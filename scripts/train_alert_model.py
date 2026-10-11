#!/usr/bin/env python3
"""Train a fixed-specification, hourly 60-minute alert-onset baseline."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from alert_model import (
    ROOT, FEATURES, REGION_NAMES, SOURCE_ENDS, attach_labels, eligible_training,
    epoch, feature_frame, fit_bundle, load_events, predict_bundle, utc,
)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return {"rows": 0}
    return {
        "rows": len(y), "positive_rows": int(y.sum()), "observed_rate": float(y.mean()),
        "mean_probability": float(p.mean()), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }


def calibration(predictions: pd.DataFrame) -> list[dict]:
    rows = []
    for split, group in predictions.groupby("split", sort=True):
        # Fixed bins, not chosen from outcomes.
        bins = np.minimum((group.probability.to_numpy() * 10).astype(int), 9)
        for b in range(10):
            sub = group.loc[bins == b]
            if len(sub):
                rows.append({"split": split, "bin_low": b / 10, "bin_high": (b + 1) / 10,
                             "rows": len(sub), "mean_probability": float(sub.probability.mean()),
                             "observed_rate": float(sub.y.mean())})
    return rows


def run_source(source: str, args) -> dict:
    out = ROOT / "reports" / "learning_baseline" / source
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / "learning_baseline" / source
    model_dir.mkdir(parents=True, exist_ok=True)
    training_start, eval_start, holdout_start = map(epoch, (args.train_start, args.eval_start, args.holdout_start))
    requested_end = epoch(args.eval_end)
    coverage_end = min(requested_end, epoch(SOURCE_ENDS[source]))
    horizon = args.horizon_minutes * 60
    step = args.cadence_minutes * 60
    events, stats = load_events(source, coverage_end - 1)
    grid = np.arange(training_start, coverage_end - horizon, step, dtype=np.int64)
    print(f"{source}: replaying {len(events):,} events at {len(grid):,} forecast times", flush=True)
    full = attach_labels(feature_frame(events, grid), events, horizon)
    source_rows = len(full)
    active_origins = int((~full.known_clear).sum())
    clear = full.loc[full.known_clear].copy()
    if not (clear.max_feature_event_time <= clear.time).all():
        raise RuntimeError("Feature event time exceeds forecast time")
    model_data = ROOT / "data" / "model"
    model_data.mkdir(parents=True, exist_ok=True)
    clear.to_csv(model_data / f"{source}_hourly_examples.csv.gz", index=False)
    coverage = []
    for region, group in full.loc[full.time >= eval_start].groupby("oblast", sort=True):
        coverage.append({"oblast": region, "evaluation_origins": len(group),
                         "known_clear_origins": int(group.known_clear.sum()),
                         "active_or_unresolved_origins": int((~group.known_clear).sum())})
    pd.DataFrame(coverage).to_csv(out / "region_coverage.csv", index=False)
    del full

    folds = []
    at = eval_start
    while at < min(holdout_start, coverage_end):
        end = min(at + args.refit_days * 86400, holdout_start, coverage_end)
        folds.append((at, end, "development"))
        at = end
    if holdout_start < coverage_end:
        folds.append((holdout_start, coverage_end, "holdout_frozen"))

    all_predictions, fold_rows = [], []
    for index, (cutoff, fold_end, split) in enumerate(folds):
        train = eligible_training(clear, cutoff, training_start)
        test = clear.loc[(clear.time >= cutoff) & (clear.time < fold_end)]
        if test.empty:
            continue
        tick = time.monotonic()
        bundle = fit_bundle(train, cutoff, source, horizon)
        bundle.update({"coverage_end_exclusive": coverage_end, "raw_sha256": stats["raw_sha256"],
                       "valid_until_exclusive": fold_end, "split": split,
                       "cadence_seconds": step, "source_path": stats["raw_path"]})
        p, region_p, calendar_p = predict_bundle(bundle, test)
        predictions = test[["time", "oblast", "oblast_id", "y", "target_end", "max_feature_event_time"]].copy()
        predictions["forecast_at_utc"] = predictions.time.map(utc)
        predictions["horizon_end_utc"] = predictions.target_end.map(utc)
        predictions["probability"] = p
        predictions["region_baseline_probability"] = region_p
        predictions["calendar_baseline_probability"] = calendar_p
        predictions["fit_cutoff_utc"] = utc(cutoff)
        predictions["training_max_target_end_utc"] = utc(bundle["max_training_target_end"])
        predictions["split"] = split
        all_predictions.append(predictions)
        model_path = model_dir / f"{datetime_tag(cutoff)}.joblib"
        joblib.dump(bundle, model_path)
        row = {"fold": index, "split": split, "fit_cutoff_utc": utc(cutoff),
               "forecast_end_exclusive_utc": utc(fold_end), "training_rows": len(train),
               "training_max_target_end_utc": utc(bundle["max_training_target_end"]),
               "fit_seconds": round(time.monotonic() - tick, 3),
               "model_path": str(model_path.relative_to(ROOT)),
               **metrics(test.y, p)}
        fold_rows.append(row)
        print(f"{source} fold {index + 1}/{len(folds)} {datetime_tag(cutoff)} {split}: "
              f"train={len(train):,}, test={len(test):,}, Brier={row['brier']:.5f}, {row['fit_seconds']:.1f}s", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / "predictions.csv.gz", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "folds.csv", index=False)
    summaries = {}
    regional = []
    for split, group in predictions.groupby("split", sort=True):
        summaries[split] = {
            "logistic": metrics(group.y, group.probability),
            "region_baseline": metrics(group.y, group.region_baseline_probability),
            "region_hour_baseline": metrics(group.y, group.calendar_baseline_probability),
        }
        a = summaries[split]
        a["brier_skill_vs_region_hour"] = 1 - a["logistic"]["brier"] / a["region_hour_baseline"]["brier"]
        for region, sub in group.groupby("oblast", sort=True):
            m = metrics(sub.y, sub.probability)
            base = metrics(sub.y, sub.calendar_baseline_probability)
            regional.append({"split": split, "oblast": region, **m,
                             "calendar_baseline_brier": base["brier"],
                             "brier_skill_vs_region_hour": 1 - m["brier"] / base["brier"] if base["brier"] else None})
    pd.DataFrame(regional).to_csv(out / "metrics_by_region.csv", index=False)
    pd.DataFrame(calibration(predictions)).to_csv(out / "calibration.csv", index=False)
    monthly = []
    for (split, month), group in predictions.groupby(["split", predictions.forecast_at_utc.str[:7]]):
        monthly.append({"split": split, "month": month, **metrics(group.y, group.probability)})
    pd.DataFrame(monthly).to_csv(out / "metrics_by_month.csv", index=False)
    result = {
        "source": source, "train_start_utc": utc(training_start), "evaluation_start_utc": utc(eval_start),
        "coverage_end_exclusive_utc": utc(coverage_end), "holdout_start_utc": utc(holdout_start),
        "horizon_minutes": args.horizon_minutes, "cadence_minutes": args.cadence_minutes,
        "development_refit_days": args.refit_days, "source_audit": stats,
        "feature_columns": FEATURES, "all_grid_rows": source_rows,
        "active_or_unresolved_grid_rows": active_origins, "clear_training_and_evaluation_rows": len(clear),
        "fold_count": len(fold_rows), "metrics": summaries,
        "audit_checks": {"all_feature_event_times_at_or_before_forecast": True,
                         "all_training_target_ends_strictly_before_fit": True,
                         "no_predictions_before_fit": True,
                         "holdout_model_frozen": True},
        "limitations": [
            "Snapshot event times proxy public availability; reception/revision histories and feed uptime are absent.",
            "Absent events are treated as negatives conditional on archive completeness; this is not verified no-alert ground truth.",
            "Only regions reconstructed clear receive onset probabilities; active or unresolved states abstain.",
            "Any subregion becoming active can start a regional episode; this is not necessarily an oblast-wide declaration.",
            "No Telegram, threat-type, eventual duration, canonical reconciliation, or future cluster features are used.",
            "Official and volunteer runs are source sensitivity analyses, not independent corroboration or interchangeable live models.",
            "Probabilities require calibration review; holdout results were not used for parameter selection.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def datetime_tag(t: int) -> str:
    return utc(t)[:10]


def write_summary(results: list[dict]) -> None:
    config = results[0]
    lines = ["# First regional alert-onset learning run", "",
             f"Target: an archive-reconstructed clear region becomes active in the next {config['horizon_minutes']} minutes. "
             f"Forecasts are made every {config['cadence_minutes']} minutes. Any active locality counts as regional activity.", "",
             f"Training starts at {config['train_start_utc']}. Evaluation starts at {config['evaluation_start_utc']} "
             f"with {config['development_refit_days']}-day expanding-window refits. "
             f"The holdout model is frozen at {config['holdout_start_utc']}; no holdout outcomes enter that fit. "
             "Each source stops at its own coverage boundary (see metrics.json); source periods differ.", "",
             "| Source | Evaluation | Examples | Event rate | Model Brier | Region/hour Brier | Brier skill | ROC AUC |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        for split, comparison in result["metrics"].items():
            m, b = comparison["logistic"], comparison["region_hour_baseline"]
            lines.append(f"| {result['source']} | {split} | {m['rows']:,} | {m['observed_rate']:.2%} | "
                         f"{m['brier']:.5f} | {b['brier']:.5f} | {comparison['brier_skill_vs_region_hour']:.2%} | {m['roc_auc']:.3f} |")
    lines += ["", "Lower Brier is better. Positive Brier skill means improvement over the regional/hour-of-day "
              "baseline. ROC AUC measures ranking, not probability calibration. These are conditional "
              "hourly scores, not the fraction of all alerts successfully warned about. Pooled ranking "
              "can reflect differences between regions; inspect within-region metrics before assessing "
              "usefulness for a selected region. Rows share time periods and events, so they are not "
              "independent trials; no statistical confidence claim is made here.", "",
              "## Availability and leakage", "",
              "All inputs use events timestamped at or before the forecast. Training excludes rows whose full "
              "forecast outcome window has not ended strictly before fitting. Scalers and coefficients are "
              "fit on those rows only. Future ends do not enter start identities or features; synthetic or "
              "ambiguous ends are not replayed. There are no Telegram inputs or retrospective cross-source "
              "matching features. Locality state is replayed from the raw archives, including pre-2024 warm-up.", "",
              "The original public reception times, revisions, and feed uptime are unavailable. Therefore this "
              "is a causal event-time simulation under an archive-availability assumption, not proof that "
              "every record could have been retrieved in exactly this form at each historical forecast time. "
              "Retrospective source corrections and archive omissions remain limitations.", "",
              "## Outputs and scope", "",
              "Per-source folders contain predictions, fold cutoffs, monthly/region scores, calibration bins "
              "and region coverage. Saved models are in `models/learning_baseline/`. The training examples "
              "are in `data/model/`. Current alerts, missing coverage and unsupported dates produce no forecast.", "",
              "This first model estimates general alert onset. It does not yet estimate remaining duration or "
              "separate missile/UAV probabilities. Those require additional target construction and reviewed "
              "threat labels; the 2024–2025 alert archive has no threat-type labels.", ""]
    (ROOT / "reports" / "learning_baseline" / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["official", "volunteer", "both"], default="both")
    parser.add_argument("--train-start", default="2024-01-01")
    parser.add_argument("--eval-start", default="2026-01-01")
    parser.add_argument("--holdout-start", default="2026-08-01")
    parser.add_argument("--eval-end", default="2026-10-01")
    parser.add_argument("--refit-days", type=int, default=5)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--cadence-minutes", type=int, default=60)
    args = parser.parse_args()
    if not epoch(args.train_start) < epoch(args.eval_start) < epoch(args.holdout_start) < epoch(args.eval_end):
        parser.error("Require train-start < eval-start < holdout-start < eval-end")
    if min(args.refit_days, args.horizon_minutes, args.cadence_minutes) <= 0:
        parser.error("Time intervals must be positive")
    sources = ["official", "volunteer"] if args.source == "both" else [args.source]
    with threadpool_limits(limits=2):
        results = [run_source(source, args) for source in sources]
    write_summary(results)
    print("Finished: reports/learning_baseline/summary.md", flush=True)


if __name__ == "__main__":
    main()
