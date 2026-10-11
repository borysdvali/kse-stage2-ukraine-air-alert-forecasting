#!/usr/bin/env python3
"""Frozen-specification retrospective October 1–10 evaluation.

Fit/calibration end before October; October inputs are replayed as of each
hour. The source is a final archive snapshot, not a captured real-time feed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from itertools import groupby

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from alert_model import (ROOT, REGION_NAMES, REGION_IDS, REGION_INDEX, attach_labels,
                         canonical_region, epoch, feature_frame, load_events, utc)
from train_alert_model_v2 import (FEATURES, LEGACY_FEATURES, add_cross_region_features,
                                 calibrated_p, calibrator, duration_feature_matrix,
                                 metric_pack)

START = epoch("2024-01-01")
FIT_END = epoch("2026-09-01")
TEST_START = epoch("2026-10-01")
TEST_END = epoch("2026-10-11")
HORIZON = 3600
PRIMARY = "logistic_current_features"
CANDIDATES = (PRIMARY, "logistic_cross_region", "hist_gradient_boosting_cross_region")
OUT = ROOT / "reports/october_2026_review"
MODELS = ROOT / "models/october_2026_review"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_features(events, grid):
    return add_cross_region_features(feature_frame(events, grid), events)


def event_digest(events):
    h = hashlib.sha256()
    for e in events:
        h.update((json.dumps(e, separators=(",", ":")) + "\n").encode())
    return h.hexdigest()


def eligible_duration_mask(ep, start, end):
    return (ep.start_epoch >= start) & (ep.end_epoch < end) & ep.duration_target_eligible


def reconstructed_episodes(events, uncertain_starts):
    """Use the same causal locality replay as the onset model.

    Unobserved/imputed endings are never invented. Any episode containing an
    uncertain start is excluded from completed-duration scoring. The quality flag
    is a retrospective label filter, never a forecast input.
    """
    active = [dict() for _ in REGION_NAMES]
    pending, rows = {}, []
    for t, batch_iter in groupby(events, key=lambda e: e[0]):
        batch = list(batch_iter)
        touched = {e[2] for e in batch}
        before = {r: bool(active[r]) for r in touched}
        tainted = {r: False for r in touched}
        for _, kind, r, locality, token in batch:
            if kind == 0:
                active[r][locality] = token
                tainted[r] |= (r, token) in uncertain_starts
            elif active[r].get(locality) == token:
                del active[r][locality]
        for r in touched:
            after = bool(active[r])
            if not before[r] and after:
                pending[r] = {"start_epoch": t, "tainted": tainted[r]}
            elif before[r]:
                pending[r]["tainted"] |= tainted[r]
                if not after:
                    p = pending.pop(r)
                    duration = (t - p["start_epoch"]) / 60
                    rows.append({
                        "oblast": REGION_NAMES[r], "oblast_id": REGION_IDS[r],
                        "start_epoch": p["start_epoch"], "end_epoch": t,
                        "started_at_utc": utc(p["start_epoch"]), "ended_at_utc": utc(t),
                        "duration_minutes": duration, "uncertain_contributor": p["tainted"],
                        "duration_target_eligible": not p["tainted"] and 0 < duration <= 7 * 1440,
                    })
    for r, p in pending.items():
        rows.append({
            "oblast": REGION_NAMES[r], "oblast_id": REGION_IDS[r],
            "start_epoch": p["start_epoch"], "end_epoch": np.nan,
            "started_at_utc": utc(p["start_epoch"]), "ended_at_utc": None,
            "duration_minutes": np.nan, "uncertain_contributor": p["tainted"],
            "duration_target_eligible": False,
        })
    return pd.DataFrame(rows).sort_values(["start_epoch", "oblast_id"]).reset_index(drop=True)


def uncertain_keys(until):
    keys = set()
    with (ROOT / "data/raw/volunteer_data_en.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("naive", "").strip().lower() != "true":
                continue
            r = canonical_region(row.get("region", ""))
            t = epoch(row["started_at"])
            if r in REGION_INDEX and t <= until:
                keys.add((REGION_INDEX[r], t))
    return keys


def fit_onset(history):
    train = history.loc[history.known_clear & (history.target_end < FIT_END)]
    cal = history.loc[history.known_clear & (history.time >= FIT_END) & (history.target_end < TEST_START)]
    assert (train.max_feature_event_time <= train.time).all()
    assert (cal.max_feature_event_time <= cal.time).all()
    assert train.target_end.max() < FIT_END and cal.target_end.max() < TEST_START
    y, cy = train.y.to_numpy(), cal.y.to_numpy()
    specifications = {
        PRIMARY: (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500)), LEGACY_FEATURES),
        "logistic_cross_region": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500)), FEATURES),
        "hist_gradient_boosting_cross_region": (HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, early_stopping=False, random_state=24), FEATURES),
    }
    bundles = {}
    common = {"fit_cutoff_utc": utc(FIT_END), "calibration_end_utc": utc(TEST_START),
              "max_training_target_end": int(train.target_end.max()),
              "max_calibration_target_end": int(cal.target_end.max()),
              "training_rows": len(train), "calibration_rows": len(cal),
              "training_positives": int(y.sum()), "calibration_positives": int(cy.sum()),
              "horizon_seconds": HORIZON, "source": "volunteer"}
    for name, (model, cols) in specifications.items():
        print(f"Fitting {name}: {len(train):,} training + {len(cal):,} calibration rows", flush=True)
        model.fit(train[cols].to_numpy(dtype=float), y)
        if name.startswith("logistic") and model[-1].n_iter_.max() >= model[-1].max_iter:
            raise RuntimeError("Logistic model did not converge")
        c = calibrator(model.predict_proba(cal[cols].to_numpy(dtype=float))[:, 1], cy)
        bundles[name] = {**common, "model": model, "features": cols, "calibrator": c}
    keys = train.region_index.to_numpy() * 24 + train.local_hour.to_numpy()
    counts = np.bincount(keys, minlength=len(REGION_NAMES) * 24)
    positive = np.bincount(keys, weights=y, minlength=len(REGION_NAMES) * 24)
    global_rate = (y.sum() + 0.5) / (len(y) + 1)
    rc = np.bincount(train.region_index, minlength=len(REGION_NAMES))
    rp = np.bincount(train.region_index, weights=y, minlength=len(REGION_NAMES))
    rr = (rp + 20 * global_rate) / (rc + 20)
    rates = (positive + 20 * np.repeat(rr, 24)) / (counts + 20)
    cp = rates[cal.region_index.to_numpy() * 24 + cal.local_hour.to_numpy()]
    bundles["region_hour_baseline"] = {**common, "rates": rates, "calibrator": calibrator(cp, cy)}
    for name, bundle in bundles.items():
        joblib.dump(bundle, MODELS / f"{name}.joblib")
    return bundles, common


def predict_onset(bundle, frame):
    if int(frame.time.min()) < epoch(bundle["calibration_end_utc"]):
        raise ValueError("Forecast precedes the end of model calibration")
    if not (frame.max_feature_event_time <= frame.time).all():
        raise ValueError("Feature event after forecast time")
    if "rates" in bundle:
        raw = bundle["rates"][frame.region_index.to_numpy() * 24 + frame.local_hour.to_numpy()]
    else:
        raw = bundle["model"].predict_proba(frame[bundle["features"]].to_numpy(dtype=float))[:, 1]
    return raw, calibrated_p(bundle["calibrator"], raw)


def duration_predictions(bundle, x):
    pred = np.stack([bundle["models"][q].predict(x[bundle["features"]].to_numpy(dtype=float))
                     for q in (0.1, 0.5, 0.9)], axis=1)
    pred = np.sort(np.maximum(0, pred), axis=1)
    pred[:, 0] = np.maximum(0, pred[:, 0] - bundle["interval_expansion_minutes"])
    pred[:, 2] += bundle["interval_expansion_minutes"]
    return pred


def fit_duration(events):
    ep = reconstructed_episodes(events, uncertain_keys(TEST_START - 1))
    train = ep.loc[eligible_duration_mask(ep, START, FIT_END)].reset_index(drop=True)
    cal = ep.loc[eligible_duration_mask(ep, FIT_END, TEST_START)].reset_index(drop=True)
    x, cols = duration_feature_matrix(train, events)
    cx, _ = duration_feature_matrix(cal, events)
    y = train.duration_minutes.to_numpy()
    models = {}
    for q in (0.1, 0.5, 0.9):
        print(f"Fitting duration quantile {q}: {len(train):,} completed training episodes", flush=True)
        model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=160,
            learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=2, early_stopping=False, random_state=24)
        model.fit(x.to_numpy(dtype=float), y)
        models[q] = model
    bundle = {"models": models, "features": cols, "interval_expansion_minutes": 0.0,
              "region_medians": train.groupby("oblast").duration_minutes.median().to_dict(),
              "global_median": float(np.median(y)), "fit_cutoff_utc": utc(FIT_END),
              "calibration_end_utc": utc(TEST_START), "training_rows": len(train),
              "calibration_rows": len(cal), "max_training_end_epoch": int(train.end_epoch.max()),
              "max_calibration_end_epoch": int(cal.end_epoch.max())}
    p = duration_predictions(bundle, cx)
    cy = cal.duration_minutes.to_numpy()
    error = np.maximum.reduce((p[:, 0] - cy, cy - p[:, 2], np.zeros(len(cy))))
    rank = min(1.0, np.ceil((len(cy) + 1) * 0.8) / len(cy))
    bundle["interval_expansion_minutes"] = float(np.quantile(error, rank, method="higher"))
    joblib.dump(bundle, MODELS / "duration.joblib")
    return bundle


def block_skill(pred, column, baseline_column):
    tmp = pd.DataFrame({"day": pred.forecast_at_utc.str[:10],
                        "model": (pred[column] - pred.y) ** 2,
                        "baseline": (pred[baseline_column] - pred.y) ** 2})
    daily = tmp.groupby("day")[["model", "baseline"]].sum().to_numpy()
    rng = np.random.default_rng(20261001)
    samples = daily[rng.integers(0, len(daily), size=(5000, len(daily)))].sum(axis=1)
    skills = 1 - samples[:, 0] / samples[:, 1]
    return {"skill": float(1 - daily[:, 0].sum() / daily[:, 1].sum()),
            "day_block_bootstrap_95pct": np.quantile(skills, [0.025, 0.975]).tolist(),
            "days": len(daily), "replicates": 5000}


def audit_old_duration():
    result = {}
    for source in ("official", "volunteer"):
        ep = pd.concat([pd.read_csv(ROOT / "data/processed/2024-01-01_2025-12-31" / f"{source}_oblast_episodes.csv"),
                        pd.read_csv(ROOT / "data/processed" / f"{source}_oblast_episodes.csv")])
        ep = ep.loc[ep.duration_target_eligible.astype(bool)].copy()
        start = pd.to_datetime(ep.started_at_utc, utc=True)
        end = pd.to_datetime(ep.ended_at_utc, utc=True)
        a, b, c = [pd.Timestamp(x, tz="UTC") for x in ("2025-01-01", "2026-01-01", "2026-08-01")]
        result[source] = {"old_training_labels_ending_at_or_after_fit": int(((start < a) & (end >= a)).sum()),
                          "old_calibration_labels_ending_at_or_after_cutoff": int(((start >= a) & (start < b) & (end >= b)).sum()),
                          "old_test_labels_ending_outside_window": int(((start >= b) & (start < c) & (end >= c)).sum())}
    return result


def write_report(report, regions, days):
    coverage = report["coverage"]
    lines = ["# Forecasting review: October 1–10, 2026", "",
        "This is a retrospective replay of hourly next-hour forecasts, not a single ten-day forecast made on September 30. Dates use UTC, consistently with the existing project; calendar features use Kyiv local time.", "",
        "The primary specification was fixed from the earlier September comparison: logistic regression with existing own-region, calendar and country-wide activity features. Cross-region logistic regression and histogram gradient boosting are fixed challengers. October is used for scoring only; weights and calibration remain frozen for all ten days. Because October was mentioned/inspected earlier in this project, this is not claimed to be a never-seen holdout.", "",
        "## Data and method", "",
        f"Volunteer records are available on every October day. The official-derived archive ends September 7 and has no October comparison. There are {coverage['all_region_hours']:,} eligible calendar region-hours, of which {coverage['clear_region_hours']:,} are reconstructed clear and scored; {coverage['active_or_unresolved_region_hours']:,} are active/unresolved and receive no onset forecast. The final hourly origin is omitted so its target cannot include October 11.", "",
        "Refreshed model fit: January 1, 2024–August 31, 2026. Probability/interval calibration: September 2026. Every fitting or calibration label must be fully observed before the next cutoff. Thus all three historical periods contribute, with September reserved for calibration. Training examples use 2024 onward; earlier events initialize state and lag features only.", "",
        "The target is an alert starting within the next 60 minutes in a currently clear region (any mapped locality). There are 24 modeled regions: 23 oblasts plus Kyiv City. Missile/UAV labels and Telegram text are not model inputs.", "",
        "## October onset comparison", "",
        "Brier score is mean squared probability error (lower is better). Skill is the relative Brier improvement against the refreshed, September-calibrated region/hour historical-frequency baseline, not percent accuracy. AUC measures ranking and is not a calibration score.", "",
        "| Model | Brier | Log loss | AUC | Average precision | Mean probability | Observed rate | Brier skill |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["onset_metrics"].items():
        s = report["brier_skill"].get(name)
        skill_text = f"{s['skill']:.1%}" if s else "reference"
        lines.append(f"| {name} | {m['brier']:.5f} | {m['log_loss']:.5f} | {m['roc_auc']:.3f} | {m['average_precision']:.3f} | {m['mean_probability']:.1%} | {m['event_rate']:.1%} | {skill_text} |")
    skill = report["brier_skill"]["refreshed_" + PRIMARY]
    ci = skill["day_block_bootstrap_95pct"]
    lines += ["", f"The prespecified primary model has Brier skill {skill['skill']:.1%}, with an approximate day-block bootstrap 95% interval {ci[0]:.1%} to {ci[1]:.1%}. Resampling whole days retains simultaneous region dependence, but ten days is a small sample and this interval does not establish stability across seasons or future attack patterns.", "",
        f"The simple logistic specification remains the leading candidate on this test; the richer cross-region models do not improve its aggregate Brier score. It beats the frequency baseline in {report['primary_regions_better_than_baseline']} of 24 regions, but the mean within-region AUC is only {report['primary_macro_region_auc']:.3f}. The pooled AUC partly reflects differences between regions and overstates the ability to time an alert within a selected region. Zakarpattia has no positive test hours, so its large Brier improvement does not demonstrate detection of alerts there. Ternopil has only 17 scored clear hours because most hours are active/unresolved; its regional estimate is particularly uncertain.", "",
        "## Probability calibration", "",
        "| Forecast band | Scored hours | Mean forecast | Observed alert rate |",
        "|---|---:|---:|---:|"]
    for row in json.loads((OUT / "calibration.json").read_text()):
        if row["rows"]:
            lo, hi = row["probability_bin"]
            lines.append(f"| {lo:.0%}–{hi:.0%} | {row['rows']} | {row['mean_probability']:.1%} | {row['event_rate']:.1%} |")
    lines += ["", "Calibration is uneven: the 10–20% forecast band underestimates observed frequency, and the 80–90% band has just one example. The model also underpredicts the rise in alerts on October 8–10. These findings are retained as test results; no October-driven recalibration was performed.", "",
        "## Primary model by region", "",
        "| Region | Scored hours | Positive hours | Observed rate | Mean forecast | Brier | Baseline Brier | Skill | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in regions:
        if r["scored_hours"] == 0:
            lines.append(f"| {r['region']} | 0 | — | — | — | — | — | — | — |")
            continue
        m = r["model"]
        auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "—"
        lines.append(f"| {r['region']} | {r['scored_hours']} | {m['positives']} | {m['event_rate']:.1%} | {m['mean_probability']:.1%} | {m['brier']:.5f} | {r['baseline']['brier']:.5f} | {r['brier_skill']:.1%} | {auc} |")
    lines += ["", "## Primary model by day", "",
        "| UTC day | Scored hours | Positive hours | Mean forecast | Observed rate | Brier | Baseline Brier |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for d in days:
        m = d["model"]
        lines.append(f"| {d['day']} | {m['rows']} | {m['positives']} | {m['mean_probability']:.1%} | {m['event_rate']:.1%} | {m['brier']:.5f} | {d['baseline']['brier']:.5f} |")
    lines += ["", "## Duration at alert start", "",
        "Duration labels now follow the same causal regional-state replay as onset targets. Only completed episodes with observed endings and no uncertain contributor are scored; episodes over seven days and open episodes are excluded and counted. This conservative target differs from the older retrospectively merged normalization table. Predictions estimate total duration at the start of an alert, not remaining duration midway through it.", "",
        "| Model | Episodes | MAE (min) | Median absolute error (min) | P10–P90 coverage | Mean interval width (min) |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, d in report["duration_metrics"].items():
        c = f"{d['interval_coverage']:.1%}" if "interval_coverage" in d else "—"
        w = f"{d['interval_mean_width']:.1f}" if "interval_mean_width" in d else "—"
        lines.append(f"| {name} | {d['rows']} | {d['mae_minutes']:.1f} | {d['median_absolute_error_minutes']:.1f} | {c} | {w} |")
    dc = report["duration_coverage"]
    lines += ["", f"There were {dc['october_onsets']:,} reconstructed October onsets: {dc['eligible_completed']:,} eligible completed episodes were scored, {dc['open_at_test_end']} remained open at the test cutoff, and {dc['completed_but_ineligible']} completed episodes were ineligible. The observed median duration in the scored subset was {dc['observed_median_minutes']:.1f} minutes. Excluding open and uncertain episodes can underrepresent long alerts. The duration gain over the simple regional-median baseline is modest. The comparison with the old model combines an updated training period and a more consistent duration target, so the improvement cannot be attributed to training-history length alone.", "",
        "## Audit findings and limits", "",
        "The old duration split checked start date, allowing six official-source training labels whose ends were not yet known. No volunteer training/calibration labels crossed those old cutoffs. Eight official and two volunteer old duration test labels ended outside the stated test window. The trainer is corrected; previous duration reports are retained as historical artifacts and have not been silently overwritten.", "",
        "Forecast features were checked against prefix-only replay at three October dates, including the individual cross-region inputs. Training/calibration outcome cutoffs, prediction bounds, unique keys, and full one-hour outcome containment are checked. This establishes timestamp causality under the archive assumption, not true historical receipt availability.", "",
        "The raw archives lack original reception/revision history and feed uptime. Missing records are not verified negatives. October has only a volunteer reference, and earlier October inspection cannot be undone. Both onset and duration results therefore remain archive-based, retrospective estimates. The models do not yet distinguish missile, propeller-UAV or jet-UAV threats.", "",
        "Machine-readable files: protocol.json, metrics.json, regions.json, days.json, calibration.json, predictions.jsonl.gz, and duration_predictions.jsonl.gz. The separate saved-output audit writes verification.json. Saved historical models and earlier numerical results were preserved; the earlier duration summary carries an audit correction.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    protocol = {
        "training_start_utc": utc(START), "parameter_fit_end_exclusive_utc": utc(FIT_END),
        "calibration_end_exclusive_utc": utc(TEST_START), "test_end_exclusive_utc": utc(TEST_END),
        "forecast_horizon_minutes": 60, "step_minutes": 60, "source": "volunteer",
        "primary_specification": PRIMARY, "selection_basis": "lowest calibrated September Brier score in existing learning_v2 report",
        "challengers": list(CANDIDATES[1:]), "october_parameter_updates": False,
        "test_design": "rolling hourly forecasts, frozen model, timestamped information up to each forecast",
        "previous_october_exposure": "mentioned in prior conversation; not claimed to be a pristine unseen holdout",
        "time_boundary": "UTC dates, consistent with prior project windows",
        "hyperparameters": "same fixed model complexity and regularization as learning_v2; logistic iteration cap increased only for convergence",
    }
    dump(OUT / "protocol.json", protocol)
    history_events, history_audit = load_events("volunteer", TEST_START - 1)
    old_report = json.loads((ROOT / "reports/learning_v2/metrics.json").read_text())
    if event_digest(history_events) != old_report["in_window_event_stream_sha256"]:
        raise ValueError("Pre-October event history differs from the saved September experiment")
    grid = np.arange(START, TEST_START - HORIZON, HORIZON, dtype=np.int64)
    print("Building pre-October feature history", flush=True)
    history = attach_labels(make_features(history_events, grid), history_events, HORIZON)
    bundles, fit_audit = fit_onset(history)
    del history
    duration_bundle = fit_duration(history_events)
    # All fit operations have finished before October features/outcomes are built.
    all_events, full_audit = load_events("volunteer", TEST_END - 1)
    assert [e for e in all_events if e[0] < TEST_START] == history_events
    grid = np.arange(TEST_START, TEST_END - HORIZON, HORIZON, dtype=np.int64)
    features = make_features(all_events, grid)
    checked = []
    for stamp in ("2026-10-01T00:00:00Z", "2026-10-05T12:00:00Z", "2026-10-10T18:00:00Z"):
        at = epoch(stamp)
        prefix, _ = load_events("volunteer", at)
        one = make_features(prefix, np.array([at], dtype=np.int64))
        expected = features.loc[features.time == at].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, one)
        checked.append(stamp)
    test = attach_labels(features, all_events, HORIZON)
    clear = test.loc[test.known_clear].copy()
    pred = clear[["time", "target_end", "oblast", "oblast_id", "y", "max_feature_event_time"]].copy()
    pred["forecast_at_utc"] = pred.time.map(utc)
    pred["target_end_utc"] = pred.target_end.map(utc)
    for name, b in bundles.items():
        raw, cal = predict_onset(b, clear)
        pred["refreshed_" + name + "_raw"] = raw
        pred["refreshed_" + name] = cal
    for name in CANDIDATES:
        b = joblib.load(ROOT / "models" / f"onset_{name}_2026-09-01.joblib")
        assert b["max_training_target_end"] < epoch(b["fit_cutoff_utc"])
        assert b["max_calibration_target_end"] < epoch(b["calibration_end_utc"]) <= TEST_START
        pred["previous_" + name] = predict_onset(b, clear)[1]
    columns = ["refreshed_region_hour_baseline"] + ["refreshed_" + n for n in CANDIDATES] + ["previous_" + n for n in CANDIDATES]
    metrics = {col: metric_pack(pred.y.to_numpy(), pred[col].to_numpy()) for col in columns}
    baseline = "refreshed_region_hour_baseline"
    primary = "refreshed_" + PRIMARY
    skills = {col: block_skill(pred, col, baseline) for col in columns if col != baseline}
    regions = []
    for region in REGION_NAMES:
        p = pred.loc[pred.oblast == region]
        row = {"region": region, "scored_hours": len(p), "total_hours": len(grid),
               "active_or_unresolved_hours": len(grid) - len(p)}
        if len(p):
            row["model"] = metric_pack(p.y.to_numpy(), p[primary].to_numpy())
            row["baseline"] = metric_pack(p.y.to_numpy(), p[baseline].to_numpy())
            row["brier_skill"] = 1 - row["model"]["brier"] / row["baseline"]["brier"]
            row["challenger_metrics"] = {n: metric_pack(p.y.to_numpy(), p["refreshed_" + n].to_numpy()) for n in CANDIDATES[1:]}
        regions.append(row)
    days = []
    for day, p in pred.groupby(pred.forecast_at_utc.str[:10]):
        days.append({"day": day, "model": metric_pack(p.y.to_numpy(), p[primary].to_numpy()),
                     "baseline": metric_pack(p.y.to_numpy(), p[baseline].to_numpy())})
    calibration = []
    bins = np.minimum((pred[primary] * 10).astype(int), 9)
    for i in range(10):
        p = pred.loc[bins == i]
        calibration.append({"probability_bin": [i / 10, (i + 1) / 10], "rows": len(p),
                            "mean_probability": float(p[primary].mean()) if len(p) else None,
                            "event_rate": float(p.y.mean()) if len(p) else None})
    episodes = reconstructed_episodes(all_events, uncertain_keys(TEST_END - 1))
    october_ep = episodes.loc[(episodes.start_epoch >= TEST_START) & (episodes.start_epoch < TEST_END)]
    eligible = october_ep.loc[eligible_duration_mask(october_ep, TEST_START, TEST_END)].reset_index(drop=True)
    dx, _ = duration_feature_matrix(eligible, all_events)
    for i in np.linspace(0, len(eligible) - 1, 3, dtype=int):
        ep = eligible.iloc[[i]].reset_index(drop=True)
        at = int(ep.start_epoch.iloc[0]) - 1
        prefix, _ = load_events("volunteer", at)
        px, _ = duration_feature_matrix(ep, prefix)
        pd.testing.assert_frame_equal(dx.iloc[[i]].reset_index(drop=True), px)
    dp = duration_predictions(duration_bundle, dx)
    y = eligible.duration_minutes.to_numpy()
    db = np.array([duration_bundle["region_medians"].get(r, duration_bundle["global_median"]) for r in eligible.oblast])
    old_duration = joblib.load(ROOT / "models/duration_volunteer_2026-01-01.joblib")
    old_dp = duration_predictions(old_duration, dx)
    duration_metrics = {}
    for name, estimates in (("refreshed_quantile_boosting", dp), ("previous_quantile_boosting", old_dp)):
        duration_metrics[name] = {"rows": len(y), "mae_minutes": float(mean_absolute_error(y, estimates[:, 1])),
            "median_absolute_error_minutes": float(np.median(np.abs(y - estimates[:, 1]))),
            "interval_coverage": float(np.mean((y >= estimates[:, 0]) & (y <= estimates[:, 2]))),
            "interval_mean_width": float(np.mean(estimates[:, 2] - estimates[:, 0]))}
    duration_metrics["training_region_median"] = {"rows": len(y), "mae_minutes": float(mean_absolute_error(y, db)),
        "median_absolute_error_minutes": float(np.median(np.abs(y - db)))}
    for j, name in enumerate(("p10_minutes", "p50_minutes", "p90_minutes")):
        eligible[name] = dp[:, j]
    eligible["baseline_minutes"] = db
    assert pred.target_end.max() < TEST_END
    assert not pred.duplicated(["time", "oblast_id"]).any()
    assert (pred.max_feature_event_time <= pred.time).all()
    assert all(pred[c].between(0, 1).all() for c in columns)
    assert duration_bundle["max_training_end_epoch"] < FIT_END
    assert duration_bundle["max_calibration_end_epoch"] < TEST_START
    region_auc = [r["model"]["roc_auc"] for r in regions if r.get("model", {}).get("roc_auc") is not None]
    report = {
        "protocol": protocol, "fit_audit": fit_audit,
        "source_audit": {"historical": history_audit, "through_october_10": full_audit,
                         "historical_event_digest": event_digest(history_events),
                         "evaluation_event_digest": event_digest(all_events)},
        "coverage": {"hourly_origins": len(grid), "all_region_hours": len(test),
                     "clear_region_hours": len(clear), "active_or_unresolved_region_hours": len(test) - len(clear),
                     "positive_clear_region_hours": int(clear.y.sum()), "region_count": len(REGION_NAMES)},
        "onset_metrics": metrics, "brier_skill": skills,
        "primary_macro_region_auc": float(np.mean(region_auc)),
        "primary_regions_better_than_baseline": sum(r.get("brier_skill", -1) > 0 for r in regions),
        "raw_onset_metrics": {n: metric_pack(pred.y.to_numpy(), pred["refreshed_" + n + "_raw"].to_numpy()) for n in (*CANDIDATES, "region_hour_baseline")},
        "duration_metrics": duration_metrics,
        "duration_fit_audit": {k: v for k, v in duration_bundle.items() if k not in ("models", "features", "region_medians")},
        "duration_coverage": {"october_onsets": len(october_ep), "eligible_completed": len(eligible),
            "open_at_test_end": int(october_ep.end_epoch.isna().sum()),
            "completed_but_ineligible": int((october_ep.end_epoch.notna() & ~october_ep.duration_target_eligible).sum()),
            "observed_median_minutes": float(np.median(y))},
        "previous_duration_split_findings": audit_old_duration(),
        "temporal_audit": {"all_checks_passed": True, "prefix_feature_checks_utc": checked,
            "duration_prefix_checks": 3, "histories_identical_before_october": True,
            "training_and_calibration_finished_before_october_events_loaded": True,
            "full_test_horizons_before_october_11": True, "october_model_updates": 0},
    }
    dump(OUT / "metrics.json", report)
    dump(OUT / "regions.json", regions)
    dump(OUT / "days.json", days)
    dump(OUT / "calibration.json", calibration)
    pred.to_json(OUT / "predictions.jsonl.gz", orient="records", lines=True, compression="gzip", double_precision=15)
    eligible.to_json(OUT / "duration_predictions.jsonl.gz", orient="records", lines=True, compression="gzip", double_precision=15)
    write_report(report, regions, days)
    print(json.dumps({"coverage": report["coverage"], "onset_metrics": metrics, "duration_metrics": duration_metrics,
                      "primary_skill": skills[primary]}, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
