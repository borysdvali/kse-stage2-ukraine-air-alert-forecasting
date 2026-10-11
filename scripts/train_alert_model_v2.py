#!/usr/bin/env python3
"""Cross-region onset model comparison and completed-alert duration models.

Historical timestamps proxy data availability; source archives do not preserve
reception/revision times. Evaluation metrics are therefore archive-based estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, mean_absolute_error, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alert_model import (  # noqa: E402
    ROOT, REGION_IDS, REGION_NAMES, attach_labels, epoch, feature_frame,
    load_events, replay, utc,
)

TRAIN_START = epoch("2024-01-01T00:00:00+00:00")
CALIBRATION_START = epoch("2026-08-01T00:00:00+00:00")
EVALUATION_START = epoch("2026-09-01T00:00:00+00:00")
ANALYSIS_END_EXCLUSIVE = epoch("2026-10-01T00:00:00+00:00")
HORIZON = 3600
STEP = 3600
LEGACY_FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "year_sin", "year_cos",
    "since_last_onset_log_hours", "own_onsets_6h_log", "own_onsets_24h_log",
    "own_onsets_7d_log", "country_active_fraction", "country_onsets_1h_log",
    "country_onsets_6h_log", "country_onsets_24h_log",
] + [f"region_{rid}" for rid in REGION_IDS]
CROSS_FEATURES = [f"other_active_{rid}" for rid in REGION_IDS] + [
    f"other_onsets_6h_{rid}" for rid in REGION_IDS
]
FEATURES = LEGACY_FEATURES + CROSS_FEATURES


def add_cross_region_features(frame: pd.DataFrame, events: list) -> pd.DataFrame:
    """Attach other-region status/recent onsets using only state at forecast time."""
    onsets, changes = replay(events)
    grid = frame.time.drop_duplicates().to_numpy(dtype=np.int64)
    state_by_region = []
    for times, values in changes:
        ix = np.searchsorted(times, grid, side="right") - 1
        state = np.zeros(len(grid), dtype=np.int8)
        ok = ix >= 0
        state[ok] = values[ix[ok]].astype(np.int8)
        state_by_region.append(state)
    onset_by_region = [
        np.searchsorted(times, grid, side="right") - np.searchsorted(times, grid - 6 * 3600, side="right")
        for times in onsets
    ]
    index = {int(t): j for j, t in enumerate(grid)}
    result = frame.copy()
    row_ix = result.time.map(index).to_numpy(dtype=int)
    own = result.region_index.to_numpy(dtype=int)
    for origin, rid in enumerate(REGION_IDS):
        active = state_by_region[origin][row_ix].copy()
        recent = onset_by_region[origin][row_ix].astype(np.int16)
        self_mask = own == origin
        active[self_mask] = 0
        recent[self_mask] = 0
        result[f"other_active_{rid}"] = active
        result[f"other_onsets_6h_{rid}"] = np.log1p(recent)
    return result


def metric_pack(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": int(len(y)), "positives": int(y.sum()), "event_rate": float(y.mean()),
        "mean_probability": float(p.mean()), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
    }


def calibrator(raw_p: np.ndarray, y: np.ndarray):
    z = np.log(np.clip(raw_p, 1e-6, 1 - 1e-6) / np.clip(1 - raw_p, 1e-6, 1))
    cal = LogisticRegression(C=10.0, max_iter=500)
    cal.fit(z.reshape(-1, 1), y)
    return cal


def calibrated_p(cal, raw_p: np.ndarray) -> np.ndarray:
    z = np.log(np.clip(raw_p, 1e-6, 1 - 1e-6) / np.clip(1 - raw_p, 1e-6, 1))
    return cal.predict_proba(z.reshape(-1, 1))[:, 1]


def fit_onset_models(frame: pd.DataFrame) -> dict:
    train = frame.loc[(frame.time >= TRAIN_START) & (frame.target_end < CALIBRATION_START) & frame.known_clear]
    cal = frame.loc[(frame.time >= CALIBRATION_START) & (frame.target_end < EVALUATION_START) & frame.known_clear]
    test = frame.loc[(frame.time >= EVALUATION_START) & (frame.target_end < ANALYSIS_END_EXCLUSIVE) & frame.known_clear]
    if not len(train) or not len(cal) or not len(test):
        raise ValueError("Training/calibration/evaluation split is empty")
    if not (train.target_end < CALIBRATION_START).all() or not (cal.target_end < EVALUATION_START).all():
        raise ValueError("Target outcome crosses a fitting cutoff")
    if not (test.target_end < ANALYSIS_END_EXCLUSIVE).all():
        raise ValueError("Evaluation target extends beyond 30 September 2026")
    for split in (train, cal, test):
        if not (split.max_feature_event_time <= split.time).all():
            raise ValueError("Future event used as an onset feature")
    y_train = train.y.to_numpy(dtype=np.int8)
    y_cal = cal.y.to_numpy(dtype=np.int8)
    y_test = test.y.to_numpy(dtype=np.int8)
    models = {
        "logistic_current_features": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=300)), LEGACY_FEATURES),
        "logistic_cross_region": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=300)), FEATURES),
        "hist_gradient_boosting_cross_region": (HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=1.0, early_stopping=False, random_state=24,
        ), FEATURES),
    }
    results, preds, bundles = {}, test[["time", "oblast", "oblast_id", "y", "target_end"]].copy(), {}
    for name, (model, columns) in models.items():
        model.fit(train[columns].to_numpy(dtype=float), y_train)
        raw_cal = model.predict_proba(cal[columns].to_numpy(dtype=float))[:, 1]
        cal_model = calibrator(raw_cal, y_cal)
        raw_test = model.predict_proba(test[columns].to_numpy(dtype=float))[:, 1]
        p_calibrated = calibrated_p(cal_model, raw_test)
        results[name] = {"raw": metric_pack(y_test, raw_test), "platt_calibrated": metric_pack(y_test, p_calibrated)}
        preds[f"{name}_raw"] = raw_test
        preds[f"{name}_calibrated"] = p_calibrated
        bundles[name] = {"model": model, "calibrator": cal_model, "features": columns,
                         "fit_cutoff_utc": utc(CALIBRATION_START), "calibration_end_utc": utc(EVALUATION_START),
                         "evaluation_start_utc": utc(EVALUATION_START), "horizon_seconds": HORIZON,
                         "training_rows": len(train), "calibration_rows": len(cal),
                         "max_training_target_end": int(train.target_end.max()),
                         "max_calibration_target_end": int(cal.target_end.max())}
    # Training-only region/hour baseline, then calibrated on August in the same way.
    keys_train = train.region_index.to_numpy() * 24 + train.local_hour.to_numpy()
    counts = np.bincount(keys_train, minlength=len(REGION_NAMES) * 24)
    positives = np.bincount(keys_train, weights=y_train, minlength=len(REGION_NAMES) * 24)
    global_rate = (y_train.sum() + 0.5) / (len(y_train) + 1)
    region_count = np.bincount(train.region_index.to_numpy(), minlength=len(REGION_NAMES))
    region_pos = np.bincount(train.region_index.to_numpy(), weights=y_train, minlength=len(REGION_NAMES))
    region_rate = (region_pos + 20 * global_rate) / (region_count + 20)
    baseline = (positives + 20 * np.repeat(region_rate, 24)) / (counts + 20)
    cal_keys = cal.region_index.to_numpy() * 24 + cal.local_hour.to_numpy()
    test_keys = test.region_index.to_numpy() * 24 + test.local_hour.to_numpy()
    cal_base, test_base = baseline[cal_keys], baseline[test_keys]
    base_cal = calibrator(cal_base, y_cal)
    test_base_cal = calibrated_p(base_cal, test_base)
    results["training_region_hour_baseline"] = {"raw": metric_pack(y_test, test_base),
                                                  "platt_calibrated": metric_pack(y_test, test_base_cal)}
    preds["training_region_hour_baseline_raw"] = test_base
    preds["training_region_hour_baseline_calibrated"] = test_base_cal
    for name in models:
        results[name]["brier_skill_vs_region_hour_raw"] = 1 - results[name]["platt_calibrated"]["brier"] / results["training_region_hour_baseline"]["raw"]["brier"]
        results[name]["brier_skill_vs_region_hour_calibrated"] = 1 - results[name]["platt_calibrated"]["brier"] / results["training_region_hour_baseline"]["platt_calibrated"]["brier"]
    preds["forecast_at_utc"] = preds.time.map(utc)
    preds["target_end_utc"] = preds.target_end.map(utc)
    return {"results": results, "predictions": preds, "bundles": bundles,
            "counts": {"training": len(train), "calibration": len(cal), "evaluation_clear": len(test),
                       "evaluation_active_or_unresolved": int(((frame.time >= EVALUATION_START) &
                                                                  (frame.target_end < ANALYSIS_END_EXCLUSIVE) &
                                                                  ~frame.known_clear).sum()),
                       "evaluation_positive": int(y_test.sum())}}


def duration_feature_matrix(episodes: pd.DataFrame, events: list) -> tuple[pd.DataFrame, list[str]]:
    """Construct onset-time-only features for episode duration prediction."""
    onsets, changes = replay(events)
    start = pd.to_datetime(episodes.started_at_utc, utc=True).dt.as_unit("s").astype("int64").to_numpy()
    query = start - 1  # exclude the target episode's own start from input state
    local = pd.to_datetime(start, unit="s", utc=True).tz_convert("Europe/Kyiv")
    hour = np.asarray(local.hour) + np.asarray(local.minute) / 60
    weekday = np.asarray(local.dayofweek)
    doy = np.asarray(local.dayofyear)
    x = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7), "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "year_sin": np.sin(2 * np.pi * doy / 365.25), "year_cos": np.cos(2 * np.pi * doy / 365.25),
    })
    reg_index = episodes.oblast.map({r: i for i, r in enumerate(REGION_NAMES)}).to_numpy(dtype=int)
    for i, rid in enumerate(REGION_IDS):
        x[f"region_{rid}"] = (reg_index == i).astype(np.int8)
    active_columns = []
    active_total = np.zeros(len(query), dtype=np.int8)
    for i, rid in enumerate(REGION_IDS):
        times, values = changes[i]
        ix = np.searchsorted(times, query, side="right") - 1
        active = np.zeros(len(query), dtype=np.int8)
        ok = ix >= 0
        active[ok] = values[ix[ok]].astype(np.int8)
        active[reg_index == i] = 0
        x[f"other_active_{rid}"] = active
        active_columns.append(f"other_active_{rid}")
        active_total += active
        onset_times = onsets[i]
        count = np.searchsorted(onset_times, query, side="right") - np.searchsorted(onset_times, query - 6 * 3600, side="right")
        count[reg_index == i] = 0
        x[f"other_onsets_6h_{rid}"] = np.log1p(count)
    x["other_regions_active_count"] = active_total
    columns = list(x.columns)
    return x, columns


def fit_duration_models() -> dict:
    out_dir = ROOT / "reports" / "learning_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    results, predictions = {}, []
    train_cutoff = epoch("2025-01-01T00:00:00+00:00")
    calibration_cutoff = epoch("2026-01-01T00:00:00+00:00")
    eval_end = epoch("2026-08-01T00:00:00+00:00")
    for source in ("official", "volunteer"):
        events, _ = load_events(source, ANALYSIS_END_EXCLUSIVE - 1)
        if events and events[-1][0] >= ANALYSIS_END_EXCLUSIVE:
            raise ValueError("Duration feature event exceeds the agreed analysis window")
        windows = [ROOT / "data/processed/2024-01-01_2025-12-31" / f"{source}_oblast_episodes.csv",
                   ROOT / "data/processed" / f"{source}_oblast_episodes.csv"]
        frames = [pd.read_csv(p) for p in windows]
        ep = pd.concat(frames, ignore_index=True)
        ep["start_epoch"] = pd.to_datetime(ep.started_at_utc, utc=True).dt.as_unit("s").astype("int64")
        ep["end_epoch"] = pd.to_datetime(ep.ended_at_utc, utc=True).dt.as_unit("s").astype("int64")
        ep["duration_minutes"] = pd.to_numeric(ep.duration_minutes, errors="coerce")
        ep = ep.loc[ep.duration_target_eligible.astype(bool) & ep.duration_minutes.notna() &
                    np.isfinite(ep.duration_minutes) & (ep.duration_minutes > 0)].copy().reset_index(drop=True)
        x, cols = duration_feature_matrix(ep, events)
        cutoff = ep.start_epoch.to_numpy()
        # A duration label is available only after its observed end, not its start.
        train_mask = (cutoff < train_cutoff) & (ep.start_epoch >= TRAIN_START) & (ep.end_epoch < train_cutoff)
        calibration_mask = (ep.start_epoch >= train_cutoff) & (ep.end_epoch < calibration_cutoff)
        test_mask = (ep.start_epoch >= calibration_cutoff) & (ep.end_epoch < eval_end)
        y = ep.duration_minutes.to_numpy(dtype=float)
        if train_mask.sum() < 1000 or calibration_mask.sum() < 100 or test_mask.sum() < 100:
            results[source] = {"status": "insufficient completed episodes", "train_n": int(train_mask.sum()), "calibration_n": int(calibration_mask.sum()), "test_n": int(test_mask.sum())}
            continue
        models = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=160,
                  learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=2.0, early_stopping=False, random_state=24) for q in (0.1, 0.5, 0.9)}
        for m in models.values():
            m.fit(x.loc[train_mask, cols].to_numpy(dtype=float), y[train_mask])
        Xcal = x.loc[calibration_mask, cols].to_numpy(dtype=float)
        cal_lo = np.maximum(0, models[0.1].predict(Xcal))
        cal_hi = np.maximum(cal_lo, models[0.9].predict(Xcal))
        cal_y = y[calibration_mask]
        nonconformity = np.maximum.reduce((cal_lo - cal_y, cal_y - cal_hi, np.zeros(len(cal_y))))
        # 80% conformal expansion, calibrated on 2025 only; evaluation remains 2026.
        rank = min(1.0, np.ceil((len(nonconformity) + 1) * 0.80) / len(nonconformity))
        interval_adjustment = float(np.quantile(nonconformity, rank, method="higher"))
        Xtest = x.loc[test_mask, cols].to_numpy(dtype=float)
        lo = np.maximum(0, models[0.1].predict(Xtest) - interval_adjustment)
        med = np.maximum(0, models[0.5].predict(Xtest))
        hi = np.maximum(med, models[0.9].predict(Xtest) + interval_adjustment)
        ytest = y[test_mask]
        reg_test = ep.loc[test_mask, "oblast"].to_numpy()
        regional_median = ep.loc[train_mask].groupby("oblast").duration_minutes.median().to_dict()
        baseline = np.asarray([regional_median.get(r, float(np.median(y[train_mask]))) for r in reg_test])
        results[source] = {
            "training_rows": int(train_mask.sum()), "interval_calibration_rows": int(calibration_mask.sum()),
            "evaluation_rows": int(test_mask.sum()),
            "training_cutoff_utc": utc(train_cutoff), "evaluation_end_exclusive_utc": utc(eval_end),
            "interval_calibration_end_exclusive_utc": utc(calibration_cutoff),
            "target": "total duration from start, observed and eligible episodes only",
            "median_model_mae_minutes": float(mean_absolute_error(ytest, med)),
            "region_median_baseline_mae_minutes": float(mean_absolute_error(ytest, baseline)),
            "median_model_median_absolute_error_minutes": float(np.median(np.abs(ytest - med))),
            "p10_p90_interval_coverage": float(np.mean((ytest >= lo) & (ytest <= hi))),
            "p10_p90_mean_width_minutes": float(np.mean(hi - lo)),
            "2025_conformal_interval_expansion_minutes": interval_adjustment,
            "observed_duration_median_minutes": float(np.median(ytest)),
            "predicted_duration_median_minutes": float(np.median(med)),
            "excluded_ineligible_episodes": int((~pd.concat(frames, ignore_index=True).duration_target_eligible.astype(bool)).sum()),
        }
        p = ep.loc[test_mask, ["episode_id", "oblast", "started_at_utc", "duration_minutes"]].copy()
        p["p10_minutes"], p["p50_minutes"], p["p90_minutes"] = lo, med, hi
        p["region_median_baseline_minutes"] = baseline
        p["source"] = source
        predictions.append(p)
        joblib.dump({"models": models, "features": cols, "source": source,
                     "fit_cutoff_utc": utc(train_cutoff), "training_rows": int(train_mask.sum()),
                     "interval_calibration_cutoff_utc": utc(calibration_cutoff),
                     "interval_expansion_minutes": interval_adjustment,
                     "target": "eligible complete episode duration in minutes"},
                    ROOT / "models" / f"duration_{source}_2026-01-01.joblib")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(out_dir / "duration_predictions.csv.gz", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-only", action="store_true", help="skip onset model and refit duration models only")
    args = parser.parse_args()
    if args.duration_only:
        duration = fit_duration_models()
        (ROOT / "reports/learning_v2/duration_metrics.json").write_text(json.dumps(duration, indent=2, allow_nan=False) + "\n")
        print(json.dumps(duration, indent=2), flush=True)
        return
    events, _ = load_events("volunteer", ANALYSIS_END_EXCLUSIVE - 1)
    if events and events[-1][0] >= ANALYSIS_END_EXCLUSIVE:
        raise ValueError("Onset event exceeds the agreed analysis window")
    # Exclude the final hour: its outcome could include an event at 1 October 00:00.
    grid = np.arange(TRAIN_START, ANALYSIS_END_EXCLUSIVE - HORIZON, STEP, dtype=np.int64)
    print(f"Building volunteer hourly feature history: {len(grid):,} origins × {len(REGION_NAMES)} regions", flush=True)
    frame = attach_labels(feature_frame(events, grid), events, HORIZON)
    frame = add_cross_region_features(frame, events)
    result = fit_onset_models(frame)
    out_dir = ROOT / "reports" / "learning_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(out_dir / "onset_evaluation_predictions.csv.gz", index=False)
    for name, bundle in result["bundles"].items():
        joblib.dump(bundle, ROOT / "models" / f"onset_{name}_2026-09-01.joblib")
    duration = fit_duration_models()
    (out_dir / "duration_metrics.json").write_text(json.dumps(duration, indent=2, allow_nan=False) + "\n")
    digest = hashlib.sha256()
    for event in events:
        digest.update(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    report = {
        "onset_source": "volunteer", "in_window_event_stream_sha256": digest.hexdigest(),
        "in_window_event_count": len(events),
        "training_start_utc": utc(TRAIN_START), "model_fit_cutoff_utc": utc(CALIBRATION_START),
        "calibration_start_utc": utc(CALIBRATION_START), "evaluation_start_utc": utc(EVALUATION_START),
        "analysis_end_exclusive_utc": utc(ANALYSIS_END_EXCLUSIVE), "horizon_minutes": 60,
        "training_calibration_evaluation_counts": result["counts"], "onset_metrics": result["results"],
        "duration_metrics": duration,
        "leakage_audit": {"training_outcomes_end_before_calibration": True,
                           "calibration_outcomes_end_before_evaluation": True,
                           "all_evaluation_outcomes_end_by_30_september_2026": True,
                           "feature_event_times_at_or_before_origin": True,
                           "duration_features_use_state_at_start_minus_one_second": True,
                           "duration_labels_use_eligible_complete_episodes_only": True},
        "limitations": [
            "September 2026 is a volunteer-only evaluation period and was previously inspected; it is not a fresh holdout.",
            "Archive timestamps proxy availability; source receipt/edit histories and feed uptime are absent.",
            "Absent records are not verified negatives. Results are conditional on reconstructed clear state and archive completeness.",
            "Duration models predict total duration at alert onset, not remaining duration later in an alert.",
            "Duration labels exclude imputed, ambiguous, clipped, and >7-day union episodes; selection may bias durations downward.",
            "Generic onset probabilities are not separated by missile/UAV because historic 2024–25 alerts lack reliable threat labels.",
            "Cross-region covariates are associations, not routes or causal effects; September is insufficient for strong regional conclusions.",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Model iteration 2: cross-region onset and alert duration", "",
        "## Onset probability", "",
        "Question: from a reconstructed clear state, what is the probability a region becomes active in the next 60 minutes?",
        f"Evaluation uses volunteer records through 30 September 2026. The September 1–30 comparison contains {result['counts']['evaluation_clear']:,} clear-region forecast hours; {result['counts']['evaluation_active_or_unresolved']:,} additional region-hours were already active or unresolved and abstained. September has been inspected previously, so this is a development comparison rather than a fresh holdout.", "",
        "| Model | Calibration | Brier | Log loss | ROC AUC | Average precision | Brier skill vs calibrated region/hour |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("logistic_current_features", "logistic_cross_region", "hist_gradient_boosting_cross_region"):
        row = report["onset_metrics"][name]
        for label, cal_key in (("raw", "raw"), ("Platt", "platt_calibrated")):
            m = row[cal_key]
            skill = row.get("brier_skill_vs_region_hour_calibrated") if label == "Platt" else None
            lines.append(f"| {name.replace('_', ' ')} | {label} | {m['brier']:.5f} | {m['log_loss']:.5f} | {m['roc_auc']:.3f} | {m['average_precision']:.3f} | {f'{skill:.1%}' if skill is not None else '—'} |")
    winner = min((name for name in ("logistic_current_features", "logistic_cross_region", "hist_gradient_boosting_cross_region")),
                 key=lambda name: report["onset_metrics"][name]["platt_calibrated"]["brier"])
    lines += ["", f"Among these three specifications, {winner.replace('_', ' ')} has the lowest calibrated Brier score in September. Platt calibration was fit on August. These results are exploratory because September was previously inspected and there is no later untouched test period within the agreed cutoff.", "",
              "## Duration at alert start", "",
              "Separate source-specific quantile gradient-boosting models estimate total episode duration in minutes at its start. Models use 2024 complete episodes, calibrate interval width on 2025, and evaluate January–July 2026. The evaluation includes completed, non-imputed episodes only.", "",
              "| Source | Train / calibration / evaluation | Model MAE (min) | Region-median baseline MAE | P10–P90 coverage | Mean interval width (min) |",
              "|---|---:|---:|---:|---:|---:|"]
    for source, d in report["duration_metrics"].items():
        lines.append(f"| {source} | {d.get('training_rows', 0):,} / {d.get('interval_calibration_rows', 0):,} / {d.get('evaluation_rows', 0):,} | {d.get('median_model_mae_minutes', float('nan')):.1f} | {d.get('region_median_baseline_mae_minutes', float('nan')):.1f} | {d.get('p10_p90_interval_coverage', float('nan')):.1%} | {d.get('p10_p90_mean_width_minutes', float('nan')):.1f} |")
    lines += ["", "Duration predictions are not yet a demonstrated improvement over the simple region-median benchmark, and the nominal 80% intervals cover fewer than 80% of 2026 cases. Do not present them as reliable operational estimates. They predict total duration at onset, not remaining time once an alert is underway.", "",
              "## Leakage controls and limits", "",
              "The onset model trains through July, calibrates on August, and is evaluated on September 1–30. No event after 30 September is replayed; the final hourly origin is omitted because its full target window would reach 1 October. Training and calibration targets end before the next split, and every onset feature is computed from source events timestamped no later than the forecast origin. Duration models train through 2024, calibrate only on 2025, and evaluate on January–July 2026; duration inputs use state immediately before the episode start. Censored, estimated, ambiguous, window-clipped and >7-day union durations are excluded.", "",
              "Archive event timestamps proxy availability. Telegram reception/edit history and historical feed uptime are unavailable. Absence from an archive is not verified ground truth. The official-derived archive stops on September 7, so full September onset evaluation is volunteer-only. The target remains any mapped locality active in a region; threat-type separation is not yet learned.", ""]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"counts": result["counts"], "onset_metrics": result["results"], "duration_metrics": duration}, indent=2), flush=True)


if __name__ == "__main__":
    main()
