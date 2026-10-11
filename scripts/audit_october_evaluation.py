#!/usr/bin/env python3
"""Validate saved October scores and reproduce forecasts from event prefixes."""
import csv
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from alert_model import ROOT, epoch, load_events, sha256
from evaluate_october_2026 import (CANDIDATES, FIT_END, MODELS, OUT, TEST_END,
    TEST_START, duration_predictions, make_features, predict_onset)
from train_alert_model_v2 import duration_feature_matrix, metric_pack


def main():
    report = json.loads((OUT / "metrics.json").read_text())
    pred = pd.read_json(OUT / "predictions.jsonl.gz", lines=True, convert_dates=False)
    assert not pred.duplicated(["time", "oblast_id"]).any()
    assert (pred.time >= TEST_START).all()
    assert (pred.target_end == pred.time + 3600).all()
    assert (pred.target_end < TEST_END).all()
    assert (pred.max_feature_event_time <= pred.time).all()
    assert len(pred) == report["coverage"]["clear_region_hours"]
    for name, recorded in report["onset_metrics"].items():
        calculated = metric_pack(pred.y.to_numpy(), pred[name].to_numpy())
        for metric in ("brier", "log_loss", "roc_auc", "average_precision", "mean_probability"):
            np.testing.assert_allclose(calculated[metric], recorded[metric], atol=1e-12, rtol=0)
    reproduced = []
    for stamp in ("2026-10-01T00:00:00Z", "2026-10-05T12:00:00Z", "2026-10-10T18:00:00Z"):
        at = epoch(stamp)
        events, audit = load_events("volunteer", at)
        assert audit["raw_sha256"] == report["source_audit"]["historical"]["raw_sha256"]
        frame = make_features(events, np.array([at], dtype=np.int64))
        frame = frame.loc[frame.known_clear].sort_values("oblast_id")
        saved = pred.loc[pred.time == at].sort_values("oblast_id")
        assert frame.oblast_id.tolist() == saved.oblast_id.tolist()
        for name in (*CANDIDATES, "region_hour_baseline"):
            b = joblib.load(MODELS / f"{name}.joblib")
            assert b["max_training_target_end"] < FIT_END
            assert b["max_calibration_target_end"] < TEST_START <= at
            p = predict_onset(b, frame)[1]
            np.testing.assert_allclose(p, saved["refreshed_" + name], atol=1e-12, rtol=0)
        reproduced.append({"forecast_utc": stamp, "regions": len(saved), "models": 4})
    duration = pd.read_json(OUT / "duration_predictions.jsonl.gz", lines=True, convert_dates=False)
    b = joblib.load(MODELS / "duration.joblib")
    assert b["max_training_end_epoch"] < FIT_END
    assert b["max_calibration_end_epoch"] < TEST_START
    assert (duration.end_epoch < TEST_END).all()
    assert (duration.start_epoch >= TEST_START).all()
    assert duration.duration_target_eligible.all()
    assert (duration.p10_minutes <= duration.p50_minutes).all()
    assert (duration.p50_minutes <= duration.p90_minutes).all()
    sample = duration.iloc[[len(duration) // 2]].copy()
    events, _ = load_events("volunteer", int(sample.start_epoch.iloc[0]) - 1)
    x, _ = duration_feature_matrix(sample, events)
    np.testing.assert_allclose(duration_predictions(b, x)[0],
        sample[["p10_minutes", "p50_minutes", "p90_minutes"]].iloc[0].to_numpy(dtype=float), atol=1e-10, rtol=0)
    raw_days = Counter()
    with (ROOT / "data/raw/volunteer_data_en.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            start = row["started_at"]
            if "2026-10-01" <= start < "2026-10-11":
                raw_days[start[:10]] += 1
    assert len(raw_days) == 10
    result = {"all_checks_passed": True, "saved_onset_rows_checked": len(pred),
        "saved_duration_rows_checked": len(duration), "prefix_predictions_reproduced": reproduced,
        "duration_prefix_prediction_reproduced": True,
        "raw_volunteer_october_start_rows_by_day": dict(sorted(raw_days.items())),
        "code_sha256": {name: sha256(ROOT / name) for name in (
            "scripts/alert_model.py", "scripts/train_alert_model_v2.py",
            "scripts/evaluate_october_2026.py", "scripts/audit_october_evaluation.py")},
        "scope": "Timestamp and reproducibility checks; archive receipt/revision history remains unavailable"}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
