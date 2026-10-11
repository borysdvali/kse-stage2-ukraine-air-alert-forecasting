#!/usr/bin/env python3
"""Independently check saved predictions and reproduce a holdout forecast."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from alert_model import ROOT, epoch, feature_frame, load_events, predict_bundle


def main():
    results = {}
    for source in ("official", "volunteer"):
        folder = ROOT / "reports" / "learning_baseline" / source
        p = pd.read_csv(folder / "predictions.csv.gz")
        metadata = json.loads((folder / "metrics.json").read_text())
        fit = pd.to_datetime(p.fit_cutoff_utc, utc=True)
        end = pd.to_datetime(p.training_max_target_end_utc, utc=True)
        origin = pd.to_datetime(p.forecast_at_utc, utc=True)
        assert (end < fit).all()
        assert (fit <= origin).all()
        assert (p.max_feature_event_time <= p.time).all()
        assert (p.target_end == p.time + metadata["horizon_minutes"] * 60).all()
        assert (p.target_end < epoch(metadata["coverage_end_exclusive_utc"])).all()
        assert p.probability.between(0, 1).all()
        assert not p.duplicated(["time", "oblast_id"]).any()
        h = p.loc[p.split == "holdout_frozen"]
        assert h.fit_cutoff_utc.nunique() == 1
        sample = h.iloc[len(h) // 2]
        model_path = ROOT / "models" / "learning_baseline" / source / (sample.fit_cutoff_utc[:10] + ".joblib")
        bundle = joblib.load(model_path)
        at = int(sample.time)
        events, audit = load_events(source, at)
        assert audit["raw_sha256"] == bundle["raw_sha256"]
        frame = feature_frame(events, np.array([at]))
        frame = frame.loc[frame.oblast_id == sample.oblast_id]
        assert frame.known_clear.all()
        with threadpool_limits(limits=2):
            probability = float(predict_bundle(bundle, frame)[0][0])
        np.testing.assert_allclose(probability, sample.probability, rtol=0, atol=1e-12)
        future_model_rejected = False
        try:
            predict_bundle({**bundle, "fit_cutoff": at + 1}, frame)
        except ValueError:
            future_model_rejected = True
        assert future_model_rejected
        results[source] = {
            "prediction_rows_checked": len(p), "all_checks_passed": True,
            "sample_origin_utc": sample.forecast_at_utc, "sample_oblast_id": sample.oblast_id,
            "sample_probability_reproduced": probability, "absolute_tolerance": 1e-12,
            "checks": ["training label embargo", "model precedes forecast", "feature event cutoff",
                       "label horizon length and source coverage",
                       "probability bounds", "unique forecast keys", "frozen holdout",
                       "unchanged source hash", "prefix-only forecast reproduction", "future model rejection"],
        }
    path = ROOT / "reports" / "learning_baseline" / "verification.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
