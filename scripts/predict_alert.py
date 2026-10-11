#!/usr/bin/env python3
"""Make a historical forecast using a saved model and an as-of event replay."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from alert_model import REGION_IDS, REGION_NAMES, epoch, feature_frame, load_events, predict_bundle, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trusted local joblib model from this project")
    parser.add_argument("--as-of", required=True, help="UTC timestamp, e.g. 2026-08-03T12:00:00+00:00")
    parser.add_argument("--region", action="append", help="ISO oblast ID, repeatable; default all")
    args = parser.parse_args()
    # joblib uses pickle: only load locally generated trusted model files.
    bundle = joblib.load(args.model)
    at = epoch(args.as_of)
    if at < bundle["fit_cutoff"]:
        parser.error("This model was trained after the requested time")
    if at >= bundle["valid_until_exclusive"]:
        parser.error("Requested time is outside this model's evaluated validity interval")
    if (at - bundle["training_first_origin"]) % bundle["cadence_seconds"]:
        parser.error("Requested time must align with the evaluated forecast cadence")
    if at + bundle["horizon_seconds"] >= bundle["coverage_end_exclusive"]:
        parser.error("The archive does not cover the full forecast horizon")
    selected = args.region or REGION_IDS
    if any(r not in REGION_IDS for r in selected):
        parser.error("Unknown or unsupported region ID")
    events, audit = load_events(bundle["source"], at)
    if audit["raw_sha256"] != bundle["raw_sha256"]:
        parser.error("Source snapshot changed since training; rerun the evaluation")
    frame = feature_frame(events, np.array([at], dtype=np.int64))
    frame = frame.loc[frame.oblast_id.isin(selected)]
    result = []
    with threadpool_limits(limits=2):
        for _, row in frame.iterrows():
            item = {"oblast": row.oblast, "oblast_id": row.oblast_id, "forecast_at_utc": utc(at),
                    "horizon_minutes": bundle["horizon_seconds"] // 60,
                    "source": bundle["source"], "fit_cutoff_utc": utc(bundle["fit_cutoff"])}
            if not row.known_clear:
                item.update(status="active_or_unresolved_in_archive", probability=None)
            elif bundle["region_counts"][int(row.region_index)] == 0:
                item.update(status="no_training_examples_for_region", probability=None)
            else:
                p, _, _ = predict_bundle(bundle, frame.loc[[row.name]])
                item.update(status="historical_research_forecast", probability=float(p[0]))
            result.append(item)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
