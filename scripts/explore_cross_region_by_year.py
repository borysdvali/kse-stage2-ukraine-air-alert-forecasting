#!/usr/bin/env python3
"""Recalculate every directed one-hour region pair for 2024, 2025, Jan–Sep 2026.

An origin is active and a different target is clear at the hourly observation
time. The outcome is a target onset strictly after that time and within one hour.
Direction is temporal association, not a tracked route or causal transfer.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from alert_model import ROOT, REGION_NAMES, SOURCE_ENDS, epoch, load_events, utc
from explore_cross_region import (
    MIN_MATCHED_ACTIVE_HOURS, MIN_MATCHED_POSITIVES,
    analyze_period, construct_hourly_state,
)

PERIODS = (
    (2024, "2024-01-01", "2025-01-01"),
    (2025, "2025-01-01", "2026-01-01"),
    (2026, "2026-01-01", "2026-10-01"),
)
FIRST = epoch("2024-01-01")
FINAL_EXCLUSIVE = epoch("2026-10-01")
OUTPUT = ROOT / "reports" / "cross_region_by_year"


def serialize(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def top_supported(group: pd.DataFrame, from_region: bool) -> tuple[str | None, float | None]:
    ranked = group.loc[group.rankable & group.matched_risk_ratio.notna()]
    if ranked.empty:
        return None, None
    row = ranked.sort_values(["matched_risk_ratio", "matched_active_onsets"], ascending=False).iloc[0]
    return (str(row.target_region if from_region else row.origin_region), float(row.matched_risk_ratio))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pair_frames, region_rows, coverage_rows, source_metadata = [], [], [], {}

    for source in ("official", "volunteer"):
        # SOURCE_ENDS already conservatively cuts official coverage at Sep 7 00:00
        # and volunteer coverage at Oct 1 00:00. No October event is replayed.
        source_end = min(epoch(SOURCE_ENDS[source]), FINAL_EXCLUSIVE)
        events, audit = load_events(source, source_end - 1)
        if events and max(e[0] for e in events) >= FINAL_EXCLUSIVE:
            raise ValueError("An event after 30 September 2026 entered the analysis")
        grid = np.arange(FIRST, source_end - 3600, 3600, dtype=np.int64)
        state, outcome, _ = construct_hourly_state(events, grid)
        local = pd.to_datetime(grid, unit="s", utc=True).tz_convert("Europe/Kyiv")
        digest = hashlib.sha256()
        for event in events:
            digest.update(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
        source_metadata[source] = {
            "raw_path": audit["raw_path"],
            "in_window_event_stream_sha256": digest.hexdigest(),
            "unique_replayed_events": len(events),
            "source_coverage_end_exclusive_utc": utc(source_end),
        }

        for year, start_text, requested_end_text in PERIODS:
            start = epoch(start_text)
            period_end = min(epoch(requested_end_text), source_end)
            if start >= period_end - 3600:
                continue
            # Keep the whole target hour inside the period and available source
            # coverage. Since the outcome includes an onset exactly at t + 1h,
            # the final origin is two hours before this exclusive boundary.
            final_origin_exclusive = period_end - 3600
            frame = analyze_period(
                source, str(year), start_text, utc(final_origin_exclusive),
                state, outcome, local, grid_origin=FIRST,
            )
            frame.insert(0, "year", year)
            frame["coverage_end_exclusive_utc"] = utc(period_end)
            frame["full_requested_period"] = period_end == epoch(requested_end_text)
            frame["association_direction"] = np.where(
                ~frame.rankable, "insufficient matched support",
                np.where(frame.matched_risk_ratio > 1, "higher onset rate",
                         np.where(frame.matched_risk_ratio < 1, "lower onset rate", "equal observed rate")),
            )
            frame["reverse_matched_risk_ratio"] = [
                frame.loc[(frame.origin_region == r.target_region) &
                          (frame.target_region == r.origin_region), "matched_risk_ratio"].iloc[0]
                for r in frame.itertuples(index=False)
            ]
            pair_frames.append(frame)

            ix_start = (start - FIRST) // 3600
            ix_end = (final_origin_exclusive - FIRST) // 3600
            period_state = state[:, ix_start:ix_end]
            origin_hours = period_state.shape[1]
            if origin_hours != int((period_end - start) // 3600 - 1):
                raise RuntimeError("Period hourly-origin count does not reconcile")
            coverage_rows.append({
                "year": year, "source": source,
                "requested_end_exclusive_utc": utc(epoch(requested_end_text)),
                "actual_end_exclusive_utc": utc(period_end),
                "full_requested_period": bool(period_end == epoch(requested_end_text)),
                "hourly_origins": int(origin_hours), "directed_pairs": len(frame),
                "pairs_passing_support_rule": int(frame.rankable.sum()),
                "supported_higher_rate_pairs": int((frame.rankable & (frame.matched_risk_ratio > 1)).sum()),
                "supported_lower_rate_pairs": int((frame.rankable & (frame.matched_risk_ratio < 1)).sum()),
            })

            for i, region in enumerate(REGION_NAMES):
                outgoing = frame.loc[frame.origin_region == region]
                incoming = frame.loc[frame.target_region == region]
                top_out, top_out_ratio = top_supported(outgoing, from_region=True)
                top_in, top_in_ratio = top_supported(incoming, from_region=False)
                region_rows.append({
                    "year": year, "source": source, "region": region,
                    "coverage_end_exclusive_utc": utc(period_end),
                    "source_active_hours": int(period_state[i].sum()),
                    "target_clear_hours": int((~period_state[i]).sum()),
                    "supported_outgoing_pairs": int(outgoing.rankable.sum()),
                    "supported_outgoing_higher_rate_pairs": int((outgoing.rankable & (outgoing.matched_risk_ratio > 1)).sum()),
                    "top_outgoing_target": top_out,
                    "top_outgoing_matched_risk_ratio": top_out_ratio,
                    "supported_incoming_pairs": int(incoming.rankable.sum()),
                    "supported_incoming_higher_rate_pairs": int((incoming.rankable & (incoming.matched_risk_ratio > 1)).sum()),
                    "top_incoming_source": top_in,
                    "top_incoming_matched_risk_ratio": top_in_ratio,
                })
            print(f"{source} {year}: {len(frame)} directions; {int(frame.rankable.sum())} pass support rule; "
                  f"coverage through {utc(period_end)}", flush=True)

    pairs = pd.concat(pair_frames, ignore_index=True)
    if len(pairs) != 2 * 3 * len(REGION_NAMES) * (len(REGION_NAMES) - 1):
        raise RuntimeError("Expected one row for every source/year/directed pair")
    if pairs.duplicated(["source", "year", "origin_region", "target_region"]).any():
        raise RuntimeError("Duplicate directed pair")
    region_summary = pd.DataFrame(region_rows)
    coverage = pd.DataFrame(coverage_rows).sort_values(["year", "source"]).reset_index(drop=True)

    persistent = []
    for (origin, target), group in pairs.groupby(["origin_region", "target_region"], sort=True):
        if len(group) != 6:
            continue
        if (group.rankable & (group.matched_risk_ratio > 1)).all():
            persistent.append({
                "origin_region": origin, "target_region": target,
                "min_matched_risk_ratio": float(group.matched_risk_ratio.min()),
                "mean_matched_risk_ratio": float(group.matched_risk_ratio.mean()),
                "min_active_onsets": int(group.matched_active_onsets.min()),
                "six_periods_supported": 6,
            })
    persistent.sort(key=lambda row: (-row["min_matched_risk_ratio"], row["origin_region"], row["target_region"]))
    payload = {
        "method": {
            "meaning": "origin region active and target region clear at an hourly origin; target onset in the following hour",
            "matched_strata": ["calendar quarter", "Kyiv local six-hour block", "other active regions: 0, 1, 2+"],
            "support_rule": {"min_matched_active_hours": MIN_MATCHED_ACTIVE_HOURS,
                             "min_matched_target_onsets": MIN_MATCHED_POSITIVES},
            "period_rule": "last one-hour origin omitted so outcome remains inside the requested year/coverage",
            "cutoff_inclusive_utc": "2026-09-30T23:59:59+00:00",
            "interpretation": "temporal association, not flight route, threat type, or causal transfer",
        },
        "source_metadata": source_metadata,
        "coverage": serialize(coverage),
        "pairs": serialize(pairs),
        "regions": serialize(region_summary),
        "persistent_higher_rate_pairs": persistent,
    }
    (OUTPUT / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

    lines = [
        "# Directed regional alert patterns by year", "",
        "At each hourly observation, one region is active and another is clear. The table measures target onsets in the following hour. Direction describes the order of alert states, not a verified physical route.", "",
        "| Year | Archive | Coverage end (exclusive UTC) | Hourly origins | Directed pairs | Pairs passing support | Supported higher-rate pairs |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in coverage.itertuples(index=False):
        lines.append(f"| {r.year} | {r.source} | {r.actual_end_exclusive_utc} | {r.hourly_origins:,} | {r.directed_pairs} | {r.pairs_passing_support_rule} | {r.supported_higher_rate_pairs} |")
    lines += ["", "Support means at least 200 matched active/clear hours and 10 target onsets. The matched comparison controls coarsely for calendar quarter, Kyiv local six-hour block, and the number of other active regions. It is descriptive; 552 directed comparisons per archive and year can produce large ratios by chance, and hourly observations are dependent.", "",
              "The official-derived source ends before September 7, 2026, so its 2026 column does not cover the full nine months. Volunteer records extend through September 30. Neither archive labels 2024–25 alerts by missile versus UAV.", "",
              "## Higher-rate directions repeated in all six source/year comparisons", "",
              "| Active source region | Clear target region | Minimum matched ratio | Minimum target onsets while exposed |",
              "|---|---|---:|---:|"]
    for row in persistent[:25]:
        lines.append(f"| {row['origin_region']} | {row['target_region']} | {row['min_matched_risk_ratio']:.2f}× | {row['min_active_onsets']} |")
    if not persistent:
        lines.append("| None met the support and direction rules in all six comparisons |  |  |  |")
    lines += ["", f"{len(persistent)} directed pairs meet this repeated-direction rule. The complete 3,312-row table and a region-by-region incoming/outgoing summary are in the accompanying workbook.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines))
    print(f"All pairs: {len(pairs)}; repeated higher-rate directions: {len(persistent)}", flush=True)


if __name__ == "__main__":
    main()
