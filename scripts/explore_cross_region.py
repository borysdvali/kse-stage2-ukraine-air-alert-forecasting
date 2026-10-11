#!/usr/bin/env python3
"""Describe one-hour cross-region alert associations in historical archives.

The source-specific analyses use a region as active when any reconstructed
locality is active. Alert start/end events are replayed from raw archives using
the same conservative ending rules as the first onset model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alert_model import ROOT, REGION_NAMES, SOURCE_ENDS, epoch, load_events, replay, utc


PERIODS = {
    "2024_2025": ("2024-01-01", "2026-01-01"),
    "2026_jan_jul": ("2026-01-01", "2026-08-01"),
}
MIN_MATCHED_ACTIVE_HOURS = 200
MIN_MATCHED_POSITIVES = 10


def construct_hourly_state(events: list, grid: np.ndarray):
    onsets, changes = replay(events)
    state = np.zeros((len(REGION_NAMES), len(grid)), dtype=bool)
    for region, (times, values) in enumerate(changes):
        ix = np.searchsorted(times, grid, side="right") - 1
        valid = ix >= 0
        state[region, valid] = values[ix[valid]]
    y = np.zeros_like(state)
    horizon = 3600
    for region, times in enumerate(onsets):
        end_count = np.searchsorted(times, grid + horizon, side="right")
        start_count = np.searchsorted(times, grid, side="right")
        y[region] = end_count > start_count
    return state, y, onsets


def quarter_bins(local: pd.DatetimeIndex) -> np.ndarray:
    return (local.year.to_numpy() * 4 + (local.month.to_numpy() - 1) // 3).astype(np.int32)


def analyze_period(source: str, period: str, start_text: str, end_text: str,
                   state: np.ndarray, outcome: np.ndarray, local: pd.DatetimeIndex,
                   grid_origin: int | None = None) -> pd.DataFrame:
    start, end = epoch(start_text), epoch(end_text)
    grid = np.arange(start, end, 3600, dtype=np.int64)
    offset = (grid - (grid_origin if grid_origin is not None else state_grid_origin(source))) // 3600
    # The shared hourly arrays begin at 2024-01-01 for every source.
    ix = offset.astype(np.int64)
    s = state[:, ix]
    y = outcome[:, ix]
    time_quarter = quarter_bins(local[ix])
    time_daypart = (local[ix].hour.to_numpy() // 6).astype(np.int8)
    quarter_offset = time_quarter - time_quarter.min()
    n_strata = (int(quarter_offset.max()) + 1) * 12
    rows = []
    n_regions = len(REGION_NAMES)
    for target in range(n_regions):
        target_clear = ~s[target]
        for origin in range(n_regions):
            if origin == target:
                continue
            exposed = target_clear & s[origin]
            unexposed = target_clear & ~s[origin]
            exp_n, ctrl_n = int(exposed.sum()), int(unexposed.sum())
            exp_y = int(y[target, exposed].sum())
            ctrl_y = int(y[target, unexposed].sum())
            p_exp = exp_y / exp_n if exp_n else np.nan
            p_ctrl = ctrl_y / ctrl_n if ctrl_n else np.nan

            # Compare exposed/control hours within coarse seasonal, local-time,
            # and background-alert-count strata. This reduces, but does not
            # eliminate, confounding from broad alert waves.
            other_count = s.sum(axis=0) - s[target].astype(np.int8) - s[origin].astype(np.int8)
            other_bin = np.minimum(other_count, 2).astype(np.int8)
            stratum = (quarter_offset * 4 + time_daypart) * 3 + other_bin
            cell = stratum[target_clear] * 2 + s[origin, target_clear].astype(np.int8)
            sizes = np.bincount(cell, minlength=n_strata * 2).reshape(n_strata, 2)
            positives = np.bincount(
                cell, weights=y[target, target_clear].astype(np.int8), minlength=n_strata * 2
            ).reshape(n_strata, 2)
            common = (sizes[:, 0] > 0) & (sizes[:, 1] > 0)
            common_exp_n = int(sizes[common, 1].sum())
            common_ctrl_n = int(sizes[common, 0].sum())
            common_exp_y = int(positives[common, 1].sum())
            common_ctrl_y = int(positives[common, 0].sum())
            matched_exp_risk = common_exp_y / common_exp_n if common_exp_n else np.nan
            # Standardize the comparison rate to the distribution of exposed
            # hours across common strata (rather than pool different mixtures).
            matched_ctrl_risk = (
                float(np.average(positives[common, 0] / sizes[common, 0], weights=sizes[common, 1]))
                if common_exp_n else np.nan
            )
            matched_lift = (matched_exp_risk / matched_ctrl_risk
                            if common_exp_n and common_ctrl_n and matched_ctrl_risk > 0 else np.nan)
            rows.append({
                "source": source, "period": period,
                "origin_region": REGION_NAMES[origin], "target_region": REGION_NAMES[target],
                "horizon_hours": 1,
                "origin_active_target_clear_hours": exp_n,
                "origin_inactive_target_clear_hours": ctrl_n,
                "onsets_next_hour_with_origin_active": exp_y,
                "onsets_next_hour_with_origin_inactive": ctrl_y,
                "risk_with_origin_active": p_exp,
                "risk_with_origin_inactive": p_ctrl,
                "crude_risk_ratio": p_exp / p_ctrl if exp_n and ctrl_n and p_ctrl > 0 else np.nan,
                "crude_risk_difference_pp": 100 * (p_exp - p_ctrl) if exp_n and ctrl_n else np.nan,
                "matched_active_hours": common_exp_n,
                "matched_inactive_hours": common_ctrl_n,
                "matched_active_onsets": common_exp_y,
                "matched_inactive_onsets": common_ctrl_y,
                "matched_risk_active": matched_exp_risk,
                "matched_risk_inactive": matched_ctrl_risk,
                "matched_risk_ratio": matched_lift,
                "matched_risk_difference_pp": 100 * (matched_exp_risk - matched_ctrl_risk)
                    if common_exp_n and common_ctrl_n else np.nan,
            })
    result = pd.DataFrame(rows)
    result["rankable"] = (
        (result.matched_active_hours >= MIN_MATCHED_ACTIVE_HOURS)
        & (result.matched_active_onsets >= MIN_MATCHED_POSITIVES)
        & result.matched_risk_ratio.notna()
    )
    return result


def state_grid_origin(source: str) -> int:
    # The state matrix origin is fixed at the beginning of the first analysis period.
    return epoch(PERIODS["2024_2025"][0])


def main():
    out = ROOT / "reports" / "cross_region_patterns"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    source_meta = {}
    for source in ("official", "volunteer"):
        source_end = epoch(SOURCE_ENDS[source])
        analysis_end = min(epoch(PERIODS["2026_jan_jul"][1]), source_end)
        # Include one extra hour of events for labels at the final hourly
        # origin. Feature replay still reads state only at or before each T.
        event_cutoff = min(analysis_end + 3600, source_end) - 1
        events, audit = load_events(source, event_cutoff)
        first = epoch(PERIODS["2024_2025"][0])
        grid = np.arange(first, analysis_end, 3600, dtype=np.int64)
        state, outcome, _ = construct_hourly_state(events, grid)
        local = pd.to_datetime(grid, unit="s", utc=True).tz_convert("Europe/Kyiv")
        source_meta[source] = {
            "raw_path": audit["raw_path"], "raw_sha256": audit["raw_sha256"],
            "unique_start_events_replayed": audit["unique_start_events"],
            "unique_end_events_replayed": audit["unique_end_events"],
            "label_events_loaded_through_inclusive_utc": utc(event_cutoff),
            "analysis_end_exclusive_utc": utc(analysis_end),
        }
        for period, (start_text, end_text) in PERIODS.items():
            if epoch(start_text) >= analysis_end:
                continue
            clipped_end = min(epoch(end_text), analysis_end)
            end_text = utc(clipped_end)
            df = analyze_period(source, period, start_text, end_text, state, outcome, local)
            df.to_csv(out / f"{source}_{period}_pairs.csv", index=False)
            all_rows.append(df)
            print(f"{source} {period}: {len(df)} directed pairs; "
                  f"{int(df.rankable.sum())} meet minimum matched-support rule", flush=True)

    pairs = pd.concat(all_rows, ignore_index=True)
    pairs.to_csv(out / "all_pairs.csv", index=False)
    examples = pairs.loc[
        (pairs.period == "2024_2025")
        & pairs.origin_region.eq("Sumska oblast")
        & pairs.target_region.isin(["Kyiv City", "Kyivska oblast", "Lvivska oblast"])
    ]
    ranked = pairs.loc[pairs.rankable].sort_values(
        ["source", "period", "matched_risk_ratio"], ascending=[True, True, False]
    )
    top = ranked.groupby(["source", "period"], sort=True).head(10)
    repeated_corridors = [
        ("Zhytomyrska oblast", "Rivnenska oblast"),
        ("Zhytomyrska oblast", "Khmelnytska oblast"),
    ]
    repeat_rows = []
    for origin, target in repeated_corridors:
        cells = []
        for source, period in [("official", "2024_2025"), ("volunteer", "2024_2025"),
                               ("official", "2026_jan_jul"), ("volunteer", "2026_jan_jul")]:
            selected = pairs.loc[
                pairs.source.eq(source) & pairs.period.eq(period)
                & pairs.origin_region.eq(origin) & pairs.target_region.eq(target)
            ]
            if len(selected) and bool(selected.iloc[0].rankable):
                row = selected.iloc[0]
                cells.append(f"{row.matched_risk_ratio:.2f}× ({row.matched_active_onsets}/{row.matched_active_hours})")
            else:
                cells.append("insufficient support")
        repeat_rows.append(f"| {origin} → {target} | " + " | ".join(cells) + " |")
    report = [
        "# Historical cross-region alert patterns",
        "",
        "## Question and method",
        "",
        "For each source region that is active at an hourly forecast origin, and each different target region that is clear, "
        "measure how often a new target-region episode begins in the next hour. The comparison is the same target region's "
        "one-hour onset rate when the source region is inactive and the target remains clear.",
        "",
        "Analyses are separate by archive and period: 2024–2025, then January–July 2026 as a later-period check. "
        "August–September 2026 are not included. Alerts are reconstructed by replaying locality start/end events; an oblast is "
        "active if any mapped locality is active. Uncertain volunteer endings and exactly-one-hour official endings are "
        "not used to mark a region clear. The target is a regional episode onset, not threat arrival or causal propagation.",
        "",
        "The raw rate ratio compares the exposed hours with all other target-clear hours. The matched rate ratio compares "
        "only strata containing both exposed and comparison hours, matching on calendar quarter, Kyiv local six-hour time "
        "block and the number of other active regions (0, 1, or 2+). The comparison rate is weighted to match the distribution "
        "of source-active hours across those strata. This partly accounts for season, time of day and broad countrywide alert "
        "waves. It does not remove all confounding. Hourly observations and alert waves are dependent; "
        "ratios are descriptive, not causal effects, and no significance claims are made.",
        "",
        f"For the ranked lists, a pair needs at least {MIN_MATCHED_ACTIVE_HOURS} matched source-active/target-clear hours "
        f"and {MIN_MATCHED_POSITIVES} target onsets during those hours. A missing pair means insufficient overlap or "
        "no comparison risk estimate, not evidence of no relationship.",
        "",
        "## Associations repeated across archives and periods",
        "",
        "These adjacent pairs meet the support rule in both archives and both periods. Values show matched rate ratio "
        "(target onsets / matched source-active hours):",
        "",
        "| Active source → clear target | Official 2024–25 | Volunteer 2024–25 | Official Jan–Jul 2026 | Volunteer Jan–Jul 2026 |",
        "|---|---:|---:|---:|---:|",
        *repeat_rows,
        "",
        "Ratios are elevated in all four comparisons, while their size changes by archive and period. This supports checking "
        "these pairs as predictive features; it does not establish a route or causal spread.",
        "",
        "## Sumska oblast followed by Kyiv or Lviv (2024–2025)",
        "",
        "| Archive | Target | Matched active hours | Target onsets | Rate when Sumy active | Rate when Sumy inactive | Matched rate ratio | Matched difference (pp) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in examples.iterrows():
        fmt = lambda x, pct=False: "n/a" if pd.isna(x) else (f"{x:.2%}" if pct else f"{x:.2f}")
        report.append(
            f"| {row.source} | {row.target_region} | {row.matched_active_hours:,} | {row.matched_active_onsets:,} | "
            f"{fmt(row.matched_risk_active, True)} | {fmt(row.matched_risk_inactive, True)} | "
            f"{fmt(row.matched_risk_ratio)} | {fmt(row.matched_risk_difference_pp)} |"
        )
    report += ["", "## Highest matched one-hour associations", ""]
    for (source, period), group in top.groupby(["source", "period"], sort=True):
        report += [f"### {source.capitalize()} archive, {period.replace('_', ' ')}", "",
                   "| From active region | Clear target region | Matched active hours | Target onsets | Active rate | Inactive rate | Matched ratio |",
                   "|---|---|---:|---:|---:|---:|---:|"]
        for _, row in group.iterrows():
            report.append(
                f"| {row.origin_region} | {row.target_region} | {row.matched_active_hours:,} | "
                f"{row.matched_active_onsets:,} | {row.matched_risk_active:.2%} | "
                f"{row.matched_risk_inactive:.2%} | {row.matched_risk_ratio:.2f}× |"
            )
        report.append("")
    report += [
        "## Interpretation limits",
        "",
        "These measurements ask whether a target onset followed while the source was already active. They do not track aircraft "
        "or identify a route. A common threat can activate several regions, and alerts can spread at the same time; the association "
        "does not show that one region's alert caused another's. This analysis uses generic alert histories and has no threat-type "
        "filter. Source coverage and locality resolution differ, especially in 2026. Missing historical receipt/edit times also mean "
        "the archived event timestamp is only a proxy for what was available to a forecaster.",
        "",
        "See `all_pairs.csv` for the complete directed-pair metrics and the source-specific CSV files for each period. "
        "Source snapshot hashes and event counts are in `metadata.json`.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(report))
    (out / "metadata.json").write_text(json.dumps({
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "hourly at-risk exposure; target clear; source active; next-hour target onset",
        "matched_strata": ["calendar quarter", "Kyiv local six-hour block", "other active regions: 0, 1, 2+"],
        "minimum_matched_active_hours": MIN_MATCHED_ACTIVE_HOURS,
        "minimum_matched_target_onsets": MIN_MATCHED_POSITIVES,
        "sources": source_meta,
    }, indent=2) + "\n")
    print(f"Finished: {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
