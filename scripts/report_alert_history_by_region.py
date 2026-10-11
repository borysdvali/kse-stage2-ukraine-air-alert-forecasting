#!/usr/bin/env python3
"""Descriptive regional alert counts, observed durations, and candidate threat links."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alert_model import REGION_NAMES, ROOT

OUT = ROOT / "reports/regional_history_2024_2026"
PAST = ROOT / "data/processed/2024-01-01_2025-12-31"
NOW = ROOT / "data/processed"
FILES = {
    "volunteer": (PAST / "volunteer_oblast_episodes.csv", NOW / "volunteer_oblast_episodes.csv"),
    "official_priority": (PAST / "canonical_oblast_alert_episodes.csv", NOW / "canonical_oblast_alert_episodes.csv"),
}
EVENTS = NOW / "kpszsu_threat_events.csv"
LINKS = NOW / "alert_threat_event_links.csv"
PERIODS = (("2024", "2024-01-01", "2025-01-01"),
           ("2025", "2025-01-01", "2026-01-01"),
           ("9m 2026", "2026-01-01", "2026-10-01"))
CATEGORIES = {
    "Any UAV": ("propeller_or_unspecified_attack_uav", "jet_attack_uav", "reconnaissance_uav"),
    "Propeller/unspecified attack UAV": ("propeller_or_unspecified_attack_uav",),
    "Jet attack UAV": ("jet_attack_uav",),
    "Reconnaissance UAV": ("reconnaissance_uav",),
    "Any missile": ("ballistic_missile", "cruise_or_air_launched_missile"),
    "Ballistic missile": ("ballistic_missile",),
    "Cruise/air-launched missile": ("cruise_or_air_launched_missile",),
    "Guided aerial bomb": ("guided_aerial_bomb",),
    "Tactical aviation": ("tactical_aviation",),
}
FLAGS = ("has_imputed_end_contributor", "has_ambiguous_official_one_hour_end",
         "needs_long_union_review", "left_censored_by_window", "right_censored_by_window")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    d = pd.read_csv(path)
    if d.episode_id.duplicated().any() or not d.oblast.isin(REGION_NAMES).all():
        raise ValueError(f"Duplicate ID or unmapped region in {path}")
    d["start"] = pd.to_datetime(d.started_at_utc, utc=True)
    d["end"] = pd.to_datetime(d.ended_at_utc, utc=True)
    if (d.end < d.start).any():
        raise ValueError(f"Negative duration in {path}")
    for col in (*FLAGS, "duration_target_eligible"):
        d[col] = d[col].astype(bool)
    if (d.duration_target_eligible & d[list(FLAGS)].any(axis=1)).any():
        raise ValueError(f"Eligible duration conflicts with flags in {path}")
    return d


def days_inside(d, start, end):
    if d.empty:
        return 0.0
    return float(((d.end.clip(upper=end) - d.start.clip(lower=start)).dt.total_seconds()
                  .clip(lower=0).sum()) / 86400)


def longest(d, qualified):
    if qualified:
        d = d.loc[d.duration_target_eligible]
    if d.empty:
        return {"start_utc": None, "end_utc": None, "duration_days": None,
                "quality": "no eligible episode" if qualified else "no recorded episode",
                "episode_id": None, "source_basis": None}
    r = d.sort_values(["duration_minutes", "started_at_utc", "episode_id"],
                      ascending=[False, True, True]).iloc[0]
    flags = [name for name in FLAGS if r[name]]
    return {"start_utc": r.started_at_utc, "end_utc": r.ended_at_utc,
            "duration_days": float((r.end - r.start).total_seconds() / 86400),
            "quality": ", ".join(flags) if flags else "observed ending",
            "episode_id": r.episode_id, "source_basis": r.source_basis}


def source_summary(source, paths):
    past, current = map(load, paths)
    ranks, longest_rows, averages, coverage = [], [], [], []
    for label, a, b in PERIODS:
        start, end = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        d = current if label == "9m 2026" else past
        started = d.loc[(d.start >= start) & (d.start < end)]
        observed = d.loc[d.duration_target_eligible & (d.start < end) & (d.end > start)]
        overlapping = d.loc[(d.start < end) & (d.end > start)]
        if source == "volunteer" and started.oblast.nunique() != len(REGION_NAMES):
            raise ValueError(f"Missing volunteer region in {label}")
        coverage.append({"period": label, "source": source, "alerts_starting": len(started),
                         "eligible_starting": int(started.duration_target_eligible.sum()),
                         "excluded_starting": int((~started.duration_target_eligible).sum()),
                         "long_unions_starting": int(started.needs_long_union_review.sum()),
                         "regions_with_starts": int(started.oblast.nunique())})
        for region in REGION_NAMES:
            begins = started.loc[started.oblast == region]
            eligible = begins.loc[begins.duration_target_eligible]
            o = observed.loc[observed.oblast == region]
            ranks.append({"period": label, "source": source, "region": region,
                          "alert_count": len(begins), "observed_alert_days": days_inside(o, start, end),
                          "duration_eligible_alerts": len(eligible),
                          "excluded_duration_alerts": len(begins) - len(eligible),
                          "long_union_alerts": int(begins.needs_long_union_review.sum()),
                          "unqualified_alerts_overlapping_period": int((~overlapping.loc[overlapping.oblast == region].duration_target_eligible).sum()),
                          "carried_over_eligible_alerts": int((o.start < start).sum())})
            longest_rows.append({"period": label, "source": source, "region": region,
                                 "longest_recorded": longest(begins, False),
                                 "longest_eligible": longest(begins, True)})
            if label == "9m 2026":
                mean = float(((eligible.end - eligible.start).dt.total_seconds() / 86400).mean()) if len(eligible) else None
                averages.append({"source": source, "region": region, "eligible_alerts": len(eligible),
                                 "excluded_duration_alerts": len(begins) - len(eligible),
                                 "average_duration_days": mean,
                                 "average_duration_hours": mean * 24 if mean is not None else None})
    frame = pd.DataFrame(ranks)
    frame["count_rank"] = frame.groupby(["period", "source"]).alert_count.rank(ascending=False, method="min").astype(int)
    frame["time_rank"] = frame.groupby(["period", "source"]).observed_alert_days.rank(ascending=False, method="min").astype(int)
    return frame, longest_rows, averages, coverage


def candidate_threats(canonical):
    event = pd.read_csv(EVENTS)
    link = pd.read_csv(LINKS)
    if event.threat_event_id.duplicated().any() or canonical.canonical_episode_id.duplicated().any():
        raise ValueError("Duplicate threat or canonical episode IDs")
    joined = link.merge(event[["threat_event_id", "event_confidence", "needs_manual_review"]],
                        on="threat_event_id", validate="many_to_one")
    chosen = joined.loc[(joined.link_confidence == "medium_temporal_and_region_overlap") &
                        (joined.event_confidence == "medium") &
                        (~joined.needs_manual_review.astype(bool))].copy()
    chosen = chosen.merge(canonical[["canonical_episode_id", "oblast", "start", "end",
                                     "duration_target_eligible"]], on="canonical_episode_id",
                          validate="many_to_one", suffixes=("_link", ""))
    if (chosen.oblast_link != chosen.oblast).any() or chosen.start.isna().any():
        raise ValueError("Threat links do not match canonical regions")
    start, end = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-10-01", tz="UTC")
    if not ((chosen.start >= start) & (chosen.start < end)).all():
        raise ValueError("Linked episode is outside 2026 window")
    rows = []
    for category, families in CATEGORIES.items():
        subset = chosen.loc[chosen.event_family.isin(families)].drop_duplicates("canonical_episode_id")
        for region in REGION_NAMES:
            d = subset.loc[subset.oblast == region]
            good = d.loc[d.duration_target_eligible]
            rows.append({"period": "9m 2026", "category": category, "region": region,
                         "linked_alert_count": len(d), "linked_alert_days": days_inside(good, start, end),
                         "linked_alerts_with_usable_end": len(good),
                         "linked_alerts_excluded_duration": len(d) - len(good)})
    result = pd.DataFrame(rows)
    result["count_rank"] = result.groupby("category").linked_alert_count.rank(ascending=False, method="min").astype(int)
    result["time_rank"] = result.groupby("category").linked_alert_days.rank(ascending=False, method="min").astype(int)
    audit = {"all_canonical_alerts": len(canonical), "all_candidate_link_rows": len(link),
             "all_candidate_linked_alerts": int(link.canonical_episode_id.nunique()),
             "selected_link_rows": len(chosen), "selected_unique_alerts": int(chosen.canonical_episode_id.nunique()),
             "selection": "medium regional/time link, corroborated event, no manual-review flag",
             "multi_label": "an alert may appear in several threat categories; totals are not additive",
             "causal_label": False}
    return result, audit


def records(d):
    return json.loads(d.to_json(orient="records", double_precision=12))


def markdown(payload):
    lines = ["# Regional air-alert history, 2024–September 2026", "",
             "Period boundaries are UTC. The headline rankings use volunteer region episodes for consistent coverage across all three periods. Official-priority reconciled rankings are a separate cross-check.", "",
             "No collected Telegram archive labels alerts by threat type in 2024 or 2025. Threat rankings for those periods are unavailable, not zero. The 2026 threat results are candidate associations from @kpszsu posts, not confirmed causes. They include only corroborated events with a medium regional/time match and no manual-review flag. They cover " + f"{payload['threat_audit']['selected_unique_alerts']:,} of {payload['threat_audit']['all_canonical_alerts']:,} canonical episodes. An alert may appear in several categories; categories cannot be added together.", "",
             "Alert counts use episode starts in each period. Alert days sum only observed, quality-eligible portions within the period. Estimated, ambiguous, window-censored or >7-day merged episodes are excluded from measured time. An episode crossing a year boundary contributes time to each year it overlaps. All 24 regions, both sources, flags and ranks are in the workbook.", "",
             "## Source coverage", "",
             "| Period | Basis | Alerts starting | Eligible endings | Excluded endings | Long unions |",
             "|---|---|---:|---:|---:|---:|"]
    for r in payload["coverage"]:
        lines.append(f"| {r['period']} | {r['source']} | {r['alerts_starting']:,} | {r['eligible_starting']:,} | {r['excluded_starting']:,} | {r['long_unions_starting']:,} |")
    for metric, title, unit in (("alert_count", "Alert count", "alerts"),
                                ("observed_alert_days", "Measured alert time", "days")):
        lines += ["", "## Five highest regions by " + title.lower(), "",
                  f"| Period | Rank | Region | {title} ({unit}) |", "|---|---:|---|---:|"]
        for period, _, _ in PERIODS:
            group = [r for r in payload["rankings"] if r["period"] == period and r["source"] == "volunteer"]
            for position, r in enumerate(sorted(group, key=lambda x: (-x[metric], x["region"]))[:5], 1):
                value = f"{r[metric]:,}" if metric == "alert_count" else f"{r[metric]:.2f}"
                lines.append(f"| {period} | {position} | {r['region']} | {value} |")
    lines += ["", "## 2026 threat-linked leading regions (official-priority basis)", "",
              "| Category | First by count | Alerts | First by time | Days |",
              "|---|---|---:|---|---:|"]
    for category in CATEGORIES:
        group = [r for r in payload["threat_rankings"] if r["category"] == category]
        count = sorted(group, key=lambda x: (-x["linked_alert_count"], x["region"]))[0]
        time = sorted(group, key=lambda x: (-x["linked_alert_days"], x["region"]))[0]
        lines.append(f"| {category} | {count['region']} | {count['linked_alert_count']:,} | {time['region']} | {time['linked_alert_days']:.2f} |")
    lines += ["", "Threat-linked days use the official-priority canonical regional state. The earlier all-alert days use the volunteer regional state, so the two tables measure different alert definitions and should not be compared or subtracted directly. The linked days are the full eligible alert durations, not the observed duration of each threat.", "",
              "## Longest alerts and 2026 averages", "",
              "For each region and period, the workbook shows exact UTC start and end times for both the longest recorded episode and the longest episode with a usable observed end. The former can be an artificial regional union. In the official-priority source, Kharkiv has one flagged record from 12 May 2024 through the 2024–25 window end; it should not be read as one continuous, verified alert.", "",
              "The table below gives the longest quality-eligible volunteer alert for every region and period. The workbook also shows the longest raw record and its quality flags.", "",
              "| Period | Region | Start UTC | End UTC | Duration (days) |",
              "|---|---|---|---|---:|"]
    for period, _, _ in PERIODS:
        for row in sorted((r for r in payload["longest"] if r["period"] == period and r["source"] == "volunteer"), key=lambda r: r["region"]):
            x = row["longest_eligible"]
            if x["start_utc"] is None:
                lines.append(f"| {period} | {row['region']} | unavailable | unavailable | unavailable |")
            else:
                begin = x["start_utc"].replace("T", " ").replace("+00:00", "")
                finish = x["end_utc"].replace("T", " ").replace("+00:00", "")
                lines.append(f"| {period} | {row['region']} | {begin} | {finish} | {x['duration_days']:.3f} |")
    lines += ["", "The 2026 average is the arithmetic mean of complete, quality-eligible volunteer alert durations. Estimates and censored endings are excluded.", "",
              "| Region | Eligible alerts | Excluded endings | Average (days) | Average (hours) |",
              "|---|---:|---:|---:|---:|"]
    for row in sorted((r for r in payload["averages_2026"] if r["source"] == "volunteer"), key=lambda r: r["region"]):
        lines.append(f"| {row['region']} | {row['eligible_alerts']:,} | {row['excluded_duration_alerts']:,} | {row['average_duration_days']:.4f} | {row['average_duration_hours']:.2f} |")
    lines += ["", "Official-derived source coverage stops on 7 September 2026. Its later 2026 canonical episodes are volunteer supplements. Source counts differ and cannot be summed.", ""]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts, longest_rows, averages, coverage = [], [], [], []
    for source, paths in FILES.items():
        rank, long, avg, cover = source_summary(source, paths)
        parts.append(rank)
        longest_rows.extend(long)
        averages.extend(avg)
        coverage.extend(cover)
    rank = pd.concat(parts, ignore_index=True)
    threat, threat_audit = candidate_threats(load(FILES["official_priority"][1]))
    if len(rank) != 3 * 2 * len(REGION_NAMES) or len(longest_rows) != len(rank) or len(threat) != len(CATEGORIES) * len(REGION_NAMES):
        raise ValueError("Expected one row per region, period and requested category")
    for c in coverage:
        g = rank.loc[(rank.period == c["period"]) & (rank.source == c["source"])]
        assert int(g.alert_count.sum()) == c["alerts_starting"]
        assert int(g.duration_eligible_alerts.sum()) == c["eligible_starting"]
    data = {"method": {"periods": [{"label": p, "start_inclusive_utc": a, "end_exclusive_utc": b}
                                  for p, a, b in PERIODS],
                       "time_metric": "quality-eligible alert overlap with period, days",
                       "average_metric": "mean eligible complete 2026 alert duration, days",
                       "covered_regions": len(REGION_NAMES)},
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for pair in FILES.values() for p in pair} |
                             {str(p.relative_to(ROOT)): sha(p) for p in (EVENTS, LINKS)},
            "coverage": coverage, "rankings": records(rank), "longest": longest_rows,
            "averages_2026": averages, "threat_rankings": records(threat), "threat_audit": threat_audit,
            "threat_period_availability": {"2024": "unavailable", "2025": "unavailable",
                                           "9m 2026": "provisional candidate links"}}
    (OUT / "analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (OUT / "summary.md").write_text(markdown(data))
    print(json.dumps({"coverage": coverage, "threat_audit": threat_audit,
                      "ranking_rows": len(rank), "longest_rows": len(longest_rows),
                      "threat_ranking_rows": len(threat)}, indent=2))


if __name__ == "__main__":
    main()
