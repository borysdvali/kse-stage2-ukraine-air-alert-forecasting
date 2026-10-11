#!/usr/bin/env python3
"""Normalize alert intervals and relate public KPSZSU posts to them.

All decisions are deterministic and recorded with provenance. Threat labels,
clusters, and links are rule-based candidates for review, not official labels.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 1, tzinfo=timezone.utc)
WINDOW_TAG = "2026-01-01_2026-09-30"
OFFICIAL_CUTOFF = datetime(2026, 9, 7, 5, 28, 22, tzinfo=timezone.utc)
LINK_LOOKBACK = timedelta(minutes=20)
LINK_LOOKAHEAD = timedelta(minutes=20)

REGIONS: dict[str, dict[str, Any]] = {
    "Vinnytska oblast": {"id": "UA-05", "aliases": ["вінницьк", "вінниччин", "вінниц"]},
    "Volynska oblast": {"id": "UA-07", "aliases": ["волинськ", "волинь", "волині", "волинщин"]},
    "Dnipropetrovska oblast": {"id": "UA-12", "aliases": ["дніпропетровськ", "дніпропетровщин", "дніпропетровщину", "дніпро", "дніпрі"]},
    "Donetska oblast": {"id": "UA-14", "aliases": ["донецьк", "донеччин", "донеччину"]},
    "Zhytomyrska oblast": {"id": "UA-18", "aliases": ["житомирськ", "житомирщин", "житомирщину"]},
    "Zakarpatska oblast": {"id": "UA-21", "aliases": ["закарпатськ", "закарпатт"]},
    "Zaporizka oblast": {"id": "UA-23", "aliases": ["запорізьк", "запоріжж", "запоріжжя"]},
    "Ivano-Frankivska oblast": {"id": "UA-26", "aliases": ["івано-франківськ", "івано франківськ", "прикарпатт", "франківщин"]},
    "Kyivska oblast": {"id": "UA-32", "aliases": ["київщин", "київськ", "київська област"]},
    "Kyiv City": {"id": "UA-30", "aliases": ["місто київ", "м. київ", "м.київ", "київ"]},
    "Kirovohradska oblast": {"id": "UA-35", "aliases": ["кіровоградськ", "кіровоградщин", "кропивницьк", "кропивниччин"]},
    "Luhanska oblast": {"id": "UA-09", "aliases": ["луганськ", "луганщин", "луганщину"]},
    "Lvivska oblast": {"id": "UA-46", "aliases": ["львівськ", "львівщин", "львівщину"]},
    "Mykolaivska oblast": {"id": "UA-48", "aliases": ["миколаївськ", "миколаївщин", "миколаївщину"]},
    "Odeska oblast": {"id": "UA-51", "aliases": ["одеськ", "одещин", "одещину", "одеса", "одесі"]},
    "Poltavska oblast": {"id": "UA-53", "aliases": ["полтавськ", "полтавщин", "полтавщину"]},
    "Rivnenska oblast": {"id": "UA-56", "aliases": ["рівненськ", "рівненщин", "рівненщину"]},
    "Sumska oblast": {"id": "UA-59", "aliases": ["сумськ", "сумщин", "сумщину", "сумщині", "суми"]},
    "Ternopilska oblast": {"id": "UA-61", "aliases": ["тернопільськ", "тернопільщин"]},
    "Kharkivska oblast": {"id": "UA-63", "aliases": ["харківськ", "харківщин", "харківщину", "харків"]},
    "Khersonska oblast": {"id": "UA-65", "aliases": ["херсонськ", "херсонщин", "херсонщину", "херсон"]},
    "Khmelnytska oblast": {"id": "UA-68", "aliases": ["хмельницьк", "хмельниччин"]},
    "Cherkaska oblast": {"id": "UA-71", "aliases": ["черкаськ", "черкащин", "черкащину", "черкаси"]},
    "Chernivetska oblast": {"id": "UA-77", "aliases": ["чернівецьк", "чернівеччин", "чернівці"]},
    "Chernihivska oblast": {"id": "UA-74", "aliases": ["чернігівськ", "чернігівщин", "чернігівщину", "чернігів"]},
}

OBLAST_ALIASES = {
    alias: region
    for region, values in REGIONS.items()
    for alias in values["aliases"]
}
ALIAS_PATTERNS = [
    (alias, region)
    for alias, region in sorted(OBLAST_ALIASES.items(), key=lambda item: -len(item[0]))
]

THREAT_PATTERNS: dict[str, tuple[str, ...]] = {
    "ballistic_missile": ("баліст", "кинджал", "іскандер-м", "kn-23"),
    "cruise_or_air_launched_missile": (
        "крилат", "ракет", "ракета", "ракети", "швидкісна ціль", "швидкісну ціль",
        "аеробаліст", "калібр", "х-101", "х-59", "х-69", "циркон",
    ),
    "jet_attack_uav": ("реактивн", "shahed-238", "шахед-238"),
    "propeller_or_unspecified_attack_uav": (
        "бпла", "бпл", "дрон", "шахед", "shahed", "гербер", "італмас", "мопед",
    ),
    "guided_aerial_bomb": ("каб", "керовані авіабомб", "керованої авіабомб"),
    "tactical_aviation": ("тактичної авіації", "тактичної авіаці", "авіаційн", "авіаційного озброєння"),
    "reconnaissance_uav": ("розвідувальн", "розвіддрон"),
}
ALL_CLEAR_TERMS = ("відбій", "загрозу знято", "загрози знято")
GENERIC_ALERT_TERMS = ("повітряна тривога", "ракетна небезпека", "ракетна загроза", "не ігноруйте сигнал")
DIRECTION_TERMS = (
    "курс", "курсом", "напрям", "напрямку", "вектор", "із", "з", "через", "повз", "рухається", "рух"
)
BEARING_TERMS = (
    "північний", "північна", "північному", "північ", "південний", "південна", "південному", "південь",
    "східний", "східна", "східному", "схід", "західний", "західна", "західному", "захід",
    "північно-східний", "північно-західний", "південно-східний", "південно-західний",
)


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso(dt: datetime | None) -> str:
    return dt.astimezone(timezone.utc).isoformat() if dt else ""


def stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").strip().lower()
    value = value.replace("’", "'").replace("ʼ", "'").replace("ё", "е")
    return re.sub(r"\s+", " ", value)


def canonical_region(value: str) -> str | None:
    text = norm_text(value)
    for region in REGIONS:
        if norm_text(region) == text:
            return region
    for alias, region in ALIAS_PATTERNS:
        if text.startswith(alias):
            return region
    return None


def mentioned_regions(text: str) -> list[str]:
    normalized = norm_text(text)
    found: set[str] = set()
    for alias, region in ALIAS_PATTERNS:
        if alias in normalized:
            # Do not interpret a bare Kyiv city mention as both city and oblast.
            found.add(region)
    if "київщин" in normalized or "київськ" in normalized:
        if not any(marker in normalized for marker in ("м. київ", "м.київ", "місто київ")):
            found.discard("Kyiv City")
    if "Kyiv City" in found and ("київщин" in normalized or "київська област" in normalized):
        found.add("Kyivska oblast")
    return sorted(found, key=lambda r: REGIONS[r]["id"])


def location_mentions_with_roles(text: str) -> list[dict[str, Any]]:
    normalized = norm_text(text)
    allowed = set(mentioned_regions(text))
    found: dict[str, dict[str, Any]] = {}
    for alias, region in ALIAS_PATTERNS:
        if region not in allowed:
            continue
        index = normalized.find(alias)
        if index < 0:
            continue
        prefix = normalized[max(0, index - 18):index]
        role = "location"
        if re.search(r"(?:^|\s)(?:на|до|у|в)\s+$", prefix):
            role = "possible_destination"
        elif re.search(r"(?:^|\s)(?:з|із|від)\s+$", prefix):
            role = "possible_origin"
        candidate = {
            "oblast": region,
            "oblast_id": REGIONS[region]["id"],
            "text_offset": index,
            "matched_alias": alias,
            "role": role,
        }
        if region not in found or index < found[region]["text_offset"]:
            found[region] = candidate
    return sorted(found.values(), key=lambda item: (item["text_offset"], item["oblast_id"]))


def classify_threat(text: str) -> tuple[list[str], str, str]:
    normalized = norm_text(text)
    explicit_all_clear = bool(
        re.search(r"(?<!до )відбій(?:\s|[.!,:;]|$)", normalized)
        or any(term in normalized for term in ALL_CLEAR_TERMS[1:])
    )
    if explicit_all_clear:
        return [], "all_clear", "high"
    classes = [
        family for family, patterns in THREAT_PATTERNS.items()
        if any(pattern in normalized for pattern in patterns)
    ]
    if not classes and any(term in normalized for term in GENERIC_ALERT_TERMS):
        return ["generic_alert_or_unspecified"], "generic_alert_or_unspecified", "medium"
    if not classes:
        return [], "other_or_non_threat", "low"
    if "reconnaissance_uav" in classes and "propeller_or_unspecified_attack_uav" in classes:
        classes.remove("propeller_or_unspecified_attack_uav")
    # Generic missile words occur in some ballistic posts; retain both searchable
    # labels, but use the more specific ballistic class as the event family.
    if "ballistic_missile" in classes and "cruise_or_air_launched_missile" in classes:
        classes.remove("cruise_or_air_launched_missile")
    if "jet_attack_uav" in classes and "propeller_or_unspecified_attack_uav" in classes:
        classes.remove("propeller_or_unspecified_attack_uav")
    family = classes[0] if len(classes) == 1 else "mixed_or_multi_threat"
    confidence = "medium" if family == "mixed_or_multi_threat" else "high"
    return classes, family, confidence


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def deduplicate_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    out: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}
    for source, filename in (
        ("official", f"official_overlap_{WINDOW_TAG}.csv"),
        ("volunteer", f"volunteer_overlap_{WINDOW_TAG}.csv"),
    ):
        input_rows = read_csv_rows(INTERIM / filename)
        unique: dict[tuple[str, ...], dict[str, Any]] = {}
        dupes = 0
        fields = list(input_rows[0]) if input_rows else []
        for row in input_rows:
            key = tuple(row.get(field, "") for field in fields)
            if key in unique:
                dupes += 1
                continue
            start = parse_ts(row.get("started_at"))
            end = parse_ts(row.get("finished_at"))
            if start is None:
                continue
            oblast = canonical_region(row.get("oblast") or row.get("region") or "")
            volunteer_naive = source == "volunteer" and row.get("naive", "").strip().lower() == "true"
            official_one_hour_ambiguous = (
                source == "official" and end is not None
                and (end - start).total_seconds() == 3600
            )
            record = {
                **row,
                "source_dataset": source,
                "source_record_id": stable_id(source, *key),
                "oblast_canonical": oblast or "",
                "oblast_id": REGIONS[oblast]["id"] if oblast else "",
                "started_at_utc": iso(start),
                "finished_at_utc": iso(end),
                "duration_minutes": round((end - start).total_seconds() / 60, 3) if end and end >= start else "",
                "end_observed": not (volunteer_naive or official_one_hour_ambiguous) and end is not None,
                "end_imputed": volunteer_naive,
                "official_one_hour_end_ambiguous": official_one_hour_ambiguous,
            }
            if end and end < start:
                record["duration_minutes"] = ""
                record["end_observed"] = False
                record["quality_flag"] = "end_before_start"
            else:
                record["quality_flag"] = "official_one_hour_end_ambiguous" if official_one_hour_ambiguous else ""
            unique[key] = record
        out[source] = list(unique.values())
        write_csv(PROCESSED / f"{source}_records_deduplicated.csv", out[source])
        summary[source] = {
            "input_rows": len(input_rows),
            "deduplicated_rows": len(out[source]),
            "exact_duplicates_removed": dupes,
            "rows_without_canonical_oblast": sum(not row["oblast_canonical"] for row in out[source]),
            "naive_endings": sum(bool(row["end_imputed"]) for row in out[source]),
            "official_one_hour_end_ambiguous": sum(bool(row["official_one_hour_end_ambiguous"]) for row in out[source]),
        }
    return out["official"], out["volunteer"], summary


def merge_region_intervals(records: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        region = record.get("oblast_canonical")
        start = parse_ts(record.get("started_at_utc"))
        end = parse_ts(record.get("finished_at_utc"))
        if region and start and end and end >= start:
            if start < WINDOW_END and end >= WINDOW_START:
                by_region[region].append(record)

    episodes: list[dict[str, Any]] = []
    for region, items in by_region.items():
        items.sort(key=lambda row: (parse_ts(row["started_at_utc"]), parse_ts(row["finished_at_utc"])))
        current: dict[str, Any] | None = None
        for item in items:
            start = parse_ts(item["started_at_utc"])
            end = parse_ts(item["finished_at_utc"])
            assert start is not None and end is not None
            clipped_start = max(start, WINDOW_START)
            clipped_end = min(end, WINDOW_END)
            if current is None or start > current["end"]:
                if current:
                    episodes.append(finalize_episode(current, source_name, region))
                current = {
                    "start": clipped_start,
                    "end": clipped_end,
                    "record_ids": [item["source_record_id"]],
                    "levels": {item.get("level", "oblast") or "oblast"},
                    "imputed_any": bool(item.get("end_imputed")),
                    "ambiguous_official_end_any": bool(item.get("official_one_hour_end_ambiguous")),
                    "observed_end_count": int(bool(item.get("end_observed"))),
                    "raw_start_min": start,
                    "raw_end_max": end,
                    "left_censored": start < WINDOW_START,
                    "right_censored": end > WINDOW_END,
                }
            else:
                current["end"] = max(current["end"], clipped_end)
                current["record_ids"].append(item["source_record_id"])
                current["levels"].add(item.get("level", "oblast") or "oblast")
                current["imputed_any"] = current["imputed_any"] or bool(item.get("end_imputed"))
                current["ambiguous_official_end_any"] = (
                    current["ambiguous_official_end_any"] or bool(item.get("official_one_hour_end_ambiguous"))
                )
                current["observed_end_count"] += int(bool(item.get("end_observed")))
                current["raw_end_max"] = max(current["raw_end_max"], end)
                current["left_censored"] = current["left_censored"] or start < WINDOW_START
                current["right_censored"] = current["right_censored"] or end > WINDOW_END
        if current:
            episodes.append(finalize_episode(current, source_name, region))
    return sorted(episodes, key=lambda row: (row["oblast_id"], row["started_at_utc"]))


def finalize_episode(current: dict[str, Any], source: str, region: str) -> dict[str, Any]:
    start = current["start"]
    end = current["end"]
    ids = sorted(set(current["record_ids"]))
    long_union = end - start > timedelta(days=7)
    return {
        "episode_id": stable_id(source, region, iso(start), iso(end)),
        "source_basis": source,
        "oblast": region,
        "oblast_id": REGIONS[region]["id"],
        "started_at_utc": iso(start),
        "ended_at_utc": iso(end),
        "duration_minutes": round((end - start).total_seconds() / 60, 3),
        "contributing_record_count": len(ids),
        "contributing_source_record_ids": json.dumps(ids, ensure_ascii=False),
        "contributing_levels": json.dumps(sorted(current["levels"]), ensure_ascii=False),
        "has_imputed_end_contributor": current["imputed_any"],
        "has_ambiguous_official_one_hour_end": current["ambiguous_official_end_any"],
        "needs_long_union_review": long_union,
        "left_censored_by_window": current["left_censored"],
        "right_censored_by_window": current["right_censored"],
        "observed_end_record_count": current["observed_end_count"],
        "duration_target_eligible": not (
            current["imputed_any"] or current["ambiguous_official_end_any"]
            or current["left_censored"] or current["right_censored"] or long_union
        ),
    }


def episodes_overlap(a: dict[str, Any], b: dict[str, Any], tolerance: timedelta = timedelta(minutes=30)) -> bool:
    a_start, a_end = parse_ts(a["started_at_utc"]), parse_ts(a["ended_at_utc"])
    b_start, b_end = parse_ts(b["started_at_utc"]), parse_ts(b["ended_at_utc"])
    assert a_start and a_end and b_start and b_end
    return a_start <= b_end + tolerance and b_start <= a_end + tolerance


def reconcile_sources(official_eps: list[dict[str, Any]], volunteer_eps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    volunteer_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in official_eps:
        official_by_region[episode["oblast"]].append(episode)
    for episode in volunteer_eps:
        volunteer_by_region[episode["oblast"]].append(episode)

    def indexed_matches(
        region_index: dict[str, list[dict[str, Any]]],
        start_index: dict[str, list[datetime]],
        end_index: dict[str, list[datetime]],
        episode: dict[str, Any],
    ) -> list[dict[str, Any]]:
        region = episode["oblast"]
        rows = region_index.get(region, [])
        if not rows:
            return []
        start = parse_ts(episode["started_at_utc"])
        end = parse_ts(episode["ended_at_utc"])
        assert start and end
        tolerance = timedelta(minutes=30)
        first = bisect_left(end_index[region], start - tolerance)
        after_last = bisect_right(start_index[region], end + tolerance)
        return rows[first:after_last]

    official_starts = {
        region: [parse_ts(row["started_at_utc"]) for row in rows]
        for region, rows in official_by_region.items()
    }
    official_ends = {
        region: [parse_ts(row["ended_at_utc"]) for row in rows]
        for region, rows in official_by_region.items()
    }
    volunteer_starts = {
        region: [parse_ts(row["started_at_utc"]) for row in rows]
        for region, rows in volunteer_by_region.items()
    }
    volunteer_ends = {
        region: [parse_ts(row["ended_at_utc"]) for row in rows]
        for region, rows in volunteer_by_region.items()
    }

    recon: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    matched_volunteer: set[str] = set()
    official_ids = set()
    for region, episodes in official_by_region.items():
        for official in episodes:
            official_ids.add(official["episode_id"])
            matches = indexed_matches(volunteer_by_region, volunteer_starts, volunteer_ends, official)
            for volunteer in matches:
                matched_volunteer.add(volunteer["episode_id"])
            deltas = [
                abs((parse_ts(official["started_at_utc"]) - parse_ts(v["started_at_utc"])).total_seconds())
                for v in matches
            ]
            min_delta = min(deltas) if deltas else ""
            material_conflict = bool(deltas and min(deltas) > 5 * 60)
            recon.append({
                "official_episode_id": official["episode_id"],
                "volunteer_episode_ids": json.dumps([v["episode_id"] for v in matches]),
                "oblast": region,
                "match_status": "material_timing_conflict" if material_conflict else ("matched" if matches else "official_only"),
                "start_delta_seconds_min": min_delta,
                "official_start_utc": official["started_at_utc"],
                "volunteer_starts_utc": json.dumps([v["started_at_utc"] for v in matches]),
                "official_end_utc": official["ended_at_utc"],
                "volunteer_ends_utc": json.dumps([v["ended_at_utc"] for v in matches]),
                "volunteer_has_imputed_end": any(v["has_imputed_end_contributor"] for v in matches),
            })
            row = dict(official)
            row.update({
                "canonical_episode_id": stable_id("canonical", official["episode_id"]),
                "canonical_source": "official",
                "volunteer_match_count": len(matches),
                "volunteer_episode_ids": json.dumps([v["episode_id"] for v in matches]),
                "source_agreement": bool(matches),
                "source_conflict": material_conflict,
                "confidence": "medium_timing_conflict" if material_conflict else ("high" if matches else "high_official_only"),
            })
            canonical.append(row)

    # Volunteer-only episodes are retained only when the official-derived
    # publisher has no nearby oblast episode. This matters after its Sep 7 cutoff.
    for region, episodes in volunteer_by_region.items():
        for volunteer in episodes:
            if volunteer["episode_id"] in matched_volunteer:
                continue
            nearby = indexed_matches(official_by_region, official_starts, official_ends, volunteer)
            if nearby:
                # An ambiguous many-to-many overlap is logged; do not create a
                # second canonical episode for what may be the same wave.
                recon.append({
                    "official_episode_id": json.dumps([o["episode_id"] for o in nearby]),
                    "volunteer_episode_ids": json.dumps([volunteer["episode_id"]]),
                    "oblast": region,
                    "match_status": "ambiguous_overlapping_volunteer",
                    "start_delta_seconds_min": min(abs((parse_ts(o["started_at_utc"]) - parse_ts(volunteer["started_at_utc"])).total_seconds()) for o in nearby),
                    "official_start_utc": json.dumps([o["started_at_utc"] for o in nearby]),
                    "volunteer_starts_utc": volunteer["started_at_utc"],
                    "official_end_utc": json.dumps([o["ended_at_utc"] for o in nearby]),
                    "volunteer_ends_utc": volunteer["ended_at_utc"],
                    "volunteer_has_imputed_end": volunteer["has_imputed_end_contributor"],
                })
                matched_volunteer.add(volunteer["episode_id"])
                continue
            recon.append({
                "official_episode_id": "",
                "volunteer_episode_ids": json.dumps([volunteer["episode_id"]]),
                "oblast": region,
                "match_status": "volunteer_only_supplement",
                "start_delta_seconds_min": "",
                "official_start_utc": "",
                "volunteer_starts_utc": volunteer["started_at_utc"],
                "official_end_utc": "",
                "volunteer_ends_utc": volunteer["ended_at_utc"],
                "volunteer_has_imputed_end": volunteer["has_imputed_end_contributor"],
            })
            row = dict(volunteer)
            row.update({
                "canonical_episode_id": stable_id("canonical", volunteer["episode_id"]),
                "canonical_source": "volunteer_supplement",
                "volunteer_match_count": 1,
                "volunteer_episode_ids": json.dumps([volunteer["episode_id"]]),
                "source_agreement": False,
                "source_conflict": False,
                "confidence": "low_volunteer_imputed_end" if volunteer["has_imputed_end_contributor"] else "medium_volunteer_only",
            })
            canonical.append(row)

    return sorted(canonical, key=lambda row: (row["oblast_id"], row["started_at_utc"])), recon


def extract_posts() -> list[dict[str, Any]]:
    path = RAW / "kpszsu_2026-01-01_2026-09-30.jsonl"
    posts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            published = parse_ts(raw["published_at_utc"])
            if published is None or not (WINDOW_START <= published < WINDOW_END):
                continue
            text = str(raw.get("text", ""))
            classes, family, class_conf = classify_threat(text)
            regions = mentioned_regions(text)
            normalized = norm_text(text)
            direction_cues = [term for term in DIRECTION_TERMS if term in normalized]
            bearing_matches = [term for term in BEARING_TERMS if term in normalized]
            bearings = [
                term for term in bearing_matches
                if not any(term != other and term in other for other in bearing_matches)
            ]
            locations_in_text_order = location_mentions_with_roles(text)
            post = {
                **raw,
                "published_at_utc": iso(published),
                "post_type": family,
                "threat_classes": json.dumps(classes, ensure_ascii=False),
                "oblast_mentions": json.dumps(regions, ensure_ascii=False),
                "oblast_ids": json.dumps([REGIONS[r]["id"] for r in regions]),
                "location_mentions": json.dumps(locations_in_text_order, ensure_ascii=False),
                "regional_sequence": json.dumps([item["oblast"] for item in locations_in_text_order], ensure_ascii=False),
                "bearing_mentions": json.dumps(bearings, ensure_ascii=False),
                "direction_cues": json.dumps(direction_cues, ensure_ascii=False),
                "classification_confidence": class_conf,
                "needs_manual_review": family in {"mixed_or_multi_threat", "generic_alert_or_unspecified", "other_or_non_threat"} or not regions,
            }
            posts.append(post)
    return posts


def cluster_threat_events(posts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active = [p for p in posts if p["post_type"] not in {"all_clear", "other_or_non_threat"}]
    parent = list(range(len(active)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_family: dict[str, list[int]] = defaultdict(list)
    id_to_index: dict[int, int] = {}
    for i, post in enumerate(active):
        by_family[post["post_type"]].append(i)
        id_to_index[int(post["post_id"])] = i

    max_gap = timedelta(minutes=45)
    for family, indices in by_family.items():
        for pos, i in enumerate(indices):
            current = active[i]
            current_time = parse_ts(current["published_at_utc"])
            current_regions = set(json.loads(current["oblast_mentions"]))
            reply_id = current.get("reply_to_post_id")
            if reply_id and int(reply_id) in id_to_index:
                j = id_to_index[int(reply_id)]
                if active[j]["post_type"] == family:
                    union(i, j)
            for j in reversed(indices[:pos]):
                prior_time = parse_ts(active[j]["published_at_utc"])
                if current_time - prior_time > max_gap:
                    break
                prior_regions = set(json.loads(active[j]["oblast_mentions"]))
                if current_regions and prior_regions and current_regions.intersection(prior_regions):
                    union(i, j)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, post in enumerate(active):
        groups[find(i)].append(post)
    events: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    post_event: dict[int, str] = {}
    for members in groups.values():
        members.sort(key=lambda p: (p["published_at_utc"], int(p["post_id"])))
        post_ids = sorted(int(p["post_id"]) for p in members)
        event_id = stable_id("threat", *post_ids)
        region_set = sorted({r for p in members for r in json.loads(p["oblast_mentions"])}, key=lambda r: REGIONS[r]["id"])
        region_observations = []
        for post in members:
            for location in json.loads(post["location_mentions"]):
                region_observations.append({
                    "post_id": post["post_id"],
                    "observed_at_utc": post["published_at_utc"],
                    "oblast": location["oblast"],
                    "role": location["role"],
                    "text_offset": location["text_offset"],
                })
        region_observations.sort(key=lambda item: (item["observed_at_utc"], item["post_id"], item["text_offset"]))
        regional_sequence = []
        for observation in region_observations:
            if not regional_sequence or regional_sequence[-1]["oblast"] != observation["oblast"]:
                regional_sequence.append({
                    "oblast": observation["oblast"],
                    "oblast_id": REGIONS[observation["oblast"]]["id"],
                    "first_observed_at_utc": observation["observed_at_utc"],
                    "role": observation["role"],
                    "post_id": observation["post_id"],
                })
        classes = sorted({c for p in members for c in json.loads(p["threat_classes"])})
        start = min(parse_ts(p["published_at_utc"]) for p in members)
        end = max(parse_ts(p["published_at_utc"]) for p in members)
        family_set = sorted({p["post_type"] for p in members})
        event = {
            "threat_event_id": event_id,
            "event_family": family_set[0] if len(family_set) == 1 else "mixed_or_multi_threat",
            "threat_classes": json.dumps(classes, ensure_ascii=False),
            "first_observed_at_utc": iso(start),
            "last_observed_at_utc": iso(end),
            "post_count": len(members),
            "oblast_mentions": json.dumps(region_set, ensure_ascii=False),
            "oblast_ids": json.dumps([REGIONS[r]["id"] for r in region_set]),
            "regional_observation_sequence": json.dumps(regional_sequence, ensure_ascii=False),
            "region_observation_times_utc": json.dumps(
                {
                    region: {
                        "first": min(p["published_at_utc"] for p in members if region in json.loads(p["oblast_mentions"])),
                        "last": max(p["published_at_utc"] for p in members if region in json.loads(p["oblast_mentions"])),
                    }
                    for region in region_set
                },
                ensure_ascii=False,
            ),
            "post_ids": json.dumps(post_ids),
            "post_urls": json.dumps([p["url"] for p in members]),
            "event_confidence": "medium" if region_set and len(members) > 1 else "low_single_post_or_no_region",
            "needs_manual_review": not region_set or len(family_set) > 1 or any(p["needs_manual_review"] for p in members),
        }
        events.append(event)
        for post in members:
            post_event[int(post["post_id"])] = event_id
            links.append({
                "post_id": post["post_id"],
                "threat_event_id": event_id,
                "post_url": post["url"],
                "post_type": post["post_type"],
            })

    # All-clear posts can attach to a threat through their Telegram reply target.
    for post in posts:
        if post["post_type"] != "all_clear":
            continue
        reply = post.get("reply_to_post_id")
        linked = post_event.get(int(reply)) if reply and int(reply) in post_event else ""
        links.append({
            "post_id": post["post_id"],
            "threat_event_id": linked,
            "post_url": post["url"],
            "post_type": "all_clear",
        })
    return sorted(events, key=lambda row: row["first_observed_at_utc"]), links


def link_events_to_alerts(events: list[dict[str, Any]], canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in canonical:
        by_region[episode["oblast"]].append(episode)
    for rows in by_region.values():
        rows.sort(key=lambda row: row["started_at_utc"])

    links: list[dict[str, Any]] = []
    for event in events:
        regions = json.loads(event["oblast_mentions"])
        region_times = json.loads(event["region_observation_times_utc"])
        for region in regions:
            event_start = parse_ts(region_times[region]["first"])
            event_end = parse_ts(region_times[region]["last"])
            assert event_start and event_end
            for episode in by_region.get(region, []):
                alert_start = parse_ts(episode["started_at_utc"])
                alert_end = parse_ts(episode["ended_at_utc"])
                assert alert_start and alert_end
                if event_start <= alert_end + LINK_LOOKAHEAD and event_end >= alert_start - LINK_LOOKBACK:
                    if event_start <= alert_end and event_end >= alert_start:
                        confidence = "medium_temporal_and_region_overlap"
                    else:
                        confidence = "low_temporal_nearby"
                    links.append({
                        "threat_event_id": event["threat_event_id"],
                        "canonical_episode_id": episode["canonical_episode_id"],
                        "oblast": region,
                        "event_family": event["event_family"],
                        "event_first_observed_utc": event["first_observed_at_utc"],
                        "alert_started_at_utc": episode["started_at_utc"],
                        "alert_ended_at_utc": episode["ended_at_utc"],
                        "canonical_source": episode["canonical_source"],
                        "link_confidence": confidence,
                        "link_method": "explicit_oblast_mention_and_temporal_proximity",
                    })
    return links


def main() -> None:
    global WINDOW_START, WINDOW_END, WINDOW_TAG, PROCESSED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01", help="Inclusive UTC date (YYYY-MM-DD)")
    parser.add_argument("--end-exclusive", default="2026-10-01", help="Exclusive UTC date (YYYY-MM-DD)")
    args = parser.parse_args()
    WINDOW_START = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    WINDOW_END = datetime.fromisoformat(args.end_exclusive).replace(tzinfo=timezone.utc)
    if WINDOW_START >= WINDOW_END:
        parser.error("--start must be before --end-exclusive")
    WINDOW_TAG = f"{WINDOW_START.date().isoformat()}_{(WINDOW_END - timedelta(days=1)).date().isoformat()}"
    if WINDOW_TAG != "2026-01-01_2026-09-30":
        PROCESSED = PROCESSED / WINDOW_TAG
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    official, volunteer, dedup_summary = deduplicate_records()
    official_eps = merge_region_intervals(official, "official")
    volunteer_eps = merge_region_intervals(volunteer, "volunteer")
    canonical, reconciliation = reconcile_sources(official_eps, volunteer_eps)
    telegram_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    telegram_end = datetime(2026, 10, 1, tzinfo=timezone.utc)
    posts = extract_posts() if WINDOW_START < telegram_end and WINDOW_END > telegram_start else []
    threat_events, post_event_links = cluster_threat_events(posts)
    alert_event_links = link_events_to_alerts(threat_events, canonical)

    write_csv(PROCESSED / "official_oblast_episodes.csv", official_eps)
    write_csv(PROCESSED / "volunteer_oblast_episodes.csv", volunteer_eps)
    write_csv(PROCESSED / "canonical_oblast_alert_episodes.csv", canonical)
    write_csv(PROCESSED / "source_reconciliation.csv", reconciliation)
    if posts:
        with (PROCESSED / "kpszsu_posts_enriched.jsonl").open("w", encoding="utf-8") as handle:
            for post in posts:
                handle.write(json.dumps(post, ensure_ascii=False) + "\n")
        write_csv(PROCESSED / "kpszsu_threat_events.csv", threat_events)
        write_csv(PROCESSED / "kpszsu_post_event_links.csv", post_event_links)
        write_csv(PROCESSED / "alert_threat_event_links.csv", alert_event_links)

    reconciliation_counts = Counter(row["match_status"] for row in reconciliation)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_inclusive_utc": iso(WINDOW_START),
        "window_end_exclusive_utc": iso(WINDOW_END),
        "official_source_cutoff_utc": iso(OFFICIAL_CUTOFF) if WINDOW_START <= OFFICIAL_CUTOFF < WINDOW_END else None,
        "deduplication": dedup_summary,
        "oblast_episode_counts": {
            "official": len(official_eps),
            "volunteer": len(volunteer_eps),
            "canonical": len(canonical),
        },
        "source_reconciliation_status_counts": dict(reconciliation_counts),
        "canonical_source_counts": dict(Counter(row["canonical_source"] for row in canonical)),
        "canonical_duration_target_eligible": sum(bool(row["duration_target_eligible"]) for row in canonical),
        "canonical_duration_ineligible": sum(not bool(row["duration_target_eligible"]) for row in canonical),
        "canonical_long_unions_needing_review": sum(bool(row["needs_long_union_review"]) for row in canonical),
        "kpszsu_posts": len(posts),
        "kpszsu_post_type_counts": dict(Counter(row["post_type"] for row in posts)),
        "kpszsu_posts_with_region_mentions": sum(bool(json.loads(row["oblast_mentions"])) for row in posts),
        "kpszsu_posts_needing_manual_review": sum(bool(row["needs_manual_review"]) for row in posts),
        "threat_events": len(threat_events),
        "threat_event_family_counts": dict(Counter(row["event_family"] for row in threat_events)),
        "threat_event_links_to_alert_episodes": len(alert_event_links),
        "files": {
            "official_deduplicated": str((PROCESSED / "official_records_deduplicated.csv").relative_to(ROOT)),
            "volunteer_deduplicated": str((PROCESSED / "volunteer_records_deduplicated.csv").relative_to(ROOT)),
            "official_oblast_episodes": str((PROCESSED / "official_oblast_episodes.csv").relative_to(ROOT)),
            "volunteer_oblast_episodes": str((PROCESSED / "volunteer_oblast_episodes.csv").relative_to(ROOT)),
            "canonical_alert_episodes": str((PROCESSED / "canonical_oblast_alert_episodes.csv").relative_to(ROOT)),
            "source_reconciliation": str((PROCESSED / "source_reconciliation.csv").relative_to(ROOT)),
        },
        "method_notes": [
            "Exact duplicate rows are removed from processed outputs; raw snapshots are untouched.",
            "Overlapping district, hromada and oblast intervals are unioned into continuous oblast active episodes.",
            "Official-derived episodes are canonical when available; unmatched volunteer episodes are retained as supplements.",
            "Duration targets exclude imputed volunteer ends, ambiguous official one-hour ends, window-clipped episodes, and unions longer than seven days pending review.",
            "The official collector can synthesize a one-hour end without marking it in the CSV; exact one-hour rows are conservatively flagged as ambiguous.",
            "An oblast union means at least one contributing subregion is active, not that the entire oblast is simultaneously under alert.",
            "Threat classes, event clusters, and alert-threat links are deterministic text/time heuristics and require quality review before predictive use.",
        ],
    }
    if posts:
        report["files"].update({
            "enriched_kpszsu_posts": str((PROCESSED / "kpszsu_posts_enriched.jsonl").relative_to(ROOT)),
            "kpszsu_threat_events": str((PROCESSED / "kpszsu_threat_events.csv").relative_to(ROOT)),
            "post_event_links": str((PROCESSED / "kpszsu_post_event_links.csv").relative_to(ROOT)),
            "alert_event_links": str((PROCESSED / "alert_threat_event_links.csv").relative_to(ROOT)),
        })
    if WINDOW_START <= OFFICIAL_CUTOFF < WINDOW_END:
        report["method_notes"].append(
            "The public official-derived source ends on 2026-09-07; later canonical supplements rely on volunteer history."
        )
    output = REPORTS / f"normalization_report_{WINDOW_TAG}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
