#!/usr/bin/env python3
"""Create reproducible alert-data subsets for a UTC date window.

The script uses only the Python standard library so the raw-data audit can run
before a project environment is installed.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "reports"

WINDOW_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 10, 1, tzinfo=timezone.utc)  # exclusive
WINDOW_TAG = "2026-01-01_2026-09-30"

SOURCES = {
    "official": {
        "path": RAW / "official_data_en.csv",
        "url": (
            "https://raw.githubusercontent.com/Vadimkin/"
            "ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
        ),
    },
    "volunteer": {
        "path": RAW / "volunteer_data_en.csv",
        "url": (
            "https://raw.githubusercontent.com/Vadimkin/"
            "ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
        ),
    },
}


def parse_timestamp(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes"}


def overlaps_window(start: datetime, end: datetime | None) -> bool:
    return start < WINDOW_END and (end is None or end >= WINDOW_START)


def audit_dataset(name: str, config: dict[str, object]) -> dict[str, object]:
    path = Path(config["path"])
    starts_output = INTERIM / f"{name}_starts_{WINDOW_TAG}.csv"
    overlap_output = INTERIM / f"{name}_overlap_{WINDOW_TAG}.csv"

    total = 0
    invalid_timestamps = 0
    missing_end_total = 0
    missing_end_in_starts = 0
    start_rows = []
    overlap_rows = []
    all_rows_seen: set[tuple[str, ...]] = set()
    exact_duplicates = 0
    start_rows_seen: set[tuple[str, ...]] = set()
    start_exact_duplicates = 0
    levels: Counter[str] = Counter()
    locations: Counter[str] = Counter()
    naive_count = 0
    source_counts: Counter[str] = Counter()
    monthly_counts: Counter[str] = Counter()
    active_dates: set[str] = set()
    earliest_start: datetime | None = None
    latest_start: datetime | None = None

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        for row in reader:
            total += 1
            row_key = tuple(row.get(field, "") for field in fieldnames)
            if row_key in all_rows_seen:
                exact_duplicates += 1
            else:
                all_rows_seen.add(row_key)

            try:
                start = parse_timestamp(row.get("started_at", ""))
                end = parse_timestamp(row.get("finished_at", ""))
            except ValueError:
                invalid_timestamps += 1
                continue
            if start is None:
                invalid_timestamps += 1
                continue

            earliest_start = start if earliest_start is None else min(earliest_start, start)
            latest_start = start if latest_start is None else max(latest_start, start)
            if end is None:
                missing_end_total += 1

            if WINDOW_START <= start < WINDOW_END:
                start_rows.append(row)
                monthly_counts[start.strftime("%Y-%m")] += 1
                active_dates.add(start.date().isoformat())
                if row_key in start_rows_seen:
                    start_exact_duplicates += 1
                else:
                    start_rows_seen.add(row_key)
                if end is None:
                    missing_end_in_starts += 1
                if name == "official":
                    levels[(row.get("level") or "missing").strip()] += 1
                    locations[(row.get("oblast") or "missing").strip()] += 1
                    source_counts[(row.get("source") or "missing").strip()] += 1
                else:
                    locations[(row.get("region") or "missing").strip()] += 1
                    naive_count += int(bool_value(row.get("naive", "")))

            if overlaps_window(start, end):
                overlap_rows.append(row)

    INTERIM.mkdir(parents=True, exist_ok=True)
    for output, rows in ((starts_output, start_rows), (overlap_output, overlap_rows)):
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "source_url": config["url"],
        "raw_path": str(path.relative_to(ROOT)),
        "raw_sha256": sha256(path),
        "raw_rows": total,
        "raw_exact_duplicates": exact_duplicates,
        "invalid_timestamp_rows": invalid_timestamps,
        "missing_end_total": missing_end_total,
        "earliest_start_utc": earliest_start.isoformat() if earliest_start else None,
        "latest_start_utc": latest_start.isoformat() if latest_start else None,
        "starts_in_window_rows": len(start_rows),
        "starts_in_window_exact_duplicates": start_exact_duplicates,
        "overlap_window_rows": len(overlap_rows),
        "missing_end_in_start_window": missing_end_in_starts,
        "distinct_locations_in_start_window": len(locations),
        "distinct_start_dates_in_window": len(active_dates),
        "start_window_monthly_counts": dict(sorted(monthly_counts.items())),
        "start_window_location_counts": dict(locations.most_common()),
        "starts_output": str(starts_output.relative_to(ROOT)),
        "overlap_output": str(overlap_output.relative_to(ROOT)),
    }
    if name == "official":
        report["start_window_level_counts"] = dict(levels.most_common())
        report["start_window_source_counts"] = dict(source_counts.most_common())
    else:
        report["naive_endings_in_start_window"] = naive_count
    return report


def main() -> None:
    global WINDOW_START, WINDOW_END, WINDOW_TAG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-01-01", help="Inclusive UTC date (YYYY-MM-DD)")
    parser.add_argument("--end-exclusive", default="2026-10-01", help="Exclusive UTC date (YYYY-MM-DD)")
    args = parser.parse_args()
    WINDOW_START = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    WINDOW_END = datetime.fromisoformat(args.end_exclusive).replace(tzinfo=timezone.utc)
    if WINDOW_START >= WINDOW_END:
        parser.error("--start must be before --end-exclusive")
    WINDOW_TAG = f"{WINDOW_START.date().isoformat()}_{(WINDOW_END - timedelta(days=1)).date().isoformat()}"
    INTERIM.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    datasets = {name: audit_dataset(name, config) for name, config in SOURCES.items()}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start_inclusive_utc": WINDOW_START.isoformat(),
            "end_exclusive_utc": WINDOW_END.isoformat(),
            "description": f"UTC window {WINDOW_TAG}",
        },
        "datasets": datasets,
    }
    output = REPORTS / f"historical_data_audit_{WINDOW_TAG}.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
