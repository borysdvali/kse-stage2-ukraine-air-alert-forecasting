#!/usr/bin/env python3
"""Summarize a public Telegram JSONL snapshot produced by the collector."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for line in args.snapshot.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    ids = [int(record["post_id"]) for record in records]
    timestamps = [
        datetime.fromisoformat(record["published_at_utc"]).astimezone(timezone.utc)
        for record in records
    ]
    months = Counter(timestamp.strftime("%Y-%m") for timestamp in timestamps)
    dates = Counter(timestamp.date().isoformat() for timestamp in timestamps)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(args.snapshot),
        "sha256": sha256(args.snapshot),
        "records": len(records),
        "distinct_post_ids": len(set(ids)),
        "minimum_post_id": min(ids) if ids else None,
        "maximum_post_id": max(ids) if ids else None,
        "earliest_timestamp_utc": min(timestamps).isoformat() if timestamps else None,
        "latest_timestamp_utc": max(timestamps).isoformat() if timestamps else None,
        "distinct_dates": len(dates),
        "monthly_counts": dict(sorted(months.items())),
        "empty_text_records": sum(not record.get("text") for record in records),
        "edited_records": sum(bool(record.get("is_edited")) for record in records),
        "reply_records": sum(record.get("reply_to_post_id") is not None for record in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
