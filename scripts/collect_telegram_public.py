#!/usr/bin/env python3
"""Collect a date-bounded public Telegram channel archive.

This uses Telegram's public web preview, requires no account or API token, and
writes a resumable JSONL snapshot. It is intended for public-channel research,
not real-time safety use.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


MESSAGE_START_RE = re.compile(
    r'<div class="tgme_widget_message[^>]*?"[^>]*?data-post="(?P<post>[^"]+)"[^>]*>',
    re.DOTALL,
)
DATETIME_RE = re.compile(r'<time[^>]+datetime="([^"]+)"', re.DOTALL)
TEXT_RE = re.compile(
    r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>', re.DOTALL
)
REPLY_RE = re.compile(
    r'<a class="tgme_widget_message_reply[^"]*" '
    r'href="https://t\.me/[^/]+/(\d+)"',
    re.DOTALL,
)
EDITED_RE = re.compile(r'tgme_widget_message_meta[^>]*>\s*edited\b')
TAG_RE = re.compile(r"<[^>]+>")
BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self.parts.append("\n")


def clean_text(fragment: str) -> str:
    parser = TextExtractor()
    parser.feed(fragment)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_page(page: str, channel: str) -> list[dict[str, object]]:
    starts = list(MESSAGE_START_RE.finditer(page))
    messages: list[dict[str, object]] = []
    for index, match in enumerate(starts):
        block_end = starts[index + 1].start() if index + 1 < len(starts) else len(page)
        block = page[match.start():block_end]
        post = match.group("post")
        if not post.startswith(f"{channel}/"):
            continue
        post_id = int(post.rsplit("/", 1)[1])
        datetime_match = DATETIME_RE.search(block)
        if not datetime_match:
            continue
        published_at = datetime.fromisoformat(
            html.unescape(datetime_match.group(1)).replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        text_match = TEXT_RE.search(block)
        reply_ids = [int(value) for value in REPLY_RE.findall(block)]
        messages.append(
            {
                "channel": channel,
                "post_id": post_id,
                "published_at_utc": published_at.isoformat(),
                "text": clean_text(text_match.group(1)) if text_match else "",
                "reply_to_post_id": reply_ids[0] if reply_ids else None,
                "is_edited": bool(EDITED_RE.search(block)),
                "url": f"https://t.me/{channel}/{post_id}",
            }
        )
    return messages


def fetch(url: str, timeout: int = 30) -> str:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            "--user-agent",
            "Mozilla/5.0 (compatible; historical-public-channel-research/1.0)",
            url,
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="kpszsu")
    parser.add_argument("--start", default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-10-01T00:00:00+00:00")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--max-pages", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def persist(
    output: Path,
    checkpoint_path: Path,
    records: dict[int, dict[str, object]],
    channel: str,
    next_before: int,
    pages: int,
    oldest: datetime,
) -> None:
    ordered = [records[key] for key in sorted(records)]
    with output.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    checkpoint_path.write_text(
        json.dumps(
            {
                "channel": channel,
                "next_before": next_before,
                "pages_fetched_this_run": pages,
                "records_in_window": len(records),
                "oldest_page_timestamp_utc": oldest.isoformat(),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end).astimezone(timezone.utc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    before: int | None = None
    seen_ids: set[int] = set()
    records: dict[int, dict[str, object]] = {}
    pages = 0
    if args.checkpoint.exists():
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        before = checkpoint.get("next_before")
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[int(record["post_id"])] = record
                seen_ids.add(int(record["post_id"]))

    while pages < args.max_pages:
        url = f"https://t.me/s/{args.channel}"
        if before is not None:
            url += f"?before={before}"
        page = fetch(url)
        batch = parse_page(page, args.channel)
        if not batch:
            raise RuntimeError(f"No messages parsed from {url}")
        pages += 1

        for record in batch:
            published = datetime.fromisoformat(str(record["published_at_utc"]))
            if start <= published < end:
                records[int(record["post_id"])] = record
                seen_ids.add(int(record["post_id"]))

        oldest = min(datetime.fromisoformat(str(row["published_at_utc"])) for row in batch)
        minimum_id = min(int(row["post_id"]) for row in batch)
        next_before = minimum_id

        if pages % args.checkpoint_every == 0 or oldest < start:
            persist(
                args.output,
                args.checkpoint,
                records,
                args.channel,
                next_before,
                pages,
                oldest,
            )

        if oldest < start:
            break
        if before == next_before:
            raise RuntimeError(f"Pagination stalled at post {next_before}")
        before = next_before
        time.sleep(args.delay)
    else:
        raise RuntimeError(f"Reached max-pages={args.max_pages} before start date")

    print(
        json.dumps(
            {
                "channel": args.channel,
                "records": len(records),
                "pages_fetched_this_run": pages,
                "output": str(args.output),
                "complete": oldest < start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
