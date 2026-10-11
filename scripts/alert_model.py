"""Causal, archive-based regional alert onset modelling.

Availability assumption: source timestamps approximate public message times.
The archives do not contain reception or revision histories. Code-level temporal
causality can be checked; true historical availability cannot be certified.
"""
from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from normalize_and_reconcile import REGIONS, canonical_region

ROOT = Path(__file__).resolve().parents[1]
# Fixed geography, not learned from future rows. The source omits Luhansk.
REGION_NAMES = sorted(r for r in REGIONS if r != "Luhanska oblast")
REGION_IDS = [REGIONS[r]["id"] for r in REGION_NAMES]
REGION_INDEX = {r: i for i, r in enumerate(REGION_NAMES)}
SOURCE_ENDS = {"official": "2026-09-07", "volunteer": "2026-10-01"}
NUMERIC_FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "year_sin", "year_cos", "since_last_onset_log_hours",
    "own_onsets_6h_log", "own_onsets_24h_log", "own_onsets_7d_log",
    "country_active_fraction", "country_onsets_1h_log",
    "country_onsets_6h_log", "country_onsets_24h_log",
]
FEATURES = NUMERIC_FEATURES + [f"region_{r}" for r in REGION_IDS]


def epoch(value: str) -> int:
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def utc(value: int) -> str:
    return datetime.fromtimestamp(int(value), timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_events(source: str, until: int) -> tuple[list, dict]:
    """Keep all starts; omit only uncertain end events, never their starts.

    Events: (message time, kind [start=0,end=1], region, locality, start time).
    Identity does not depend on the eventual end or an end-derived record hash.
    No reconciliation against another source or future episode merging is used.
    """
    path = ROOT / "data" / "raw" / f"{source}_data_en.csv"
    events = set()
    stats = defaultdict(int)
    region_cache = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_region = row.get("oblast") or row.get("region") or ""
            if raw_region not in region_cache:
                region_cache[raw_region] = canonical_region(raw_region)
            region = region_cache[raw_region]
            if region not in REGION_INDEX:
                stats["unmapped_or_unsupported_rows"] += 1
                continue
            try:
                start = epoch(row["started_at"])
            except (ValueError, KeyError):
                stats["invalid_starts"] += 1
                continue
            if start > until:
                continue
            stats["raw_rows_with_start_by_cutoff"] += 1
            i = REGION_INDEX[region]
            locality = (row.get("level", "oblast"), row.get("raion", ""), row.get("hromada", ""))
            events.add((start, 0, i, locality, start))
            try:
                end = epoch(row["finished_at"])
            except (ValueError, KeyError):
                stats["missing_or_invalid_ends"] += 1
                continue
            if end < start:
                stats["end_before_start"] += 1
                continue
            uncertain = (
                (source == "volunteer" and row.get("naive", "").lower() == "true")
                or (source == "official" and end - start == 3600)
            )
            if uncertain:
                stats["uncertain_end_rows_omitted"] += 1
            elif end <= until:
                events.add((end, 1, i, locality, start))
    events = sorted(events)
    stats["unique_start_events"] = sum(e[1] == 0 for e in events)
    stats["unique_end_events"] = sum(e[1] == 1 for e in events)
    stats["raw_sha256"] = sha256(path)
    stats["raw_path"] = str(path.relative_to(ROOT))
    return events, dict(stats)


def replay(events: list) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Replay locality state without looking at future event contents.

    Same-timestamp events are atomic: a touching end/start is continuous activity.
    A late end of an older record cannot close a newer activation at that locality.
    """
    active = [dict() for _ in REGION_NAMES]
    onsets = [[] for _ in REGION_NAMES]
    changes = [[] for _ in REGION_NAMES]
    for timestamp, batch_iter in groupby(sorted(events), key=lambda e: e[0]):
        batch = list(batch_iter)
        touched = {e[2] for e in batch}
        before = {r: bool(active[r]) for r in touched}
        for _, kind, region, locality, token in batch:
            if kind == 0:
                active[region][locality] = token
            elif active[region].get(locality) == token:
                del active[region][locality]
        for region in touched:
            after = bool(active[region])
            if after != before[region]:
                changes[region].append((timestamp, after))
                if after:
                    onsets[region].append(timestamp)
    return (
        [np.asarray(v, dtype=np.int64) for v in onsets],
        [(np.asarray([v[0] for v in c], dtype=np.int64),
          np.asarray([v[1] for v in c], dtype=bool)) for c in changes],
    )


def counts(times: np.ndarray, grid: np.ndarray, seconds: int) -> np.ndarray:
    return np.searchsorted(times, grid, side="right") - np.searchsorted(times, grid - seconds, side="right")


def feature_frame(events: list, grid: np.ndarray) -> pd.DataFrame:
    """Features and reported state only; labels live in a separate function."""
    onsets, changes = replay(events)
    n = len(grid)
    states = []
    recent = []
    for times, values in changes:
        ix = np.searchsorted(times, grid, side="right") - 1
        state = np.zeros(n, dtype=bool)
        valid = ix >= 0
        state[valid] = values[ix[valid]]
        states.append(state)
    for times in onsets:
        recent.append({h: counts(times, grid, h * 3600) for h in (1, 6, 24, 168)})
    country_active = np.stack(states).sum(axis=0) / len(REGION_NAMES)
    country_counts = {h: sum(v[h] for v in recent) for h in (1, 6, 24)}
    local = pd.to_datetime(grid, unit="s", utc=True).tz_convert("Europe/Kyiv")
    hour = np.asarray(local.hour) + np.asarray(local.minute) / 60
    weekday = np.asarray(local.dayofweek)
    doy = np.asarray(local.dayofyear)
    event_times = np.asarray([e[0] for e in events], dtype=np.int64)
    event_ix = np.searchsorted(event_times, grid, side="right") - 1
    max_seen = np.full(n, -1, dtype=np.int64)
    ok = event_ix >= 0
    max_seen[ok] = event_times[event_ix[ok]]
    frames = []
    for i, region in enumerate(REGION_NAMES):
        times = onsets[i]
        ix = np.searchsorted(times, grid, side="right") - 1
        since = np.full(n, 168.0)
        valid = ix >= 0
        since[valid] = np.minimum((grid[valid] - times[ix[valid]]) / 3600, 168)
        data = {
            "time": grid, "region_index": np.full(n, i), "oblast": region,
            "oblast_id": REGION_IDS[i], "known_clear": ~states[i],
            "max_feature_event_time": max_seen,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "year_sin": np.sin(2 * np.pi * doy / 365.25),
            "year_cos": np.cos(2 * np.pi * doy / 365.25),
            "since_last_onset_log_hours": np.log1p(since),
            "country_active_fraction": country_active,
            **{f"own_onsets_{label}_log": np.log1p(recent[i][h]) for h, label in ((6, "6h"), (24, "24h"), (168, "7d"))},
            **{f"country_onsets_{h}h_log": np.log1p(country_counts[h]) for h in (1, 6, 24)},
            "local_hour": np.asarray(local.hour),
        }
        for j, region_id in enumerate(REGION_IDS):
            data[f"region_{region_id}"] = float(i == j)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True).sort_values(["time", "region_index"]).reset_index(drop=True)


def attach_labels(frame: pd.DataFrame, events: list, horizon_seconds: int) -> pd.DataFrame:
    onsets, _ = replay(events)
    frame = frame.copy()
    frame["target_end"] = frame["time"] + horizon_seconds
    frame["y"] = np.zeros(len(frame), dtype=np.int8)
    for i, times in enumerate(onsets):
        mask = frame.region_index == i
        grid = frame.loc[mask, "time"].to_numpy()
        n = np.searchsorted(times, grid + horizon_seconds, side="right") - np.searchsorted(times, grid, side="right")
        frame.loc[mask, "y"] = (n > 0).astype(np.int8)
    return frame


def eligible_training(frame: pd.DataFrame, cutoff: int, training_start: int) -> pd.DataFrame:
    # Strict embargo: even an hour whose target ends exactly at refit time waits.
    return frame.loc[frame.known_clear & (frame.time >= training_start) & (frame.target_end < cutoff)]


def fit_bundle(train: pd.DataFrame, cutoff: int, source: str, horizon: int) -> dict:
    if train.empty or train.y.nunique() < 2:
        raise ValueError("Training needs both positive and negative matured examples")
    if not (train.target_end < cutoff).all():
        raise ValueError("Training labels extend to or beyond the fitting cutoff")
    if not (train.max_feature_event_time <= train.time).all():
        raise ValueError("Future event detected in training features")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, tol=1e-5))
    model.fit(train[FEATURES].to_numpy(dtype=float), train.y.to_numpy())
    fit = model[-1]
    if int(fit.n_iter_.max()) >= fit.max_iter:
        raise RuntimeError("Logistic regression did not converge; refusing to report a completed fit")
    region_counts = np.bincount(train.region_index, minlength=len(REGION_NAMES))
    region_pos = np.bincount(train.region_index, weights=train.y, minlength=len(REGION_NAMES))
    global_rate = (float(train.y.sum()) + 0.5) / (len(train) + 1)
    region_rates = (region_pos + 20 * global_rate) / (region_counts + 20)
    keys = train.region_index.to_numpy() * 24 + train.local_hour.to_numpy()
    hour_counts = np.bincount(keys, minlength=len(REGION_NAMES) * 24)
    hour_pos = np.bincount(keys, weights=train.y, minlength=len(REGION_NAMES) * 24)
    hour_rates = (hour_pos + 20 * np.repeat(region_rates, 24)) / (hour_counts + 20)
    return {
        "model": model, "features": FEATURES, "source": source,
        "fit_cutoff": cutoff, "max_training_target_end": int(train.target_end.max()),
        "training_rows": len(train), "training_positives": int(train.y.sum()),
        "training_first_origin": int(train.time.min()),
        "region_rates": region_rates, "hour_rates": hour_rates,
        "region_counts": region_counts, "horizon_seconds": horizon,
        "availability_assumption": "public event timestamp; no reception/revision history",
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(frame) and int(frame.time.min()) < bundle["fit_cutoff"]:
        raise ValueError("Model was trained after the requested forecast time")
    if not (frame.max_feature_event_time <= frame.time).all():
        raise ValueError("Future event detected in prediction features")
    p = bundle["model"].predict_proba(frame[FEATURES].to_numpy(dtype=float))[:, 1]
    regions = frame.region_index.to_numpy()
    return p, bundle["region_rates"][regions], bundle["hour_rates"][regions * 24 + frame.local_hour.to_numpy()]
