"""Presentation helpers for historical forecasts; no model fitting or scoring logic."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from alert_model import REGION_IDS, REGION_NAMES, ROOT

REGION_LABELS = dict(zip(REGION_IDS, REGION_NAMES))
BOUNDARY_PATH = ROOT / "data" / "geo" / "ukraine_adm1_2017_simplified.geojson"
BOUNDARY_SOURCE = "https://www.geoboundaries.org/api/current/gbOpen/UKR/ADM1/"

# geoBoundaries gbOpen UKR ADM1 names, pinned 2017 reference geometry. Crimea,
# Sevastopol and Luhansk are shown as unsupported, never as zero probability.
MAP_SHAPE_TO_ID = {
    "Kherson Oblast": "UA-65", "Volyn Oblast": "UA-07",
    "Rivne Oblast": "UA-56", "Zhytomyr Oblast": "UA-18",
    "Kyiv Oblast": "UA-32", "Chernihiv Oblast": "UA-74",
    "Sumy Oblast": "UA-59", "Kharkiv Oblast": "UA-63",
    "Donetsk Oblast": "UA-14", "Zaporizhia Oblast": "UA-23",
    "Lviv Oblast": "UA-46", "Ivano-Frankivsk Oblast": "UA-26",
    "Zakarpattia Oblast": "UA-21", "Ternopil Oblast": "UA-61",
    "Chernivtsi Oblast": "UA-77", "Odessa Oblast": "UA-51",
    "Mykolaiv Oblast": "UA-48", "Vinnytsia Oblast": "UA-05",
    "Khmelnytskyi Oblast": "UA-68", "Cherkasy Oblast": "UA-71",
    "Poltava Oblast": "UA-53", "Dnipropetrovsk Oblast": "UA-12",
    "Kirovohrad Oblast": "UA-35", "Kyiv": "UA-30",
}
UNSUPPORTED_SHAPES = {"Luhansk Oblast", "Autonomous Republic of Crimea", "Sevastopol"}


def select_model(source: str, as_of: datetime, root: Path = ROOT) -> Path:
    """Choose the latest saved baseline fold no later than the UTC origin.

    The prediction command independently validates the model's actual validity
    interval, source hash, cadence, and complete forecast horizon.
    """
    if source not in {"official", "volunteer"}:
        raise ValueError("Unknown alert archive")
    if as_of.tzinfo is None:
        raise ValueError("Forecast time must include a timezone")
    folder = root / "models" / "learning_baseline" / source
    candidates = sorted(folder.glob("????-??-??.joblib"))
    day = as_of.astimezone(timezone.utc).date().isoformat()
    eligible = [path for path in candidates if path.stem <= day]
    if not eligible:
        raise FileNotFoundError(
            f"No saved {source} model at or before {day}. Train the baseline first."
        )
    return eligible[-1]


def run_historical_forecast(model: Path, as_of: datetime, root: Path = ROOT) -> list[dict]:
    """Use the existing CLI so all point-in-time and snapshot checks stay intact."""
    if as_of.tzinfo is None:
        raise ValueError("Forecast time must include a timezone")
    command = [
        sys.executable, str(root / "scripts" / "predict_alert.py"),
        "--model", str(model), "--as-of", as_of.astimezone(timezone.utc).isoformat(),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    if result.returncode:
        message = result.stderr.strip() or "Historical forecast failed."
        raise ValueError(message)
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prediction command returned no regional forecasts")
    return rows


def display_rows(rows: list[dict]) -> list[dict]:
    """Keep abstentions distinct from a numerical zero-probability forecast."""
    out = []
    seen = set()
    for row in rows:
        region_id = row["oblast_id"]
        if region_id not in REGION_LABELS or region_id in seen:
            raise ValueError("Unsupported or duplicate region in prediction output")
        seen.add(region_id)
        status = row["status"]
        probability = row["probability"]
        if status == "historical_research_forecast":
            if probability is None or not 0 <= probability <= 1:
                raise ValueError("Forecast probability must be between zero and one")
            state = "Reconstructed clear"
            percent = 100 * probability
        elif status == "active_or_unresolved_in_archive":
            if probability is not None:
                raise ValueError("An active or unresolved region must abstain")
            state, percent = "Active or unresolved", None
        elif status == "no_training_examples_for_region":
            if probability is not None:
                raise ValueError("A region without training examples must abstain")
            state, percent = "Insufficient training data", None
        else:
            raise ValueError(f"Unknown prediction status: {status}")
        out.append({"Region": row["oblast"], "Region ID": region_id,
                    "Archive state": state, "Probability (%)": percent,
                    "status": status})
    return out


def format_forecast_time(forecast_at_utc: str, horizon_minutes: int) -> tuple[str, str]:
    start = datetime.fromisoformat(forecast_at_utc.replace("Z", "+00:00"))
    if start.tzinfo is None:
        raise ValueError("Forecast time must include a timezone")
    from datetime import timedelta
    end = start + timedelta(minutes=horizon_minutes)
    kyiv = ZoneInfo("Europe/Kyiv")
    return (start.astimezone(kyiv).strftime("%d %B %Y, %H:%M %Z"),
            end.astimezone(kyiv).strftime("%H:%M %Z"))


def _color(percent: float | None, state: str) -> list[int]:
    if state == "Active or unresolved":
        return [226, 151, 58, 205]
    if percent is None:
        return [174, 181, 191, 170]
    fraction = percent / 100
    return [int(210 - 170 * fraction), int(229 - 120 * fraction),
            int(245 - 55 * fraction), 215]


def map_geojson(boundaries: dict, rows: list[dict]) -> dict:
    """Join real administrative polygons to forecasts without inventing values."""
    if boundaries.get("type") != "FeatureCollection":
        raise ValueError("Map boundaries must be a GeoJSON FeatureCollection")
    by_id = {row["Region ID"]: row for row in rows}
    features = []
    names = set()
    for feature in boundaries.get("features", []):
        properties = dict(feature.get("properties", {}))
        shape_name = properties.get("shapeName")
        if shape_name in names or (shape_name not in MAP_SHAPE_TO_ID and shape_name not in UNSUPPORTED_SHAPES):
            raise ValueError(f"Unknown or duplicate map region: {shape_name}")
        names.add(shape_name)
        region_id = MAP_SHAPE_TO_ID.get(shape_name)
        row = by_id.get(region_id) if region_id else None
        state = row["Archive state"] if row else "Not modeled"
        percent = row["Probability (%)"] if row else None
        properties.update({
            "model_region_id": region_id or "",
            "display_name": row["Region"] if row else shape_name,
            "archive_state": state,
            "forecast_text": f"{percent:.1f}% next hour" if percent is not None else "No onset forecast",
            "fill_color": _color(percent, state),
        })
        features.append({"type": "Feature", "geometry": feature["geometry"],
                         "properties": properties})
    if set(MAP_SHAPE_TO_ID) - names:
        raise ValueError("Map file is missing modeled administrative regions")
    return {"type": "FeatureCollection", "features": features}
