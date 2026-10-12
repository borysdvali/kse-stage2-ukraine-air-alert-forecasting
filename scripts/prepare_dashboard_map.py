#!/usr/bin/env python3
"""Download the pinned, simplified Ukraine ADM1 polygons used by the local dashboard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from dashboard_data import BOUNDARY_PATH, MAP_SHAPE_TO_ID, UNSUPPORTED_SHAPES

URL = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/"
    "releaseData/gbOpen/UKR/ADM1/geoBoundaries-UKR-ADM1_simplified.geojson"
)
EXPECTED_SHA256 = "4a5947e7497574d51f93255dfa7e03dce0c7acf6ecee52d25773a5205854a399"


def validate(raw: bytes) -> dict:
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Boundary file differs from the pinned geoBoundaries snapshot")
    geometry = json.loads(raw)
    names = {f["properties"]["shapeName"] for f in geometry["features"]}
    if geometry.get("type") != "FeatureCollection" or names != set(MAP_SHAPE_TO_ID) | UNSUPPORTED_SHAPES:
        raise ValueError("Unexpected administrative geometry or region names")
    return geometry


def main() -> None:
    with urlopen(URL, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Unexpectedly large boundary download")
    geometry = validate(raw)
    BOUNDARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    BOUNDARY_PATH.write_bytes(raw)
    print(f"Saved {len(geometry['features'])} administrative areas to {BOUNDARY_PATH}")
    print("Source: geoBoundaries gbOpen UKR ADM1 (2017 reference), ODbL 1.0")


if __name__ == "__main__":
    main()
