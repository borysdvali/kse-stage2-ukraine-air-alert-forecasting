"""Presentation tests: preserve abstentions and avoid inventing map forecasts."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dashboard_data import (  # noqa: E402
    MAP_SHAPE_TO_ID, REGION_LABELS, UNSUPPORTED_SHAPES,
    display_rows, format_forecast_time, map_geojson, select_model,
)


class DashboardDataTests(unittest.TestCase):
    def test_selects_latest_model_no_later_than_origin(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "models" / "learning_baseline" / "volunteer"
            folder.mkdir(parents=True)
            for name in ("2026-01-01.joblib", "2026-01-06.joblib"):
                (folder / name).touch()
            selected = select_model("volunteer", datetime(2026, 1, 4, tzinfo=timezone.utc), Path(temp))
            self.assertEqual(selected.name, "2026-01-01.joblib")
            with self.assertRaises(FileNotFoundError):
                select_model("volunteer", datetime(2025, 12, 31, tzinfo=timezone.utc), Path(temp))

    def test_abstention_is_not_zero_probability(self):
        rows = display_rows([
            {"oblast_id": "UA-05", "oblast": "Vinnytska oblast",
             "status": "historical_research_forecast", "probability": 0.0},
            {"oblast_id": "UA-51", "oblast": "Odeska oblast",
             "status": "active_or_unresolved_in_archive", "probability": None},
        ])
        self.assertEqual(rows[0]["Probability (%)"], 0.0)
        self.assertIsNone(rows[1]["Probability (%)"])
        self.assertEqual(rows[1]["Archive state"], "Active or unresolved")

    def test_invalid_probability_and_active_forecast_are_rejected(self):
        for status, probability in (("historical_research_forecast", 1.1),
                                    ("active_or_unresolved_in_archive", 0.2)):
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    display_rows([{"oblast_id": "UA-05", "oblast": "Vinnytska oblast",
                                   "status": status, "probability": probability}])

    def test_map_join_preserves_unsupported_and_missing_forecasts(self):
        geometry = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"shapeName": shape},
             "geometry": {"type": "Polygon", "coordinates": []}}
            for shape in sorted(set(MAP_SHAPE_TO_ID) | UNSUPPORTED_SHAPES)
        ]}
        rows = display_rows([
            {"oblast_id": "UA-05", "oblast": REGION_LABELS["UA-05"],
             "status": "historical_research_forecast", "probability": 0.25},
        ])
        mapped = map_geojson(geometry, rows)
        by_name = {f["properties"]["shapeName"]: f["properties"] for f in mapped["features"]}
        self.assertEqual(by_name["Vinnytsia Oblast"]["forecast_text"], "25.0% next hour")
        self.assertEqual(by_name["Kyiv Oblast"]["forecast_text"], "No onset forecast")
        self.assertEqual(by_name["Autonomous Republic of Crimea"]["archive_state"], "Not modeled")
        self.assertEqual(by_name["Autonomous Republic of Crimea"]["model_region_id"], "")

    def test_kyiv_time_handles_daylight_saving(self):
        summer = format_forecast_time("2026-09-03T12:00:00+00:00", 60)
        winter = format_forecast_time("2026-01-03T12:00:00+00:00", 60)
        self.assertIn("15:00 EEST", summer[0])
        self.assertIn("14:00 EET", winter[0])


if __name__ == "__main__":
    unittest.main()
