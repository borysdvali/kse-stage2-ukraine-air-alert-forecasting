import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alert_model as m


def event(t, kind, token=None, region=0, locality=("oblast", "", "")):
    return (t, kind, region, locality, t if token is None else token)


class CausalityTests(unittest.TestCase):
    def test_future_records_do_not_change_past_features(self):
        events = sorted([event(10, 0), event(50, 1, 10), event(80, 0),
                         event(150, 1, 80), event(170, 0, region=1)])
        grid = np.array([20, 60, 100], dtype=np.int64)
        full = m.feature_frame(events, grid)
        for t in grid:
            prefix = m.feature_frame([e for e in events if e[0] <= t], np.array([t]))
            pd.testing.assert_frame_equal(full.loc[full.time == t].reset_index(drop=True), prefix)

    def test_old_end_cannot_close_new_activation(self):
        events = [event(10, 0), event(20, 0), event(30, 1, 10), event(40, 1, 20)]
        onsets, changes = m.replay(events)
        self.assertEqual(onsets[0].tolist(), [10])
        self.assertEqual(changes[0][0].tolist(), [10, 40])

    def test_touching_intervals_are_continuous(self):
        onsets, changes = m.replay([event(10, 0), event(20, 1, 10), event(20, 0), event(30, 1, 20)])
        self.assertEqual(onsets[0].tolist(), [10])
        self.assertEqual(changes[0][0].tolist(), [10, 30])

    def test_zero_length_record_does_not_leave_active_state(self):
        onsets, _ = m.replay([event(10, 0), event(10, 1, 10)])
        self.assertEqual(onsets[0].tolist(), [])

    def test_label_window_excludes_origin_includes_right_edge(self):
        for start, expected in [(100, 0), (160, 1), (161, 0)]:
            events = [event(start, 0)]
            f = m.attach_labels(m.feature_frame(events, np.array([100])), events, 60)
            self.assertEqual(int(f.loc[f.region_index == 0, "y"].iloc[0]), expected)

    def test_training_embargo(self):
        f = pd.DataFrame({"time": [10, 20, 30], "target_end": [70, 80, 90], "known_clear": True})
        self.assertEqual(m.eligible_training(f, 80, 0).time.tolist(), [10])

    def test_future_model_is_rejected(self):
        f = pd.DataFrame({"time": [100]})
        with self.assertRaisesRegex(ValueError, "trained after"):
            m.predict_bundle({"fit_cutoff": 101}, f)

    def test_uncertain_ends_omit_end_not_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            for source in ("official", "volunteer"):
                path = raw / f"{source}_data_en.csv"
                with path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["oblast", "started_at", "finished_at", "naive"])
                    writer.writeheader()
                    writer.writerow({"oblast": m.REGION_NAMES[0], "started_at": "2024-01-01T00:00:00Z",
                                     "finished_at": "2024-01-01T01:00:00Z", "naive": "True"})
                with patch.object(m, "ROOT", root):
                    events, _ = m.load_events(source, m.epoch("2024-01-02"))
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0][1], 0)

    def test_fixture_archive_prefix_invariance(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "synthetic_alerts.csv"
        with fixture.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 240)
        self.assertEqual(len({row["oblast"] for row in rows}), 8)
        self.assertTrue(all(row["started_at"].startswith("2030-") for row in rows))
        self.assertTrue(any(not row["finished_at"] for row in rows))
        self.assertTrue(any(row["naive"] == "True" for row in rows))

        at = m.epoch("2030-01-25T12:00:00Z")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            for source in ("official", "volunteer"):
                shutil.copyfile(fixture, raw / f"{source}_data_en.csv")
                with patch.object(m, "ROOT", root):
                    complete, _ = m.load_events(source, at + 86400)
                    prefix, _ = m.load_events(source, at)
                grid = np.array([at])
                pd.testing.assert_frame_equal(m.feature_frame(complete, grid), m.feature_frame(prefix, grid))

    @unittest.skipUnless(
        all((m.ROOT / "data" / "raw" / f"{source}_data_en.csv").exists()
            for source in ("official", "volunteer")),
        "Full historical archives are not installed",
    )
    def test_real_archive_prefix_invariance(self):
        at = m.epoch("2026-07-01T12:00:00Z")
        for source in ("official", "volunteer"):
            complete, _ = m.load_events(source, at + 86400)
            prefix, _ = m.load_events(source, at)
            grid = np.array([at])
            pd.testing.assert_frame_equal(m.feature_frame(complete, grid), m.feature_frame(prefix, grid))


if __name__ == "__main__":
    unittest.main()
