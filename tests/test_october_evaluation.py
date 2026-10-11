import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_october_2026 import (eligible_duration_mask, make_features,
                                  predict_onset, reconstructed_episodes)


def event(t, kind, token=None, region=0):
    return (t, kind, region, ("oblast", "", ""), t if token is None else token)


class OctoberEvaluationTests(unittest.TestCase):
    def test_cross_region_features_unchanged_by_future_events(self):
        events = [event(10, 0), event(30, 1, 10), event(40, 0, region=1), event(80, 1, 40, 1)]
        grid = np.array([20, 35, 60], dtype=np.int64)
        all_features = make_features(events, grid)
        for t in grid:
            prefix = make_features([e for e in events if e[0] <= t], np.array([t]))
            pd.testing.assert_frame_equal(all_features.loc[all_features.time == t].reset_index(drop=True), prefix)

    def test_duration_label_must_end_strictly_before_cutoff(self):
        ep = pd.DataFrame({"start_epoch": [10, 10, 10, 20], "end_epoch": [19, 20, 21, np.nan],
                           "duration_target_eligible": [True, True, True, False]})
        self.assertEqual(eligible_duration_mask(ep, 0, 20).tolist(), [True, False, False, False])

    def test_unresolved_duration_not_given_invented_end(self):
        ep = reconstructed_episodes([event(10, 0)], set())
        self.assertTrue(pd.isna(ep.end_epoch.iloc[0]))
        self.assertFalse(ep.duration_target_eligible.iloc[0])

    def test_uncertain_activation_taints_merged_episode(self):
        ep = reconstructed_episodes([event(10, 0), event(20, 0), event(30, 1, 20)], {(0, 10)})
        self.assertEqual(ep.start_epoch.iloc[0], 10)
        self.assertEqual(ep.end_epoch.iloc[0], 30)
        self.assertFalse(ep.duration_target_eligible.iloc[0])

    def test_touching_duration_episodes_are_one_episode(self):
        ep = reconstructed_episodes(sorted([event(10, 0), event(20, 1, 10), event(20, 0), event(30, 1, 20)]), set())
        self.assertEqual(len(ep), 1)
        self.assertAlmostEqual(ep.duration_minutes.iloc[0], 20 / 60)

    def test_october_outcome_window_is_strictly_contained(self):
        from evaluate_october_2026 import TEST_START, TEST_END, HORIZON
        grid = np.arange(TEST_START, TEST_END - HORIZON, HORIZON)
        self.assertEqual(len(grid), 239)
        self.assertTrue((grid + HORIZON < TEST_END).all())

    def test_forecast_cannot_precede_calibration_end(self):
        f = pd.DataFrame({"time": [100]})
        with self.assertRaisesRegex(ValueError, "calibration"):
            predict_onset({"calibration_end_utc": "2026-10-01T00:00:00Z"}, f)


if __name__ == "__main__":
    unittest.main()
