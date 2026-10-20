import unittest

import numpy as np
import pandas as pd

from generate_datasets_pfm_index import fit_saturation, original, pfm_congestion, training_cutoff


class PFMIndexTests(unittest.TestCase):
    def test_formula_matches_document_example(self):
        traffic = pd.DataFrame({"id": [1], "intensity": [500.], "occupation": [20.]})
        value = pfm_congestion(traffic, pd.Series({1: 1000.}))
        self.assertAlmostEqual(float(value.iloc[0]), .6)

    def test_zero_intensity_and_saturation_clipping(self):
        traffic = pd.DataFrame({"id": [1, 1, 1], "intensity": [0., 2000., 500.], "occupation": [20., 20., 100.]})
        np.testing.assert_allclose(pfm_congestion(traffic, pd.Series({1: 1000.})), [.2, 1., 1.])

    def test_invalid_indices_follow_original_zero_fill_policy(self):
        traffic = pd.DataFrame({"id": [1, 1, 1, 2, 3], "intensity": [-1., np.inf, 10., 0., 50.], "occupation": [20., 20., 101., 20., 20.]})
        values = pfm_congestion(traffic, pd.Series({1: 100., 2: 0.}))
        self.assertTrue(values.isna().all())
        np.testing.assert_array_equal(values.fillna(0.), np.zeros(5))

    def test_percentile_is_per_sensor_and_excludes_future(self):
        times = pd.date_range("2025-01-01", periods=4, freq="15min")
        traffic = pd.DataFrame({"id": [1]*4+[2]*4, "date": list(times)*2,
                                "intensity": [10., 20., 30., 10000., 100., 200., 300., 100000.]})
        capacity = fit_saturation(traffic, times[2])
        self.assertAlmostEqual(capacity.loc[1], 29.6)
        self.assertAlmostEqual(capacity.loc[2], 296.)
        traffic.loc[traffic["date"] > times[2], "intensity"] = 1e9
        pd.testing.assert_series_equal(capacity, fit_saturation(traffic, times[2]))

    def test_negative_intensities_do_not_fit_saturation(self):
        cutoff = pd.Timestamp("2025-01-01")
        traffic = pd.DataFrame({"id": [1]*4, "date": [cutoff]*4, "intensity": [-100., np.inf, 0., 100.]})
        self.assertAlmostEqual(fit_saturation(traffic, cutoff).loc[1], 98.)

    def test_cutoff_matches_original_last_training_target(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="15min")
        values = original.jnp.arange(30, dtype=original.jnp.float32)[:, None]
        for horizon in (1, 2):
            with self.subTest(horizon=horizon):
                x, y = original.create_windows(values, window=3, horizon=horizon)
                _, train_y, _, _, _, _ = original.split_time(x, y, .7)
                position = int(train_y[-1, -1, 0, 0])
                self.assertEqual(training_cutoff(dates, 3, horizon, .7), dates[position])

    def test_formula_does_not_modify_input_or_other_columns(self):
        traffic = pd.DataFrame({"id": [1], "intensity": [500.], "occupation": [20.], "congestion": [.04], "daytype": ["WD"]})
        before = traffic.copy(deep=True)
        pfm_congestion(traffic, pd.Series({1: 1000.}))
        pd.testing.assert_frame_equal(traffic, before)

    def test_cutoff_rejects_invalid_or_too_short_splits(self):
        dates = pd.date_range("2025-01-01", periods=4, freq="15min")
        with self.assertRaises(ValueError):
            training_cutoff(dates, window=12)
        with self.assertRaises(ValueError):
            training_cutoff(dates, window=1, train_ratio=1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
