from __future__ import annotations

import unittest

import numpy as np

from quant_crypto_portfolio.offline_eval import offline_eval_with_series


class TestOfflineEvalSeries(unittest.TestCase):
    def test_offline_eval_with_series_shapes(self) -> None:
        try:
            import sklearn  # noqa: F401
        except Exception:
            self.skipTest("scikit-learn not installed")

        rng = np.random.default_rng(0)
        n = 1000
        ts = np.arange(n, dtype=np.int64) * 1_000_000_000
        mid = 100 + np.cumsum(rng.normal(0, 0.01, size=n)).astype(float)
        X = rng.normal(0, 1, size=(n, 8)).astype(float)

        res, series = offline_eval_with_series(
            X,
            ts,
            mid,
            horizon_sec=10,
            ret_threshold=0.0,
            test_frac=0.2,
            score_threshold=0.1,
        )
        self.assertIn("accuracy", res.classification)
        self.assertGreater(series.split_idx, 0)
        self.assertEqual(len(series.y_true), n - series.split_idx)
        self.assertEqual(series.score.shape, series.y_true.shape)
        self.assertEqual(series.signal.shape, series.y_true.shape)
        self.assertEqual(series.position.shape, series.y_true.shape)


if __name__ == "__main__":
    unittest.main()
