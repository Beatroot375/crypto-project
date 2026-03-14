from __future__ import annotations

import unittest

import numpy as np

from quant_crypto_portfolio.features import snapshot_to_feature_vector


class TestFeatures(unittest.TestCase):
    def test_feature_vector_shape(self) -> None:
        snap = {
            "bid_px_1": 100.0,
            "ask_px_1": 101.0,
            "bid_sz_1": 1.0,
            "ask_sz_1": 2.0,
            "bid_px_2": 99.0,
            "bid_sz_2": 1.0,
            "ask_px_2": 102.0,
            "ask_sz_2": 1.0,
        }
        x = snapshot_to_feature_vector(snap, depth=2)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (8,))


if __name__ == "__main__":
    unittest.main()
