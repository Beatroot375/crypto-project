from __future__ import annotations

import unittest

import numpy as np

from quant_crypto_portfolio.strategy import positions_from_signal, trades_from_positions


class TestStrategy(unittest.TestCase):
    def test_positions_and_trades(self) -> None:
        signal = np.array([0, 1, 1, 0, -1, -1, 0], dtype=int)
        pos = positions_from_signal(signal)
        self.assertListEqual(pos.tolist(), [0, 1, 1, 0, -1, -1, 0])

        mid = np.array([100, 101, 102, 103, 102, 101, 100], dtype=float)
        trades = trades_from_positions(mid, pos)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0].side, 1)
        self.assertEqual(trades[0].entry_idx, 1)
        self.assertEqual(trades[0].exit_idx, 3)
        self.assertEqual(trades[1].side, -1)
        self.assertEqual(trades[1].entry_idx, 4)
        self.assertEqual(trades[1].exit_idx, 6)


if __name__ == "__main__":
    unittest.main()

