from __future__ import annotations

import unittest

from quant_crypto_portfolio.aggtrades import AggTradeWindow


class TestAggTrades(unittest.TestCase):
    def test_window_prunes_and_counts(self) -> None:
        w = AggTradeWindow(window_ns=1_000_000_000, maxlen=10)

        # Older than window (2s ago)
        w.on_agg_trade({"T": 1_000, "p": "100.0", "q": "1.0", "m": False})
        # Recent (0.5s ago)
        w.on_agg_trade({"T": 2_500, "p": "101.0", "q": "2.0", "m": True})

        # "Now" at 3s
        stats = w.stats(now_ts_ns=3_000 * 1_000_000)
        self.assertEqual(stats["agg_trade_count"], 1)
        self.assertAlmostEqual(stats["agg_sell_qty"], 2.0)
        self.assertAlmostEqual(stats["agg_buy_qty"], 0.0)

    def test_imbalance_and_vwap(self) -> None:
        w = AggTradeWindow(window_ns=10_000_000_000, maxlen=10)
        w.on_agg_trade({"T": 1_000, "p": "100.0", "q": "1.0", "m": False})  # taker buy
        w.on_agg_trade({"T": 1_001, "p": "110.0", "q": "1.0", "m": True})  # taker sell

        s = w.stats(now_ts_ns=2_000 * 1_000_000)
        self.assertEqual(s["agg_trade_count"], 2)
        self.assertAlmostEqual(s["agg_vwap"], 105.0)
        self.assertAlmostEqual(s["agg_imbalance"], 0.0)


if __name__ == "__main__":
    unittest.main()

