from __future__ import annotations

import unittest

from quant_crypto_portfolio.orderbook import LocalOrderBook


class TestOrderBook(unittest.TestCase):
    def test_diff_sequence_accepts_overlap(self) -> None:
        book = LocalOrderBook(symbol="BTCUSDT")
        book.apply_snapshot(
            {
                "lastUpdateId": 10,
                "bids": [["100.0", "1.0"]],
                "asks": [["101.0", "1.0"]],
            }
        )

        ok = book.apply_diff(
            {
                "U": 11,
                "u": 12,
                "b": [["100.0", "2.0"]],
                "a": [],
            }
        )
        self.assertTrue(ok)
        self.assertEqual(book.last_update_id, 12)
        self.assertEqual(book.bids[100.0], 2.0)

    def test_diff_sequence_gap_detected(self) -> None:
        book = LocalOrderBook(symbol="BTCUSDT")
        book.apply_snapshot({"lastUpdateId": 10, "bids": [], "asks": []})

        ok = book.apply_diff({"U": 20, "u": 21, "b": [], "a": []})
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
