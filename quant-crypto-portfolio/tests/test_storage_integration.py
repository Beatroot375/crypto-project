from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from quant_crypto_portfolio.data_io import TruncatedGzipError, load_dataset
from quant_crypto_portfolio.storage import DailyPerSymbolGzipJsonlWriter


class TestStorageIntegration(unittest.TestCase):
    def test_writer_and_loader_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "data" / "l2"
            w = DailyPerSymbolGzipJsonlWriter(
                data_dir=base,
                validate=True,
                on_validation_error="raise",
            )

            ts = int(datetime(2026, 3, 14, 0, 0, 0, tzinfo=UTC).timestamp() * 1_000_000_000)
            row = {
                "ts_ns": ts,
                "symbol": "BTCUSDT",
                "mid": 100.0,
                "best_bid": 99.0,
                "best_ask": 101.0,
                "bid_px_1": 99.0,
                "bid_sz_1": 1.0,
                "ask_px_1": 101.0,
                "ask_sz_1": 1.0,
            }
            w.write("BTCUSDT", ts, row)
            w.close()

            ds = load_dataset(
                data_dir=base,
                symbol="BTCUSDT",
                from_day="2026-03-14",
                to_day="2026-03-14",
                max_rows=10,
                stride=1,
                signal_field=None,
                pred_class_field=None,
            )
            self.assertEqual(len(ds.mid), 1)

    def test_writer_validation_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "data" / "l2"
            w = DailyPerSymbolGzipJsonlWriter(
                data_dir=base,
                validate=True,
                on_validation_error="raise",
            )
            ts = int(datetime(2026, 3, 14, 0, 0, 0, tzinfo=UTC).timestamp() * 1_000_000_000)
            with self.assertRaises(ValueError):
                w.write("BTCUSDT", ts, {"ts_ns": ts, "symbol": "BTCUSDT"})

    def test_truncated_gzip_raise_when_disallowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "data" / "l2"
            w = DailyPerSymbolGzipJsonlWriter(data_dir=base)
            ts = int(datetime(2026, 3, 14, 0, 0, 0, tzinfo=UTC).timestamp() * 1_000_000_000)
            row = {
                "ts_ns": ts,
                "symbol": "BTCUSDT",
                "mid": 100.0,
                "best_bid": 99.0,
                "best_ask": 101.0,
                "bid_px_1": 99.0,
                "bid_sz_1": 1.0,
                "ask_px_1": 101.0,
                "ask_sz_1": 1.0,
            }
            w.write("BTCUSDT", ts, row)
            w.close()

            f = base / "2026-03-14" / "2026-03-14_btcusdt_l2.jsonl.gz"
            data = f.read_bytes()
            f.write_bytes(data[:-8])

            with self.assertRaises(TruncatedGzipError):
                load_dataset(
                    data_dir=base,
                    symbol="BTCUSDT",
                    from_day="2026-03-14",
                    to_day="2026-03-14",
                    max_rows=10,
                    stride=1,
                    signal_field=None,
                    pred_class_field=None,
                    allow_truncated=False,
                )


if __name__ == "__main__":
    unittest.main()
