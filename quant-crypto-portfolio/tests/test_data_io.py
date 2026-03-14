from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from quant_crypto_portfolio.data_io import iter_snapshots


class TestDataIo(unittest.TestCase):
    def test_iter_snapshots_tolerates_missing_gzip_footer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.jsonl.gz"

            rows = [{"a": 1}, {"a": 2}, {"a": 3}]
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            # Truncate footer (CRC32 + ISIZE = 8 bytes)
            # to simulate a gzip stream without a footer yet.
            data = path.read_bytes()
            path.write_bytes(data[:-8])

            out = list(iter_snapshots([path]))
            self.assertGreaterEqual(len(out), 1)
            self.assertEqual(out[0]["a"], 1)

    def test_load_dataset_extra_fields(self) -> None:
        import json

        import numpy as np

        from quant_crypto_portfolio.data_io import load_dataset

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "data" / "l2" / "2026-03-14"
            base.mkdir(parents=True, exist_ok=True)
            fpath = base / "2026-03-14_btcusdt_l2.jsonl.gz"

            rows = [
                {"ts_ns": 1, "mid": 100.0, "bid_px_1": 99.0, "ask_px_1": 101.0, "x": 1.0},
                {"ts_ns": 2, "mid": 101.0, "bid_px_1": 100.0, "ask_px_1": 102.0, "x": None},
            ]
            with gzip.open(fpath, "wt", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

            ds = load_dataset(
                data_dir=Path(td) / "data" / "l2",
                symbol="BTCUSDT",
                from_day="2026-03-14",
                to_day="2026-03-14",
                max_rows=10,
                stride=1,
                signal_field=None,
                pred_class_field=None,
                extra_fields=["x"],
            )
            self.assertIsNotNone(ds.extra)
            assert ds.extra is not None
            self.assertIn("x", ds.extra)
            self.assertTrue(np.isfinite(ds.extra["x"][0]))
            self.assertTrue(np.isnan(ds.extra["x"][1]))


if __name__ == "__main__":
    unittest.main()
