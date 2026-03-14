from __future__ import annotations

import gzip
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_REQUIRED_FIELDS: tuple[str, ...] = ("ts_ns", "symbol", "mid", "best_bid", "best_ask")


def _is_finite_number(v: object) -> bool:
    try:
        x = float(v)  # type: ignore[arg-type]
    except Exception:
        return False
    return math.isfinite(x)


def validate_snapshot_row(
    row: dict[str, Any],
    *,
    required_fields: tuple[str, ...] = DEFAULT_REQUIRED_FIELDS,
) -> None:
    missing = [k for k in required_fields if k not in row]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if not _is_finite_number(row.get("ts_ns")):
        raise ValueError("Invalid ts_ns")
    if not isinstance(row.get("symbol"), str) or not row["symbol"]:
        raise ValueError("Invalid symbol")
    for k in ("mid", "best_bid", "best_ask"):
        if not _is_finite_number(row.get(k)):
            raise ValueError(f"Invalid {k}")


def sanitize_row_for_json(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
        else:
            out[k] = v
    return out


@dataclass
class DailyPerSymbolGzipJsonlWriter:
    data_dir: Path
    flush_interval_sec: float = 1.0
    encoding: str = "utf-8"
    validate: bool = False
    on_validation_error: str = "skip"  # "skip" or "raise"
    required_fields: tuple[str, ...] = DEFAULT_REQUIRED_FIELDS

    _handles: dict[str, gzip.GzipFile] = field(default_factory=dict, init=False)
    _day_str: dict[str, str] = field(default_factory=dict, init=False)
    _last_flush: float = field(default_factory=time.time, init=False)

    def write(self, symbol: str, ts_ns: int, row: dict[str, Any]) -> None:
        ts = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)
        day_str = ts.date().isoformat()
        sym = symbol.upper()

        if self._day_str.get(sym) != day_str:
            self._rotate(sym, day_str)

        handle = self._handles[sym]
        if self.validate:
            row = dict(row)
            row.setdefault("ts_ns", ts_ns)
            row.setdefault("symbol", sym)
            try:
                validate_snapshot_row(row, required_fields=self.required_fields)
            except Exception as e:
                if self.on_validation_error == "raise":
                    raise
                log.warning("Skipping invalid row for %s (%s)", sym, e)
                return
            row = sanitize_row_for_json(row)

        try:
            payload = json.dumps(row, allow_nan=not self.validate) + "\n"
        except ValueError:
            payload = json.dumps(sanitize_row_for_json(row), allow_nan=True) + "\n"
        handle.write(payload.encode(self.encoding))

    def maybe_flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_flush) < self.flush_interval_sec:
            return
        for handle in self._handles.values():
            handle.flush()
        self._last_flush = now

    def close(self) -> None:
        for handle in self._handles.values():
            handle.flush()
            handle.close()
        self._handles.clear()
        self._day_str.clear()

    def _rotate(self, symbol: str, day_str: str) -> None:
        sym = symbol.upper()
        if sym in self._handles:
            self._handles[sym].flush()
            self._handles[sym].close()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        day_dir = self.data_dir / day_str
        day_dir.mkdir(parents=True, exist_ok=True)

        file_path = day_dir / f"{day_str}_{sym.lower()}_l2.jsonl.gz"
        self._handles[sym] = gzip.open(file_path, "ab")
        self._day_str[sym] = day_str
