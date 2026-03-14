from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .features import snapshot_to_feature_vector

log = logging.getLogger(__name__)
_TRUNCATED_WARNED: set[str] = set()
_OPEN_FAILED_WARNED: set[str] = set()


def _iter_days(data_dir: Path) -> list[str]:
    days = []
    for p in sorted(data_dir.glob("20*")):
        if p.is_dir():
            try:
                date.fromisoformat(p.name)
            except ValueError:
                continue
            days.append(p.name)
    return days


def iter_snapshot_files(
    data_dir: Path,
    symbol: str,
    from_day: str | None = None,
    to_day: str | None = None,
) -> list[Path]:
    sym = symbol.upper()
    days = _iter_days(data_dir)
    if from_day:
        date.fromisoformat(from_day)
        days = [d for d in days if d >= from_day]
    if to_day:
        date.fromisoformat(to_day)
        days = [d for d in days if d <= to_day]
    out: list[Path] = []
    for day in days:
        path = data_dir / day / f"{day}_{sym.lower()}_l2.jsonl.gz"
        if path.exists():
            out.append(path)
    return out


class TruncatedGzipError(RuntimeError):
    pass


def iter_snapshots(
    paths: Iterable[Path],
    *,
    allow_truncated: bool = True,
) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            f = gzip.open(path, "rt", encoding="utf-8")
        except OSError as e:
            key = str(path)
            if key not in _OPEN_FAILED_WARNED:
                log.warning("Failed to open gzip %s (%s); skipping", path, e)
                _OPEN_FAILED_WARNED.add(key)
            continue

        with f:
            while True:
                try:
                    line = f.readline()
                except EOFError as e:
                    # Common when reading a gzip file that is still being written (no footer yet),
                    # or when the last gzip member is truncated after an unclean shutdown.
                    if not allow_truncated:
                        raise TruncatedGzipError(f"Truncated gzip stream: {path}") from e
                    key = str(path)
                    if key not in _TRUNCATED_WARNED:
                        log.warning(
                            "Truncated gzip stream %s (%s); using readable prefix",
                            path,
                            e,
                        )
                        _TRUNCATED_WARNED.add(key)
                    break
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


@dataclass(frozen=True)
class Dataset:
    ts_ns: np.ndarray
    mid: np.ndarray
    X: np.ndarray
    raw_signals: np.ndarray | None = None
    raw_pred_class: np.ndarray | None = None
    extra: dict[str, np.ndarray] | None = None


def load_dataset(
    data_dir: Path,
    symbol: str,
    *,
    from_day: str | None = None,
    to_day: str | None = None,
    depth: int = 200,
    max_rows: int | None = 200_000,
    stride: int = 1,
    signal_field: str | None = "online_signal",
    pred_class_field: str | None = "online_pred_class",
    extra_fields: list[str] | None = None,
    allow_truncated: bool = True,
) -> Dataset:
    paths = iter_snapshot_files(data_dir, symbol, from_day=from_day, to_day=to_day)
    if not paths:
        raise FileNotFoundError(f"No snapshot files found for {symbol} in {data_dir}")

    ts_list: list[int] = []
    mid_list: list[float] = []
    X_list: list[np.ndarray] = []
    sig_list: list[int] = []
    pred_list: list[int] = []
    has_sig = False
    has_pred = False
    extra_lists: dict[str, list[float | int | None]] = {}
    if extra_fields:
        for k in extra_fields:
            extra_lists[str(k)] = []

    count = 0
    for row in iter_snapshots(paths, allow_truncated=allow_truncated):
        if stride > 1 and (count % stride) != 0:
            count += 1
            continue
        count += 1
        ts = row.get("ts_ns")
        mid = row.get("mid")
        if ts is None or mid is None:
            continue
        ts_list.append(int(ts))
        mid_list.append(float(mid))
        X_list.append(snapshot_to_feature_vector(row, depth=depth))

        if signal_field and signal_field in row:
            sig_list.append(int(row.get(signal_field) or 0))
            has_sig = True
        if pred_class_field and pred_class_field in row:
            pred_list.append(int(row.get(pred_class_field) or 0))
            has_pred = True

        if extra_lists:
            for k, lst in extra_lists.items():
                lst.append(row.get(k))

        if max_rows is not None and len(ts_list) >= max_rows:
            break

    if not ts_list:
        raise RuntimeError("No usable rows found (missing ts_ns/mid?)")

    ts_arr = np.asarray(ts_list, dtype=np.int64)
    mid_arr = np.asarray(mid_list, dtype=float)
    X = np.vstack(X_list).astype(float, copy=False)

    raw_signals = (
        np.asarray(sig_list, dtype=int) if has_sig and len(sig_list) == len(ts_list) else None
    )
    raw_pred = (
        np.asarray(pred_list, dtype=int) if has_pred and len(pred_list) == len(ts_list) else None
    )
    extra: dict[str, np.ndarray] | None = None
    if extra_lists:
        extra = {}
        for k, lst in extra_lists.items():
            # best-effort numeric conversion; preserve None as NaN for floats
            arr = np.asarray([np.nan if v is None else v for v in lst], dtype=float)
            if len(arr) != len(ts_arr):
                continue
            extra[k] = arr

    return Dataset(
        ts_ns=ts_arr,
        mid=mid_arr,
        X=X,
        raw_signals=raw_signals,
        raw_pred_class=raw_pred,
        extra=extra,
    )
