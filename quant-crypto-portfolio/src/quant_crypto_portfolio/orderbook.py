from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .time_utils import ns_to_utc_str, utc_now_ns

LOGGER = logging.getLogger(__name__)


@dataclass
class LocalOrderBook:
    symbol: str
    bids: dict[float, float] = field(default_factory=dict)  # price -> qty
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int | None = None
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid: float = 0.0

    def apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.bids.clear()
        self.asks.clear()

        for price, qty in snapshot.get("bids", []):
            self.bids[float(price)] = float(qty)
        for price, qty in snapshot.get("asks", []):
            self.asks[float(price)] = float(qty)

        self.last_update_id = int(snapshot["lastUpdateId"])
        self._update_best()

    def apply_diff(self, event: dict[str, Any]) -> bool:
        """
        Binance depthUpdate sequencing (simplified):
          - Ignore if event["u"] <= last_update_id
          - Accept if event["U"] <= last_update_id + 1 <= event["u"]
          - Otherwise gap => return False (caller should resync snapshot)
        """
        if self.last_update_id is None:
            return False

        first_update_id = int(event["U"])
        final_update_id = int(event["u"])

        if final_update_id <= self.last_update_id:
            return True

        if not (first_update_id <= self.last_update_id + 1 <= final_update_id):
            return False

        for price, qty in event.get("b", []):
            px = float(price)
            q = float(qty)
            if q == 0.0:
                self.bids.pop(px, None)
            else:
                self.bids[px] = q

        for price, qty in event.get("a", []):
            px = float(price)
            q = float(qty)
            if q == 0.0:
                self.asks.pop(px, None)
            else:
                self.asks[px] = q

        self.last_update_id = final_update_id
        self._update_best()
        return True

    def _update_best(self) -> None:
        self.best_bid = max(self.bids) if self.bids else 0.0
        self.best_ask = min(self.asks) if self.asks else 0.0
        self.mid = (self.best_bid + self.best_ask) / 2.0 if self.best_bid and self.best_ask else 0.0

    def to_snapshot_dict(self, levels: int = 200) -> dict[str, Any]:
        ts_ns = utc_now_ns()
        row: dict[str, Any] = {
            "ts_ns": ts_ns,
            "ts_utc": ns_to_utc_str(ts_ns),
            "symbol": self.symbol,
            "exchange": "BINANCE",
            "currency": "USDT",
            "snapshot_id": self.last_update_id,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid": self.mid,
        }

        n = max(1, int(levels))
        sorted_bids = sorted(self.bids.items(), reverse=True)[:n]
        sorted_asks = sorted(self.asks.items())[:n]
        for i, (p, q) in enumerate(sorted_bids, 1):
            row[f"bid_px_{i}"] = p
            row[f"bid_sz_{i}"] = q
        for i, (p, q) in enumerate(sorted_asks, 1):
            row[f"ask_px_{i}"] = p
            row[f"ask_sz_{i}"] = q
        return row


@dataclass
class MultiAssetOrderBook:
    books: dict[str, LocalOrderBook] = field(default_factory=dict)
    last_update_times: dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    total_updates: int = 0
    total_snapshots: int = 0

    def add_asset(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym in self.books:
            return
        self.books[sym] = LocalOrderBook(symbol=sym)
        self.last_update_times[sym] = time.time()
        LOGGER.info("Tracking symbol %s", sym)

    def apply_snapshot(self, symbol: str, snapshot: dict[str, Any]) -> None:
        self.add_asset(symbol)
        self.books[symbol.upper()].apply_snapshot(snapshot)
        self.last_update_times[symbol.upper()] = time.time()

    def apply_diff(self, symbol: str, event: dict[str, Any]) -> bool:
        sym = symbol.upper()
        if sym not in self.books:
            return False
        ok = self.books[sym].apply_diff(event)
        if ok:
            self.last_update_times[sym] = time.time()
            self.total_updates += 1
        return ok

    def get_snapshot(self, symbol: str, levels: int = 200) -> dict[str, Any]:
        sym = symbol.upper()
        if sym not in self.books:
            return {}
        snap = self.books[sym].to_snapshot_dict(levels=levels)
        return snap

    def get_all_snapshots(self, levels: int = 200) -> list[dict[str, Any]]:
        snaps: list[dict[str, Any]] = []
        for _sym, book in self.books.items():
            if book.mid <= 0:
                continue
            snaps.append(book.to_snapshot_dict(levels=levels))
            self.total_snapshots += 1
        return snaps

    def stats(self) -> dict[str, Any]:
        uptime = max(1e-9, time.time() - self.start_time)
        return {
            "asset_count": len(self.books),
            "total_updates": self.total_updates,
            "total_snapshots": self.total_snapshots,
            "uptime_hours": uptime / 3600.0,
            "updates_per_second": self.total_updates / uptime,
            "snapshots_per_second": self.total_snapshots / uptime,
        }
