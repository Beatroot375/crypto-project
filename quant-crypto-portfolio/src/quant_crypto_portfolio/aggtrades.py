from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AggTradeWindow:
    window_ns: int
    maxlen: int = 200_000

    _trades: deque[tuple[int, float, float, int]] = field(default_factory=deque, init=False)
    _last_px: float | None = None
    _last_qty: float | None = None
    _last_side: int | None = None  # 1 taker buy, -1 taker sell
    _buy_qty: float = 0.0
    _sell_qty: float = 0.0
    _qty_sum: float = 0.0
    _notional: float = 0.0

    def reset(self) -> None:
        self._trades.clear()
        self._last_px = None
        self._last_qty = None
        self._last_side = None
        self._buy_qty = 0.0
        self._sell_qty = 0.0
        self._qty_sum = 0.0
        self._notional = 0.0

    def on_agg_trade(self, payload: dict[str, Any]) -> None:
        """
        Binance aggTrade event fields used:
          T (trade time ms), p (price), q (qty), m (buyer is market maker)
        Side convention:
          m == True  => buyer is maker => taker sell (-1)
          m == False => buyer is taker => taker buy (+1)
        """
        t_ms = payload.get("T")
        p = payload.get("p")
        q = payload.get("q")
        if t_ms is None or p is None or q is None:
            return

        ts_ns = int(t_ms) * 1_000_000
        px = float(p)
        qty = float(q)
        side = -1 if bool(payload.get("m", False)) else 1

        self._trades.append((ts_ns, px, qty, side))
        while len(self._trades) > int(self.maxlen):
            ts0, px0, qty0, side0 = self._trades.popleft()
            self._notional -= px0 * qty0
            self._qty_sum -= qty0
            if side0 == 1:
                self._buy_qty -= qty0
            else:
                self._sell_qty -= qty0

        self._notional += px * qty
        self._qty_sum += qty
        if side == 1:
            self._buy_qty += qty
        else:
            self._sell_qty += qty

        self._last_px = px
        self._last_qty = qty
        self._last_side = side

    def _prune(self, now_ts_ns: int) -> None:
        cutoff = int(now_ts_ns) - int(self.window_ns)
        while self._trades and self._trades[0][0] < cutoff:
            _ts, px, qty, side = self._trades.popleft()
            self._notional -= px * qty
            self._qty_sum -= qty
            if side == 1:
                self._buy_qty -= qty
            else:
                self._sell_qty -= qty

    def stats(self, now_ts_ns: int) -> dict[str, Any]:
        self._prune(now_ts_ns)

        count = len(self._trades)
        buy_qty = max(0.0, float(self._buy_qty))
        sell_qty = max(0.0, float(self._sell_qty))
        qty_sum = max(0.0, float(self._qty_sum))
        notional = float(self._notional)

        vwap = (notional / qty_sum) if qty_sum > 0 else None
        denom = buy_qty + sell_qty
        imbalance = ((buy_qty - sell_qty) / denom) if denom > 0 else None

        return {
            "agg_trade_count": int(count),
            "agg_buy_qty": float(buy_qty),
            "agg_sell_qty": float(sell_qty),
            "agg_imbalance": float(imbalance) if imbalance is not None else None,
            "agg_vwap": float(vwap) if vwap is not None else None,
            "agg_last_px": self._last_px,
            "agg_last_qty": self._last_qty,
            "agg_last_side": self._last_side,
        }
