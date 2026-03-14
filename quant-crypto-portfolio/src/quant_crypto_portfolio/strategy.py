from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Trade:
    side: int  # 1 long, -1 short
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    ret: float


def positions_from_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=int)
    pos = np.zeros_like(signal, dtype=int)
    cur = 0
    for i, s in enumerate(signal):
        if cur == 0:
            if s == 1:
                cur = 1
            elif s == -1:
                cur = -1
        elif cur == 1 and s <= 0:
            cur = 0
        elif cur == -1 and s >= 0:
            cur = 0
        pos[i] = cur
    return pos


def trades_from_positions(mid: np.ndarray, pos: np.ndarray) -> list[Trade]:
    mid = np.asarray(mid, dtype=float)
    pos = np.asarray(pos, dtype=int)
    if len(mid) != len(pos):
        raise ValueError("mid and pos must have same length")
    trades: list[Trade] = []

    cur_side = 0
    entry_idx = -1
    for i in range(len(pos)):
        if cur_side == 0 and pos[i] != 0:
            cur_side = int(pos[i])
            entry_idx = i
            continue
        if cur_side != 0 and pos[i] == 0:
            exit_idx = i
            entry_px = float(mid[entry_idx])
            exit_px = float(mid[exit_idx])
            if cur_side == 1:
                r = (exit_px / max(1e-12, entry_px)) - 1.0
            else:
                r = (entry_px / max(1e-12, exit_px)) - 1.0
            trades.append(
                Trade(
                    side=cur_side,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    entry_px=entry_px,
                    exit_px=exit_px,
                    ret=float(r),
                )
            )
            cur_side = 0

    return trades


def equity_curve(mid: np.ndarray, pos: np.ndarray) -> np.ndarray:
    mid = np.asarray(mid, dtype=float)
    pos = np.asarray(pos, dtype=int)
    if len(mid) < 2:
        return np.ones_like(mid, dtype=float)
    r = (mid[1:] / mid[:-1]) - 1.0
    pnl = pos[:-1].astype(float) * r
    eq = np.empty(len(mid), dtype=float)
    eq[0] = 1.0
    eq[1:] = np.cumprod(1.0 + pnl)
    return eq


def strategy_metrics(mid: np.ndarray, pos: np.ndarray) -> dict[str, Any]:
    eq = equity_curve(mid, pos)
    trades = trades_from_positions(mid, pos)
    rets = np.array([t.ret for t in trades], dtype=float) if trades else np.array([], dtype=float)

    total_return = float(eq[-1] - 1.0)
    win_rate = float((rets > 0).mean()) if rets.size else None
    avg_trade = float(rets.mean()) if rets.size else None

    dd = np.maximum.accumulate(eq) - eq
    max_dd = float(dd.max()) if dd.size else 0.0

    return {
        "total_return": total_return,
        "trade_count": int(len(trades)),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade,
        "max_drawdown": max_dd,
    }

