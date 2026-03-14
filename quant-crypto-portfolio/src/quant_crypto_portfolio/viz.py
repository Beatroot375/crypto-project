from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .strategy import equity_curve, trades_from_positions

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


@dataclass(frozen=True)
class PlotResult:
    out_path: Path
    trade_count: int


def plot_signals(
    ts_ns: np.ndarray,
    mid: np.ndarray,
    pos: np.ndarray,
    *,
    out_path: Path,
    title: str = "Signals (entry/exit)",
) -> PlotResult:
    if plt is None:
        raise RuntimeError("Install extra: pip install -e '.[viz]'")

    ts = [datetime.fromtimestamp(int(t) / 1_000_000_000, tz=UTC) for t in ts_ns]
    mid = np.asarray(mid, dtype=float)
    pos = np.asarray(pos, dtype=int)
    trades = trades_from_positions(mid, pos)
    eq = equity_curve(mid, pos)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[3, 1])
    fig.suptitle(title)

    ax1.plot(ts, mid, linewidth=1.0, label="mid")
    for t in trades:
        ax1.scatter(ts[t.entry_idx], t.entry_px, marker="^" if t.side == 1 else "v", s=35)
        ax1.scatter(ts[t.exit_idx], t.exit_px, marker="x", s=35)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylabel("Price")

    ax2.plot(ts, eq, linewidth=1.0, label="equity")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylabel("Equity")
    ax2.set_xlabel("Time (UTC)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return PlotResult(out_path=out_path, trade_count=len(trades))
