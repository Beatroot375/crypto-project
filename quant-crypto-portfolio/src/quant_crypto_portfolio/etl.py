from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SymbolSummary:
    snapshot_count: int = 0
    spread_bps_sum: float = 0.0
    mid_min: float = float("inf")
    mid_max: float = float("-inf")

    def update(self, row: dict[str, Any]) -> None:
        mid = float(row.get("mid", 0.0) or 0.0)
        best_bid = float(row.get("best_bid", 0.0) or 0.0)
        best_ask = float(row.get("best_ask", 0.0) or 0.0)
        if mid <= 0.0:
            return
        spread_bps = ((best_ask - best_bid) / mid) * 1e4 if mid else 0.0
        self.snapshot_count += 1
        self.spread_bps_sum += float(spread_bps)
        self.mid_min = min(self.mid_min, mid)
        self.mid_max = max(self.mid_max, mid)

    def to_dict(self) -> dict[str, Any]:
        if self.snapshot_count == 0:
            return {
                "snapshot_count": 0,
                "avg_spread_bps": None,
                "min_mid_price": None,
                "max_mid_price": None,
                "price_range": None,
            }
        avg_spread = self.spread_bps_sum / self.snapshot_count
        return {
            "snapshot_count": self.snapshot_count,
            "avg_spread_bps": float(avg_spread),
            "min_mid_price": float(self.mid_min),
            "max_mid_price": float(self.mid_max),
            "price_range": float(self.mid_max - self.mid_min),
        }


def daily_etl_and_report(data_dir: Path, day: str | None = None) -> Path | None:
    """
    Produces a JSON summary report for a single UTC day.
    Defaults to yesterday (UTC) if day is None.
    """
    if day is None:
        day = (datetime.now(UTC) - timedelta(days=1)).date().isoformat()

    day_dir = data_dir / day
    if not day_dir.exists():
        log.warning("No data directory for %s (%s)", day, day_dir)
        return None

    summaries: dict[str, SymbolSummary] = {}
    files = sorted(day_dir.glob(f"{day}_*_l2.jsonl.gz"))
    if not files:
        log.warning("No data files for %s in %s", day, day_dir)
        return None

    for path in files:
        symbol = path.name.split("_")[1].split(".")[0].upper()
        summaries.setdefault(symbol, SymbolSummary())
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                summaries[symbol].update(row)

    out = {
        "date": day,
        "symbols": {sym: s.to_dict() for sym, s in sorted(summaries.items())},
    }

    report_dir = data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{day}_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote report %s", out_path)
    return out_path


def parse_day(day: str | None) -> str | None:
    if day is None:
        return None
    day = day.strip()
    if not day:
        return None
    # minimal validation
    date.fromisoformat(day)
    return day
