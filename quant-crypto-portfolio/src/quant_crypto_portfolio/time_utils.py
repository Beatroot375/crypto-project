from __future__ import annotations

from datetime import UTC, datetime


def utc_now_ns() -> int:
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)


def ns_to_utc_str(ts_ns: int) -> str:
    ts = datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)
    return ts.isoformat()
