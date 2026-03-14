from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURE_NAMES = [
    "spread_bps",
    "imbalance_l1",
    "imbalance_l3",
    "imbalance_l5",
    "imbalance_l10",
    "microprice_dev_bps",
    "book_pressure",
    "depth_slope",
]


def snapshot_to_feature_vector(snapshot: dict[str, Any], depth: int = 200) -> np.ndarray:
    def g(name: str, default: float = 0.0) -> float:
        v = snapshot.get(name, default)
        if v is None:
            return default
        if isinstance(v, float) and math.isnan(v):
            return default
        return float(v)

    best_bid = g("bid_px_1")
    best_ask = g("ask_px_1")
    mid = max(1e-9, (best_bid + best_ask) / 2.0)
    spread = max(0.0, best_ask - best_bid)
    spread_bps = spread / mid * 1e4

    bid_sz_1 = g("bid_sz_1")
    ask_sz_1 = g("ask_sz_1")
    imbalance_l1 = (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1 + 1e-12)

    def imbalance_k(k: int) -> float:
        k = max(1, min(int(k), int(depth)))
        b = sum(g(f"bid_sz_{i}") for i in range(1, k + 1))
        a = sum(g(f"ask_sz_{i}") for i in range(1, k + 1))
        return (b - a) / (b + a + 1e-12)

    imbalance_l3 = imbalance_k(3)
    imbalance_l5 = imbalance_k(5)
    imbalance_l10 = imbalance_k(10)

    microprice = ((best_ask * bid_sz_1) + (best_bid * ask_sz_1)) / (bid_sz_1 + ask_sz_1 + 1e-12)
    microprice_dev_bps = (microprice - mid) / mid * 1e4

    bid_pressure = 0.0
    ask_pressure = 0.0
    for i in range(1, max(1, int(depth)) + 1):
        w = 1.0 / i
        bid_pressure += g(f"bid_sz_{i}") * w
        ask_pressure += g(f"ask_sz_{i}") * w
    book_pressure = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-12)

    k = min(max(1, int(depth)), 1000)
    bsz = np.array([g(f"bid_sz_{i}") for i in range(1, k + 1)], dtype=float)
    asz = np.array([g(f"ask_sz_{i}") for i in range(1, k + 1)], dtype=float)
    bpx = np.array([g(f"bid_px_{i}") for i in range(1, k + 1)], dtype=float)
    apx = np.array([g(f"ask_px_{i}") for i in range(1, k + 1)], dtype=float)
    bid_vwap = float((bpx * bsz).sum() / (bsz.sum() + 1e-12))
    ask_vwap = float((apx * asz).sum() / (asz.sum() + 1e-12))
    depth_slope = (mid - bid_vwap) - (ask_vwap - mid)

    return np.array(
        [
            spread_bps,
            imbalance_l1,
            imbalance_l3,
            imbalance_l5,
            imbalance_l10,
            microprice_dev_bps,
            book_pressure,
            depth_slope,
        ],
        dtype=float,
    )
