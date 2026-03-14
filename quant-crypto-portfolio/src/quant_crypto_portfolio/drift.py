from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def population_stability_index(
    baseline: np.ndarray,
    recent: np.ndarray,
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    baseline = np.asarray(baseline, dtype=float)
    recent = np.asarray(recent, dtype=float)
    if baseline.size == 0 or recent.size == 0:
        return float("nan")

    q = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.quantile(baseline, q)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    b_hist, _ = np.histogram(baseline, bins=edges)
    r_hist, _ = np.histogram(recent, bins=edges)
    b = b_hist / max(1, baseline.size)
    r = r_hist / max(1, recent.size)

    b = np.clip(b, eps, 1.0)
    r = np.clip(r, eps, 1.0)
    psi = float(np.sum((r - b) * np.log(r / b)))
    if math.isfinite(psi):
        return psi
    return float("nan")


@dataclass(frozen=True)
class DriftReport:
    psi_by_feature: dict[str, float]
    max_psi: float
    drifted_features: list[str]


def feature_drift_report(
    X_baseline: np.ndarray,
    X_recent: np.ndarray,
    feature_names: list[str],
    *,
    bins: int = 10,
    psi_threshold: float = 0.2,
) -> DriftReport:
    if X_baseline.shape[1] != X_recent.shape[1]:
        raise ValueError("baseline and recent must have same feature count")
    if X_baseline.shape[1] != len(feature_names):
        raise ValueError("feature_names length mismatch")

    psi_by_feature: dict[str, float] = {}
    drifted: list[str] = []
    max_psi = 0.0
    for i, name in enumerate(feature_names):
        psi = population_stability_index(X_baseline[:, i], X_recent[:, i], bins=bins)
        psi_by_feature[name] = float(psi)
        if math.isfinite(psi):
            max_psi = max(max_psi, float(psi))
            if psi >= psi_threshold:
                drifted.append(name)

    return DriftReport(
        psi_by_feature=psi_by_feature,
        max_psi=float(max_psi),
        drifted_features=drifted,
    )
