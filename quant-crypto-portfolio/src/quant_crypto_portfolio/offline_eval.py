from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .strategy import positions_from_signal, strategy_metrics

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    SGDClassifier = StandardScaler = None


CLASS_TO_IDX = {-1: 0, 0: 1, 1: 2}


def make_labels(
    mid: np.ndarray,
    ts_ns: np.ndarray,
    *,
    horizon_sec: int,
    ret_threshold: float,
) -> np.ndarray:
    mid = np.asarray(mid, dtype=float)
    ts = np.asarray(ts_ns, dtype=np.int64)
    horizon_ns = int(horizon_sec) * 1_000_000_000
    j = np.searchsorted(ts, ts + horizon_ns, side="left")
    y = np.zeros(len(mid), dtype=int)
    valid = j < len(mid)
    fwd_ret = np.zeros(len(mid), dtype=float)
    fwd_ret[valid] = (mid[j[valid]] / mid[valid]) - 1.0
    y[fwd_ret > ret_threshold] = 1
    y[fwd_ret < -ret_threshold] = -1
    y[~valid] = 0
    return y


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[CLASS_TO_IDX[int(t)], CLASS_TO_IDX[int(p)]] += 1
    return cm


def f1_macro(cm: np.ndarray) -> float:
    f1s: list[float] = []
    for i in range(3):
        tp = cm[i, i]
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        f1s.append(float(f1))
    return float(np.mean(f1s))


@dataclass(frozen=True)
class OfflineEvalResult:
    classification: dict[str, Any]
    strategy: dict[str, Any]


@dataclass(frozen=True)
class OfflineSignalSeries:
    split_idx: int
    y_true: np.ndarray
    y_pred: np.ndarray
    score: np.ndarray
    signal: np.ndarray
    position: np.ndarray


def offline_eval_with_series(
    X: np.ndarray,
    ts_ns: np.ndarray,
    mid: np.ndarray,
    *,
    horizon_sec: int = 60,
    ret_threshold: float = 0.00015,
    test_frac: float = 0.2,
    score_threshold: float = 0.15,
) -> tuple[OfflineEvalResult, OfflineSignalSeries]:
    if SGDClassifier is None or StandardScaler is None:
        raise RuntimeError("Install extra: pip install -e '.[eval]'")

    y = make_labels(mid, ts_ns, horizon_sec=horizon_sec, ret_threshold=ret_threshold)
    split = int(len(X) * (1.0 - float(test_frac)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    mid_test = mid[split:]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42)
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xte)
    pred = clf.predict(Xte).astype(int)
    cm = confusion_matrix(y_test, pred)
    acc = float((pred == y_test).mean())
    macro_f1 = f1_macro(cm)

    class_to_idx = {int(c): i for i, c in enumerate(clf.classes_)}
    up_p = probs[:, class_to_idx.get(1, 0)]
    down_p = probs[:, class_to_idx.get(-1, 0)]
    score = (up_p - down_p).astype(float, copy=False)

    signal = np.where(
        score > score_threshold,
        1,
        np.where(score < -score_threshold, -1, 0),
    ).astype(int, copy=False)
    pos = positions_from_signal(signal)
    strat = strategy_metrics(mid_test, pos)

    return (
        OfflineEvalResult(
            classification={
                "accuracy": acc,
                "macro_f1": macro_f1,
                "confusion_matrix": cm.tolist(),
                "test_size": int(len(y_test)),
            },
            strategy=strat,
        ),
        OfflineSignalSeries(
            split_idx=int(split),
            y_true=y_test.astype(int, copy=False),
            y_pred=pred.astype(int, copy=False),
            score=score,
            signal=signal,
            position=pos.astype(int, copy=False),
        ),
    )


def offline_evaluate(
    X: np.ndarray,
    ts_ns: np.ndarray,
    mid: np.ndarray,
    *,
    horizon_sec: int = 60,
    ret_threshold: float = 0.00015,
    test_frac: float = 0.2,
    score_threshold: float = 0.15,
) -> OfflineEvalResult:
    res, _series = offline_eval_with_series(
        X,
        ts_ns,
        mid,
        horizon_sec=horizon_sec,
        ret_threshold=ret_threshold,
        test_frac=test_frac,
        score_threshold=score_threshold,
    )
    return res
