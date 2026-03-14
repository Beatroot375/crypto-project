from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    SGDClassifier = StandardScaler = None


class OnlineL2Model:
    """
    Online classifier for short-horizon direction using partial_fit.
    Labels:
      1  -> forward return > up_threshold
      0  -> neutral
      -1 -> forward return < down_threshold
    """

    def __init__(
        self,
        horizon_sec: int = 1,
        up_threshold: float = 0.0001,
        down_threshold: float = -0.0001,
    ):
        if SGDClassifier is None or StandardScaler is None:
            raise RuntimeError("Install extra: pip install -e '.[online]'")

        self.horizon_ns = int(horizon_sec * 1_000_000_000)
        self.up_threshold = float(up_threshold)
        self.down_threshold = float(down_threshold)
        self.pending: deque[tuple[int, float, np.ndarray]] = deque()  # (ts_ns, mid, x)

        self.scaler = StandardScaler()
        self.model = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42)
        self.classes = np.array([-1, 0, 1], dtype=int)

        self.ready = False
        self.samples_seen = 0
        self.samples_trained = 0

    def on_snapshot(self, ts_ns: int, mid: float, x: np.ndarray) -> tuple[float, int, int]:
        score = 0.0
        pred_class = 0

        if self.ready:
            xs = self.scaler.transform(x.reshape(1, -1))
            probs = self.model.predict_proba(xs)[0]
            class_to_idx = {int(c): i for i, c in enumerate(self.model.classes_)}
            up_p = float(probs[class_to_idx.get(1, 0)])
            down_p = float(probs[class_to_idx.get(-1, 0)])
            score = up_p - down_p
            pred_class = int(self.model.classes_[int(np.argmax(probs))])

        self.pending.append((int(ts_ns), float(mid), np.asarray(x, dtype=float)))
        trained_now = self._train_matured(now_ts_ns=int(ts_ns), now_mid=float(mid))
        self.samples_seen += 1
        return score, pred_class, trained_now

    def _train_matured(self, now_ts_ns: int, now_mid: float) -> int:
        xs: list[np.ndarray] = []
        ys: list[int] = []

        while self.pending and now_ts_ns - self.pending[0][0] >= self.horizon_ns:
            _, past_mid, past_x = self.pending.popleft()
            fwd_ret = (now_mid - past_mid) / max(1e-12, past_mid)
            if fwd_ret > self.up_threshold:
                y = 1
            elif fwd_ret < self.down_threshold:
                y = -1
            else:
                y = 0
            xs.append(past_x)
            ys.append(y)

        if not xs:
            return 0

        X = np.vstack(xs)
        y = np.asarray(ys, dtype=int)
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)

        if not self.ready:
            self.model.partial_fit(Xs, y, classes=self.classes)
            self.ready = True
        else:
            self.model.partial_fit(Xs, y)

        trained = int(len(y))
        self.samples_trained += trained
        return trained

    def to_state(self) -> dict[str, Any]:
        return {
            "horizon_ns": self.horizon_ns,
            "up_threshold": self.up_threshold,
            "down_threshold": self.down_threshold,
            "pending": list(self.pending),
            "scaler": self.scaler,
            "model": self.model,
            "classes": self.classes,
            "ready": self.ready,
            "samples_seen": self.samples_seen,
            "samples_trained": self.samples_trained,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> OnlineL2Model:
        horizon_sec = max(1, int(state.get("horizon_ns", 1_000_000_000) / 1_000_000_000))
        obj = cls(
            horizon_sec=horizon_sec,
            up_threshold=float(state.get("up_threshold", 0.0001)),
            down_threshold=float(state.get("down_threshold", -0.0001)),
        )
        obj.horizon_ns = int(state.get("horizon_ns", obj.horizon_ns))
        obj.pending = deque(state.get("pending", []))
        obj.scaler = state.get("scaler", obj.scaler)
        obj.model = state.get("model", obj.model)
        obj.classes = state.get("classes", obj.classes)
        obj.ready = bool(state.get("ready", obj.ready))
        obj.samples_seen = int(state.get("samples_seen", 0))
        obj.samples_trained = int(state.get("samples_trained", 0))
        return obj
