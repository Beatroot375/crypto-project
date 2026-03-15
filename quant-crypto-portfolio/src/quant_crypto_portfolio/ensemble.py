from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np

from .features import snapshot_to_feature_vector

log = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
except ImportError:  # pragma: no cover
    lgb = xgb = cb = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def prepare_ensemble_data(data_dir: Path, horizon_sec: int = 60, symbol: str | None = None):
    if pd is None:
        raise RuntimeError("Install extra: pip install -e '.[train]'")

    symbol_lc = symbol.lower() if symbol else None
    dfs = []

    for day_dir in sorted(data_dir.glob("20*")):
        if not day_dir.is_dir():
            continue
        pattern = f"*_{symbol_lc}_l2.jsonl.gz" if symbol_lc else "*_l2.jsonl.gz"
        for f in sorted(day_dir.glob(pattern)):
            try:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    day_df = pd.read_json(fh, lines=True)
                dfs.append(day_df)
            except Exception as e:
                log.warning(f"Skipping corrupted file {f}: {e}")
                continue

    if not dfs:
        raise RuntimeError("No data found to train on")

    df = pd.concat(dfs, ignore_index=True).sort_values("ts_ns").reset_index(drop=True)
    log.info("Loaded %d snapshots", len(df))

    it = df.iterrows()
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="Feature engineering")

    features = [snapshot_to_feature_vector(row.to_dict(), depth=200) for _, row in it]
    feat_df = pd.DataFrame(
        features,
        columns=[
            "spread_bps",
            "imbalance_l1",
            "imbalance_l3",
            "imbalance_l5",
            "imbalance_l10",
            "microprice_dev_bps",
            "book_pressure",
            "depth_slope",
            "is_buyer_maker",
        ],
    )

    ts = df["ts_ns"].to_numpy(dtype=np.int64)
    mid = df["mid"].to_numpy(dtype=float)
    horizon_ns = int(horizon_sec * 1_000_000_000)
    target_idx = np.searchsorted(ts, ts + horizon_ns, side="left")
    valid = target_idx < len(ts)

    fwd_mid = np.full_like(mid, np.nan, dtype=float)
    fwd_mid[valid] = mid[target_idx[valid]]
    feat_df["fwd_ret"] = (fwd_mid / mid) - 1.0

    out = feat_df.dropna().reset_index(drop=True)
    return out


def train_multi_model_ensemble(feat_df, test_frac: float = 0.2) -> dict[str, float]:
    if pd is None or joblib is None or lgb is None or xgb is None or cb is None:
        raise RuntimeError("Install extra: pip install -e '.[train]'")

    X = feat_df.drop(columns=["fwd_ret"])
    y_ret = feat_df["fwd_ret"]
    y_class = pd.cut(y_ret, bins=[-np.inf, -0.00015, 0.00015, np.inf], labels=[0, 1, 2]).astype(int)

    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y_class.iloc[:split], y_class.iloc[split:]
    y_ret_test = y_ret.iloc[split:]

    models = []
    for Model, name in [
        (lgb.LGBMClassifier, "LightGBM"),
        (xgb.XGBClassifier, "XGBoost"),
        (cb.CatBoostClassifier, "CatBoost"),
    ]:
        m = Model(n_estimators=400, learning_rate=0.05, max_depth=7, random_state=42, verbose=0)
        m.fit(X_train, y_train)
        models.append(m)
        log.info("%s trained", name)

    probs_list = [m.predict_proba(X_test) for m in models]
    ensemble_probs = np.mean(probs_list, axis=0)
    pred_class = ensemble_probs.argmax(axis=1)

    acc = float((pred_class == y_test).mean())
    direction_score = ensemble_probs[:, 2] - ensemble_probs[:, 0]
    pearson = float(np.corrcoef(direction_score, y_ret_test)[0, 1])

    Path("models").mkdir(exist_ok=True)
    joblib.dump(models, "models/crypto_ensemble.pkl")
    return {"pearson_correlation": pearson, "accuracy": acc}
