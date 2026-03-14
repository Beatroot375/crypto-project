from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from quant_crypto_portfolio.data_io import load_dataset
from quant_crypto_portfolio.drift import feature_drift_report
from quant_crypto_portfolio.features import FEATURE_NAMES
from quant_crypto_portfolio.offline_eval import offline_eval_with_series
from quant_crypto_portfolio.strategy import (
    positions_from_signal,
    strategy_metrics,
    trades_from_positions,
)


def build_dashboard() -> None:
    import pandas as pd
    import streamlit as st
    
    def is_na(v: object) -> bool:
        return bool(pd.isna(v))

    def show_strategy_metrics(title: str, m: dict) -> None:
        st.markdown(f"**{title}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total return", f"{(m.get('total_return') or 0.0):.2%}")
        c2.metric("Trades", f"{int(m.get('trade_count') or 0):,}")
        wr = m.get("win_rate")
        c3.metric("Win rate", "—" if wr is None else f"{float(wr):.1%}")
        atr = m.get("avg_trade_return")
        c4.metric("Avg trade", "—" if atr is None else f"{float(atr):.2%}")
        c5.metric("Max drawdown", f"{float(m.get('max_drawdown') or 0.0):.2%}")

    def show_classification_metrics(title: str, m: dict) -> None:
        st.markdown(f"**{title}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{float(m.get('accuracy') or 0.0):.2%}")
        c2.metric("Macro F1", f"{float(m.get('macro_f1') or 0.0):.3f}")
        c3.metric("Test size", f"{int(m.get('test_size') or 0):,}")

    st.set_page_config(page_title="QCP Dashboard", layout="wide")
    st.title("Quant Crypto Portfolio — Dashboard")

    with st.sidebar:
        st.header("Data")
        data_dir = st.text_input("Data directory", value=str(Path("data/l2")))
        symbol = st.text_input("Symbol", value="BTCUSDT").upper().strip() or "BTCUSDT"
        from_day = st.text_input("From day (YYYY-MM-DD)", value="")
        to_day = st.text_input("To day (YYYY-MM-DD)", value="")
        max_rows = st.number_input(
            "Max rows",
            min_value=1000,
            max_value=5_000_000,
            value=200_000,
            step=10_000,
        )
        stride = st.number_input(
            "Stride (downsample)",
            min_value=1,
            max_value=10_000,
            value=10,
            step=1,
        )

        st.divider()
        st.header("Offline Eval")
        horizon_sec = st.number_input(
            "Horizon (sec)",
            min_value=1,
            max_value=3600,
            value=60,
            step=1,
        )
        ret_threshold = st.number_input(
            "Return threshold",
            min_value=0.0,
            max_value=0.01,
            value=0.00015,
            step=0.00005,
            format="%.6f",
        )
        score_threshold = st.number_input(
            "Score threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            format="%.3f",
        )
        test_frac = st.number_input(
            "Test fraction",
            min_value=0.05,
            max_value=0.9,
            value=0.2,
            step=0.05,
            format="%.2f",
        )

        st.divider()
        st.header("Drift")
        baseline_rows = st.number_input(
            "Baseline rows",
            min_value=1000,
            max_value=1_000_000,
            value=50_000,
            step=10_000,
        )
        recent_rows = st.number_input(
            "Recent rows",
            min_value=1000,
            max_value=1_000_000,
            value=50_000,
            step=10_000,
        )
        psi_threshold = st.number_input(
            "PSI threshold",
            min_value=0.0,
            max_value=2.0,
            value=0.2,
            step=0.05,
            format="%.2f",
        )

    try:
        ds = load_dataset(
            Path(data_dir),
            symbol,
            from_day=from_day or None,
            to_day=to_day or None,
            max_rows=int(max_rows),
            stride=int(stride),
            extra_fields=[
                "best_bid",
                "best_ask",
                "online_score",
                "online_signal",
                "online_pred_class",
                "agg_trade_count",
                "agg_imbalance",
                "agg_vwap",
                "agg_buy_qty",
                "agg_sell_qty",
            ],
        )
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    ts = pd.to_datetime(ds.ts_ns / 1_000_000_000, unit="s", utc=True)
    df = pd.DataFrame({"ts": ts, "mid": ds.mid})
    for i, name in enumerate(FEATURE_NAMES):
        df[name] = ds.X[:, i]

    extra = ds.extra or {}
    for k, arr in extra.items():
        if len(arr) == len(df):
            df[k] = arr

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Last mid", f"{df['mid'].iloc[-1]:.2f}" if len(df) else "—")
    if "best_bid" in df.columns and "best_ask" in df.columns and len(df):
        spr_bps = (
            (df["best_ask"].iloc[-1] - df["best_bid"].iloc[-1]) / max(1e-12, df["mid"].iloc[-1])
        ) * 1e4
        col3.metric("Last spread (bps)", f"{spr_bps:.2f}")
    if "agg_trade_count" in df.columns and len(df):
        v = df["agg_trade_count"].iloc[-1]
        if is_na(v):
            col4.metric("AggTrades (window)", "—")
        else:
            col4.metric("AggTrades (window)", f"{int(v):,}")

    tab_overview, tab_signals, tab_features, tab_drift, tab_eval, tab_files, tab_export = st.tabs(
        ["Overview", "Signals", "Features", "Drift", "Offline Eval", "Files", "Export"]
    )

    with tab_overview:
        st.subheader("Latest snapshot")
        rows = [
            ("symbol", symbol),
            ("ts_utc", df["ts"].iloc[-1].isoformat() if len(df) else None),
            ("mid", float(df["mid"].iloc[-1]) if len(df) else None),
        ]
        for k in [
            "online_score",
            "online_signal",
            "online_pred_class",
            "agg_trade_count",
            "agg_imbalance",
            "agg_vwap",
            "agg_buy_qty",
            "agg_sell_qty",
        ]:
            if k in df.columns and len(df):
                v = df[k].iloc[-1]
                rows.append((k, None if is_na(v) else float(v)))
        kv = pd.DataFrame(rows, columns=["field", "value"])

        def fmt(v: object) -> str:
            if v is None or is_na(v):
                return "—"
            if isinstance(v, str):
                return v
            try:
                return f"{float(v):.6g}"
            except Exception:
                return str(v)

        kv["value"] = kv["value"].map(fmt)
        st.dataframe(kv, hide_index=True, width="content")

    with tab_signals:
        st.subheader("Signals and entry/exit points")
        source = st.radio(
            "Signal source",
            ["Stored online signal (collector)", "Offline SGDClassifier signal (test set)"],
            horizontal=True,
        )
        tail = st.number_input(
            "Plot last N points (0 = all)",
            min_value=0,
            max_value=int(len(df)),
            value=min(10_000, int(len(df))),
            step=1000,
        )

        if source.startswith("Stored"):
            if ds.raw_signals is None:
                st.info("No `online_signal` field found. Collect with `--online` to store signals.")
            else:
                sig = ds.raw_signals.astype(int, copy=False)
                pos = positions_from_signal(sig)
                m = strategy_metrics(ds.mid, pos)

                df_plot = df if tail == 0 else df.iloc[-int(tail) :]
                idx0 = len(df) - len(df_plot)
                pos_plot = pos[idx0:]
                trades = trades_from_positions(df_plot["mid"].to_numpy(), pos_plot)

                points = []
                for t in trades:
                    points.append({"ts": df_plot["ts"].iloc[t.entry_idx], "px": t.entry_px})
                    points.append({"ts": df_plot["ts"].iloc[t.exit_idx], "px": t.exit_px})
                pts = pd.DataFrame(points) if points else pd.DataFrame(columns=["ts", "px"])

                st.line_chart(df_plot.set_index("ts")[["mid"]])
                if len(pts):
                    st.scatter_chart(pts.set_index("ts")[["px"]])
                show_strategy_metrics("Strategy metrics (stored online signal)", m)
        else:
            st.caption("Generated from an offline SGDClassifier trained on the pre-test split.")
            try:
                res, series = offline_eval_with_series(
                    ds.X,
                    ds.ts_ns,
                    ds.mid,
                    horizon_sec=int(horizon_sec),
                    ret_threshold=float(ret_threshold),
                    test_frac=float(test_frac),
                    score_threshold=float(score_threshold),
                )
            except Exception as e:
                st.error(f"Offline evaluation failed: {e}")
                series = None
            if series is not None:
                split = series.split_idx
                df_test = df.iloc[split:].copy()
                df_test["sgd_score"] = series.score
                df_test["sgd_signal"] = series.signal
                df_test["sgd_pos"] = series.position

                df_plot = df_test if tail == 0 else df_test.iloc[-int(tail) :]
                pos_plot = df_plot["sgd_pos"].to_numpy(dtype=int)
                trades = trades_from_positions(df_plot["mid"].to_numpy(), pos_plot)

                points = []
                for t in trades:
                    points.append({"ts": df_plot["ts"].iloc[t.entry_idx], "px": t.entry_px})
                    points.append({"ts": df_plot["ts"].iloc[t.exit_idx], "px": t.exit_px})
                pts = pd.DataFrame(points) if points else pd.DataFrame(columns=["ts", "px"])

                st.line_chart(df_plot.set_index("ts")[["mid", "sgd_score"]])
                if len(pts):
                    st.scatter_chart(pts.set_index("ts")[["px"]])
                show_strategy_metrics("Strategy metrics (offline SGD signal)", res.strategy)

    with tab_features:
        st.subheader("Feature time series")
        selected = st.multiselect(
            "Features",
            FEATURE_NAMES,
            default=["spread_bps", "book_pressure"],
        )
        if selected:
            st.line_chart(df.set_index("ts")[selected])

        cols = []
        for k in ["agg_imbalance", "agg_vwap", "agg_trade_count"]:
            if k in df.columns:
                cols.append(k)
        if cols:
            st.subheader("AggTrade window stats")
            st.line_chart(df.set_index("ts")[cols])

        cols = []
        for k in ["online_score", "online_signal"]:
            if k in df.columns:
                cols.append(k)
        if cols:
            st.subheader("Online outputs (if present)")
            st.line_chart(df.set_index("ts")[cols])

    with tab_drift:
        st.subheader("Feature drift (PSI)")
        b = min(int(baseline_rows), len(ds.X))
        r = min(int(recent_rows), len(ds.X))
        if b < 1000 or r < 1000 or (b + r) > len(ds.X):
            st.info("Need enough rows: baseline+recent must fit within loaded rows (>=1000 each).")
        else:
            rep = feature_drift_report(
                ds.X[:b],
                ds.X[-r:],
                FEATURE_NAMES,
                psi_threshold=float(psi_threshold),
            )
            st.write(f"Max PSI: `{rep.max_psi:.4f}` | Drifted: `{len(rep.drifted_features)}`")
            psi_df = pd.DataFrame(
                {
                    "feature": list(rep.psi_by_feature.keys()),
                    "psi": list(rep.psi_by_feature.values()),
                }
            ).set_index("feature")
            st.bar_chart(psi_df)
            if rep.drifted_features:
                st.warning("Retrain recommended (PSI threshold exceeded).")
            else:
                st.success("No drift detected above threshold.")

    with tab_eval:
        st.subheader("Offline evaluation (walk-forward split)")
        st.caption("Requires `pip install -e '.[eval]'`")
        try:
            res, series = offline_eval_with_series(
                ds.X,
                ds.ts_ns,
                ds.mid,
                horizon_sec=int(horizon_sec),
                ret_threshold=float(ret_threshold),
                test_frac=float(test_frac),
                score_threshold=float(score_threshold),
            )
            show_classification_metrics("Classification metrics", res.classification)
            show_strategy_metrics("Strategy metrics (SGD signal on test set)", res.strategy)

            cm = res.classification.get("confusion_matrix")
            if isinstance(cm, list) and len(cm) == 3:
                st.markdown("**Confusion matrix (rows=true, cols=pred)**")
                cm_df = pd.DataFrame(cm, index=["-1", "0", "1"], columns=["-1", "0", "1"])
                st.dataframe(cm_df, width="content")

            if series is not None:
                st.markdown("**Signal distribution (test set)**")
                counts = pd.Series(series.signal).value_counts().sort_index()
                dist = pd.DataFrame({"signal": counts.index.astype(int), "count": counts.values})
                st.dataframe(dist, hide_index=True, width="content")
        except Exception as e:
            st.error(f"Offline evaluation failed: {e}")

    with tab_files:
        st.subheader("Storage / file health")
        base = Path(data_dir)
        day_dirs = [p for p in sorted(base.glob("20*")) if p.is_dir()]
        if not day_dirs:
            st.info(f"No day directories found under {base}")
        else:
            rows = []
            for d in day_dirs[-30:]:
                p = d / f"{d.name}_{symbol.lower()}_l2.jsonl.gz"
                rows.append(
                    {
                        "day": d.name,
                        "file_exists": p.exists(),
                        "size_mb": (p.stat().st_size / 1_048_576) if p.exists() else None,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")

    with tab_export:
        st.subheader("Export")
        payload = {
            "symbol": symbol,
            "generated_utc": datetime.now(UTC).isoformat(),
            "rows": int(len(df)),
            "from_day": from_day or None,
            "to_day": to_day or None,
        }
        st.download_button(
            "Download metadata JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"{symbol.lower()}_dashboard_meta.json",
            mime="application/json",
        )


if __name__ == "__main__":
    build_dashboard()
