from __future__ import annotations

import asyncio
import json
import logging
from argparse import ArgumentParser
from pathlib import Path

from .collector_binance_ws import collect_binance_depth_ws
from .data_io import load_dataset
from .drift import feature_drift_report
from .ensemble import prepare_ensemble_data, train_multi_model_ensemble
from .etl import daily_etl_and_report, parse_day
from .features import FEATURE_NAMES
from .logging_utils import setup_logging
from .offline_eval import make_labels, offline_evaluate
from .strategy import positions_from_signal, strategy_metrics

log = logging.getLogger(__name__)


DEFAULT_DATA_DIR = Path("data/l2")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="qcp", description="Quant Crypto Portfolio (L2)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_collect = sub.add_parser("collect", help="Collect Binance L2 via WebSocket")
    p_collect.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_collect.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        help="Symbols to collect (space-separated)",
    )
    p_collect.add_argument("--depth", type=int, default=1000)
    p_collect.add_argument(
        "--levels",
        type=int,
        default=20,
        help="Levels to store per side in each snapshot",
    )
    p_collect.add_argument("--snapshot-ms", type=int, default=100)
    p_collect.add_argument("--status-sec", type=int, default=30, help="Heartbeat log interval")
    p_collect.add_argument(
        "--aggtrades",
        default=True,
        help="Also subscribe to aggTrade streams",
    )
    p_collect.add_argument(
        "--agg-window-sec",
        type=float,
        default=0.1,
        help="Rolling aggTrade window length (seconds)",
    )
    p_collect.add_argument(
        "--agg-maxlen",
        type=int,
        default=200_000,
        help="Max aggTrades kept per symbol",
    )
    p_collect.add_argument("--online", action="store_true")
    p_collect.add_argument("--online-eval", action="store_true", help="Enable rolling evaluation metrics for online model")
    p_collect.add_argument("--online-horizon-sec", type=int, default=1)
    p_collect.add_argument("--score-threshold", type=float, default=0.15)
    p_collect.add_argument("--checkpoint-sec", type=int, default=60)
    p_collect.add_argument("--model-dir", type=Path, default=Path("models"))

    p_etl = sub.add_parser("etl", help="Generate daily summary report")
    p_etl.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_etl.add_argument("--day", default=None, help="UTC day YYYY-MM-DD (defaults to yesterday)")

    p_train = sub.add_parser("train", help="Train ensemble on collected data")
    p_train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_train.add_argument("--symbol", default=None, help="Filter training to a single symbol")
    p_train.add_argument("--horizon-sec", type=int, default=60)
    p_train.add_argument("--test-frac", type=float, default=0.2)

    p_eval = sub.add_parser("evaluate", help="Offline evaluation (accuracy + strategy metrics)")
    p_eval.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_eval.add_argument("--symbol", required=True)
    p_eval.add_argument("--from-day", default=None)
    p_eval.add_argument("--to-day", default=None)
    p_eval.add_argument("--max-rows", type=int, default=200_000)
    p_eval.add_argument("--stride", type=int, default=10)
    p_eval.add_argument("--horizon-sec", type=int, default=60)
    p_eval.add_argument("--ret-threshold", type=float, default=0.00015)
    p_eval.add_argument("--score-threshold", type=float, default=0.15)
    p_eval.add_argument("--test-frac", type=float, default=0.2)
    p_eval.add_argument("--out-json", type=Path, default=None)

    p_drift = sub.add_parser("drift", help="Feature drift test (PSI) + retrain threshold")
    p_drift.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_drift.add_argument("--symbol", required=True)
    p_drift.add_argument("--from-day", default=None)
    p_drift.add_argument("--to-day", default=None)
    p_drift.add_argument("--max-rows", type=int, default=200_000)
    p_drift.add_argument("--stride", type=int, default=10)
    p_drift.add_argument("--baseline-rows", type=int, default=50_000)
    p_drift.add_argument("--recent-rows", type=int, default=50_000)
    p_drift.add_argument("--bins", type=int, default=10)
    p_drift.add_argument("--psi-threshold", type=float, default=0.2)
    p_drift.add_argument("--out-json", type=Path, default=None)

    p_plot = sub.add_parser("plot", help="Plot price + entry/exit markers from stored signals")
    p_plot.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p_plot.add_argument("--symbol", required=True)
    p_plot.add_argument("--from-day", default=None)
    p_plot.add_argument("--to-day", default=None)
    p_plot.add_argument("--max-rows", type=int, default=50_000)
    p_plot.add_argument("--stride", type=int, default=10)
    p_plot.add_argument("--signal-field", default="online_signal")
    p_plot.add_argument("--out", type=Path, required=True)
    p_plot.add_argument("--title", default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    if args.cmd == "collect":
        asyncio.run(
            collect_binance_depth_ws(
                data_dir=args.data_dir,
                symbols=args.symbols,
                depth=args.depth,
                snapshot_interval_ms=args.snapshot_ms,
                levels=args.levels,
                status_interval_sec=args.status_sec,
                aggtrades=args.aggtrades,
                agg_window_sec=args.agg_window_sec,
                agg_maxlen=args.agg_maxlen,
                online_learning=args.online,
                online_horizon_sec=args.online_horizon_sec,
                score_threshold=args.score_threshold,
                checkpoint_sec=args.checkpoint_sec,
                model_dir=args.model_dir,
                online_eval=args.online_eval,
            )
        )
        return

    if args.cmd == "etl":
        daily_etl_and_report(args.data_dir, day=parse_day(args.day))
        return

    if args.cmd == "train":
        feat_df = prepare_ensemble_data(
            args.data_dir,
            horizon_sec=args.horizon_sec,
            symbol=args.symbol,
        )
        metrics = train_multi_model_ensemble(feat_df, test_frac=args.test_frac)
        print(metrics)
        return

    if args.cmd == "evaluate":
        ds = load_dataset(
            args.data_dir,
            args.symbol,
            from_day=args.from_day,
            to_day=args.to_day,
            max_rows=args.max_rows,
            stride=args.stride,
        )
        result = offline_evaluate(
            ds.X,
            ds.ts_ns,
            ds.mid,
            horizon_sec=args.horizon_sec,
            ret_threshold=args.ret_threshold,
            test_frac=args.test_frac,
            score_threshold=args.score_threshold,
        )

        out: dict[str, object] = {
            "symbol": args.symbol.upper(),
            "from_day": args.from_day,
            "to_day": args.to_day,
            "rows": int(len(ds.mid)),
            "offline_model": {
                "classification": result.classification,
                "strategy": result.strategy,
            },
        }

        y = make_labels(
            ds.mid,
            ds.ts_ns,
            horizon_sec=args.horizon_sec,
            ret_threshold=args.ret_threshold,
        )
        split = int(len(ds.X) * (1.0 - float(args.test_frac)))
        y_test = y[split:]
        mid_test = ds.mid[split:]

        if ds.raw_signals is not None:
            pos = positions_from_signal(ds.raw_signals[split:])
            out["online_signal"] = {"strategy": strategy_metrics(mid_test, pos)}

        if ds.raw_pred_class is not None:
            pred = ds.raw_pred_class[split:].astype(int, copy=False)
            out["online_pred_class"] = {
                "accuracy": float((pred == y_test).mean()),
                "test_size": int(len(y_test)),
            }

        payload = json.dumps(out, indent=2)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(payload, encoding="utf-8")
            log.info("Wrote %s", args.out_json)
        else:
            print(payload)
        return

    if args.cmd == "drift":
        ds = load_dataset(
            args.data_dir,
            args.symbol,
            from_day=args.from_day,
            to_day=args.to_day,
            max_rows=args.max_rows,
            stride=args.stride,
            signal_field=None,
            pred_class_field=None,
        )
        b = min(int(args.baseline_rows), len(ds.X))
        r = min(int(args.recent_rows), len(ds.X))
        if b < 1000 or r < 1000 or (b + r) > len(ds.X):
            raise SystemExit(
                "Need enough rows: baseline+recent must fit within loaded rows (>=1000 each)."
            )

        X_baseline = ds.X[:b]
        X_recent = ds.X[-r:]
        report = feature_drift_report(
            X_baseline,
            X_recent,
            FEATURE_NAMES,
            bins=int(args.bins),
            psi_threshold=float(args.psi_threshold),
        )
        out = {
            "symbol": args.symbol.upper(),
            "from_day": args.from_day,
            "to_day": args.to_day,
            "rows": int(len(ds.X)),
            "baseline_rows": b,
            "recent_rows": r,
            "psi_threshold": float(args.psi_threshold),
            "max_psi": report.max_psi,
            "drifted_features": report.drifted_features,
            "psi_by_feature": report.psi_by_feature,
            "retrain_recommended": bool(report.drifted_features),
        }
        payload = json.dumps(out, indent=2)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(payload, encoding="utf-8")
            log.info("Wrote %s", args.out_json)
        else:
            print(payload)
        return

    if args.cmd == "plot":
        from .viz import plot_signals

        ds = load_dataset(
            args.data_dir,
            args.symbol,
            from_day=args.from_day,
            to_day=args.to_day,
            max_rows=args.max_rows,
            stride=args.stride,
            signal_field=str(args.signal_field),
            pred_class_field=None,
        )
        if ds.raw_signals is None:
            raise SystemExit(f"Signal field not found in data: {args.signal_field}")
        pos = positions_from_signal(ds.raw_signals)
        title = args.title or f"{args.symbol.upper()} ({args.signal_field})"
        res = plot_signals(ds.ts_ns, ds.mid, pos, out_path=args.out, title=title)
        log.info("Wrote plot %s (trades=%d)", res.out_path, res.trade_count)
        return
