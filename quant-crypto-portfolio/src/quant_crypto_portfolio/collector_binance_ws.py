from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
import math
import numpy as np
from collections import defaultdict
from datetime import datetime

log = logging.getLogger(__name__)

# Separate logger for predictions
pred_log = logging.getLogger('predictions')
pred_log.setLevel(logging.INFO)
pred_handler = logging.FileHandler('predictions.log')
pred_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
pred_log.addHandler(pred_handler)

from .aggtrades import AggTradeWindow
from .features import snapshot_to_feature_vector
from .online import OnlineL2Model
from .orderbook import MultiAssetOrderBook
from .storage import DailyPerSymbolGzipJsonlWriter

log = logging.getLogger(__name__)


async def sync_with_snapshot(client, books: MultiAssetOrderBook, symbol: str, depth: int) -> None:
    snapshot = await client.get_order_book(symbol=symbol.upper(), limit=int(depth))
    books.apply_snapshot(symbol.upper(), snapshot)
    book = books.books[symbol.upper()]
    log.info(
        "Snapshot sync %s | lastUpdateId=%d | bid/ask=%.2f/%.2f",
        symbol.upper(),
        book.last_update_id or -1,
        book.best_bid,
        book.best_ask,
    )


async def collect_binance_depth_ws(
    data_dir: Path,
    symbols: Sequence[str],
    depth: int = 1000,
    snapshot_interval_ms: int = 100,
    levels: int = 200,
    status_interval_sec: int = 30,
    aggtrades: bool = True,
    agg_window_sec: float = 1.0,
    agg_maxlen: int = 200_000,
    max_consecutive_errors: int = 20,
    online_learning: bool = False,
    online_horizon_sec: int = 1,
    score_threshold: float = 0.15,
    checkpoint_sec: int = 60,
    model_dir: Path = Path("models"),
    online_eval: bool = False,
) -> None:
    try:
        from binance import AsyncClient, BinanceSocketManager
        from binance.exceptions import ReadLoopClosed
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install extra: pip install -e '.[binance]'") from e

    if online_learning:
        try:
            import joblib
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install extra: pip install -e '.[online]'") from e
    else:
        joblib = None

    sym_list = [s.upper() for s in symbols]
    if not sym_list:
        raise ValueError("No symbols provided")

    books = MultiAssetOrderBook()
    writer = DailyPerSymbolGzipJsonlWriter(data_dir=data_dir)

    ws_depthupdate_count = 0
    ws_aggtrade_count = 0
    snapshot_count = 0

    eval_total: dict[str, int] = {}
    eval_correct: dict[str, int] = {}
    last_mid: dict[str, float] = {}
    agg_windows = {}
    # Hourly aggregation
    hourly_features_first: defaultdict[str, list[list[float]]] = defaultdict(list)
    hourly_features_last: defaultdict[str, list[list[float]]] = defaultdict(list)
    hourly_predictions: defaultdict[str, list[int]] = defaultdict(list)
    hourly_true_classes: defaultdict[str, list[int]] = defaultdict(list)
    current_hour: str = ""
    if aggtrades:
        window_ns = int(float(agg_window_sec) * 1_000_000_000)
        for sym in sym_list:
            agg_windows[sym] = AggTradeWindow(window_ns=window_ns, maxlen=int(agg_maxlen))

    if online_eval:
        for sym in sym_list:
            eval_total[sym] = 0
            eval_correct[sym] = 0
            last_mid[sym] = float('nan')

    online_models = {}

    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)

    if online_learning:
        model_dir.mkdir(parents=True, exist_ok=True)
        for sym in sym_list:
            path = model_dir / f"{sym.lower()}_online_l2.pkl"
            if path.exists():
                try:
                    state = joblib.load(path)
                    online_models[sym] = OnlineL2Model.from_state(state)
                    log.info(
                        "Loaded online checkpoint %s | trained=%d",
                        path,
                        online_models[sym].samples_trained,
                    )
                except Exception:
                    online_models[sym] = OnlineL2Model(horizon_sec=online_horizon_sec)
            else:
                online_models[sym] = OnlineL2Model(horizon_sec=online_horizon_sec)

    last_checkpoint = 0.0
    last_status = 0.0

    async def ws_loop(bm: BinanceSocketManager, client: AsyncClient) -> None:
        nonlocal ws_aggtrade_count, ws_depthupdate_count
        listen_keys: list[str] = []
        for sym in sym_list:
            listen_keys.append(f"{sym.lower()}@depth@100ms")
            if aggtrades:
                listen_keys.append(f"{sym.lower()}@aggTrade")
        reconnect_attempt = 0

        while True:
            try:
                async with bm.multiplex_socket(listen_keys) as sock:
                    reconnect_attempt = 0
                    log.info("WebSocket connected (%d streams)", len(listen_keys))

                    # After (re)connect, resync snapshots to avoid prolonged gap warnings.
                    for sym in sym_list:
                        try:
                            await sync_with_snapshot(client, books, sym, depth)
                        except Exception as e:
                            log.warning("Snapshot sync failed for %s (%s)", sym, e)
                        if aggtrades:
                            agg_windows[sym].reset()

                    while True:
                        msg = await sock.recv()
                        stream = (msg or {}).get("stream", "")
                        payload = (msg or {}).get("data", {})
                        ev = payload.get("e")

                        if "@" in stream:
                            symbol = stream.split("@", 1)[0].upper()
                        else:
                            symbol = str(payload.get("s", "")).upper()
                        if not symbol:
                            continue

                        if ev == "aggTrade" and aggtrades:
                            if symbol in agg_windows:
                                agg_windows[symbol].on_agg_trade(payload)
                                ws_aggtrade_count += 1
                            continue

                        if ev != "depthUpdate":
                            continue
                        ws_depthupdate_count += 1

                        ok = books.apply_diff(symbol, payload)
                        if not ok:
                            log.warning("Gap detected for %s; resync snapshot", symbol)
                            try:
                                await sync_with_snapshot(client, books, symbol, depth)
                            except Exception as e:
                                log.warning("Snapshot resync failed for %s (%s)", symbol, e)

            except asyncio.CancelledError:
                raise
            except ReadLoopClosed as e:
                reconnect_attempt += 1
                delay = min(30.0, 2.0**min(reconnect_attempt, 5))
                log.warning("WebSocket closed (%s). Reconnecting in %.1fs...", e, delay)
                await asyncio.sleep(delay)
            except Exception as e:
                reconnect_attempt += 1
                if reconnect_attempt >= max_consecutive_errors:
                    raise
                delay = min(30.0, 2.0**min(reconnect_attempt, 5))
                log.exception("WebSocket error (%s). Reconnecting in %.1fs...", e, delay)
                await asyncio.sleep(delay)

    async def snapshot_loop() -> None:
        nonlocal snapshot_count, last_status, current_hour
        interval = max(1, int(snapshot_interval_ms)) / 1000.0
        last_snap = None
        while True:
            await asyncio.sleep(interval)
            snaps = books.get_all_snapshots(levels=levels)
            for snap in snaps:
                sym = str(snap.get("symbol", "")).upper()
                if not sym:
                    continue

                # Initialize current_hour on first snapshot
                if current_hour == "":
                    current_hour = datetime.fromtimestamp(snap["ts_ns"] / 1_000_000_000).strftime("%Y-%m-%d-%H")

                if aggtrades and sym in agg_windows:
                    snap.update(agg_windows[sym].stats(int(snap["ts_ns"])))

                if online_learning and sym in online_models:
                    x = snapshot_to_feature_vector(snap, depth=min(levels, 200))
                    score, pred_class, _trained = online_models[sym].on_snapshot(
                        snap["ts_ns"],
                        snap["mid"],
                        x,
                    )
                    signal = 1 if score > score_threshold else (
                        -1 if score < -score_threshold else 0
                    )
                    snap["online_score"] = float(score)
                    snap["online_pred_class"] = int(pred_class)
                    snap["online_signal"] = int(signal)

                    # Accumulate hourly data
                    hourly_features_first[sym].append(x[:5].tolist())
                    hourly_features_last[sym].append(x[-5:].tolist())
                    hourly_predictions[sym].append(pred_class)
                    if online_eval and sym in eval_total and not math.isnan(last_mid[sym]):
                        true_change = 1 if snap["mid"] > last_mid[sym] else (-1 if snap["mid"] < last_mid[sym] else 0)
                        hourly_true_classes[sym].append(true_change)
                        eval_total[sym] += 1
                        if true_change == pred_class:
                            eval_correct[sym] += 1
                    last_mid[sym] = snap["mid"]

                writer.write(sym, int(snap["ts_ns"]), snap)
                snapshot_count += 1
                last_snap = snap  # Update last_snap

            writer.maybe_flush()
            await checkpoint_models()

            now = asyncio.get_running_loop().time()
            if now - last_status >= max(5.0, float(status_interval_sec)):
                stats = books.stats()
                log.info(
                    (
                        "Status | depth=%d aggtrades=%d snapshots=%d assets=%d "
                        "ups=%.1f sps=%.1f uptime=%.2fh"
                    ),
                    ws_depthupdate_count,
                    ws_aggtrade_count,
                    snapshot_count,
                    stats["asset_count"],
                    stats["updates_per_second"],
                    stats["snapshots_per_second"],
                    stats["uptime_hours"],
                )
                for sym in sym_list:
                    book = books.books.get(sym)
                    if book is None or book.last_update_id is None:
                        log.info("Symbol %s | not synced yet", sym)
                        continue
                    log.info(
                        "Symbol %s | lastUpdateId=%d | bid/ask=%.2f/%.2f | mid=%.2f",
                        sym,
                        book.last_update_id,
                        book.best_bid,
                        book.best_ask,
                        book.mid,
                    )
                    if online_eval and sym in eval_total and eval_total[sym] > 0:
                        accuracy = 100.0 * eval_correct[sym] / eval_total[sym]
                        log.info(
                            "Eval %s | total=%d correct=%d accuracy=%.2f%% | last_true=%.2f last_pred=%d",
                            sym,
                            eval_total[sym],
                            eval_correct[sym],
                            accuracy,
                            last_mid[sym],
                            last_snap["online_pred_class"] if last_snap and "online_pred_class" in last_snap else 0,
                        )

                # Check for hour change and log summary (moved here from per-snapshot to status interval)
                if last_snap:
                    current_ts = datetime.fromtimestamp(last_snap["ts_ns"] / 1_000_000_000)
                    new_hour = current_ts.strftime("%Y-%m-%d-%H")
                    if new_hour != current_hour and current_hour != "":
                        for sym in sym_list:
                            if sym in hourly_features_first and hourly_features_first[sym]:
                                # Compute feature stats
                                first_arrays = np.array(hourly_features_first[sym])
                                last_arrays = np.array(hourly_features_last[sym])
                                first_mean = np.mean(first_arrays, axis=0).tolist()
                                first_std = np.std(first_arrays, axis=0).tolist()
                                first_min = np.min(first_arrays, axis=0).tolist()
                                first_max = np.max(first_arrays, axis=0).tolist()
                                last_mean = np.mean(last_arrays, axis=0).tolist()
                                last_std = np.std(last_arrays, axis=0).tolist()
                                last_min = np.min(last_arrays, axis=0).tolist()
                                last_max = np.max(last_arrays, axis=0).tolist()

                                # Compute prediction stats
                                pred_total = len(hourly_predictions[sym])
                                pred_correct = sum(1 for p, t in zip(hourly_predictions[sym], hourly_true_classes[sym]) if p == t)
                                pred_accuracy = 100.0 * pred_correct / pred_total if pred_total > 0 else 0.0

                                pred_log.info(
                                    f"HOURLY_SUMMARY {sym} hour={current_hour} | "
                                    f"features_first_mean={first_mean} std={first_std} min={first_min} max={first_max} | "
                                    f"features_last_mean={last_mean} std={last_std} min={last_min} max={last_max} | "
                                    f"predictions_total={pred_total} correct={pred_correct} accuracy={pred_accuracy:.2f}%"
                                )

                                # Reset hourly accumulators
                                hourly_features_first[sym].clear()
                                hourly_features_last[sym].clear()
                                hourly_predictions[sym].clear()
                                hourly_true_classes[sym].clear()
                        current_hour = new_hour

                last_status = now

    async def checkpoint_models(force: bool = False) -> None:
        nonlocal last_checkpoint
        if not online_learning:
            return
        now = asyncio.get_running_loop().time()
        if not force and (now - last_checkpoint) < float(checkpoint_sec):
            return
        for sym, model in online_models.items():
            path = model_dir / f"{sym.lower()}_online_l2.pkl"
            joblib.dump(model.to_state(), path)
        last_checkpoint = now

    try:
        ws_task = asyncio.create_task(ws_loop(bm, client), name="ws_loop")
        snap_task = asyncio.create_task(snapshot_loop(), name="snapshot_loop")
        await asyncio.gather(ws_task, snap_task)
    finally:
        try:
            await checkpoint_models(force=True)
        except Exception:
            pass
        writer.maybe_flush(force=True)
        writer.close()
        await client.close_connection()
