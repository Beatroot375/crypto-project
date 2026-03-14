# Performance notes

## Hot spots

- JSON encoding + gzip compression dominates CPU for high-frequency snapshots.
- Feature computation (`snapshot_to_feature_vector`) is `O(depth)` for pressure/VWAP-style features.
- Frequent flushes increase syscalls and reduce throughput.

## Practical knobs

- Reduce stored levels: `qcp collect --levels 50` (default is 200).
- Increase snapshot interval: `--snapshot-ms 250` or `500` if you don’t need 10Hz.
- Downsample in analysis: `--stride 10` / `--max-rows ...` for `evaluate`, `drift`, and the dashboard.
- Prefer running `evaluate`/`drift` on completed days (`--to-day ...`) to avoid reading an in-progress gzip member.

## Profiling

CPU profile a loader/eval run:

```bash
python -m cProfile -o /tmp/qcp.prof -m quant_crypto_portfolio.cli evaluate --data-dir data/l2 --symbol BTCUSDT
```

Then inspect with `snakeviz` (optional) or `python -c` tooling.

## Future optimizations

- Use faster JSON encoders (e.g. `orjson`) and write uncompressed JSONL, compressing offline.
- Maintain incremental order book best bid/ask updates without recomputing `max/min` frequently.
- Persist data in columnar format (Parquet) for faster analytics at scale.

