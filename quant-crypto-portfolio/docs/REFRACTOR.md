# Refactor notes (from `btc_ibkr_l2_nautilus_backtest.py`)

This folder is a production-oriented extraction of the original single-file prototype into a maintainable,
scalable GitHub-ready structure.

## What changed

- Split monolithic script into a package under `src/quant_crypto_portfolio/` with clear responsibilities:
  - `orderbook.py`: local order book + multi-asset manager
  - `collector_binance_ws.py`: Binance WebSocket collection loop + snapshot loop
  - `storage.py`: daily-per-symbol gzipped JSONL writer with rotation + periodic flushing
  - `features.py`: feature engineering
  - `online.py`: optional online learner (sklearn)
  - `etl.py`: streaming daily summaries (no pandas required)
  - `ensemble.py`: optional offline ensemble training
  - `cli.py`: CLI entrypoint (`qcp`)
- Fixed correctness issues in the prototype:
  - Depth update sequencing now follows the Binance `U/u` overlap rule (gap detection is reliable).
  - Removed high-frequency busy polling (`10ms` checks inside the message handler). Snapshotting runs in a
    dedicated async loop at `--snapshot-ms`.
  - Data/ETL paths are consistent with multi-symbol storage (`YYYY-MM-DD_<symbol>_l2.jsonl.gz`).
- Simplified operations: ETL is a first-class CLI command intended to be run by external schedulers (cron),
  instead of embedding a third-party scheduler inside the collector process.
- Added basic repo hygiene:
  - `pyproject.toml` for packaging + extras (`binance`, `online`, `train`, `dev`)
  - Unit tests for critical pure logic (order book sequencing, feature vector shape)
  - GitHub Actions CI (ruff + pytest)

## Motivation

- Maintainability: smaller modules, isolated optional dependencies, and a stable CLI.
- Scalability: avoids per-message polling and per-snapshot forced flush; ETL reads data in a streaming fashion.
- Production readiness: repeatable installs (extras), automated checks (CI), and tests around the most fragile logic.
