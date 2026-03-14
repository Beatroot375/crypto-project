# Dashboard

The dashboard is a local Streamlit app that visualizes the main components of the system:

- Latest snapshot health (mid, spread, online score/signal if present)
- Signal/trade entry & exit markers (from stored `online_signal`)
- Feature time series (from `FEATURE_NAMES`)
- AggTrade rolling window stats (when collected with `--aggtrades`)
- Drift report (PSI vs threshold; retrain recommendation)
- Offline evaluation summary (accuracy, macro-F1, strategy metrics)
- Storage/file health (per-day file existence and size)

## Install

```bash
pip install -e ".[dashboard,eval]"
```

## Run

```bash
qcp-dashboard
```

The app reads from `data/l2` by default; override in the sidebar.

