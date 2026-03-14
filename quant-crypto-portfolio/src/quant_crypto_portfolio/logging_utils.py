from __future__ import annotations

import logging
import os
import time
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None, *, force: bool = True) -> None:
    """
    Configure root logging with UTC timestamps.

    Environment overrides:
      - QCP_LOG_LEVEL
    """
    level = os.environ.get("QCP_LOG_LEVEL", level)
    root = logging.getLogger()
    if force:
        root.handlers.clear()
    elif root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    fmt.converter = time.gmtime
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)
