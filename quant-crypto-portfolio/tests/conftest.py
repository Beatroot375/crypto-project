from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """
    Ensure `src/` is importable when running pytest without an editable install.

    This keeps `pytest` usable even when invoked from a different Python environment
    (e.g. conda base) as long as the repo is the working directory.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))

