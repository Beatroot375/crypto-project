from __future__ import annotations

import sys


def main() -> None:
    """
    Launch the Streamlit dashboard.

    Requires: `pip install -e '.[dashboard]'`
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install extra: pip install -e '.[dashboard]'") from e

    from . import dashboard_app

    # Streamlit expects a file path to execute.
    app_path = dashboard_app.__file__
    if not app_path:
        raise SystemExit("Could not locate dashboard app file")

    sys.argv = ["streamlit", "run", app_path]
    raise SystemExit(stcli.main())

