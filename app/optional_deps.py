"""
Optional dependencies: safe imports that do not fail at import time.
Use try_import_*() at runtime; if None, feature is unavailable.
"""
from __future__ import annotations


def try_import_plotly_events():
    """Return plotly_events from streamlit_plotly_events if installed; else None. No ImportError."""
    try:
        from streamlit_plotly_events import plotly_events
        return plotly_events
    except Exception:
        return None
