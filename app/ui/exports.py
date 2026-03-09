"""
Shared export utility for all tables and NLQ results.
Single place for CSV generation, row caps, and deterministic filenames.
"""
from __future__ import annotations

import time
from datetime import timezone
from typing import Callable

import pandas as pd
import streamlit as st

DEFAULT_VIEW_MAX_ROWS = 5000
FULL_EXPORT_MAX_ROWS = 50000

PERF_QUERY_LOG_KEY = "perf_query_log"
PERF_QUERY_LOG_MAX = 50


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return CSV bytes from DataFrame (index=False). Safe for empty df."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return b""
    return df.to_csv(index=False).encode("utf-8")


def _iso_minutes() -> str:
    """Timestamp in ISO minutes, UTC, safe for filenames (no colons)."""
    from datetime import datetime
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")


def _log_export_full(elapsed_ms: float, rows: int, base_filename: str) -> None:
    """Append a synthetic perf_query_log entry for full export."""
    if st is None:
        return
    log = st.session_state.get(PERF_QUERY_LOG_KEY)
    if not isinstance(log, list):
        st.session_state[PERF_QUERY_LOG_KEY] = []
        log = st.session_state[PERF_QUERY_LOG_KEY]
    entry = {
        "name": "export_full",
        "cache_key": base_filename,
        "elapsed_ms": round(elapsed_ms, 2),
        "rows": rows,
        "hit": False,
        "budget_ms": None,
        "warning": None,
    }
    log.append(entry)
    st.session_state[PERF_QUERY_LOG_KEY] = log[-PERF_QUERY_LOG_MAX:]


def render_export_buttons(
    df_view: pd.DataFrame,
    df_full_provider: Callable[[], pd.DataFrame] | None = None,
    base_filename: str = "export",
    *,
    view_label: str = "Download CSV (current view)",
    full_label: str = "Export full CSV (may take longer)",
    allow_full: bool = False,
    max_view_rows: int = DEFAULT_VIEW_MAX_ROWS,
    max_full_rows: int = FULL_EXPORT_MAX_ROWS,
    spinner_text: str = "Preparing export…",
) -> None:
    """
    Render download buttons for current view (and optional full export).
    - Always show current-view download; clamp to max_view_rows and warn if truncated.
    - Show full-export button only if allow_full=True and df_full_provider is provided.
      On full: run provider in spinner, clamp to max_full_rows, log to perf_query_log.
    - Filenames: {base_filename}__rows-{n}__{timestamp}.csv (timestamp ISO minutes UTC).
    """
    if df_view is None:
        df_view = pd.DataFrame()
    view_df = df_view if isinstance(df_view, pd.DataFrame) else pd.DataFrame()
    view_capped = False
    if len(view_df) > max_view_rows:
        view_df = view_df.head(max_view_rows)
        view_capped = True
    n_view = len(view_df)
    ts = _iso_minutes()
    view_filename = f"{base_filename}__rows-{n_view}__{ts}.csv"
    view_bytes = dataframe_to_csv_bytes(view_df)
    if view_capped:
        st.caption(f"View capped to {max_view_rows} rows for download.")
    st.download_button(
        view_label,
        data=view_bytes,
        file_name=view_filename,
        mime="text/csv",
        key=f"export_view_{base_filename}",
    )

    if allow_full and df_full_provider is not None:
        full_key = f"export_full_{base_filename}"
        full_state_key = f"_export_full_{base_filename}"
        if st.button(full_label, key=full_key):
            t0 = time.perf_counter()
            with st.spinner(spinner_text):
                full_df = df_full_provider()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if full_df is None or not isinstance(full_df, pd.DataFrame):
                full_df = pd.DataFrame()
            if len(full_df) > max_full_rows:
                full_df = full_df.head(max_full_rows)
                st.caption(f"Full export capped to {max_full_rows} rows.")
            n_full = len(full_df)
            _log_export_full(elapsed_ms, n_full, base_filename)
            full_filename = f"{base_filename}__rows-{n_full}__{_iso_minutes()}.csv"
            full_bytes = dataframe_to_csv_bytes(full_df)
            st.session_state[full_state_key] = {"bytes": full_bytes, "filename": full_filename}
        if full_state_key in st.session_state:
            payload = st.session_state[full_state_key]
            st.download_button(
                "Download full CSV",
                data=payload["bytes"],
                file_name=payload["filename"],
                mime="text/csv",
                key=f"export_full_dl_{base_filename}",
            )
            del st.session_state[full_state_key]
