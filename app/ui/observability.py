"""
Dev-only observability panel shared across Tabs 1–3.
Shows dataset_version, filter/drill/QuerySpec JSON, cache stats, and recent query log.
Controlled by DEV_MODE env or sidebar toggle; hidden in non-dev mode.
"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import streamlit as st

# Session keys used by data_gateway cached_call
PERF_CACHE_STATS_KEY = "perf_cache_stats"
PERF_QUERY_LOG_KEY = "perf_query_log"
OBS_QUERY_LOG_DISPLAY = 20


def _is_dev_mode() -> bool:
    """True when DEV_MODE=1 or config/secrets or in-page toggle is on."""
    if os.environ.get("DEV_MODE") == "1":
        return True
    if st is not None:
        try:
            if bool(st.secrets.get("DEV_MODE")):
                return True
        except Exception:
            pass
        # In-page toggle: show checkbox so user can enable without env
        try:
            show = st.checkbox(
                "Dev mode (observability)",
                value=st.session_state.get("observability_dev_toggle", False),
                key="obs_sidebar_dev_toggle",
            )
            st.session_state["observability_dev_toggle"] = show
            if show:
                return True
        except Exception:
            pass
        return bool(st.session_state.get("observability_dev_toggle", False))
    return False


def _clear_perf_logs() -> None:
    """Reset perf_query_log and optionally perf_cache_stats for dev."""
    if st is None:
        return
    st.session_state[PERF_QUERY_LOG_KEY] = []
    st.session_state[PERF_CACHE_STATS_KEY] = {"hit": 0, "miss": 0}


def render_observability_panel(
    filters: Any = None,
    drill_state: Any = None,
    queryspec: Any = None,
) -> None:
    """
    Render dev-only observability panel: dataset_version, filter/drill/QuerySpec JSON,
    cache stats (perf_cache_stats), and recent query log table (last 20).
    Hidden when not in dev mode (DEV_MODE=1 or sidebar toggle).
    """
    if not _is_dev_mode():
        return

    with st.expander("Observability (dev)", expanded=False):
        # Dataset version
        dataset_version = st.session_state.get("dataset_version") or "—"
        st.text(f"dataset_version: {dataset_version}")

        # Filter state JSON (canonical, sorted keys)
        if filters is not None:
            try:
                fd = getattr(filters, "to_dict", lambda: filters)()
                filter_json = json.dumps(fd, sort_keys=True, indent=2, default=str)
            except Exception:
                filter_json = str(filters)
            st.text("Filter state (canonical):")
            st.code(filter_json, language="json")
        else:
            st.caption("Filter state: not provided")

        # Drill state JSON (if provided)
        if drill_state is not None:
            try:
                dd = getattr(drill_state, "to_dict", lambda: drill_state)()
                drill_json = json.dumps(dd, sort_keys=True, indent=2, default=str)
            except Exception:
                drill_json = str(drill_state)
            st.text("Drill state:")
            st.code(drill_json, language="json")
        else:
            st.caption("Drill state: not provided")

        # QuerySpec JSON (if provided)
        if queryspec is not None:
            try:
                qs_dict = getattr(queryspec, "model_dump", None)
                if callable(qs_dict):
                    qs_dict = qs_dict(mode="json")
                else:
                    qs_dict = queryspec
                qs_json = json.dumps(qs_dict, sort_keys=True, indent=2, default=str)
            except Exception:
                qs_json = str(queryspec)
            st.text("QuerySpec:")
            st.code(qs_json, language="json")
        else:
            st.caption("QuerySpec: not provided")

        # Cache stats
        st.subheader("Cache stats")
        stats = st.session_state.get(PERF_CACHE_STATS_KEY)
        if isinstance(stats, dict):
            st.json(stats)
        else:
            st.caption("No cache stats (gateway not used yet)")

        # Recent query log table (last 20)
        st.subheader("Recent query log")
        log = st.session_state.get(PERF_QUERY_LOG_KEY)
        if isinstance(log, list) and log:
            rows = log[-OBS_QUERY_LOG_DISPLAY:][::-1]
            table_data = []
            for e in rows:
                table_data.append({
                    "name": e.get("name") or "—",
                    "elapsed_ms": e.get("elapsed_ms") if e.get("elapsed_ms") is not None else "—",
                    "rows": e.get("rows") if e.get("rows") is not None else "—",
                    "hit": e.get("hit", False),
                    "budget_ms": e.get("budget_ms") or "—",
                    "warning": (e.get("warning") or "—")[:80],
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, width="stretch", hide_index=True, height=min(400, 50 * len(df) + 38))
        else:
            st.caption("No query log entries yet")

        # Clear perf logs button
        if st.button("Clear perf logs", key="obs_clear_perf_logs"):
            _clear_perf_logs()
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
