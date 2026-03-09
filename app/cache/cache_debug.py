"""
Cache debug instrumentation: last 50 calls with level, name, keys, elapsed_ms, size.
Optional dev-only panel shows cache keys, timings, and cache bust reason.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None

CACHE_DEBUG_MAX = 50
DEV_MODE = os.environ.get("DEV_MODE") == "1"


def record_cache_call(
    level: str,
    name: str,
    dataset_version: str,
    filter_hash: str,
    elapsed_ms: float,
    rowcount_or_size: int | str,
    ttl_seconds: int | None = None,
) -> None:
    """
    Append one cache call to st.session_state['cache_debug'] (last CACHE_DEBUG_MAX).
    Computes bust_reason by comparing with previous entry for same level+name.
    """
    if st is None or not DEV_MODE:
        return
    if "cache_debug" not in st.session_state:
        st.session_state["cache_debug"] = []

    prev_list = st.session_state["cache_debug"]
    bust_reason: str | None = None
    if prev_list:
        prev = prev_list[-1]
        if prev.get("level") == level and prev.get("name") == name:
            if prev.get("dataset_version") != dataset_version:
                bust_reason = "dataset_version changed"
            elif prev.get("filter_hash") != filter_hash:
                bust_reason = "filter changed"
            else:
                bust_reason = "cache hit"
        else:
            bust_reason = "name changed"

    hash_val = filter_hash[:16] if filter_hash else ""
    entry: dict[str, Any] = {
        "level": level,
        "name": name,
        "query_name": name,
        "dataset_version": dataset_version,
        "filter_hash": hash_val,
        "filter_state_hash": hash_val,
        "elapsed_ms": round(elapsed_ms, 2),
        "rowcount_or_size": rowcount_or_size,
        "bust_reason": bust_reason,
    }
    if ttl_seconds is not None:
        entry["ttl"] = ttl_seconds
    st.session_state["cache_debug"] = (prev_list + [entry])[-CACHE_DEBUG_MAX:]


def render_cache_debug_panel() -> None:
    """
    Dev-only: render a panel showing last cache keys, timings, and bust reason.
    Call optionally from main.py (e.g. in sidebar when DEV_MODE=1).
    """
    if st is None or not DEV_MODE:
        return
    entries = st.session_state.get("cache_debug") or []
    if not entries:
        st.caption("Cache debug: no entries yet (set DEV_MODE=1).")
        return
    st.subheader("Cache debug (last 50)")
    import pandas as pd
    df = pd.DataFrame(entries)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("level, name, dataset_version, filter_hash, elapsed_ms, rowcount_or_size, bust_reason")
