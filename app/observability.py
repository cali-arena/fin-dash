"""
Observability panel: query timing, cache hit/miss, dataset_version per tab.
Initializes st.session_state["obs"]; record_query appends per-tab; render_obs_panel shows in expander.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None

OBS_KEY = "obs"
QUERIES_CAP = 100
CURRENT_TAB_KEY = "current_tab"


def _default_obs() -> dict[str, Any]:
    return {
        "dataset_version": "",
        "last_refresh": "",
        "tabs": {
            "visualisations": {"queries": [], "cache": {"hit": 0, "miss": 0}},
            "dynamic_report": {"queries": [], "cache": {"hit": 0, "miss": 0}},
            "nlq_chat": {"queries": [], "cache": {"hit": 0, "miss": 0}},
        },
    }


def init_obs() -> None:
    """Initialize st.session_state['obs'] with dataset_version, tabs (queries + cache hit/miss)."""
    if st is None:
        return
    if OBS_KEY not in st.session_state:
        st.session_state[OBS_KEY] = _default_obs()
    obs = st.session_state[OBS_KEY]
    if "tabs" not in obs:
        obs["tabs"] = _default_obs()["tabs"]
    for tab_id, tab_data in _default_obs()["tabs"].items():
        if tab_id not in obs["tabs"]:
            obs["tabs"][tab_id] = {"queries": [], "cache": {"hit": 0, "miss": 0}}


def set_obs_meta(dataset_version: str = "", last_refresh: str = "") -> None:
    """Set top-level dataset_version and last_refresh (e.g. from header)."""
    if st is None:
        return
    init_obs()
    obs = st.session_state[OBS_KEY]
    if dataset_version:
        obs["dataset_version"] = dataset_version
    if last_refresh:
        obs["last_refresh"] = last_refresh


def record_query(tab: str, query_name: str, timing_ms: float, cache_status: str) -> None:
    """Append query to obs for tab and increment cache hit or miss."""
    if st is None:
        return
    init_obs()
    obs = st.session_state[OBS_KEY]
    tabs = obs["tabs"]
    if tab not in tabs:
        tabs[tab] = {"queries": [], "cache": {"hit": 0, "miss": 0}}
    t = tabs[tab]
    t["queries"].append({
        "query_name": query_name,
        "ms": round(timing_ms, 2),
        "cache_status": cache_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    t["queries"] = t["queries"][-QUERIES_CAP:]
    if cache_status == "hit":
        t["cache"]["hit"] = t["cache"].get("hit", 0) + 1
    else:
        t["cache"]["miss"] = t["cache"].get("miss", 0) + 1


def render_obs_panel(tab: str) -> None:
    """Render Observability expander (collapsed): dataset_version, filter_state_hash, last_refresh, queries table, cache hit/miss."""
    if st is None:
        return
    init_obs()
    obs = st.session_state[OBS_KEY]

    with st.expander("Observability", expanded=False):
        st.text(f"dataset_version: {obs.get('dataset_version') or '—'}")
        try:
            from app.state import get_filter_state
            state = get_filter_state()
            st.text(f"filter_state_hash: {state.filter_state_hash()}")
        except Exception:
            st.text("filter_state_hash: —")
        st.text(f"last_refresh: {obs.get('last_refresh') or '—'}")

        tab_data = obs["tabs"].get(tab, {"queries": [], "cache": {"hit": 0, "miss": 0}})
        queries = tab_data.get("queries", [])[-30:]
        if queries:
            import pandas as pd
            df = pd.DataFrame(queries)
            if not df.empty:
                st.caption("Recent queries")
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No queries recorded for this tab yet.")

        cache = tab_data.get("cache", {"hit": 0, "miss": 0})
        hit = cache.get("hit", 0)
        miss = cache.get("miss", 0)
        st.metric("Cache hits", hit)
        st.metric("Cache misses", miss)
