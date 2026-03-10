from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so "app" package resolves (e.g. when run from app/ or IDE)
_root = Path(__file__).resolve().parent.parent
ROOT = _root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.config.contract import ensure_contract_checked, load_ui_contract
from app.config.tab1_defaults import (
    get_scope_label_from_state,
    get_scope_mode_from_state,
    get_tab1_dimension_keys,
    TAB1_DEFAULT_PERIOD,
)
from app.data_contract import get_data_contract_cached
from app.pages.dynamic_report import render as render_dynamic_report
from app.pages.nlq_chat import render as render_nlq_chat
from app.pages.visualisations import render as render_visualisations
from app.state import ensure_tab1_defaults_initialized, get_filter_state
from app.ui.filters import render_global_filters
from app.ui.theme import configure_plotly_defaults, inject_global_theme_css

PAGE_RENDERERS = {
    "visualisations": render_visualisations,
    "dynamic_report": render_dynamic_report,
    "nlq_chat": render_nlq_chat,
}

TAB_LABELS = {
    "visualisations": "Executive Dashboard",
    "dynamic_report": "Investment Commentary",
    "nlq_chat": "Intelligence Desk",
}

LAST_CONTRACT_FINGERPRINT_KEY = "app_last_data_contract_fingerprint"
DATASET_FINGERPRINT_CHANGED_KEY = "app_dataset_fingerprint_changed"


def _invalidate_cache_if_fingerprint_changed(current_fingerprint: str) -> None:
    """Clear Streamlit data cache when dataset fingerprint changes so stale outputs do not persist."""
    last = st.session_state.get(LAST_CONTRACT_FINGERPRINT_KEY)
    if last is not None and last != current_fingerprint:
        st.cache_data.clear()
        st.session_state[DATASET_FINGERPRINT_CHANGED_KEY] = True
    else:
        st.session_state[DATASET_FINGERPRINT_CHANGED_KEY] = False
    st.session_state[LAST_CONTRACT_FINGERPRINT_KEY] = current_fingerprint


def _render_state_cache_debug(state, data_contract) -> None:
    """Debug expander: filter state, scope mode, period mode, cache-sensitive inputs; dev-only cache reset."""
    tab1_snapshot = {k: st.session_state.get(k, "All") for k in get_tab1_dimension_keys()}
    tab1_snapshot["tab1_period"] = st.session_state.get("tab1_period", TAB1_DEFAULT_PERIOD)
    scope_mode = get_scope_mode_from_state(tab1_snapshot)
    scope_label = get_scope_label_from_state(tab1_snapshot)
    filter_hash = state.filter_state_hash() if hasattr(state, "filter_state_hash") else "-"
    with st.expander("State / Cache (filter, scope, period)", expanded=False):
        st.text(f"Filter state: date_start={state.date_start}, date_end={state.date_end}, period_mode={state.period_mode}")
        st.text(f"Tab 1 period: {tab1_snapshot.get('tab1_period', TAB1_DEFAULT_PERIOD)}")
        st.text(f"Scope mode: {scope_mode}")
        st.text(f"Active scope for KPIs: {scope_label}")
        st.text(f"Cache-sensitive: dataset_version={data_contract.dataset_version}, filter_hash={filter_hash[:16]}...")
        if st.session_state.get("observability_dev_toggle") or __import__("os").environ.get("DEV_MODE") == "1":
            if st.button("Clear data cache (dev)"):
                st.cache_data.clear()
                st.session_state.pop(LAST_CONTRACT_FINGERPRINT_KEY, None)
                st.rerun()


def _parity_debug_enabled() -> bool:
    import os
    return bool(
        st.session_state.get("observability_dev_toggle")
        or st.session_state.get("dev_mode")
        or os.environ.get("DEV_MODE") == "1"
        or os.environ.get("SHOW_PARITY_DEBUG") == "1"
    )


def main() -> None:
    st.set_page_config(
        page_title="Finance Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display:none;}
        [data-testid="collapsedControl"] {display:none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    configure_plotly_defaults()
    ensure_contract_checked()
    ui_contract = load_ui_contract()
    tabs = ui_contract.get("tabs") or ["visualisations", "dynamic_report", "nlq_chat"]
    inject_global_theme_css()

    st.title("AI infin8 | Institutional Asset Management Intelligence")
    render_global_filters()
    ensure_tab1_defaults_initialized()
    state = get_filter_state()

    # Single canonical data contract: resolves path, version, fingerprint; fails loudly if dataset missing
    data_contract = get_data_contract_cached(ROOT)
    _invalidate_cache_if_fingerprint_changed(data_contract.fingerprint)
    st.session_state["dataset_version"] = data_contract.dataset_version
    st.session_state["dataset_fingerprint"] = data_contract.fingerprint
    st.session_state["dataset_path"] = data_contract.resolved_path
    st.session_state["dataset_backend"] = data_contract.backend

    if st.session_state.get(DATASET_FINGERPRINT_CHANGED_KEY):
        st.warning(
            "Dataset fingerprint changed in this session. Data cache was cleared to prevent stale KPI outputs."
        )

    if _parity_debug_enabled():
        tab1_snapshot = {k: st.session_state.get(k, "All") for k in get_tab1_dimension_keys()}
        tab1_snapshot["tab1_period"] = st.session_state.get("tab1_period", TAB1_DEFAULT_PERIOD)
        active_scope = get_scope_label_from_state(tab1_snapshot)
        active_period = tab1_snapshot.get("tab1_period", TAB1_DEFAULT_PERIOD)
        with st.expander("Debug / Data Contract", expanded=False):
            st.text(f"Environment: {data_contract.environment}")
            st.text(f"Dataset path: {data_contract.resolved_path}")
            st.text(f"Dataset version (DATA_VERSION): {data_contract.dataset_version}")
            st.text(f"Dataset fingerprint: {data_contract.fingerprint}")
            st.text(f"Row count: {data_contract.row_count}")
            st.text(f"Date range: {data_contract.min_date or '-'} to {data_contract.max_date or '-'}")
            st.text(f"Sum End AUM: {data_contract.sum_end_aum:,.0f}")
            st.text(f"Sum NNB: {data_contract.sum_nnb:,.0f}")
            st.text(f"Sum NNF: {data_contract.sum_nnf:,.0f}")
            st.text(f"Active scope: {active_scope}")
            st.text(f"Active period mode: {active_period}")
            st.caption(f"Backend: {data_contract.backend}")
        _render_state_cache_debug(state, data_contract)

    tab_widgets = st.tabs([TAB_LABELS.get(t, t.replace("_", " ").title()) for t in tabs])
    for tab_name, tab_widget in zip(tabs, tab_widgets):
        renderer = PAGE_RENDERERS.get(tab_name)
        with tab_widget:
            if renderer is None:
                st.info("This tab is not available.")
                continue
            renderer(state, {"tab_id": tab_name})


if __name__ == "__main__":
    main()
