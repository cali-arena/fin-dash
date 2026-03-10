from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure project root is on path so "app" package resolves (e.g. when run from app/ or IDE)
_root = Path(__file__).resolve().parent.parent
ROOT = _root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

try:
    from app.config.contract import ensure_contract_checked, load_ui_contract
    from app.config.tab1_defaults import (
        get_scope_label_from_state,
        get_scope_mode_from_state,
        get_tab1_dimension_keys,
        TAB1_DEFAULT_PERIOD,
    )
    from app.data_contract import get_data_contract_cached
    from app.data.data_gateway import Q_FIRM_MONTHLY, run_query as gateway_run_query
    from app.pages.dynamic_report import render as render_dynamic_report
    from app.pages.nlq_chat import render as render_nlq_chat
    from app.pages.visualisations import render as render_visualisations
    from app.state import ensure_tab1_defaults_initialized, get_filter_state
    from app.ui.filters import render_global_filters
    from app.ui.theme import configure_plotly_defaults, inject_global_theme_css
except Exception as _import_err:
    st.set_page_config(page_title="Finance Dashboard", layout="wide")
    st.error("App failed during import — check Streamlit Cloud logs.")
    st.exception(_import_err)
    with st.expander("Traceback", expanded=True):
        st.code(traceback.format_exc(), language="text")
    st.stop()

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
        or os.environ.get("DEBUG_DATA_PARITY") == "1"
    )


def _render_data_parity_debug(app_root: Path, data_contract) -> None:
    """Temporary debug block for localhost vs cloud parity: path, exists, row count, dates, NNB/NNF/market sums, scope sample."""
    import pandas as pd
    with st.expander("Data parity (DEBUG_DATA_PARITY / SHOW_PARITY_DEBUG)", expanded=False):
        resolved = data_contract.resolved_path
        path_exists = Path(resolved).exists() if resolved else False
        st.text(f"Resolved path: {resolved}")
        st.text(f"File exists: {path_exists}")
        st.text(f"Backend: {data_contract.backend}")
        st.text(f"Dataset version (cache key): {data_contract.dataset_version}")
        try:
            df = gateway_run_query(Q_FIRM_MONTHLY, {}, root=app_root)
            if df is not None and isinstance(df, pd.DataFrame):
                st.text(f"Firm monthly rows (unfiltered): {len(df)}")
                if "month_end" in df.columns:
                    me = pd.to_datetime(df["month_end"], errors="coerce").dropna()
                    if not me.empty:
                        st.text(f"Month range: {me.min()} to {me.max()}")
                for col, label in (("nnb", "Sum NNB"), ("nnf", "Sum NNF"), ("end_aum", "Sum End AUM")):
                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce").sum()
                        st.text(f"{label}: {s:,.2f}")
                for col in ("market_impact", "market_impact_rate", "market_pnl"):
                    if col in df.columns:
                        s = pd.to_numeric(df[col], errors="coerce").sum()
                        st.text(f"Sum {col}: {s:,.2f}")
                if "ogr" in df.columns:
                    last_ogr = pd.to_numeric(df["ogr"], errors="coerce").dropna()
                    if not last_ogr.empty:
                        st.text(f"OGR (last row): {last_ogr.iloc[-1]:.4f}")
            else:
                st.caption("Firm query returned no DataFrame.")
        except Exception as e:
            st.exception(e)


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

    # Single canonical data contract: resolves path, version, fingerprint
    try:
        data_contract = get_data_contract_cached(ROOT)
    except FileNotFoundError as e:
        st.error("Dataset not found — app cannot load data.")
        st.code(str(e), language="text")
        st.caption(
            "On Streamlit Cloud: ensure data/agg/firm_monthly.parquet (or analytics.duckdb) is in the repo, "
            "or set APP_DATA_BACKEND and add the required file. See docs/DEPLOY.md."
        )
        return
    except Exception as e:
        st.error("App failed to load data contract.")
        st.exception(e)
        return
    _invalidate_cache_if_fingerprint_changed(data_contract.fingerprint)
    st.session_state["dataset_version"] = data_contract.dataset_version
    st.session_state["dataset_fingerprint"] = data_contract.fingerprint
    st.session_state["dataset_path"] = data_contract.resolved_path
    st.session_state["dataset_backend"] = data_contract.backend
    st.session_state["app_root"] = str(ROOT)

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
        _render_data_parity_debug(ROOT, data_contract)
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
    try:
        main()
    except Exception as e:
        st.set_page_config(page_title="Finance Dashboard", layout="wide")
        st.error("Error running app.")
        st.exception(e)
        with st.expander("Traceback", expanded=True):
            st.code(traceback.format_exc(), language="text")
        st.caption(
            "Common causes: missing data (data/agg/firm_monthly.parquet or analytics.duckdb), "
            "or missing env. Check Streamlit Cloud logs for details."
        )
