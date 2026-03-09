"""
Global filter bar UI: fixed order, session persistence (filters + filter_hash).
Uses app/config/filters.yml via app/filters_contract; no hardcoded modes beyond fallbacks.
Wires validate_and_heal_filters for auto-clear and user warnings; no broken queries.
"""
from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.filters_contract import (
    is_optional_filter_enabled,
    load_filters_contract,
)
from app.filters_validation import validate_and_heal_filters
from app.state import FilterState, get_filter_state, set_filter_state, update_filter_state

# Session keys for persistence (prompt requirement)
FILTERS_KEY = "filters"
FILTER_HASH_KEY = "filter_hash"
PREV_CHANNEL_VIEW_KEY = "prev_channel_view"

# Widget key prefix (stable, avoid clashes)
GF = "gf"

# Fallback enums when contract is missing
FALLBACK_PERIOD_MODES = ["1M", "QoQ", "YTD", "YoY"]
FALLBACK_CHANNEL_VIEWS = ["raw", "standard", "best", "canonical"]
FALLBACK_GEO_DIMS = ["src_country", "product_country"]
FALLBACK_PRODUCT_DIMS = ["ticker", "segment", "sub_segment"]

LABEL_ALL = "All"
VIEW_FOR_COLUMNS = "v_firm_monthly"


def _default_date_end() -> str:
    """Last day of current month (month_end aligned)."""
    today = date.today()
    _, last_day = calendar.monthrange(today.year, today.month)
    return date(today.year, today.month, last_day).isoformat()


def _default_date_start() -> str:
    """Month-end 12 months before default date_end."""
    end_d = date.fromisoformat(_default_date_end())
    year_s, month_s = end_d.year, end_d.month - 12
    if month_s <= 0:
        month_s += 12
        year_s -= 1
    _, last_s = calendar.monthrange(year_s, month_s)
    return date(year_s, month_s, last_s).isoformat()


def _options_from_contract(contract: dict[str, Any], *path: str) -> list[str]:
    """Get enum list from contract path."""
    node = contract
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return []
    return list(node) if isinstance(node, (list, tuple)) else []


def _get_available_values(gw: Any, dimension: str, contract: dict[str, Any]) -> list[str]:
    """Fetch available values for a dimension from gateway if supported; else return [All] only."""
    if gw is None:
        return [LABEL_ALL]
    getter = getattr(gw, "get_available_values", None)
    if callable(getter):
        try:
            values = getter(dimension, contract)
            if isinstance(values, (list, tuple)):
                return [LABEL_ALL] + [str(v) for v in values]
        except Exception:
            pass
    return [LABEL_ALL]


def _sync_filters_and_hash() -> None:
    """Write current FilterState to session_state['filters'] and filter_state_hash to session_state['filter_hash']."""
    state = get_filter_state()
    st.session_state[FILTERS_KEY] = state
    st.session_state[FILTER_HASH_KEY] = state.filter_state_hash()


def _state_canonical_repr(state: FilterState) -> tuple[Any, Any, Any]:
    """Canonical representation for comparing state (dict repr, geo_values, product_values)."""
    d = state.to_dict()
    gv = getattr(state, "geo_values", None)
    pv = getattr(state, "product_values", None)
    gv = tuple(sorted(gv)) if gv else ()
    pv = tuple(sorted(pv)) if pv else ()
    return (tuple(sorted(d.items())), gv, pv)


def render_global_filters(gw: Any, contract: dict[str, Any] | None = None) -> FilterState:
    """
    Render global filter bar in-page (fixed order). On any change, update session_state['filters']
    and session_state['filter_hash']. Runs validate_and_heal_filters before returning; shows warnings/infos.
    Returns the (possibly healed) FilterState.
    """
    contract = contract or {}
    filters_contract = load_filters_contract()

    # Available columns for custodian visibility
    available_columns: set[str] = set()
    if gw is not None and hasattr(gw, "available_columns"):
        try:
            available_columns = gw.available_columns(VIEW_FOR_COLUMNS)
        except Exception:
            pass
    custodian_enabled = is_optional_filter_enabled("custodian_firm", available_columns, filters_contract)

    state = get_filter_state()

    # Custodian not present: clear state and do not render widget
    if not custodian_enabled and state.custodian_firm:
        update_filter_state(custodian_firm=None)
        _sync_filters_and_hash()
        state = get_filter_state()

    # Ensure session keys exist
    if FILTERS_KEY not in st.session_state:
        st.session_state[FILTERS_KEY] = state
    if FILTER_HASH_KEY not in st.session_state:
        st.session_state[FILTER_HASH_KEY] = state.filter_state_hash()

    # (1) Date range (month_end aligned) + period_mode
    period_opts = _options_from_contract(filters_contract, "time", "period_mode", "enum") or FALLBACK_PERIOD_MODES
    with st.container():
        st.caption("Global filters")

        col1, col2 = st.columns(2)
        with col1:
            start_val = st.date_input(
                "Date from",
                value=pd.Timestamp(state.date_start).date(),
                key=f"{GF}_date_start",
            )
        with col2:
            end_val = st.date_input(
                "Date to",
                value=pd.Timestamp(state.date_end).date(),
                key=f"{GF}_date_end",
            )
        if start_val is not None and end_val is not None:
            start_iso = start_val.isoformat()
            end_iso = end_val.isoformat()
            if start_iso != state.date_start or end_iso != state.date_end:
                update_filter_state(date_start=start_iso, date_end=end_iso)
                _sync_filters_and_hash()

        period_idx = period_opts.index(state.period_mode) if state.period_mode in period_opts else 0
        period_sel = st.selectbox(
            "Period mode",
            options=period_opts,
            index=min(period_idx, len(period_opts) - 1),
            key=f"{GF}_period_mode",
        )
        if period_sel is not None and period_sel != state.period_mode:
            update_filter_state(period_mode=period_sel)
            _sync_filters_and_hash()

        # (2) Channel view selector — on change, clear channel-specific slice before validation
        channel_opts = _options_from_contract(filters_contract, "channel_view", "enum") or FALLBACK_CHANNEL_VIEWS
        ch_idx = channel_opts.index(state.channel_view) if state.channel_view in channel_opts else 0
        channel_sel = st.selectbox(
            "Channel view",
            options=channel_opts,
            index=min(ch_idx, len(channel_opts) - 1),
            key=f"{GF}_channel_view",
        )
        if channel_sel is not None and channel_sel != state.channel_view:
            prev_cv = st.session_state.get(PREV_CHANNEL_VIEW_KEY)
            update_filter_state(channel_view=channel_sel, slice_dim=None, slice_value=None)
            st.session_state[PREV_CHANNEL_VIEW_KEY] = channel_sel
            _sync_filters_and_hash()
            if prev_cv is not None:
                st.warning("Channel view changed; channel slice selection cleared.")
            state = get_filter_state()

        # (3) Geo dimension selector + value multiselect (optional)
        geo_opts = _options_from_contract(filters_contract, "geo", "geo_dim", "enum") or FALLBACK_GEO_DIMS
        geo_dim_idx = geo_opts.index(state.geo_dim) if state.geo_dim in geo_opts else 0
        geo_dim_sel = st.selectbox(
            "Geo dimension",
            options=geo_opts,
            index=min(geo_dim_idx, len(geo_opts) - 1),
            key=f"{GF}_geo_dim",
        )
        if geo_dim_sel is not None and geo_dim_sel != state.geo_dim:
            update_filter_state(geo_dim=geo_dim_sel)
            _sync_filters_and_hash()

        geo_values_key = f"{GF}_geo_values"
        geo_value_options = _get_available_values(gw, "geo", filters_contract)
        current_geo_values = st.session_state.get("filter_geo_values") or []
        default_geo = [v for v in current_geo_values if v in geo_value_options] if current_geo_values else []
        if not default_geo and LABEL_ALL not in current_geo_values:
            default_geo = [LABEL_ALL]
        geo_values_sel = st.multiselect(
            "Geo values (optional)",
            options=geo_value_options,
            default=default_geo if default_geo else [LABEL_ALL],
            key=geo_values_key,
        )
        if geo_values_sel is not None:
            st.session_state["filter_geo_values"] = geo_values_sel

        # (4) Product dimension selector + value multiselect (optional)
        product_opts = _options_from_contract(filters_contract, "product", "product_dim", "enum") or FALLBACK_PRODUCT_DIMS
        product_dim_idx = product_opts.index(state.product_dim) if state.product_dim in product_opts else 0
        product_dim_sel = st.selectbox(
            "Product dimension",
            options=product_opts,
            index=min(product_dim_idx, len(product_opts) - 1),
            key=f"{GF}_product_dim",
        )
        if product_dim_sel is not None and product_dim_sel != state.product_dim:
            update_filter_state(product_dim=product_dim_sel)
            _sync_filters_and_hash()

        product_values_key = f"{GF}_product_values"
        product_value_options = _get_available_values(gw, "product", filters_contract)
        current_product_values = st.session_state.get("filter_product_values") or []
        default_product = [v for v in current_product_values if v in product_value_options] if current_product_values else []
        if not default_product and LABEL_ALL not in current_product_values:
            default_product = [LABEL_ALL]
        product_values_sel = st.multiselect(
            "Product values (optional)",
            options=product_value_options,
            default=default_product if default_product else [LABEL_ALL],
            key=product_values_key,
        )
        if product_values_sel is not None:
            st.session_state["filter_product_values"] = product_values_sel

        # (5) Custodian firm selector — only if column present in dataset
        if custodian_enabled:
            custodian_val = st.text_input(
                "Custodian firm",
                value=state.custodian_firm or "",
                placeholder="Optional",
                key=f"{GF}_custodian_firm",
            )
            if custodian_val is not None:
                new_val = custodian_val.strip() or None
                if new_val != state.custodian_firm:
                    update_filter_state(custodian_firm=new_val)
                    _sync_filters_and_hash()

    # Build candidate state with geo/product from session for validation
    state = get_filter_state()
    gv = st.session_state.get("filter_geo_values") or []
    if not gv or gv == [LABEL_ALL]:
        setattr(state, "geo_values", None)
    else:
        setattr(state, "geo_values", [str(x).strip() for x in gv if str(x).strip() and str(x).strip().lower() != "all"])
    pv = st.session_state.get("filter_product_values") or []
    if not pv or pv == [LABEL_ALL]:
        setattr(state, "product_values", None)
    else:
        setattr(state, "product_values", [str(x).strip() for x in pv if str(x).strip() and str(x).strip().lower() != "all"])

    # Validate and heal; persist if changed; show warnings/infos
    contract_filters = load_filters_contract()
    healed_state, warnings, infos = validate_and_heal_filters(state, gw, contract_filters)

    if _state_canonical_repr(healed_state) != _state_canonical_repr(state):
        set_filter_state(healed_state)
        _sync_filters_and_hash()
        # Sync multiselect session from healed state
        hgv = getattr(healed_state, "geo_values", None)
        st.session_state["filter_geo_values"] = sorted(hgv) if hgv else [LABEL_ALL]
        hpv = getattr(healed_state, "product_values", None)
        st.session_state["filter_product_values"] = sorted(hpv) if hpv else [LABEL_ALL]

    with st.container():
        for w in warnings:
            st.warning(w)
        for i in infos:
            st.info(i)

    # Keep session_state['filters'] and ['filter_hash'] in sync
    state = get_filter_state()
    st.session_state[FILTERS_KEY] = state
    st.session_state[FILTER_HASH_KEY] = state.filter_state_hash()

    return get_filter_state()
