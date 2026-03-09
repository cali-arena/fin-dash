"""
Sidebar global filter panel. Driven by app/config/filters.yml; all updates via update_filter_state.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.filters_contract import (
    is_optional_filter_enabled,
    load_filters_contract,
)
from app.state import FilterState, get_filter_state, update_filter_state

# Fallback dimension list when contract does not provide one
DEFAULT_DRILL_DIMENSIONS = [
    "channel",
    "country",
    "ticker",
    "segment",
    "sub_segment",
    "product_ticker",
]


def _drill_options_from_contract(contract: dict[str, Any]) -> list[str]:
    """Dimension list from contract global_filters.drill_path.options or .default; else fallback."""
    gf = (contract or {}).get("global_filters") or {}
    dp = gf.get("drill_path")
    if isinstance(dp, dict):
        opts = dp.get("options") or dp.get("dimensions") or dp.get("default")
        if isinstance(opts, list) and opts:
            return [str(x) for x in opts]
    if isinstance(dp, list) and dp:
        return [str(x) for x in dp]
    return list(DEFAULT_DRILL_DIMENSIONS)


def _options_from_contract(contract: dict[str, Any], *path: str) -> list[str]:
    """Get enum list from contract path, e.g. ('channel_view', 'enum')."""
    node = contract
    for key in path:
        node = (node or {}).get(key)
        if node is None:
            return []
    return list(node) if isinstance(node, (list, tuple)) else []


def render_sidebar_filters(
    state: FilterState,
    contract: dict[str, Any],
    available_columns: set[str] | None = None,
) -> FilterState:
    """
    Render sidebar from filters.yml: date range, period mode, channel view, geo dim, product dim,
    custodian firm (only if enabled via available_columns), drill path, slice, currency, unit.
    All updates via update_filter_state(...). Returns updated FilterState.
    """
    filters_contract = load_filters_contract()
    gf = (contract or {}).get("global_filters") or {}
    state = get_filter_state()
    drill_options = _drill_options_from_contract(contract)

    def _sync_dates() -> None:
        start = st.session_state.get("sidebar_date_start")
        end = st.session_state.get("sidebar_date_end")
        if start is not None and end is not None:
            update_filter_state(date_start=start.isoformat(), date_end=end.isoformat())

    def _sync_period_mode() -> None:
        val = st.session_state.get("sidebar_period_mode")
        if val is not None:
            update_filter_state(period_mode=val)

    def _sync_channel_view() -> None:
        val = st.session_state.get("sidebar_channel_view")
        if val is not None:
            update_filter_state(channel_view=val)

    def _sync_geo_dim() -> None:
        val = st.session_state.get("sidebar_geo_dim")
        if val is not None:
            update_filter_state(geo_dim=val)

    def _sync_product_dim() -> None:
        val = st.session_state.get("sidebar_product_dim")
        if val is not None:
            update_filter_state(product_dim=val)

    def _sync_custodian_firm() -> None:
        val = st.session_state.get("sidebar_custodian_firm")
        if val is None or (isinstance(val, str) and val.strip() == "") or val == "—":
            update_filter_state(custodian_firm=None)
        else:
            update_filter_state(custodian_firm=str(val))

    def _sync_drill_path() -> None:
        val = st.session_state.get("sidebar_drill_path")
        if val is None or not isinstance(val, list):
            return
        if len(val) == 0:
            update_filter_state(drill_path=[drill_options[0]])
            return
        update_filter_state(drill_path=val)

    def _sync_slice() -> None:
        val = st.session_state.get("sidebar_slice")
        if val is None or (isinstance(val, str) and val.strip() == "") or val == "—":
            update_filter_state(slice_value=None)
        else:
            update_filter_state(slice_value=str(val))

    def _sync_currency() -> None:
        val = st.session_state.get("sidebar_currency")
        if val is not None:
            update_filter_state(currency=val)

    def _sync_unit() -> None:
        val = st.session_state.get("sidebar_unit")
        if val is not None:
            update_filter_state(unit=val)

    st.caption("Global filters")

    # 1) Date range (start/end)
    col1, col2 = st.columns(2)
    with col1:
        st.date_input(
            "Date from",
            value=pd.Timestamp(state.date_start).date(),
            key="sidebar_date_start",
            on_change=_sync_dates,
        )
    with col2:
        st.date_input(
            "Date to",
            value=pd.Timestamp(state.date_end).date(),
            key="sidebar_date_end",
            on_change=_sync_dates,
        )

    # 2) Period mode (1M / QoQ / YTD / YoY)
    period_opts = _options_from_contract(filters_contract, "time", "period_mode", "enum")
    if not period_opts:
        period_opts = ["1M", "QoQ", "YTD", "YoY"]
    idx_p = period_opts.index(state.period_mode) if state.period_mode in period_opts else 0
    st.selectbox(
        "Period mode",
        options=period_opts,
        index=min(idx_p, len(period_opts) - 1),
        key="sidebar_period_mode",
        on_change=_sync_period_mode,
    )

    # 3) Channel view (raw / standard / best / canonical)
    channel_opts = _options_from_contract(filters_contract, "channel_view", "enum")
    if not channel_opts:
        channel_opts = ["raw", "standard", "best", "canonical"]
    idx_c = channel_opts.index(state.channel_view) if state.channel_view in channel_opts else 0
    st.selectbox(
        "Channel view",
        options=channel_opts,
        index=min(idx_c, len(channel_opts) - 1),
        key="sidebar_channel_view",
        on_change=_sync_channel_view,
    )

    # 4) Geo dim (src_country / product_country)
    geo_opts = _options_from_contract(filters_contract, "geo", "geo_dim", "enum")
    if not geo_opts:
        geo_opts = ["src_country", "product_country"]
    idx_g = geo_opts.index(state.geo_dim) if state.geo_dim in geo_opts else 0
    st.selectbox(
        "Geo dimension",
        options=geo_opts,
        index=min(idx_g, len(geo_opts) - 1),
        key="sidebar_geo_dim",
        on_change=_sync_geo_dim,
    )

    # 5) Product dim (ticker / segment / sub_segment)
    product_opts = _options_from_contract(filters_contract, "product", "product_dim", "enum")
    if not product_opts:
        product_opts = ["ticker", "segment", "sub_segment"]
    idx_pd = product_opts.index(state.product_dim) if state.product_dim in product_opts else 0
    st.selectbox(
        "Product dimension",
        options=product_opts,
        index=min(idx_pd, len(product_opts) - 1),
        key="sidebar_product_dim",
        on_change=_sync_product_dim,
    )

    # 6) Custodian firm — only if enabled (column exists); else disabled placeholder
    custodian_enabled = (
        is_optional_filter_enabled("custodian_firm", available_columns or set(), filters_contract)
        if available_columns is not None
        else False
    )
    if custodian_enabled:
        custodian_val = state.custodian_firm or "—"
        st.text_input(
            "Custodian firm",
            value=custodian_val if custodian_val != "—" else "",
            placeholder="Optional",
            key="sidebar_custodian_firm",
            on_change=_sync_custodian_firm,
        )
    else:
        st.text_input(
            "Custodian firm",
            value="",
            placeholder="N/A (column not in view)",
            key="sidebar_custodian_firm",
            disabled=True,
        )

    # 7) Drill path (multiselect; non-empty enforced)
    current_drill = [x for x in state.drill_path if x in drill_options]
    if not current_drill:
        current_drill = [drill_options[0]]
    multiselect_val = st.multiselect(
        "Drill path",
        options=drill_options,
        default=current_drill,
        key="sidebar_drill_path",
        on_change=_sync_drill_path,
    )
    if not multiselect_val:
        st.info("Drill path must have at least one dimension; keeping previous selection.")
        try:
            update_filter_state(drill_path=current_drill if current_drill else [drill_options[0]])
        except Exception:
            update_filter_state(drill_path=[drill_options[0]])

    # 8) Slice (selectbox)
    slice_options = ["—"] + list(drill_options)
    slice_display = state.slice_value or state.slice or "—"
    if slice_display not in slice_options:
        slice_display = "—"
    slice_idx = slice_options.index(slice_display)
    st.selectbox(
        "Slice",
        options=slice_options,
        index=slice_idx,
        key="sidebar_slice",
        on_change=_sync_slice,
    )

    # 9) Optional currency / unit (from contract if present)
    currency_cfg = gf.get("currency_toggle")
    if currency_cfg is not None and (not isinstance(currency_cfg, dict) or currency_cfg.get("required") is not True):
        opts = (currency_cfg.get("values", ["native", "usd"]) if isinstance(currency_cfg, dict) else ["native", "usd"])
        cur = state.currency or "native"
        idx = opts.index(cur) if cur in opts else 0
        st.selectbox(
            "Currency",
            options=opts,
            index=min(idx, len(opts) - 1),
            key="sidebar_currency",
            on_change=_sync_currency,
        )

    unit_cfg = gf.get("unit_toggle")
    if unit_cfg is not None and (not isinstance(unit_cfg, dict) or unit_cfg.get("required") is not True):
        opts = (unit_cfg.get("values", ["units", "thousands", "millions"]) if isinstance(unit_cfg, dict) else ["units", "thousands", "millions"])
        u = state.unit or "units"
        idx = opts.index(u) if u in opts else 0
        st.selectbox(
            "Unit",
            options=opts,
            index=min(idx, len(opts) - 1),
            key="sidebar_unit",
            on_change=_sync_unit,
        )

    st.divider()
    st.caption("Active Filter Hash")
    final_state = get_filter_state()
    st.text(final_state.filter_state_hash())

    return get_filter_state()
