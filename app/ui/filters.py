from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.data_gateway import DataGateway
from app.state import (
    FilterState,
    get_dataset_date_bounds,
    get_filter_state,
    resolve_best_default_filters,
    set_filter_state,
    update_filter_state,
)

SMART_INIT_DONE_KEY = "ui_smart_init_done"
USER_CHANGED_FILTERS_KEY = "ui_filters_user_changed"


def render_global_filters() -> None:
    """Render global filters and persist changes through FilterState only."""
    if not st.session_state.get(SMART_INIT_DONE_KEY):
        root = Path(__file__).resolve().parents[2]
        best_state = resolve_best_default_filters(root=root)
        set_filter_state(best_state)
        st.session_state[USER_CHANGED_FILTERS_KEY] = False
        st.session_state[SMART_INIT_DONE_KEY] = True

    state = get_filter_state()
    root = Path(__file__).resolve().parents[2]
    min_date, max_date = get_dataset_date_bounds(root)
    start_val = pd.Timestamp(state.date_start).date()
    end_val = pd.Timestamp(state.date_end).date()
    if min_date is not None and max_date is not None:
        start_val = max(min_date, min(start_val, max_date))
        end_val = max(min_date, min(end_val, max_date))
    kwargs_start: dict = {"min_value": min_date, "max_value": max_date} if min_date is not None and max_date is not None else {}
    kwargs_end: dict = {"min_value": min_date, "max_value": max_date} if min_date is not None and max_date is not None else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start = st.date_input("Reporting period from", value=start_val, key="ui_date_start", **kwargs_start)
    with c2:
        end = st.date_input("Reporting period to", value=end_val, key="ui_date_end", **kwargs_end)
    with c3:
        unit_opts = ["units", "thousands", "millions"]
        unit = st.selectbox("Display unit", options=unit_opts, index=unit_opts.index(state.unit or "units"))
    with c4:
        st.caption("Currency: USD")

    changed = (
        start.isoformat() != state.date_start
        or end.isoformat() != state.date_end
        or unit != (state.unit or "units")
    )

    update_filter_state(
        date_start=start.isoformat(),
        date_end=end.isoformat(),
        currency="usd",
        unit=unit,
    )
    if changed:
        st.session_state[USER_CHANGED_FILTERS_KEY] = True

    # ---- Explore Data: drill-down filters at top (Channel, Geography, Product, Segment) ----
    state = get_filter_state()
    gw = DataGateway(root)
    channel_opts = ["All"] + (gw.list_channel_values(state, limit=200) or [])
    geo_opts = ["All"] + (gw.list_geo_values(state, limit=200) or [])
    product_opts = ["All"] + (gw.list_product_values(state, limit=200) or [])
    state_segment = FilterState.from_dict({**state.to_dict(), "product_dim": "segment"})
    segment_opts = ["All"] + (gw.list_product_values(state_segment, limit=200) or [])

    # Current slice: only one dimension is active; show its value in the right box
    slice_dim = (state.slice_dim or "").strip().lower() if state.slice_dim else ""
    slice_val = (state.slice_value or getattr(state, "slice", None) or "").strip()
    if slice_val and ":" in slice_val:
        slice_val = slice_val.split(":", 1)[-1].strip()
    channel_val = slice_val if slice_dim == "channel" else "All"
    geo_val = slice_val if slice_dim == "geo" else "All"
    product_val = slice_val if slice_dim == "product" and state.product_dim == "ticker" else "All"
    segment_val = slice_val if slice_dim == "product" and state.product_dim == "segment" else "All"
    # Clamp to options if current value not in list (e.g. after filter change)
    channel_val = channel_val if channel_val in channel_opts else "All"
    geo_val = geo_val if geo_val in geo_opts else "All"
    product_val = product_val if product_val in product_opts else "All"
    segment_val = segment_val if segment_val in segment_opts else "All"

    st.subheader("Explore Data")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        channel_sel = st.selectbox("Channel", options=channel_opts, index=channel_opts.index(channel_val), key="ui_explore_channel")
    with c2:
        geo_sel = st.selectbox("Geography", options=geo_opts, index=geo_opts.index(geo_val), key="ui_explore_geo")
    with c3:
        product_sel = st.selectbox("Product", options=product_opts, index=product_opts.index(product_val), key="ui_explore_product")
    with c4:
        segment_sel = st.selectbox("Segment", options=segment_opts, index=segment_opts.index(segment_val), key="ui_explore_segment")

    # Map selection to single slice (priority: Channel > Geography > Product > Segment)
    new_slice_dim: str | None = None
    new_slice_value: str | None = None
    new_product_dim: str | None = None
    if channel_sel and channel_sel != "All":
        new_slice_dim, new_slice_value = "channel", channel_sel
    elif geo_sel and geo_sel != "All":
        new_slice_dim, new_slice_value = "geo", geo_sel
    elif product_sel and product_sel != "All":
        new_slice_dim, new_slice_value = "product", product_sel
        new_product_dim = "ticker"
    elif segment_sel and segment_sel != "All":
        new_slice_dim, new_slice_value = "product", segment_sel
        new_product_dim = "segment"

    explore_changed = (
        (new_slice_dim or "") != (state.slice_dim or "")
        or (new_slice_value or "") != (state.slice_value or state.slice or "")
        or (new_product_dim is not None and new_product_dim != state.product_dim)
    )
    if explore_changed:
        update_filter_state(
            slice_dim=new_slice_dim,
            slice_value=new_slice_value,
            product_dim=new_product_dim or state.product_dim,
        )
        st.session_state[USER_CHANGED_FILTERS_KEY] = True
