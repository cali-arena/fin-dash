from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.state import (
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
    top_controls = st.container()
    with top_controls:
        st.markdown("<div class='global-controls-grid-anchor'></div>", unsafe_allow_html=True)
        row_1_col_1, row_1_col_2 = st.columns(2, gap="medium")
        with row_1_col_1:
            start = st.date_input(
                "Reporting period from",
                value=start_val,
                key="ui_date_start",
                format="YYYY/MM/DD",
                help="Start date for the reporting period (YYYY/MM/DD).",
                **kwargs_start,
            )
        with row_1_col_2:
            end = st.date_input(
                "Reporting period to",
                value=end_val,
                key="ui_date_end",
                format="YYYY/MM/DD",
                help="End date for the reporting period (YYYY/MM/DD).",
                **kwargs_end,
            )

        row_2_col_1, row_2_col_2 = st.columns(2, gap="medium")
        with row_2_col_1:
            unit_opts = ["units", "thousands", "millions"]
            unit = st.selectbox("Display unit", options=unit_opts, index=unit_opts.index(state.unit or "units"))
        with row_2_col_2:
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
