from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.state import (
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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start = st.date_input("Reporting period from", value=pd.Timestamp(state.date_start).date(), key="ui_date_start")
    with c2:
        end = st.date_input("Reporting period to", value=pd.Timestamp(state.date_end).date(), key="ui_date_end")
    with c3:
        currency = st.selectbox("Base currency", options=["native", "usd"], index=0 if state.currency == "native" else 1)
    with c4:
        unit_opts = ["units", "thousands", "millions"]
        unit = st.selectbox("Display unit", options=unit_opts, index=unit_opts.index(state.unit or "units"))

    changed = (
        start.isoformat() != state.date_start
        or end.isoformat() != state.date_end
        or currency != (state.currency or "native")
        or unit != (state.unit or "units")
    )

    update_filter_state(
        date_start=start.isoformat(),
        date_end=end.isoformat(),
        currency=currency,
        unit=unit,
    )
    if changed:
        st.session_state[USER_CHANGED_FILTERS_KEY] = True
