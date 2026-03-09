from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path so "app" package resolves (e.g. when run from app/ or IDE)
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.config.contract import ensure_contract_checked, load_ui_contract
from app.pages.dynamic_report import render as render_dynamic_report
from app.pages.nlq_chat import render as render_nlq_chat
from app.pages.visualisations import render as render_visualisations
from app.state import get_filter_state
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
    "nlq_chat": "Guided Analytics",
}


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
    contract = load_ui_contract()
    tabs = contract.get("tabs") or ["visualisations", "dynamic_report", "nlq_chat"]
    inject_global_theme_css()

    st.title("Investment Intelligence Dashboard")
    render_global_filters()
    state = get_filter_state()

    tab_widgets = st.tabs([TAB_LABELS.get(t, t.replace("_", " ").title()) for t in tabs])
    for tab_name, tab_widget in zip(tabs, tab_widgets):
        renderer = PAGE_RENDERERS.get(tab_name)
        with tab_widget:
            if renderer is None:
                st.error(f"No renderer configured for tab '{tab_name}'")
                continue
            renderer(state, {"tab_id": tab_name})


if __name__ == "__main__":
    main()
