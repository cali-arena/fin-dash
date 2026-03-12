from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PALETTE = {
    "bg": "#0b132b",
    "surface": "#111d3a",
    "surface_alt": "#17284d",
    "primary": "#8fb4ff",
    "secondary": "#4c7edb",
    "accent": "#4c7edb",
    "text": "#f8fbff",
    "text_muted": "#b7c5e3",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "market": "#f59e0b",
    "neutral": "#7f93bc",
    "grid": "#2a3d67",
}


def configure_plotly_defaults() -> None:
    """Set global Plotly defaults for a dark institutional financial theme."""
    pio.templates.default = "plotly_dark"
    pio.templates["plotly_dark"].layout.colorway = [
        PALETTE["primary"],
        PALETTE["market"],
        PALETTE["positive"],
        PALETTE["negative"],
        PALETTE["neutral"],
    ]


def inject_global_theme_css() -> None:
    """Inject global CSS for dark financial appearance across all Streamlit primitives."""
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #0b132b !important;
            color: #f8fbff !important;
            font-family: "Segoe UI", "Inter", "SF Pro Display", sans-serif !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: #0b132b !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
        }
        .stMarkdown, .stCaption, .stText, label, p, li, span {
            color: #f8fbff !important;
        }
        h1, h2, h3, h4, h5 {
            color: #f8fbff !important;
            letter-spacing: 0.01em;
        }
        h2 {
            font-size: 1.55rem !important;
            margin-bottom: 0.35rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
            margin-top: 1.0rem !important;
            margin-bottom: 0.35rem !important;
        }
        .stCaption {
            color: #b7c5e3 !important;
        }
        .stApp hr {
            border-color: #2a3d67 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #2a3d67 !important;
            border-radius: 12px !important;
            padding: 0.55rem 0.7rem !important;
            background: #111d3a !important;
            min-height: 112px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25) !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #b7c5e3 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fbff !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-weight: 700 !important;
        }
        .stSpinner > div {
            border-top-color: #f59e0b !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: #17284d !important;
            border-color: #2a3d67 !important;
            color: #f8fbff !important;
        }
        /* Centralized dark-theme form layer: text_area, text_input, chat/prompt fields, select/search */
        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label {
            color: #b7c5e3 !important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: #f8fbff !important;
            -webkit-text-fill-color: #f8fbff !important;
            caret-color: #8fb4ff !important;
            background-color: #17284d !important;
            border: 1px solid #2a3d67 !important;
            border-radius: 8px !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="textarea"] textarea::placeholder {
            color: #8b9dc3 !important;
            opacity: 1 !important;
        }
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-baseweb="input"] input:focus,
        [data-baseweb="textarea"] textarea:focus {
            color: #f8fbff !important;
            -webkit-text-fill-color: #f8fbff !important;
            background-color: #17284d !important;
            border-color: #4c7edb !important;
            box-shadow: 0 0 0 1px #4c7edb !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] input:disabled,
        [data-testid="stTextArea"] textarea:disabled,
        [data-baseweb="input"] input:disabled,
        [data-baseweb="textarea"] textarea:disabled {
            color: #c7d4ee !important;
            -webkit-text-fill-color: #c7d4ee !important;
            opacity: 1 !important;
            background-color: #1e3a5f !important;
        }
        [data-testid="stTextInput"] input::selection,
        [data-testid="stTextArea"] textarea::selection,
        [data-baseweb="input"] input::selection,
        [data-baseweb="textarea"] textarea::selection {
            background: #4c7edb !important;
            color: #f8fbff !important;
        }
        /* Prevent browser autofill from making typed text dark on dark background */
        [data-testid="stTextInput"] input:-webkit-autofill,
        [data-testid="stTextInput"] input:-webkit-autofill:hover,
        [data-testid="stTextInput"] input:-webkit-autofill:focus,
        [data-testid="stTextArea"] textarea:-webkit-autofill,
        [data-testid="stTextArea"] textarea:-webkit-autofill:hover,
        [data-testid="stTextArea"] textarea:-webkit-autofill:focus,
        [data-baseweb="input"] input:-webkit-autofill,
        [data-baseweb="input"] input:-webkit-autofill:hover,
        [data-baseweb="input"] input:-webkit-autofill:focus,
        [data-baseweb="textarea"] textarea:-webkit-autofill,
        [data-baseweb="textarea"] textarea:-webkit-autofill:hover,
        [data-baseweb="textarea"] textarea:-webkit-autofill:focus {
            -webkit-text-fill-color: #f8fbff !important;
            -webkit-box-shadow: 0 0 0 1000px #17284d inset !important;
            box-shadow: 0 0 0 1000px #17284d inset !important;
            caret-color: #8fb4ff !important;
            transition: background-color 5000s ease-in-out 0s;
        }
        /* Opened dropdown menu readability only (BaseWeb) */
        div[role="listbox"] * {
            color: #000000 !important;
        }
        div[role="option"] * {
            color: #000000 !important;
        }
        ul[role="listbox"] * {
            color: #000000 !important;
        }
        li[role="option"] * {
            color: #000000 !important;
        }
        div[role="listbox"] {
            background: #ffffff !important;
        }
        .stRadio > div, .stMultiSelect > div, .stSelectbox > div {
            color: #f8fbff !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.3rem;
            flex-wrap: wrap;
            row-gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #111d3a !important;
            border: 1px solid #2a3d67 !important;
            border-radius: 10px !important;
            color: #b7c5e3 !important;
            white-space: normal !important;
            min-height: 2.2rem;
        }
        .stTabs [aria-selected="true"] {
            background: #17284d !important;
            color: #f8fbff !important;
            border-color: #4c7edb !important;
        }
        .stButton button, .stDownloadButton button {
            background: #17284d !important;
            border: 1px solid #2a3d67 !important;
            color: #f8fbff !important;
            border-radius: 10px !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            border-color: #4c7edb !important;
            color: #f8fbff !important;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #2a3d67;
            border-radius: 10px;
            overflow: hidden;
            background: #111d3a;
        }
        [data-testid="stDataFrame"] [role="columnheader"] {
            background: #17284d !important;
            color: #f8fbff !important;
            font-weight: 700 !important;
        }
        [data-testid="stDataFrame"] [role="columnheader"] * {
            color: #f8fbff !important;
        }
        .section-frame {
            border: 1px solid #2a3d67;
            background: #111d3a;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
            margin: 0.2rem 0 0.55rem 0;
        }
        .section-subtitle {
            color: #f8fbff;
            font-size: 0.9rem;
            margin-top: -0.15rem;
            margin-bottom: 0.5rem;
        }
        .hero-narrative {
            border: 1px solid #2a3d67;
            border-left: 5px solid #8fb4ff;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            background: linear-gradient(135deg, rgba(17,29,58,0.98), rgba(23,40,77,0.95));
            margin-bottom: 1rem;
            color: #f8fbff;
        }
        .hero-narrative-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #b7c5e3;
            margin-bottom: 0.4rem;
        }
        .hero-narrative-text {
            font-size: 1rem;
            line-height: 1.55;
            color: #f8fbff;
        }
        .hero-narrative-text strong {
            color: #f8fbff;
        }
        .empty-state-card {
            border: 1px solid #2a3d67;
            border-radius: 10px;
            background: #111d3a;
            color: #f8fbff;
            padding: 0.65rem 0.8rem;
            margin: 0.2rem 0 0.45rem 0;
            font-size: 0.92rem;
        }
        .empty-state-title {
            color: #f8fbff;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .availability-note {
            border: 1px solid #2a3d67;
            border-radius: 10px;
            background: #111d3a;
            color: #f8fbff;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .nlq-mode-badge {
            border: 1px solid #2a3d67;
            border-radius: 12px;
            background: #17284d;
            color: #f8fbff;
            padding: 0.65rem 1rem;
            margin-bottom: 0.75rem;
            font-size: 0.95rem;
        }
        .nlq-mode-badge strong { color: #8fb4ff; }
        .nlq-mode-label {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #b7c5e3;
            margin-bottom: 0.25rem;
        }
        /* Reporting period date inputs: minimal scoped fix for visible value on dark navy */
        [data-testid="stDateInput"] label,
        [data-testid="stDateInput"] p {
            color: #b7c5e3 !important;
        }
        /* Visible date value: BaseWeb shows it in an input; calendar trigger is a button */
        [data-testid="stDateInput"] input {
            color: #e2e8f0 !important;
            background-color: #111d3a !important;
            border-color: #2a3d67 !important;
        }
        [data-testid="stDateInput"] input::placeholder {
            color: #94a3b8 !important;
        }
        [data-testid="stDateInput"] input:focus {
            border-color: #4c7edb !important;
            box-shadow: 0 0 0 1px #4c7edb !important;
            outline: none;
        }
        [data-testid="stDateInput"] [data-baseweb="input"] {
            background-color: #111d3a !important;
            border-color: #2a3d67 !important;
        }
        [data-testid="stDateInput"] [data-baseweb="input"] input {
            color: #e2e8f0 !important;
        }
        /* Calendar trigger button (can show date in some layouts): readable text */
        [data-testid="stDateInput"] button {
            background-color: #17284d !important;
            border-color: #2a3d67 !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stDateInput"] button:hover {
            border-color: #4c7edb !important;
            color: #e2e8f0 !important;
        }
        [data-testid="stDateInput"] button span,
        [data-testid="stDateInput"] button div {
            color: #e2e8f0 !important;
        }
        /* Selectbox / dropdown selected value on dark background */
        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] p {
            color: #b7c5e3 !important;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
            color: #f8fbff !important;
            background-color: #111d3a !important;
            border-color: #2a3d67 !important;
        }
        [data-testid="stSelectbox"] [aria-expanded="true"] {
            border-color: #4c7edb !important;
        }
        /* Responsive global controls and tab-1 filter grid (scoped via anchor markers). */
        .global-controls-grid-anchor,
        .tab1-filter-grid-anchor {
            display: block;
            height: 0;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) {
            margin-bottom: 0.5rem;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            row-gap: 0.7rem;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) div[data-testid="column"] {
            min-width: 260px;
            flex: 1 1 260px;
        }
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) {
            padding: 0.5rem 0 0.75rem 0.6rem;
            border-left: 2px solid #2a3d67;
            margin-bottom: 0.5rem;
        }
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            row-gap: 0.7rem;
        }
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) div[data-testid="column"] {
            min-width: 220px;
            flex: 1 1 220px;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) [data-testid="stWidgetLabel"],
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) [data-testid="stWidgetLabel"] {
            margin-bottom: 0.15rem !important;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) [data-testid="stWidgetLabel"] p,
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) [data-testid="stWidgetLabel"] p {
            white-space: normal !important;
            line-height: 1.25 !important;
            max-width: 100%;
        }
        div[data-testid="stVerticalBlock"]:has(.global-controls-grid-anchor) [data-testid="stCaptionContainer"],
        div[data-testid="stVerticalBlock"]:has(.tab1-filter-grid-anchor) [data-testid="stCaptionContainer"] {
            margin-top: 0.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def axis_title(text: str | None = None) -> dict:
    """Return a modern Plotly axis title config."""
    return {
        "text": text or "",
        "font": {"size": 12, "color": PALETTE["text"]},
    }


def axis_style(*, title_text: str | None = None) -> dict:
    """Return a shared axis style payload using modern Plotly syntax."""
    return {
        "title": axis_title(title_text),
        "tickfont": {"size": 11, "color": PALETTE["text_muted"]},
        "showgrid": True,
        "gridcolor": PALETTE["grid"],
        "linecolor": PALETTE["grid"],
        "zeroline": False,
    }


def apply_enterprise_plotly_theme(fig: go.Figure, *, height: int | None = None, title: str | None = None) -> go.Figure:
    """Apply consistent dark-theme chart styling with financial color semantics."""
    if title is not None:
        fig.update_layout(title={"text": title, "font": {"size": 16, "color": PALETTE["text"]}})
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font={"color": PALETTE["text"]},
        hoverlabel={"font": {"color": PALETTE["text"]}, "bgcolor": PALETTE["surface_alt"], "bordercolor": PALETTE["grid"]},
        legend={"font": {"color": PALETTE["text"]}},
    )
    fig.update_xaxes(**axis_style())
    fig.update_yaxes(**axis_style())
    if height is not None:
        fig.update_layout(height=height)
    return fig


def apply_enterprise_plotly_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Backward-compatible wrapper; ensures chart title font is white when a title exists."""
    fig = apply_enterprise_plotly_theme(fig, height=height)
    try:
        t = fig.layout.title
        if t is not None and getattr(t, "text", None):
            size = 14
            if getattr(t, "font", None) is not None and getattr(t.font, "size", None) is not None:
                size = t.font.size
            fig.update_layout(title={"text": t.text, "font": {"size": size, "color": PALETTE["text"]}})
    except Exception:
        pass
    return fig


def safe_render_plotly(
    fig_builder,
    *,
    user_message: str = "Chart unavailable for this selection.",
    width: str = "stretch",
) -> None:
    """
    Render a Plotly figure with compact user-facing failures.
    In dev mode, surface detailed exception context in an expander.
    """
    try:
        fig = fig_builder() if callable(fig_builder) else fig_builder
        st.plotly_chart(fig, width=width)
    except Exception as exc:
        st.markdown(
            (
                "<div class='empty-state-card'>"
                "<div class='empty-state-title'>Chart unavailable</div>"
                f"<div>{user_message}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        if st.session_state.get("dev_mode"):
            with st.expander("Chart error details (dev)", expanded=False):
                st.exception(exc)
