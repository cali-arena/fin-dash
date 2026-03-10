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
            background: #111d3a !important;
            border-color: #2a3d67 !important;
            color: #f8fbff !important;
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
        }
        .stTabs [data-baseweb="tab"] {
            background: #111d3a !important;
            border: 1px solid #2a3d67 !important;
            border-radius: 10px !important;
            color: #b7c5e3 !important;
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
        .insight-banner {
            border: 1px solid #36507f;
            border-left: 5px solid #8fb4ff;
            border-radius: 12px;
            padding: 0.8rem 0.95rem;
            background: linear-gradient(135deg, rgba(23,40,77,0.95), rgba(17,29,58,0.95));
            margin-bottom: 0.6rem;
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
