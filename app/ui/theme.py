from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PALETTE = {
    "primary": "#1f3b73",   # dark_blue
    "secondary": "#4c7edb", # accent
    "accent": "#4c7edb",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#64748b",
}


def configure_plotly_defaults() -> None:
    """Set global Plotly defaults for a readable white institutional theme."""
    pio.templates.default = "plotly_white"
    pio.templates["plotly_white"].layout.colorway = [
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["positive"],
        PALETTE["negative"],
        PALETTE["neutral"],
    ]


def inject_global_theme_css() -> None:
    """Inject global CSS for white background, high contrast text, and clean cards."""
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #111111 !important;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: #ffffff !important;
        }
        .stMarkdown, .stCaption, .stText, label, p, li, span {
            color: #222222 !important;
        }
        h1, h2, h3, h4, h5 {
            color: #111111 !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e6e8eb !important;
            border-radius: 10px !important;
            padding: 0.55rem 0.7rem !important;
            background: #f7f8fa !important;
            min-height: 112px;
        }
        div[data-testid="stMetricLabel"] {
            color: #222222 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #111111 !important;
            font-weight: 800 !important;
        }
        div[data-testid="stMetricDelta"] {
            font-weight: 700 !important;
        }
        .empty-state-card {
            border: 1px solid #e6e8eb;
            border-radius: 10px;
            background: #f7f8fa;
            color: #222222;
            padding: 0.65rem 0.8rem;
            margin: 0.2rem 0 0.45rem 0;
            font-size: 0.92rem;
        }
        .empty-state-title {
            color: #111111;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .availability-note {
            border: 1px solid #e6e8eb;
            border-radius: 10px;
            background: #f7f8fa;
            color: #222222;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def axis_title(text: str | None = None) -> dict:
    """Return a modern Plotly axis title config."""
    return {
        "text": text or "",
        "font": {"size": 12, "color": "#111111"},
    }


def axis_style(*, title_text: str | None = None) -> dict:
    """Return a shared axis style payload using modern Plotly syntax."""
    return {
        "title": axis_title(title_text),
        "tickfont": {"size": 11, "color": "#222222"},
        "showgrid": True,
        "gridcolor": "#e6e8eb",
        "linecolor": "#d8dde3",
        "zeroline": False,
    }


def apply_enterprise_plotly_theme(fig: go.Figure, *, height: int | None = None, title: str | None = None) -> go.Figure:
    """Apply consistent white-theme chart styling with readable axes and subtle grids."""
    if title is not None:
        fig.update_layout(title={"text": title, "font": {"size": 16, "color": "#111111"}})
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#111111"},
        hoverlabel={"font": {"color": "#111111"}, "bgcolor": "#ffffff"},
        legend={"font": {"color": "#111111"}},
    )
    fig.update_xaxes(**axis_style())
    fig.update_yaxes(**axis_style())
    if height is not None:
        fig.update_layout(height=height)
    return fig


def apply_enterprise_plotly_style(fig: go.Figure, *, height: int | None = None) -> go.Figure:
    """Backward-compatible wrapper for existing imports."""
    return apply_enterprise_plotly_theme(fig, height=height)


def safe_render_plotly(
    fig_builder,
    *,
    user_message: str = "Chart unavailable for this selection.",
    use_container_width: bool = True,
) -> None:
    """
    Render a Plotly figure with compact user-facing failures.
    In dev mode, surface detailed exception context in an expander.
    """
    try:
        fig = fig_builder() if callable(fig_builder) else fig_builder
        st.plotly_chart(fig, use_container_width=use_container_width)
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
