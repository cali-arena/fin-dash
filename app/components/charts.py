"""
Plotly chart builders. Return figures only; no state or data access.
Mock data when df is None so components run with no data.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.ui.theme import PALETTE, apply_enterprise_plotly_style, safe_render_plotly

# Exclude from chart aggregation only (so "Unassigned" / "—" / blank do not appear as categories)
CHART_EXCLUDE_DIM = frozenset({"", "Unassigned", "—", "nan"})

# Expected column names when passing real data (documentation only)
# fig_aum_over_time: df with columns like "month_end" (or "date") + "end_aum" (or "aum")
# fig_nnb_by_channel: df with "channel" (or "channel_l1") + "nnb" (or "value")
# fig_growth_quality_matrix: df with "x" / "growth", "y" / "quality", optional "label" / "name"


def _mock_series_data() -> pd.DataFrame:
    """Mock time series for AUM over time."""
    return pd.DataFrame({
        "month_end": pd.date_range("2023-01-01", periods=12, freq="ME"),
        "end_aum": [100, 105, 108, 112, 110, 115, 120, 118, 122, 125, 128, 130],
    })


def _mock_channel_data() -> pd.DataFrame:
    """Mock NNB by channel."""
    return pd.DataFrame({
        "channel": ["Wholesale", "Retail", "Institutional", "Other"],
        "nnb": [12.5, -2.1, 8.3, 1.0],
    })


def _mock_scatter_data() -> pd.DataFrame:
    """Mock growth vs quality for scatter."""
    return pd.DataFrame({
        "growth": [0.02, -0.01, 0.05, 0.03, -0.02, 0.04],
        "quality": [0.8, 0.6, 0.9, 0.7, 0.5, 0.85],
        "label": ["A", "B", "C", "D", "E", "F"],
    })


def fig_aum_over_time(df: pd.DataFrame | None = None) -> go.Figure:
    """
    Simple Plotly line chart for AUM over time.
    If df is None, uses mock data. Expects df with date-like column and value column (e.g. month_end, end_aum).
    """
    if df is None or df.empty:
        df = _mock_series_data()
    date_col = "month_end" if "month_end" in df.columns else "date" if "date" in df.columns else df.columns[0]
    value_col = "end_aum" if "end_aum" in df.columns else "aum" if "aum" in df.columns else df.columns[1]
    x = pd.to_datetime(df[date_col])
    y = df[value_col]
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers", name="AUM")])
    fig.update_traces(line=dict(color=PALETTE["primary"], width=2), marker=dict(color=PALETTE["primary"], size=7))
    fig.update_layout(
        title="AUM over time",
        xaxis_title=date_col,
        yaxis_title=value_col,
        template="plotly_white",
        height=350,
    )
    return apply_enterprise_plotly_style(fig)


def fig_nnb_by_channel(df: pd.DataFrame | None = None) -> go.Figure:
    """
    Plotly bar chart for NNB by channel.
    If df is None, uses mock data. Expects df with channel column and value column (e.g. channel, nnb).
    Placeholders (Unassigned, —, blank) are excluded from the chart only.
    """
    if df is None or df.empty:
        df = _mock_channel_data()
    label_col = "channel" if "channel" in df.columns else "channel_l1" if "channel_l1" in df.columns else df.columns[0]
    value_col = "nnb" if "nnb" in df.columns else "value" if "value" in df.columns else df.columns[1]
    work = df.copy()
    if label_col in work.columns:
        s = work[label_col].astype(str).str.strip()
        work = work[s.notna() & ~s.isin(CHART_EXCLUDE_DIM) & (s != "")]
    fig = go.Figure(data=[
        go.Bar(x=work[label_col], y=work[value_col], name="NNB", marker_color=PALETTE["primary"]),
    ])
    fig.update_layout(
        title="NNB by channel",
        xaxis_title=label_col,
        yaxis_title=value_col,
        template="plotly_white",
        height=350,
    )
    return apply_enterprise_plotly_style(fig)


def fig_growth_quality_matrix(df: pd.DataFrame | None = None) -> go.Figure:
    """
    Plotly scatter for growth vs quality matrix.
    If df is None, uses mock data. Expects df with x/growth, y/quality, optional label/name.
    """
    if df is None or df.empty:
        df = _mock_scatter_data()
    x_col = "growth" if "growth" in df.columns else "x" if "x" in df.columns else df.columns[0]
    y_col = "quality" if "quality" in df.columns else "y" if "y" in df.columns else df.columns[1]
    text_col = "label" if "label" in df.columns else "name" if "name" in df.columns else None
    x = df[x_col]
    y = df[y_col]
    text = df[text_col].astype(str).tolist() if text_col and text_col in df.columns else None
    fig = go.Figure(data=[
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text" if text else "markers",
            text=text,
            textposition="top center",
            marker=dict(size=12, color=y, colorscale=[[0.0, PALETTE["accent"]], [1.0, PALETTE["primary"]]], showscale=True),
            name="Growth vs quality",
        ),
    ])
    fig.update_layout(
        title="Growth vs quality matrix",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white",
        height=400,
    )
    return apply_enterprise_plotly_style(fig)


def fig_nnb_trend(df: pd.DataFrame | None = None) -> go.Figure:
    """
    Plotly line chart for NNB over time (monthly series).
    Expects df with month_end and nnb (e.g. from gateway chart_nnb_trend).
    """
    if df is None or df.empty:
        df = pd.DataFrame({
            "month_end": pd.date_range("2023-01-01", periods=12, freq="ME"),
            "nnb": [2.0, -0.5, 1.2, 0.8, -0.3, 1.5, 2.1, 0.0, 1.0, 0.9, 1.2, 1.5],
        })
    date_col = "month_end" if "month_end" in df.columns else df.columns[0]
    value_col = "nnb" if "nnb" in df.columns else df.columns[1]
    x = pd.to_datetime(df[date_col])
    y = df[value_col]
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers", name="NNB")])
    fig.update_traces(line=dict(color=PALETTE["secondary"], width=2), marker=dict(color=PALETTE["secondary"], size=7))
    fig.update_layout(
        title="NNB over time",
        xaxis_title=date_col,
        yaxis_title=value_col,
        template="plotly_white",
        height=350,
    )
    return apply_enterprise_plotly_style(fig)


def fig_waterfall(decomposition: dict[str, Any] | None = None) -> go.Figure:
    """
    Plotly waterfall from gateway growth_decomposition_inputs dict.
    Keys: start_aum, organic, external, market, total_change, end_aum.
    UI transforms dict → figure only; no aggregate computation.
    """
    d = decomposition or {}
    start_aum = float(d.get("start_aum", 0))
    organic = float(d.get("organic", 0))
    external = float(d.get("external", 0))
    market = float(d.get("market", 0))
    end_aum = float(d.get("end_aum", start_aum))

    x = ["Start AUM", "Organic", "External", "Market", "End AUM"]
    y = [start_aum, organic, external, market, end_aum]
    measure = ["total", "relative", "relative", "relative", "total"]

    fig = go.Figure(go.Waterfall(
        name="AUM change",
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        text=[f"{v:,.1f}" for v in y],
        textposition="outside",
        increasing=dict(marker=dict(color=PALETTE["positive"])),
        decreasing=dict(marker=dict(color=PALETTE["negative"])),
        totals=dict(marker=dict(color=PALETTE["primary"])),
        connector=dict(line=dict(color=PALETTE["neutral"], width=1)),
    ))
    fig.update_layout(
        title="AUM growth decomposition",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return apply_enterprise_plotly_style(fig)


def render_plotly_figure(fig: go.Figure, width: str = "stretch") -> None:
    """
    Render a Plotly figure in Streamlit. Use from pages: st.plotly_chart(fig, ...).
    Provided for convenience; callers may use st.plotly_chart directly.
    """
    safe_render_plotly(
        fig,
        user_message="Chart unavailable for this selection.",
        width=width,
    )
