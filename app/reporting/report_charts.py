"""
Static styled charts for Tab 2 Dynamic Report. Matplotlib/Seaborn only; no Plotly.
Optional: if matplotlib/seaborn are missing, functions return None and the tab falls back to Plotly or table.
"""
from __future__ import annotations

import io
from typing import Any

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    plt = None

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False
    sns = None

# Enterprise-style colors (match app.ui.theme where possible)
_CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#2ca02c",
    "negative": "#d62728",
    "positive": "#2ca02c",
    "grid": "#e0e0e0",
    "text": "#333333",
}


def _style_axes(ax: Any) -> None:
    if ax is None:
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=_CHART_COLORS["grid"], linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_CHART_COLORS["text"])


def fig_aum_trend_mpl(ts: pd.DataFrame, title: str = "AUM trend") -> bytes | None:
    """
    Static AUM-over-time line chart. Returns PNG bytes or None if matplotlib missing or insufficient data.
    """
    if not _HAS_MPL or ts is None or ts.empty:
        return None
    if "month_end" not in ts.columns or "end_aum" not in ts.columns:
        return None
    work = ts[["month_end", "end_aum"]].copy()
    work["month_end"] = pd.to_datetime(work["month_end"], errors="coerce")
    work["end_aum"] = pd.to_numeric(work["end_aum"], errors="coerce")
    work = work.dropna(subset=["month_end", "end_aum"]).sort_values("month_end")
    if len(work) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(work["month_end"], work["end_aum"], color=_CHART_COLORS["primary"], linewidth=2, marker="o", markersize=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(work) // 6)))
    plt.xticks(rotation=25)
    ax.set_title(title, fontsize=12, color=_CHART_COLORS["text"])
    ax.set_ylabel("End AUM", fontsize=10, color=_CHART_COLORS["text"])
    _style_axes(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_ranked_bars_mpl(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str = "Top / Bottom",
    top_n: int = 5,
    bottom_n: int = 5,
) -> bytes | None:
    """
    Horizontal bar chart: top N and bottom N by value_col. Positive values green, negative red.
    Returns PNG bytes or None.
    """
    if not _HAS_MPL or df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return None
    work = df[[label_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    work = work.sort_values(value_col, ascending=False)
    top = work.head(top_n)
    bottom = work.tail(bottom_n)
    combined = pd.concat([top, bottom]).drop_duplicates(subset=[label_col])
    if combined.empty or len(combined) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, max(3, len(combined) * 0.35)))
    y_pos = range(len(combined))
    colors = [_CHART_COLORS["positive"] if v >= 0 else _CHART_COLORS["negative"] for v in combined[value_col]]
    ax.barh(y_pos, combined[value_col], color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(combined[label_col].astype(str), fontsize=9)
    ax.set_title(title, fontsize=12, color=_CHART_COLORS["text"])
    ax.set_xlabel(value_col, fontsize=10, color=_CHART_COLORS["text"])
    _style_axes(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fig_share_bars_mpl(
    df: pd.DataFrame,
    label_col: str,
    share_col: str,
    title: str = "Share (top 5)",
    n: int = 5,
) -> bytes | None:
    """Concentration: horizontal bars for share. Single color."""
    if not _HAS_MPL or df is None or df.empty or label_col not in df.columns or share_col not in df.columns:
        return None
    work = df[[label_col, share_col]].copy()
    work[share_col] = pd.to_numeric(work[share_col], errors="coerce")
    work = work.dropna(subset=[share_col]).sort_values(share_col, ascending=False).head(n)
    if len(work) < 2:
        return None
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, len(work) * 0.4)))
    y_pos = range(len(work))
    ax.barh(y_pos, work[share_col], color=_CHART_COLORS["secondary"], height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(work[label_col].astype(str), fontsize=9)
    ax.set_title(title, fontsize=12, color=_CHART_COLORS["text"])
    ax.set_xlabel(share_col, fontsize=10, color=_CHART_COLORS["text"])
    _style_axes(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
