"""
Shared chart foundations for ALL Tab 1 visuals.
Uses Plotly only; no data queries. Receives clean dataframes/payloads from gateway.
Theme, number formatting, no-data placeholder, hover behavior, and validation only.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from app.ui.theme import PALETTE, apply_enterprise_plotly_style

# Theme defaults
TAB1_FONT_FAMILY = "Inter, system-ui, sans-serif"
TAB1_FONT_SIZE = 12
TAB1_TITLE_FONT_SIZE = 14
TAB1_MARGIN = dict(l=20, r=20, t=50, b=20)
TAB1_PAPER_BGCOLOR = "white"
TAB1_PLOT_BGCOLOR = "white"
TAB1_HOVERMODE = "closest"


def apply_tab1_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    """
    Apply shared Tab 1 theme: font, bg, margins, legend (horizontal top-right), hovermode.
    Optionally set figure title.
    """
    fig.update_layout(
        font=dict(family=TAB1_FONT_FAMILY, size=TAB1_FONT_SIZE),
        title_font_size=TAB1_TITLE_FONT_SIZE,
        paper_bgcolor=TAB1_PAPER_BGCOLOR,
        plot_bgcolor=TAB1_PLOT_BGCOLOR,
        margin=TAB1_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode=TAB1_HOVERMODE,
        hoverlabel=dict(
            font_family=TAB1_FONT_FAMILY,
            font_size=TAB1_FONT_SIZE,
            bgcolor="white",
            bordercolor="lightgray",
        ),
    )
    if title is not None:
        fig.update_layout(title=dict(text=title))
    return apply_enterprise_plotly_style(fig)


def format_currency_compact(x: Any) -> str:
    """
    None/NaN -> "—". Else format as $12.3B / $123.4M / $12.3K (no scientific notation).
    Keeps sign for negatives.
    """
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    av = abs(v)
    if av >= 1e9:
        return f"${v / 1e9:,.1f}B"
    if av >= 1e6:
        return f"${v / 1e6:,.1f}M"
    if av >= 1e3:
        return f"${v / 1e3:,.1f}K"
    return f"${v:,.2f}"


def format_percent_compact(x: Any) -> str:
    """None/NaN -> "—". Else format as 1.23% (decimal input, e.g. 0.0123)."""
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v * 100:.2f}%"


def make_no_data_figure(
    title: str = "Chart",
    subtitle: str = "No data for selected filters",
) -> go.Figure:
    """
    Empty axes, central annotation, themed. Safe placeholder when no data.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=f"<b>{title}</b><br>{subtitle}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=TAB1_FONT_SIZE + 2, family=TAB1_FONT_FAMILY),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        font=dict(family=TAB1_FONT_FAMILY, size=TAB1_FONT_SIZE),
        paper_bgcolor=TAB1_PAPER_BGCOLOR,
        plot_bgcolor=TAB1_PLOT_BGCOLOR,
        margin=TAB1_MARGIN,
        title=dict(text=title),
        height=280,
    )
    return fig


def validate_df(
    df: pd.DataFrame | None,
    required_cols: list[str],
) -> tuple[bool, str]:
    """
    Returns (ok, reason). ok is False if df is None, empty, or missing any required column.
    """
    if df is None:
        return False, "DataFrame is None"
    if not isinstance(df, pd.DataFrame):
        return False, "Not a DataFrame"
    if df.empty:
        return False, "DataFrame is empty"
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


def maybe_to_datetime(series: pd.Series | Any) -> pd.Series:
    """Coerce month_end (or any series) to datetime safely. Invalid -> NaT."""
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    if isinstance(series, pd.Series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def prepare_waterfall_inputs(snapshot: dict[str, Any] | pd.DataFrame | None) -> dict[str, float] | None:
    """
    Normalize waterfall inputs into numeric dict:
    {begin_aum, end_aum, nnb, market_impact}. Returns None if invalid/missing.
    """
    if snapshot is None:
        return None

    row: dict[str, Any]
    if isinstance(snapshot, dict):
        row = snapshot
    elif isinstance(snapshot, pd.DataFrame):
        if snapshot.empty:
            return None
        required = ["begin_aum", "end_aum", "nnb", "market_impact"]
        if any(c not in snapshot.columns for c in required):
            return None
        last_row = snapshot.iloc[-1]
        row = {k: last_row.get(k) for k in required}
    else:
        return None

    out: dict[str, float] = {}
    for key in ("begin_aum", "end_aum", "nnb"):
        value = row.get(key)
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(num) or math.isinf(num):
            return None
        out[key] = num

    market_value = row.get("market_impact")
    try:
        market_num = float(market_value) if market_value is not None else float("nan")
    except (TypeError, ValueError):
        market_num = float("nan")
    # Deterministic fallback when market impact is absent: end - begin - nnb.
    if math.isnan(market_num) or math.isinf(market_num):
        market_num = out["end_aum"] - out["begin_aum"] - out["nnb"]
    out["market_impact"] = market_num
    return out


WATERFALL_TITLE = "AUM Flow (Begin → NNB → Market → End)"


def build_aum_waterfall_figure(snapshot: dict[str, Any] | pd.DataFrame | None) -> go.Figure | None:
    """
    Plotly Waterfall: Begin (absolute), NNB (relative), Market (relative), End (total).
    Input: dict with begin_aum, end_aum, nnb, market_impact OR one-row DataFrame with those columns.
    No recomputation; if any value missing -> None.
    """
    values = prepare_waterfall_inputs(snapshot)
    if values is None:
        return None

    begin_aum = values["begin_aum"]
    end_aum = values["end_aum"]
    nnb = values["nnb"]
    market_impact = values["market_impact"]
    x_labels = ["Begin", "NNB", "Market", "End"]
    y_vals = [begin_aum, nnb, market_impact, end_aum]
    measures = ["absolute", "relative", "relative", "total"]

    # Bar text: compact currency (negatives with "-")
    text_vals = [format_currency_compact(v) for v in y_vals]
    # Cumulative for hover: Begin -> begin; after NNB -> begin+nnb; after Market -> end; End -> end
    cumulatives = [
        format_currency_compact(begin_aum),
        format_currency_compact(begin_aum + nnb),
        format_currency_compact(end_aum),
        format_currency_compact(end_aum),
    ]

    trace = go.Waterfall(
        x=x_labels,
        y=y_vals,
        measure=measures,
        text=text_vals,
        textposition="outside",
        customdata=cumulatives,
        hovertemplate="%{x}<br>Value: %{text}<br>Cumulative: %{customdata}<extra></extra>",
        connector=dict(line=dict(color=PALETTE["neutral"], width=1, dash="solid")),
        increasing=dict(marker=dict(color=PALETTE["positive"])),
        decreasing=dict(marker=dict(color=PALETTE["negative"])),
        totals=dict(marker=dict(color=PALETTE["primary"])),
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=dict(text=WATERFALL_TITLE),
        yaxis=dict(
            title="AUM",
            tickformat=",.0f",
            exponentformat="none",
        ),
        xaxis=dict(title=""),
        height=320,
        showlegend=False,
    )
    apply_tab1_theme(fig, title=WATERFALL_TITLE)
    return fig


def render_aum_waterfall(snapshot: dict[str, Any] | pd.DataFrame | None) -> go.Figure:
    """
    Backward-compatible wrapper for existing call sites.
    Returns no-data figure when inputs are invalid.
    """
    fig = build_aum_waterfall_figure(snapshot)
    if fig is None:
        return make_no_data_figure(WATERFALL_TITLE, "No data for selected filters")
    return fig


# ---- Channel treemap ----
def render_channel_treemap(
    df: pd.DataFrame | None,
    metric: str,
) -> go.Figure:
    """
    Treemap of channel sizes by value. Metric in {"aum", "nnb"} or {"end_aum", "nnb"}.
    Accepts df with ["channel", "value"] or ["channel", "nnb", "end_aum"]; maps metric to value column.
    Drops zeros/NaNs, sorts desc by value. Hover: channel, value (currency), % of total.
    """
    metric_norm = str(metric or "aum").strip().lower()
    metric_col = "nnb" if metric_norm == "nnb" else "end_aum"
    title = "Channel Contribution (Net New Business)" if metric_norm == "nnb" else "Channel Contribution (AUM)"
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return make_no_data_figure(title, "No data for selected filters")

    if "channel" not in df.columns:
        return make_no_data_figure(title, "Missing column: channel")
    if metric_col not in df.columns:
        if "value" not in df.columns:
            return make_no_data_figure(title, f"Missing column: {metric_col}")
        out = df[["channel", "value"]].copy()
    else:
        out = df[["channel", metric_col]].copy().rename(columns={metric_col: "value"})
    out["channel"] = out["channel"].astype(str).str.strip()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out = out[out["value"] > 0] if metric_norm != "nnb" else out[out["value"] != 0]
    out = out.sort_values("value", ascending=False).reset_index(drop=True)
    if out.empty:
        return make_no_data_figure(title, "No data for selected filters")
    if out["channel"].nunique() < 2:
        return make_no_data_figure(title, "Distribution is fully concentrated in a single channel.")
    if metric_norm == "nnb" and out["value"].abs().sum() == 0:
        return make_no_data_figure(title, "No meaningful net new business contribution is available.")

    total = out["value"].abs().sum() if metric_norm == "nnb" else out["value"].sum()
    if total == 0 or math.isnan(total):
        return make_no_data_figure(title, "No data for selected filters")
    pct = (out["value"].abs() / total * 100).tolist() if metric_norm == "nnb" else (out["value"] / total * 100).tolist()
    labels = out["channel"].tolist()
    signed_values = out["value"].tolist()
    values = out["value"].abs().tolist() if metric_norm == "nnb" else out["value"].tolist()
    parents = [""] * len(labels)
    metric_label = "NNB" if metric_norm == "nnb" else "AUM"
    fmt = (lambda x: f"{x:+,.2f}") if metric_norm == "nnb" else format_currency_compact
    text_hover = [
        f"{ch}<br>{metric_label}: {fmt(v)}<br>Share: {p:.1f}%"
        for ch, v, p in zip(labels, signed_values, pct)
    ]
    if metric_norm == "nnb":
        colors = [PALETTE["positive"] if float(v) >= 0 else PALETTE["negative"] for v in signed_values]
        marker = dict(colors=colors)
    else:
        marker = dict(colors=values, colorscale=[[0.0, "#dbeafe"], [1.0, PALETTE["primary"]]])
    trace = go.Treemap(
        marker=marker,
        labels=labels,
        values=values,
        parents=parents,
        textinfo="label+value",
        texttemplate="%{label}<br>%{percentRoot:.1%}",
        hovertext=text_hover,
        hovertemplate="%{hovertext}<extra></extra>",
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(title=dict(text=title), height=380, margin=dict(l=12, r=12, t=52, b=10))
    apply_tab1_theme(fig, title=title)
    return fig


# ---- Growth quality matrix ----
def render_growth_quality_matrix(
    df: pd.DataFrame | None,
    view: str,
    top_n_aum: int = 10,
) -> go.Figure:
    """
    Bubble scatter: x=NNB, y=fee_yield (fallback to OGR), size=AUM, color=channel.
    Drop rows where aum is NaN or <= 0. Labels for top N by aum + top/bottom 3 by nnb; else hover only.
    view in {"channel", "ticker"} for title. Required columns: label, nnb, aum.
    """
    title_suffix = "Channels" if (view or "").lower() == "channel" else "Tickers"
    title = f"Growth Quality Matrix ({title_suffix})"
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return make_no_data_figure(title, "No data for selected filters")
    required = ["label", "nnb", "aum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return make_no_data_figure(title, f"Missing columns: {missing}")

    out = df.copy()
    out["aum"] = pd.to_numeric(out["aum"], errors="coerce")
    out["nnb"] = pd.to_numeric(out["nnb"], errors="coerce")
    y_col = "fee_yield" if "fee_yield" in out.columns else None
    if y_col is None:
        y_col = next((c for c in ("ogr", "organic_growth_rate", "organic_growth") if c in out.columns), None)
    if y_col is None:
        out["matrix_y"] = 0.0
        y_col = "matrix_y"
        y_label = "Organic Growth Rate (%)"
    else:
        out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
        y_label = "Fee Yield (%)" if y_col == "fee_yield" else "Organic Growth Rate (%)"
    if "channel" not in out.columns:
        out["channel"] = "Unassigned"
    else:
        out["channel"] = out["channel"].astype(str).str.strip().replace("", "Unassigned")
        out["channel"] = out["channel"].fillna("Unassigned")
    out = out.dropna(subset=["aum"])
    out = out[out["aum"] > 0]
    out = out.dropna(subset=["nnb", y_col])
    if out.empty:
        return make_no_data_figure(title, "No data for selected filters")

    nnb_vals = out["nnb"].tolist()
    y_vals = out[y_col].tolist()
    aum_vals = out["aum"].tolist()
    labels = out["label"].astype(str).tolist()
    channels = out["channel"].astype(str).tolist()
    # Both fee_yield and OGR are decimal rates in dataset; display as percentage.
    y_display = [val * 100 if not (math.isnan(val) or math.isinf(val)) else 0 for val in y_vals]
    median_y = out[y_col].median()
    if math.isnan(median_y):
        median_y = 0.0
    median_y_pct = median_y * 100

    # Which points get text labels: top top_n_aum by aum + top 3 nnb + bottom 3 nnb
    out_sorted_aum = out.sort_values("aum", ascending=False)
    top_aum_idx = set(out_sorted_aum.head(top_n_aum).index.tolist())
    out_sorted_nnb = out.sort_values("nnb", ascending=False)
    top_nnb_idx = set(out_sorted_nnb.head(3).index.tolist())
    bottom_nnb_idx = set(out_sorted_nnb.tail(3).index.tolist())
    show_text_idx = top_aum_idx | top_nnb_idx | bottom_nnb_idx
    text_vals = [labels[j] if out.index[j] in show_text_idx else "" for j in range(len(out))]

    hover_lines = [
        f"{lab}<br>Channel: {ch}<br>NNB: {format_currency_compact(n)}<br>{y_label.replace(' (%)', '')}: {format_percent_compact(y)}<br>AUM: {format_currency_compact(a)}"
        for lab, ch, n, y, a in zip(labels, channels, nnb_vals, y_vals, aum_vals)
    ]
    # customdata for reliable click parsing: [ticker, channel] per point (index 0=ticker, 1=channel)
    is_ticker_view = (view or "").lower() == "ticker"
    customdata_list = [
        [lab, ch] if is_ticker_view else [None, ch]
        for lab, ch in zip(labels, channels)
    ]
    # Marker size: scale by aum (use sqrt so area ~ aum)
    aum_max = max(aum_vals) if aum_vals else 1
    sizes = [max(8, 40 * (a / aum_max) ** 0.5) for a in aum_vals]

    channel_order = sorted(set(channels))
    base_colors = [PALETTE["primary"], PALETTE["accent"], PALETTE["positive"], PALETTE["negative"], PALETTE["neutral"]]
    color_map = {ch: base_colors[i % len(base_colors)] for i, ch in enumerate(channel_order)}
    marker_colors = [color_map[ch] for ch in channels]

    scatter = go.Scatter(
        x=nnb_vals,
        y=y_display,
        text=text_vals,
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=TAB1_FONT_SIZE - 1, family=TAB1_FONT_FAMILY),
        marker=dict(
            size=sizes,
            sizemode="diameter",
            sizeref=1,
            line=dict(width=0.7, color=PALETTE["neutral"]),
            color=marker_colors,
        ),
        hovertext=hover_lines,
        hoverinfo="text",
        customdata=customdata_list,
        name=title_suffix,
    )
    fig = go.Figure(data=[scatter])
    # Quadrant lines: x=0, y=median(y metric)
    shapes = [
        dict(type="line", x0=0, x1=0, y0=0, y1=1, yref="paper", line=dict(color=PALETTE["neutral"], dash="dot", width=1)),
        dict(type="line", x0=0, x1=1, xref="paper", y0=median_y_pct, y1=median_y_pct, line=dict(color=PALETTE["neutral"], dash="dot", width=1)),
    ]
    fig.update_layout(
        title=dict(text=title),
        xaxis=dict(title="NNB", tickformat=",.0f", exponentformat="none", zeroline=True),
        yaxis=dict(title=y_label, tickformat=",.2f"),
        shapes=shapes,
        height=420,
        showlegend=False,
    )
    apply_tab1_theme(fig, title=title)
    return fig


# ---- Growth trend (OGR + Market Impact Rate + vol band) ----
GROWTH_TREND_TITLE = "OGR & Market Impact Rate (with 6M Vol Band)"
ROLLING_WINDOW = 6
ROLLING_MIN_PERIODS = 3


def render_growth_trend(df_series: pd.DataFrame | None) -> go.Figure:
    """
    Two lines: OGR (%), Market Impact Rate (%). Rolling vol band = ogr ± rolling_std (window=6, min_periods=3).
    Single y-axis (%). Hover: month, both series, vol band. No data -> make_no_data_figure("Growth Trend").
    """
    if df_series is None or not isinstance(df_series, pd.DataFrame) or df_series.empty:
        return make_no_data_figure("Growth Trend", "No data for selected filters")
    required = ["month_end", "ogr", "market_impact_rate"]
    missing = [c for c in required if c not in df_series.columns]
    if missing:
        return make_no_data_figure("Growth Trend", f"Missing columns: {missing}")

    out = df_series.copy()
    out["month_end"] = maybe_to_datetime(out["month_end"])
    out = out.dropna(subset=["month_end"])
    out["ogr"] = pd.to_numeric(out["ogr"], errors="coerce")
    out["market_impact_rate"] = pd.to_numeric(out["market_impact_rate"], errors="coerce")
    out = out.sort_values("month_end").reset_index(drop=True)
    if out.empty or len(out) < 2:
        return make_no_data_figure("Growth Trend", "No data for selected filters")

    x = out["month_end"].tolist()
    ogr_pct = (out["ogr"] * 100).tolist()
    mir_pct = (out["market_impact_rate"] * 100).tolist()
    rolling_std = out["ogr"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).std()
    month_labels = [pd.Timestamp(t).strftime("%Y-%m") if pd.notna(t) else "" for t in x]
    hover_vol = []
    for i in range(len(out)):
        rs = rolling_std.iloc[i] if i < len(rolling_std) else None
        hover_vol.append(f"Vol band: {format_percent_compact(rs)}" if pd.notna(rs) else "Vol band: —")
    hover_lines = [
        f"Month: {ml}<br>OGR: {format_percent_compact(ogr_pct[j] / 100)}<br>Market Impact Rate: {format_percent_compact(mir_pct[j] / 100)}<br>{hover_vol[j]}"
        for j, ml in enumerate(month_labels)
    ]
    upper = (out["ogr"] * 100 + rolling_std * 100).tolist()
    lower = (out["ogr"] * 100 - rolling_std * 100).tolist()

    data = []
    data.append(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(color="rgba(76, 126, 219, 0.35)", width=0.5),
            fill=None,
            showlegend=False,
        )
    )
    data.append(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(color="rgba(76, 126, 219, 0.35)", width=0.5),
            fill="tonexty",
            fillcolor="rgba(76, 126, 219, 0.14)",
            showlegend=False,
        )
    )
    data.append(
        go.Scatter(
            x=x,
            y=ogr_pct,
            mode="lines+markers",
            name="OGR",
            line=dict(color=PALETTE["primary"], width=2),
            marker=dict(size=6),
            hovertext=hover_lines,
            hoverinfo="text",
        )
    )
    data.append(
        go.Scatter(
            x=x,
            y=mir_pct,
            mode="lines+markers",
            name="Market Impact Rate",
            line=dict(color=PALETTE["secondary"], width=2),
            marker=dict(size=6),
            hovertext=hover_lines,
            hoverinfo="text",
        )
    )
    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=GROWTH_TREND_TITLE),
        xaxis=dict(title="Month", type="date", tickformat="%b %Y"),
        yaxis=dict(title="%", tickformat=",.1f", exponentformat="none"),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_tab1_theme(fig, title=GROWTH_TREND_TITLE)
    return fig
