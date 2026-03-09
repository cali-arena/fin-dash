"""
Executive Dashboard tab: KPI + charts + drilldown wired to DataGateway. No groupby in page.
Data: gateway only (no duckdb/parquet in pages).
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from app.components.charts import fig_aum_over_time, fig_nnb_trend, fig_waterfall
from app.components.kpis import kpis_from_gateway_dict, render_kpi_row
from app.analytics.commentary_engine import build_executive_insight_sections
from app.data.data_gateway import (
    HEAVY_BUDGET_MS,
    get_channel_breakdown,
    get_firm_snapshot,
    get_growth_quality,
    get_trend_series,
)
from app.panels.details_panel import render_details_panel
from app.reporting.firm_narrative import build_firm_narrative_text
try:
    from app.observability import render_obs_panel
except ImportError:
    def render_obs_panel(_tab_id: str) -> None:
        return
from app.ui.observability import render_observability_panel
from app.drill_events import parse_treemap_click
from app.optional_deps import try_import_plotly_events
from app.state import (
    DRILL_RESET_FLAG_KEY,
    FilterState,
    get_drill_state,
    get_filter_state,
    revalidate_drill_on_filter_change,
    set_drill_mode,
    set_selected_channel,
    set_selected_ticker,
    update_drill_state,
    update_filter_state,
    validate_drill_selection,
)
from app.ui.exports import render_export_buttons
from app.ui.guardrails import (
    fallback_note,
    render_chart_or_fallback,
    render_error_state,
    render_empty_state,
    render_timeout_state,
    missing_required_columns,
)
from app.ui.theme import PALETTE, apply_enterprise_plotly_style, safe_render_plotly
from app.viz.tab1_charts import (
    render_aum_waterfall,
    render_channel_treemap,
    render_growth_trend,
)

ROOT = Path(__file__).resolve().parents[2]

# Dashboard semantic colors
COLOR_POSITIVE = "#2ca02c"
COLOR_NEGATIVE = "#d62728"
COLOR_NEUTRAL = "#6c757d"
COLOR_PRIMARY = "#1f3b73"
COLOR_SECONDARY = "#4c7edb"
COLOR_BACKGROUND = "#f5f7fa"

# Session key for tracking gateway queries executed this run
VIZ_QUERIES_KEY = "_viz_gateway_queries"

# Dimension options for path/slice (fallback when contract has no list)
DEFAULT_DIM_OPTIONS = ["channel", "country", "ticker", "segment", "sub_segment", "product_ticker"]

def build_ranked_channels(df: pd.DataFrame, metric: str, top_n: int = 25) -> pd.DataFrame:
    """
    Rank channels by NNB or AUM. Returns df with columns [rank, channel, NNB, AUM].
    metric in {"NNB","AUM"} (case-insensitive). Deterministic tie-break: dimension name asc.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["rank", "channel", "NNB", "AUM"])
    if "channel" not in df.columns:
        raise ValueError(f"build_ranked_channels: missing required column 'channel'. Have: {list(df.columns)}")
    aum_col = "end_aum" if "end_aum" in df.columns else ("aum" if "aum" in df.columns else None)
    nnb_col = "nnb" if "nnb" in df.columns else None
    if not aum_col and not nnb_col:
        raise ValueError(
            "build_ranked_channels: need at least one of [end_aum, aum, nnb]. Have: " + str(list(df.columns))
        )
    metric_upper = str(metric).strip().upper()
    if metric_upper not in ("NNB", "AUM"):
        raise ValueError(f"build_ranked_channels: metric must be NNB or AUM, got {metric!r}")
    aggs = {}
    if nnb_col:
        aggs["NNB"] = (nnb_col, "sum")
    if aum_col:
        aggs["AUM"] = (aum_col, "sum")
    out = df.groupby("channel", as_index=False).agg(**aggs)
    if "NNB" not in out.columns:
        out["NNB"] = 0.0
    if "AUM" not in out.columns:
        out["AUM"] = 0.0
    out["NNB"] = pd.to_numeric(out["NNB"], errors="coerce").fillna(0)
    out["AUM"] = pd.to_numeric(out["AUM"], errors="coerce").fillna(0)
    sort_col = "NNB" if metric_upper == "NNB" else "AUM"
    out = out.sort_values(by=[sort_col, "channel"], ascending=[False, True]).reset_index(drop=True).head(top_n)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[["rank", "channel", "NNB", "AUM"]]


def build_ranked_tickers(df: pd.DataFrame, metric: str, top_n: int = 25) -> pd.DataFrame:
    """
    Rank tickers by NNB or AUM. Returns df with columns [rank, ticker, NNB, AUM].
    Dimension column: "ticker" or "label". metric in {"NNB","AUM"} (case-insensitive).
    Deterministic tie-break: dimension name asc.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["rank", "ticker", "NNB", "AUM"])
    dim_col = "ticker" if "ticker" in df.columns else ("label" if "label" in df.columns else None)
    if not dim_col:
        raise ValueError(
            f"build_ranked_tickers: missing dimension column (ticker or label). Have: {list(df.columns)}"
        )
    aum_col = "end_aum" if "end_aum" in df.columns else ("aum" if "aum" in df.columns else None)
    nnb_col = "nnb" if "nnb" in df.columns else None
    if not aum_col and not nnb_col:
        raise ValueError(
            "build_ranked_tickers: need at least one of [end_aum, aum, nnb]. Have: " + str(list(df.columns))
        )
    metric_upper = str(metric).strip().upper()
    if metric_upper not in ("NNB", "AUM"):
        raise ValueError(f"build_ranked_tickers: metric must be NNB or AUM, got {metric!r}")
    aggs = {}
    if nnb_col:
        aggs["NNB"] = (nnb_col, "sum")
    if aum_col:
        aggs["AUM"] = (aum_col, "sum")
    out = df.groupby(dim_col, as_index=False).agg(**aggs)
    out = out.rename(columns={dim_col: "ticker"})
    if "NNB" not in out.columns:
        out["NNB"] = 0.0
    if "AUM" not in out.columns:
        out["AUM"] = 0.0
    out["NNB"] = pd.to_numeric(out["NNB"], errors="coerce").fillna(0)
    out["AUM"] = pd.to_numeric(out["AUM"], errors="coerce").fillna(0)
    sort_col = "NNB" if metric_upper == "NNB" else "AUM"
    out = out.sort_values(by=[sort_col, "ticker"], ascending=[False, True]).reset_index(drop=True).head(top_n)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[["rank", "ticker", "NNB", "AUM"]]


def _render_ranked_kpi_cards(df: pd.DataFrame, label_col: str, title: str) -> None:
    """Compact KPI strip for ranked contributor datasets."""
    if df is None or df.empty:
        return
    top_row = df.sort_values(["NNB", label_col], ascending=[False, True]).iloc[0]
    bottom_row = df.sort_values(["NNB", label_col], ascending=[True, True]).iloc[0]
    total_nnb = float(pd.to_numeric(df["NNB"], errors="coerce").fillna(0).sum())
    total_aum = float(pd.to_numeric(df["AUM"], errors="coerce").fillna(0).sum())
    pos_share = float((pd.to_numeric(df["NNB"], errors="coerce").fillna(0) > 0).mean() * 100)
    st.caption(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Top {label_col.title()} (NNB)", str(top_row[label_col]), f"{float(top_row['NNB']):,.2f}")
    c2.metric(f"Lowest {label_col.title()} (NNB)", str(bottom_row[label_col]), f"{float(bottom_row['NNB']):,.2f}")
    c3.metric("Total NNB", f"{total_nnb:,.2f}")
    c4.metric("Positive Mix", f"{pos_share:,.1f}%", f"AUM {total_aum:,.2f}")


def _render_top_bottom_bar(df: pd.DataFrame, label_col: str, chart_title: str, top_n: int = 6) -> None:
    """Top/bottom contributor bar chart from ranked frames."""
    if go is None:
        render_empty_state("Plotly unavailable for contributor bar chart.", "Install plotly to render this visual.")
        return
    if df is None or df.empty:
        render_empty_state("No ranked data available.", "Widen date range or relax filters.")
        return
    pos = df.sort_values(["NNB", label_col], ascending=[False, True]).head(top_n)
    neg = df.sort_values(["NNB", label_col], ascending=[True, True]).head(top_n)
    bar_df = pd.concat([pos, neg], ignore_index=True)
    if bar_df.empty:
        render_empty_state("No contributor extremes available.", "Adjust filters.")
        return
    bar_df["color"] = bar_df["NNB"].apply(lambda v: COLOR_POSITIVE if float(v) >= 0 else COLOR_NEGATIVE)
    fig = go.Figure(
        data=[
            go.Bar(
                x=bar_df["NNB"],
                y=bar_df[label_col],
                orientation="h",
                marker=dict(color=bar_df["color"]),
                text=[f"{v:,.2f}" for v in bar_df["NNB"]],
                textposition="outside",
                hovertemplate=f"{label_col.title()}: %{{y}}<br>NNB: %{{x:,.2f}}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title="NNB",
        yaxis_title=label_col.title(),
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    apply_enterprise_plotly_style(fig, height=380)
    safe_render_plotly(fig, user_message="Contributor bar unavailable for this selection.")


def _render_growth_impact_scatter(df: pd.DataFrame, label_col: str, chart_title: str) -> None:
    """Growth vs market impact proxy scatter using NNB and AUM."""
    if go is None:
        render_empty_state("Plotly unavailable for scatter chart.", "Install plotly to render this visual.")
        return
    if df is None or df.empty:
        render_empty_state("No ranked data available.", "Widen date range or relax filters.")
        return
    work = df.copy()
    work["impact_ratio"] = work.apply(
        lambda r: float(r["NNB"]) / float(r["AUM"]) if float(r["AUM"]) != 0 else 0.0,
        axis=1,
    )
    fig = go.Figure(
        data=[
            go.Scatter(
                x=work["AUM"],
                y=work["NNB"],
                mode="markers+text",
                text=work[label_col],
                textposition="top center",
                marker=dict(
                    size=14,
                    color=work["impact_ratio"],
                    colorscale=[[0.0, PALETTE["accent"]], [1.0, PALETTE["primary"]]],
                    showscale=True,
                    colorbar=dict(title="Impact ratio"),
                ),
                hovertemplate=f"{label_col.title()}: %{{text}}<br>AUM: %{{x:,.2f}}<br>NNB: %{{y:,.2f}}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title="Market Footprint (AUM)",
        yaxis_title="Growth Contribution (NNB)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    apply_enterprise_plotly_style(fig, height=380)
    safe_render_plotly(fig, user_message="Growth vs impact scatter unavailable for this selection.")


def prepare_growth_quality_dataset(df: pd.DataFrame | None, view: str) -> pd.DataFrame:
    """
    Canonical matrix contract:
    label, nnb, aum with y-axis preference fee_yield -> ogr fallback.
    Keeps channel/ticker passthrough for drill + color encoding.
    """
    base_cols = ["label", "nnb", "aum", "fee_yield", "ogr", "organic_growth_rate", "organic_growth"]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=base_cols)
    view_norm = str(view or "channel").strip().lower()
    label_candidates = (
        ["label", "channel", "preferred_label", "channel_l1", "channel_best", "channel_standard"]
        if view_norm == "channel"
        else ["label", "ticker", "product_ticker", "preferred_label"]
    )
    label_col = next((c for c in label_candidates if c in df.columns), None)
    aum_col = next((c for c in ("aum", "end_aum", "total_aum") if c in df.columns), None)
    nnb_col = next((c for c in ("nnb", "net_new_business") if c in df.columns), None)
    fee_col = next((c for c in ("fee_yield", "fee", "yield") if c in df.columns), None)
    ogr_col = next((c for c in ("ogr", "organic_growth_rate", "organic_growth") if c in df.columns), None)
    channel_col = next(
        (c for c in ("channel", "channel_l1", "channel_best", "channel_standard", "preferred_channel") if c in df.columns),
        None,
    )
    if not label_col or not aum_col or not nnb_col:
        return pd.DataFrame(columns=base_cols)
    out = df.copy()
    out["label"] = out[label_col].astype(str).str.strip()
    out["aum"] = pd.to_numeric(out[aum_col], errors="coerce")
    out["nnb"] = pd.to_numeric(out[nnb_col], errors="coerce")
    out["fee_yield"] = pd.to_numeric(out[fee_col], errors="coerce") if fee_col else float("nan")
    if ogr_col:
        out[ogr_col] = pd.to_numeric(out[ogr_col], errors="coerce")
    if channel_col:
        out["channel"] = out[channel_col].astype(str).str.strip()
    elif view_norm == "channel":
        out["channel"] = out["label"]
    else:
        out["channel"] = "Unassigned"
    keep = [
        c for c in base_cols + ["channel"] + (["ticker"] if "ticker" in out.columns else [])
        if c in out.columns
    ]
    out = out[keep].dropna(subset=["label"])
    out = out[out["label"] != ""]
    return out


def resolve_default_growth_view(gq_channel_df: pd.DataFrame, gq_ticker_df: pd.DataFrame) -> str:
    if gq_channel_df is not None and not gq_channel_df.empty:
        return "channel"
    if gq_ticker_df is not None and not gq_ticker_df.empty:
        return "ticker"
    return "channel"


def safe_render_visual(
    *,
    title: str,
    data: pd.DataFrame | None,
    render_fn: Callable[[pd.DataFrame], None],
    empty_message: str,
) -> None:
    """Single visual render guard: empty-state card + isolated render failure."""
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        render_empty_state(empty_message, "Adjust filters or date range.")
        return
    _safe_render_chart(title, lambda: render_fn(frame))


@st.cache_resource
def _cached_gateway(root: str):
    """Cached DataGateway instance (one per session)."""
    from app.data.data_gateway import DataGateway
    return DataGateway(Path(root))


def _get_gateway():
    """Return cached DataGateway for ROOT."""
    if st is not None and hasattr(st, "cache_resource"):
        return _cached_gateway(str(ROOT))
    from app.data.data_gateway import DataGateway
    return DataGateway(ROOT)


def _filter_context(state: FilterState) -> None:
    """Compact filter context at top of page."""
    st.caption(
        f"Reporting filters: {state.date_start} â†’ {state.date_end} | "
        f"analysis path: {', '.join(state.drill_path)} | "
        f"slice: {state.slice or 'â€”'} | "
        f"{state.currency or 'native'} / {state.unit or 'units'}"
    )


def _track_query(name: str) -> None:
    """Append gateway query name to session list for provenance."""
    if st is None:
        return
    if VIZ_QUERIES_KEY not in st.session_state:
        st.session_state[VIZ_QUERIES_KEY] = []
    st.session_state[VIZ_QUERIES_KEY].append(name)


def _waterfall_reconcile_from_snapshot(snapshot: Any, payload_qa: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build waterfall reconciliation for QA: prefer payload _qa.waterfall_reconcile;
    else from snapshot row compute diff = begin + nnb + market_impact - end (no recomputation of metrics).
    """
    if payload_qa is not None:
        wf = payload_qa.get("waterfall_reconcile")
        if isinstance(wf, dict):
            return wf
    try:
        import pandas as pd
        if isinstance(snapshot, dict):
            begin, end, nnb, mi = snapshot.get("begin_aum"), snapshot.get("end_aum"), snapshot.get("nnb"), snapshot.get("market_impact")
        elif isinstance(snapshot, pd.DataFrame) and not snapshot.empty:
            row = snapshot.iloc[-1]
            begin = row.get("begin_aum") if hasattr(row, "get") else (row["begin_aum"] if "begin_aum" in row.index else None)
            end = row.get("end_aum") if hasattr(row, "get") else (row["end_aum"] if "end_aum" in row.index else None)
            nnb = row.get("nnb") if hasattr(row, "get") else (row["nnb"] if "nnb" in row.index else None)
            mi = row.get("market_impact") if hasattr(row, "get") else (row["market_impact"] if "market_impact" in row.index else None)
        else:
            return {}
        if begin is not None and end is not None and nnb is not None and mi is not None:
            b, e, n, m = float(begin), float(end), float(nnb), float(mi)
            if not (b != b or e != e or n != n or m != m):  # no NaN
                diff = (b + n + m) - e
                return {"diff": diff, "ok": abs(diff) < 1e-6}
    except Exception:
        pass
    return {}


def _pick_kpi(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return the KPI item with the given key from payload["kpis"], or None."""
    for k in payload.get("kpis") or []:
        if k.get("key") == key:
            return k
    return None


def _status_dot(status: str) -> str:
    """Minimal status indicator: good=green, bad=red, neutral=gray, na=light gray. No heavy CSS."""
    color = {"good": PALETTE["positive"], "bad": PALETTE["negative"], "neutral": PALETTE["neutral"], "na": "#d1d5db"}.get(
        (status or "").lower(), "#d1d5db"
    )
    return f'<span style="color:{color};font-size:0.9em;">â—</span>'


def _safe_render_chart(name: str, render_fn: Callable[[], None]) -> None:
    """Render a chart section without crashing the whole page."""
    try:
        render_fn()
    except Exception as exc:
        render_error_state(name, exc, "Section was skipped to keep the page responsive.")


def _render_metric_tooltips() -> None:
    """Compact KPI glossary with hover tooltips for client demos."""
    st.markdown(
        (
            "<span title='Assets Under Management at period end.'><b>AUM</b></span> | "
            "<span title='Month-over-month change in end AUM.'><b>MoM Growth</b></span> | "
            "<span title='Year-to-date change in end AUM.'><b>YTD Growth</b></span> | "
            "<span title='Net New Business: subscriptions minus redemptions.'><b>NNB</b></span> | "
            "<span title='Net New Flow: net subscriptions adjusted for transfers where available.'><b>NNF</b></span> | "
            "<span title='Organic Growth Rate from net flows versus opening AUM.'><b>OGR</b></span> | "
            "<span title='AUM change not explained by flows, typically market performance and FX.'><b>Market Impact</b></span>"
        ),
        unsafe_allow_html=True,
    )


def kpi_color(value: float | None) -> str:
    if value is None:
        return COLOR_NEUTRAL
    if value > 0:
        return COLOR_POSITIVE
    if value < 0:
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


def _kpi_color_from_status(status: str | None) -> str:
    status_norm = (status or "").lower()
    if status_norm == "good":
        return COLOR_POSITIVE
    if status_norm == "bad":
        return COLOR_NEGATIVE
    return COLOR_NEUTRAL


def _kpi_numeric_value(kpi: dict[str, Any] | None) -> float | None:
    if not isinstance(kpi, dict):
        return None
    raw = kpi.get("value")
    if isinstance(raw, (int, float)) and raw == raw:
        return float(raw)
    return None


def _render_kpi_card(label: str, value: str, delta: str | None, color: str, tooltip: str) -> None:
    delta_html = f"<div style='margin-top:4px;color:{COLOR_NEUTRAL};font-size:12px'>{delta}</div>" if delta else ""
    st.markdown(
        (
            f"<div title='{tooltip}' style='padding:12px;border-radius:8px;background:{COLOR_BACKGROUND};"
            f"border-left:6px solid {color};box-shadow:0 1px 2px rgba(0,0,0,0.06);'>"
            f"<div style='font-size:12px;color:{COLOR_NEUTRAL};font-weight:600;'>{label}</div>"
            f"<div style='font-size:20px;color:{COLOR_PRIMARY};font-weight:700;line-height:1.2'>{value}</div>"
            f"{delta_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_kpi_strip(payload: dict[str, Any]) -> None:
    """
    Render Executive Summary KPI strip from payload["kpis"] only.
    """
    kpis = payload.get("kpis") or []
    if not kpis:
        return

    if payload.get("rates_not_computable_reason"):
        st.caption("Rates not computable (first month or missing prior). MoM %, OGR, and market impact show —.")

    keys = ["end_aum", "mom_growth", "ytd_growth", "nnb", "nnf", "ogr", "market_impact"]
    tooltip_map = {
        "end_aum": "Assets Under Management at period end.",
        "mom_growth": "Month-over-month growth in ending AUM.",
        "ytd_growth": "Year-to-date growth in ending AUM.",
        "nnb": "Net New Business (subscriptions minus redemptions).",
        "nnf": "Net New Flow adjusted for transfer effects where available.",
        "ogr": "Organic Growth Rate from net flows vs opening AUM.",
        "market_impact": "Performance and market effects not explained by flows.",
    }
    directional_keys = {"mom_growth", "nnb", "ogr", "market_impact"}

    cols = st.columns(7)
    for col, key in zip(cols, keys):
        kpi = _pick_kpi(payload, key)
        if kpi is None:
            with col:
                _render_kpi_card(
                    key.replace("_", " ").title(),
                    "—",
                    None,
                    COLOR_NEUTRAL,
                    tooltip_map.get(key, "KPI value."),
                )
            continue

        name = str(kpi.get("name") or key.replace("_", " ").title())
        display = str(kpi.get("display") or "—")
        status = str(kpi.get("status") or "na")
        if display == "—":
            status = "na"

        delta_raw = kpi.get("delta")
        delta_display = kpi.get("delta_display")
        if delta_display not in (None, ""):
            delta_show = str(delta_display)
        elif delta_raw is not None and isinstance(delta_raw, (int, float)) and delta_raw == delta_raw:
            try:
                delta_show = f"{delta_raw:+,.2f}"
            except Exception:
                delta_show = None
        else:
            delta_show = None

        numeric_value = _kpi_numeric_value(kpi)
        semantic_color = kpi_color(numeric_value) if key in directional_keys else _kpi_color_from_status(status)

        with col:
            _render_kpi_card(
                name,
                display,
                delta_show,
                semantic_color,
                tooltip_map.get(key, "KPI value."),
            )

    market_pnl = _pick_kpi(payload, "market_pnl")
    if market_pnl is not None:
        with st.expander("Market PnL", expanded=False):
            display = market_pnl.get("display") or "—"
            pnl_status = market_pnl.get("status") or "na"
            if display == "—":
                pnl_status = "na"
            delta_raw = market_pnl.get("delta")
            delta_display = market_pnl.get("delta_display")
            if delta_display not in (None, ""):
                delta_show = str(delta_display)
            elif delta_raw is not None and isinstance(delta_raw, (int, float)) and delta_raw == delta_raw:
                delta_show = f"{delta_raw:+,.2f}"
            else:
                delta_show = None
            _render_kpi_card(
                str(market_pnl.get("name") or "Market PnL"),
                str(display),
                delta_show,
                _kpi_color_from_status(str(pnl_status)),
                "Market PnL contribution for the selected range.",
            )
def render_sparkline(payload: dict[str, Any]) -> None:
    """
    Mini trend sparkline for End AUM (last 12 months) from payload["series"] only.
    No KPI computation; plot raw end_aum. Skip if series missing or too short.
    """
    series = payload.get("series") or {}
    month_end = series.get("month_end") or []
    end_aum = series.get("end_aum") or []
    if not month_end or not end_aum or len(month_end) < 2 or len(month_end) != len(end_aum):
        return
    try:
        import pandas as pd
        df = pd.DataFrame({"month_end": month_end, "end_aum": end_aum})
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        df = df.dropna(subset=["month_end", "end_aum"])
        if len(df) < 2:
            return
        df = df.set_index("month_end").sort_index()
        st.line_chart(df[["end_aum"]])
    except Exception:
        pass


def render_freshness_caption(payload: dict[str, Any]) -> None:
    """Data freshness row with business-facing metadata; dev details hidden unless dev_mode."""
    meta = payload.get("_meta") or {}
    context = payload.get("context") or {}
    latest_month_end = context.get("latest_month_end") or "â€”"
    query_name = meta.get("query_name") or "firm_snapshot"
    if st.session_state.get("dev_mode"):
        dataset_version = meta.get("dataset_version") or "â€”"
        st.caption(f"Snapshot: {latest_month_end}  |  Cached: {query_name}  |  dataset_version={dataset_version}")
    else:
        st.caption(f"Snapshot: {latest_month_end}  |  Source: {query_name}")


def render_exec_context(payload: dict[str, Any]) -> None:
    """Small context box: latest / prev / YTD start dates from payload context. Uses 'â€”' if missing."""
    context = payload.get("context") or {}
    latest = context.get("latest_month_end") or "â€”"
    prev = context.get("prev_month_end") or "â€”"
    ytd_start = context.get("ytd_start_month_end") or "â€”"
    st.caption(f"Latest: {latest}  |  Prev: {prev}  |  YTD start: {ytd_start}")


def _render_exec_debug_expander(payload: dict[str, Any]) -> None:
    """Dev-only expander: _meta, context, list of KPI keys. Shown when st.session_state.get('dev_mode')."""
    if not st.session_state.get("dev_mode"):
        return
    with st.expander("Executive Summary (debug)", expanded=False):
        st.json(payload.get("_meta") or {})
        st.json(payload.get("context") or {})
        kpis = payload.get("kpis") or []
        keys = [k.get("key") for k in kpis if isinstance(k, dict)]
        st.text("KPI keys: " + ", ".join(str(k) for k in keys))


def render(state: FilterState, contract: dict[str, Any]) -> None:
    """Render Executive Dashboard tab: gateway-driven KPIs and charts; drill-down controls; no groupby."""
    st.subheader("Institutional Asset Management Analytics Platform")
    last_updated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"Last updated: {last_updated_utc}")
    _filter_context(state)

    # Reset query log for this render
    if VIZ_QUERIES_KEY in st.session_state:
        st.session_state[VIZ_QUERIES_KEY] = []

    gateway = _get_gateway()

    payload: dict[str, Any] = {}
    container_kpi = st.container()
    # ---- Firm Snapshot (no KPI formulas in page) ----
    try:
        from app.queries.firm_snapshot import get_firm_snapshot_cached
        dataset_version = gateway.get_dataset_version()
        filter_state_hash = state.filter_state_hash()
        with container_kpi:
            st.subheader("Executive Snapshot")
            payload = get_firm_snapshot_cached(
                months=24,
                dataset_version=dataset_version,
                filter_state_hash=filter_state_hash,
            )
            if not payload or not payload.get("kpis"):
                if payload and payload.get("coverage_incomplete"):
                    render_empty_state(
                        "Month coverage incomplete for selected filters.",
                        "Widen date range or ensure data exists for all months in range.",
                    )
                else:
                    render_empty_state(
                        "No firm snapshot data available for selected range.",
                        "Widen date range or relax filters.",
                    )
            else:
                render_freshness_caption(payload)
                render_kpi_strip(payload)
                _render_metric_tooltips()
                if payload.get("coverage_incomplete"):
                    st.caption(
                        "Month coverage is incomplete for the selected filters; some rates may show as unavailable."
                    )
                # Compact row: left = sparkline, right = context box
                row_c1, row_c2 = st.columns([3, 1])
                with row_c1:
                    render_sparkline(payload)
                with row_c2:
                    render_exec_context(payload)
                # Executive commentary: deterministic narrative from payload only (no metric math)
                st.markdown("#### Executive Insights")
                narrative_text = build_firm_narrative_text(payload) if payload else ""
                if narrative_text:
                    st.markdown(narrative_text)
                else:
                    st.caption("Narrative not available for the selected range.")
                if st.session_state.get("dev_mode"):
                    with st.expander("Narrative inputs (debug)", expanded=False):
                        st.json(payload.get("raw") or {})
                        st.json(payload.get("context") or {})
                _render_exec_debug_expander(payload)
    except Exception:
        render_empty_state(
            "No firm snapshot data available for selected range.",
            "Widen date range or relax filters.",
        )

    top_end_aum = "â€”"
    as_of_date = state.date_end
    if payload:
        end_kpi = _pick_kpi(payload, "end_aum")
        if end_kpi is not None:
            top_end_aum = end_kpi.get("display") or "â€”"
        as_of_date = (payload.get("context") or {}).get("latest_month_end") or state.date_end
    badge1, badge2, badge3, badge4 = st.columns([1.2, 1.2, 2.0, 1.1])
    with badge1:
        st.caption("Reporting Period")
        st.markdown(f"**{state.date_start} â†’ {state.date_end}**")
    with badge2:
        st.caption("Total AUM (As of)")
        st.markdown(f"**{top_end_aum}**  \nAs of `{as_of_date}`")
    with badge3:
        st.caption("Portfolio Slice")
        st.markdown(f"**{state.slice or 'All'}**  \nPath: `{', '.join(state.drill_path)}`")
    with badge4:
        st.caption("Coverage Context")
        st.markdown("**Firm View**")

    # Use state passed from app (single source of truth: st.session_state["filters"])

    # ---- Drill path (order); slice/filter is at top in "Explore Data" bar ----
    gf = (contract or {}).get("global_filters") or {}
    dp_cfg = gf.get("drill_path") or {}
    dim_options = (
        dp_cfg.get("options")
        or dp_cfg.get("dimensions")
        or dp_cfg.get("default")
        or DEFAULT_DIM_OPTIONS
    )
    if not isinstance(dim_options, list):
        dim_options = list(DEFAULT_DIM_OPTIONS)
    dim_options = [str(x) for x in dim_options]
    current_path = [x for x in state.drill_path if x in dim_options]
    if not current_path:
        current_path = dim_options[:3] if len(dim_options) >= 3 else dim_options

    with st.expander("Drill path (hierarchy order)", expanded=False):
        st.caption("Analysis path configuration. Use «Explore Data» at the top to filter by Channel, Geography, Product, or Segment.")
        new_path = st.multiselect(
            "Drill path (order = hierarchy)",
            options=dim_options,
            default=current_path,
            key="viz_drill_path",
        )
        if new_path and new_path != current_path:
            update_filter_state(drill_path=new_path)
            state = get_filter_state()
        elif not new_path and current_path:
            update_filter_state(drill_path=current_path)
            state = get_filter_state()

    st.divider()

    # ---- Tab 1: single fetch, then render in order (waterfall â†’ treemap|matrix â†’ trend) ----
    filters = state
    with st.spinner("Loadingâ€¦"):
        snapshot_df = get_firm_snapshot(filters, root=ROOT)
        trend_df = get_trend_series(filters, root=ROOT)
        channel_df_aum = get_channel_breakdown(filters, metric="end_aum", root=ROOT)
        channel_df_nnb = get_channel_breakdown(filters, metric="nnb", root=ROOT)
        # Growth-quality sources (same filtered dataset as visuals; used for default view + matrix + ranked tables)
        _gq_channel_df = get_growth_quality(filters, view="channel", root=ROOT)
        _gq_ticker_df = get_growth_quality(filters, view="ticker", root=ROOT)

    snapshot_df = snapshot_df if isinstance(snapshot_df, pd.DataFrame) else pd.DataFrame()
    trend_df = trend_df if isinstance(trend_df, pd.DataFrame) else pd.DataFrame()
    channel_df_aum = channel_df_aum if isinstance(channel_df_aum, pd.DataFrame) else pd.DataFrame()
    channel_df_nnb = channel_df_nnb if isinstance(channel_df_nnb, pd.DataFrame) else pd.DataFrame()
    channel_df = channel_df_aum.copy() if not channel_df_aum.empty else channel_df_nnb.copy()
    if "nnb" not in channel_df.columns and not channel_df_nnb.empty and "channel" in channel_df_nnb.columns and "nnb" in channel_df_nnb.columns:
        _nnb_join = (
            channel_df_nnb[["channel", "nnb"]]
            .copy()
            .groupby("channel", as_index=False)["nnb"]
            .sum()
        )
        if "channel" in channel_df.columns:
            channel_df = channel_df.merge(_nnb_join, on="channel", how="left")
    _gq_channel_df = _gq_channel_df if isinstance(_gq_channel_df, pd.DataFrame) else pd.DataFrame()
    _gq_ticker_df = _gq_ticker_df if isinstance(_gq_ticker_df, pd.DataFrame) else pd.DataFrame()
    gq_channel_df = prepare_growth_quality_dataset(_gq_channel_df, view="channel")
    gq_ticker_df = prepare_growth_quality_dataset(_gq_ticker_df, view="ticker")
    gq_df = pd.DataFrame()
    user_changed_filters = bool(st.session_state.get("ui_filters_user_changed", False))
    missing_sections = []
    if snapshot_df.empty:
        missing_sections.append("snapshot")
    if trend_df.empty:
        missing_sections.append("trend")
    if channel_df.empty:
        missing_sections.append("channels")
    if gq_channel_df.empty and gq_ticker_df.empty:
        missing_sections.append("tickers")
    if missing_sections and (user_changed_filters or len(missing_sections) == 4):
        st.markdown(
            (
                "<div class='availability-note'>"
                "<strong>Availability summary:</strong> "
                + ", ".join(missing_sections)
                + " unavailable for this selection. The dashboard is showing sections with valid data."
                + "</div>"
            ),
            unsafe_allow_html=True,
        )
    coverage_labels = {
        "snapshot": "KPI snapshot",
        "trend": "AUM trend",
        "channels": "distribution mix",
        "tickers": "growth matrix/movers",
    }
    if missing_sections:
        missing_readable = ", ".join(coverage_labels.get(s, s) for s in missing_sections)
        st.caption(f"Data coverage note: unavailable under current filters -> {missing_readable}.")
    else:
        st.caption("Data coverage note: full coverage for KPI snapshot, trend, distribution, and growth/movers sections.")
    # Ranked tables: canonical source for drill selectbox options (metric: AUM or NNB)
    rank_metric = st.session_state.get("tab1_rank_metric", "AUM")
    rank_metric = st.selectbox(
        "Movers ranked by",
        options=["AUM", "NNB"],
        index=0 if rank_metric == "AUM" else 1,
        key="tab1_rank_metric",
    )
    top_n = st.slider("Top N entities", min_value=5, max_value=50, value=25, step=5, key="tab1_top_n")
    try:
        ranked_channels = build_ranked_channels(
            channel_df if channel_df is not None else pd.DataFrame(),
            metric=rank_metric,
            top_n=top_n,
        )
        ranked_tickers = build_ranked_tickers(
            gq_ticker_df if gq_ticker_df is not None else pd.DataFrame(),
            metric=rank_metric,
            top_n=top_n,
        )
    except ValueError as e:
        render_empty_state(str(e), "Check filters or data mappings.", icon="ERROR")
        ranked_channels = pd.DataFrame(columns=["rank", "channel", "NNB", "AUM"])
        ranked_tickers = pd.DataFrame(columns=["rank", "ticker", "NNB", "AUM"])

    insight_sections = build_executive_insight_sections(
        snapshot_df=snapshot_df,
        ranked_channels=ranked_channels,
        ranked_tickers=ranked_tickers,
    )
    insight_text = ""
    for section in insight_sections:
        if getattr(section, "sentences", None):
            insight_text = " ".join(section.sentences[:2]).strip()
            if insight_text:
                break
    if not insight_text:
        insight_text = (
            "Asset growth dynamics are mixed across channels and products. "
            "Use distribution mix, growth matrix, and movers to isolate concentration and momentum drivers."
        )
    st.markdown(
        (
            f"<div style='background:#eef4ff;border-left:6px solid {COLOR_SECONDARY};"
            "padding:16px;border-radius:8px;margin-bottom:20px;'>"
            "<b>Key Insight</b><br>"
            f"{insight_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    # Canonical option lists from ranked tables (drill options match what user sees)
    channel_options = ranked_channels["channel"].tolist() if not ranked_channels.empty else []
    ticker_options = ranked_tickers["ticker"].tolist() if not ranked_tickers.empty else []
    available_channels = set(channel_options) if channel_options else None
    available_tickers = set(ticker_options) if ticker_options else None
    drill = revalidate_drill_on_filter_change(
        state.filter_state_hash(),
        available_channels=available_channels,
        available_tickers=available_tickers,
    )

    # Reset banner when selection was reset due to filter change
    msg = st.session_state.get(DRILL_RESET_FLAG_KEY)
    if msg:
        render_empty_state(msg, "Pick another channel/ticker or widen date range.")
        st.session_state[DRILL_RESET_FLAG_KEY] = None

    # Optional: Plotly click-to-select (progressive enhancement)
    plotly_events = try_import_plotly_events()
    has_plotly_events = plotly_events is not None
    st.session_state["cap_plotly_events"] = has_plotly_events

    # Validate drill selection each render; show empty state if reset
    valid_channels = available_channels or set()
    valid_tickers = available_tickers or set()
    ok, _ = validate_drill_selection(valid_channels, valid_tickers)
    if not ok:
        render_empty_state(
            "Selection reset (not available under current filters).",
            "Pick another channel/ticker or widen date range.",
            icon="â„¹ï¸",
        )

    # Drill UI above ranked tables (true state source: DrillState in session_state)
    drill = get_drill_state()
    mode_labels = ["Channel", "Ticker"]
    mode_values = ["channel", "ticker"]
    radio_idx = 0 if drill.drill_mode == "channel" else 1
    chosen_label = st.radio(
        "Drill focus",
        mode_labels,
        horizontal=True,
        index=radio_idx,
        key="drill_mode",
    )
    chosen_mode = mode_values[mode_labels.index(chosen_label)]
    if chosen_mode != drill.drill_mode:
        set_drill_mode(chosen_mode)
        drill = get_drill_state()

    drill_options = ["(All)"] + channel_options if drill.drill_mode == "channel" else ["(All)"] + ticker_options
    current_val = drill.selected_channel if drill.drill_mode == "channel" else drill.selected_ticker
    select_idx = 0 if current_val is None else (drill_options.index(current_val) if current_val in drill_options else 0)
    select_label = "Select distribution channel" if drill.drill_mode == "channel" else "Select product ticker"
    sel = st.selectbox(
        select_label,
        options=drill_options,
        index=select_idx,
        key="drill_select",
    )
    if drill.drill_mode == "channel":
        if sel == "(All)":
            if current_val is not None or drill.details_level != "firm":
                update_drill_state(details_level="firm")
                set_selected_channel(None)
        elif sel != current_val:
            set_selected_channel(sel)
    else:
        if sel == "(All)":
            if current_val is not None or drill.details_level != "firm":
                update_drill_state(details_level="firm")
                set_selected_ticker(None)
        elif sel != current_val:
            set_selected_ticker(sel)
    if st.button("Reset Selection", key="drill_reset_btn"):
        update_drill_state(details_level="firm")
        set_selected_channel(None)
        set_selected_ticker(None)

    # Click-to-select toggle only when streamlit-plotly-events is installed
    if has_plotly_events:
        st.toggle("Click-to-select on charts", value=False, key="enable_chart_drill")

    # Timeout/budget: if any Tab 1 gateway call went over budget, show safe message (charts still degrade to table)
    _log = st.session_state.get("perf_query_log") or []
    _is_chart_op = lambda n: (
        n in {"get_firm_snapshot", "get_trend_series"}
        or (n or "").startswith("get_channel_breakdown")
        or (n or "").startswith("get_growth_quality")
    )
    over_budget_chart_ops = [
        e for e in _log[-20:]
        if ("over_budget" in str(e.get("warning") or "")) and _is_chart_op(e.get("name"))
    ]
    # Startup should stay curated; only surface timeout warning after explicit user filter actions.
    if user_changed_filters and over_budget_chart_ops:
        render_timeout_state("Chart query", HEAVY_BUDGET_MS, "Narrow filters or reduce dimensions.")

    # Institutional layout containers
    container_trend = st.container()
    container_distribution = st.container()
    container_movers = st.container()

    with container_kpi:
        st.markdown("#### KPI Snapshot")
        _drill = get_drill_state()
        _src = _drill.selection_source or "â€”"
        st.caption(
            f"Active selection: Channel={_drill.selected_channel or 'All'} | Ticker={_drill.selected_ticker or 'All'} | source={_src}"
        )

    snapshot_for_waterfall = (
        snapshot_df.tail(1) if (snapshot_df is not None and not snapshot_df.empty) else None
    )

    with container_trend:
        st.markdown("#### AUM Trend")
        st.caption("Hover the chart to inspect exact monthly values and compare trend vs volatility band.")
        _trend_df = trend_df if (trend_df is not None and not trend_df.empty) else pd.DataFrame()
        trend_period_mode = st.radio(
            "Trend window",
            options=["1M", "3M", "YTD", "12M", "Full"],
            horizontal=True,
            key="tab1_trend_period_mode",
        )
        def _apply_trend_window(df: pd.DataFrame, mode: str) -> pd.DataFrame:
            if df is None or df.empty or "month_end" not in df.columns:
                return df
            out = df.copy()
            out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
            out = out.dropna(subset=["month_end"]).sort_values("month_end")
            if out.empty:
                return out
            if mode == "1M":
                return out.tail(1)
            if mode == "3M":
                return out.tail(3)
            if mode == "12M":
                return out.tail(12)
            if mode == "YTD":
                current_year = int(out["month_end"].max().year)
                return out[out["month_end"].dt.year == current_year]
            return out
        _trend_df = _apply_trend_window(_trend_df, trend_period_mode)
        trend_fallback_cols = [c for c in ("month_end", "end_aum", "nnb", "aum") if c in _trend_df.columns] or list(_trend_df.columns)[:5]
        trend_required_cols = ["month_end", "ogr", "market_impact_rate"]

        def _draw_trend():
            fig_trend = render_growth_trend(_trend_df)
            if go is not None:
                fig_trend.update_traces(
                    line=dict(color=COLOR_PRIMARY, width=3),
                    marker=dict(color=COLOR_SECONDARY),
                    fill="tozeroy",
                    fillcolor="rgba(76,126,219,0.15)",
                )
            safe_render_plotly(fig_trend, user_message="Trend chart unavailable for this selection.")

        trend_missing_cols = missing_required_columns(_trend_df, trend_required_cols)
        if trend_missing_cols:
            render_empty_state(
                f"Growth trend unavailable: missing required columns {trend_missing_cols}.",
                "Check upstream data mappings or relax filters.",
                icon="ERROR",
            )
        else:
            _safe_render_chart(
                "Growth Trend",
                lambda: render_chart_or_fallback(
                    _draw_trend,
                    _trend_df,
                    trend_fallback_cols,
                    fallback_note("insufficient_trend", {"min_points": 2}),
                    min_points=2,
                    empty_reason="No trend data for the selected range.",
                    empty_hint="Widen date range or relax filters.",
                ),
            )
        if go is not None and not _trend_df.empty and "month_end" in _trend_df.columns and "end_aum" in _trend_df.columns and len(_trend_df) >= 4:
            trend_band = _trend_df[["month_end", "end_aum"]].copy().sort_values("month_end")
            trend_band["aum_roll"] = trend_band["end_aum"].rolling(window=3, min_periods=2).mean()
            trend_band["aum_std"] = trend_band["end_aum"].rolling(window=3, min_periods=2).std()
            trend_band["upper"] = trend_band["aum_roll"] + trend_band["aum_std"]
            trend_band["lower"] = trend_band["aum_roll"] - trend_band["aum_std"]
            fig_band = go.Figure()
            fig_band.add_trace(
                go.Scatter(
                    x=trend_band["month_end"],
                    y=trend_band["end_aum"],
                    mode="lines+markers",
                    name="AUM",
                    line=dict(color=COLOR_PRIMARY, width=3),
                    marker=dict(color=COLOR_SECONDARY, size=7),
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=trend_band["month_end"],
                    y=trend_band["aum_roll"],
                    mode="lines",
                    name="AUM 3M rolling avg",
                    line=dict(color=COLOR_SECONDARY, width=2),
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=trend_band["month_end"],
                    y=trend_band["upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig_band.add_trace(
                go.Scatter(
                    x=trend_band["month_end"],
                    y=trend_band["lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(76,126,219,0.15)",
                    name="Volatility band (rolling std)",
                    hoverinfo="skip",
                )
            )
            fig_band.update_layout(height=320, margin=dict(l=30, r=30, t=30, b=30))
            apply_enterprise_plotly_style(fig_band, height=320)
            safe_render_plotly(fig_band, user_message="Volatility band unavailable for this selection.")

    treemap_metric = st.session_state.get("tab1_treemap_metric", "aum")

    EPS = 1e-9
    enable_chart_drill = bool(has_plotly_events and st.session_state.get("enable_chart_drill"))

    with container_distribution:
        st.markdown("#### Distribution Mix")
        treemap_metric = st.radio(
            "Channel contribution metric",
            options=["aum", "nnb"],
            horizontal=True,
            key="tab1_treemap_metric",
        )
        treemap_df = channel_df_nnb if treemap_metric == "nnb" else channel_df_aum
        treemap_df = treemap_df if (treemap_df is not None and not treemap_df.empty) else pd.DataFrame()
        metric_col = "nnb" if treemap_metric == "nnb" else "end_aum"
        metric_title = "Channel Contribution (Net New Business)" if treemap_metric == "nnb" else "Channel Contribution (AUM)"
        metric_subtitle = (
            "Shows channel-level contribution to net new business over the selected period."
            if treemap_metric == "nnb"
            else "Shows relative channel concentration by ending assets under management."
        )
        st.caption(metric_subtitle)

        channel_col = "channel" if "channel" in treemap_df.columns else None
        _work = pd.DataFrame()
        if channel_col and metric_col in treemap_df.columns:
            _work = treemap_df[[channel_col, metric_col]].copy()
            _work[channel_col] = _work[channel_col].astype(str).str.strip()
            _work[metric_col] = pd.to_numeric(_work[metric_col], errors="coerce")
            _work = _work.dropna(subset=[channel_col, metric_col])
            _work = _work[_work[channel_col] != ""]
            _work = _work.groupby(channel_col, as_index=False)[metric_col].sum()
            if treemap_metric == "aum":
                _work = _work[_work[metric_col] > 0]
            else:
                _work = _work[_work[metric_col] != 0]

        if _work.empty:
            _safe_render_chart(
                "Treemap",
                lambda: render_empty_state(
                    "No channel contribution data available for this metric.",
                    "Adjust filters or date range.",
                ),
            )
        elif treemap_metric == "nnb" and abs(float(_work[metric_col].abs().sum())) < EPS:
            _note = "No meaningful net new business contribution is available for the selected period."
            st.markdown(f"<div class='availability-note'><strong>Coverage note:</strong> {_note}</div>", unsafe_allow_html=True)
        elif _work[channel_col].nunique() == 1:
            only_channel = str(_work[channel_col].iloc[0])
            st.markdown(
                (
                    "<div class='availability-note'><strong>Concentration:</strong> "
                    f"Distribution is fully concentrated in a single channel ({only_channel}) for the selected slice."
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if go is not None:
                fig_one = go.Figure(
                    go.Bar(
                        x=[float(_work[metric_col].iloc[0])],
                        y=[only_channel],
                        orientation="h",
                        marker=dict(color=PALETTE["primary"] if treemap_metric == "aum" else (PALETTE["positive"] if float(_work[metric_col].iloc[0]) >= 0 else PALETTE["negative"])),
                        hovertemplate=f"Channel: %{{y}}<br>{metric_col}: %{{x:,.2f}}<extra></extra>",
                    )
                )
                fig_one.update_layout(
                    title=metric_title,
                    xaxis_title=metric_col,
                    yaxis_title="",
                    height=220,
                    margin=dict(l=20, r=20, t=42, b=20),
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                )
                apply_enterprise_plotly_style(fig_one, height=220)
                safe_render_plotly(fig_one, user_message="Single-channel distribution view unavailable.")
        else:
            fallback_treemap_cols = list(_work.columns)[:8]

            def _draw_treemap():
                fig_treemap = render_channel_treemap(_work, treemap_metric)
                if enable_chart_drill and plotly_events is not None:
                    events = plotly_events(
                        fig_treemap,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        override_height=360,
                    )
                    events = events if isinstance(events, list) else []
                    if events:
                        click_hash = hashlib.sha1(
                            json.dumps(events, sort_keys=True, default=str).encode()
                        ).hexdigest()
                        last_hash = st.session_state.get("last_chart_click_hash")
                        if click_hash != last_hash:
                            st.session_state["last_chart_click_hash"] = click_hash
                            label = parse_treemap_click(events)
                            if label:
                                set_drill_mode("channel")
                                set_selected_channel(label, source="chart")
                else:
                    safe_render_plotly(fig_treemap, user_message="Treemap unavailable for this selection.")

            _safe_render_chart(
                "Treemap",
                lambda: render_chart_or_fallback(
                    _draw_treemap,
                    _work,
                    fallback_treemap_cols or ["channel", metric_col],
                    fallback_note("insufficient_points", {"min_points": 2}),
                    min_points=2,
                    empty_reason="No channel data for distribution mix.",
                    empty_hint="Widen date range or relax filters.",
                ),
            )

    with container_movers:
        st.markdown("#### Movers (Products/Channels)")
        st.caption("Hover bars and points to review top contributors, laggards, and impact ratios.")
        tab_chan, tab_tick = st.tabs(["Distribution Channel Mix", "Product Leaders & Laggards"])
        with tab_chan:
            if ranked_channels.empty:
                render_empty_state("No channel data for the current filters.", "Widen date range or relax filters.")
            else:
                _render_ranked_kpi_cards(ranked_channels, "channel", "Channel contribution snapshot")
                cc1, cc2 = st.columns(2)
                with cc1:
                    _render_top_bottom_bar(ranked_channels, "channel", "Top/Bottom Channel Contributors by NNB", top_n=6)
                with cc2:
                    _render_growth_impact_scatter(ranked_channels.head(12), "channel", "Channel Growth vs Market Impact")
                render_export_buttons(ranked_channels, None, "tab1_top_channels")
        with tab_tick:
            if ranked_tickers.empty:
                render_empty_state("No ticker data for the current filters.", "Widen date range or relax filters.")
            else:
                _render_ranked_kpi_cards(ranked_tickers, "ticker", "Product/ETF contribution snapshot")
                tc1, tc2 = st.columns(2)
                with tc1:
                    _render_top_bottom_bar(ranked_tickers, "ticker", "Top/Bottom Product Contributors by NNB", top_n=6)
                with tc2:
                    _render_growth_impact_scatter(ranked_tickers.head(12), "ticker", "Product Growth vs Market Impact")
                render_export_buttons(ranked_tickers, None, "tab1_top_tickers")

    with st.expander("Guided Analytics", expanded=False):
        st.markdown("#### Net Flow Decomposition")
        def _render_waterfall() -> None:
            fig = render_aum_waterfall(snapshot_for_waterfall)
            safe_render_plotly(fig, user_message="Waterfall unavailable for this selection.")
        if snapshot_for_waterfall is None or snapshot_for_waterfall.empty:
            render_empty_state("No waterfall available for current selection.", "Adjust filters or date range.")
        else:
            _safe_render_chart("Waterfall", _render_waterfall)

        st.markdown("#### Clean Details Panel")
        render_details_panel(state, get_drill_state(), gateway)

        st.markdown("#### Supplemental KPI Monitor")
        kpi_data = gateway.kpi_firm_global(state)
        _track_query("kpi_firm_global")
        kpis = kpis_from_gateway_dict(kpi_data)
        render_kpi_row(kpis)

        st.subheader("AUM Trend (Supplemental)")
        df_aum = gateway.chart_aum_trend(state)
        _track_query("chart_aum_trend")
        df_aum = df_aum if (df_aum is not None and not df_aum.empty) else pd.DataFrame()
        aum_fallback = [c for c in ("month_end", "date", "end_aum", "aum") if c in df_aum.columns] or list(df_aum.columns)[:5]
        def _draw_aum():
            fig_aum = fig_aum_over_time(df_aum)
            if go is not None:
                fig_aum.update_traces(
                    line=dict(color=COLOR_PRIMARY, width=3),
                    marker=dict(color=COLOR_SECONDARY),
                    fill="tozeroy",
                    fillcolor="rgba(76,126,219,0.15)",
                )
            safe_render_plotly(fig_aum, user_message="AUM trend unavailable for this selection.")
        render_chart_or_fallback(
            _draw_aum,
            df_aum,
            aum_fallback,
            fallback_note("insufficient_trend", {"min_points": 2}),
            min_points=2,
            empty_reason="No AUM trend for the selected range.",
            empty_hint="Widen date range or relax filters.",
        )

        st.subheader("Net New Business Trend (Supplemental)")
        df_nnb = gateway.chart_nnb_trend(state)
        _track_query("chart_nnb_trend")
        df_nnb = df_nnb if (df_nnb is not None and not df_nnb.empty) else pd.DataFrame()
        nnb_fallback = [c for c in ("month_end", "date", "nnb") if c in df_nnb.columns] or list(df_nnb.columns)[:5]
        def _draw_nnb():
            safe_render_plotly(lambda: fig_nnb_trend(df_nnb), user_message="NNB trend unavailable for this selection.")
        render_chart_or_fallback(
            _draw_nnb,
            df_nnb,
            nnb_fallback,
            fallback_note("insufficient_trend", {"min_points": 2}),
            min_points=2,
            empty_reason="No NNB trend for the selected range.",
            empty_hint="Widen date range or relax filters.",
        )

        st.subheader("Net Flow Decomposition (Supplemental)")
        decomp = gateway.growth_decomposition_inputs(state)
        _track_query("growth_decomposition_inputs")
        if not decomp:
            render_empty_state("No growth decomposition data for the selected range.", "Widen date range or relax filters.")
        else:
            def _draw_decomp() -> None:
                fig = fig_waterfall(decomp)
                safe_render_plotly(fig, user_message="Net flow decomposition chart unavailable for this selection.")
            _safe_render_chart("Net flow decomposition", _draw_decomp)
    # ---- Dev-only QA / Debug ----
    dev_mode = st.session_state.get("dev_mode", False)
    if dev_mode:
        dataset_version = st.session_state.get("dataset_version") or gateway.get_dataset_version()
        filter_state_hash = state.filter_state_hash()
        filters_json = state.to_dict()
        payload_qa = None  # Tab 1 uses gateway DataFrame; no payload _qa here
        waterfall = _waterfall_reconcile_from_snapshot(snapshot_for_waterfall, payload_qa)
        drill_state = get_drill_state()
        diagnostics = {
            "filters": filters_json,
            "rows": {
                "snapshot_df": int(len(snapshot_df)) if snapshot_df is not None else 0,
                "trend_df": int(len(trend_df)) if trend_df is not None else 0,
                "channel_df": int(len(channel_df)) if channel_df is not None else 0,
                "growth_quality_df": int(len(gq_df)) if gq_df is not None else 0,
                "ranked_channels": int(len(ranked_channels)) if ranked_channels is not None else 0,
                "ranked_tickers": int(len(ranked_tickers)) if ranked_tickers is not None else 0,
            },
            "snapshot_for_waterfall_empty": bool(snapshot_for_waterfall is None or snapshot_for_waterfall.empty),
            "columns": {
                "snapshot_df": list(snapshot_df.columns) if snapshot_df is not None else [],
                "trend_df": list(trend_df.columns) if trend_df is not None else [],
                "channel_df": list(channel_df.columns) if channel_df is not None else [],
                "growth_quality_df": list(gq_df.columns) if gq_df is not None else [],
            },
            "drill_state": {
                "mode": drill_state.drill_mode,
                "selected_channel": drill_state.selected_channel,
                "selected_ticker": drill_state.selected_ticker,
                "details_level": drill_state.details_level,
                "selection_source": drill_state.selection_source,
            },
        }
        with st.expander("QA / Debug", expanded=False):
            st.write("dataset_version:", dataset_version)
            st.write("filter_state_hash:", filter_state_hash)
            st.json(diagnostics)
            st.json(waterfall)
            cache_hits = st.session_state.get("cache_hits")
            cache_misses = st.session_state.get("cache_misses")
            if cache_hits is not None or cache_misses is not None:
                st.json({"cache_hits": cache_hits or {}, "cache_misses": cache_misses or {}})
            else:
                st.caption("Cache hit/miss not tracked.")

    # ---- Data provenance ----
    if st.session_state.get("dev_mode"):
        st.divider()
        with st.expander("Operational Metadata", expanded=False):
            st.text(f"dataset_version: {gateway.get_dataset_version()}")
            st.text(f"filter_state_hash: {state.filter_state_hash()}")
            queries = st.session_state.get(VIZ_QUERIES_KEY, [])
            st.text("Gateway queries executed:")
            st.code("\n".join(queries) if queries else "(none)", language="text")

    st.caption("Visuals refresh when reporting period, drill path, or portfolio slice changes.")

    render_observability_panel(filters=state, drill_state=get_drill_state(), queryspec=None)
    render_obs_panel(contract.get("tab_id", "visualisations"))

