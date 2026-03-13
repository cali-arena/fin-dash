"""
Tab 2: Dynamic Report - fully deterministic Investment Commentary.
No LLM: all narrative from Python-calculated metrics and conditional templates (app.reporting.nlg_templates).
Six sections: Executive Overview, Channel Analysis, Product and ETF Analysis, Geographic Analysis,
Anomalies and Flags, Recommendations. Export: HTML, Markdown, PDF.
"""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Exclude from chart aggregation only (so "Unassigned" / "—" / blank do not appear as bar labels)
CHART_EXCLUDE_LABELS = frozenset({"", "Unassigned", "—", "nan"})

from app.data.data_gateway import HEAVY_BUDGET_MS
from app.config.tab1_defaults import (
    TAB1_DEFAULT_CHANNEL,
    TAB1_DEFAULT_COUNTRY,
    TAB1_DEFAULT_PERIOD,
    TAB1_DEFAULT_PRODUCT_TICKER,
    TAB1_DEFAULT_SALES_FOCUS,
    TAB1_DEFAULT_SUB_CHANNEL,
    TAB1_DEFAULT_SUB_SEGMENT,
    get_scope_label_from_state,
)
from app.components.kpis import (
    build_executive_overview_primary_kpis,
    build_executive_overview_secondary_kpis,
    render_kpi_row,
)
from app.metrics.shared_payload import build_metric_payload
from app.reporting.html_export import _safe_filename, build_report_html
from app.reporting.reconciliation import run_reconciliation
from app.reporting.report_engine import (
    SectionOutput,
    render_overview,
    render_recommendations,
)
from app.state import FilterState, get_filter_state
from app.ui.exports import render_export_buttons
from app.ui.formatters import fmt_currency_kpi, fmt_percent, format_df, infer_common_formats
from app.ui.guardrails import render_error_state, render_timeout_state

try:
    from app.export_utils import make_pdf_with_footer
except ImportError:
    make_pdf_with_footer = None
from app.ui.observability import render_observability_panel
from app.ui.theme import PALETTE, apply_enterprise_plotly_style

try:
    from app.observability import render_obs_panel
except ImportError:
    def render_obs_panel(_tab_id: str) -> None:
        pass

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def _tab1_snapshot_for_parity() -> dict[str, Any]:
    """Tab 1 state for report parity; Segment filter removed — source is always Fixed Income."""
    return {
        "tab1_period": st.session_state.get("tab1_period", TAB1_DEFAULT_PERIOD),
        "tab1_filter_channel": st.session_state.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL),
        "tab1_filter_sub_channel": st.session_state.get("tab1_filter_sub_channel", TAB1_DEFAULT_SUB_CHANNEL),
        "tab1_filter_country": st.session_state.get("tab1_filter_country", TAB1_DEFAULT_COUNTRY),
        "tab1_filter_sub_segment": st.session_state.get("tab1_filter_sub_segment", TAB1_DEFAULT_SUB_SEGMENT),
        "tab1_filter_sales_focus": st.session_state.get("tab1_filter_sales_focus", TAB1_DEFAULT_SALES_FOCUS),
        "tab1_filter_ticker": st.session_state.get("tab1_filter_ticker", TAB1_DEFAULT_PRODUCT_TICKER),
    }


def _overview_bullets_from_payload(payload: dict[str, Any], *, period: str) -> list[str]:
    end_aum = fmt_currency_kpi(payload.get("end_aum"))
    begin_aum = fmt_currency_kpi(payload.get("begin_aum"))
    nnb = fmt_currency_kpi(payload.get("nnb"))
    nnf = fmt_currency_kpi(payload.get("nnf"))
    market = fmt_currency_kpi(payload.get("market_pnl"))
    ogr = fmt_percent(payload.get("ogr"), decimals=2, signed=True)
    market_rate = fmt_percent(payload.get("market_impact"), decimals=2, signed=True)
    fee_yield = fmt_percent(payload.get("fee_yield"), decimals=2, signed=True)
    return [
        f"{period} snapshot: Selected Scope End AUM {end_aum}, Begin AUM {begin_aum}.",
        f"Net flows: NNB {nnb}, NNF {nnf}.",
        f"Growth decomposition: OGR {ogr}, Market Impact {market_rate} ({market}).",
        f"Fee Yield: {fee_yield}.",
    ]


def _build_dim_rank(df: pd.DataFrame, dim_col: str) -> pd.DataFrame:
    if df is None or df.empty or dim_col not in df.columns:
        return pd.DataFrame(columns=[dim_col, "nnb", "nnf", "end_aum", "share"])
    out = (
        df.groupby(dim_col, as_index=False)[["nnb", "nnf", "end_aum"]]
        .sum(min_count=1)
        .sort_values("nnb", ascending=False)
        .reset_index(drop=True)
    )
    total_aum = float(pd.to_numeric(out["end_aum"], errors="coerce").fillna(0).sum())
    out["share"] = out["end_aum"] / total_aum if total_aum else float("nan")
    return out


def _section_bullets_from_rank(df: pd.DataFrame, label_col: str, label_name: str) -> list[str]:
    if df is None or df.empty or label_col not in df.columns:
        return [f"{label_name} analysis is unavailable for the selected scope."]
    work = df.copy()
    work["nnb"] = pd.to_numeric(work.get("nnb"), errors="coerce").fillna(0.0)
    work["end_aum"] = pd.to_numeric(work.get("end_aum"), errors="coerce").fillna(0.0)
    if work.empty:
        return [f"{label_name} analysis is unavailable for the selected scope."]
    sorted_leader = work.sort_values(["nnb", label_col], ascending=[False, True])
    sorted_laggard = work.sort_values(["nnb", label_col], ascending=[True, True])
    if sorted_leader.empty or sorted_laggard.empty:
        return [f"{label_name} analysis is unavailable for the selected scope."]
    leader = sorted_leader.iloc[0]
    laggard = sorted_laggard.iloc[0]
    return [
        f"Top {label_name.lower()} by NNB: {leader[label_col]} ({fmt_currency_kpi(leader['nnb'], decimals=2)}).",
        f"Weakest {label_name.lower()} by NNB: {laggard[label_col]} ({fmt_currency_kpi(laggard['nnb'], decimals=2)}).",
    ]


def _anomaly_bullets_from_shared(anomalies_df: pd.DataFrame | None) -> list[str]:
    """Build anomaly summary bullets. Handles None, empty, missing columns, and sparse data without crashing."""
    fallback = ["No statistically significant anomalies were flagged in the selected scope and period."]
    if anomalies_df is None or not isinstance(anomalies_df, pd.DataFrame) or anomalies_df.empty:
        return fallback
    work = anomalies_df.copy()
    required = {"zscore", "value_current"}
    if not required.issubset(set(work.columns)):
        return fallback
    if "entity" not in work.columns:
        work["entity"] = "selected scope"
    work["zscore"] = pd.to_numeric(work["zscore"], errors="coerce")
    work["value_current"] = pd.to_numeric(work["value_current"], errors="coerce")
    if work["zscore"].dropna().empty:
        return fallback
    work = work.dropna(subset=["zscore", "value_current"])
    if work.empty:
        return fallback
    try:
        # Use positional selection after index reset to avoid label/duplicate-index hazards.
        work = work.reset_index(drop=True)
        peak_pos = int(work["zscore"].abs().values.argmax())
        peak = work.iloc[peak_pos]
    except (ValueError, KeyError, IndexError, TypeError):
        return fallback
    entity_name = str(peak.get("entity", peak.get("dim_value", "selected scope")))
    try:
        val_cur = float(peak["value_current"])
        zval = float(peak["zscore"])
    except (TypeError, ValueError):
        return fallback
    direction = "positive" if val_cur >= 0 else "negative"
    return [
        f"Largest flow anomaly occurred in {entity_name} with {direction} NNB of {fmt_currency_kpi(val_cur)}.",
        f"Anomaly intensity (|z-score|) peaked at {abs(zval):.2f} in the selected period.",
    ]


@st.cache_resource
def _cached_gateway(root: str):
    from app.data.data_gateway import DataGateway
    return DataGateway(Path(root))


def _get_gateway():
    return _cached_gateway(str(ROOT))


def has_meaningful_variation(series: pd.Series | None) -> bool:
    if series is None or not isinstance(series, pd.Series):
        return False
    s = pd.to_numeric(series, errors="coerce").dropna()
    return len(s) >= 2 and s.nunique() >= 2 and not bool((s == 0).all())


def has_minimum_categories(df: pd.DataFrame | None, col: str, n: int = 2) -> bool:
    if df is None or df.empty or col not in df.columns:
        return False
    return int(df[col].astype(str).str.strip().replace("", pd.NA).dropna().nunique()) >= n


def should_render_chart(df: pd.DataFrame | None, required_cols: list[str], min_rows: int = 2, nonzero_cols: list[str] | None = None) -> bool:
    if df is None or df.empty or any(c not in df.columns for c in required_cols):
        return False
    work = df[required_cols].dropna(how="any")
    if len(work) < min_rows:
        return False
    for c in nonzero_cols or []:
        if c in work.columns and not has_meaningful_variation(work[c]):
            return False
    return True


def _first_row(df: pd.DataFrame | None) -> dict[str, Any]:
    return {} if df is None or df.empty else df.iloc[0].to_dict()


def _label_col(df: pd.DataFrame) -> str | None:
    for c in ["dim_value", "name", "channel", "product_ticker", "ticker", "country", "geo", "entity"]:
        if c in df.columns:
            return c
    return None


def _metric_col(df: pd.DataFrame, prefs: list[str]) -> str | None:
    for c in prefs:
        if c in df.columns:
            return c
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None


def _note(text: str, title: str = "Coverage Note") -> None:
    st.markdown(f"<div class='availability-note'><strong>{title}:</strong> {text}</div>", unsafe_allow_html=True)


def _normalize_bullets(bullets: list[str] | None, limit: int = 5) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    repl = {
        "No anomalies triggered under current thresholds.": "No statistically significant anomalies were flagged under current thresholds.",
        "No channel rank data available.": "Channel coverage is insufficient for comparative ranking in the selected slice.",
        "No product rank data available.": "Product coverage is insufficient for comparative ranking in the selected slice.",
        "Geo breakdown not available under current dataset columns.": "Geographic coverage is insufficient for comparative analysis in the selected slice.",
    }
    for b in bullets or []:
        txt = repl.get(b, b)
        txt = re.sub(r"^Mix shift:\s*(.+?)\s+share moved\s+(.+)\.$", r"Allocation mix shifted in \1 by \2.", txt)
        txt = re.sub(r"^Top channel by NNB:\s*(.+)\s+\((.+)\)\.$", r"Channel leadership was concentrated in \1, contributing \2 in net new business.", txt)
        txt = re.sub(r"^Top ticker by NNB:\s*(.+)\s+\((.+)\)\.$", r"Product leadership was driven by \1 with \2 in net new business.", txt)
        txt = re.sub(r"^Top geography by NNB:\s*(.+)\s+\((.+)\)\.$", r"Geographic leadership was concentrated in \1 with \2 in net new business.", txt)
        if txt and txt not in seen:
            out.append(txt)
            seen.add(txt)
        if len(out) >= limit:
            break
    return out


def _ranked_bar(df: pd.DataFrame, title: str, key: str, prefs: list[str]) -> bool:
    if go is None or df is None or df.empty:
        return False
    label = _label_col(df)
    metric = _metric_col(df, prefs)
    if label is None or metric is None:
        return False
    work = df[[label, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work = work[work[label].astype(str).str.strip() != ""]
    work = work[~work[label].astype(str).str.strip().isin(CHART_EXCLUDE_LABELS)]
    if work.empty:
        return False
    work = pd.concat([work.sort_values(metric, ascending=False).head(5), work.sort_values(metric, ascending=True).head(5)])
    work = work.drop_duplicates(subset=[label])
    if work.empty or not has_minimum_categories(work, label, 2) or not has_meaningful_variation(work[metric]):
        return False
    fig = go.Figure(go.Bar(x=work[metric], y=work[label].astype(str), orientation="h", marker=dict(color=[PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in work[metric]])))
    fig.update_layout(title=title, xaxis_title=metric, yaxis_title="", height=320, margin=dict(l=8, r=8, t=42, b=8))
    apply_enterprise_plotly_style(fig, height=320)
    st.plotly_chart(fig, width="stretch", key=key)
    return True


def _executive_chart(ts: pd.DataFrame) -> bool:
    if go is None or ts is None or ts.empty:
        return False
    if not should_render_chart(ts, ["month_end", "end_aum"], min_rows=2, nonzero_cols=["end_aum"]):
        return False
    cols = ["month_end", "end_aum"]
    if "nnb" in ts.columns:
        cols.append("nnb")
    if "market_impact_abs" in ts.columns:
        cols.append("market_impact_abs")
    elif "market_impact" in ts.columns:
        cols.append("market_impact")
    work = ts[cols].copy()
    work["month_end"] = pd.to_datetime(work["month_end"], errors="coerce")
    work["end_aum"] = pd.to_numeric(work["end_aum"], errors="coerce")
    work = work.dropna(subset=["month_end", "end_aum"]).sort_values("month_end")
    if len(work) < 2 or not has_meaningful_variation(work["end_aum"]):
        return False
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=work["month_end"], y=work["end_aum"], mode="lines+markers", name="End AUM (selected slice)", line=dict(color=PALETTE["primary"], width=2.4)))
    market_col = "market_impact_abs" if "market_impact_abs" in work.columns else ("market_impact" if "market_impact" in work.columns else None)
    if market_col is not None and has_meaningful_variation(work[market_col]):
        work[market_col] = pd.to_numeric(work[market_col], errors="coerce")
        market_last = float(work[market_col].dropna().iloc[-1]) if not work[market_col].dropna().empty else 0.0
        market_color = PALETTE["positive"] if market_last >= 0 else PALETTE["negative"]
        fig.add_trace(go.Scatter(x=work["month_end"], y=work[market_col], mode="lines+markers", name="Market Movement", line=dict(color=market_color, width=2.1), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Market Movement", overlaying="y", side="right", showgrid=False))
    elif "nnb" in work.columns and has_meaningful_variation(work["nnb"]):
        work["nnb"] = pd.to_numeric(work["nnb"], errors="coerce")
        nnb_colors = [PALETTE["positive"] if float(v) >= 0 else PALETTE["negative"] for v in work["nnb"].fillna(0.0)]
        fig.add_trace(go.Bar(x=work["month_end"], y=work["nnb"], name="NNB", marker=dict(color=nnb_colors, opacity=0.45), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="NNB", overlaying="y", side="right", showgrid=False))
    fig.update_layout(title="AUM trend (selected scope)", xaxis_title="Month", yaxis_title="Selected Scope End AUM", height=320, margin=dict(l=8, r=8, t=42, b=8))
    apply_enterprise_plotly_style(fig, height=320)
    st.plotly_chart(fig, width="stretch", key="tab2_exec_ts")
    return True


def _mix_shift_or_concentration(df: pd.DataFrame, base_key: str, section_name: str) -> bool:
    if go is None or df is None or df.empty:
        return False
    label = _label_col(df)
    if label is None:
        return False
    delta = _metric_col(df, ["aum_share_delta", "nnb_share_delta", "share_delta"])
    if delta is not None and has_minimum_categories(df, label, 2):
        mix = df[[label, delta]].copy()
        mix[delta] = pd.to_numeric(mix[delta], errors="coerce")
        mix = mix.dropna(subset=[delta]).sort_values(delta, ascending=False).head(8)
        if len(mix) >= 2 and has_meaningful_variation(mix[delta]):
            fig = go.Figure(go.Bar(x=mix[label].astype(str), y=mix[delta], marker=dict(color=[PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in mix[delta]])))
            fig.update_layout(title=f"{section_name} mix shift", xaxis_title="", yaxis_title=delta, height=290, margin=dict(l=8, r=8, t=42, b=8))
            apply_enterprise_plotly_style(fig, height=290)
            st.plotly_chart(fig, width="stretch", key=f"{base_key}_mix")
            return True
    share = _metric_col(df, ["share"])
    if share is not None and has_minimum_categories(df, label, 2):
        conc = df[[label, share]].copy()
        conc[share] = pd.to_numeric(conc[share], errors="coerce")
        conc = conc.dropna(subset=[share]).sort_values(share, ascending=False).head(5)
        if len(conc) >= 2 and has_meaningful_variation(conc[share]):
            fig = go.Figure(go.Bar(x=conc[label].astype(str), y=conc[share], marker=dict(color=PALETTE["secondary"])))
            fig.update_layout(title=f"{section_name} concentration (top 5)", xaxis_title="", yaxis_title="Share", height=290, margin=dict(l=8, r=8, t=42, b=8))
            apply_enterprise_plotly_style(fig, height=290)
            st.plotly_chart(fig, width="stretch", key=f"{base_key}_conc")
            return True
    return False


def _section_to_markdown(title: str, out: SectionOutput) -> list[str]:
    lines = [f"## {title}", ""]
    for b in out.bullets or []:
        lines.append(f"- {b}")
    return lines + ["", "---", ""]


def _export_markdown(meta: dict[str, Any], sections: list[tuple[str, SectionOutput]]) -> str:
    lines = ["# Investment Commentary", "", f"- **Dataset version:** {meta.get('dataset_version', '-')}", f"- **Filter hash:** `{meta.get('filter_hash', '')}`", "", "---", ""]
    for title, out in sections:
        lines.extend(_section_to_markdown(title, out))
    return "\n".join(lines)


def render(state: FilterState, contract: dict[str, Any]) -> None:
    st.subheader("Investment Commentary")
    filters = get_filter_state()
    tab1 = _tab1_snapshot_for_parity()
    period = str(tab1.get("tab1_period", TAB1_DEFAULT_PERIOD))
    scope_label = get_scope_label_from_state(tab1)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    h1, h2, h3 = st.columns(3)
    h1.caption(f"Reporting window: {filters.date_start} to {filters.date_end}")
    h2.caption(f"Portfolio scope: {scope_label}")
    h3.caption(f"Data refresh timestamp: {updated}")
    st.markdown("<div class='section-subtitle'>Portfolio commentary and evidence; narrative from internal data.</div>", unsafe_allow_html=True)

    try:
        with st.spinner("Loading investment commentary..."):
            gateway = _get_gateway()
            pack = gateway.get_report_pack(filters)
            shared_payload = build_metric_payload(
                gateway=gateway,
                state=filters,
                scope_label=scope_label,
                period=period,
                channel=tab1.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL),
                sub_channel=tab1.get("tab1_filter_sub_channel", TAB1_DEFAULT_SUB_CHANNEL),
                country=tab1.get("tab1_filter_country", TAB1_DEFAULT_COUNTRY),
                segment=None,  # Segment filter removed: source is always Fixed Income
                sub_segment=tab1.get("tab1_filter_sub_segment", TAB1_DEFAULT_SUB_SEGMENT),
                ticker=tab1.get("tab1_filter_ticker", TAB1_DEFAULT_PRODUCT_TICKER),
            )
    except Exception as e:
        LOGGER.warning("Investment Commentary load failed: %s", e, exc_info=False)
        render_error_state("Investment Commentary", e, "Try adjusting filters or reload the page.")
        return
    log = st.session_state.get("perf_query_log") or []
    if any(e.get("name") == "get_report_pack" and ("over_budget" in str(e.get("warning") or "")) for e in log[-10:]):
        render_timeout_state("Report generation", HEAVY_BUDGET_MS, "Try a shorter date range.")

    rec = run_reconciliation(pack)
    meta = {
        "dataset_version": getattr(pack, "dataset_version", "-"),
        "filter_hash": getattr(pack, "filter_hash", "-"),
        "filters_used": filters.to_dict(),
        "report_timestamp": datetime.now(timezone.utc).isoformat(),
        "reconciliation": rec,
    }

    overview = render_overview(pack)
    recs = render_recommendations(pack)

    snap_src = getattr(pack, "firm_snapshot", pd.DataFrame())
    kpi = shared_payload.kpi_snapshot
    period_df = shared_payload.df_period
    channel_df = _build_dim_rank(period_df, "channel")
    product_df = _build_dim_rank(period_df, "product_ticker")
    geo_df = _build_dim_rank(period_df, "country")
    anomalies_df = pd.DataFrame()
    if shared_payload.monthly_period is not None and not shared_payload.monthly_period.empty and "nnb" in shared_payload.monthly_period.columns:
        nnb_series = pd.to_numeric(shared_payload.monthly_period["nnb"], errors="coerce")
        mean_nnb = float(nnb_series.mean()) if not nnb_series.dropna().empty else 0.0
        std_nnb = float(nnb_series.std()) if float(nnb_series.std()) > 0 else float("nan")
        zscore = (nnb_series - mean_nnb) / std_nnb if std_nnb == std_nnb else pd.Series([0.0] * len(nnb_series))
        anomalies_df = pd.DataFrame(
            {
                "entity": shared_payload.monthly_period["month_end"].dt.strftime("%Y-%m"),
                "value_current": nnb_series,
                "zscore": zscore,
            }
        )
    suppressed: list[str] = []
    LOGGER.info(
        "commentary_payload scope=%s period=%s filters=%s rows_filtered=%d rows_period=%d kpi_end_aum=%.6f kpi_nnb=%.6f kpi_nnf=%.6f kpi_market=%.6f reconciled=%s variance=%.6f",
        scope_label,
        period,
        {
            "channel": tab1.get("tab1_filter_channel"),
            "sub_channel": tab1.get("tab1_filter_sub_channel"),
            "country": tab1.get("tab1_filter_country"),
            "sub_segment": tab1.get("tab1_filter_sub_segment"),
            "sales_focus": tab1.get("tab1_filter_sales_focus"),
            "ticker": tab1.get("tab1_filter_ticker"),
            "date_start": filters.date_start,
            "date_end": filters.date_end,
        },
        int(len(shared_payload.df_filtered)),
        int(len(shared_payload.df_period)),
        float(pd.to_numeric(kpi.get("end_aum"), errors="coerce")),
        float(pd.to_numeric(kpi.get("nnb"), errors="coerce")),
        float(pd.to_numeric(kpi.get("nnf"), errors="coerce")),
        float(pd.to_numeric(kpi.get("market_pnl"), errors="coerce")),
        bool(shared_payload.reconciliation.get("ok")),
        float(pd.to_numeric(shared_payload.reconciliation.get("variance"), errors="coerce")),
    )

    st.markdown("#### Executive Overview")
    st.markdown("<div class='section-subtitle'>Growth direction, revenue quality, and market contribution.</div>", unsafe_allow_html=True)
    primary_kpis = build_executive_overview_primary_kpis(kpi)
    LOGGER.debug("Investment Commentary KPI strings: %r", {item.get("label"): item.get("value") for item in primary_kpis})
    render_kpi_row(primary_kpis)
    # Second row: supporting indicators; same formatter policy as tab 1 primary row.
    render_kpi_row(build_executive_overview_secondary_kpis(kpi))
    st.caption(f"Report slice: **{scope_label}**. All values are from the same filtered monthly source as Executive Dashboard KPIs.")
    overview_list = _overview_bullets_from_payload(kpi, period=period)
    for b in overview_list:
        st.markdown(f"- {b}")
    if not _executive_chart(shared_payload.monthly_period.rename(columns={"market_pnl": "market_impact_abs"})):
        _note("Time-series data in the selected range is insufficient for the executive chart.")

    st.markdown("#### Channel Analysis")
    st.markdown("<div class='section-subtitle'>Channel concentration, leadership, and mix shift.</div>", unsafe_allow_html=True)
    for b in _section_bullets_from_rank(channel_df, "channel", "Channel"):
        st.markdown(f"- {b}")
    if not _ranked_bar(channel_df, "Channel contributors by net new business", "tab2_channel", ["nnb", "end_aum", "share"]):
        if has_minimum_categories(channel_df, _label_col(channel_df) or "", 1):
            _note("Channel concentration is fully driven by one distribution group.", "Concentration")
        else:
            suppressed.append("Channel Analysis")
    elif not _mix_shift_or_concentration(channel_df, "tab2_channel", "Channel"):
        _note("Channel mix lacks enough variation for a secondary mix-shift view.", "Signal Note")

    st.markdown("#### Product and ETF Analysis")
    st.markdown("<div class='section-subtitle'>Product-level flow, concentration, and ETF contribution.</div>", unsafe_allow_html=True)
    for b in _section_bullets_from_rank(product_df, "product_ticker", "Product"):
        st.markdown(f"- {b}")
    if not _ranked_bar(product_df, "Product leaders and laggards", "tab2_product", ["nnb", "end_aum", "share"]):
        _note("Product mix shows limited cross-sectional dispersion in the selected slice.")
    elif not _mix_shift_or_concentration(product_df, "tab2_product", "Product"):
        _note("Product allocation shifts are limited under the selected slice.", "Signal Note")

    st.markdown("#### Geographic Analysis")
    st.markdown("<div class='section-subtitle'>What this shows: regional contribution and mix shift. Why it matters: see which geographies are driving or lagging.</div>", unsafe_allow_html=True)
    for b in _section_bullets_from_rank(geo_df, "country", "Geography"):
        st.markdown(f"- {b}")
    geo_label = _label_col(geo_df) or ""
    if has_minimum_categories(geo_df, geo_label, 2):
        if not _ranked_bar(geo_df, "Geographic contribution by net new business", "tab2_geo", ["nnb", "end_aum", "share"]):
            _note("Geographic evidence exists but lacks meaningful variation for charting.")
    else:
        _note("Geographic view is concentrated in a single market, so no comparative chart is shown.", "Concentration")

    st.markdown("#### Anomalies and Flags")
    st.markdown("<div class='section-subtitle'>Outliers and patterns for review.</div>", unsafe_allow_html=True)
    anomaly_bullets = _anomaly_bullets_from_shared(anomalies_df)
    for b in anomaly_bullets:
        st.markdown(f"- {b}")
    if anomalies_df is not None and not anomalies_df.empty:
        if not _ranked_bar(anomalies_df, "Anomaly intensity snapshot", "tab2_risk", ["zscore", "value_current"]):
            _note("Anomaly chart not shown: fewer than two meaningful data points in the selected range.")

    st.markdown("#### Recommendations")
    st.markdown("<div class='section-subtitle'>Suggested next actions from anomalies and mix.</div>", unsafe_allow_html=True)
    actions = _normalize_bullets(recs.bullets, 6)
    for b in actions:
        st.markdown(f"- {b}")

    export_sections: list[tuple[str, SectionOutput]] = [
        ("Executive Overview", SectionOutput(overview_list, overview.table_title, overview.table, overview.meta)),
        ("Channel Analysis", SectionOutput(_section_bullets_from_rank(channel_df, "channel", "Channel"), "Channel evidence", channel_df, {"source": "shared_payload.df_period"})),
        ("Product and ETF Analysis", SectionOutput(_section_bullets_from_rank(product_df, "product_ticker", "Product"), "Product evidence", product_df, {"source": "shared_payload.df_period"})),
        ("Geographic Analysis", SectionOutput(_section_bullets_from_rank(geo_df, "country", "Geography"), "Geographic evidence", geo_df, {"source": "shared_payload.df_period"})),
        ("Anomalies and Flags", SectionOutput(anomaly_bullets, "Anomaly evidence", anomalies_df, {"source": "shared_payload.monthly_period"})),
        ("Recommendations", SectionOutput(actions, recs.table_title, recs.table, recs.meta)),
    ]

    st.divider()
    if suppressed:
        st.caption("Sections not shown (insufficient data in selected range): " + ", ".join(suppressed))

    with st.expander("Evidence Pack and Consistency Checks", expanded=False):
        recon_df = pd.DataFrame(rec)
        if not recon_df.empty:
            st.dataframe(format_df(recon_df, infer_common_formats(recon_df)), width="stretch", hide_index=True, height=min(220, 56 * len(recon_df) + 40))
        for name, tbl in [
            ("Executive evidence", overview.table),
            ("Channel evidence", channel_df),
            ("Product evidence", product_df),
            ("Geographic evidence", geo_df),
            ("Anomaly evidence", anomalies_df),
            ("Recommendations evidence", recs.table),
        ]:
            if tbl is not None and not tbl.empty:
                st.caption(name)
                st.dataframe(format_df(tbl.head(12), infer_common_formats(tbl.head(12))), width="stretch", hide_index=True, height=min(220, 56 * len(tbl.head(12)) + 40))
                render_export_buttons(tbl, None, f"tab2_{name.lower().replace(' ', '_')}")

        html = build_report_html("Investment Commentary", export_sections, meta)
        md = _export_markdown(meta, export_sections)
        safe_dv = _safe_filename(getattr(pack, "dataset_version", "") or meta.get("dataset_version", "report"))
        safe_fh = _safe_filename(getattr(pack, "filter_hash", "") or meta.get("filter_hash", ""))
        ds_ver = getattr(pack, "dataset_version", "") or meta.get("dataset_version", "report")
        pipeline_ver = meta.get("report_timestamp", "")[:19] or "report"
        body_lines: list[str] = []
        for sec_title, out in export_sections:
            body_lines.append(sec_title)
            for b in out.bullets or []:
                body_lines.append("  - " + b[:120])
            body_lines.append("")
        pdf_bytes: bytes | None = None
        if make_pdf_with_footer is not None:
            pdf_bytes = make_pdf_with_footer("Investment Commentary", body_lines, ds_ver, pipeline_ver)
        mcol, hcol, pcol = st.columns(3)
        mcol.download_button("Download Markdown (.md)", data=md, file_name="investment_commentary.md", mime="text/markdown", key="dl_dr_md")
        hcol.download_button("Download HTML", data=html.encode("utf-8"), file_name=f"report_{safe_dv}_{safe_fh}.html", mime="text/html", key="dl_dr_html")
        if pdf_bytes is not None:
            pcol.download_button("Download PDF", data=pdf_bytes, file_name=f"report_{safe_dv}_{safe_fh}.pdf", mime="application/pdf", key="dl_dr_pdf")
        else:
            pcol.caption("PDF export requires reportlab.")

    render_observability_panel(filters=filters)
    render_obs_panel(contract.get("tab_id", "dynamic_report"))

    if st.session_state.get("dev_mode"):
        with st.expander("Investment Commentary (debug)", expanded=False):
            parity = {
                "shared_scope_label": shared_payload.scope_label,
                "shared_period": shared_payload.period,
                "shared_kpi": kpi,
                "shared_reconciliation": shared_payload.reconciliation,
            }
            # Lightweight parity check against pack snapshot where available.
            pack_snap = _first_row(snap_src if isinstance(snap_src, pd.DataFrame) else pd.DataFrame())
            if pack_snap:
                parity["pack_snapshot_compare"] = {
                    "end_aum_delta": float(pd.to_numeric(kpi.get("end_aum"), errors="coerce") - pd.to_numeric(pack_snap.get("end_aum"), errors="coerce")),
                    "nnb_delta": float(pd.to_numeric(kpi.get("nnb"), errors="coerce") - pd.to_numeric(pack_snap.get("nnb"), errors="coerce")),
                    "nnf_delta": float(pd.to_numeric(kpi.get("nnf"), errors="coerce") - pd.to_numeric(pack_snap.get("nnf"), errors="coerce")),
                }
            st.json({"dataset_version": meta.get("dataset_version"), "filter_hash": meta.get("filter_hash"), "source_diagnostics": (getattr(pack, "meta", {}) or {}).get("source_diagnostics", {}), "parity": parity})
            try:
                from app.metrics.snapshot import get_metrics_debug_info, validation_required_metrics
                shared_snap_df = pd.DataFrame([kpi])
                missing = validation_required_metrics(kpi) if kpi else ["no snapshot"]
                st.caption("Required metrics (missing -> fix source or period)")
                st.json({"missing_required_metrics": missing})
                st.caption("Canonical formulas / period debug")
                st.json(get_metrics_debug_info(shared_snap_df, None))
            except Exception:
                pass
