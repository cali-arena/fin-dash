"""
Tab 2: Dynamic Report - fully deterministic Investment Commentary.
No LLM: all narrative from Python-calculated metrics and conditional templates (app.reporting.nlg_templates).
Six sections: Executive Overview, Channel Analysis, Product and ETF Analysis, Geographic Analysis,
Anomalies and Flags, Recommendations. Export: HTML, Markdown, PDF.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

from app.data.data_gateway import HEAVY_BUDGET_MS
from app.reporting.html_export import _safe_filename, build_report_html
from app.reporting.reconciliation import run_reconciliation
from app.reporting.report_charts import fig_aum_trend_mpl
from app.reporting.report_engine import (
    SectionOutput,
    render_anomalies,
    render_channel_commentary,
    render_geo_commentary,
    render_overview,
    render_product_commentary,
    render_recommendations,
)
from app.state import FilterState, get_filter_state
from app.ui.exports import render_export_buttons
from app.ui.formatters import fmt_currency, fmt_percent, format_df, infer_common_formats
from app.ui.guardrails import render_empty_state, render_timeout_state

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
    for c in ["dim_value", "name", "channel", "ticker", "geo", "entity"]:
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
    if go is None:
        return False
    label = _label_col(df)
    metric = _metric_col(df, prefs)
    if label is None or metric is None:
        return False
    work = df[[label, metric]].copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    work = work[work[label].astype(str).str.strip() != ""]
    work = pd.concat([work.sort_values(metric, ascending=False).head(5), work.sort_values(metric, ascending=True).head(5)])
    work = work.drop_duplicates(subset=[label])
    if not has_minimum_categories(work, label, 2) or not has_meaningful_variation(work[metric]):
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
        fig.add_trace(go.Scatter(x=work["month_end"], y=work[market_col], mode="lines+markers", name="Market Movement", line=dict(color=PALETTE["market"], width=2.1), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="Market Movement", overlaying="y", side="right", showgrid=False))
    elif "nnb" in work.columns and has_meaningful_variation(work["nnb"]):
        work["nnb"] = pd.to_numeric(work["nnb"], errors="coerce")
        fig.add_trace(go.Bar(x=work["month_end"], y=work["nnb"], name="NNB", marker=dict(color=PALETTE["secondary"], opacity=0.35), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title="NNB", overlaying="y", side="right", showgrid=False))
    fig.update_layout(title="AUM trend (selected slice)", xaxis_title="Month", yaxis_title="End AUM (selected slice)", height=320, margin=dict(l=8, r=8, t=42, b=8))
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
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    h1, h2, h3 = st.columns(3)
    h1.caption(f"Reporting window: {filters.date_start} to {filters.date_end}")
    h2.caption(f"Portfolio scope: {filters.slice_dim + ': ' + filters.slice_value if filters.slice_dim and filters.slice_value else 'Enterprise-wide portfolio'}")
    h3.caption(f"Data refresh timestamp: {updated}")
    st.markdown("<div class='section-subtitle'>Portfolio commentary and supporting evidence for decisions. Narrative and metrics are deterministic from your internal data.</div>", unsafe_allow_html=True)

    with st.spinner("Loading investment commentary..."):
        pack = _get_gateway().get_report_pack(filters)
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
    channel = render_channel_commentary(pack)
    product = render_product_commentary(pack)
    geo = render_geo_commentary(pack)
    anomalies = render_anomalies(pack)
    recs = render_recommendations(pack)

    snap_src = getattr(pack, "firm_snapshot", pd.DataFrame())
    snap = _first_row(snap_src if isinstance(snap_src, pd.DataFrame) else pd.DataFrame())
    channel_df = getattr(pack, "channel_rank", pd.DataFrame())
    product_df = getattr(pack, "ticker_rank", pd.DataFrame())
    geo_df = getattr(pack, "geo_rank", pd.DataFrame())
    anomalies_df = getattr(pack, "anomalies", pd.DataFrame())
    channel_df = channel_df if isinstance(channel_df, pd.DataFrame) else pd.DataFrame()
    product_df = product_df if isinstance(product_df, pd.DataFrame) else pd.DataFrame()
    geo_df = geo_df if isinstance(geo_df, pd.DataFrame) else pd.DataFrame()
    anomalies_df = anomalies_df if isinstance(anomalies_df, pd.DataFrame) else pd.DataFrame()
    suppressed: list[str] = []

    st.markdown("#### Executive Overview")
    st.markdown("<div class='section-subtitle'>Growth direction, revenue quality, and market contribution.</div>", unsafe_allow_html=True)
    report_scope = (filters.slice_dim + ": " + filters.slice_value) if (filters.slice_dim and filters.slice_value) else "Firm-wide portfolio"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("End AUM (selected slice)", fmt_currency(snap.get("end_aum"), unit="auto", decimals=2))
    c2.metric("MoM Growth", fmt_percent(snap.get("mom_pct"), decimals=2, signed=True))
    c3.metric("YTD Growth", fmt_percent(snap.get("ytd_pct"), decimals=2, signed=True))
    c4.metric("Net New Business", fmt_currency(snap.get("nnb"), unit="auto", decimals=2))
    c5.metric("Market Impact", fmt_currency(snap.get("market_impact_abs"), unit="auto", decimals=2))
    st.caption(f"Report slice: **{report_scope}**. All values = selected slice only (not enterprise AUM unless slice is Firm-wide).")
    overview_list = _normalize_bullets(overview.bullets, 8)
    if overview_list and len(overview_list[0]) > 120:
        st.markdown(overview_list[0])
        for b in overview_list[1:]:
            st.markdown(f"- {b}")
    else:
        for b in overview_list:
            st.markdown(f"- {b}")
    if not _executive_chart(getattr(pack, "time_series", pd.DataFrame())):
        _note("Time-series data in the selected range is insufficient for the executive chart.")

    st.markdown("#### Channel Analysis")
    st.markdown("<div class='section-subtitle'>What this shows: channel concentration, leadership, and mix shift. Why it matters: see where flow and AUM are coming from and where to reallocate.</div>", unsafe_allow_html=True)
    for b in _normalize_bullets(channel.bullets, 6):
        st.markdown(f"- {b}")
    if not _ranked_bar(channel_df, "Channel contributors by net new business", "tab2_channel", ["nnb", "aum_end", "value"]):
        if has_minimum_categories(channel_df, _label_col(channel_df) or "", 1):
            _note("Channel concentration is fully driven by one distribution group.", "Concentration")
        else:
            suppressed.append("Channel Analysis")
    elif not _mix_shift_or_concentration(channel_df, "tab2_channel", "Channel"):
        _note("Channel mix lacks enough variation for a secondary mix-shift view.", "Signal Note")

    st.markdown("#### Product and ETF Analysis")
    st.markdown("<div class='section-subtitle'>What this shows: product-level flow quality, concentration, and ETF contribution. Why it matters: spot leaders, laggards, and concentration risk.</div>", unsafe_allow_html=True)
    for b in _normalize_bullets(product.bullets, 6):
        st.markdown(f"- {b}")
    if not _ranked_bar(product_df, "Product leaders and laggards", "tab2_product", ["nnb", "aum_end", "value"]):
        _note("Product mix shows limited cross-sectional dispersion in the selected slice.")
    elif not _mix_shift_or_concentration(product_df, "tab2_product", "Product"):
        _note("Product allocation shifts are limited under the selected slice.", "Signal Note")

    st.markdown("#### Geographic Analysis")
    st.markdown("<div class='section-subtitle'>What this shows: regional contribution and mix shift. Why it matters: see which geographies are driving or lagging.</div>", unsafe_allow_html=True)
    for b in _normalize_bullets(geo.bullets, 6):
        st.markdown(f"- {b}")
    geo_label = _label_col(geo_df) or ""
    if has_minimum_categories(geo_df, geo_label, 2):
        if not _ranked_bar(geo_df, "Geographic contribution by net new business", "tab2_geo", ["nnb", "aum_end", "value"]):
            _note("Geographic evidence exists but lacks meaningful variation for charting.")
    else:
        _note("Geographic view is concentrated in a single market, so no comparative chart is shown.", "Concentration")

    st.markdown("#### Anomalies and Flags")
    st.markdown("<div class='section-subtitle'>What this shows: outliers and patterns that may need attention. Why it matters: focus review on items that deviate from baseline.</div>", unsafe_allow_html=True)
    for b in _normalize_bullets(anomalies.bullets, 6):
        st.markdown(f"- {b}")
    if anomalies_df is not None and not anomalies_df.empty:
        if not _ranked_bar(anomalies_df, "Anomaly intensity snapshot", "tab2_risk", ["zscore", "value_current"]):
            _note("Anomaly chart not shown: fewer than two meaningful data points in the selected range.")

    st.markdown("#### Recommendations")
    st.markdown("<div class='section-subtitle'>What this shows: suggested next actions from anomalies, concentration, and mix shift. Why it matters: prioritize where to act.</div>", unsafe_allow_html=True)
    actions = _normalize_bullets(recs.bullets, 6)
    for b in actions:
        st.markdown(f"- {b}")

    export_sections: list[tuple[str, SectionOutput]] = [
        ("Executive Overview", SectionOutput(_normalize_bullets(overview.bullets, 8), overview.table_title, overview.table, overview.meta)),
        ("Channel Analysis", SectionOutput(_normalize_bullets(channel.bullets, 6), channel.table_title, channel.table, channel.meta)),
        ("Product and ETF Analysis", SectionOutput(_normalize_bullets(product.bullets, 6), product.table_title, product.table, product.meta)),
        ("Geographic Analysis", SectionOutput(_normalize_bullets(geo.bullets, 6), geo.table_title, geo.table, geo.meta)),
        ("Anomalies and Flags", SectionOutput(_normalize_bullets(anomalies.bullets, 6), anomalies.table_title, anomalies.table, anomalies.meta)),
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
            ("Channel evidence", channel.table),
            ("Product evidence", product.table),
            ("Geographic evidence", geo.table),
            ("Anomaly evidence", anomalies.table),
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
            st.json({"dataset_version": meta.get("dataset_version"), "filter_hash": meta.get("filter_hash"), "source_diagnostics": (getattr(pack, "meta", {}) or {}).get("source_diagnostics", {})})
            try:
                from app.metrics.snapshot import get_metrics_debug_info, validation_required_metrics
                missing = validation_required_metrics(snap) if snap else ["no snapshot"]
                st.caption("Required metrics (missing -> fix source or period)")
                st.json({"missing_required_metrics": missing})
                st.caption("Canonical formulas / period debug")
                st.json(get_metrics_debug_info(snap_src if isinstance(snap_src, pd.DataFrame) else None, None))
            except Exception:
                pass
