from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None

from app.config.tab1_defaults import (
    get_scope_label_from_state,
    get_scope_mode_from_state,
    TAB1_DEFAULT_CHANNEL,
    TAB1_DEFAULT_COUNTRY,
    TAB1_DEFAULT_PERIOD,
    TAB1_DEFAULT_PRODUCT_TICKER,
    TAB1_DEFAULT_SEGMENT,
    TAB1_DEFAULT_SUB_CHANNEL,
    TAB1_DEFAULT_SUB_SEGMENT,
)
from app.data.data_gateway import DataGateway
from app.kpi.service import (
    KPIResult,
    apply_period_canonical,
    compute_kpi,
    validate_kpi_against_latest_row,
)
from app.metrics.metric_contract import (
    compute_fee_yield,
    compute_fee_yield_nnf_nnb,
    compute_market_impact,
    compute_market_impact_rate,
    compute_ogr,
)
from app.state import FilterState
from app.ui.exports import render_export_buttons
from app.ui.formatters import fmt_bps, fmt_currency, format_df, infer_common_formats
from app.ui.theme import PALETTE, apply_enterprise_plotly_style, safe_render_plotly

ROOT = Path(__file__).resolve().parents[2]
PERIOD_OPTIONS = ("1M", "QoQ", "YTD", "YoY")


@dataclass(frozen=True)
class NarrativeFacts:
    month_end: str
    end_aum: float
    nnb: float
    ogr: float
    market_impact_rate: float


def _coerce_num(x: Any) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _build_hero_narrative_context(monthly: pd.DataFrame, monthly_full: pd.DataFrame) -> dict[str, Any]:
    """
    Build context for the hero narrative from period-filtered monthly (current) and full-history monthly_full.
    Returns dict with latest, prior_row, yoy_row, roll36_avg, highest_since_aum, lowest_since_aum, highest_since_ogr, lowest_since_ogr.
    All values are Python-calculated from the provided data only.
    """
    out: dict[str, Any] = {
        "latest": None,
        "prior_row": None,
        "yoy_row": None,
        "roll36_avg_end_aum": float("nan"),
        "roll36_avg_ogr": float("nan"),
        "mom_pct": float("nan"),
        "yoy_pct": float("nan"),
        "prior_ogr": float("nan"),
        "yoy_ogr": float("nan"),
        "highest_since_aum": None,
        "lowest_since_aum": None,
        "highest_since_ogr": None,
        "lowest_since_ogr": None,
    }
    if monthly is None or monthly.empty:
        return out
    sorted_full = monthly_full.sort_values("month_end").reset_index(drop=True) if monthly_full is not None and not monthly_full.empty else pd.DataFrame()
    latest = monthly.sort_values("month_end").iloc[-1]
    out["latest"] = latest
    current_me = pd.Timestamp(latest["month_end"])
    current_end = _coerce_num(latest.get("end_aum"))
    current_ogr = _coerce_num(latest.get("ogr"))
    current_nnb = _coerce_num(latest.get("nnb"))
    current_mi = _coerce_num(latest.get("market_impact"))

    if len(sorted_full) >= 2:
        idx = sorted_full[sorted_full["month_end"] == current_me].index
        if len(idx) > 0:
            i = int(idx[0])
            if i >= 1:
                out["prior_row"] = sorted_full.iloc[i - 1]
                prior_end = _coerce_num(out["prior_row"].get("end_aum"))
                if prior_end and prior_end > 0 and current_end == current_end:
                    out["mom_pct"] = (current_end - prior_end) / prior_end
                out["prior_ogr"] = _coerce_num(out["prior_row"].get("ogr"))
    if sorted_full.shape[0] >= 1:
        target_yoy = current_me - pd.DateOffset(months=12)
        yoy_candidates = sorted_full[pd.to_datetime(sorted_full["month_end"]).dt.normalize() == pd.Timestamp(target_yoy).normalize()]
        if not yoy_candidates.empty:
            out["yoy_row"] = yoy_candidates.iloc[0]
            yoy_end = _coerce_num(out["yoy_row"].get("end_aum"))
            if yoy_end and yoy_end > 0 and current_end == current_end:
                out["yoy_pct"] = (current_end - yoy_end) / yoy_end
            out["yoy_ogr"] = _coerce_num(out["yoy_row"].get("ogr"))
        roll36 = sorted_full.tail(36)
        if len(roll36) >= 12:
            out["roll36_avg_end_aum"] = float(roll36["end_aum"].mean())
            ogr_vals = pd.to_numeric(roll36["ogr"], errors="coerce").dropna()
            if not ogr_vals.empty:
                out["roll36_avg_ogr"] = float(ogr_vals.mean())
        if len(sorted_full) >= 2:
            hist = sorted_full[sorted_full["month_end"] <= current_me]
            if not hist.empty and "end_aum" in hist.columns:
                aum_vals = pd.to_numeric(hist["end_aum"], errors="coerce").dropna()
                if not aum_vals.empty and current_end == current_end:
                    if current_end >= aum_vals.max():
                        out["highest_since_aum"] = hist["month_end"].min()
                    elif current_end <= aum_vals.min():
                        out["lowest_since_aum"] = hist["month_end"].min()
            if "ogr" in hist.columns:
                ogr_vals = pd.to_numeric(hist["ogr"], errors="coerce").dropna()
                if not ogr_vals.empty and current_ogr == current_ogr:
                    if current_ogr >= ogr_vals.max():
                        out["highest_since_ogr"] = hist["month_end"].min()
                    elif current_ogr <= ogr_vals.min():
                        out["lowest_since_ogr"] = hist["month_end"].min()
    return out


def _build_hero_narrative_paragraph(context: dict[str, Any], scope: str) -> str:
    """
    Build one institutional asset management commentary paragraph from verified context only.
    Matches client style: AUM growth %, OGR, market impact rate, prior month, same month last year,
    3-year average, highest-since logic. All values Python-calculated; 3–4 concise sentences.
    """
    latest = context.get("latest")
    if latest is None:
        return f"No portfolio snapshot is available for the selected scope ({scope})."
    current_me = pd.Timestamp(latest["month_end"])
    month_name = current_me.strftime("%B")
    month_year = current_me.strftime("%B %Y")
    end_aum = _coerce_num(latest.get("end_aum"))
    ogr = _coerce_num(latest.get("ogr"))
    market_rate = _coerce_num(latest.get("market_impact_rate"))
    mom_pct = context.get("mom_pct", float("nan"))
    prior_ogr = context.get("prior_ogr", float("nan"))
    yoy_ogr = context.get("yoy_ogr", float("nan"))
    roll36_avg_ogr = context.get("roll36_avg_ogr", float("nan"))
    highest_since_ogr = context.get("highest_since_ogr")
    lowest_since_ogr = context.get("lowest_since_ogr")

    sentences: list[str] = []

    # Sentence 1: AUM growth during the month; OGR and market movement
    if mom_pct == mom_pct:
        if mom_pct > 0:
            sentences.append(f"Assets under management grew by {_fmt_pct_signed(mom_pct)} during the month of {month_name}.")
        elif mom_pct < 0:
            sentences.append(f"Assets under management declined by {_fmt_pct_signed(mom_pct)} during the month of {month_name}.")
        else:
            sentences.append(f"Assets under management were flat during the month of {month_name}.")
    if ogr == ogr and market_rate == market_rate:
        if market_rate < 0:
            sentences.append(f"Organic growth accounted for {_fmt_pct_signed(ogr)} but was offset by market movement of {_fmt_pct_signed(market_rate)}.")
        elif market_rate > 0:
            sentences.append(f"Organic growth accounted for {_fmt_pct_signed(ogr)} and was supported by market movement of {_fmt_pct_signed(market_rate)}.")
        else:
            sentences.append(f"Organic growth accounted for {_fmt_pct_signed(ogr)}; market movement was flat.")
    elif ogr == ogr:
        sentences.append(f"Organic growth accounted for {_fmt_pct_signed(ogr)}.")

    # Sentence 2: Prior month OGR vs same month last year OGR
    prior_ok = prior_ogr == prior_ogr and prior_ogr is not None
    yoy_ok = yoy_ogr == yoy_ogr and yoy_ogr is not None
    if prior_ok or yoy_ok:
        clauses = []
        if prior_ok and ogr == ogr:
            if ogr >= prior_ogr:
                clauses.append(f"{month_name} beat previous month organic growth of {_fmt_pct_signed(prior_ogr)}")
            else:
                clauses.append(f"{month_name} lagged previous month organic growth of {_fmt_pct_signed(prior_ogr)}")
        if yoy_ok and ogr == ogr:
            if ogr >= yoy_ogr:
                clauses.append(f"was higher than last {month_name}'s growth of {_fmt_pct_signed(yoy_ogr)}" if prior_ok else f"{month_name} was higher than last {month_name}'s growth of {_fmt_pct_signed(yoy_ogr)}")
            else:
                clauses.append(f"was lower than last {month_name}'s growth of {_fmt_pct_signed(yoy_ogr)}" if prior_ok else f"{month_name} was lower than last {month_name}'s growth of {_fmt_pct_signed(yoy_ogr)}")
        if clauses:
            sentences.append((clauses[0] + " but " + clauses[1] if len(clauses) == 2 else clauses[0]) + ".")

    # Sentence 3: Highest/lowest since and 3-year average
    if highest_since_ogr is not None:
        since_str = pd.Timestamp(highest_since_ogr).strftime("%B %Y")
        if roll36_avg_ogr == roll36_avg_ogr and ogr == ogr:
            if ogr >= roll36_avg_ogr:
                sentences.append(f"{month_year} was the highest OGR since {since_str} and was higher than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")
            else:
                sentences.append(f"{month_year} was the highest OGR since {since_str} but was lower than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")
        else:
            sentences.append(f"{month_year} was the highest OGR since {since_str}.")
    elif lowest_since_ogr is not None:
        since_str = pd.Timestamp(lowest_since_ogr).strftime("%B %Y")
        if roll36_avg_ogr == roll36_avg_ogr and ogr == ogr:
            if ogr >= roll36_avg_ogr:
                sentences.append(f"{month_year} was the lowest OGR since {since_str} but was higher than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")
            else:
                sentences.append(f"{month_year} was the lowest OGR since {since_str} and was lower than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")
        else:
            sentences.append(f"{month_year} was the lowest OGR since {since_str}.")
    elif roll36_avg_ogr == roll36_avg_ogr and ogr == ogr:
        if ogr >= roll36_avg_ogr:
            sentences.append(f"{month_name} was higher than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")
        else:
            sentences.append(f"{month_name} was lower than the 3-year average growth rate of {_fmt_pct_signed(roll36_avg_ogr)}.")

    if not sentences:
        sentences.append(f"As of {month_year}, selected-slice AUM stood at {_fmt_currency(end_aum)}; organic growth was {_fmt_pct_signed(ogr) if ogr == ogr else '—'}.")
    return " ".join(sentences).strip()


class NarrativeWordingService:
    """Deterministic wording fallback until external narrative provider is wired."""

    def summarize(self, facts: NarrativeFacts) -> str:
        direction = "positive" if facts.nnb >= 0 else "negative"
        market_tone = "supportive" if facts.market_impact_rate >= 0 else "detractive"
        return (
            f"As of {facts.month_end}, End AUM (selected slice) is {_fmt_currency(facts.end_aum)}. "
            f"Net new business remained {direction} at {_fmt_currency(facts.nnb)}, "
            f"with organic growth of {_fmt_pct(facts.ogr)} and {market_tone} market movement of {_fmt_pct(facts.market_impact_rate)}."
        )


def _fmt_currency(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(v):
        return "-"
    return fmt_currency(v, unit="auto", decimals=2)


def _fmt_pct(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(v):
        return "-"
    return f"{v * 100:.2f}%"


def _fmt_pct_signed(x: Any, decimals: int = 2) -> str:
    """Institutional style: (+1.75%) or (-2.15%) for narrative commentary."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    pct = v * 100.0
    if pct > 0:
        return f"(+{pct:.{decimals}f}%)"
    if pct < 0:
        return f"({pct:.{decimals}f}%)"
    return "(0.00%)"


def _fmt_fee_yield(x: Any) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(v):
        return "-"
    return f"{_fmt_pct(v)} ({fmt_bps(v, decimals=0)})"


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def _tab1_snapshot() -> dict[str, Any]:
    """Current Tab 1 state with governed defaults as fallback (single source of truth)."""
    from app.config.tab1_defaults import get_tab1_dimension_keys
    keys = get_tab1_dimension_keys() + ["tab1_period"]
    defaults = {
        "tab1_period": TAB1_DEFAULT_PERIOD,
        "tab1_filter_channel": TAB1_DEFAULT_CHANNEL,
        "tab1_filter_sub_channel": TAB1_DEFAULT_SUB_CHANNEL,
        "tab1_filter_country": TAB1_DEFAULT_COUNTRY,
        "tab1_filter_segment": TAB1_DEFAULT_SEGMENT,
        "tab1_filter_sub_segment": TAB1_DEFAULT_SUB_SEGMENT,
        "tab1_filter_ticker": TAB1_DEFAULT_PRODUCT_TICKER,
    }
    return {k: st.session_state.get(k, defaults.get(k, "All")) for k in keys}


def _aum_scope_label() -> str:
    """Return a short, explicit label for the current portfolio scope (for AUM and related metrics). Governed defaults."""
    return get_scope_label_from_state(_tab1_snapshot())


def _render_aum_glossary() -> None:
    """Compact client-facing definitions so Enterprise vs Selected Slice vs Matrix Universe AUM cannot be confused."""
    with st.expander("What these AUM numbers mean", expanded=False):
        st.markdown(
            "- **Enterprise AUM (firm-wide):** Total assets under management across the entire firm. "
            "The single largest AUM number when you have not applied any segment, channel, or product filter.\n"
            "- **Selected-slice AUM:** AUM for the segment you have chosen (e.g. one channel, one country, or one product). "
            "The KPI card **End AUM (selected slice)** and the AUM waterfall use this.\n"
            "- **Matrix universe AUM:** Sum of product-level AUM in the Growth Quality Matrix only. "
            "Can be larger than selected-slice End AUM if the matrix includes products outside the current date/scope aggregation, or different; always refer to the chart caption for context."
        )


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


# Known ETF tickers for drill-down (client list). Detection: ticker in this set or label contains "ETF".
KNOWN_ETF_TICKERS = frozenset(
    {"AGG", "HYG", "TIP", "MUB", "MBB", "IUSB", "SUB"}
)


def _is_etf_ticker(series: pd.Series) -> pd.Series:
    """True where ticker is in KNOWN_ETF_TICKERS or string contains 'ETF'. Handles NaN/str."""
    if series.empty:
        return pd.Series(dtype=bool)
    up = series.astype(str).str.strip().str.upper()
    in_set = up.isin(KNOWN_ETF_TICKERS)
    has_etf = series.astype(str).str.contains("ETF", case=False, na=False)
    return in_set | has_etf


def _normalize_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["month_end"] = pd.to_datetime(out.get("month_end"), errors="coerce")

    channel_col = _pick_col(out, ("channel", "channel_l1", "channel_best", "channel_standard", "preferred_label"))
    sub_channel_col = _pick_col(out, ("channel_l2", "sub_channel"))
    country_col = _pick_col(out, ("src_country_canonical", "product_country_canonical", "geo", "country", "region"))
    segment_col = _pick_col(out, ("segment",))
    sub_segment_col = _pick_col(out, ("sub_segment", "subsegment"))
    ticker_col = _pick_col(out, ("product_ticker", "ticker", "label"))

    out["channel"] = out[channel_col].astype(str).str.strip() if channel_col else "Unassigned"
    out["sub_channel"] = out[sub_channel_col].astype(str).str.strip() if sub_channel_col else "Unassigned"
    out["country"] = out[country_col].astype(str).str.strip() if country_col else "Unassigned"
    out["segment"] = out[segment_col].astype(str).str.strip() if segment_col else "Unassigned"
    out["sub_segment"] = out[sub_segment_col].astype(str).str.strip() if sub_segment_col else "Unassigned"
    out["product_ticker"] = out[ticker_col].astype(str).str.strip() if ticker_col else "Unassigned"

    for col in ("begin_aum", "end_aum", "nnb", "nnf"):
        if col not in out.columns:
            flow_col = _pick_col(out, ("net_flow", "net_flows", "flow", "flows"))
            if col == "nnb" and flow_col:
                out["nnb"] = pd.to_numeric(out[flow_col], errors="coerce")
            else:
                out[col] = float("nan")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["month_end"])
    for col in ("channel", "sub_channel", "country", "segment", "sub_segment", "product_ticker"):
        out[col] = out[col].replace("", "Unassigned").fillna("Unassigned")
    return out


def _apply_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Tab 1 period windowing delegates to the canonical KPI service implementation."""
    return apply_period_canonical(df, period)


def _build_monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by month_end and derive market_impact, ogr, fee_yield. Prior-period: begin_aum = prior row end_aum when missing. For top-level KPIs use app.kpi.service.compute_kpi only."""
    if df.empty:
        return pd.DataFrame(columns=["month_end", "begin_aum", "end_aum", "nnb", "nnf", "market_impact", "ogr", "market_impact_rate", "fee_yield"])
    monthly = (
        df.groupby("month_end", as_index=False)[["begin_aum", "end_aum", "nnb", "nnf"]]
        .sum(min_count=1)
        .sort_values("month_end")
    )
    if monthly["begin_aum"].isna().all() and "end_aum" in monthly.columns:
        monthly["begin_aum"] = monthly["end_aum"].shift(1)

    monthly["market_impact"] = monthly.apply(
        lambda r: compute_market_impact(r.get("begin_aum"), r.get("end_aum"), r.get("nnb")), axis=1
    )
    monthly["ogr"] = monthly.apply(lambda r: compute_ogr(r.get("nnb"), r.get("begin_aum")), axis=1)
    monthly["market_impact_rate"] = monthly.apply(
        lambda r: compute_market_impact_rate(r.get("market_impact"), r.get("begin_aum")), axis=1
    )
    monthly["fee_yield"] = monthly.apply(
        lambda r: compute_fee_yield(r.get("nnf"), r.get("begin_aum"), r.get("end_aum"), nnb=r.get("nnb")), axis=1
    )
    return monthly


def _selectbox_with_all(label: str, key: str, options: list[str]) -> str:
    opts = ["All"] + sorted({o for o in options if isinstance(o, str) and o.strip()})
    if len(opts) == 1:
        opts.append("Unassigned")
    current = st.session_state.get(key, "All")
    if current not in opts:
        current = "All"
    return st.selectbox(label, opts, index=opts.index(current), key=key)


def _apply_business_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    s = _tab1_snapshot()
    filters = {
        "channel": s.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL),
        "sub_channel": s.get("tab1_filter_sub_channel", TAB1_DEFAULT_SUB_CHANNEL),
        "country": s.get("tab1_filter_country", TAB1_DEFAULT_COUNTRY),
        "segment": s.get("tab1_filter_segment", TAB1_DEFAULT_SEGMENT),
        "sub_segment": s.get("tab1_filter_sub_segment", TAB1_DEFAULT_SUB_SEGMENT),
        "product_ticker": s.get("tab1_filter_ticker", TAB1_DEFAULT_PRODUCT_TICKER),
    }
    for col, val in filters.items():
        if val not in (None, "", "All") and col in out.columns:
            out = out[out[col] == val]
    return out


def _apply_scoped_filters(df: pd.DataFrame, filter_map: dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    s = _tab1_snapshot()
    defaults = {"tab1_filter_channel": TAB1_DEFAULT_CHANNEL, "tab1_filter_sub_channel": TAB1_DEFAULT_SUB_CHANNEL, "tab1_filter_country": TAB1_DEFAULT_COUNTRY, "tab1_filter_segment": TAB1_DEFAULT_SEGMENT, "tab1_filter_sub_segment": TAB1_DEFAULT_SUB_SEGMENT, "tab1_filter_ticker": TAB1_DEFAULT_PRODUCT_TICKER}
    for col, key in filter_map.items():
        val = s.get(key, defaults.get(key, "All"))
        if col in out.columns and val not in (None, "", "All"):
            col_vals = out[col].astype(str).str.strip()
            supported = bool(((col_vals != "") & (col_vals != "Unassigned")).any())
            if not supported:
                continue
            out = out[out[col] == val]
    return out


def _institutional_note(title: str, detail: str) -> None:
    st.markdown(
        (
            f"<div class='availability-note'><strong>{title}:</strong> {detail}</div>"
        ),
        unsafe_allow_html=True,
    )


def _semantic_colors(values: pd.Series, *, pos: str = PALETTE["positive"], neg: str = PALETTE["negative"]) -> list[str]:
    series = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return [pos if float(v) >= 0 else neg for v in series]


def _render_core_metrics(kpi_result: KPIResult, firm_monthly: pd.DataFrame | None = None) -> None:
    """Render top-level KPIs from the single governed KPI result. Scope and period are explicit."""
    scope_suffix = " (firm-wide)" if kpi_result.scope_mode == "firm" else " (selected slice)"
    st.caption(f"**Active scope for KPI calculation:** {kpi_result.scope_label}{scope_suffix}")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("End AUM" + scope_suffix, _fmt_currency(kpi_result.end_aum))
    k2.metric("Net New Business", _fmt_currency(kpi_result.nnb))
    k3.metric("Net New Flow", _fmt_currency(kpi_result.nnf))
    k4.metric("Organic Growth", _fmt_pct(kpi_result.ogr))
    k5.metric("Market Movement", _fmt_currency(kpi_result.market_movement))

    st.caption(f"Scope: **{kpi_result.scope_label}**. Values are {kpi_result.scope_mode}-level; firm-wide only when scope is Firm-wide.")
    if firm_monthly is not None and not firm_monthly.empty and "end_aum" in firm_monthly.columns and kpi_result.scope_mode == "slice":
        firm_latest = firm_monthly.sort_values("month_end").iloc[-1]
        firm_end_aum = float(firm_latest.get("end_aum", float("nan")))
        if pd.notna(firm_end_aum) and pd.notna(kpi_result.end_aum) and firm_end_aum != 0:
            scope_share = kpi_result.end_aum / firm_end_aum
            st.caption(
                f"**Enterprise AUM (firm-wide):** {_fmt_currency(firm_end_aum)}. "
                f"**Selected-slice End AUM** (the card above) = {_fmt_pct(scope_share)} of enterprise AUM."
            )
    recon = kpi_result.begin_aum + kpi_result.nnb + kpi_result.market_movement - kpi_result.end_aum
    if pd.notna(recon):
        if abs(float(recon)) <= 1.0:
            st.caption("AUM consistency: Begin AUM + NNB + Market = End AUM (selected slice; reconciled).")
        else:
            st.caption(f"AUM consistency variance (selected slice): {_fmt_currency(recon)}. Verify source aggregation.")
    _render_aum_glossary()
    if kpi_result.validation.get("warnings"):
        with st.expander("KPI validation", expanded=False):
            st.json(kpi_result.validation)


def _build_driver_cards(
    ch: pd.DataFrame,
    tk: pd.DataFrame,
    monthly: pd.DataFrame,
) -> list[str]:
    """
    Build four commercial insight cards from verified metrics only.
    Themes: channel economics, fee yield mix shift, ETF concentration, pricing / growth quality.
    Returns list of exactly 4 strings, plain financial language.
    """
    fallback = "Insufficient data for this view in the selected slice."
    cards = [fallback, fallback, fallback, fallback]

    # 1) Channel economics - flow vs fee share gap
    if ch is not None and not ch.empty and ch["channel"].nunique() >= 2:
        total_nnb = float(ch["nnb"].sum())
        total_nnf = float(ch["nnf"].sum())
        if total_nnb != 0 or total_nnf != 0:
            lead_nnb = ch.sort_values("nnb", ascending=False).iloc[0]
            lead_channel = str(lead_nnb["channel"]).strip() or "Leading channel"
            nnb_pct = (float(lead_nnb["nnb"]) / total_nnb * 100.0) if total_nnb else 0.0
            fee_pct = (float(lead_nnb["nnf"]) / total_nnf * 100.0) if total_nnf else 0.0
            gap = nnb_pct - fee_pct
            if abs(gap) >= 1.0:
                if gap > 0:
                    cards[0] = f"{lead_channel}: {nnb_pct:.0f}% of flow, {fee_pct:.0f}% of fee revenue - pricing or mix is lagging flow."
                else:
                    cards[0] = f"{lead_channel}: {fee_pct:.0f}% of fee revenue on {nnb_pct:.0f}% of flow; margin quality ahead of scale."
            else:
                cards[0] = f"{lead_channel} in line: {nnb_pct:.0f}% NNB, {fee_pct:.0f}% fee flow."

    # 2) Fee yield mix - best channel and spread
    if ch is not None and not ch.empty and "end_aum" in ch.columns:
        fy = ch.copy()
        fy["fee_yield_proxy"] = fy.apply(
            lambda r: (float(r["nnf"]) / float(r["end_aum"])) if float(r.get("end_aum", 0)) else float("nan"), axis=1
        )
        fy = fy.dropna(subset=["fee_yield_proxy"])
        if len(fy) >= 2:
            best = fy.loc[fy["fee_yield_proxy"].idxmax()]
            worst = fy.loc[fy["fee_yield_proxy"].idxmin()]
            best_name = str(best["channel"]).strip() or "Top channel"
            best_fy = best["fee_yield_proxy"]
            spread = float(best_fy) - float(worst["fee_yield_proxy"])
            if spread > 0 and best_fy > 0:
                cards[1] = f"{best_name} leads fee yield at {_fmt_fee_yield(best_fy)}. Wide spread vs peers - scope to improve pricing or shift mix to higher-yield business."
            elif best_fy > 0:
                cards[1] = f"{best_name} best fee yield {_fmt_fee_yield(best_fy)}; benchmark for pricing and mix."
        elif len(fy) == 1 and float(fy.iloc[0]["fee_yield_proxy"]) > 0:
            best_name = str(fy.iloc[0]["channel"]).strip() or "Channel"
            cards[1] = f"{best_name} fee yield {_fmt_fee_yield(fy.iloc[0]['fee_yield_proxy'])} (single channel in scope)."

    # 3) ETF concentration
    if tk is not None and not tk.empty:
        etf = tk[_is_etf_ticker(tk["product_ticker"])].copy()
        if not etf.empty:
            etf_nnb = float(etf["nnb"].sum())
            total_nnb = float(tk["nnb"].sum())
            etf_top = etf.sort_values("nnb", ascending=False).iloc[0]
            top_ticker = str(etf_top["product_ticker"]).strip()
            share = (float(etf_top["nnb"]) / etf_nnb * 100.0) if etf_nnb else 0.0
            etf_pct_of_total = (etf_nnb / total_nnb * 100.0) if total_nnb else 0.0
            if share >= 50:
                cards[2] = f"{top_ticker} is {share:.0f}% of ETF NNB - concentrated. Decide: diversify or lean into concentration."
            elif etf_pct_of_total >= 30:
                cards[2] = f"ETFs {etf_pct_of_total:.0f}% of total NNB; {top_ticker} leads with {share:.0f}% of ETF flow."
            else:
                cards[2] = f"{top_ticker} leads ETF flow ({share:.0f}% of ETF NNB); ETFs are {etf_pct_of_total:.0f}% of total flow."

    # 4) Pricing / growth quality - fee yield trend
    if monthly is not None and len(monthly) >= 2:
        first = monthly.sort_values("month_end").iloc[0]
        last = monthly.sort_values("month_end").iloc[-1]
        fee_first = _coerce_num(first.get("fee_yield"))
        fee_last = _coerce_num(last.get("fee_yield"))
        if fee_first == fee_first and fee_last == fee_last:
            shift = fee_last - fee_first
            if abs(shift) >= 1e-6:
                if shift > 0:
                    cards[3] = f"Fee yield up {_fmt_fee_yield(shift)} over the period - mix tilting to higher-yield business."
                else:
                    cards[3] = f"Fee yield down {_fmt_fee_yield(abs(shift))}; review mix and pricing to protect margin."
            else:
                cards[3] = "Fee yield flat over the period; no material mix shift."
        else:
            ogr_last = _coerce_num(last.get("ogr"))
            if ogr_last == ogr_last:
                cards[3] = f"OGR {_fmt_pct(ogr_last)}. Track fee yield when available to link growth to margin."
    elif monthly is not None and len(monthly) == 1:
        ogr = _coerce_num(monthly.iloc[0].get("ogr"))
        if ogr == ogr:
            cards[3] = f"OGR {_fmt_pct(ogr)} this period. Widen date range for mix-shift and growth-quality trends."

    return cards


def _render_narrative_and_drivers(
    monthly: pd.DataFrame,
    monthly_full: pd.DataFrame,
    channel_scoped: pd.DataFrame,
    ticker_scoped: pd.DataFrame,
    firm_monthly: pd.DataFrame | None = None,
    kpi_result: KPIResult | None = None,
) -> None:
    _section_header(
        "Portfolio Snapshot",
        "What this shows: executive summary and key drivers for your selected scope. Why it matters: quick read on growth, market contribution, and where to focus.",
    )
    if monthly.empty:
        st.info("No portfolio data is available for the selected filter set.")
        return
    if kpi_result is None:
        raise ValueError("_render_narrative_and_drivers requires canonical kpi_result.")

    scope = kpi_result.scope_label
    context = _build_hero_narrative_context(monthly, monthly_full)
    hero_paragraph = _build_hero_narrative_paragraph(context, scope)
    st.markdown(
        (
            "<div class='hero-narrative'>"
            "<div class='hero-narrative-label'>Executive summary</div>"
            f"<div class='hero-narrative-text'>{hero_paragraph}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    _render_core_metrics(kpi_result, firm_monthly)

    ch = channel_scoped.groupby("channel", as_index=False).agg(nnb=("nnb", "sum"), nnf=("nnf", "sum"), end_aum=("end_aum", "sum"))
    tk = ticker_scoped.groupby("product_ticker", as_index=False).agg(nnb=("nnb", "sum"), nnf=("nnf", "sum"), end_aum=("end_aum", "sum"))

    driver_text = _build_driver_cards(ch, tk, monthly)

    cols = st.columns(4)
    for col, text in zip(cols, driver_text):
        col.markdown(
            (
                f"<div style='border:1px solid {PALETTE['grid']};border-radius:10px;padding:12px;background:{PALETTE['surface']};min-height:118px;'>"
                f"{text}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _render_aum_waterfall(monthly: pd.DataFrame, kpi_result: KPIResult) -> None:
    _section_header("1) AUM Waterfall", "What this shows: how opening selected-slice AUM moved to closing selected-slice AUM via flows and market. Why it matters: confirms reconciliation and shows the bridge for your selected slice.")
    if go is None or monthly.empty:
        _institutional_note("AUM Waterfall", "The selected scope does not have enough data points for a reconciled AUM bridge.")
        return
    begin_aum = float(kpi_result.begin_aum) if kpi_result.begin_aum == kpi_result.begin_aum else 0.0
    nnb = float(kpi_result.nnb) if kpi_result.nnb == kpi_result.nnb else 0.0
    market = float(kpi_result.market_movement) if kpi_result.market_movement == kpi_result.market_movement else 0.0
    end_aum = float(kpi_result.end_aum) if kpi_result.end_aum == kpi_result.end_aum else 0.0
    x = ["Begin AUM (slice)", "NNB", "Market Impact", "End AUM (slice)"]
    start_after_begin = begin_aum
    end_after_nnb = begin_aum + nnb
    end_after_market = begin_aum + nnb + market
    bases = [
        0.0,
        min(start_after_begin, end_after_nnb),
        min(end_after_nnb, end_after_market),
        0.0,
    ]
    heights = [
        begin_aum,
        abs(nnb),
        abs(market),
        end_aum,
    ]
    colors = [
        PALETTE["primary"],
        PALETTE["positive"] if nnb >= 0 else PALETTE["negative"],
        PALETTE["market"],
        PALETTE["primary"],
    ]
    custom = [_fmt_currency(begin_aum), _fmt_currency(nnb), _fmt_currency(market), _fmt_currency(end_aum)]
    fig = go.Figure(
        go.Bar(
            x=x,
            y=heights,
            base=bases,
            marker=dict(color=colors, line=dict(color=PALETTE["grid"], width=1)),
            customdata=custom,
            hovertemplate="%{x}<br>Value: %{customdata}<extra></extra>",
        )
    )
    connector_y = [begin_aum, end_after_nnb, end_after_market, end_aum]
    fig.add_scatter(
        x=x,
        y=connector_y,
        mode="lines",
        line=dict(color=PALETTE["grid"], width=1.2, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
    apply_enterprise_plotly_style(fig, height=330)
    safe_render_plotly(fig)
    scope = _aum_scope_label()
    st.caption(f"Scope: **{scope}**. All amounts above are **selected-slice AUM** only. Not enterprise AUM unless scope is Firm-wide.")


def _render_channel_breakdown(channel_scoped: pd.DataFrame) -> None:
    _section_header(
        "2) Distribution Channel Breakdown (NNB + NNF)",
        "What this shows: NNB and NNF by channel (bars or treemap), with optional drill to sub-channels. Why it matters: see where flow is concentrated and where to focus distribution.",
    )
    if go is None or channel_scoped.empty:
        _institutional_note("Distribution View", "Channel flow data in the selected scope is insufficient for comparative NNB and NNF analysis.")
        return
    by_channel = channel_scoped.groupby("channel", as_index=False).agg(nnb=("nnb", "sum"), nnf=("nnf", "sum"))
    by_channel = by_channel.sort_values("nnb", ascending=False).head(15)
    if by_channel.empty or by_channel["channel"].nunique() < 2:
        _institutional_note("Distribution View", "Flow contribution is concentrated in a single channel under the current slice.")
        return

    st.markdown("**Step 1 - Chart type:** Choose how to view channels (grouped bars or treemap).")
    view_mode = st.radio(
        "View channels as",
        ["Grouped bars", "Treemap"],
        horizontal=True,
        key="tab1_channel_breakdown_view",
        help="Grouped bars: side-by-side NNB and NNF per channel. Treemap: size = magnitude, color = sign.",
        label_visibility="collapsed",
    )

    if view_mode == "Treemap":
        t1, t2 = st.columns(2)
        tm = by_channel.copy()
        tm["nnb_abs"] = tm["nnb"].abs()
        tm["nnf_abs"] = tm["nnf"].abs()
        with t1:
            fig_nnb = go.Figure(
                go.Treemap(
                    labels=tm["channel"],
                    parents=[""] * len(tm),
                    values=tm["nnb_abs"],
                    marker=dict(
                        colors=tm["nnb"],
                        colorscale=[[0.0, PALETTE["negative"]], [0.5, PALETTE["neutral"]], [1.0, PALETTE["positive"]]],
                        cmid=0.0,
                    ),
                    customdata=tm["nnb"].map(_fmt_currency),
                    hovertemplate="Channel: %{label}<br>NNB: %{customdata}<extra></extra>",
                )
            )
            fig_nnb.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), title="NNB by Channel (treemap)")
            apply_enterprise_plotly_style(fig_nnb, height=330)
            safe_render_plotly(fig_nnb)
        with t2:
            fig_nnf = go.Figure(
                go.Treemap(
                    labels=tm["channel"],
                    parents=[""] * len(tm),
                    values=tm["nnf_abs"],
                    marker=dict(
                        colors=tm["nnf"],
                        colorscale=[[0.0, PALETTE["negative"]], [0.5, PALETTE["neutral"]], [1.0, PALETTE["positive"]]],
                        cmid=0.0,
                    ),
                    customdata=tm["nnf"].map(_fmt_currency),
                    hovertemplate="Channel: %{label}<br>NNF: %{customdata}<extra></extra>",
                )
            )
            fig_nnf.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), title="NNF by Channel (treemap)")
            apply_enterprise_plotly_style(fig_nnf, height=330)
            safe_render_plotly(fig_nnf)
    else:
        fig = go.Figure()
        fig.add_bar(
            x=by_channel["channel"],
            y=by_channel["nnb"],
            name="NNB",
            marker_color=_semantic_colors(by_channel["nnb"]),
            customdata=by_channel["nnb"].map(_fmt_currency),
            hovertemplate="Channel: %{x}<br>NNB: %{customdata}<extra></extra>",
        )
        fig.add_bar(
            x=by_channel["channel"],
            y=by_channel["nnf"],
            name="NNF",
            marker_color=_semantic_colors(by_channel["nnf"]),
            customdata=by_channel["nnf"].map(_fmt_currency),
            hovertemplate="Channel: %{x}<br>NNF: %{customdata}<extra></extra>",
        )
        fig.update_layout(barmode="group", height=360, margin=dict(l=20, r=20, t=40, b=40), xaxis_title="Channel")
        apply_enterprise_plotly_style(fig, height=360)
        safe_render_plotly(fig)

    if "sub_channel" in channel_scoped.columns:
        ch_opts = by_channel["channel"].astype(str).tolist()
        st.markdown("---")
        st.markdown("**Step 2 - Drill to sub-channel (optional):** Select a channel to see NNB and NNF by sub-channel.")
        drill_channel = st.selectbox(
            "Select a channel to drill into sub-channel detail",
            options=["- View channels only (no drill) -"] + ch_opts,
            index=0,
            key="tab1_channel_drill",
            help="Choosing a channel shows a second chart below with that channel's sub-channels.",
        )
        if drill_channel and drill_channel != "- View channels only (no drill) -":
            sub = channel_scoped[channel_scoped["channel"].astype(str) == str(drill_channel)].copy()
            by_sub = sub.groupby("sub_channel", as_index=False).agg(nnb=("nnb", "sum"), nnf=("nnf", "sum"))
            by_sub = by_sub.sort_values("nnb", ascending=False).head(12)
            if not by_sub.empty and by_sub["sub_channel"].nunique() >= 1:
                sub_fig = go.Figure()
                sub_fig.add_bar(
                    x=by_sub["sub_channel"],
                    y=by_sub["nnb"],
                    name="NNB",
                    marker_color=_semantic_colors(by_sub["nnb"]),
                    customdata=by_sub["nnb"].map(_fmt_currency),
                    hovertemplate="Sub-channel: %{x}<br>NNB: %{customdata}<extra></extra>",
                )
                sub_fig.add_bar(
                    x=by_sub["sub_channel"],
                    y=by_sub["nnf"],
                    name="NNF",
                    marker_color=_semantic_colors(by_sub["nnf"]),
                    customdata=by_sub["nnf"].map(_fmt_currency),
                    hovertemplate="Sub-channel: %{x}<br>NNF: %{customdata}<extra></extra>",
                )
                sub_fig.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=20, r=20, t=35, b=40),
                    title=f"Sub-channel detail: {drill_channel}",
                    xaxis_title="Sub-channel",
                )
                apply_enterprise_plotly_style(sub_fig, height=320)
                safe_render_plotly(sub_fig)
        else:
            st.caption("Select a channel above to see its sub-channel breakdown here.")
        st.caption("All data is for the selected scope. Use the dropdown to drill from any channel into its sub-channels.")
    else:
        st.caption("All data is for the selected scope. Sub-channel drill is not available for this slice.")


def _render_growth_quality_matrix(ticker_scoped: pd.DataFrame, monthly: pd.DataFrame) -> None:
    _section_header(
        "3) Growth Quality Matrix",
        "What this shows: products by growth contribution (x) and fee-yield quality (y); size = product AUM in this matrix, color = market contribution. Why it matters: spot star performers, pricing issues, and underutilised opportunities.",
    )
    _render_aum_glossary()
    if go is None or ticker_scoped.empty:
        _institutional_note("Growth Quality Matrix", "Product-level flow and pricing data in the selected scope is insufficient for quadrant analysis.")
        return
    mat = ticker_scoped.groupby("product_ticker", as_index=False).agg(
        nnb=("nnb", "sum"),
        nnf=("nnf", "sum"),
        aum=("end_aum", "sum"),
        begin_aum=("begin_aum", "sum"),
    )
    mat = mat[mat["aum"].fillna(0) > 0]
    if mat.empty or mat["product_ticker"].nunique() < 2:
        _institutional_note("Growth Quality Matrix", "The selected slice is too concentrated for meaningful cross-product quality comparison.")
        return
    # Client: Fee Yield = NNF / NNB; use NNF/NNB when NNB > 0 to avoid divide-by-zero
    def _fee_yield_row(r: pd.Series) -> float:
        nnb_val = r.get("nnb")
        nnf_val = r.get("nnf")
        aum_val = r.get("aum")
        if nnb_val is not None and pd.notna(nnb_val) and float(nnb_val) > 0:
            return compute_fee_yield_nnf_nnb(nnf_val, nnb_val)
        return compute_fee_yield(nnf_val, aum_val, aum_val, nnb=nnb_val)
    mat["fee_yield"] = mat.apply(_fee_yield_row, axis=1)
    mat["fee_yield_pct"] = mat["fee_yield"].fillna(0) * 100
    mat["market_impact_abs"] = mat["aum"] - mat["begin_aum"] - mat["nnb"]
    mat["nnb_m"] = mat["nnb"] / 1_000_000.0
    mat["market_impact_m"] = mat["market_impact_abs"] / 1_000_000.0
    x_med = float(mat["nnb_m"].median())
    y_med = float(mat["fee_yield_pct"].median())
    aum_max = max(float(mat["aum"].max()), 1.0)
    mat["size"] = mat["aum"].apply(lambda x: max(9.0, 40.0 * (float(x) / aum_max) ** 0.5))

    def _quadrant_row(r: pd.Series) -> str:
        if float(r["nnb_m"]) >= x_med and float(r["fee_yield_pct"]) >= y_med:
            return "High Growth / High Yield"
        if float(r["nnb_m"]) >= x_med and float(r["fee_yield_pct"]) < y_med:
            return "High Growth / Low Yield"
        if float(r["nnb_m"]) < x_med and float(r["fee_yield_pct"]) >= y_med:
            return "Low Growth / High Yield"
        return "Low Growth / Low Yield"

    action_map = {
        "High Growth / High Yield": "Scale aggressively",
        "High Growth / Low Yield": "Grow then optimize monetization",
        "Low Growth / High Yield": "Protect margin",
        "Low Growth / Low Yield": "Review / reposition",
    }
    quadrant_label_map = {
        "High Growth / High Yield": "Star Performer",
        "High Growth / Low Yield": "Underutilised Opportunity",
        "Low Growth / High Yield": "Pricing Issue",
        "Low Growth / Low Yield": "Review Required",
    }
    mat["quadrant"] = mat.apply(_quadrant_row, axis=1)
    mat["strategic_action"] = mat["quadrant"].map(action_map).fillna("Review")
    mat["quadrant_label"] = mat["quadrant"].map(quadrant_label_map).fillna("Review Required")

    total_aum = float(mat["aum"].sum())
    total_begin = float(mat["begin_aum"].sum())
    total_nnb = float(mat["nnb"].sum())
    total_nnf = float(mat["nnf"].sum())
    total_market = float(mat["market_impact_abs"].sum())
    ogr_total = (total_nnb / total_begin) if total_begin else float("nan")
    fee_total = compute_fee_yield_nnf_nnb(total_nnf, total_nnb) if total_nnb > 0 else float("nan")

    scope = _aum_scope_label()
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Matrix universe AUM (sum of products in chart)", _fmt_currency(total_aum))
    k2.metric("Net New Business", _fmt_currency(total_nnb))
    k3.metric("Organic Growth Rate", _fmt_pct(ogr_total))
    k4.metric("Fee Yield", _fmt_fee_yield(fee_total))
    k5.metric("Market Impact", _fmt_currency(total_market))
    hh_share = float((mat["quadrant"] == "High Growth / High Yield").sum()) / max(float(len(mat)), 1.0)
    k6.metric("Composite Opportunity", _fmt_pct(hh_share))

    top_growth = mat.sort_values(["nnb_m", "product_ticker"], ascending=[False, True]).iloc[0]
    scaleable = mat[mat["nnb_m"] >= x_med].copy()
    if scaleable.empty:
        scaleable = mat
    best_yield_scaleable = scaleable.sort_values(["fee_yield_pct", "product_ticker"], ascending=[False, True]).iloc[0]
    biggest_drag = mat.sort_values(["nnb_m", "product_ticker"], ascending=[True, True]).iloc[0]
    needs_attention = mat[mat["quadrant"] == "Low Growth / Low Yield"].copy()
    if needs_attention.empty:
        needs_attention = mat
    needs_attention = needs_attention.sort_values(["nnb", "fee_yield_pct"], ascending=[True, True]).iloc[0]

    i1, i2, i3, i4 = st.columns(4)
    i1.markdown(
        f"<div class='section-frame'><strong>Top growth driver</strong><br>{top_growth['product_ticker']} contributing {_fmt_currency(top_growth['nnb'])} in NNB.</div>",
        unsafe_allow_html=True,
    )
    i2.markdown(
        f"<div class='section-frame'><strong>Highest-yield scalable product</strong><br>{best_yield_scaleable['product_ticker']} at {_fmt_fee_yield(best_yield_scaleable['fee_yield'])} with positive growth scale.</div>",
        unsafe_allow_html=True,
    )
    i3.markdown(
        f"<div class='section-frame'><strong>Biggest drag</strong><br>{biggest_drag['product_ticker']} with {_fmt_currency(biggest_drag['nnb'])} NNB requires remediation.</div>",
        unsafe_allow_html=True,
    )
    i4.markdown(
        f"<div class='section-frame'><strong>Product needing attention</strong><br>{needs_attention['product_ticker']} sits in {needs_attention['quadrant_label']} and is flagged for {needs_attention['strategic_action'].lower()}.</div>",
        unsafe_allow_html=True,
    )
    st.caption("**Matrix universe AUM** = sum of product AUM in this chart. **Selected-slice End AUM** = KPI from the top of the page for your current scope. **Enterprise AUM** = firm-wide total (see above when scope is a slice).")
    scope_end = float("nan")
    if monthly is not None and not monthly.empty and "end_aum" in monthly.columns:
        scope_end = float(monthly.sort_values("month_end").iloc[-1].get("end_aum", float("nan")))
    if pd.notna(scope_end) and scope_end != 0:
        coverage = total_aum / scope_end
        st.caption(
            f"Scope: **{scope}**. Matrix universe AUM: {_fmt_currency(total_aum)}. "
            f"Selected-slice End AUM: {_fmt_currency(scope_end)} (matrix = {_fmt_pct(coverage)} of slice)."
        )
    else:
        st.caption(
            f"Scope: **{scope}**. Matrix universe AUM above = sum of product End AUM in this matrix (selected slice only)."
        )

    top_n_labels = 8
    label_candidates = pd.concat(
        [
            mat.nlargest(top_n_labels, "nnb_m"),
            mat.nlargest(top_n_labels, "aum"),
        ],
        ignore_index=True,
    )
    nnb_std = float(mat["nnb_m"].std()) if pd.notna(mat["nnb_m"].std()) else 0.0
    fy_std = float(mat["fee_yield_pct"].std()) if pd.notna(mat["fee_yield_pct"].std()) else 0.0
    nnb_outliers = mat[abs(mat["nnb_m"] - mat["nnb_m"].median()) > (1.8 * nnb_std)] if nnb_std > 0 else pd.DataFrame()
    fy_outliers = mat[abs(mat["fee_yield_pct"] - mat["fee_yield_pct"].median()) > (1.8 * fy_std)] if fy_std > 0 else pd.DataFrame()
    label_candidates = pd.concat([label_candidates, nnb_outliers, fy_outliers], ignore_index=True).drop_duplicates(subset=["product_ticker"])
    label_set = set(label_candidates["product_ticker"].astype(str).tolist())
    mat["display_label"] = mat["product_ticker"].astype(str).where(mat["product_ticker"].astype(str).isin(label_set), "")

    fig = go.Figure(
        go.Scatter(
            x=mat["nnb_m"],
            y=mat["fee_yield_pct"],
            mode="markers+text",
            text=mat["display_label"],
            textposition="top center",
            marker=dict(
                size=mat["size"],
                color=mat["market_impact_m"],
                colorscale=[[0.0, PALETTE["negative"]], [0.5, "#f3f4f6"], [1.0, PALETTE["positive"]]],
                showscale=True,
                cmin=float(mat["market_impact_m"].min()),
                cmax=float(mat["market_impact_m"].max()),
                cmid=0.0,
                colorbar=dict(title="Market Contribution ($)"),
                line=dict(width=1, color=PALETTE["grid"]),
                opacity=0.88,
            ),
            customdata=pd.DataFrame(
                {
                    "ticker": mat["product_ticker"].astype(str),
                    "nnb_fmt": mat["nnb"].map(_fmt_currency),
                    "fy_fmt": mat["fee_yield"].map(_fmt_fee_yield),
                    "aum_fmt": mat["aum"].map(_fmt_currency),
                    "mkt_fmt": mat["market_impact_abs"].map(_fmt_currency),
                    "q": mat["quadrant_label"],
                    "a": mat["strategic_action"],
                }
            ),
            hovertemplate=(
                "Product: %{customdata[0]}<br>"
                "Growth Contribution: %{customdata[1]}<br>"
                "Fee Yield Quality: %{customdata[2]}<br>"
                "Product AUM (matrix): %{customdata[3]}<br>"
                "Market Contribution: %{customdata[4]}<br>"
                "Quadrant: %{customdata[5]}<br>"
                "Action: %{customdata[6]}<extra></extra>"
            ),
        )
    )
    highlight_df = pd.DataFrame([top_growth, best_yield_scaleable, biggest_drag]).drop_duplicates(subset=["product_ticker"])
    fig.add_trace(
        go.Scatter(
            x=highlight_df["nnb_m"],
            y=highlight_df["fee_yield_pct"],
            mode="markers",
            marker=dict(
                size=highlight_df["size"] + 7,
                color="rgba(0,0,0,0)",
                line=dict(color=PALETTE["market"], width=2.4),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_vline(x=x_med, line_dash="dash", line_color=PALETTE["neutral"], line_width=1.8)
    fig.add_hline(y=y_med, line_dash="dash", line_color=PALETTE["neutral"], line_width=1.8)
    x_lo = float(mat["nnb_m"].min())
    x_hi = float(mat["nnb_m"].max())
    y_lo = float(mat["fee_yield_pct"].min())
    y_hi = float(mat["fee_yield_pct"].max())
    if x_hi != x_lo and y_hi != y_lo:
        x_left = x_lo + (x_med - x_lo) * 0.5
        x_right = x_med + (x_hi - x_med) * 0.5
        y_low = y_lo + (y_med - y_lo) * 0.5
        y_high = y_med + (y_hi - y_med) * 0.5
        fig.add_annotation(x=x_right, y=y_high, text="<b>Star Performer</b><br>Scale aggressively", showarrow=False, align="center", font=dict(color=PALETTE["text_muted"], size=11))
        fig.add_annotation(x=x_right, y=y_low, text="<b>Underutilised Opportunity</b><br>Grow then optimize monetization", showarrow=False, align="center", font=dict(color=PALETTE["text_muted"], size=11))
        fig.add_annotation(x=x_left, y=y_high, text="<b>Pricing Issue</b><br>Protect margin", showarrow=False, align="center", font=dict(color=PALETTE["text_muted"], size=11))
        fig.add_annotation(x=x_left, y=y_low, text="<b>Review Required</b><br>Review / reposition", showarrow=False, align="center", font=dict(color=PALETTE["text_muted"], size=11))
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=30),
        xaxis_title="Net Growth Contribution ($M)",
        yaxis_title="Fee Yield (%)",
    )
    apply_enterprise_plotly_style(fig, height=500)
    fig.update_xaxes(tickformat=",.1f", ticksuffix="M", gridcolor="rgba(127,147,188,0.25)")
    fig.update_yaxes(tickformat=".2f", ticksuffix="%", gridcolor="rgba(127,147,188,0.25)")
    safe_render_plotly(fig)
    st.caption("Key products and outliers are labeled; hover for detail. Quadrants guide where to invest, improve pricing, or review.")
    st.markdown(
        (
            "<div class='availability-note' style='margin-top:0.35rem; margin-bottom:0.7rem; padding:0.7rem 0.85rem;'>"
            "<strong>How to read this chart:</strong> "
            "X axis = growth contribution, Y axis = fee yield, bubble size = product AUM in this matrix, color = market contribution."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    priority = mat.copy()
    priority["growth_rank"] = priority["nnb_m"].rank(pct=True, method="average")
    priority["yield_rank"] = priority["fee_yield_pct"].rank(pct=True, method="average")
    priority["scale_rank"] = priority["aum"].rank(pct=True, method="average")
    priority["priority_score"] = (0.45 * priority["growth_rank"]) + (0.35 * priority["yield_rank"]) + (0.20 * priority["scale_rank"])
    priority = priority.sort_values(["priority_score", "nnb_m"], ascending=[False, False]).head(12)
    action_table = priority.rename(
        columns={
            "product_ticker": "Product",
            "quadrant_label": "Quadrant",
            "nnb": "Growth Contribution",
            "fee_yield": "Fee Yield",
            "aum": "Product AUM (matrix)",
            "strategic_action": "Strategic Action",
        }
    )[
        ["Product", "Quadrant", "Growth Contribution", "Fee Yield", "Product AUM (matrix)", "Strategic Action"]
    ]
    action_formats = infer_common_formats(action_table)
    action_formats["Growth Contribution"] = lambda x: fmt_currency(x, unit="auto", decimals=1)
    action_formats["Product AUM (matrix)"] = lambda x: fmt_currency(x, unit="auto", decimals=2)
    action_formats["Fee Yield"] = lambda x: (_fmt_pct(x) if pd.notna(pd.to_numeric(x, errors="coerce")) else "-")
    st.dataframe(format_df(action_table, action_formats), use_container_width=True, hide_index=True)
    st.caption("Products are ranked by growth, fee yield, and scale to support prioritization by quadrant.")

    top_strip = priority.head(3)["product_ticker"].astype(str).tolist()
    if top_strip and "month_end" in ticker_scoped.columns:
        trend_src = ticker_scoped[ticker_scoped["product_ticker"].astype(str).isin(top_strip)].copy()
        trend_src = trend_src.groupby(["month_end", "product_ticker"], as_index=False)["nnb"].sum()
        trend_src["month_end"] = pd.to_datetime(trend_src["month_end"], errors="coerce")
        trend_src = trend_src.dropna(subset=["month_end"]).sort_values(["product_ticker", "month_end"])
        if not trend_src.empty:
            strip = go.Figure()
            for tkr in top_strip:
                tdf = trend_src[trend_src["product_ticker"].astype(str) == tkr]
                if tdf.empty:
                    continue
                strip.add_scatter(
                    x=tdf["month_end"],
                    y=tdf["nnb"],
                    mode="lines",
                    name=tkr,
                    hovertemplate=f"{tkr}<br>Month: %{{x|%b %Y}}<br>NNB: %{{y:,.0f}}<extra></extra>",
                )
            strip.update_layout(height=180, margin=dict(l=20, r=20, t=25, b=20), title="Compact trend strip: top priority products")
            apply_enterprise_plotly_style(strip, height=180)
            safe_render_plotly(strip)


def _etf_flag(by_etf: pd.DataFrame) -> pd.Series:
    """High NNB + low fee yield => red risk flag. Low NNB + high fee yield => green opportunity flag."""
    fee_yield = by_etf.apply(
        lambda r: (float(r["nnf"]) / float(r["nnb"])) if r["nnb"] and float(r["nnb"]) != 0 else float("nan"), axis=1
    )
    nnb_med = by_etf["nnb"].median()
    fy_med = fee_yield.median()
    flag = []
    for i, row in by_etf.iterrows():
        nnb, fy = float(row["nnb"]), fee_yield.loc[i] if i in fee_yield.index else float("nan")
        if fy == fy and nnb_med == nnb_med and fy_med == fy_med:
            if nnb >= nnb_med and fy < fy_med:
                flag.append("Red Flag: Pricing Issue")
            elif nnb < nnb_med and fy >= fy_med:
                flag.append("Green Flag: Yield Opportunity")
            else:
                flag.append("Neutral")
        else:
            flag.append("Neutral")
    return pd.Series(flag, index=by_etf.index)


def _render_etf_drilldown(ticker_scoped: pd.DataFrame) -> None:
    _section_header(
        "4) ETF Drill-Down",
        "What this shows: top ETFs by NNB and NNF with pricing flags. Why it matters: red = pricing issue (high NNB, low fee yield); green = opportunity (low NNB, high fee yield).",
    )
    if go is None or ticker_scoped.empty:
        _institutional_note("ETF Drill-Down", "ETF-labelled product data is not available for the selected scope.")
        return
    etf = ticker_scoped[_is_etf_ticker(ticker_scoped["product_ticker"])].copy()
    if etf.empty:
        _institutional_note("ETF Drill-Down", "No ETF-labelled tickers are present in the selected period and scope.")
        return

    by_etf = etf.groupby("product_ticker", as_index=False).agg(
        nnb=("nnb", "sum"),
        nnf=("nnf", "sum"),
        aum=("end_aum", "sum"),
    )
    by_etf["rank_nnb"] = by_etf["nnb"].rank(method="first", ascending=False).astype(int)
    by_etf["rank_nnf"] = by_etf["nnf"].rank(method="first", ascending=False).astype(int)
    by_etf["Flag"] = _etf_flag(by_etf)

    by_nnb = by_etf.sort_values(["nnb", "product_ticker"], ascending=[False, True]).head(10)
    by_nnf = by_etf.sort_values(["nnf", "product_ticker"], ascending=[False, True]).head(10)

    def _bar_colors_by_flag(df: pd.DataFrame) -> list[str]:
        return [
            PALETTE["negative"]
            if f == "Red Flag: Pricing Issue"
            else (PALETTE["positive"] if f == "Green Flag: Yield Opportunity" else PALETTE["neutral"])
            for f in df["Flag"]
        ]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(
            go.Bar(
                x=by_nnb["product_ticker"],
                y=by_nnb["nnb"],
                marker_color=_bar_colors_by_flag(by_nnb),
                customdata=by_nnb["nnb"].map(_fmt_currency),
                hovertemplate="ETF: %{x}<br>NNB: %{customdata}<extra></extra>",
            )
        )
        if "Flag" in by_nnb.columns:
            fig.update_traces(
                customdata=list(zip(by_nnb["nnb"].map(_fmt_currency).tolist(), by_nnb["Flag"].tolist())),
                hovertemplate="ETF: %{x}<br>NNB: %{customdata[0]}<br>Flag: %{customdata[1]}<extra></extra>",
            )
        else:
            fig.update_traces(hovertemplate="ETF: %{x}<br>NNB: %{customdata}<extra></extra>")
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="ETF Ticker",
            yaxis_title="NNB",
            title={"text": "Top 10 ETFs by Net New Business", "font": {"size": 14, "color": PALETTE["text"]}},
        )
        apply_enterprise_plotly_style(fig, height=320)
        safe_render_plotly(fig)
    with c2:
        fig2 = go.Figure(
            go.Bar(
                x=by_nnf["product_ticker"],
                y=by_nnf["nnf"],
                marker_color=_bar_colors_by_flag(by_nnf),
                customdata=by_nnf["nnf"].map(_fmt_currency),
                hovertemplate="ETF: %{x}<br>NNF: %{customdata}<extra></extra>",
            )
        )
        if "Flag" in by_nnf.columns:
            fig2.update_traces(
                customdata=list(zip(by_nnf["nnf"].map(_fmt_currency).tolist(), by_nnf["Flag"].tolist())),
                hovertemplate="ETF: %{x}<br>NNF: %{customdata[0]}<br>Flag: %{customdata[1]}<extra></extra>",
            )
        else:
            fig2.update_traces(hovertemplate="ETF: %{x}<br>NNF: %{customdata}<extra></extra>")
        fig2.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="ETF Ticker",
            yaxis_title="NNF",
            title={"text": "Top 10 ETFs by Net New Flow", "font": {"size": 14, "color": PALETTE["text"]}},
        )
        apply_enterprise_plotly_style(fig2, height=320)
        safe_render_plotly(fig2)

    scope = _aum_scope_label()
    etf_display = by_etf.rename(
        columns={
            "product_ticker": "ETF Ticker",
            "nnb": "NNB",
            "nnf": "NNF",
            "aum": "End AUM (per ETF)",
            "rank_nnb": "Rank (NNB)",
            "rank_nnf": "Rank (NNF)",
        }
    )
    if "Flag" not in etf_display.columns:
        etf_display["Flag"] = by_etf["Flag"].values
    col_order = ["ETF Ticker", "Flag", "NNB", "NNF", "End AUM (per ETF)", "Rank (NNB)", "Rank (NNF)"]
    etf_display = etf_display[[c for c in col_order if c in etf_display.columns]]
    etf_display = format_df(etf_display, infer_common_formats(etf_display))

    def _flag_style(v: str) -> str:
        if v == "Red Flag: Pricing Issue":
            return "background-color: #ef4444; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 600;"
        if v == "Green Flag: Yield Opportunity":
            return "background-color: #22c55e; color: #fff; padding: 2px 8px; border-radius: 4px; font-weight: 600;"
        return ""

    styled = etf_display.style.apply(
        lambda col: [_flag_style(v) for v in col] if col.name == "Flag" else [""] * len(col),
        axis=0,
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        f"Scope: **{scope}**. Red = pricing issue; green = opportunity. Table and charts use the same flags."
    )


def _resample_to_quarter(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly to quarter: first begin_aum, last end_aum, sum nnb/market_impact; compute ogr and market_impact_rate."""
    if monthly.empty or "month_end" not in monthly.columns:
        return pd.DataFrame()
    m = monthly.copy()
    m["month_end"] = pd.to_datetime(m["month_end"], errors="coerce")
    m = m.dropna(subset=["month_end"]).sort_values("month_end")
    m["quarter"] = m["month_end"].dt.to_period("Q")
    agg = m.groupby("quarter", as_index=False).agg(
        begin_aum=("begin_aum", "first"),
        end_aum=("end_aum", "last"),
        nnb=("nnb", "sum"),
        market_impact=("market_impact", "sum"),
    )
    agg["month_end"] = agg["quarter"].dt.to_timestamp("M") + pd.offsets.MonthEnd(0)
    agg["ogr"] = agg.apply(lambda r: compute_ogr(r["nnb"], r["begin_aum"]), axis=1)
    agg["market_impact_rate"] = agg.apply(
        lambda r: compute_market_impact_rate(r["market_impact"], r["begin_aum"]), axis=1
    )
    return agg


def _prior_year_series(
    df: pd.DataFrame,
    full: pd.DataFrame,
    period_col: str,
    value_cols: list[str],
    offset_months: int = 12,
) -> pd.DataFrame:
    """Merge prior-year values: for each row in df, attach full's value from (period - offset_months)."""
    if df.empty or full.empty or period_col not in full.columns:
        return df.copy()
    out = df.copy()
    out["_dt"] = pd.to_datetime(out[period_col], errors="coerce")
    full = full.sort_values(period_col).drop_duplicates(subset=[period_col], keep="last")
    full["_dt"] = pd.to_datetime(full[period_col], errors="coerce")
    prior_key = out["_dt"] - pd.DateOffset(months=offset_months)
    right = full[["_dt"] + [c for c in value_cols if c in full.columns]].copy()
    right = right.rename(columns={c: f"prior_{c}" for c in value_cols if c in right.columns})
    right = right.rename(columns={"_dt": "_prior_dt"})
    out["_prior_dt"] = prior_key
    out = out.merge(right, on="_prior_dt", how="left")
    for c in ["_dt", "_prior_dt"]:
        if c in out.columns:
            out = out.drop(columns=[c])
    return out


def _render_trend_analysis(monthly: pd.DataFrame, monthly_full: pd.DataFrame | None = None) -> None:
    _section_header(
        "5) Trend Analysis",
        "What this shows: organic growth rate and market impact rate over time, with a volatility band and prior-year comparison. Why it matters: track momentum and compare to last year.",
    )
    if go is None or monthly.empty:
        _institutional_note("Trend Analysis", "Time-series coverage in the selected scope is insufficient for trend diagnostics.")
        return
    scope = _aum_scope_label()
    use_qoq = st.radio(
        "Period view",
        ["Month-over-Month (MoM)", "Quarter-over-Quarter (QoQ)"],
        index=0,
        key="trend_mom_qoq",
        horizontal=True,
    )
    is_qoq = "QoQ" in use_qoq

    if is_qoq:
        df = _resample_to_quarter(monthly)
        full = (
            _resample_to_quarter(monthly_full)
            if monthly_full is not None and not monthly_full.empty
            else pd.DataFrame()
        )
        period_label = "Quarter"
        roll_window = 2
    else:
        df = monthly.sort_values("month_end").copy()
        df["month_end"] = pd.to_datetime(df["month_end"], errors="coerce")
        full = monthly_full.sort_values("month_end").copy() if monthly_full is not None and not monthly_full.empty else pd.DataFrame()
        if not full.empty:
            full["month_end"] = pd.to_datetime(full["month_end"], errors="coerce")
        period_label = "Month"
        roll_window = 6

    if df.empty or "ogr" not in df.columns or "market_impact_rate" not in df.columns:
        _institutional_note("Trend Analysis", "OGR or market impact rate was not computed for the selected period.")
        return

    df = _prior_year_series(df, full if not full.empty else df, "month_end", ["ogr", "market_impact_rate"], offset_months=12)

    ogr = pd.to_numeric(df["ogr"], errors="coerce")
    mir = pd.to_numeric(df["market_impact_rate"], errors="coerce")
    x = df["month_end"]

    roll_mean = ogr.rolling(window=min(roll_window, len(ogr)), min_periods=1).mean()
    roll_std = ogr.rolling(window=min(roll_window, len(ogr)), min_periods=1).std().fillna(0)
    upper = roll_mean + roll_std
    lower = roll_mean - roll_std

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(143, 180, 255, 0.2)",
            line=dict(width=0),
            name="OGR volatility band",
            showlegend=True,
        )
    )
    fig.add_scatter(
        x=x,
        y=ogr,
        mode="lines+markers",
        name="OGR",
        line=dict(color=PALETTE["primary"], width=2),
        customdata=ogr.map(_fmt_pct),
        hovertemplate=f"{period_label}: %{{x|%b %Y}}<br>OGR: %{{customdata}}<extra></extra>",
    )
    fig.add_scatter(
        x=x,
        y=mir,
        mode="lines+markers",
        name="Market Impact Rate",
        line=dict(color=PALETTE["market"], width=2),
        customdata=mir.map(_fmt_pct),
        hovertemplate=f"{period_label}: %{{x|%b %Y}}<br>Market Impact Rate: %{{customdata}}<extra></extra>",
    )
    if "prior_ogr" in df.columns:
        prior_ogr = pd.to_numeric(df["prior_ogr"], errors="coerce")
        fig.add_scatter(
            x=x,
            y=prior_ogr,
            mode="lines+markers",
            name="OGR (same period last year)",
            line=dict(color=PALETTE["primary"], width=1, dash="dash"),
            customdata=prior_ogr.map(_fmt_pct),
            hovertemplate=f"{period_label}: %{{x|%b %Y}}<br>OGR prior year: %{{customdata}}<extra></extra>",
        )
    if "prior_market_impact_rate" in df.columns:
        prior_mir = pd.to_numeric(df["prior_market_impact_rate"], errors="coerce")
        fig.add_scatter(
            x=x,
            y=prior_mir,
            mode="lines+markers",
            name="Market Impact Rate (same period last year)",
            line=dict(color=PALETTE["market"], width=1, dash="dash"),
            customdata=prior_mir.map(_fmt_pct),
            hovertemplate=f"{period_label}: %{{x|%b %Y}}<br>Market Impact Rate prior year: %{{customdata}}<extra></extra>",
        )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis=dict(title=period_label),
        yaxis=dict(title="Rate (%)", tickformat=".1%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    apply_enterprise_plotly_style(fig, height=420)
    safe_render_plotly(fig)
    st.caption(f"Scope: **{scope}**. Band = OGR volatility; dashed lines = same period last year.")


def _render_correlation(monthly: pd.DataFrame, channel_scoped: pd.DataFrame) -> None:
    _section_header(
        "6) Correlation Analysis",
        "What this shows: executive driver contribution plus a correlation heatmap diagnostic for NNB by channel, fee yield by channel, and AUM growth. Why it matters: translates statistical relationships into plain-English strategy implications.",
    )
    if go is None or monthly.empty:
        _institutional_note("Correlation Analysis", "The selected scope does not include enough observations for driver diagnostics.")
        return
    st.markdown("#### Driver Contribution Analysis")
    st.caption("Relative impact of key portfolio drivers on growth and performance.")
    driver_map = {
        "Net New Business": "nnb",
        "Net Fund Flows": "nnf",
        "Organic Growth Rate": "ogr",
        "Market Impact": "market_impact" if "market_impact" in monthly.columns else "market_impact_rate",
        "Fee Yield": "fee_yield",
    }
    driver_desc = {
        "Net New Business": "Represents new capital entering or leaving the portfolio.",
        "Net Fund Flows": "Tracks fee-bearing flow momentum across products.",
        "Organic Growth Rate": "Reflects organic expansion of portfolio value.",
        "Market Impact": "Captures valuation movement from market performance.",
        "Fee Yield": "Measures revenue efficiency of the portfolio mix.",
    }

    scores: list[dict[str, Any]] = []
    for label, col in driver_map.items():
        if col not in monthly.columns:
            continue
        s = pd.to_numeric(monthly[col], errors="coerce").dropna()
        if s.empty:
            continue
        std = float(s.std()) if pd.notna(s.std()) else 0.0
        if std > 0:
            influence = float(((s - float(s.mean())) / std).abs().mean())
        else:
            influence = float(s.abs().mean())
        scores.append({"driver": label, "column": col, "influence": influence})

    if not scores:
        _institutional_note("Correlation Analysis", "At least one driver series with sufficient variation is required to estimate contribution influence.")
        return

    df = pd.DataFrame(scores)
    total = float(df["influence"].sum()) if float(df["influence"].sum()) else 0.0
    if total <= 0:
        _institutional_note("Correlation Analysis", "Driver influence could not be estimated; signal is low or flat in the selected period.")
        return
    df["contribution_pct"] = (df["influence"] / total) * 100.0
    df = df.sort_values("contribution_pct", ascending=False).reset_index(drop=True)

    left, right = st.columns([2.0, 1.0])
    with left:
        bar_colors = [PALETTE["market"] if str(d) == "Market Impact" else PALETTE["primary"] for d in df["driver"]]
        fig = go.Figure(
            go.Bar(
                x=df["contribution_pct"],
                y=df["driver"],
                orientation="h",
                marker=dict(color=bar_colors, line=dict(color=PALETTE["grid"], width=1)),
                text=df["contribution_pct"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                customdata=pd.DataFrame(
                    {
                        "driver": df["driver"],
                        "desc": df["driver"].map(driver_desc),
                        "column": df["column"],
                    }
                ),
                hovertemplate=(
                    "Driver: %{customdata[0]}<br>"
                    "Estimated Contribution: %{x:.1f}%<br>"
                    "%{customdata[1]}<br>"
                    "Source Metric: %{customdata[2]}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=340,
            margin=dict(l=20, r=30, t=15, b=20),
            xaxis_title="Relative Contribution (%)",
            yaxis_title="Driver",
            yaxis=dict(autorange="reversed"),
        )
        apply_enterprise_plotly_style(fig, height=340)
        fig.update_xaxes(range=[0, max(100.0, float(df["contribution_pct"].max()) * 1.25)])
        safe_render_plotly(fig)
        st.caption("Ranking reflects each driver's relative influence over the selected period.")

    with right:
        top = df["driver"].iloc[0] if len(df) > 0 else "-"
        second = df["driver"].iloc[1] if len(df) > 1 else "-"
        third = df["driver"].iloc[2] if len(df) > 2 else "-"
        fourth = df["driver"].iloc[3] if len(df) > 3 else "-"
        st.markdown("#### Key Growth Drivers")
        st.markdown(f"**Top driver**  \n{top}")
        st.markdown(f"**Secondary driver**  \n{second}")
        st.markdown(f"**Supporting factor**  \n{third}")
        st.markdown(f"**Moderate influence**  \n{fourth}")

    st.caption("Bar length = relative influence of each driver in the period.")

    st.markdown("#### Correlation Heatmap")
    st.caption("Diagnostic view of cross-channel relationships.")
    corr_cols = ["nnb_channel", "fee_yield_channel", "aum_growth"]
    corr_ready = pd.DataFrame(columns=corr_cols)
    if channel_scoped is not None and not channel_scoped.empty:
        by_channel = channel_scoped.groupby("channel", as_index=False).agg(
            nnb_channel=("nnb", "sum"),
            nnf_channel=("nnf", "sum"),
            begin_aum=("begin_aum", "sum"),
            end_aum=("end_aum", "sum"),
        )
        by_channel["fee_yield_channel"] = by_channel.apply(
            lambda r: compute_fee_yield_nnf_nnb(r.get("nnf_channel"), r.get("nnb_channel"))
            if pd.notna(r.get("nnb_channel")) and float(r.get("nnb_channel")) > 0
            else compute_fee_yield(r.get("nnf_channel"), r.get("begin_aum"), r.get("end_aum"), nnb=r.get("nnb_channel")),
            axis=1,
        )
        by_channel["aum_growth"] = by_channel.apply(
            lambda r: compute_ogr(r.get("end_aum") - r.get("begin_aum"), r.get("begin_aum")),
            axis=1,
        )
        corr_ready = by_channel[corr_cols].copy()
        corr_ready = corr_ready.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if go is None or corr_ready.shape[0] < 2:
        _institutional_note(
            "Correlation Diagnostic",
            "Insufficient channel variation to estimate NNB, fee-yield, and AUM-growth correlation in this slice.",
        )
        return

    corr = corr_ready.corr()
    labels = {
        "nnb_channel": "NNB by Channel",
        "fee_yield_channel": "Fee Yield by Channel",
        "aum_growth": "AUM Growth",
    }
    corr_plot = corr.rename(index=labels, columns=labels)
    corr_fig = go.Figure(
        data=go.Heatmap(
            z=corr_plot.values,
            x=list(corr_plot.columns),
            y=list(corr_plot.index),
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, PALETTE["negative"]], [0.5, "#f3f4f6"], [1.0, PALETTE["positive"]]],
            colorbar=dict(title="Correlation"),
            text=[[f"{v:.2f}" for v in row] for row in corr_plot.values],
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    corr_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=10))
    apply_enterprise_plotly_style(corr_fig, height=300)
    safe_render_plotly(corr_fig)

    upper_pairs = []
    cols_list = list(corr.columns)
    for i in range(len(cols_list)):
        for j in range(i + 1, len(cols_list)):
            c1 = cols_list[i]
            c2 = cols_list[j]
            v = float(corr.loc[c1, c2])
            if pd.notna(v):
                upper_pairs.append((c1, c2, v))
    if upper_pairs:
        c1, c2, v = max(upper_pairs, key=lambda t: abs(t[2]))
        direction = "positive" if v >= 0 else "negative"
        strength = "strong" if abs(v) >= 0.6 else ("moderate" if abs(v) >= 0.3 else "weak")
        st.markdown(
            (
                f"Across channels, the strongest observed relationship is between **{labels[c1]}** and "
                f"**{labels[c2]}** ({strength} {direction} correlation, **{v:.2f}**). "
                "This is a diagnostic relationship indicator, not a causal estimate."
            )
        )
    else:
        st.markdown(
            "Cross-channel correlations are currently too limited for a reliable interpretation in this slice."
        )


def _render_top_bottom_table(ticker_scoped: pd.DataFrame) -> None:
    _section_header(
        "7) Top and Bottom Contributors Table",
        "What this shows: product-level NNB, NNF, and End AUM (per product) within the selected slice, with filters by channel, country, and thresholds. Why it matters: identify top and bottom contributors and export for action; CSV reflects current view.",
    )
    if ticker_scoped.empty:
        _institutional_note("Contributors Table", "Product-level contribution data is not available for the selected filters.")
        return

    agg_dict: dict[str, Any] = {
        "nnb": ("nnb", "sum"),
        "nnf": ("nnf", "sum"),
        "end_aum": ("end_aum", "sum"),
    }
    if "channel" in ticker_scoped.columns:
        agg_dict["channel"] = ("channel", "first")
    if "sub_channel" in ticker_scoped.columns:
        agg_dict["sub_channel"] = ("sub_channel", "first")
    if "country" in ticker_scoped.columns:
        agg_dict["country"] = ("country", "first")

    contributors = ticker_scoped.groupby("product_ticker", as_index=False).agg(**{k: v for k, v in agg_dict.items()})
    contributors["fee_yield_proxy"] = contributors.apply(
        lambda r: (float(r["nnf"]) / float(r["end_aum"])) if r.get("end_aum") and float(r["end_aum"]) else float("nan"), axis=1
    )

    scope = _aum_scope_label()
    filter_key = "tab1_contributors_filter"

    with st.expander("Filters and sort", expanded=False):
        c1, c2, c3 = st.columns(3)
        channel_options = sorted(contributors["channel"].dropna().astype(str).unique().tolist()) if "channel" in contributors.columns else []
        sub_options = sorted(contributors["sub_channel"].dropna().astype(str).unique().tolist()) if "sub_channel" in contributors.columns else []
        country_options = sorted(contributors["country"].dropna().astype(str).unique().tolist()) if "country" in contributors.columns else []
        ticker_options = sorted(contributors["product_ticker"].dropna().astype(str).unique().tolist())

        with c1:
            sel_channel = st.multiselect("Channel", options=channel_options, default=channel_options, key=f"{filter_key}_channel")
            sel_sub = st.multiselect("Sub-Channel", options=sub_options, default=sub_options, key=f"{filter_key}_sub")
        with c2:
            sel_country = st.multiselect("Country", options=country_options, default=country_options, key=f"{filter_key}_country")
            sel_ticker = st.multiselect("Product Ticker", options=ticker_options, default=ticker_options, key=f"{filter_key}_ticker")
        with c3:
            nnb_min = st.number_input("NNB min ($)", min_value=0, value=0, step=10000, key=f"{filter_key}_nnb_min", help="e.g. 100000 for NNB above $100k")
            nnf_min = st.number_input("NNF min ($)", min_value=0, value=0, step=10000, key=f"{filter_key}_nnf_min")
            aum_min = st.number_input("End AUM min ($, per product)", min_value=0, value=0, step=100000, key=f"{filter_key}_aum_min")

        sort_col = st.selectbox(
            "Sort by",
            options=["NNB", "NNF", "Fee Yield Proxy", "End AUM (per product)", "Product Ticker"] + (["Channel", "Country"] if "channel" in contributors.columns else []),
            index=0,
            key=f"{filter_key}_sort",
        )
        sort_asc = st.radio("Order", options=["Descending", "Ascending"], index=0, key=f"{filter_key}_order", horizontal=True)
        limit_top_bottom = st.checkbox("Show only top and bottom 10 by sort", value=False, key=f"{filter_key}_limit")

    filtered = contributors.copy()
    if "channel" in filtered.columns and sel_channel:
        filtered = filtered[filtered["channel"].astype(str).isin(sel_channel)]
    if "sub_channel" in filtered.columns and sel_sub:
        filtered = filtered[filtered["sub_channel"].astype(str).isin(sel_sub)]
    if "country" in filtered.columns and sel_country:
        filtered = filtered[filtered["country"].astype(str).isin(sel_country)]
    if sel_ticker:
        filtered = filtered[filtered["product_ticker"].astype(str).isin(sel_ticker)]
    if nnb_min is not None and nnb_min > 0:
        filtered = filtered[filtered["nnb"] >= nnb_min]
    if nnf_min is not None and nnf_min > 0:
        filtered = filtered[filtered["nnf"] >= nnf_min]
    if aum_min is not None and aum_min > 0:
        filtered = filtered[filtered["end_aum"] >= aum_min]

    sort_col_internal = {
        "NNB": "nnb",
        "NNF": "nnf",
        "Fee Yield Proxy": "fee_yield_proxy",
        "End AUM (per product)": "end_aum",
        "Product Ticker": "product_ticker",
        "Channel": "channel",
        "Country": "country",
    }.get(sort_col, "nnb")
    ascending = sort_asc == "Ascending"
    filtered = filtered.sort_values(sort_col_internal, ascending=ascending, na_position="last")

    if limit_top_bottom:
        top_n = filtered.head(10)
        bot_n = filtered.tail(10).sort_values(sort_col_internal, ascending=ascending, na_position="last")
        display_df = pd.concat([top_n, bot_n]).drop_duplicates()
    else:
        display_df = filtered.head(500)

    rename_map = {
        "product_ticker": "Product Ticker",
        "nnb": "NNB",
        "nnf": "NNF",
        "end_aum": "End AUM (per product)",
        "fee_yield_proxy": "Fee Yield Proxy",
        "channel": "Channel",
        "sub_channel": "Sub-Channel",
        "country": "Country",
    }
    table_renamed = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
    col_order = [c for c in ["Product Ticker", "Channel", "Sub-Channel", "Country", "NNB", "NNF", "End AUM (per product)", "Fee Yield Proxy"] if c in table_renamed.columns]
    table_renamed = table_renamed[[c for c in col_order if c in table_renamed.columns]] if col_order else table_renamed

    st.dataframe(format_df(table_renamed, infer_common_formats(table_renamed)), use_container_width=True, hide_index=True)
    st.caption(f"Scope: **{scope}**. Table is sortable and filterable; CSV export reflects the current filtered and sorted view.")
    render_export_buttons(table_renamed, None, "tab1_top_bottom_contributors")


def render(state: FilterState, contract: dict[str, Any]) -> None:
    _ = contract
    gateway = DataGateway(ROOT)

    st.markdown("## Institutional Asset Management Dashboard")
    st.markdown(
        "<div class='section-subtitle'>Portfolio growth, distribution quality, ETF focus, and top contributors. Scope by period and dimension using the filters below.</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading dashboard data..."):
        firm_df = _normalize_base_frame(gateway.run_query("firm_monthly", state))
        channel_df = _normalize_base_frame(gateway.run_query("channel_monthly", state))
        ticker_df = _normalize_base_frame(gateway.run_query("ticker_monthly", state))
        geo_df = _normalize_base_frame(gateway.run_query("geo_monthly", state))
        segment_df = _normalize_base_frame(gateway.run_query("segment_monthly", state))
    if firm_df.empty and channel_df.empty and ticker_df.empty:
        st.info("No data is available for the current reporting range.")
        return

    st.markdown("### Scope & filters")
    period = st.radio("Time Period", PERIOD_OPTIONS, horizontal=True, key="tab1_period")
    firm_period = _apply_period(firm_df, period)
    channel_period = _apply_period(channel_df, period)
    ticker_period = _apply_period(ticker_df, period)
    geo_period = _apply_period(geo_df, period)
    segment_period = _apply_period(segment_df, period)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _selectbox_with_all("Channel", "tab1_filter_channel", channel_period["channel"].astype(str).tolist() if not channel_period.empty else [])

    tab1 = _tab1_snapshot()
    sub_source = channel_period
    if tab1.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL) not in (None, "", "All"):
        sub_source = sub_source[sub_source["channel"] == tab1["tab1_filter_channel"]]
    with c2:
        _selectbox_with_all("Sub-Channel", "tab1_filter_sub_channel", sub_source["sub_channel"].astype(str).tolist() if not sub_source.empty else [])

    with c3:
        _selectbox_with_all("Country", "tab1_filter_country", geo_period["country"].astype(str).tolist() if not geo_period.empty else [])
    with c4:
        _selectbox_with_all("Segment", "tab1_filter_segment", segment_period["segment"].astype(str).tolist() if not segment_period.empty else [])

    seg_source = segment_period
    if tab1.get("tab1_filter_segment", TAB1_DEFAULT_SEGMENT) not in (None, "", "All"):
        seg_source = seg_source[seg_source["segment"] == tab1["tab1_filter_segment"]]
    with c5:
        _selectbox_with_all("Sub-Segment", "tab1_filter_sub_segment", seg_source["sub_segment"].astype(str).tolist() if not seg_source.empty else [])
    with c6:
        _selectbox_with_all("Product Ticker", "tab1_filter_ticker", ticker_period["product_ticker"].astype(str).tolist() if not ticker_period.empty else [])

    # Refresh snapshot after all selectors so downstream scope and KPI logic use current selection.
    tab1 = _tab1_snapshot()

    common_filter_map = {
        "channel": "tab1_filter_channel",
        "sub_channel": "tab1_filter_sub_channel",
        "country": "tab1_filter_country",
        "segment": "tab1_filter_segment",
        "sub_segment": "tab1_filter_sub_segment",
        "product_ticker": "tab1_filter_ticker",
    }
    channel_scoped = _apply_scoped_filters(
        channel_period,
        common_filter_map,
    )
    ticker_scoped = _apply_scoped_filters(
        ticker_period,
        common_filter_map,
    )
    geo_scoped = _apply_scoped_filters(
        geo_period,
        common_filter_map,
    )
    segment_scoped = _apply_scoped_filters(
        segment_period,
        common_filter_map,
    )

    source_for_monthly = firm_period
    if tab1.get("tab1_filter_ticker", TAB1_DEFAULT_PRODUCT_TICKER) not in (None, "", "All") and not ticker_scoped.empty:
        source_for_monthly = ticker_scoped
    elif tab1.get("tab1_filter_sub_segment", TAB1_DEFAULT_SUB_SEGMENT) not in (None, "", "All") and not segment_scoped.empty:
        source_for_monthly = segment_scoped
    elif tab1.get("tab1_filter_segment", TAB1_DEFAULT_SEGMENT) not in (None, "", "All") and not segment_scoped.empty:
        source_for_monthly = segment_scoped
    elif tab1.get("tab1_filter_country", TAB1_DEFAULT_COUNTRY) not in (None, "", "All") and not geo_scoped.empty:
        source_for_monthly = geo_scoped
    elif tab1.get("tab1_filter_sub_channel", TAB1_DEFAULT_SUB_CHANNEL) not in (None, "", "All") and not channel_scoped.empty:
        source_for_monthly = channel_scoped
    elif tab1.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL) not in (None, "", "All") and not channel_scoped.empty:
        source_for_monthly = channel_scoped
    monthly = _build_monthly_metrics(source_for_monthly)

    source_for_monthly_full = firm_df
    if tab1.get("tab1_filter_ticker", TAB1_DEFAULT_PRODUCT_TICKER) not in (None, "", "All") and not ticker_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(ticker_df, common_filter_map)
    elif tab1.get("tab1_filter_sub_segment", TAB1_DEFAULT_SUB_SEGMENT) not in (None, "", "All") and not segment_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(segment_df, common_filter_map)
    elif tab1.get("tab1_filter_segment", TAB1_DEFAULT_SEGMENT) not in (None, "", "All") and not segment_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(segment_df, common_filter_map)
    elif tab1.get("tab1_filter_country", TAB1_DEFAULT_COUNTRY) not in (None, "", "All") and not geo_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(geo_df, common_filter_map)
    elif tab1.get("tab1_filter_sub_channel", TAB1_DEFAULT_SUB_CHANNEL) not in (None, "", "All") and not channel_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(channel_df, common_filter_map)
    elif tab1.get("tab1_filter_channel", TAB1_DEFAULT_CHANNEL) not in (None, "", "All") and not channel_scoped.empty:
        source_for_monthly_full = _apply_scoped_filters(channel_df, common_filter_map)
    monthly_full = _build_monthly_metrics(source_for_monthly_full) if not source_for_monthly_full.empty else monthly

    period = tab1.get("tab1_period", TAB1_DEFAULT_PERIOD)
    scope_label = get_scope_label_from_state(tab1)
    scope_mode = get_scope_mode_from_state(tab1)
    kpi_result = compute_kpi(monthly, period, scope_label, scope_mode)
    parity = validate_kpi_against_latest_row(monthly, kpi_result)

    if not kpi_result.prior_period_used:
        st.warning("Prior-period AUM is unavailable for this slice; OGR and Market Movement may be unavailable.")
    current_scope_mode = get_scope_mode_from_state(_tab1_snapshot())
    if kpi_result.scope_mode != current_scope_mode:
        st.warning(
            "Scope label/state mismatch detected for KPI calculation. The dashboard recomputed from current selection, but review scope controls."
        )
    if parity.get("warnings"):
        for msg in parity["warnings"]:
            st.warning(msg)
    # Unit consistency: compact professional QA note when NNF/NNB magnitude ratio is suspicious
    unit_status = kpi_result.validation.get("unit_consistency", "ok")
    if unit_status == "possible_mismatch":
        st.caption(
            "**Metric note:** NNF is reported in lower magnitude than NNB in this period. "
            "If both are intended to be in the same currency unit, verify source scaling for NNF; "
            "otherwise fee revenue (NNF) can correctly be materially smaller than flow (NNB)."
        )
    if st.session_state.get("observability_dev_toggle") or st.session_state.get("dev_mode"):
        with st.expander("KPI parity check", expanded=False):
            displayed = {
                "End AUM": _fmt_currency(kpi_result.end_aum),
                "NNB": _fmt_currency(kpi_result.nnb),
                "NNF": _fmt_currency(kpi_result.nnf),
                "OGR": _fmt_pct(kpi_result.ogr),
                "Market Movement": _fmt_currency(kpi_result.market_movement),
            }
            st.caption("Displayed on cards (same as above):")
            st.json(displayed)
            st.caption("Validation (raw source totals, unit check, reconciliation):")
            st.json(
                {
                    "scope_mode": kpi_result.scope_mode,
                    "scope_label": kpi_result.scope_label,
                    "period": kpi_result.period,
                    "kpi_validation": kpi_result.validation,
                    "latest_row_parity": parity,
                }
            )
            try:
                from app.metrics.qa_guardrails import run_metric_qa
                qa_report = run_metric_qa(kpi_result)
                st.caption("Metric QA guardrails (dev):")
                st.json({"ok": qa_report.get("ok"), "messages": qa_report.get("messages", []), "checks": {k: {"ok": v.get("ok"), "message": v.get("message")} for k, v in (qa_report.get("checks") or {}).items()}})
            except Exception:
                pass

    with st.spinner("Preparing portfolio snapshot..."):
        _render_narrative_and_drivers(monthly, monthly_full, channel_scoped, ticker_scoped, firm_period, kpi_result=kpi_result)

    st.divider()
    with st.spinner("Rendering analytics sections..."):
        _render_aum_waterfall(monthly, kpi_result)
        _render_channel_breakdown(channel_scoped)
        _render_growth_quality_matrix(ticker_scoped, monthly)
        _render_etf_drilldown(ticker_scoped)
        _render_trend_analysis(monthly, monthly_full)
        _render_correlation(monthly, channel_scoped)
        _render_top_bottom_table(ticker_scoped)
