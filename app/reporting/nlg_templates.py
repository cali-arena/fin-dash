"""
Deterministic NLG for Tab 2 Dynamic Report. No LLM.
Conditional template library: OGR, market impact, fee yield, channel/product share,
anomalies, pricing/mix-shift, ETF concentration. All text from Python if/else and metrics only.
Financial wording only; no technical jargon.
"""
from __future__ import annotations

import math
from typing import Any

# --- Thresholds (align with report_engine) -------------------------------------
OGR_STRONG = 0.02
OGR_WEAK_NEG = -0.01
MKT_TAILWIND = 0.02
MKT_HEADWIND = 0.0
FEE_YIELD_IMPROVING_THRESHOLD = 0.0   # delta > 0 = improving
CONCENTRATION_TOP_SHARE = 0.5          # top entity share >= 50% = concentrated
MIX_SHIFT_REC_THRESHOLD = 0.01         # recommend on share delta >= 1%


def _num(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _is_na(x: float) -> bool:
    return x != x or math.isnan(x)


def _has_columns(obj: Any) -> bool:
    """True when object exposes a non-empty columns Index/list."""
    cols = getattr(obj, "columns", None)
    return cols is not None and len(cols) > 0


def _is_empty_like(obj: Any) -> bool:
    """Safe emptiness check for DataFrame-like objects; never uses pandas objects directly in boolean context."""
    if obj is None:
        return True
    empty_attr = getattr(obj, "empty", None)
    if empty_attr is not None:
        try:
            return bool(empty_attr)
        except Exception:
            return False
    try:
        return len(obj) == 0
    except Exception:
        return True


# --- Executive Overview: Are we growing? Is growth good? Where from? What changed? ---

EXEC_PARAGRAPH_OPENING = (
    "This period summarizes portfolio growth, revenue quality, and driver mix for the selected coverage. "
    "The commentary addresses growth direction, sustainability of economics, source of performance, and change versus the prior period."
)

EXEC_GROWTH_TEMPLATES = {
    "growth_strong_flows_tailwind": "Portfolio growth is strong, supported by both client flows and favourable market movement.",
    "growth_strong_flows_headwind": "Growth remains solid despite market headwinds, with client flows carrying the period.",
    "growth_weak_flows_tailwind": "Performance is primarily market-led while organic flows remain limited; durability depends on flow improvement.",
    "growth_weak_flows_headwind": "Growth conditions are weak, with soft flows and adverse market contribution weighing on results.",
    "growth_modest": "Growth is present but moderate, with neither flows nor market contribution showing decisive momentum.",
    "growth_unknown": "Growth direction cannot be confirmed from the selected data window.",
}

EXEC_QUALITY_TEMPLATES = {
    "quality_high_nnb_low_fee_yield": "Growth quality is constructive: strong flows are paired with disciplined fee-yield outcomes.",
    "quality_high_nnb_high_fee_yield": "Growth is profitable; fee yield is elevated—review pricing and mix to ensure sustainability.",
    "quality_low_nnb": "Organic momentum is limited; improving retention and net sales should be a near-term priority.",
    "quality_fee_improving": "Fee yield has improved versus the prior period, supporting better revenue quality.",
    "quality_fee_deteriorating": "Fee yield has softened versus the prior period; pricing and mix should be reassessed.",
    "quality_fee_stable": "Fee yield is broadly stable versus the prior period.",
    "quality_unknown": "Revenue quality cannot be reliably assessed from current inputs.",
}

EXEC_SOURCE_TEMPLATES = {
    "source_nnb_dominant": "Client flow was the principal driver of AUM change in the period.",
    "source_market_dominant": "Market movement was the principal driver of AUM change in the period.",
    "source_mixed": "AUM progression reflects a balanced contribution from flows and market movement.",
    "source_unknown": "The source mix of AUM change is not fully observable in this cut of data.",
}

EXEC_WHAT_CHANGED_TEMPLATES = {
    "changed_strong_flow_mkt_down": "Versus the prior period, flow momentum strengthened while market contribution turned negative, shifting the mix toward client activity.",
    "changed_weak_flow_mkt_up": "Versus the prior period, market contribution improved while flows remained soft, indicating a more market-led profile.",
    "changed_both_positive": "Versus the prior period, both flows and market contribution improved in tandem.",
    "changed_both_negative": "Versus the prior period, both flows and market contribution weakened, creating a more challenging backdrop.",
    "changed_modest": "Versus the prior period, the driver mix remains broadly stable with moderate movement in both factors.",
    "changed_unknown": "Period-over-period comparison is not available for the selected window.",
}


def select_executive_overview(snap: dict[str, Any], fmt_money: Any, fmt_pct: Any) -> list[str]:
    """
    Deterministic Executive Overview: opening paragraph plus bullets.
    Answers: Are we growing? Is growth good (profitable/sustainable)? Where is it coming from? What has changed?
    """
    bullets: list[str] = []
    ogr = _num(snap.get("ogr"))
    mkt_rate = _num(snap.get("market_impact_rate"))
    mkt_abs = _num(snap.get("market_impact_abs"))
    nnb = _num(snap.get("nnb"))
    end_aum = _num(snap.get("end_aum"))
    mom_pct = _num(snap.get("mom_pct"))
    ytd_pct = _num(snap.get("ytd_pct"))
    fee_yield = _num(snap.get("fee_yield"))
    fee_yield_prior = _num(snap.get("fee_yield_prior"))
    month_end = snap.get("month_end", "-")

    # Opening narrative paragraph
    bullets.append(EXEC_PARAGRAPH_OPENING)

    # Headline: AUM, MoM, YTD, NNB, OGR
    bullets.append(
        f"As of {month_end}, month-end AUM is {fmt_money(end_aum)}; "
        f"MoM change is {fmt_pct(mom_pct)} and YTD change is {fmt_pct(ytd_pct)}. "
        f"Net new business is {fmt_money(nnb)} with organic growth at {fmt_pct(ogr)}."
    )

    # Are we growing? (directional)
    if not _is_na(ogr) and not _is_na(mkt_rate):
        if ogr > OGR_STRONG and mkt_rate >= MKT_TAILWIND:
            bullets.append(EXEC_GROWTH_TEMPLATES["growth_strong_flows_tailwind"])
        elif ogr > OGR_STRONG and mkt_rate < MKT_HEADWIND:
            bullets.append(EXEC_GROWTH_TEMPLATES["growth_strong_flows_headwind"])
        elif ogr <= OGR_WEAK_NEG and mkt_rate >= MKT_TAILWIND:
            bullets.append(EXEC_GROWTH_TEMPLATES["growth_weak_flows_tailwind"])
        elif ogr <= OGR_WEAK_NEG and mkt_rate < MKT_HEADWIND:
            bullets.append(EXEC_GROWTH_TEMPLATES["growth_weak_flows_headwind"])
        else:
            bullets.append(EXEC_GROWTH_TEMPLATES["growth_modest"])
    else:
        bullets.append(EXEC_GROWTH_TEMPLATES["growth_unknown"])

    # What has changed versus last period
    if not _is_na(ogr) and not _is_na(mkt_rate):
        if ogr > OGR_STRONG and mkt_rate < MKT_HEADWIND:
            bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_strong_flow_mkt_down"])
        elif ogr <= OGR_WEAK_NEG and mkt_rate > MKT_TAILWIND:
            bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_weak_flow_mkt_up"])
        elif ogr > OGR_STRONG and mkt_rate >= MKT_TAILWIND:
            bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_both_positive"])
        elif ogr <= OGR_WEAK_NEG and mkt_rate < MKT_HEADWIND:
            bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_both_negative"])
        else:
            bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_modest"])
    else:
        bullets.append(EXEC_WHAT_CHANGED_TEMPLATES["changed_unknown"])

    # Flows vs market (driver detail)
    if not _is_na(ogr) and not _is_na(mkt_rate):
        bullets.append(f"Organic growth rate {fmt_pct(ogr)}; market impact rate {fmt_pct(mkt_rate)}.")
    if not _is_na(mkt_abs):
        bullets.append(f"Market contribution (absolute): {fmt_money(mkt_abs)}.")

    # Is growth good? (fee yield, quality) — support fee yield delta when available
    if not _is_na(fee_yield):
        bullets.append(f"Fee yield: {fmt_pct(fee_yield)}.")
        if not _is_na(fee_yield_prior) and fee_yield_prior != 0:
            delta = fee_yield - fee_yield_prior
            if delta > FEE_YIELD_IMPROVING_THRESHOLD:
                bullets.append(EXEC_QUALITY_TEMPLATES["quality_fee_improving"])
            elif delta < -0.0001:
                bullets.append(EXEC_QUALITY_TEMPLATES["quality_fee_deteriorating"])
            else:
                bullets.append(EXEC_QUALITY_TEMPLATES["quality_fee_stable"])
        elif ogr > OGR_STRONG and fee_yield < 0.02:
            bullets.append(EXEC_QUALITY_TEMPLATES["quality_high_nnb_low_fee_yield"])
        elif ogr > OGR_STRONG:
            bullets.append(EXEC_QUALITY_TEMPLATES["quality_high_nnb_high_fee_yield"])
        elif ogr <= OGR_WEAK_NEG:
            bullets.append(EXEC_QUALITY_TEMPLATES["quality_low_nnb"])
        else:
            bullets.append(EXEC_QUALITY_TEMPLATES["quality_fee_stable"])
    else:
        bullets.append(EXEC_QUALITY_TEMPLATES["quality_unknown"])

    # Where is it coming from?
    if not _is_na(ogr) and not _is_na(mkt_abs):
        if abs(ogr) > 1e-9 and (abs(mkt_abs) < 1e-9 or abs(nnb) >= abs(mkt_abs)):
            bullets.append(EXEC_SOURCE_TEMPLATES["source_nnb_dominant"])
        elif abs(mkt_abs) > 1e-9 and (abs(nnb) < 1e-9 or abs(mkt_abs) >= abs(nnb)):
            bullets.append(EXEC_SOURCE_TEMPLATES["source_market_dominant"])
        else:
            bullets.append(EXEC_SOURCE_TEMPLATES["source_mixed"])
    else:
        bullets.append(EXEC_SOURCE_TEMPLATES["source_unknown"])

    return bullets[:8]


# --- Channel: share gain/loss, concentration ------------------------------------

CHANNEL_TEMPLATES = {
    "top_bottom": "Top channel by NNB: {top_name} ({top_value}). Bottom channel by NNB: {bottom_name} ({bottom_value}).",
    "share_gain": "Distribution mix rotated toward {name}, gaining {delta} of AUM share versus the prior period.",
    "share_loss": "Distribution mix rotated away from {name}, which lost {delta} of AUM share versus the prior period.",
    "concentration_high": "Channel concentration is elevated; the largest distribution channel accounts for {share} of total AUM.",
    "concentration_modest": "Channel mix remains reasonably diversified across distribution groups.",
    "no_data": "No channel rank data available for the selected slice.",
}


def _share_col(cols: Any) -> str | None:
    """First available share column: gateway uses aum_share / nnb_share; fallback share."""
    if cols is None:
        return None
    for c in ("share", "aum_share", "nnb_share"):
        if c in cols:
            return c
    return None


def select_channel_commentary(bullets_from_rules: list[str], rank: Any, snap: dict[str, Any], fmt_money: Any, fmt_pct: Any) -> list[str]:
    """Add deterministic channel share gain/loss and concentration from rank table. Financial wording only."""
    out = list(bullets_from_rules)
    if rank is None or getattr(rank, "empty", True):
        return out
    cols = getattr(rank, "columns", None)
    dim_col = "dim_value" if (cols is not None and "dim_value" in cols) else "name"
    for share_delta_col in ("aum_share_delta", "nnb_share_delta", "share_delta"):
        if cols is not None and share_delta_col in cols:
            try:
                idx = rank[share_delta_col].abs().idxmax()
                row = rank.loc[idx]
            except Exception:
                continue
            delta = _num(row.get(share_delta_col))
            name = row.get(dim_col, "-")
            if delta > 0.01:
                out.append(CHANNEL_TEMPLATES["share_gain"].format(name=name, delta=fmt_pct(delta)))
            elif delta < -0.01:
                out.append(CHANNEL_TEMPLATES["share_loss"].format(name=name, delta=fmt_pct(abs(delta))))
            break
    share_col = _share_col(cols)
    if share_col:
        try:
            top_share = rank[share_col].max()
        except Exception:
            top_share = float("nan")
        if not _is_na(top_share) and top_share >= CONCENTRATION_TOP_SHARE:
            out.append(CHANNEL_TEMPLATES["concentration_high"].format(share=fmt_pct(top_share)))
        elif not _is_na(top_share):
            out.append(CHANNEL_TEMPLATES["concentration_modest"])
    return out[:6]


# --- Product and ETF: concentration, mix shift ---------------------------------

PRODUCT_TEMPLATES = {
    "concentration_high": "Product concentration is elevated: the leading product or strategy represents {share} of net new business.",
    "concentration_modest": "Product inflow is distributed across several products, with concentration risk in a moderate range.",
    "top_etf": "ETF flows were led by {name} with {value} in net new business.",
    "mix_shift_product": "Product mix shifted, with {name} gaining {delta} share of flows versus the prior period.",
    "no_data": "No product rank data available for the selected slice.",
}


def select_product_commentary(bullets_from_rules: list[str], rank: Any, fmt_money: Any, fmt_pct: Any) -> list[str]:
    """Add product concentration (AUM or NNB share) and optional mix-shift. Financial wording only."""
    out = list(bullets_from_rules)
    if rank is None or getattr(rank, "empty", True):
        return out
    cols = getattr(rank, "columns", None)
    share_col = _share_col(cols)
    if share_col:
        try:
            top_share = rank[share_col].max()
        except Exception:
            top_share = float("nan")
        if not _is_na(top_share) and top_share >= CONCENTRATION_TOP_SHARE:
            out.append(PRODUCT_TEMPLATES["concentration_high"].format(share=fmt_pct(top_share)))
        elif not _is_na(top_share):
            out.append(PRODUCT_TEMPLATES["concentration_modest"])
    dim_col = "dim_value" if (cols is not None and "dim_value" in cols) else "name"
    for share_delta_col in ("nnb_share_delta", "aum_share_delta", "share_delta"):
        if cols is not None and share_delta_col in cols:
            try:
                idx = rank[share_delta_col].abs().idxmax()
                row = rank.loc[idx]
            except Exception:
                continue
            delta = _num(row.get(share_delta_col))
            if abs(delta) >= 0.01:
                name = row.get(dim_col, "-")
                out.append(PRODUCT_TEMPLATES["mix_shift_product"].format(name=name, delta=fmt_pct(abs(delta))))
            break
    return out[:6]


# --- Geographic analysis --------------------------------------------------------

GEO_TEMPLATES = {
    "share_gain": "Geographic mix shifted toward {name}, gaining {delta} of AUM share versus the prior period.",
    "share_loss": "Geographic mix shifted away from {name}, losing {delta} of AUM share versus the prior period.",
    "concentration_high": "Geographic concentration is elevated; the largest region represents {share} of total AUM.",
    "no_data": "Geographic coverage is insufficient for comparative analysis in the selected slice.",
}


def select_geo_commentary(bullets_from_rules: list[str], rank: Any, fmt_pct: Any) -> list[str]:
    """Add geographic share gain/loss and optional concentration. Financial wording only."""
    out = list(bullets_from_rules)
    if rank is None or getattr(rank, "empty", True):
        return out
    cols = getattr(rank, "columns", None)
    dim_col = "dim_value" if (cols is not None and "dim_value" in cols) else "name"
    for share_delta_col in ("aum_share_delta", "nnb_share_delta", "share_delta"):
        if cols is not None and share_delta_col in cols:
            try:
                idx = rank[share_delta_col].abs().idxmax()
                row = rank.loc[idx]
            except Exception:
                continue
            delta = _num(row.get(share_delta_col))
            name = row.get(dim_col, "-")
            if delta > 0.01:
                out.append(GEO_TEMPLATES["share_gain"].format(name=name, delta=fmt_pct(delta)))
            elif delta < -0.01:
                out.append(GEO_TEMPLATES["share_loss"].format(name=name, delta=fmt_pct(abs(delta))))
            break
    share_col = _share_col(cols)
    if share_col:
        try:
            top_share = rank[share_col].max()
        except Exception:
            top_share = float("nan")
        if not _is_na(top_share) and top_share >= CONCENTRATION_TOP_SHARE:
            out.append(GEO_TEMPLATES["concentration_high"].format(share=fmt_pct(top_share)))
    return out[:6]


# --- Anomalies and Flags -------------------------------------------------------

ANOMALY_TEMPLATES = {
    "none": "No statistically significant anomalies were flagged under current thresholds; no immediate intervention is indicated.",
    "count": "{n} items were flagged for review under the configured statistical thresholds.",
    "high_severity": "High-priority flag: {metric} at {entity} diverged materially from baseline (current value {value}); immediate driver review is advised.",
    "medium_severity": "Review recommended: {metric} at {entity} showed a meaningful deviation from baseline.",
    "reversal": "Flow reversal at {entity}: net new business changed sign versus the prior period; confirm underlying client or product drivers.",
}


def select_anomaly_bullets(anom: Any, fmt_num: Any) -> list[str]:
    """Deterministic anomaly bullets from anomaly table only. Financial wording; no pandas truth-value on DataFrame."""
    if _is_empty_like(anom) or not _has_columns(anom):
        return [ANOMALY_TEMPLATES["none"]]
    cols = getattr(anom, "columns", None)
    if cols is None:
        return [ANOMALY_TEMPLATES["none"]]
    col_set = set(cols)
    if not {"metric", "entity"}.issubset(col_set):
        return [ANOMALY_TEMPLATES["none"]]
    n_rows = len(anom)
    bullets = [ANOMALY_TEMPLATES["count"].format(n=n_rows)]
    head = anom.head(5)
    for i in range(len(head)):
        r = head.iloc[i]
        metric = r.get("metric", "-")
        entity = r.get("entity", "-")
        zscore = r.get("zscore")
        val = r.get("value_current")
        sev = (str(r.get("severity") or "")).strip().lower()
        rule_id = str(r.get("rule_id") or "")
        val_str = fmt_num(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else "-"
        if rule_id == "reversal":
            bullets.append(ANOMALY_TEMPLATES["reversal"].format(entity=entity))
        elif sev == "high":
            bullets.append(ANOMALY_TEMPLATES["high_severity"].format(metric=metric, entity=entity, zscore=fmt_num(zscore), value=val_str))
        else:
            bullets.append(ANOMALY_TEMPLATES["medium_severity"].format(metric=metric, entity=entity, zscore=fmt_num(zscore)))
    return bullets[:6]


# --- Recommendations: what should we do next? ---------------------------------

REC_TEMPLATES = {
    "strong_flows_negative_market": "Review hedging posture and risk messaging: strong flows are offset by negative market contribution.",
    "allocate_sales_channel": "Reallocate distribution focus toward {name}, where channel mix shift is meaningful versus the prior period.",
    "allocate_sales_product": "Reallocate product or sales focus toward {name}, where product mix shift is meaningful versus the prior period.",
    "allocate_sales_geo": "Reallocate regional focus toward {name}, where geographic mix shift is meaningful versus the prior period.",
    "investigate_ticker": "Review drivers for {entity}: distinguish flow effect from pricing or mix.",
    "investigate_high_severity": "Review {entity} in light of the high-priority flag and validate sustainability of the current trend.",
    "investigate_reversal": "Review {entity} following the flow reversal and confirm the underlying demand shift.",
    "weak_flows_retention": "Organic growth is limited; prioritize flow generation and client retention initiatives.",
    "fee_deteriorating": "Fee yield has deteriorated; review pricing, mix, and cost levers to protect revenue quality.",
    "concentration_diversify": "Concentration is elevated; diversify across channels or products to reduce single-point dependency.",
    "monitor_modest": "AUM growth is moderate; monitor breadth across channels and products for early signs of acceleration or stress.",
    "no_actions": "No specific actions are triggered under current anomaly or mix-shift thresholds; continue standard monitoring.",
}


def select_recommendations(
    pack: Any,
    snap: dict[str, Any],
    fmt_pct: Any,
    ogr_strong: float = OGR_STRONG,
    mkt_negative: bool = False,
) -> list[str]:
    """Deterministic recommendations tied to metrics: anomalies, mix shift, OGR, market, fee yield, concentration."""
    bullets: list[str] = []
    ogr = _num(snap.get("ogr"))
    fee_yield = _num(snap.get("fee_yield"))
    fee_yield_prior = _num(snap.get("fee_yield_prior"))

    if ogr > ogr_strong and mkt_negative:
        bullets.append(REC_TEMPLATES["strong_flows_negative_market"])
    if ogr <= OGR_WEAK_NEG:
        bullets.append(REC_TEMPLATES["weak_flows_retention"])
    if not _is_na(fee_yield_prior) and not _is_na(fee_yield) and fee_yield < fee_yield_prior - 0.0001:
        bullets.append(REC_TEMPLATES["fee_deteriorating"])

    for rank_df, key in [
        (getattr(pack, "channel_rank", None), "allocate_sales_channel"),
        (getattr(pack, "ticker_rank", None), "allocate_sales_product"),
        (getattr(pack, "geo_rank", None), "allocate_sales_geo"),
    ]:
        if rank_df is None or getattr(rank_df, "empty", True):
            continue
        cols = getattr(rank_df, "columns", None)
        if cols is None:
            continue
        dim_col = "dim_value" if "dim_value" in cols else "name"
        for share_delta_col in ("aum_share_delta", "nnb_share_delta", "share_delta"):
            if share_delta_col not in cols:
                continue
            try:
                idx = rank_df[share_delta_col].abs().idxmax()
                row = rank_df.loc[idx]
            except Exception:
                continue
            if _num(row.get(share_delta_col)) >= MIX_SHIFT_REC_THRESHOLD:
                bullets.append(REC_TEMPLATES[key].format(name=row.get(dim_col, "-")))
            break

    anom = getattr(pack, "anomalies", None)
    if not _is_empty_like(anom) and _has_columns(anom):
        try:
            cols = set(getattr(anom, "columns", []))
            if {"severity", "entity"}.issubset(cols):
                high = anom[anom["severity"] == "high"]
                n_high = len(high)
                if n_high > 0:
                    if "level" in cols:
                        ticker_high = high[high["level"] == "ticker"]
                        if not _is_empty_like(ticker_high):
                            bullets.append(REC_TEMPLATES["investigate_ticker"].format(entity=ticker_high.iloc[0].get("entity", "-")))
                        else:
                            bullets.append(REC_TEMPLATES["investigate_high_severity"].format(entity=high.iloc[0].get("entity", "-")))
                    else:
                        bullets.append(REC_TEMPLATES["investigate_high_severity"].format(entity=high.iloc[0].get("entity", "-")))
            if "rule_id" in cols:
                rev = anom[anom["rule_id"].fillna("").astype(str).str.strip() == "reversal"]
                if len(rev) > 0:
                    bullets.append(REC_TEMPLATES["investigate_reversal"].format(entity=rev.iloc[0].get("entity", "-")))
        except Exception:
            pass

    for rank_df in (getattr(pack, "channel_rank", None), getattr(pack, "ticker_rank", None)):
        if rank_df is None or getattr(rank_df, "empty", True):
            continue
        cols = getattr(rank_df, "columns", None)
        share_col = _share_col(cols)
        if not share_col:
            continue
        try:
            top_share = rank_df[share_col].max()
        except Exception:
            continue
        if not _is_na(top_share) and top_share >= CONCENTRATION_TOP_SHARE:
            bullets.append(REC_TEMPLATES["concentration_diversify"])
            break

    if not bullets:
        bullets.append(REC_TEMPLATES["no_actions"])
        bullets.append(REC_TEMPLATES["monitor_modest"])
    elif len(bullets) == 1 and "no_actions" in bullets[0]:
        bullets.append(REC_TEMPLATES["monitor_modest"])
    return bullets[:6]
