"""
Deterministic report engine: section renderers produce SectionOutput from ReportPack.
No LLM calls; bullets from templates + rule triggers only; all numbers from pack.
Uses app.ui.formatters for currency, percent, and generic number display.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.reporting.report_pack import ReportPack
from app.reporting import rules
from app.ui.formatters import fmt_currency, fmt_number, fmt_percent

# --- Constants (template / rule thresholds) ------------------------------------

OGR_STRONG = 0.02           # 2% organic growth = "strong flows"
MKT_TAILWIND = 0.02         # 2% market impact rate = "tailwind"
MKT_HEADWIND_THRESHOLD = 0  # market_impact_rate < 0 = headwind
TOP_N = rules.TOP_N
BOTTOM_N = rules.BOTTOM_N
MIX_SHIFT_THRESHOLD = rules.MIX_SHIFT_THRESHOLD
MIN_BULLETS = rules.MIN_BULLETS
MAX_BULLETS = rules.MAX_BULLETS


def _fmt_money(x: float | None) -> str:
    """Currency for report bullets (full integer with comma)."""
    return fmt_currency(x, unit=" ", decimals=0)


def _fmt_pct(x: float | None) -> str:
    """Percent for report bullets (fraction in -> percent out)."""
    return fmt_percent(x, decimals=2, signed=False)


def _fmt_num(x: float | None) -> str:
    """Generic number for report bullets (e.g. zscore, value)."""
    return fmt_number(x, decimals=2)


def _safe_first_row(df: pd.DataFrame) -> dict[str, Any]:
    """Return first row as dict; {} if empty."""
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _rank_dim_col(rank: pd.DataFrame) -> str:
    """Entity column: gateway uses dim_value, fallback uses name."""
    if rank is None or rank.empty:
        return "name"
    return "dim_value" if "dim_value" in rank.columns else "name"


def _rank_value_col(rank: pd.DataFrame) -> str:
    """AUM/value column: gateway uses aum_end, fallback uses value."""
    if rank is None or rank.empty:
        return "value"
    return "aum_end" if "aum_end" in rank.columns else "value"


def _rank_nnb_col(rank: pd.DataFrame) -> str | None:
    """NNB column if present."""
    if rank is None or rank.empty:
        return None
    return "nnb" if "nnb" in rank.columns else None


def _rank_share_delta_cols(rank: pd.DataFrame) -> list[str]:
    """Share-delta columns for mix shift: gateway has aum_share_delta, nnb_share_delta; fallback has share_delta."""
    if rank is None or rank.empty:
        return ["share_delta"]
    out = []
    for c in ("aum_share_delta", "nnb_share_delta", "share_delta"):
        if c in rank.columns:
            out.append(c)
    return out if out else ["share_delta"]


def _ensure_bullet_count(bullets: list[str], snap: dict[str, Any], fallback_templates: list[str]) -> list[str]:
    """Cap at MAX_BULLETS; if fewer than MIN_BULLETS, append fallback bullets with numbers from snap."""
    bullets = list(bullets)[:MAX_BULLETS]
    snap = snap or {}
    kwargs = {"end_aum": _fmt_money(snap.get("end_aum")), "month_end": snap.get("month_end", "—")}
    for t in fallback_templates:
        if len(bullets) >= MIN_BULLETS:
            break
        try:
            bullets.append(t.format(**kwargs))
        except KeyError:
            bullets.append(t)
    return bullets[:MAX_BULLETS]


# --- Section contract ---------------------------------------------------------

@dataclass(frozen=True)
class SectionOutput:
    bullets: list[str]       # 2–5 bullets (unless no data)
    table_title: str
    table: pd.DataFrame       # small supporting table (top movers)
    meta: dict[str, Any]      # triggered rules, thresholds used


# --- Overview -----------------------------------------------------------------

def render_overview(pack: ReportPack) -> SectionOutput:
    """Overview: firm KPI headline, flows vs markets, fee yield if present. Table: Top Movers (Channels) by NNB."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "firm_snapshot, time_series", "thresholds_used": ["OGR_STRONG", "MKT_TAILWIND"]}
    table = pd.DataFrame()
    table_title = "Top Movers (Channels)"
    snap = pack.firm_snapshot
    ts = pack.time_series
    row = _safe_first_row(snap)
    if row:
        end_aum = row.get("end_aum")
        nnb = row.get("nnb")
        mom_pct = row.get("mom_pct")
        ytd_pct = row.get("ytd_pct")
        ogr = row.get("ogr")
        month_end = row.get("month_end")
        market_rate = row.get("market_impact_rate") if "market_impact_rate" in row else None
        try:
            ogr_f = float(ogr) if ogr is not None and pd.notna(ogr) else float("nan")
            mkt_f = float(market_rate) if market_rate is not None and pd.notna(market_rate) else float("nan")
        except (TypeError, ValueError):
            ogr_f, mkt_f = float("nan"), float("nan")

        bullets.append(f"Month-end AUM: {_fmt_money(end_aum)} (as of {month_end}); MoM {_fmt_pct(mom_pct)}, YTD {_fmt_pct(ytd_pct)}. NNB {_fmt_money(nnb)}, OGR {_fmt_pct(ogr)}.")

        if ogr_f > OGR_STRONG and mkt_f < MKT_HEADWIND_THRESHOLD:
            bullets.append("Flows strong, markets headwind; growth driven by net new business.")
        elif ogr_f < 0 and mkt_f > MKT_TAILWIND:
            bullets.append("Flows weak, markets tailwind; AUM change largely from market move.")
        else:
            bullets.append(f"Flows {_fmt_pct(ogr)}; market impact rate {_fmt_pct(market_rate)}.")

        if "fee_yield" in row and row.get("fee_yield") is not None and pd.notna(row.get("fee_yield")):
            bullets.append(f"Fee yield: {_fmt_pct(row.get('fee_yield'))}.")

        table = pd.DataFrame()
        ch = pack.channel_rank
        if ch is not None and not ch.empty:
            dim_col = _rank_dim_col(ch)
            nnb_col = _rank_nnb_col(ch)
            metric_col = nnb_col if nnb_col else _rank_value_col(ch)
            if metric_col in ch.columns and dim_col in ch.columns:
                try:
                    table = rules.select_top_bottom(ch, metric_col, dim_col, top_n=TOP_N, bottom_n=BOTTOM_N, caller="render_overview")
                except ValueError:
                    table = ch.head(TOP_N + BOTTOM_N) if len(ch) else pd.DataFrame()
        if table.empty and ch is not None and not ch.empty:
            table = ch.head(TOP_N + BOTTOM_N)
        bullets = _ensure_bullet_count(bullets, row, ["Snapshot month: {month_end}.", "Firm AUM: {end_aum}."])
    else:
        bullets.append("No firm snapshot data available.")
        bullets.append("Firm AUM: —. Run report with valid filters.")
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)


# --- Channel commentary -------------------------------------------------------

def render_channel_commentary(pack: ReportPack) -> SectionOutput:
    """Top/bottom channel by NNB with values; mix shift callout. Table: Top/Bottom Channels."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "channel_rank", "thresholds_used": [f"MIX_SHIFT_THRESHOLD={MIX_SHIFT_THRESHOLD}"]}
    table_title = "Top/Bottom Channels"
    rank = pack.channel_rank
    snap = _safe_first_row(pack.firm_snapshot)

    if rank is None or rank.empty:
        bullets.append("No channel rank data available.")
        bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot month: {month_end}."])
        return SectionOutput(bullets=bullets, table_title=table_title, table=pd.DataFrame(), meta=meta)

    dim_col = _rank_dim_col(rank)
    value_col = _rank_value_col(rank)
    nnb_col = _rank_nnb_col(rank)
    metric_col = nnb_col if nnb_col else value_col
    bucket_col = "bucket" if "bucket" in rank.columns else "segment"

    try:
        top_bottom = rules.select_top_bottom(rank, metric_col, dim_col, top_n=TOP_N, bottom_n=BOTTOM_N, caller="render_channel_commentary")
    except ValueError:
        top_bottom = pd.DataFrame()
    if not top_bottom.empty:
        top_block = top_bottom[top_bottom[bucket_col] == "top"]
        bottom_block = top_bottom[top_bottom[bucket_col] == "bottom"]
        if not top_block.empty:
            r = top_block.iloc[0]
            name = r.get(dim_col, "—")
            val = r.get(metric_col)
            bullets.append(f"Top channel by NNB: {name} ({_fmt_money(val)}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            name = r.get(dim_col, "—")
            val = r.get(metric_col)
            bullets.append(f"Bottom channel by NNB: {name} ({_fmt_money(val)}).")
        table = top_bottom
    else:
        top_block = rank[rank[bucket_col] == "top"] if bucket_col in rank.columns else rank.head(TOP_N)
        bottom_block = rank[rank[bucket_col] == "bottom"] if bucket_col in rank.columns else rank.tail(BOTTOM_N)
        if not top_block.empty:
            r = top_block.iloc[0]
            name = r.get(dim_col, r.get("name", "—"))
            val = r.get(metric_col, r.get(value_col))
            bullets.append(f"Top channel: {name} ({_fmt_money(val)}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            name = r.get(dim_col, r.get("name", "—"))
            val = r.get(metric_col, r.get(value_col))
            bullets.append(f"Bottom channel: {name} ({_fmt_money(val)}).")
        table = rank.head(TOP_N + BOTTOM_N)

    for share_delta_col in _rank_share_delta_cols(rank):
        if share_delta_col not in rank.columns:
            continue
        try:
            shifted = rules.detect_mix_shift(rank, share_delta_col, MIX_SHIFT_THRESHOLD, dim_col, caller="render_channel_commentary")
        except ValueError:
            shifted = pd.DataFrame()
        if not shifted.empty:
            r = shifted.iloc[0]
            ch_name = r.get(dim_col, "—")
            delta = r.get(share_delta_col)
            bullets.append(f"Mix shift: {ch_name} share moved {_fmt_pct(delta)}.")
            break

    bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot: {month_end}."])
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)


# --- Product commentary -------------------------------------------------------

def render_product_commentary(pack: ReportPack) -> SectionOutput:
    """Top/bottom ticker by NNB, mix shift; if etf_rank available, one bullet for top ETF mover. Table: ticker top/bottom."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "ticker_rank, etf_rank", "thresholds_used": [f"MIX_SHIFT_THRESHOLD={MIX_SHIFT_THRESHOLD}"]}
    table_title = "Top/Bottom Products (Tickers)"
    rank = pack.ticker_rank
    etf = pack.etf_rank
    snap = _safe_first_row(pack.firm_snapshot)

    if rank is None or rank.empty:
        bullets.append("No product rank data available.")
        bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot month: {month_end}."])
        return SectionOutput(bullets=bullets, table_title=table_title, table=pd.DataFrame(), meta=meta)

    dim_col = _rank_dim_col(rank)
    value_col = _rank_value_col(rank)
    nnb_col = _rank_nnb_col(rank)
    metric_col = nnb_col if nnb_col else value_col
    bucket_col = "bucket" if "bucket" in rank.columns else "segment"

    try:
        top_bottom = rules.select_top_bottom(rank, metric_col, dim_col, top_n=TOP_N, bottom_n=BOTTOM_N, caller="render_product_commentary")
    except ValueError:
        top_bottom = pd.DataFrame()
    if not top_bottom.empty:
        top_block = top_bottom[top_bottom[bucket_col] == "top"]
        bottom_block = top_bottom[top_bottom[bucket_col] == "bottom"]
        if not top_block.empty:
            r = top_block.iloc[0]
            bullets.append(f"Top ticker by NNB: {r.get(dim_col, '—')} ({_fmt_money(r.get(metric_col))}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            bullets.append(f"Bottom ticker by NNB: {r.get(dim_col, '—')} ({_fmt_money(r.get(metric_col))}).")
        table = top_bottom
    else:
        top_block = rank[rank[bucket_col] == "top"] if bucket_col in rank.columns else rank.head(TOP_N)
        bottom_block = rank[rank[bucket_col] == "bottom"] if bucket_col in rank.columns else rank.tail(BOTTOM_N)
        if not top_block.empty:
            r = top_block.iloc[0]
            bullets.append(f"Top ticker: {r.get(dim_col, r.get('name', '—'))} ({_fmt_money(r.get(metric_col, r.get(value_col)))}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            bullets.append(f"Bottom ticker: {r.get(dim_col, r.get('name', '—'))} ({_fmt_money(r.get(metric_col, r.get(value_col)))}).")
        table = rank.head(TOP_N + BOTTOM_N)

    for share_delta_col in _rank_share_delta_cols(rank):
        if share_delta_col not in rank.columns:
            continue
        try:
            shifted = rules.detect_mix_shift(rank, share_delta_col, MIX_SHIFT_THRESHOLD, dim_col, caller="render_product_commentary")
        except ValueError:
            shifted = pd.DataFrame()
        if not shifted.empty:
            r = shifted.iloc[0]
            bullets.append(f"Mix shift: {r.get(dim_col, '—')} share moved {_fmt_pct(r.get(share_delta_col))}.")
            break

    if etf is not None and not etf.empty:
        edim = _rank_dim_col(etf)
        emetric = _rank_nnb_col(etf) or _rank_value_col(etf)
        if emetric in etf.columns and edim in etf.columns:
            ebucket = "bucket" if "bucket" in etf.columns else "segment"
            top_etf = etf[etf[ebucket] == "top"] if ebucket in etf.columns else etf.head(1)
            if top_etf.empty:
                top_etf = etf.head(1)
            if not top_etf.empty:
                r = top_etf.iloc[0]
                bullets.append(f"Top ETF mover: {r.get(edim, '—')} ({_fmt_money(r.get(emetric))}).")

    bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot: {month_end}."])
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)


# --- Geo commentary -----------------------------------------------------------

def render_geo_commentary(pack: ReportPack) -> SectionOutput:
    """If geo_rank empty: one geo-unavailable bullet + one from firm context. Else: top/bottom geo by NNB, mix shift. Table: Top/Bottom Geographies."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "geo_rank", "thresholds_used": [f"MIX_SHIFT_THRESHOLD={MIX_SHIFT_THRESHOLD}"]}
    table_title = "Top/Bottom Geographies"
    rank = pack.geo_rank
    snap = _safe_first_row(pack.firm_snapshot)

    if rank is None or rank.empty:
        bullets.append("Geo breakdown not available under current dataset columns.")
        bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot month: {month_end}."])
        return SectionOutput(bullets=bullets, table_title=table_title, table=pd.DataFrame(), meta=meta)

    dim_col = _rank_dim_col(rank)
    value_col = _rank_value_col(rank)
    nnb_col = _rank_nnb_col(rank)
    metric_col = nnb_col if nnb_col else value_col
    bucket_col = "bucket" if "bucket" in rank.columns else "segment"

    try:
        top_bottom = rules.select_top_bottom(rank, metric_col, dim_col, top_n=TOP_N, bottom_n=BOTTOM_N, caller="render_geo_commentary")
    except ValueError:
        top_bottom = pd.DataFrame()
    if not top_bottom.empty:
        top_block = top_bottom[top_bottom[bucket_col] == "top"]
        bottom_block = top_bottom[top_bottom[bucket_col] == "bottom"]
        if not top_block.empty:
            r = top_block.iloc[0]
            bullets.append(f"Top geography by NNB: {r.get(dim_col, '—')} ({_fmt_money(r.get(metric_col))}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            bullets.append(f"Bottom geography by NNB: {r.get(dim_col, '—')} ({_fmt_money(r.get(metric_col))}).")
        table = top_bottom
    else:
        top_block = rank[rank[bucket_col] == "top"] if bucket_col in rank.columns else rank.head(TOP_N)
        bottom_block = rank[rank[bucket_col] == "bottom"] if bucket_col in rank.columns else rank.tail(BOTTOM_N)
        if not top_block.empty:
            r = top_block.iloc[0]
            bullets.append(f"Top geography: {r.get(dim_col, r.get('name', '—'))} ({_fmt_money(r.get(metric_col, r.get(value_col)))}).")
        if not bottom_block.empty:
            r = bottom_block.iloc[0]
            bullets.append(f"Bottom geography: {r.get(dim_col, r.get('name', '—'))} ({_fmt_money(r.get(metric_col, r.get(value_col)))}).")
        table = rank.head(TOP_N + BOTTOM_N)

    for share_delta_col in _rank_share_delta_cols(rank):
        if share_delta_col not in rank.columns:
            continue
        try:
            shifted = rules.detect_mix_shift(rank, share_delta_col, MIX_SHIFT_THRESHOLD, dim_col, caller="render_geo_commentary")
        except ValueError:
            shifted = pd.DataFrame()
        if not shifted.empty:
            r = shifted.iloc[0]
            bullets.append(f"Mix shift: {r.get(dim_col, '—')} share moved {_fmt_pct(r.get(share_delta_col))}.")
            break

    bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot: {month_end}."])
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)


# --- Anomalies -----------------------------------------------------------------

def render_anomalies(pack: ReportPack) -> SectionOutput:
    """If empty: two bullets (none triggered + one from overview). Else: top 1–3 high-severity by severity then abs(zscore); each bullet metric, entity, zscore, value. Table: top 10."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "anomalies", "triggered_rules": [], "thresholds_used": []}
    anom = pack.anomalies
    snap = _safe_first_row(pack.firm_snapshot)

    if anom is None or anom.empty:
        bullets.append("No anomalies triggered under current thresholds.")
        bullets = _ensure_bullet_count(bullets, snap, ["Firm snapshot month: {month_end}; AUM {end_aum}."])
        table = pd.DataFrame(columns=anom.columns) if anom is not None and hasattr(anom, "columns") else pd.DataFrame()
        return SectionOutput(bullets=bullets, table_title="Anomalies (none)", table=table, meta=meta)

    table_title = "Anomalies (top 10)"
    n = len(anom)
    bullets.append(f"Total anomalies triggered: {n}.")
    if "rule_id" in anom.columns:
        meta["triggered_rules"] = anom["rule_id"].dropna().unique().tolist()

    has_z = "zscore" in anom.columns
    has_sev = "severity" in anom.columns
    sev_order = {"high": 0, "medium": 1, "med": 1, "low": 2}
    sorted_anom = anom
    if has_sev and has_z:
        sorted_anom = anom.copy()
        sorted_anom["_sev_ord"] = sorted_anom["severity"].map(sev_order).fillna(1)
        sorted_anom["_abs_z"] = sorted_anom["zscore"].abs()
        sorted_anom = sorted_anom.sort_values(by=["_sev_ord", "_abs_z"], ascending=[True, False]).drop(columns=["_sev_ord", "_abs_z"], errors="ignore")
    elif has_sev:
        sorted_anom = anom.copy()
        sorted_anom["_sev_ord"] = sorted_anom["severity"].map(sev_order).fillna(1)
        sorted_anom = sorted_anom.sort_values(by=["_sev_ord"], ascending=[True]).drop(columns=["_sev_ord"], errors="ignore")
    elif has_z:
        sorted_anom = anom.copy()
        sorted_anom["_abs_z"] = sorted_anom["zscore"].abs()
        sorted_anom = sorted_anom.sort_values(by=["_abs_z"], ascending=[False]).drop(columns=["_abs_z"], errors="ignore")
    top_anom = sorted_anom.head(3)
    for _, r in top_anom.iterrows():
        metric = r.get("metric", "—")
        entity = r.get("entity", "—")
        zscore = r.get("zscore")
        val = r.get("value_current")
        bullets.append(f"{metric} ({entity}): z={_fmt_num(zscore)}, value={_fmt_num(val)}.")
    table = anom.head(10)
    bullets = _ensure_bullet_count(bullets, snap, ["Snapshot: {month_end}.", "Firm AUM: {end_aum}."])
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)


# --- Recommendations ----------------------------------------------------------

def render_recommendations(pack: ReportPack) -> SectionOutput:
    """Deterministic actions from anomalies and mix shift only: strong flows + negative markets, mix shift by channel, high-severity ticker. Cap 5. Table: triggered anomalies or mix shift."""
    bullets: list[str] = []
    meta: dict[str, Any] = {"source": "anomalies, channel/ticker/geo rank mix shift", "thresholds_used": []}
    table_title = "Recommendations"
    anom = pack.anomalies
    snap = _safe_first_row(pack.firm_snapshot)
    row = snap

    ogr_f = float("nan")
    mkt_neg = False
    if row:
        try:
            ogr_f = float(row.get("ogr")) if row.get("ogr") is not None and pd.notna(row.get("ogr")) else float("nan")
        except (TypeError, ValueError):
            pass
        rate = row.get("market_impact_rate")
        abs_val = row.get("market_impact_abs")
        if rate is not None and pd.notna(rate) and float(rate) < 0:
            mkt_neg = True
        elif abs_val is not None and pd.notna(abs_val) and float(abs_val) < 0:
            mkt_neg = True

    if ogr_f > OGR_STRONG and mkt_neg:
        bullets.append("Review hedging / risk messaging (strong flows, negative markets).")

    for rank, level_name in [(pack.channel_rank, "channel"), (pack.ticker_rank, "ticker"), (pack.geo_rank, "geo")]:
        if rank is None or rank.empty:
            continue
        dim_col = _rank_dim_col(rank)
        for share_delta_col in _rank_share_delta_cols(rank):
            if share_delta_col not in rank.columns:
                continue
            try:
                shifted = rules.detect_mix_shift(rank, share_delta_col, MIX_SHIFT_THRESHOLD, dim_col, caller="render_recommendations")
            except ValueError:
                shifted = pd.DataFrame()
            if not shifted.empty:
                r = shifted.iloc[0]
                name = r.get(dim_col, "—")
                bullets.append(f"Allocate sales focus to {name} (significant {level_name} mix shift).")
                meta["thresholds_used"] = list(set(meta["thresholds_used"] + [f"MIX_SHIFT_THRESHOLD={MIX_SHIFT_THRESHOLD}"]))
                break

    if anom is not None and not anom.empty and "severity" in anom.columns and "entity" in anom.columns:
        high_anom = anom[anom["severity"] == "high"]
        ticker_high = high_anom[high_anom["level"] == "ticker"] if "level" in anom.columns else high_anom
        if not ticker_high.empty:
            e = ticker_high.iloc[0].get("entity", "—")
            bullets.append(f"Investigate {e} driver: flows vs pricing.")
        elif not high_anom.empty:
            e = high_anom.iloc[0].get("entity", "—")
            bullets.append(f"Investigate {e} (high-severity anomaly).")

    bullets = bullets[:MAX_BULLETS]
    table = pd.DataFrame()
    if anom is not None and not anom.empty:
        table = anom.head(10)
    else:
        for rank in (pack.channel_rank, pack.ticker_rank, pack.geo_rank):
            if rank is None or rank.empty:
                continue
            dim_col = _rank_dim_col(rank)
            for share_delta_col in _rank_share_delta_cols(rank):
                if share_delta_col not in rank.columns:
                    continue
                try:
                    shifted = rules.detect_mix_shift(rank, share_delta_col, MIX_SHIFT_THRESHOLD, dim_col, caller="render_recommendations")
                except ValueError:
                    shifted = pd.DataFrame()
                if not shifted.empty and (table.empty or len(shifted) > len(table)):
                    table = shifted.head(10)
    if bullets == []:
        bullets.append("No recommendations triggered by anomalies or mix shift.")
        bullets = _ensure_bullet_count(bullets, snap, ["Firm AUM: {end_aum}.", "Snapshot: {month_end}."])
    return SectionOutput(bullets=bullets, table_title=table_title, table=table, meta=meta)
