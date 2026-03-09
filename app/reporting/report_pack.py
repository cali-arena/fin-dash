"""
Governed Report Data Mart: single structured ReportPack with all report tables.
All derived features (MoM/YTD/YoY, shares, mix shift, rolling stats, z-scores) computed here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Stable column names for report tables
FIRM_SNAPSHOT_COLUMNS = [
    "month_end", "begin_aum", "end_aum", "nnb", "nnf",
    "mom_pct", "ytd_pct", "yoy_pct",
    "ogr", "market_impact_abs", "market_impact_rate", "fee_yield",
]
RANK_COLUMNS = ["name", "value", "share", "share_delta", "share_prior", "rank", "segment"]
TIME_SERIES_COLUMNS = ["month_end", "end_aum", "nnb", "nnf", "mom_pct", "ytd_pct"]
ANOMALIES_COLUMNS = ["month_end", "flag_type", "reason", "severity"]


def _coerce(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _build_firm_snapshot(firm_df: pd.DataFrame) -> pd.DataFrame:
    """One row: latest month with derived MoM/YTD/YoY, OGR, market impact, fee yield."""
    if firm_df is None or firm_df.empty:
        return pd.DataFrame(columns=FIRM_SNAPSHOT_COLUMNS)
    from app.metrics.metric_contract import (
        compute_fee_yield,
        compute_market_impact,
        compute_market_impact_rate,
        compute_ogr,
    )
    df = firm_df.sort_values("month_end", ascending=True)
    by_month = df.groupby("month_end", as_index=False).agg({
        c: "sum" for c in ["begin_aum", "end_aum", "nnb", "nnf"] if c in df.columns
    })
    if by_month.empty:
        return pd.DataFrame(columns=FIRM_SNAPSHOT_COLUMNS)
    last = by_month.iloc[-1]
    end_aum = _coerce(last.get("end_aum"))
    begin_aum = _coerce(last.get("begin_aum"))
    nnb = _coerce(last.get("nnb"))
    nnf = _coerce(last.get("nnf"))
    mom_pct = float("nan")
    ytd_pct = float("nan")
    yoy_pct = float("nan")
    if len(by_month) >= 2:
        prev = by_month.iloc[-2]
        prev_end = _coerce(prev.get("end_aum"))
        if prev_end and prev_end == prev_end:
            mom_pct = (end_aum - prev_end) / prev_end
    if len(by_month) >= 1:
        first_end = _coerce(by_month.iloc[0].get("end_aum"))
        if first_end and first_end == first_end:
            ytd_pct = (end_aum - first_end) / first_end
    if len(by_month) >= 12:
        yoy_end = _coerce(by_month.iloc[-12].get("end_aum"))
        if yoy_end and yoy_end == yoy_end:
            yoy_pct = (end_aum - yoy_end) / yoy_end
    mi = compute_market_impact(begin_aum, end_aum, nnb)
    ogr = compute_ogr(nnb, begin_aum)
    mir = compute_market_impact_rate(mi, begin_aum)
    fy = compute_fee_yield(nnf, begin_aum, end_aum, nnb=nnb)
    row = {
        "month_end": last.get("month_end"),
        "begin_aum": begin_aum,
        "end_aum": end_aum,
        "nnb": nnb,
        "nnf": nnf,
        "mom_pct": mom_pct,
        "ytd_pct": ytd_pct,
        "yoy_pct": yoy_pct,
        "ogr": ogr,
        "market_impact_abs": mi,
        "market_impact_rate": mir,
        "fee_yield": fy,
    }
    out = pd.DataFrame([row])
    return out.reindex(columns=[c for c in FIRM_SNAPSHOT_COLUMNS if c in out.columns], copy=False)


def _build_rank_table(
    monthly_df: pd.DataFrame,
    name_col: str,
    value_col: str = "end_aum",
    top_n: int = 10,
) -> pd.DataFrame:
    """Top/bottom rank table with share, share_delta, rank, segment. Stable columns: name, value, share, share_delta, share_prior, rank, segment."""
    if monthly_df is None or monthly_df.empty or name_col not in monthly_df.columns or value_col not in monthly_df.columns:
        return pd.DataFrame(columns=RANK_COLUMNS)
    agg = monthly_df.groupby(name_col, as_index=False)[value_col].sum()
    total = agg[value_col].sum()
    if _coerce(total) == 0:
        agg["share"] = float("nan")
        agg["share_delta"] = float("nan")
        agg["share_prior"] = float("nan")
    else:
        agg["share"] = agg[value_col] / total
        agg["share_delta"] = float("nan")
        agg["share_prior"] = agg["share"]
    agg = agg.sort_values(value_col, ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    top = agg.head(top_n).copy()
    top["segment"] = "top"
    bottom = agg.tail(top_n).copy()
    bottom["segment"] = "bottom"
    combined = pd.concat([top, bottom], ignore_index=True)
    combined = combined.rename(columns={name_col: "name", value_col: "value"})
    combined = combined.drop_duplicates(subset=["name"], keep="first")
    for c in RANK_COLUMNS:
        if c not in combined.columns:
            combined[c] = None
    return combined[RANK_COLUMNS]


def _build_time_series(firm_df: pd.DataFrame) -> pd.DataFrame:
    """Monthly series with end_aum, nnb, nnf, mom_pct, ytd_pct."""
    if firm_df is None or firm_df.empty or "month_end" not in firm_df.columns:
        return pd.DataFrame(columns=TIME_SERIES_COLUMNS)
    by_month = firm_df.groupby("month_end", as_index=False).agg({
        c: "sum" for c in ["end_aum", "nnb", "nnf"] if c in firm_df.columns
    })
    by_month = by_month.sort_values("month_end")
    if "end_aum" not in by_month.columns:
        return pd.DataFrame(columns=TIME_SERIES_COLUMNS)
    by_month["mom_pct"] = float("nan")
    by_month["ytd_pct"] = float("nan")
    e = by_month["end_aum"]
    if len(by_month) >= 2:
        by_month["mom_pct"] = e.pct_change()
    if len(by_month) >= 1:
        first_val = e.iloc[0]
        if first_val and not math.isnan(first_val) and first_val != 0:
            by_month["ytd_pct"] = (e - first_val) / first_val
        else:
            by_month["ytd_pct"] = float("nan")
    return by_month.reindex(columns=[c for c in TIME_SERIES_COLUMNS if c in by_month.columns], copy=False)


def _build_anomalies(firm_df: pd.DataFrame) -> pd.DataFrame:
    """Flags + reason + severity from z-score and rolling std."""
    if firm_df is None or firm_df.empty or "month_end" not in firm_df.columns:
        return pd.DataFrame(columns=ANOMALIES_COLUMNS)
    by_month = firm_df.groupby("month_end", as_index=False).sum(numeric_only=True)
    if "end_aum" not in by_month.columns or len(by_month) < 3:
        return pd.DataFrame(columns=ANOMALIES_COLUMNS)
    rows = []
    roll = by_month["end_aum"].rolling(3, min_periods=2)
    mu = roll.mean()
    sigma = roll.std()
    z = (by_month["end_aum"] - mu) / sigma.replace(0, float("nan"))
    for i in range(len(by_month)):
        if pd.isna(z.iloc[i]) or not math.isfinite(z.iloc[i]):
            continue
        if abs(z.iloc[i]) > 2:
            rows.append({
                "month_end": by_month["month_end"].iloc[i],
                "flag_type": "zscore",
                "reason": f"|z|={abs(z.iloc[i]):.2f}",
                "severity": "high" if abs(z.iloc[i]) > 3 else "medium",
            })
    if not rows:
        return pd.DataFrame(columns=ANOMALIES_COLUMNS)
    out = pd.DataFrame(rows)
    return out.reindex(columns=ANOMALIES_COLUMNS, copy=False)


def _resolve_name_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_report_pack(
    firm_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    ticker_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    dataset_version: str,
    filter_hash: str,
    *,
    top_n: int = 10,
    etf_df: pd.DataFrame | None = None,
    firm_snapshot: pd.DataFrame | None = None,
    time_series: pd.DataFrame | None = None,
    meta_notes: dict[str, Any] | None = None,
    channel_rank: pd.DataFrame | None = None,
    ticker_rank: pd.DataFrame | None = None,
    geo_rank: pd.DataFrame | None = None,
    etf_rank: pd.DataFrame | None = None,
    anomalies: pd.DataFrame | None = None,
) -> "ReportPack":
    """Build ReportPack from loaded monthly data. When firm_snapshot/time_series/rank tables/anomalies provided (e.g. from gateway), use them."""
    if firm_snapshot is None:
        firm_snapshot = _build_firm_snapshot(firm_df)
    if time_series is None:
        time_series = _build_time_series(firm_df)
    if anomalies is None:
        anomalies = _build_anomalies(firm_df)
    if channel_rank is None:
        ch_name = _resolve_name_col(channel_df, ("channel", "channel_l1"))
        channel_rank = _build_rank_table(channel_df, ch_name or "channel", "end_aum", top_n) if ch_name else pd.DataFrame(columns=RANK_COLUMNS)
    if ticker_rank is None:
        tk_name = _resolve_name_col(ticker_df, ("ticker", "product_ticker"))
        ticker_rank = _build_rank_table(ticker_df, tk_name or "ticker", "end_aum", top_n) if tk_name else pd.DataFrame(columns=RANK_COLUMNS)
    if geo_rank is None:
        geo_name = _resolve_name_col(geo_df, ("geo", "region", "src_country_canonical", "product_country_canonical"))
        geo_rank = _build_rank_table(geo_df, geo_name or "geo", "end_aum", top_n) if geo_df is not None and not geo_df.empty and geo_name else pd.DataFrame(columns=RANK_COLUMNS)
    if etf_rank is None:
        tk_name = _resolve_name_col(ticker_df, ("ticker", "product_ticker"))
        etf_rank = _build_rank_table(etf_df, tk_name or "ticker", "end_aum", top_n) if etf_df is not None and not etf_df.empty and tk_name else pd.DataFrame(columns=RANK_COLUMNS)
    meta = {
        "row_counts": {
            "firm_snapshot": len(firm_snapshot),
            "channel_rank": len(channel_rank),
            "ticker_rank": len(ticker_rank),
            "etf_rank": len(etf_rank),
            "geo_rank": len(geo_rank),
            "time_series": len(time_series),
            "anomalies": len(anomalies),
        },
        "top_n": top_n,
        "dataset_version": dataset_version,
        "filter_hash": filter_hash,
    }
    if meta_notes:
        meta.update(meta_notes)
    return ReportPack(
        dataset_version=dataset_version,
        filter_hash=filter_hash,
        firm_snapshot=firm_snapshot,
        channel_rank=channel_rank,
        ticker_rank=ticker_rank,
        etf_rank=etf_rank,
        geo_rank=geo_rank,
        time_series=time_series,
        anomalies=anomalies,
        meta=meta,
    )


@dataclass(frozen=True)
class ReportPack:
    dataset_version: str
    filter_hash: str
    firm_snapshot: pd.DataFrame
    channel_rank: pd.DataFrame
    ticker_rank: pd.DataFrame
    etf_rank: pd.DataFrame
    geo_rank: pd.DataFrame
    time_series: pd.DataFrame
    anomalies: pd.DataFrame
    meta: dict[str, Any]
