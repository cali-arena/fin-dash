from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.kpi.service import apply_period_canonical
from app.state import FilterState


@dataclass(frozen=True)
class MetricPayload:
    source_df: pd.DataFrame
    df_filtered: pd.DataFrame
    df_period: pd.DataFrame
    monthly_full: pd.DataFrame
    monthly_period: pd.DataFrame
    kpi_snapshot: dict[str, Any]
    reconciliation: dict[str, Any]
    scope_label: str
    period: str


def _pick_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_currency(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return 0.0
        return float(value)
    s = str(value).strip()
    if s in {"", "-", "—", "–", "nan", "None"}:
        return 0.0
    negative = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").replace("$", "").replace(",", "").strip()
    try:
        parsed = float(s)
    except ValueError:
        return 0.0
    return -abs(parsed) if negative else parsed


def normalize_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    date_col = _pick_col(out, ("month_end", "Date", "date", "Month End"))
    out["month_end"] = pd.to_datetime(out[date_col], errors="coerce") if date_col else pd.NaT

    channel_col = _pick_col(out, ("standard_channel", "Standard Channel", "channel_standard", "channel", "channel_l1", "channel_best", "preferred_label"))
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

    numeric_candidates = {
        "begin_aum": ("begin_aum", "Begin AUM", "beginning_aum", "opening_aum"),
        "end_aum": ("end_aum", "End AUM", "ending_aum", "closing_aum"),
        "nnb": ("nnb", "NNB", "Net New Business"),
        "nnf": ("nnf", "NNF", "Net New Flow"),
    }
    for canonical, candidates in numeric_candidates.items():
        src_col = _pick_col(out, candidates)
        if src_col is None:
            out[canonical] = float("nan")
            continue
        out[canonical] = out[src_col].apply(parse_currency)

    out = out.dropna(subset=["month_end"])
    for col in ("channel", "sub_channel", "country", "segment", "sub_segment", "product_ticker"):
        out[col] = out[col].replace("", "Unassigned").fillna("Unassigned")
    return out


def apply_filters(
    df: pd.DataFrame,
    *,
    channel: str | None = None,
    sub_channel: str | None = None,
    country: str | None = None,
    segment: str | None = None,
    sub_segment: str | None = None,
    ticker: str | None = None,
    date_from: Any | None = None,
    date_to: Any | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else None)
    out = df.copy()
    if date_from is not None:
        out = out[out["month_end"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        out = out[out["month_end"] <= pd.Timestamp(date_to)]
    eq_filters = {
        "channel": channel,
        "sub_channel": sub_channel,
        "country": country,
        "segment": segment,
        "sub_segment": sub_segment,
        "product_ticker": ticker,
    }
    for col, value in eq_filters.items():
        if value in (None, "", "All"):
            continue
        if col in out.columns:
            out = out[out[col].astype(str) == str(value)]
    return out


def calculate_metrics(df_filtered: pd.DataFrame, *, tol: float = 1.0) -> pd.DataFrame:
    cols = ["month_end", "end_aum", "nnb", "nnf"]
    if df_filtered is None or df_filtered.empty:
        return pd.DataFrame(columns=cols + ["begin_aum", "market_pnl", "ogr", "market_impact", "market_impact_rate", "fee_yield", "reconciled"])

    monthly = (
        df_filtered.groupby("month_end", as_index=False)[["end_aum", "nnb", "nnf"]]
        .sum(min_count=1)
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    monthly["begin_aum"] = monthly["end_aum"].shift(1)
    monthly = monthly.dropna(subset=["begin_aum"]).reset_index(drop=True)
    monthly["market_pnl"] = monthly["end_aum"] - monthly["begin_aum"] - monthly["nnb"]
    denom = monthly["begin_aum"].replace(0, pd.NA)
    monthly["ogr"] = monthly["nnb"] / denom
    monthly["market_impact"] = monthly["market_pnl"] / denom
    monthly["market_impact_rate"] = monthly["market_impact"]
    monthly["fee_yield"] = monthly["nnf"] / monthly["nnb"].replace(0, pd.NA)
    recon_diff = (monthly["begin_aum"] + monthly["nnb"] + monthly["market_pnl"] - monthly["end_aum"]).abs()
    monthly["reconciled"] = recon_diff <= tol
    return monthly


def get_kpi_snapshot(monthly: pd.DataFrame, period: str = "1M") -> dict[str, Any]:
    normalized_period = "QoQ" if str(period).upper() == "QTD" else period
    out = {
        "period": normalized_period,
        "month_end": None,
        "begin_aum": float("nan"),
        "end_aum": float("nan"),
        "nnb": float("nan"),
        "nnf": float("nan"),
        "market_pnl": float("nan"),
        "ogr": float("nan"),
        "market_impact": float("nan"),
        "fee_yield": float("nan"),
        "reconciled": False,
    }
    if monthly is None or monthly.empty:
        return out
    work = apply_period_canonical(monthly, normalized_period).sort_values("month_end")
    if work.empty:
        return out
    latest = work.iloc[-1]
    begin_aum = float(work.iloc[0].get("begin_aum", float("nan")))
    end_aum = float(latest.get("end_aum", float("nan")))
    nnb = float(work["nnb"].sum())
    nnf = float(work["nnf"].sum())
    market_pnl = float(work["market_pnl"].sum())
    denom = begin_aum if begin_aum not in (0.0,) and pd.notna(begin_aum) else float("nan")
    out.update(
        {
            "month_end": latest.get("month_end"),
            "begin_aum": begin_aum,
            "end_aum": end_aum,
            "nnb": nnb,
            "nnf": nnf,
            "market_pnl": market_pnl,
            "ogr": (nnb / denom) if pd.notna(denom) else float("nan"),
            "market_impact": (market_pnl / denom) if pd.notna(denom) else float("nan"),
            "fee_yield": (nnf / nnb) if nnb not in (0.0,) and pd.notna(nnb) else float("nan"),
            "reconciled": abs(begin_aum + nnb + market_pnl - end_aum) <= 1.0,
        }
    )
    return out


def reconciliation_check(kpi: dict[str, Any], *, tolerance: float = 1.0) -> dict[str, Any]:
    begin_aum = pd.to_numeric(kpi.get("begin_aum"), errors="coerce")
    nnb = pd.to_numeric(kpi.get("nnb"), errors="coerce")
    market_pnl = pd.to_numeric(kpi.get("market_pnl"), errors="coerce")
    end_aum = pd.to_numeric(kpi.get("end_aum"), errors="coerce")
    variance = begin_aum + nnb + market_pnl - end_aum
    ok = (
        pd.notna(begin_aum)
        and pd.notna(nnb)
        and pd.notna(market_pnl)
        and pd.notna(end_aum)
        and abs(float(variance)) <= tolerance
    )
    return {"ok": bool(ok), "variance": float(variance) if pd.notna(variance) else float("nan")}


def build_metric_payload(
    *,
    gateway: Any,
    state: FilterState,
    scope_label: str,
    period: str,
    channel: str | None,
    sub_channel: str | None,
    country: str | None,
    segment: str | None,
    sub_segment: str | None,
    ticker: str | None,
) -> MetricPayload:
    frames = [
        normalize_base_frame(gateway.run_query("ticker_monthly", state)),
        normalize_base_frame(gateway.run_query("channel_monthly", state)),
        normalize_base_frame(gateway.run_query("segment_monthly", state)),
        normalize_base_frame(gateway.run_query("geo_monthly", state)),
        normalize_base_frame(gateway.run_query("firm_monthly", state)),
    ]
    source_df = next((f for f in frames if f is not None and not f.empty), pd.DataFrame())
    if source_df.empty:
        empty = pd.DataFrame()
        empty_kpi = get_kpi_snapshot(empty, period)
        return MetricPayload(
            source_df=empty,
            df_filtered=empty,
            df_period=empty,
            monthly_full=empty,
            monthly_period=empty,
            kpi_snapshot=empty_kpi,
            reconciliation=reconciliation_check(empty_kpi),
            scope_label=scope_label,
            period=period,
        )

    df_filtered = apply_filters(
        source_df,
        channel=channel,
        sub_channel=sub_channel,
        country=country,
        segment=segment,
        sub_segment=sub_segment,
        ticker=ticker,
        date_from=state.date_start,
        date_to=state.date_end,
    )
    monthly_full = calculate_metrics(df_filtered)
    monthly_period = apply_period_canonical(monthly_full, period)
    df_period = apply_period_canonical(df_filtered, period)
    kpi_snapshot = get_kpi_snapshot(monthly_full, period=period)
    return MetricPayload(
        source_df=source_df,
        df_filtered=df_filtered,
        df_period=df_period,
        monthly_full=monthly_full,
        monthly_period=monthly_period,
        kpi_snapshot=kpi_snapshot,
        reconciliation=reconciliation_check(kpi_snapshot),
        scope_label=scope_label,
        period=period,
    )
