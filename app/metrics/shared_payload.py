from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.kpi.service import apply_period_canonical
from app.state import FilterState

STD_CHANNEL_CANDIDATES = (
    "standard_channel",
    "Standard Channel",
    "channel_standard",
    "channel_l1",
    "channel_best",
    "preferred_label",
    "channel",
)
RAW_CHANNEL_CANDIDATES = ("Channel", "channel_raw")

# Mapping: std_channel_name (channel_final) → ibp_channel grouped level (channel_group)
# Display names match dim_lookup: "Bwm" → "National Private Bank", "Non-Us" → "Non-US / International"
CHANNEL_STD_TO_GROUP: dict[str, str] = {
    "Broker Dealer": "Broker Dealer",
    "National Private Bank": "National Private Bank",
    "Bank Brokerage": "National Private Bank",
    "Latam": "Non-US / International",
    "Emea": "Non-US / International",
    "Apac": "Non-US / International",
    "Canada": "Non-US / International",
    "Direct": "Direct",
    "Ml Direct": "Direct",
    "Fidelity Strategic Advisors": "Direct",
    "Gam": "Institutional",
    "Insurance": "Institutional",
    "Pension": "Institutional",
    "Fofe": "Institutional",
    "Blackrock": "Institutional",
    "Retirement Insurance": "Retirement Insurance",
    "Undefined Client": "Other",
    "Hedge Fund": "Other",
    "Bank Custody": "Other",
    "Plug": "Other",
    "Unassigned": "Unassigned",
}


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


def _normalize_blank_to_na(ser: pd.Series) -> pd.Series:
    """Strip whitespace and treat blank strings as NA."""
    if ser is None or ser.empty:
        return ser
    s = ser.astype(str).str.strip()
    return s.replace(r"^\s*$", pd.NA, regex=True).replace("nan", pd.NA).replace("None", pd.NA)


def _canonical_channel_from_std_raw(
    df: pd.DataFrame,
    std_candidates: tuple[str, ...],
    raw_candidates: tuple[str, ...],
    fallback: str = "Unassigned",
) -> pd.Series:
    """
    Build canonical channel: prefer non-empty standardized, else raw, else fallback.
    Treats blank/whitespace-only as empty.
    """
    std_col = _pick_col(df, std_candidates)
    raw_col = _pick_col(df, raw_candidates)
    std_ser = _normalize_blank_to_na(df[std_col]) if std_col else pd.Series([pd.NA] * len(df), index=df.index)
    raw_ser = _normalize_blank_to_na(df[raw_col]) if raw_col else pd.Series([pd.NA] * len(df), index=df.index)
    return std_ser.fillna(raw_ser).fillna(fallback)


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

    # Canonical channel: prefer standardized/mapped column if non-empty, else raw source Channel, else "Unassigned".
    std_channel_col = _pick_col(out, STD_CHANNEL_CANDIDATES)
    raw_channel_col = _pick_col(out, RAW_CHANNEL_CANDIDATES)
    out["channel_standard"] = _normalize_blank_to_na(out[std_channel_col]) if std_channel_col else pd.Series([pd.NA] * len(out), index=out.index)
    out["channel_raw"] = _normalize_blank_to_na(out[raw_channel_col]) if raw_channel_col else pd.Series([pd.NA] * len(out), index=out.index)
    out["channel_final"] = out["channel_standard"].fillna(out["channel_raw"]).fillna("Unassigned")
    # Backward compatibility for callers that still expect "channel".
    out["channel"] = out["channel_final"]

    # Sub-channel: same pattern (prefer channel_l2 / sub_channel, else raw Sub-Channel if present)
    std_sub_candidates = ("channel_l2", "sub_channel")
    raw_sub_candidates = ("Sub-Channel", "Sub-channel", "sub_channel_raw")
    out["sub_channel"] = _canonical_channel_from_std_raw(out, std_sub_candidates, raw_sub_candidates)

    country_col = _pick_col(out, ("src_country_canonical", "src_country", "product_country_canonical", "geo", "country", "region"))
    segment_col = _pick_col(out, ("segment",))
    sub_segment_col = _pick_col(out, ("sub_segment", "subsegment"))
    ticker_col = _pick_col(out, ("product_ticker", "ticker", "label"))
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
    for col in ("channel_standard", "channel_raw", "channel_final", "channel", "sub_channel", "country", "segment", "sub_segment", "product_ticker"):
        out[col] = out[col].replace("", "Unassigned").fillna("Unassigned")
    # channel_group: ibp_channel grouped level (Institutional, Broker Dealer, etc.)
    out["channel_group"] = out["channel_final"].map(CHANNEL_STD_TO_GROUP).fillna(out["channel_final"])
    return out


def _has_meaningful_values(df: pd.DataFrame, col: str) -> bool:
    if df is None or df.empty or col not in df.columns:
        return False
    s = df[col].astype(str).str.strip().replace("", "Unassigned")
    return bool((s != "Unassigned").any())


def _pick_source_frame(
    frames: dict[str, pd.DataFrame],
    *,
    channel: str | None,
    sub_channel: str | None,
    country: str | None,
    segment: str | None,
    sub_segment: str | None,
    ticker: str | None,
    ticker_allowlist: "list[str] | None" = None,
) -> pd.DataFrame:
    # Keep one canonical source per request, but choose the source that can honor active filters.
    if ticker_allowlist is not None and _has_meaningful_values(frames["ticker_monthly"], "product_ticker"):
        return frames["ticker_monthly"]
    if channel not in (None, "", "All") and _has_meaningful_values(frames["channel_monthly"], "channel_group"):
        return frames["channel_monthly"]
    # sub_channel is std_channel_name level → channel_final in normalized frames
    if sub_channel not in (None, "", "All") and _has_meaningful_values(frames["channel_monthly"], "channel_final"):
        return frames["channel_monthly"]
    if ticker not in (None, "", "All") and _has_meaningful_values(frames["ticker_monthly"], "product_ticker"):
        return frames["ticker_monthly"]
    if segment not in (None, "", "All") and _has_meaningful_values(frames["segment_monthly"], "segment"):
        return frames["segment_monthly"]
    if sub_segment not in (None, "", "All") and _has_meaningful_values(frames["segment_monthly"], "sub_segment"):
        return frames["segment_monthly"]
    if country not in (None, "", "All") and _has_meaningful_values(frames["geo_monthly"], "country"):
        return frames["geo_monthly"]
    for name in ("ticker_monthly", "channel_monthly", "segment_monthly", "geo_monthly", "firm_monthly"):
        df = frames.get(name, pd.DataFrame())
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


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
    ticker_allowlist: "list[str] | None" = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(df.columns) if isinstance(df, pd.DataFrame) else None)
    out = df.copy()
    if date_from is not None:
        out = out[out["month_end"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        out = out[out["month_end"] <= pd.Timestamp(date_to)]
    # channel → filter on channel_group (ibp_channel grouped level); fallback to legacy "channel" col
    if channel not in (None, "", "All"):
        if "channel_group" in out.columns:
            out = out[out["channel_group"].astype(str) == str(channel)]
        elif "channel" in out.columns:
            out = out[out["channel"].astype(str) == str(channel)]
    # sub_channel → filter on channel_final (std_channel_name level)
    if sub_channel not in (None, "", "All"):
        if "channel_final" in out.columns:
            out = out[out["channel_final"].astype(str) == str(sub_channel)]
        elif "sub_channel" in out.columns:
            out = out[out["sub_channel"].astype(str) == str(sub_channel)]
    # remaining equality filters
    eq_filters = {
        "country": country,
        "sub_segment": sub_segment,
        "product_ticker": ticker,
    }
    for col, value in eq_filters.items():
        if value in (None, "", "All"):
            continue
        if col in out.columns:
            out = out[out[col].astype(str) == str(value)]
    # ticker allowlist: filters to a set of tickers (used for sales_focus dimension)
    if ticker_allowlist is not None:
        if "product_ticker" in out.columns:
            allowed = {str(t) for t in ticker_allowlist}
            out = out[out["product_ticker"].astype(str).isin(allowed)]
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
    ticker_allowlist: "list[str] | None" = None,
) -> MetricPayload:
    frames = {
        "ticker_monthly": normalize_base_frame(gateway.run_query("ticker_monthly", state)),
        "channel_monthly": normalize_base_frame(gateway.run_query("channel_monthly", state)),
        "segment_monthly": normalize_base_frame(gateway.run_query("segment_monthly", state)),
        "geo_monthly": normalize_base_frame(gateway.run_query("geo_monthly", state)),
        "firm_monthly": normalize_base_frame(gateway.run_query("firm_monthly", state)),
    }
    source_df = _pick_source_frame(
        frames,
        channel=channel,
        sub_channel=sub_channel,
        country=country,
        segment=segment,
        sub_segment=sub_segment,
        ticker=ticker,
        ticker_allowlist=ticker_allowlist,
    )
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
        ticker_allowlist=ticker_allowlist,
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
