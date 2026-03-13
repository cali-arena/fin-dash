"""Shared formatting helpers for the dashboard UI."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable

import pandas as pd

NA_STR = "-"

CURRENCY_COLUMNS = frozenset(
    {
        "aum",
        "end_aum",
        "begin_aum",
        "nnb",
        "nnf",
        "market_pnl",
        "market_impact",
        "market_impact_abs",
        "total_end_aum",
        "total_begin_aum",
        "nnb_total",
        "nnf_total",
    }
)
PERCENT_COLUMNS = frozenset(
    {"ogr", "mom_pct", "ytd_pct", "yoy_pct", "market_impact_rate", "fee_yield", "fee_yield_proxy"}
)


def fmt_currency(x: float | None, unit: str = "auto", decimals: int = 2, symbol: str = "$") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return NA_STR
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA_STR
    if unit != "auto":
        return f"{symbol}{v:,.{decimals}f}"
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{symbol}{v / 1e9:,.{decimals}f}B"
    if abs_v >= 1e6:
        return f"{symbol}{v / 1e6:,.{decimals}f}M"
    if abs_v >= 1e3:
        return f"{symbol}{v / 1e3:,.{decimals}f}K"
    return f"{symbol}{v:,.{decimals}f}"


def fmt_currency_kpi(x: float | None, decimals: int = 2, symbol: str = "$") -> str:
    """Canonical KPI/card currency formatter for dashboard and commentary. Same B/M/K abbreviation everywhere."""
    return fmt_currency(x, unit="auto", decimals=decimals, symbol=symbol)


def fmt_percent(x: float | None, decimals: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return NA_STR
    try:
        val = float(x) * 100.0
    except (TypeError, ValueError):
        return NA_STR
    if signed and val > 0:
        return f"+{val:.{decimals}f}%"
    return f"{val:.{decimals}f}%"


def fmt_bps(x: float | None, decimals: int = 0) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return NA_STR
    try:
        bps = float(x) * 10000.0
    except (TypeError, ValueError):
        return NA_STR
    return f"{bps:.{decimals}f} bps"


def fmt_number(x: float | None, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return NA_STR
    try:
        return f"{float(x):,.{decimals}f}"
    except (TypeError, ValueError):
        return NA_STR


def fmt_date(month_end: date | datetime | str | None) -> str:
    if month_end is None:
        return NA_STR
    if isinstance(month_end, str):
        s = month_end.strip()
        if not s:
            return NA_STR
        try:
            return pd.to_datetime(s).strftime("%Y-%m")
        except Exception:
            return s
    try:
        return pd.Timestamp(month_end).strftime("%Y-%m")
    except Exception:
        return NA_STR


def hover_template_currency(label: str = "Value") -> str:
    return f"{label}: %{{y:,.2f}}<extra></extra>"


def hover_template_percent(label: str = "Rate") -> str:
    return f"{label}: %{{y:.2%}}<extra></extra>"


def format_df(df: pd.DataFrame, col_formats: dict[str, Callable[[Any], str]]) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    for col, formatter in col_formats.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v: formatter(v)).astype(str)
    return out


def infer_common_formats(df: pd.DataFrame) -> dict[str, Callable[[Any], str]]:
    if df is None or not hasattr(df, "columns"):
        return {}
    result: dict[str, Callable[[Any], str]] = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        if key in CURRENCY_COLUMNS:
            result[c] = lambda x: fmt_currency_kpi(x, decimals=2)
        elif key in PERCENT_COLUMNS:
            result[c] = lambda x: fmt_percent(x, decimals=2, signed=False)
    return result
