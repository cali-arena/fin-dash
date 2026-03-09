"""
DATA SUMMARY checksum formulas: single source of truth for firm-level rates.
Used by qa/validate_vs_data_summary and app firm_snapshot/time_series so definitions match exactly.

Formulas (DATA SUMMARY):
- asset_growth_rate = (end_aum - begin_aum) / begin_aum
- organic_growth_rate = nnb / begin_aum
- external_growth_rate = market_impact / begin_aum  with market_impact = end_aum - begin_aum - nnb (residual)
All rates: NaN when begin_aum missing or <= 0.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_rate(num: Any, den: Any) -> pd.Series | float:
    """num/den where den > 0, else NaN. Works for Series or scalar."""
    if isinstance(num, pd.Series) and isinstance(den, pd.Series):
        out = num / den
        out = out.where(den > 0)
        return out
    try:
        d = float(den) if den is not None else float("nan")
        n = float(num) if num is not None else float("nan")
        if d <= 0 or d != d:
            return float("nan")
        return n / d
    except (TypeError, ValueError):
        return float("nan")


def asset_growth_rate(begin_aum: Any, end_aum: Any) -> pd.Series | float:
    """(end_aum - begin_aum) / begin_aum. NaN when begin_aum missing or <= 0."""
    delta = end_aum - begin_aum if isinstance(begin_aum, pd.Series) else (float(end_aum or 0) - float(begin_aum or 0))
    return _safe_rate(delta, begin_aum)


def organic_growth_rate(nnb: Any, begin_aum: Any) -> pd.Series | float:
    """nnb / begin_aum. NaN when begin_aum missing or <= 0."""
    return _safe_rate(nnb, begin_aum)


def market_impact_residual(begin_aum: Any, end_aum: Any, nnb: Any) -> pd.Series | float:
    """end_aum - begin_aum - nnb (residual)."""
    if isinstance(end_aum, pd.Series):
        return end_aum - begin_aum - nnb
    return float(end_aum or 0) - float(begin_aum or 0) - float(nnb or 0)


def external_growth_rate(begin_aum: Any, end_aum: Any, nnb: Any) -> pd.Series | float:
    """market_impact / begin_aum with market_impact = end - begin - nnb. NaN when begin_aum missing or <= 0."""
    mi = market_impact_residual(begin_aum, end_aum, nnb)
    return _safe_rate(mi, begin_aum)


def compute_firm_rates_df(firm: pd.DataFrame) -> pd.DataFrame:
    """
    Add DATA SUMMARY rate columns to a firm-level DataFrame.
    Expects columns: begin_aum, end_aum, nnb. Adds market_impact (residual) and
    asset_growth_rate_calc, organic_growth_rate_calc, external_growth_rate_calc.
    """
    out = firm.copy()
    if "market_impact" not in out.columns:
        out["market_impact"] = market_impact_residual(
            out["begin_aum"], out["end_aum"], out["nnb"]
        )
    out["asset_growth_rate_calc"] = asset_growth_rate(out["begin_aum"], out["end_aum"])
    out["organic_growth_rate_calc"] = organic_growth_rate(out["nnb"], out["begin_aum"])
    out["external_growth_rate_calc"] = external_growth_rate(
        out["begin_aum"], out["end_aum"], out["nnb"]
    )
    return out
