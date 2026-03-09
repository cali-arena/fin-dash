"""
Canonical date alignment helpers for month_end sequences.
Uses only month_end values present in the data (gap-aware) to avoid DATE_ALIGNMENT mismatches vs DATA SUMMARY.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _sorted_month_ends(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted unique month_end values from df; empty list if no valid column or empty."""
    if df is None or df.empty or "month_end" not in df.columns:
        return []
    s = pd.to_datetime(df["month_end"], errors="coerce")
    s = s.dropna()
    if s.empty:
        return []
    return sorted(s.unique().tolist())


def get_latest_month_end(df: pd.DataFrame) -> pd.Timestamp | None:
    """
    Latest (max) month_end present in the dataframe.
    Returns None if df is empty, has no 'month_end' column, or no valid dates.
    """
    months = _sorted_month_ends(df)
    return months[-1] if months else None


def get_prior_month_end(df: pd.DataFrame, current_month_end: Any) -> pd.Timestamp | None:
    """
    Prior available month_end: largest month_end in the data strictly before current_month_end.
    Gap-aware: does not assume calendar previous month exists in data.
    Returns None if there is no earlier month in the data.
    """
    months = _sorted_month_ends(df)
    if not months:
        return None
    current_ts = pd.Timestamp(current_month_end)
    prior = [m for m in months if m < current_ts]
    return prior[-1] if prior else None


def get_year_start_month_end(df: pd.DataFrame, current_month_end: Any) -> pd.Timestamp | None:
    """
    First month_end in the same calendar year as current_month_end that appears in the data.
    Gap-aware: e.g. if data has only Jan, Mar, Jun for that year, returns Jan.
    Returns None if no month in that year exists in the data.
    """
    months = _sorted_month_ends(df)
    if not months:
        return None
    current_ts = pd.Timestamp(current_month_end)
    year = current_ts.year
    in_year = [m for m in months if m.year == year]
    return in_year[0] if in_year else None


def is_single_month(df: pd.DataFrame) -> bool:
    """True if df has exactly one distinct month_end (single-month dataset)."""
    months = _sorted_month_ends(df)
    return len(months) == 1


def has_month_gaps(df: pd.DataFrame) -> bool:
    """
    True if month_end series has gaps (non-consecutive months).
    E.g. Jan, Mar, May -> True. Jan, Feb, Mar -> False.
    """
    months = _sorted_month_ends(df)
    if len(months) < 2:
        return False
    for i in range(1, len(months)):
        m1, m2 = months[i - 1], months[i]
        # Consecutive months: (year2*12 + month2) - (year1*12 + month1) == 1
        diff = (m2.year * 12 + m2.month) - (m1.year * 12 + m1.month)
        if diff != 1:
            return True
    return False
