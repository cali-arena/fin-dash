"""
Filter helper for agg tables: date range on month_end + equality/IN on dimension columns.
No groupby; validates columns exist and applies masks in deterministic order.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.agg_store import DEFAULT_TIME_COL, MEASURE_COLS


def dimension_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are not time and not measure (candidates for dimension filters)."""
    dims: list[str] = []
    for c in df.columns:
        if c == DEFAULT_TIME_COL or c in MEASURE_COLS:
            continue
        dims.append(c)
    return sorted(dims)


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to df. Validates filter columns exist; applies masks in deterministic order.
    Single combined mask then one .loc to avoid full-table copies in a loop (sub-100ms on large agg).
    - filters["month_end_range"]: (min_ts, max_ts) or None → boolean mask on month_end
    - filters[col]: list of values → keep rows where df[col].isin(values); empty list = no filter
    Returns filtered DataFrame (one copy). No groupby.
    """
    mask = None
    # month_end first
    month_range = filters.get("month_end_range")
    if month_range is not None:
        min_ts, max_ts = month_range
        if DEFAULT_TIME_COL not in df.columns:
            raise ValueError(f"Filter month_end_range requires column {DEFAULT_TIME_COL!r}; columns: {list(df.columns)}")
        mask = df[DEFAULT_TIME_COL].notna()
        if min_ts is not None:
            mask = mask & (df[DEFAULT_TIME_COL] >= pd.Timestamp(min_ts))
        if max_ts is not None:
            mask = mask & (df[DEFAULT_TIME_COL] <= pd.Timestamp(max_ts))

    for key in sorted(filters.keys()):
        if key == "month_end_range":
            continue
        values = filters[key]
        if values is None or (isinstance(values, (list, tuple)) and len(values) == 0):
            continue
        if key not in df.columns:
            raise ValueError(f"Filter column {key!r} not in DataFrame; columns: {list(df.columns)}")
        if not isinstance(values, (list, tuple)):
            values = [values]
        col_mask = df[key].isin(values)
        mask = col_mask if mask is None else (mask & col_mask)

    if mask is None:
        return df.copy()
    return df.loc[mask].reset_index(drop=True)
