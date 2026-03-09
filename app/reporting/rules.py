"""
Deterministic rule helpers for report sections.
Top/bottom selection, mix-shift detection, negative market month, latest period.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# --- Constants (single place for thresholds and bullet limits) -----------------

TOP_N = 5
BOTTOM_N = 5
MIX_SHIFT_THRESHOLD = 0.01  # 1% share delta
MIN_BULLETS = 2
MAX_BULLETS = 5


def _require_columns(df: pd.DataFrame, required: list[str], caller: str) -> None:
    """Raise ValueError if any required column is missing; message lists missing cols and caller."""
    if df is None:
        raise ValueError(f"{caller}: input DataFrame is None")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{caller}: missing columns: {missing}")


def select_top_bottom(
    df: pd.DataFrame,
    metric_col: str,
    dim_col: str,
    top_n: int = TOP_N,
    bottom_n: int = BOTTOM_N,
    caller: str = "select_top_bottom",
) -> pd.DataFrame:
    """
    Deterministic top/bottom by metric. Drops NaN metric rows; tie-break by dim_col asc.
    Returns combined table with "bucket" ('top'/'bottom') and "rank" (1-based within bucket).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    _require_columns(df, [metric_col, dim_col], caller)

    clean = df.dropna(subset=[metric_col])
    if clean.empty:
        return pd.DataFrame()

    # Top: sort metric desc, then dim_col asc for ties
    top_df = (
        clean.sort_values(by=[metric_col, dim_col], ascending=[False, True])
        .head(top_n)
        .copy()
    )
    top_df["bucket"] = "top"
    top_df["rank"] = range(1, len(top_df) + 1)

    # Bottom: sort metric asc, then dim_col asc for ties
    bottom_df = (
        clean.sort_values(by=[metric_col, dim_col], ascending=[True, True])
        .head(bottom_n)
        .copy()
    )
    bottom_df["bucket"] = "bottom"
    bottom_df["rank"] = range(1, len(bottom_df) + 1)

    return pd.concat([top_df, bottom_df], ignore_index=True)


def detect_mix_shift(
    rank_df: pd.DataFrame,
    share_delta_col: str,
    threshold: float = MIX_SHIFT_THRESHOLD,
    dim_col: str = "name",
    caller: str = "detect_mix_shift",
) -> pd.DataFrame:
    """
    Rows where abs(share_delta) >= threshold. Sorted by abs(share_delta) desc, tie by dim_col asc.
    """
    if rank_df is None or rank_df.empty:
        return pd.DataFrame()
    _require_columns(rank_df, [share_delta_col, dim_col], caller)

    clean = rank_df.dropna(subset=[share_delta_col])
    if clean.empty:
        return pd.DataFrame()

    shifted = clean[clean[share_delta_col].abs() >= threshold].copy()
    if shifted.empty:
        return pd.DataFrame()

    shifted["_abs_delta"] = shifted[share_delta_col].abs()
    shifted = shifted.sort_values(by=["_abs_delta", dim_col], ascending=[False, True])
    shifted = shifted.drop(columns=["_abs_delta"])
    return shifted.reset_index(drop=True)


def get_latest_month(ts_df: pd.DataFrame, caller: str = "get_latest_month") -> pd.Timestamp | None:
    """Latest month_end in ts_df; None if empty or column missing."""
    if ts_df is None or ts_df.empty:
        return None
    _require_columns(ts_df, ["month_end"], caller)
    sorted_df = ts_df.sort_values("month_end", ascending=False)
    val = sorted_df.iloc[0]["month_end"]
    if pd.isna(val):
        return None
    return pd.Timestamp(val) if not isinstance(val, pd.Timestamp) else val


def detect_negative_market_month(
    ts_df: pd.DataFrame,
    caller: str = "detect_negative_market_month",
) -> bool:
    """
    True if latest month has market_impact_rate < 0, or market_impact_abs < 0 if rate missing.
    """
    if ts_df is None or ts_df.empty:
        return False
    _require_columns(ts_df, ["month_end"], caller)
    sorted_df = ts_df.sort_values("month_end", ascending=False)
    row = sorted_df.iloc[0]

    if "market_impact_rate" in row.index:
        rate = row["market_impact_rate"]
        if pd.notna(rate) and float(rate) < 0:
            return True
    if "market_impact_abs" in row.index:
        abs_val = row["market_impact_abs"]
        if pd.notna(abs_val) and float(abs_val) < 0:
            return True
    return False
