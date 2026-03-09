"""
Utilities for analyzing unmapped channel key combinations after apply_channel_map.
No I/O; deterministic.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

KEY_COLS = ["channel_raw", "channel_standard", "channel_best"]
STATUS_COL = "channel_map_status"
STATUS_MAPPED = "MAPPED"


def _to_iso_date(series: pd.Series) -> pd.Series:
    """Convert datetime-like series to YYYY-MM-DD string, preserving NaT as None."""
    if series.empty:
        return series.astype("string")
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def extract_unmapped_channel_keys(
    df: pd.DataFrame,
    *,
    max_samples_per_key: int = 3,  # noqa: ARG001 - reserved for future sampling needs
) -> pd.DataFrame:
    """
    Aggregate unmapped channel key combinations with counts and representative samples.

    - Filters rows where channel_map_status != MAPPED.
    - Groups by KEY_COLS and computes:
        - row_count (count)
        - distinct_months (nunique month_end)
        - distinct_tickers (nunique product_ticker)
        - sample_month_end (min month_end as ISO YYYY-MM-DD)
        - sample_ticker (first non-null product_ticker)
        - sample_status (mode of channel_map_status; tie broken deterministically)
    - Returns dataframe sorted by row_count desc, then key cols asc.
    """
    required_cols = ["month_end", "product_ticker", STATUS_COL, *KEY_COLS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {missing}. Required: {required_cols}.")

    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                *KEY_COLS,
                "row_count",
                "distinct_months",
                "distinct_tickers",
                "sample_month_end",
                "sample_ticker",
                "sample_status",
            ]
        )

    df_unmapped = df[df[STATUS_COL] != STATUS_MAPPED].copy()
    if df_unmapped.empty:
        return pd.DataFrame(
            columns=[
                *KEY_COLS,
                "row_count",
                "distinct_months",
                "distinct_tickers",
                "sample_month_end",
                "sample_ticker",
                "sample_status",
            ]
        )

    grouped = df_unmapped.groupby(KEY_COLS, dropna=False, sort=False)

    # Base aggregates
    agg = grouped.agg(
        row_count=("channel_raw", "size"),
        distinct_months=("month_end", "nunique"),
        distinct_tickers=("product_ticker", "nunique"),
    ).reset_index()

    # Representative samples per key (one row per key)
    # sample_month_end: earliest month_end in group
    sample_dates = grouped["month_end"].min().reset_index(name="sample_month_end")
    # sample_ticker: first non-null product_ticker in group (deterministic due to original order)
    sample_tickers = (
        grouped["product_ticker"]
        .apply(lambda s: s.dropna().iloc[0] if not s.dropna().empty else None)
        .reset_index(name="sample_ticker")
    )

    # sample_status: mode of channel_map_status; if multiple, pick lexicographically smallest
    def _status_mode(s: pd.Series) -> Any:
        vc = s.value_counts(dropna=False)
        if vc.empty:
            return None
        max_count = vc.max()
        candidates = sorted([val for val, cnt in vc.items() if cnt == max_count])
        return candidates[0]

    sample_status = grouped[STATUS_COL].apply(_status_mode).reset_index(name="sample_status")

    out = agg.merge(sample_dates, on=KEY_COLS, how="left").merge(
        sample_tickers, on=KEY_COLS, how="left"
    ).merge(sample_status, on=KEY_COLS, how="left")

    # Format dates as ISO strings (YYYY-MM-DD)
    out["sample_month_end"] = _to_iso_date(out["sample_month_end"])

    # Sort deterministically
    out = out.sort_values(
        by=["row_count", *KEY_COLS],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out[
        [
            *KEY_COLS,
            "row_count",
            "distinct_months",
            "distinct_tickers",
            "sample_month_end",
            "sample_ticker",
            "sample_status",
        ]
    ]


def unmapped_summary(df_unmapped: pd.DataFrame) -> dict[str, Any]:
    """
    Summarize unmapped keys dataframe:
      - total_unmapped_keys: distinct key combinations (rows)
      - total_unmapped_rows: sum of row_count
    """
    if df_unmapped.empty:
        return {
            "total_unmapped_keys": 0,
            "total_unmapped_rows": 0,
        }

    if "row_count" not in df_unmapped.columns:
        raise ValueError("df_unmapped must include 'row_count' column.")

    total_unmapped_keys = int(len(df_unmapped))
    total_unmapped_rows = int(df_unmapped["row_count"].sum())
    return {
        "total_unmapped_keys": total_unmapped_keys,
        "total_unmapped_rows": total_unmapped_rows,
    }

