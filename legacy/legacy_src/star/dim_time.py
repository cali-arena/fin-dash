"""
Build dim_time from fact_monthly. SCD Type 1; deterministic.
"""
from __future__ import annotations

import pandas as pd

COL = "month_end"


def build_dim_time(fact_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per distinct month_end with year, quarter, month, month_name, is_q_end, is_y_end.
    Natural key: month_end. Sorted by month_end.
    """
    if COL not in fact_monthly.columns:
        raise ValueError(f"fact_monthly missing column {COL!r}.")
    df = fact_monthly[[COL]].drop_duplicates()
    if df.empty:
        return _empty_dim_time()
    dt = pd.to_datetime(df[COL])
    out = pd.DataFrame({
        "month_end": dt,
        "year": dt.dt.year.astype("int64"),
        "quarter": dt.dt.quarter.astype("int64"),
        "month": dt.dt.month.astype("int64"),
        "month_name": dt.dt.strftime("%B").astype("string"),
        "is_q_end": dt.dt.month.isin([3, 6, 9, 12]).values,
        "is_y_end": (dt.dt.month == 12).values,
    })
    return out.sort_values("month_end", kind="mergesort").reset_index(drop=True)


def _empty_dim_time() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "month_end", "year", "quarter", "month", "month_name", "is_q_end", "is_y_end"
    ])
