"""
Data helpers for dashboard. KPIs and visuals use curated/metrics_monthly.parquet
filtered by (path_id, slice_id) from the drill path contract. No groupby in UI; only filtering + plotting.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS_MONTHLY_REL = "curated/metrics_monthly.parquet"
COL_MONTH_END = "month_end"
COL_END_AUM = "end_aum"
COL_NNB = "nnb"
COL_NNF = "nnf"
COL_BEGIN_AUM = "begin_aum"
COL_AUM_GROWTH_RATE = "aum_growth_rate"


def load_metrics_monthly(root: Path) -> pd.DataFrame:
    """Load curated/metrics_monthly.parquet. Empty DataFrame if missing."""
    path = root / METRICS_MONTHLY_REL
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_metrics_for_slice(metrics_df: pd.DataFrame, path_id: str, slice_id: str) -> pd.DataFrame:
    """
    Filter metrics to the given (path_id, slice_id). Returns view sorted by month_end.
    No groupby; filtering only.
    """
    if metrics_df.empty:
        return metrics_df
    out = metrics_df[(metrics_df["path_id"] == path_id) & (metrics_df["slice_id"] == slice_id)].copy()
    out = out.sort_values(COL_MONTH_END, kind="mergesort").reset_index(drop=True)
    return out


def get_kpi_from_metrics(slice_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build KPI table from already-filtered metrics (one path_id + slice_id).
    No groupby; only aggregation on single-slice df: total_aum (last end_aum), month_count, sum nnb/nnf.
    """
    if slice_metrics_df.empty:
        return pd.DataFrame(columns=["metric", "value"])
    rows = []
    if COL_END_AUM in slice_metrics_df.columns:
        last_aum = slice_metrics_df[COL_END_AUM].iloc[-1] if len(slice_metrics_df) else None
        rows.append({"metric": "total_aum", "value": last_aum})
    rows.append({"metric": "month_count", "value": len(slice_metrics_df)})
    if COL_NNB in slice_metrics_df.columns:
        rows.append({"metric": "nnb_total", "value": slice_metrics_df[COL_NNB].sum()})
    if COL_NNF in slice_metrics_df.columns:
        rows.append({"metric": "nnf_total", "value": slice_metrics_df[COL_NNF].sum()})
    return pd.DataFrame(rows)
