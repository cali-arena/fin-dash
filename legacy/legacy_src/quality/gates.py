"""
Quality gates before persisting DATA RAW. No I/O; explicit errors and stats.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def run_quality_gates(
    df: pd.DataFrame,
    *,
    key_fields: list[str],
    numeric_fields: list[str],
    min_rows: int = 1,
    max_nan_ratio_by_col: float = 0.01,
    max_total_nan_ratio: float = 0.001,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Run gates: row count, no missing key fields, numeric NaN ratios.
    Returns (ok, errors, stats). Errors list exact columns failing thresholds.
    """
    errors: list[str] = []
    stats: dict[str, Any] = {"row_count": len(df)}

    # 1) Row count >= min_rows
    if len(df) < min_rows:
        errors.append(f"row_count={len(df)} below min_rows={min_rows}")

    # 2) No missing in key_fields (null or empty string after strip)
    key_fields_present = [c for c in key_fields if c in df.columns]
    missing_key_cols: list[str] = []
    for col in key_fields_present:
        ser = df[col]
        stripped = ser.astype(str).str.strip()
        null_or_empty = pd.isna(ser) | stripped.eq("") | stripped.str.lower().eq("nan")
        if null_or_empty.any():
            missing_key_cols.append(col)
    if missing_key_cols:
        errors.append(f"key_fields have null or empty values: {missing_key_cols}")

    # 3) Numeric fields: per-column nan_ratio and total nan_ratio
    numeric_present = [c for c in numeric_fields if c in df.columns]
    nan_ratio_by_col: dict[str, float] = {}
    total_numeric_cells = 0
    total_nan_cells = 0

    for col in numeric_present:
        ser = df[col]
        n = len(ser)
        nan_count = int(pd.isna(ser).sum())
        total_numeric_cells += n
        total_nan_cells += nan_count
        ratio = nan_count / n if n else 0.0
        nan_ratio_by_col[col] = ratio
        if ratio > max_nan_ratio_by_col:
            errors.append(
                f"numeric col {col!r} nan_ratio={ratio:.6f} > max_nan_ratio_by_col={max_nan_ratio_by_col}"
            )

    if total_numeric_cells > 0:
        total_nan_ratio = total_nan_cells / total_numeric_cells
        stats["total_nan_ratio"] = total_nan_ratio
        stats["total_numeric_cells"] = total_numeric_cells
        stats["total_nan_cells"] = total_nan_cells
        if total_nan_ratio > max_total_nan_ratio:
            errors.append(
                f"total_nan_ratio={total_nan_ratio:.6f} > max_total_nan_ratio={max_total_nan_ratio}"
            )
    stats["nan_ratio_by_col"] = nan_ratio_by_col

    ok = len(errors) == 0
    return ok, errors, stats


def summarize_rejects(df_rejects: pd.DataFrame, reason_col: str = "_reject_reason") -> dict[str, Any]:
    """
    Summarize reject DataFrame: total count and counts by reason.
    """
    if df_rejects.empty:
        return {"total": 0, "by_reason": {}}
    total = len(df_rejects)
    by_reason = df_rejects[reason_col].value_counts().to_dict() if reason_col in df_rejects.columns else {}
    return {"total": total, "by_reason": {str(k): int(v) for k, v in by_reason.items()}}
