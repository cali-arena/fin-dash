"""
Optional profiling to decide whether to cast id columns to category.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def profile_cardinality(
    df: pd.DataFrame,
    cols: list[str],
    *,
    max_unique_for_category: int = 200,
) -> dict[str, dict[str, Any]]:
    """
    Profile cardinality for the given columns. Returns a dict per column with
    n_rows, n_unique, unique_ratio, and suggest_category (True when n_unique <= max_unique_for_category).
    """
    n_rows = len(df)
    result: dict[str, dict[str, Any]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        ser = df[col].dropna()
        n_valid = len(ser)
        n_unique = int(ser.nunique())
        unique_ratio = n_unique / n_valid if n_valid else 0.0
        suggest_category = n_unique <= max_unique_for_category and n_unique >= 0
        result[col] = {
            "n_rows": n_rows,
            "n_unique": n_unique,
            "unique_ratio": round(unique_ratio, 6),
            "suggest_category": suggest_category,
        }
    return result


def apply_optional_categoricals(df: pd.DataFrame, profile: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    For columns where profile[col]["suggest_category"] is True, cast to category.
    Others are left as-is (e.g. string).
    """
    out = df.copy()
    for col, info in profile.items():
        if col not in out.columns:
            continue
        if info.get("suggest_category"):
            out[col] = out[col].astype("category")
    return out
