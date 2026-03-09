"""
Deterministic channel_map artifact from DATA MAPPING. Strict QA; no I/O.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

KEY_COLS = ["channel_raw", "channel_standard", "channel_best"]
OUT_COLS = ["channel_l1", "channel_l2", "preferred_label"]

# Max sample of bad keys to include in error messages.
MAX_BAD_KEYS_IN_ERROR = 10


def _normalize_cell(value: Any) -> str:
    """Strip and collapse internal whitespace; do not change case. Empty/NaN/pd.NA -> empty string for detection."""
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _is_null_or_empty(s: str) -> bool:
    return s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip() == ""


def validate_mapping_inputs(df_mapping: pd.DataFrame) -> tuple[bool, list[str], dict[str, Any]]:
    """
    QA rules on canonicalized df_mapping:
    - No null/empty in KEY_COLS.
    - Uniqueness on KEY_COLS (no duplicates).
    - Column preferred_label must exist; no null/empty preferred_label (full coverage).
    Returns (ok, errors, stats).
    """
    errors: list[str] = []
    stats: dict[str, Any] = {
        "rowcount": len(df_mapping),
        "null_key_counts": {},
        "preferred_label_null_count": 0,
        "duplicate_key_count": 0,
    }

    if df_mapping.empty:
        errors.append("df_mapping is empty.")
        return False, errors, stats

    # Required columns
    missing_key = [c for c in KEY_COLS if c not in df_mapping.columns]
    if missing_key:
        errors.append(f"Missing key columns: {missing_key}.")
        return False, errors, stats
    if "preferred_label" not in df_mapping.columns:
        errors.append("Column 'preferred_label' must exist.")
        return False, errors, stats

    # Null keys
    for col in KEY_COLS:
        ser = df_mapping[col]
        null_or_empty = ser.apply(lambda v: _is_null_or_empty(v) if not pd.isna(v) else True)
        null_or_empty = null_or_empty | pd.isna(ser)
        n = int(null_or_empty.sum())
        stats["null_key_counts"][col] = n
        if n > 0:
            bad_keys_sample = df_mapping.loc[null_or_empty, KEY_COLS].head(MAX_BAD_KEYS_IN_ERROR)
            sample_tuples = [tuple(row) for _, row in bad_keys_sample.iterrows()]
            errors.append(f"Key column {col!r} has {n} null/empty value(s). Sample keys (first {MAX_BAD_KEYS_IN_ERROR}): {sample_tuples}.")

    # Preferred label coverage
    pl = df_mapping["preferred_label"]
    pl_empty = pl.apply(lambda v: _is_null_or_empty(v) if not pd.isna(v) else True)
    pl_empty = pl_empty | pd.isna(pl)
    stats["preferred_label_null_count"] = int(pl_empty.sum())
    if stats["preferred_label_null_count"] > 0:
        bad = df_mapping.loc[pl_empty, KEY_COLS].head(MAX_BAD_KEYS_IN_ERROR)
        sample_tuples = [tuple(row) for _, row in bad.iterrows()]
        errors.append(
            f"preferred_label has {stats['preferred_label_null_count']} null/empty row(s). "
            f"Sample keys (first {MAX_BAD_KEYS_IN_ERROR}): {sample_tuples}."
        )

    # Uniqueness on KEY_COLS
    dup = df_mapping.duplicated(subset=KEY_COLS, keep=False)
    stats["duplicate_key_count"] = int(dup.sum())
    if stats["duplicate_key_count"] > 0:
        dup_keys_df = df_mapping.loc[dup, KEY_COLS].drop_duplicates().head(MAX_BAD_KEYS_IN_ERROR)
        sample_tuples = [tuple(row) for _, row in dup_keys_df.iterrows()]
        errors.append(
            f"Duplicate keys on {KEY_COLS}: {stats['duplicate_key_count']} row(s) involved. "
            f"Sample keys (first {MAX_BAD_KEYS_IN_ERROR}): {sample_tuples}."
        )

    ok = len(errors) == 0
    return ok, errors, stats


def build_channel_map(df_mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build deterministic channel_map from canonicalized df_mapping.
    Select KEY_COLS + OUT_COLS, normalize strings (strip/collapse whitespace, no case change),
    sort by KEY_COLS, validate (hard fail if errors), return df_channel_map + report.
    """
    needed = [c for c in KEY_COLS + OUT_COLS if c in df_mapping.columns]
    missing = [c for c in KEY_COLS + OUT_COLS if c not in df_mapping.columns]
    if missing:
        raise ValueError(f"df_mapping missing columns for channel_map: {missing}. Required: {KEY_COLS + OUT_COLS}.")

    df = df_mapping[KEY_COLS + OUT_COLS].copy()
    for col in df.columns:
        df[col] = df[col].apply(_normalize_cell)

    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    ok, errors, stats = validate_mapping_inputs(df)
    if not ok:
        raise ValueError("Channel map validation failed: " + "; ".join(errors)) from None

    report: dict[str, Any] = {
        "rowcount": stats["rowcount"],
        "null_key_counts": stats["null_key_counts"],
        "preferred_label_null_count": stats["preferred_label_null_count"],
        "duplicate_key_count": stats["duplicate_key_count"],
    }
    return df, report


def channel_map_dict(df_channel_map: pd.DataFrame) -> dict[tuple[Any, ...], dict[str, Any]]:
    """
    Return dict keyed by tuple(key cols) -> dict of output column values.
    Deterministic: iterate rows in sorted order (assumes df is already sorted by KEY_COLS).
    """
    if df_channel_map.empty:
        return {}
    for c in KEY_COLS + OUT_COLS:
        if c not in df_channel_map.columns:
            raise ValueError(f"df_channel_map missing column {c!r} for channel_map_dict.")
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _, row in df_channel_map.sort_values(KEY_COLS).iterrows():
        key = tuple(row[c] for c in KEY_COLS)
        out[key] = {c: row[c] for c in OUT_COLS if c in row.index}
    return out
