"""
Deterministic channel mapping application to fact_monthly: left join + fallback rules + status.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

KEY_COLS = ["channel_raw", "channel_standard", "channel_best"]
OUT_COLS = ["channel_l1", "channel_l2", "preferred_label"]
STATUS_COL = "channel_map_status"

STATUS_MAPPED = "MAPPED"
STATUS_FALLBACK_BEST = "FALLBACK_BEST"
STATUS_FALLBACK_STANDARD = "FALLBACK_STANDARD"
STATUS_FALLBACK_RAW = "FALLBACK_RAW"

UNMAPPED_PLACEHOLDER = "UNMAPPED"


def _required_columns_fact() -> list[str]:
    return list(KEY_COLS)


def _required_columns_map() -> list[str]:
    return list(KEY_COLS) + list(OUT_COLS)


def _str_trim(s: Any) -> str:
    """Trim and collapse internal whitespace; empty/NA -> empty string."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    x = str(s).strip()
    return re.sub(r"\s+", " ", x)


def _non_empty(s: str) -> bool:
    return s != ""


def apply_channel_map(
    df_fact: pd.DataFrame,
    df_channel_map: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply channel_map to fact_monthly: left join on KEY_COLS, then fallback rules for unmapped rows.

    Outputs added/overwritten: channel_l1, channel_l2, preferred_label, channel_map_status.
    All output strings trimmed and collapsed; StringDtype.
    Preserves row order. No rows dropped.
    """
    missing_fact = [c for c in _required_columns_fact() if c not in df_fact.columns]
    if missing_fact:
        raise ValueError(f"df_fact missing required columns: {missing_fact}. Required: {_required_columns_fact()}.")

    missing_map = [c for c in _required_columns_map() if c not in df_channel_map.columns]
    if missing_map:
        raise ValueError(f"df_channel_map missing required columns: {missing_map}. Required: {_required_columns_map()}.")

    rows_in = len(df_fact)
    if rows_in == 0:
        out = df_fact.copy()
        for c in OUT_COLS + [STATUS_COL]:
            out[c] = pd.Series(dtype="string")
        return out, {
            "rows_in": 0,
            "mapped_count": 0,
            "fallback_best_count": 0,
            "fallback_standard_count": 0,
            "fallback_raw_count": 0,
            "unmapped_keys_sample": [],
        }

    # Left join: bring map columns with suffix _map to avoid overwriting key cols
    right = df_channel_map[KEY_COLS + OUT_COLS].copy()
    right = right.rename(columns={c: f"{c}_map" for c in OUT_COLS})
    merged = df_fact.merge(right, on=KEY_COLS, how="left", sort=False)

    # Trim key cols for consistent empty check
    raw = merged["channel_raw"].apply(_str_trim)
    std = merged["channel_standard"].apply(_str_trim)
    best = merged["channel_best"].apply(_str_trim)

    # Mapped: preferred_label from map is non-null and non-empty
    pl_map = merged["preferred_label_map"].apply(_str_trim)
    is_mapped = pl_map.apply(_non_empty)

    # Fallbacks for unmapped
    fallback_best = ~is_mapped & best.apply(_non_empty)
    fallback_standard = ~is_mapped & ~fallback_best & std.apply(_non_empty)
    fallback_raw = ~is_mapped & ~fallback_best & ~fallback_standard

    # Build output columns (out has same index as df_fact)
    out = df_fact.copy()
    out[STATUS_COL] = STATUS_FALLBACK_RAW  # default
    out.loc[is_mapped, STATUS_COL] = STATUS_MAPPED
    out.loc[fallback_best, STATUS_COL] = STATUS_FALLBACK_BEST
    out.loc[fallback_standard, STATUS_COL] = STATUS_FALLBACK_STANDARD
    out.loc[fallback_raw, STATUS_COL] = STATUS_FALLBACK_RAW

    # preferred_label: from map when MAPPED, else fallback best -> standard -> raw
    out["preferred_label"] = pl_map.where(is_mapped)
    out.loc[fallback_best, "preferred_label"] = merged.loc[fallback_best, "channel_best"].apply(_str_trim).values
    out.loc[fallback_standard, "preferred_label"] = merged.loc[fallback_standard, "channel_standard"].apply(_str_trim).values
    out.loc[fallback_raw, "preferred_label"] = merged.loc[fallback_raw, "channel_raw"].apply(_str_trim).values

    # channel_l1, channel_l2: from map when MAPPED, else "UNMAPPED" (Rule A)
    out["channel_l1"] = merged["channel_l1_map"].where(is_mapped).fillna(UNMAPPED_PLACEHOLDER)
    out["channel_l2"] = merged["channel_l2_map"].where(is_mapped).fillna(UNMAPPED_PLACEHOLDER)

    # Ensure StringDtype and trim (collapse whitespace)
    for c in OUT_COLS + [STATUS_COL]:
        out[c] = out[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        out[c] = out[c].astype("string")

    # Stats
    mapped_count = int(is_mapped.sum())
    fallback_best_count = int(fallback_best.sum())
    fallback_standard_count = int(fallback_standard.sum())
    fallback_raw_count = int(fallback_raw.sum())

    unmapped_mask = ~is_mapped
    unmapped_keys_sample: list[tuple[Any, ...]] = []
    if unmapped_mask.any():
        unmapped_df = out.loc[unmapped_mask, KEY_COLS].drop_duplicates().head(20)
        unmapped_keys_sample = [tuple(row) for _, row in unmapped_df.iterrows()]

    stats: dict[str, Any] = {
        "rows_in": rows_in,
        "mapped_count": mapped_count,
        "fallback_best_count": fallback_best_count,
        "fallback_standard_count": fallback_standard_count,
        "fallback_raw_count": fallback_raw_count,
        "unmapped_keys_sample": unmapped_keys_sample,
    }
    return out, stats
