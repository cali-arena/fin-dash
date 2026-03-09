"""
Channel mapping coverage checks and reporting. No I/O; deterministic.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

KEY_COLS = ["channel_raw", "channel_standard", "channel_best"]
STATUS_COL = "channel_map_status"
STATUS_MAPPED = "MAPPED"


def compute_channel_map_coverage(
    df_fact: pd.DataFrame,
    df_channel_map: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute coverage of fact keys by channel_map (distinct keys only).
    Returns fact_distinct_keys, map_distinct_keys, matched_distinct_keys,
    unmatched_distinct_keys, unmatched_keys_sample (up to 50, deterministic order).
    """
    missing_fact = [c for c in KEY_COLS if c not in df_fact.columns]
    if missing_fact:
        raise ValueError(f"df_fact missing key columns: {missing_fact}. Required: {KEY_COLS}.")
    missing_map = [c for c in KEY_COLS if c not in df_channel_map.columns]
    if missing_map:
        raise ValueError(f"df_channel_map missing key columns: {missing_map}. Required: {KEY_COLS}.")

    fact_keys = df_fact[KEY_COLS].drop_duplicates()
    map_keys = df_channel_map[KEY_COLS].drop_duplicates()

    fact_distinct_keys = len(fact_keys)
    map_distinct_keys = len(map_keys)

    # Matched: fact key appears in map (merge inner on KEY_COLS and count distinct fact keys)
    merged = fact_keys.merge(map_keys, on=KEY_COLS, how="inner")
    matched_distinct_keys = len(merged)

    # Unmatched: fact keys not in map
    unmatched_keys_df = fact_keys.merge(
        map_keys, on=KEY_COLS, how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns=["_merge"])[KEY_COLS]
    unmatched_distinct_keys = len(unmatched_keys_df)

    # Sample: sort for determinism, take first 50, convert to list of dicts
    unmatched_keys_sample: list[dict[str, Any]] = []
    if unmatched_distinct_keys > 0:
        sample_df = unmatched_keys_df.sort_values(KEY_COLS).head(50)
        unmatched_keys_sample = sample_df.to_dict("records")

    return {
        "fact_distinct_keys": fact_distinct_keys,
        "map_distinct_keys": map_distinct_keys,
        "matched_distinct_keys": matched_distinct_keys,
        "unmatched_distinct_keys": unmatched_distinct_keys,
        "unmatched_keys_sample": unmatched_keys_sample,
    }


def gate_minimum_mapping_rate(
    df_out: pd.DataFrame,
    min_mapped_ratio: float = 0.90,
) -> tuple[bool, list[str]]:
    """
    Gate on share of rows with channel_map_status == "MAPPED".
    If ratio < min_mapped_ratio, return (False, [error message]); else (True, []).
    """
    errors: list[str] = []
    if STATUS_COL not in df_out.columns:
        errors.append(f"df_out missing column {STATUS_COL!r}; cannot compute mapping rate.")
        return False, errors

    n = len(df_out)
    if n == 0:
        return True, []

    mapped_count = (df_out[STATUS_COL] == STATUS_MAPPED).sum()
    ratio = float(mapped_count) / n
    if ratio < min_mapped_ratio:
        errors.append(
            f"Channel mapping rate {ratio:.4f} ({mapped_count}/{n} rows MAPPED) "
            f"is below minimum {min_mapped_ratio:.2f}."
        )
        return False, errors
    return True, errors
