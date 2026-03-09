"""
Channel enrichment join using DATA_MAPPING as authority.
Deterministic: left join on (channel_raw, channel_standard, channel_best).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

KEY_COLS = ["channel_raw", "channel_standard", "channel_best"]
ENRICH_COLS = ["channel_l1", "channel_l2", "preferred_label"]


def enrich_channels(
    df_fact: pd.DataFrame,
    df_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Enrich fact_monthly with channel hierarchy/label columns from DATA_MAPPING.

    Rules:
      1) Validate df_mapping uniqueness already enforced (assert no duplicates on KEY_COLS).
      2) Left join df_fact on KEY_COLS.
      3) Report:
         - rows_in_fact
         - rows_mapped (any enrichment column non-null)
         - rows_unmapped
         - top_unmapped_keys_sample (first 20 distinct key tuples)
      4) Do NOT silently fill: leave enrichment fields null if unmapped.
    """
    if not set(KEY_COLS).issubset(df_mapping.columns):
        raise ValueError(
            f"df_mapping must contain key columns {KEY_COLS}, got {list(df_mapping.columns)}"
        )

    # 1) Validate uniqueness on KEY_COLS
    dup = df_mapping.duplicated(subset=KEY_COLS, keep=False)
    if dup.any():
        dup_count = int(dup.sum())
        dup_df = df_mapping.loc[dup, KEY_COLS]
        sample_keys_df = dup_df.drop_duplicates().head(10)
        sample_keys = [tuple(row) for _, row in sample_keys_df.iterrows()]
        raise ValueError(
            "df_mapping has duplicate keys for channel enrichment: "
            f"{dup_count} row(s) duplicate on {KEY_COLS}. Sample keys: {sample_keys}"
        )

    # 2) Left join on KEY_COLS
    enrich_present = [c for c in ENRICH_COLS if c in df_mapping.columns]
    if not enrich_present:
        # Nothing to enrich; return original with empty report.
        report = {
            "rows_in_fact": len(df_fact),
            "rows_mapped": 0,
            "rows_unmapped": len(df_fact),
            "top_unmapped_keys_sample": [],
        }
        return df_fact.copy(), report

    right = df_mapping[KEY_COLS + enrich_present].copy()

    # Ensure we don't accidentally overwrite existing enrichment columns in fact.
    overlap = [c for c in enrich_present if c in df_fact.columns]
    if overlap:
        raise ValueError(
            f"df_fact already has enrichment columns {overlap}; enrich_channels refuses to overwrite."
        )

    merged = df_fact.merge(
        right,
        on=KEY_COLS,
        how="left",
        sort=False,
        suffixes=("", "_enrich"),
    )

    # 3) Mapping stats
    rows_in_fact = len(df_fact)
    if enrich_present:
        enrich_block = merged[enrich_present]
        mapped_mask = enrich_block.notna().any(axis=1)
        rows_mapped = int(mapped_mask.sum())
    else:
        mapped_mask = pd.Series(False, index=merged.index)
        rows_mapped = 0
    rows_unmapped = rows_in_fact - rows_mapped

    unmapped_keys_sample: list[tuple[Any, ...]] = []
    if rows_unmapped > 0:
        # Keys for unmapped rows (where any enrich col is null across all enrich cols)
        unmapped = merged.loc[~mapped_mask, KEY_COLS]
        sample_df = unmapped.drop_duplicates().head(20)
        unmapped_keys_sample = [tuple(row) for _, row in sample_df.iterrows()]

    report: dict[str, Any] = {
        "rows_in_fact": rows_in_fact,
        "rows_mapped": rows_mapped,
        "rows_unmapped": rows_unmapped,
        "top_unmapped_keys_sample": unmapped_keys_sample,
    }

    return merged, report

