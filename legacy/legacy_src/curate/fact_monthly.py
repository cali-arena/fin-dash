"""
Build curated fact_monthly at single monthly grain with enforced aggregation and channel derivation.
Output sorted by grain; stats for logging/audit.
Optional channel mapping: pass mapping_df directly or mapping_path to load from DATA_MAPPING.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.curate.apply_channel_map import OUT_COLS as APPLY_OUT_COLS, apply_channel_map
from legacy.legacy_src.mapping.channel_enrichment import ENRICH_COLS, KEY_COLS, enrich_channels

# Target grain: one row per (month_end, product_ticker, channel_best, src_country, product_country, segment, sub_segment).
GRAIN = [
    "month_end",
    "product_ticker",
    "channel_best",
    "src_country",
    "product_country",
    "segment",
    "sub_segment",
]

# Column names used in raw/curated (canonical).
COL_MONTH_END = "month_end"
COL_CHANNEL_RAW = "channel_raw"
COL_CHANNEL_STANDARD = "channel_standard"
COL_CHANNEL_BEST = "channel_best"
COL_NNB = "net_new_business"
COL_NNF = "net_new_base_fees"
COL_AUM = "asset_under_management"
COL_DATE_RAW = "date_raw"


def derive_channels(
    df: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Derive channel_standard and channel_best when missing.

    - channel_raw: from df["channel_raw"] (already canonical).
    - channel_standard: if non-null in df, keep; else map from mapping_df (channel_raw -> channel_standard); else fallback = channel_raw.
    - channel_best: if non-null in df, keep; else fallback = channel_standard.

    mapping_df must have columns "channel_raw" and "channel_standard" if provided.
    """
    out = df.copy()
    if COL_CHANNEL_RAW not in out.columns:
        out[COL_CHANNEL_RAW] = pd.NA

    # channel_standard
    if COL_CHANNEL_STANDARD not in out.columns:
        out[COL_CHANNEL_STANDARD] = pd.NA
    need_standard = out[COL_CHANNEL_STANDARD].isna()
    if need_standard.any() and mapping_df is not None and len(mapping_df) > 0:
        if "channel_raw" in mapping_df.columns and "channel_standard" in mapping_df.columns:
            lookup = mapping_df.drop_duplicates("channel_raw").set_index("channel_raw")["channel_standard"]
            out.loc[need_standard, COL_CHANNEL_STANDARD] = out.loc[need_standard, COL_CHANNEL_RAW].map(lookup)
    out[COL_CHANNEL_STANDARD] = out[COL_CHANNEL_STANDARD].fillna(out[COL_CHANNEL_RAW])

    # channel_best
    if COL_CHANNEL_BEST not in out.columns:
        out[COL_CHANNEL_BEST] = pd.NA
    out[COL_CHANNEL_BEST] = out[COL_CHANNEL_BEST].fillna(out[COL_CHANNEL_STANDARD])

    return out


def aum_snapshot_rule(group: pd.DataFrame) -> float:
    """
    Snapshot AUM for one grain group.

    Default logic (documented):
    - Prefer last non-null AUM by observation date: if column "date_raw" (or similar) exists, sort by it and take last non-null asset_under_management.
    - Else use max(asset_under_management) as a stable snapshot proxy.
    - Return NaN only if the whole group has no non-null AUM.
    """
    if COL_AUM not in group.columns:
        return float("nan")
    aum = group[COL_AUM]
    non_null = aum.notna()
    if not non_null.any():
        return float("nan")
    if COL_DATE_RAW in group.columns:
        # Last non-null by date_raw
        valid = group.loc[non_null].sort_values(COL_DATE_RAW)
        return float(valid[COL_AUM].iloc[-1])
    return float(aum.max())


def build_fact_monthly(
    df_raw: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
    mapping_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build curated fact at monthly grain: derive channels, aggregate nnb/nnf/aum, enforce unique grain.
    Returns (df_fact, stats). Deterministic: output sorted by grain.

    Channel mapping: pass mapping_df (columns channel_raw, channel_standard), and/or mapping_path.
    If mapping_path is set and mapping_df is not already provided, loads from path and extracts
    channel mapping; if extraction succeeds (non-empty), passes it to derive_channels.
    """
    if COL_MONTH_END not in df_raw.columns:
        raise ValueError("build_fact_monthly requires column 'month_end' in df_raw")

    rows_in = len(df_raw)
    for g in GRAIN:
        if g not in df_raw.columns:
            raise ValueError(f"build_fact_monthly requires grain column '{g}' in df_raw (or derived).")

    # Resolve channel mapping: explicit df wins; else load from path if given
    channel_mapping_used = False
    if mapping_df is None and mapping_path is not None:
        try:
            from legacy.legacy_src.curate.mapping_loader import load_data_mapping, extract_channel_mapping

            df_loaded = load_data_mapping(mapping_path)
            mapping_df = extract_channel_mapping(df_loaded)
            if mapping_df is not None and len(mapping_df) > 0:
                channel_mapping_used = True
            else:
                mapping_df = None
        except FileNotFoundError:
            mapping_df = None
    elif mapping_df is not None and len(mapping_df) > 0:
        channel_mapping_used = True

    # 1) Derive channels
    df = derive_channels(df_raw, mapping_df=mapping_df)

    # 2) Pre-aggregation duplicate report at target grain
    grain_counts = df.groupby(GRAIN, dropna=False).size()
    duplicate_groups_count = int((grain_counts > 1).sum())
    duplicate_keys_sample: list[tuple] = []
    if duplicate_groups_count > 0:
        dup_keys = grain_counts[grain_counts > 1].index
        duplicate_keys_sample = [tuple(k) for k in dup_keys[:10].tolist()]

    # 3) Aggregate to grain
    agg_dict: dict[str, Any] = {
        COL_NNB: "sum",
        COL_NNF: "sum",
    }
    # Dimensions: take first per group (representative)
    for col in [COL_CHANNEL_RAW, COL_CHANNEL_STANDARD]:
        if col in df.columns:
            agg_dict[col] = "first"
    for col in ["display_firm", "master_custodian_firm"]:
        if col in df.columns:
            agg_dict[col] = "first"

    grouped = df.groupby(GRAIN, dropna=False)
    result = grouped.agg(agg_dict)

    # AUM via snapshot rule (group apply)
    aum_series = grouped.apply(aum_snapshot_rule, include_groups=False)
    result[COL_AUM] = aum_series.values
    aum_rule_used = "last_non_null_by_date_raw" if COL_DATE_RAW in df.columns else "max"

    # 4) Enforce unique grain (no duplicates after aggregation)
    result = result.reset_index()
    n_out = len(result)
    dup = result.duplicated(subset=GRAIN, keep="first")
    if dup.any():
        raise ValueError(f"Duplicate grain rows after aggregation: {int(dup.sum())} duplicates.")

    # Optional channel mapping: apply_channel_map (trace + status) or legacy enrich_channels.
    channel_enrichment_report: dict[str, Any] | None = None
    mapping_stats: dict[str, Any] | None = None
    if mapping_df is not None and all(c in mapping_df.columns for c in KEY_COLS):
        if all(c in mapping_df.columns for c in APPLY_OUT_COLS):
            result, mapping_stats = apply_channel_map(result, mapping_df)
        elif any(c in mapping_df.columns for c in ENRICH_COLS):
            result, channel_enrichment_report = enrich_channels(result, mapping_df)

    # 5) Sort by grain for deterministic output
    result = result.sort_values(GRAIN).reset_index(drop=True)

    # 6) Null counts in grain fields
    null_counts = {g: int(result[g].isna().sum()) for g in GRAIN if g in result.columns}

    stats: dict[str, Any] = {
        "rows_in": rows_in,
        "rows_out": n_out,
        "duplicate_groups_count": duplicate_groups_count,
        "duplicate_keys_sample": duplicate_keys_sample,
        "aum_rule_used": aum_rule_used,
        "null_counts_grain": null_counts,
        "channel_mapping_used": channel_mapping_used,
        "channel_enrichment": channel_enrichment_report,
        "mapping_stats": mapping_stats,
    }

    return result, stats
