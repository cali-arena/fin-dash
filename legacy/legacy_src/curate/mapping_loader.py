"""
Load DATA_MAPPING file and extract channel raw -> standard mapping for optional use in curation.
No business logic beyond extracting (channel_raw, channel_standard) pairs.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd


def _is_empty_or_unnamed(name: str) -> bool:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return True
    s = str(name).strip()
    return s == "" or bool(re.match(r"^Unnamed\s*:\s*\d+$", s, re.IGNORECASE))


def load_data_mapping(
    path: str | Path = "data/input/DATA_MAPPING.csv",
) -> pd.DataFrame:
    """
    Load DATA_MAPPING CSV as strings. If the header row is empty or unnamed,
    assign placeholder column names col1, col2, ... colN.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Normalize to str for header check
    cols = [str(c).strip() for c in df.columns]
    if all(_is_empty_or_unnamed(c) or c == "" for c in cols):
        n = len(df.columns)
        df.columns = [f"col{i + 1}" for i in range(n)]
    return df


# Names that suggest "raw/source" and "standard/target" column roles (for heuristic 1).
_RAW_LIKE = re.compile(
    r"^(channel_?raw|source|from|original|input|key)\s*$",
    re.IGNORECASE,
)
_STANDARD_LIKE = re.compile(
    r"^(channel_?standard|standard|target|to|display|mapped)\s*$",
    re.IGNORECASE,
)

# Standard channel display names we look for in rows (heuristic 2).
_CHANNEL_STANDARD_NAMES = frozenset(
    {"Channel", "Standard Channel", "Group Channel", "Branding Channel"}
)


def extract_channel_mapping(df_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a two-column mapping (channel_raw, channel_standard) from a loaded DATA_MAPPING frame.

    Heuristics:
    1) If columns exist that look like source and standard (e.g. channel_raw / channel_standard,
       source / standard, from / to), use those columns' values as (channel_raw, channel_standard).
    2) Else, scan rows for cells equal to \"Channel\", \"Standard Channel\", \"Group Channel\",
       or \"Branding Channel\"; treat the column containing that value as standard and the
       preceding column (or a fixed source column) as raw, and build (raw, standard) pairs.

    If extraction cannot be done confidently, returns an empty DataFrame and issues a warning.
    No business logic beyond raw -> standard channel mapping.
    """
    if df_mapping is None or df_mapping.empty:
        return pd.DataFrame(columns=["channel_raw", "channel_standard"])

    # Heuristic 1: column names that look like raw + standard
    raw_col: str | None = None
    standard_col: str | None = None
    for c in df_mapping.columns:
        cs = str(c).strip()
        if _RAW_LIKE.search(cs):
            raw_col = c
        if _STANDARD_LIKE.search(cs):
            standard_col = c
    if raw_col is not None and standard_col is not None and raw_col != standard_col:
        out = (
            df_mapping[[raw_col, standard_col]]
            .copy()
            .rename(columns={raw_col: "channel_raw", standard_col: "channel_standard"})
        )
        out = out.drop_duplicates()
        # Drop rows where both are empty
        out["channel_raw"] = out["channel_raw"].astype(str).str.strip()
        out["channel_standard"] = out["channel_standard"].astype(str).str.strip()
        out = out.loc[~(out["channel_raw"].eq("") & out["channel_standard"].eq(""))]
        return out.reset_index(drop=True)

    # Heuristic 2: rows where some cell is a known "standard channel" name
    cols = list(df_mapping.columns)
    if len(cols) < 2:
        warnings.warn(
            "extract_channel_mapping: could not find channel_raw/channel_standard columns "
            "and fewer than 2 columns for row-scan heuristic; returning empty mapping.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["channel_raw", "channel_standard"])

    # Find which column (if any) often contains standard channel names
    standard_col_idx: int | None = None
    for i, c in enumerate(cols):
        vals = df_mapping.iloc[:, i].astype(str).str.strip()
        if vals.isin(_CHANNEL_STANDARD_NAMES).any():
            standard_col_idx = i
            break
    if standard_col_idx is None:
        warnings.warn(
            "extract_channel_mapping: no column contained standard channel names "
            "(Channel, Standard Channel, Group Channel, Branding Channel); returning empty mapping.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=["channel_raw", "channel_standard"])

    # Use preceding column as raw (e.g. col2 = source_column, col3 = standard_name)
    raw_col_idx = standard_col_idx - 1 if standard_col_idx > 0 else 0
    raw_col_name = cols[raw_col_idx]
    standard_col_name = cols[standard_col_idx]

    subset = df_mapping.loc[
        df_mapping.iloc[:, standard_col_idx].astype(str).str.strip().isin(_CHANNEL_STANDARD_NAMES)
    ]
    if subset.empty:
        return pd.DataFrame(columns=["channel_raw", "channel_standard"])

    out = subset.iloc[:, [raw_col_idx, standard_col_idx]].copy()
    out.columns = ["channel_raw", "channel_standard"]
    out["channel_raw"] = out["channel_raw"].astype(str).str.strip()
    out["channel_standard"] = out["channel_standard"].astype(str).str.strip()
    out = out.drop_duplicates().reset_index(drop=True)
    return out
