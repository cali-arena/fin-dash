"""
Drill path contract for UI: load config + slices_index; expose enabled paths and slices per path.
UI must use only these paths and slices (no ad-hoc groupbys).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
SLICES_INDEX_REL = "curated/slices_index.parquet"


def _load_config(root: Path) -> dict[str, Any]:
    """Load drill_paths.yml."""
    path = root / DEFAULT_DRILL_PATHS_CONFIG
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        logger.warning("Could not load drill_paths config from %s: %s", path, e)
        return {}


def _load_slices_index(root: Path) -> pd.DataFrame:
    """Load curated/slices_index.parquet; empty DataFrame if missing."""
    path = root / SLICES_INDEX_REL
    if not path.exists():
        return pd.DataFrame(columns=["path_id", "path_label", "keys", "slice_id", "slice_key", "row_count"])
    return pd.read_parquet(path)


def get_enabled_paths(root: Path) -> list[dict[str, str]]:
    """
    Return list of enabled drill paths from config + slices_index.
    Each item: {"path_id": str, "path_label": str}.
    Uses slices_index to ensure only paths that have slices are returned.
    """
    config = _load_config(root)
    index_df = _load_slices_index(root)
    if index_df.empty:
        return []
    # Unique path_id / path_label from index (index is built from config, so only enabled paths)
    paths = index_df[["path_id", "path_label"]].drop_duplicates()
    paths = paths.sort_values("path_id", kind="mergesort")
    return paths.to_dict("records")


def get_slices_for_path(root: Path, path_id: str) -> pd.DataFrame:
    """
    Return DataFrame of slices for the given path_id.
    Columns: path_id, path_label, keys, slice_id, slice_key, row_count (or subset for UI).
    Backed by slice_id; UI shows slice_key, stores slice_id for filtering.
    """
    index_df = _load_slices_index(root)
    if index_df.empty or "path_id" not in index_df.columns:
        return pd.DataFrame(columns=["path_id", "path_label", "slice_id", "slice_key"])
    out = index_df[index_df["path_id"] == path_id].copy()
    out = out.sort_values("slice_key", kind="mergesort").reset_index(drop=True)
    return out[["path_id", "path_label", "slice_id", "slice_key"]]
