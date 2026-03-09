"""
Pytest: channel_map builder QA (null key, preferred_label required) + store (hash determinism, artifacts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.mapping.channel_map_builder import KEY_COLS, build_channel_map
from legacy.legacy_src.persist.channel_map_store import dataframe_content_hash, persist_channel_map


def _valid_channel_map_df(rows: int = 2) -> pd.DataFrame:
    """Minimal valid df_mapping with KEY_COLS + OUT_COLS."""
    return pd.DataFrame({
        "channel_raw": ["RIA", "BD"][:rows],
        "channel_standard": ["RIA", "BD"][:rows],
        "channel_best": ["RIA", "BD"][:rows],
        "channel_l1": ["L1", "L1"][:rows],
        "channel_l2": ["L2", "L2"][:rows],
        "preferred_label": ["RIA Label", "BD Label"][:rows],
    })


def test_null_key_rejection_empty_string() -> None:
    """df_mapping with channel_raw = '' -> build_channel_map raises ValueError mentioning null keys."""
    df = _valid_channel_map_df(2)
    df.loc[0, "channel_raw"] = ""
    with pytest.raises(ValueError) as exc_info:
        build_channel_map(df)
    assert "null" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
    assert "channel_raw" in str(exc_info.value)


def test_null_key_rejection_na() -> None:
    """df_mapping with channel_raw = NA -> build_channel_map raises ValueError mentioning null keys."""
    df = _valid_channel_map_df(2)
    df.loc[0, "channel_raw"] = pd.NA
    with pytest.raises(ValueError) as exc_info:
        build_channel_map(df)
    assert "null" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
    assert "channel_raw" in str(exc_info.value)


def test_preferred_label_required_raises_with_count() -> None:
    """df_mapping with preferred_label null -> raises ValueError with count."""
    df = _valid_channel_map_df(2)
    df.loc[1, "preferred_label"] = pd.NA
    with pytest.raises(ValueError) as exc_info:
        build_channel_map(df)
    msg = str(exc_info.value)
    assert "preferred_label" in msg.lower()
    assert "1" in msg  # count of null/empty rows


def test_hash_determinism_reorder_rows(tmp_path: Path) -> None:
    """Same data in different row order yields identical content_hash after persist."""
    df1 = _valid_channel_map_df(2)
    df1, _ = build_channel_map(df1)
    meta1 = persist_channel_map(
        df1,
        dataset_version="v1",
        root=tmp_path,
    )
    hash1 = meta1["content_hash"]

    # Reorder rows (reverse)
    df2 = df1.iloc[::-1].reset_index(drop=True)
    meta2 = persist_channel_map(
        df2,
        dataset_version="v1",
        root=tmp_path,
    )
    hash2 = meta2["content_hash"]
    assert hash1 == hash2

    # Also assert dataframe_content_hash directly
    assert dataframe_content_hash(df1, KEY_COLS) == dataframe_content_hash(df2, KEY_COLS)


def test_artifacts_written_parquet_and_meta(tmp_path: Path) -> None:
    """persist_channel_map writes parquet (or csv fallback) + meta json; meta has rowcount, content_hash, dataset_version."""
    df = _valid_channel_map_df(2)
    df, _ = build_channel_map(df)
    meta = persist_channel_map(
        df,
        dataset_version="test-dv",
        root=tmp_path,
    )
    curated = tmp_path / "curated"
    path_parquet = curated / "channel_map.parquet"
    path_csv = curated / "channel_map.csv"
    path_meta = curated / "channel_map.meta.json"

    assert path_meta.exists()
    assert path_parquet.exists() or path_csv.exists()

    import json
    with open(path_meta, encoding="utf-8") as f:
        read_meta = json.load(f)
    assert read_meta["rowcount"] == 2
    assert "content_hash" in read_meta and len(read_meta["content_hash"]) == 64
    assert read_meta["dataset_version"] == "test-dv"
