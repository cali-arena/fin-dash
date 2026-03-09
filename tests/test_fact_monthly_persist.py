"""
Pytest: fact_monthly_store persist_fact_monthly — parquet + meta, validation failure writes meta with validation_errors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.persist.fact_monthly_store import persist_fact_monthly


def _valid_fact_df(rows: int = 2) -> pd.DataFrame:
    """Small df_fact with correct schema/dtypes for fact_monthly.schema.yml."""
    return pd.DataFrame({
        "month_end": pd.to_datetime(["2021-01-31", "2021-02-28"][:rows]),
        "product_ticker": ["AGG", "BND"][:rows],
        "channel_raw": ["RIA", "RIA"][:rows],
        "channel_standard": ["RIA", "RIA"][:rows],
        "channel_best": ["RIA", "RIA"][:rows],
        "src_country": ["US", "US"][:rows],
        "product_country": ["US", "US"][:rows],
        "segment": ["S1", "S2"][:rows],
        "sub_segment": ["SS1", "SS2"][:rows],
        "display_firm": ["F", "F"][:rows],
        "master_custodian_firm": ["M", "M"][:rows],
        "asset_under_management": [100.0, 200.0][:rows],
        "net_new_business": [1.0, 2.0][:rows],
        "net_new_base_fees": [0.1, 0.2][:rows],
    })


def test_persist_writes_parquet_and_meta() -> None:
    """persist_fact_monthly writes curated/fact_monthly.parquet and curated/fact_monthly.meta.json."""
    df = _valid_fact_df(rows=2)
    root = PROJECT_ROOT / "tmp_fact_persist_test"
    root.mkdir(exist_ok=True)
    path_parquet = root / "curated" / "fact_monthly.parquet"
    path_meta = root / "curated" / "fact_monthly.meta.json"
    try:
        meta = persist_fact_monthly(
            df,
            dataset_version="test_v1",
            pipeline_version="p1",
            schema_path=PROJECT_ROOT / "schemas" / "fact_monthly.schema.yml",
            root=root,
        )
        assert path_parquet.exists()
        assert path_meta.exists()
        assert meta["dataset_version"] == "test_v1"
        assert meta["rowcount"] == 2
        assert "grain" in meta and len(meta["grain"]) > 0
        assert meta["month_end_min"] == "2021-01-31"
        assert meta["month_end_max"] == "2021-02-28"
        assert "schema_hash" in meta and isinstance(meta["schema_hash"], str) and len(meta["schema_hash"]) == 64
        assert "dtypes" in meta and isinstance(meta["dtypes"], dict)
    finally:
        if path_parquet.exists():
            path_parquet.unlink()
        if path_meta.exists():
            path_meta.unlink()
        if (root / "curated").exists():
            (root / "curated").rmdir()
        if root.exists():
            root.rmdir()


def test_persist_duplicate_grain_fails_and_writes_meta_with_validation_errors() -> None:
    """DataFrame with duplicate grain: persist_fact_monthly fails and writes meta with validation_errors."""
    df = _valid_fact_df(rows=2)
    # Duplicate grain: same (month_end, product_ticker, channel_best, ...)
    df = pd.concat([df, df.iloc[:1]], ignore_index=True)
    root = PROJECT_ROOT / "tmp_fact_persist_dup_test"
    root.mkdir(exist_ok=True)
    path_parquet = root / "curated" / "fact_monthly.parquet"
    path_meta = root / "curated" / "fact_monthly.meta.json"
    try:
        with pytest.raises(ValueError, match="validation failed"):
            persist_fact_monthly(
                df,
                dataset_version="test_v2",
                pipeline_version="p1",
                schema_path=PROJECT_ROOT / "schemas" / "fact_monthly.schema.yml",
                root=root,
            )
        assert not path_parquet.exists()
        assert path_meta.exists()
        with open(path_meta, encoding="utf-8") as f:
            meta = json.load(f)
        assert "validation_errors" in meta
        assert len(meta["validation_errors"]) > 0
        assert any("unique" in e.lower() or "duplicate" in e.lower() for e in meta["validation_errors"])
    finally:
        if path_parquet.exists():
            path_parquet.unlink()
        if path_meta.exists():
            path_meta.unlink()
        if (root / "curated").exists():
            (root / "curated").rmdir()
        if root.exists():
            root.rmdir()
