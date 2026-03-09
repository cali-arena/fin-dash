"""
Tests for pipelines.agg.meta_utils: dataset_version stability, schema_hash sensitivity.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.agg import meta_utils
from legacy.legacy_pipelines.contracts.agg_policy_contract import load_and_validate_agg_policy

POLICY_YAML = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum", "nnb"]
    rates: []
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "store_numerators"
    weights: {}
"""


def test_dataset_version_stable_given_same_inputs(tmp_path: Path) -> None:
    """Same source_version + policy_hash + pipeline_version yields same dataset_version."""
    (tmp_path / "p.yml").write_text(POLICY_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    policy_hash_val = meta_utils.hash_policy(policy)

    v1 = meta_utils.compute_agg_dataset_version("v1", policy_hash_val, "pipe1")
    v2 = meta_utils.compute_agg_dataset_version("v1", policy_hash_val, "pipe1")
    assert v1 == v2
    assert len(v1) == 40  # sha1 hex

    v3 = meta_utils.compute_agg_dataset_version("v2", policy_hash_val, "pipe1")
    assert v3 != v1
    v4 = meta_utils.compute_agg_dataset_version("v1", policy_hash_val + "x", "pipe1")
    assert v4 != v1
    v5 = meta_utils.compute_agg_dataset_version("v1", policy_hash_val, "pipe2")
    assert v5 != v1


def test_schema_hash_changes_if_column_order_changes() -> None:
    """schema_hash differs when column order differs (columns in order, not sorted)."""
    df1 = pd.DataFrame({"a": [1], "b": [2.0]})
    df2 = pd.DataFrame({"b": [2.0], "a": [1]})
    h1 = meta_utils.hash_schema(df1)
    h2 = meta_utils.hash_schema(df2)
    assert h1 != h2
    assert len(h1) == 40
    assert len(h2) == 40


def test_schema_hash_changes_if_dtype_changes() -> None:
    """schema_hash differs when dtype of a column changes."""
    df1 = pd.DataFrame({"x": [1, 2]})  # int64
    df2 = pd.DataFrame({"x": [1.0, 2.0]})  # float64
    h1 = meta_utils.hash_schema(df1)
    h2 = meta_utils.hash_schema(df2)
    assert h1 != h2


def test_schema_hash_stable_same_order_and_dtype() -> None:
    """Same column order and dtypes give same schema_hash."""
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-31")], "begin_aum": [100.0]})
    assert meta_utils.hash_schema(df) == meta_utils.hash_schema(df)


def test_load_source_metrics_version_prefers_curated_meta(tmp_path: Path) -> None:
    """Prefer curated/metrics_monthly.meta.json dataset_version over data/.version.json."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        json.dumps({"dataset_version": "from_curated"}), encoding="utf-8"
    )
    (tmp_path / "data" / ".version.json").write_text(
        json.dumps({"dataset_version": "from_data"}), encoding="utf-8"
    )
    v = meta_utils.load_source_metrics_version(tmp_path, "curated/metrics_monthly.parquet")
    assert v == "from_curated"


def test_load_source_metrics_version_fallback_to_version_json(tmp_path: Path) -> None:
    """Fallback to data/.version.json when curated meta missing."""
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / ".version.json").write_text(
        json.dumps({"dataset_version": "fallback"}), encoding="utf-8"
    )
    v = meta_utils.load_source_metrics_version(tmp_path, "curated/metrics_monthly.parquet")
    assert v == "fallback"


def test_get_pipeline_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """PIPELINE_VERSION env is used when set."""
    monkeypatch.setenv("PIPELINE_VERSION", "env-1.0")
    assert meta_utils.get_pipeline_version() == "env-1.0"


def test_write_meta_atomic(tmp_path: Path) -> None:
    """write_meta writes JSON atomically."""
    path = tmp_path / "out.meta.json"
    meta = {"dataset_version": "abc", "rowcount": 2}
    meta_utils.write_meta(path, meta)
    assert path.exists()
    assert json.loads(path.read_text()) == meta
