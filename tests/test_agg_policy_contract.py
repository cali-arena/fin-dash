"""
Unit tests for pipelines.contracts.agg_policy_contract: load_and_validate_agg_policy,
missing grains, invalid dims, weighted_avg without weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.contracts.agg_policy_contract import (
    AggPolicy,
    AggPolicyError,
    load_and_validate_agg_policy,
    policy_hash,
)

VALID_AGG_YAML = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum", "end_aum", "nnb"]
    rates: ["ogr", "fee_yield"]
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
    sub_segment: "sub_segment"
    channel_l2: "channel_l2"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"], ["segment", "sub_segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "recompute"
    weights:
      ogr: "begin_aum"
      fee_yield: "nnb"
"""


def test_valid_agg_policy_loads(tmp_path: Path) -> None:
    """Valid YAML loads and returns AggPolicy with expected fields."""
    (tmp_path / "agg_policy.yml").write_text(VALID_AGG_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    assert isinstance(policy, AggPolicy)
    assert policy.source_table == "curated/metrics_monthly.parquet"
    assert policy.time_key == "month_end"
    assert policy.measures.additive == ["begin_aum", "end_aum", "nnb"]
    assert policy.measures.rates == ["ogr", "fee_yield"]
    assert "channel_l1" in policy.dims
    assert policy.grains["firm_monthly"] == [[]]
    assert policy.grains["ticker_monthly"] == [["product_ticker"]]
    assert policy.null_handling.strategy == "UNKNOWN"
    assert policy.rollup.rates_method == "recompute"


def test_agg_policy_missing_grains_raises(tmp_path: Path) -> None:
    """Missing required grains raises AggPolicyError."""
    yaml_missing = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum"]
    rates: []
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
    sub_segment: "sub_segment"
    channel_l2: "channel_l2"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "recompute"
    weights: {}
"""
    (tmp_path / "agg_policy.yml").write_text(yaml_missing, encoding="utf-8")
    with pytest.raises(AggPolicyError) as exc_info:
        load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    assert "missing required keys" in str(exc_info.value).lower() or "segment_monthly" in str(exc_info.value)


def test_agg_policy_invalid_dims_raises(tmp_path: Path) -> None:
    """Grain referencing a dim not in dims raises AggPolicyError."""
    yaml_bad_dim = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum"]
    rates: []
  dims:
    channel_l1: "channel_l1"
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
    rates_method: "recompute"
    weights: {}
"""
    (tmp_path / "agg_policy.yml").write_text(yaml_bad_dim, encoding="utf-8")
    with pytest.raises(AggPolicyError) as exc_info:
        load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    assert "product_ticker" in str(exc_info.value) or "not in agg.dims" in str(exc_info.value) or "dims" in str(exc_info.value).lower()


def test_agg_policy_weighted_avg_without_weights_raises(tmp_path: Path) -> None:
    """rates_method weighted_avg without weights for each rate raises AggPolicyError."""
    yaml_no_weights = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum"]
    rates: ["ogr", "fee_yield"]
  dims:
    channel_l1: "channel_l1"
    channel_l2: "channel_l2"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
    sub_segment: "sub_segment"
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
    rates_method: "weighted_avg"
    weights:
      ogr: "begin_aum"
"""
    (tmp_path / "agg_policy.yml").write_text(yaml_no_weights, encoding="utf-8")
    with pytest.raises(AggPolicyError) as exc_info:
        load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    assert "weighted_avg" in str(exc_info.value).lower() or "weights" in str(exc_info.value).lower()
    assert "fee_yield" in str(exc_info.value)


def test_agg_policy_duplicate_grain_raises(tmp_path: Path) -> None:
    """Duplicate grain definition (same dim list twice) raises AggPolicyError."""
    yaml_dup = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum"]
    rates: []
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
    sub_segment: "sub_segment"
    channel_l2: "channel_l2"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"], ["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "recompute"
    weights: {}
"""
    (tmp_path / "agg_policy.yml").write_text(yaml_dup, encoding="utf-8")
    with pytest.raises(AggPolicyError) as exc_info:
        load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    assert "duplicate" in str(exc_info.value).lower()


def test_agg_policy_hash_stable(tmp_path: Path) -> None:
    """policy_hash is stable for the same policy."""
    (tmp_path / "agg_policy.yml").write_text(VALID_AGG_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "agg_policy.yml")
    h1 = policy_hash(policy)
    h2 = policy_hash(policy)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)
