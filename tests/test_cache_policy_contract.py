"""
Tests for pipelines.contracts.cache_policy_contract: missing keys, invalid ttl, invalid class.
"""
from pathlib import Path

import pytest

from legacy.legacy_pipelines.contracts.cache_policy_contract import (
    CachePolicyError,
    load_and_validate_cache_policy,
)


def _write_yml(tmp_path: Path, content: str, name: str = "cache_policy.yml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


VALID_MINIMAL = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
    channel_monthly: "fast"
    ticker_monthly: "fast"
    geo_monthly: "fast"
    segment_monthly: "fast"
    kpi_cards: "medium"
    correlation_matrix: "heavy"
  max_entries:
    fast: 256
    medium: 128
    heavy: 32
"""


def test_load_valid_example(tmp_path: Path) -> None:
    """Load a valid config returns CachePolicy with expected values."""
    path = _write_yml(tmp_path, VALID_MINIMAL)
    policy = load_and_validate_cache_policy(path)
    assert policy.dataset_version_source_path == "curated/metrics_monthly.meta.json"
    assert policy.dataset_version_source_key == "dataset_version"
    assert "dataset_version" in policy.cache_keys
    assert "filter_state_hash" in policy.cache_keys
    assert "query_name" in policy.cache_keys
    assert policy.ttl_seconds["fast"] == 60
    assert policy.ttl_seconds["medium"] == 300
    assert policy.ttl_seconds["heavy"] == 1800
    assert policy.query_classes["firm_monthly"] == "fast"
    assert policy.query_classes["correlation_matrix"] == "heavy"
    assert policy.max_entries is not None
    assert policy.max_entries["fast"] == 256
    assert policy.max_entries["heavy"] == 32


def test_missing_keys(tmp_path: Path) -> None:
    """Missing required top-level keys raises CachePolicyError."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  # cache_keys missing
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "cache_keys" in str(exc_info.value).lower() or "must" in str(exc_info.value).lower()


def test_cache_keys_missing_required(tmp_path: Path) -> None:
    """cache_keys missing dataset_version / filter_state_hash / query_name raises."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    # filter_state_hash and query_name missing
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "cache_keys" in str(exc_info.value).lower()
    assert "filter_state_hash" in str(exc_info.value) or "query_name" in str(exc_info.value)


def test_invalid_ttl(tmp_path: Path) -> None:
    """ttl_seconds with zero or missing fast/medium/heavy raises."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 0
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "ttl_seconds" in str(exc_info.value).lower()
    assert "positive" in str(exc_info.value).lower() or "0" in str(exc_info.value)


def test_invalid_ttl_missing_class(tmp_path: Path) -> None:
    """ttl_seconds missing 'heavy' raises."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "heavy" in str(exc_info.value).lower() or "ttl_seconds" in str(exc_info.value).lower()


def test_invalid_query_class(tmp_path: Path) -> None:
    """query_classes value not in fast/medium/heavy raises."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "slow"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "query_classes" in str(exc_info.value).lower()
    assert "fast" in str(exc_info.value) or "medium" in str(exc_info.value) or "heavy" in str(exc_info.value)


def test_dataset_version_source_path_not_json(tmp_path: Path) -> None:
    """dataset_version_source.path must end with .json."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.yaml"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert ".json" in str(exc_info.value).lower()


def test_max_entries_invalid_when_present(tmp_path: Path) -> None:
    """max_entries when present must have fast/medium/heavy and values > 0."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
  max_entries:
    fast: 0
    medium: 128
    heavy: 32
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy(path)
    assert "max_entries" in str(exc_info.value).lower()


def test_max_entries_optional(tmp_path: Path) -> None:
    """Valid config without max_entries returns policy with max_entries=None."""
    yml = """
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys:
    - "dataset_version"
    - "filter_state_hash"
    - "query_name"
  ttl_seconds:
    fast: 60
    medium: 300
    heavy: 1800
  query_classes:
    firm_monthly: "fast"
"""
    path = _write_yml(tmp_path, yml)
    policy = load_and_validate_cache_policy(path)
    assert policy.max_entries is None


def test_config_not_found() -> None:
    """Non-existent path raises CachePolicyError."""
    with pytest.raises(CachePolicyError) as exc_info:
        load_and_validate_cache_policy("configs/nonexistent_cache_policy.yml")
    assert "not found" in str(exc_info.value).lower()
