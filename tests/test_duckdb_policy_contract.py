"""
Tests for pipelines.contracts.duckdb_policy_contract: missing keys, invalid refresh_mode, invalid view source.
"""
from pathlib import Path

import pytest

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)


def _write_yml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "duckdb_policy.yml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_valid_example(tmp_path: Path) -> None:
    """Load a valid config returns DuckDBPolicy with expected values."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    curated: {}
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
    v_channel_monthly: { source: "agg.channel_monthly" }
    v_ticker_monthly: { source: "agg.ticker_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    policy = load_and_validate_duckdb_policy(path)
    assert policy.db_path == "analytics/analytics.duckdb"
    assert policy.schema == "analytics"
    assert policy.refresh_mode == "rebuild"
    assert policy.source_paths["agg"]["firm_monthly"] == "agg/firm_monthly.parquet"
    assert policy.views_to_create["v_firm_monthly"]["source"] == "agg.firm_monthly"
    assert policy.reads_views_only is True


def test_missing_keys(tmp_path: Path) -> None:
    """Missing required keys raise DuckDBPolicyError."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  # refresh_mode missing
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "refresh_mode" in str(exc_info.value).lower() or "must" in str(exc_info.value).lower()


def test_missing_source_paths_agg(tmp_path: Path) -> None:
    """source_paths.agg missing required firm/channel/ticker raises."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      # channel_monthly, ticker_monthly missing
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "source_paths.agg" in str(exc_info.value) or "channel_monthly" in str(exc_info.value) or "ticker" in str(exc_info.value)


def test_invalid_refresh_mode(tmp_path: Path) -> None:
    """refresh_mode not in {rebuild, incremental} raises DuckDBPolicyError."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "full_refresh"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "rebuild" in str(exc_info.value).lower() or "incremental" in str(exc_info.value).lower()
    assert "full_refresh" in str(exc_info.value) or "got" in str(exc_info.value).lower()


def test_invalid_view_source_not_in_agg(tmp_path: Path) -> None:
    """View source referencing a table not in source_paths.agg raises."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
    v_geo_monthly: { source: "agg.geo_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "geo_monthly" in str(exc_info.value)
    assert "source_paths.agg" in str(exc_info.value) or "not in" in str(exc_info.value).lower()


def test_invalid_view_source_format(tmp_path: Path) -> None:
    """View source that is not schema.table format raises."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "agg.firm_monthly" in str(exc_info.value) or "reference" in str(exc_info.value).lower()


def test_view_key_must_start_with_v_(tmp_path: Path) -> None:
    """views_to_create key not starting with v_ raises."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "v_" in str(exc_info.value)


def test_ui_rule_reads_views_only_must_be_true(tmp_path: Path) -> None:
    """ui_rule.reads_views_only must be true."""
    yml = """
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: false
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert "reads_views_only" in str(exc_info.value)
    assert "true" in str(exc_info.value).lower()


def test_db_path_must_endswith_duckdb(tmp_path: Path) -> None:
    """db_path must end with .duckdb."""
    yml = """
duckdb:
  db_path: "analytics/analytics.db"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
"""
    path = _write_yml(tmp_path, yml)
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy(path)
    assert ".duckdb" in str(exc_info.value)


def test_config_not_found() -> None:
    """Missing config file raises DuckDBPolicyError."""
    with pytest.raises(DuckDBPolicyError) as exc_info:
        load_and_validate_duckdb_policy("configs/nonexistent_duckdb_policy.yml")
    assert "not found" in str(exc_info.value).lower()
