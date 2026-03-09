"""
Unit tests for pipelines.contracts.validation_policy_contract: canonical keys,
duplicate mapping detection, summarize_policy, policy_hash stability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    DEFAULT_HIGHLIGHTED,
    ExpectedColumnsConfig,
    FailFastConfig,
    NormalizationConfig,
    ToleranceConfig,
    ValidationPolicy,
    ValidationPolicyError,
    WorkbookConfig,
    load_and_validate_validation_policy,
    policy_hash,
    summarize_policy,
)


def _make_policy(
    workbook_path: str = "data/summary.xlsx",
    asset_col: str = "AssetGrowth",
    organic_col: str = "OrganicGrowth",
    external_col: str = "ExternalMarketGrowth",
) -> ValidationPolicy:
    return ValidationPolicy(
        workbook=WorkbookConfig(
            path=workbook_path,
            sheet="DATA SUMMARY",
            month_column="Month",
            month_format=None,
        ),
        expected_columns=ExpectedColumnsConfig(
            asset_growth_rate=asset_col,
            organic_growth_rate=organic_col,
            external_market_growth_rate=external_col,
        ),
        normalization=NormalizationConfig(
            percent_to_decimal=False,
            percent_scale=100.0,
            month_align="month_end",
            timezone_naive=True,
        ),
        tolerance=ToleranceConfig(abs_tol=0.0, rel_tol=0.0),
        fail_fast=FailFastConfig(
            max_mismatched_months=10,
            max_deviation=1.0,
            fail_on_missing_months=True,
        ),
        highlighted=DEFAULT_HIGHLIGHTED,
    )


VALID_YAML = """
validation:
  workbook:
    path: data/summary.xlsx
    sheet: DATA SUMMARY
    month_column: Month
    month_format: null
  expected_columns:
    asset_growth_rate: AssetGrowth
    organic_growth_rate: OrganicGrowth
    external_market_growth_rate: ExternalMarketGrowth
  normalization:
    percent_to_decimal: false
    percent_scale: 100
    month_align: month_end
    timezone_naive: true
  tolerance:
    abs_tol: 0.0
    rel_tol: 0.0
  fail_fast:
    max_mismatched_months: 10
    max_deviation: 1.0
    fail_on_missing_months: true
"""

DUPLICATE_MAPPING_YAML = """
validation:
  workbook:
    path: data/summary.xlsx
    sheet: DATA SUMMARY
    month_column: Month
    month_format: null
  expected_columns:
    asset_growth_rate: SameCol
    organic_growth_rate: SameCol
    external_market_growth_rate: ExternalMarketGrowth
  normalization:
    percent_to_decimal: false
    percent_scale: 100
    month_align: month_end
    timezone_naive: true
  tolerance:
    abs_tol: 0.0
    rel_tol: 0.0
  fail_fast:
    max_mismatched_months: 10
    max_deviation: 1.0
    fail_on_missing_months: true
"""


def test_duplicate_mapping_raises(tmp_path: Path) -> None:
    """Two canonical keys mapping to the same Excel column must raise ValidationPolicyError."""
    yaml_path = tmp_path / "validation_policy.yml"
    yaml_path.write_text(DUPLICATE_MAPPING_YAML, encoding="utf-8")
    with pytest.raises(ValidationPolicyError, match="duplicate mapping") as exc_info:
        load_and_validate_validation_policy(yaml_path)
    assert "SameCol" in str(exc_info.value)
    assert "asset_growth_rate" in str(exc_info.value) or "organic_growth_rate" in str(exc_info.value)


def test_valid_yaml_loads(tmp_path: Path) -> None:
    """Valid YAML loads and has distinct column mappings."""
    yaml_path = tmp_path / "validation_policy.yml"
    yaml_path.write_text(VALID_YAML, encoding="utf-8")
    policy = load_and_validate_validation_policy(yaml_path)
    assert policy.workbook.sheet == "DATA SUMMARY"
    assert policy.expected_columns.asset_growth_rate == "AssetGrowth"
    assert policy.expected_columns.organic_growth_rate == "OrganicGrowth"
    assert policy.expected_columns.external_market_growth_rate == "ExternalMarketGrowth"


def test_summarize_policy_structure() -> None:
    """summarize_policy returns dict with expected keys; all JSON-serializable."""
    policy = _make_policy()
    summary = summarize_policy(policy)
    assert "workbook_path" in summary
    assert "sheet" in summary
    assert "month_column" in summary
    assert "expected_columns" in summary
    assert "normalization" in summary
    assert "tolerance" in summary
    assert "fail_fast" in summary
    assert summary["workbook_path"] == "data/summary.xlsx"
    assert summary["sheet"] == "DATA SUMMARY"
    assert summary["month_column"] == "Month"
    assert summary["expected_columns"] == {
        "asset_growth_rate": "AssetGrowth",
        "organic_growth_rate": "OrganicGrowth",
        "external_market_growth_rate": "ExternalMarketGrowth",
    }
    assert summary["normalization"]["percent_to_decimal"] is False
    assert summary["tolerance"]["abs_tol"] == 0.0
    assert summary["fail_fast"]["max_mismatched_months"] == 10
    # Must be JSON-serializable (no custom objects)
    import json
    json.dumps(summary)


def test_policy_hash_stability_same_policy() -> None:
    """Same policy produces the same hash every time."""
    p1 = _make_policy()
    p2 = _make_policy()
    assert policy_hash(p1) == policy_hash(p2)
    assert len(policy_hash(p1)) == 64  # SHA-256 hex


def test_policy_hash_changes_with_policy() -> None:
    """Any change in summarized policy changes the hash."""
    base = _make_policy()
    h_base = policy_hash(base)
    # Change workbook path
    p2 = _make_policy(workbook_path="other/path.xlsx")
    assert policy_hash(p2) != h_base
    # Change one expected column mapping
    p3 = _make_policy(organic_col="OrganicGrowthRate")
    assert policy_hash(p3) != h_base
    # Change tolerance
    p4 = ValidationPolicy(
        workbook=base.workbook,
        expected_columns=base.expected_columns,
        normalization=base.normalization,
        tolerance=ToleranceConfig(abs_tol=1e-5, rel_tol=0.0),
        fail_fast=base.fail_fast,
        highlighted=base.highlighted,
    )
    assert policy_hash(p4) != h_base


def test_canonical_json_deterministic() -> None:
    """Policy hash is deterministic for same content (sorted keys, minimal separators)."""
    from legacy.legacy_pipelines.contracts.validation_policy_contract import _canonical_json_dumps
    policy = _make_policy()
    summary = summarize_policy(policy)
    s1 = _canonical_json_dumps(summary)
    s2 = _canonical_json_dumps(summary)
    assert s1 == s2
    # No unnecessary whitespace (separators are ",", ":" only)
    assert "\n" not in s1
    assert "  " not in s1
    assert policy_hash(policy) == policy_hash(policy)
