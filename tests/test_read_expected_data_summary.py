"""
Unit tests for pipelines.validation.read_expected_data_summary: month parsing,
percent-to-decimal, float coercion, and normalize_expected_rates_frame.

Uses small constructed DataFrames only (no real Excel file required).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.validation.read_expected_data_summary import (
    _parse_month_to_end,
    _percent_to_decimal_series,
    _coerce_float64_series,
    normalize_expected_rates_frame,
)


# --- _parse_month_to_end ---


def test_parse_month_to_end_iso_strings() -> None:
    s = pd.Series(["2024-01-15", "2024-02-01", "2024-03-31"])
    out = _parse_month_to_end(s, month_format=None, timezone_naive=True)
    assert len(out) == 3
    assert out.iloc[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[1] == pd.Timestamp("2024-02-29")  # 2024 leap year
    assert out.iloc[2] == pd.Timestamp("2024-03-31")


def test_parse_month_to_end_with_format() -> None:
    s = pd.Series(["Jan-2024", "Feb-2024"])
    out = _parse_month_to_end(s, month_format="%b-%Y", timezone_naive=True)
    assert out.iloc[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[1] == pd.Timestamp("2024-02-29")


def test_parse_month_to_end_invalid_returns_nat() -> None:
    s = pd.Series(["2024-01-15", "not-a-date", "2024-03-01"])
    out = _parse_month_to_end(s, month_format=None, timezone_naive=True)
    assert pd.isna(out.iloc[1])
    assert out.iloc[0] == pd.Timestamp("2024-01-31")
    assert out.iloc[2] == pd.Timestamp("2024-03-31")


# --- _percent_to_decimal_series ---


def test_percent_to_decimal_string_with_pct() -> None:
    s = pd.Series(["3.2%", " 5.5 % ", "10%"])
    out = _percent_to_decimal_series(s, percent_scale=100.0)
    assert out.iloc[0] == pytest.approx(0.032)
    assert out.iloc[1] == pytest.approx(0.055)
    assert out.iloc[2] == pytest.approx(0.10)


def test_percent_to_decimal_numeric_uses_scale() -> None:
    s = pd.Series([3.2, 100.0, 0.5])
    out = _percent_to_decimal_series(s, percent_scale=100.0)
    assert out.iloc[0] == pytest.approx(0.032)
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(0.005)


def test_percent_to_decimal_invalid_becomes_nan() -> None:
    s = pd.Series(["3.2%", "nope", 1.0])
    out = _percent_to_decimal_series(s, percent_scale=100.0)
    assert out.iloc[0] == pytest.approx(0.032)
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.01)


# --- _coerce_float64_series ---


def test_coerce_float64_series_numeric() -> None:
    s = pd.Series([1.0, 2, "3.5"])
    out = _coerce_float64_series(s)
    assert out.dtype == "float64"
    assert out.iloc[0] == 1.0
    assert out.iloc[1] == 2.0
    assert out.iloc[2] == 3.5


def test_coerce_float64_series_invalid_to_nan() -> None:
    s = pd.Series([1.0, "x", "3.5"])
    out = _coerce_float64_series(s)
    assert pd.isna(out.iloc[1])
    assert out.iloc[0] == 1.0
    assert out.iloc[2] == 3.5


# --- normalize_expected_rates_frame ---


def test_normalize_expected_rates_frame_basic() -> None:
    df = pd.DataFrame({
        "month_end": ["2024-01-15", "2024-02-01"],
        "asset_growth_rate": [1.5, 2.0],
        "organic_growth_rate": [0.5, 0.6],
        "external_market_growth_rate": [1.0, 1.1],
    })
    clean, rejects = normalize_expected_rates_frame(
        df,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=None,
        timezone_naive=True,
        percent_to_decimal=False,
        percent_scale=100.0,
    )
    assert len(clean) == 2
    assert clean["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert clean["month_end"].iloc[1] == pd.Timestamp("2024-02-29")
    assert clean["asset_growth_rate"].iloc[0] == 1.5
    assert rejects.empty


def test_normalize_expected_rates_frame_percent_to_decimal() -> None:
    df = pd.DataFrame({
        "month_end": ["2024-01-01"],
        "asset_growth_rate": ["3.2%"],
        "organic_growth_rate": [1.5],  # numeric: divide by scale
        "external_market_growth_rate": [100.0],
    })
    clean, rejects = normalize_expected_rates_frame(
        df,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=None,
        timezone_naive=True,
        percent_to_decimal=True,
        percent_scale=100.0,
    )
    assert len(clean) == 1
    assert clean["asset_growth_rate"].iloc[0] == pytest.approx(0.032)
    assert clean["organic_growth_rate"].iloc[0] == pytest.approx(0.015)
    assert clean["external_market_growth_rate"].iloc[0] == pytest.approx(1.0)
    assert rejects.empty


def test_normalize_expected_rates_frame_invalid_month_rejected() -> None:
    df = pd.DataFrame({
        "month_end": ["2024-01-15", "bad", "2024-03-01"],
        "asset_growth_rate": [1.0, 2.0, 3.0],
        "organic_growth_rate": [0.1, 0.2, 0.3],
        "external_market_growth_rate": [0.5, 0.5, 0.5],
    })
    clean, rejects = normalize_expected_rates_frame(
        df,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=None,
        timezone_naive=True,
        percent_to_decimal=False,
        percent_scale=100.0,
    )
    assert len(clean) == 2
    assert "invalid_month" in rejects["reason"].values
    assert rejects["original_index"].tolist() == [1]


def test_normalize_expected_rates_frame_duplicate_month_end_keeps_first() -> None:
    df = pd.DataFrame({
        "month_end": ["2024-01-15", "2024-01-20", "2024-02-01"],
        "asset_growth_rate": [1.0, 2.0, 3.0],
        "organic_growth_rate": [0.1, 0.2, 0.3],
        "external_market_growth_rate": [0.5, 0.5, 0.5],
    })
    clean, rejects = normalize_expected_rates_frame(
        df,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=None,
        timezone_naive=True,
        percent_to_decimal=False,
        percent_scale=100.0,
    )
    assert len(clean) == 2
    assert clean["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert clean["asset_growth_rate"].iloc[0] == 1.0  # first occurrence kept
    assert "duplicate_month_end" in rejects["reason"].values
    assert 1 in rejects["original_index"].values


def test_normalize_expected_rates_frame_empty() -> None:
    df = pd.DataFrame(columns=["month_end", "asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"])
    clean, rejects = normalize_expected_rates_frame(
        df,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=None,
        timezone_naive=True,
        percent_to_decimal=False,
        percent_scale=100.0,
    )
    assert clean.empty
    assert list(rejects.columns) == ["original_index", "reason"]
    assert len(rejects) == 0


# --- load_expected_rates (with minimal Excel + mock policy) ---


def test_load_expected_rates_missing_columns_raises(tmp_path: Path) -> None:
    """Required columns missing in sheet must raise ValueError."""
    from openpyxl import Workbook
    from legacy.legacy_pipelines.contracts.validation_policy_contract import (
        DEFAULT_HIGHLIGHTED,
        ExpectedColumnsConfig,
        FailFastConfig,
        NormalizationConfig,
        ToleranceConfig,
        ValidationPolicy,
        WorkbookConfig,
    )
    from legacy.legacy_pipelines.validation.read_expected_data_summary import load_expected_rates

    xlsx = tmp_path / "summary.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "DATA SUMMARY"
    ws.append(["Month", "AssetGrowth"])  # missing OrganicGrowth, ExternalMarketGrowth
    ws.append(["2024-01-15", 1.5])
    wb.save(xlsx)

    policy = ValidationPolicy(
        workbook=WorkbookConfig(path=str(xlsx), sheet="DATA SUMMARY", month_column="Month", month_format=None),
        expected_columns=ExpectedColumnsConfig(
            asset_growth_rate="AssetGrowth",
            organic_growth_rate="OrganicGrowth",
            external_market_growth_rate="ExternalMarketGrowth",
        ),
        normalization=NormalizationConfig(
            percent_to_decimal=False, percent_scale=100.0, month_align="month_end", timezone_naive=True
        ),
        tolerance=ToleranceConfig(abs_tol=0.0, rel_tol=0.0),
        fail_fast=FailFastConfig(max_mismatched_months=10, max_deviation=1.0, fail_on_missing_months=True),
        highlighted=DEFAULT_HIGHLIGHTED,
    )
    with pytest.raises(ValueError, match="Required columns missing"):
        load_expected_rates(policy)


def test_load_expected_rates_success_writes_no_rejects(tmp_path: Path) -> None:
    """Full load with valid data returns clean df and no rejects file (or empty)."""
    from openpyxl import Workbook
    from legacy.legacy_pipelines.contracts.validation_policy_contract import (
        DEFAULT_HIGHLIGHTED,
        ExpectedColumnsConfig,
        FailFastConfig,
        NormalizationConfig,
        ToleranceConfig,
        ValidationPolicy,
        WorkbookConfig,
    )
    from legacy.legacy_pipelines.validation.read_expected_data_summary import load_expected_rates

    xlsx = tmp_path / "summary.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "DATA SUMMARY"
    ws.append(["Month", "AssetGrowth", "OrganicGrowth", "ExternalMarketGrowth"])
    ws.append(["2024-01-15", 1.5, 0.5, 1.0])
    ws.append(["2024-02-01", 2.0, 0.6, 1.1])
    wb.save(xlsx)

    policy = ValidationPolicy(
        workbook=WorkbookConfig(path=str(xlsx), sheet="DATA SUMMARY", month_column="Month", month_format=None),
        expected_columns=ExpectedColumnsConfig(
            asset_growth_rate="AssetGrowth",
            organic_growth_rate="OrganicGrowth",
            external_market_growth_rate="ExternalMarketGrowth",
        ),
        normalization=NormalizationConfig(
            percent_to_decimal=False, percent_scale=100.0, month_align="month_end", timezone_naive=True
        ),
        tolerance=ToleranceConfig(abs_tol=0.0, rel_tol=0.0),
        fail_fast=FailFastConfig(max_mismatched_months=10, max_deviation=1.0, fail_on_missing_months=True),
        highlighted=DEFAULT_HIGHLIGHTED,
    )
    df = load_expected_rates(policy)
    assert len(df) == 2
    assert list(df.columns) == ["month_end", "asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]
    assert df["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert df["month_end"].iloc[1] == pd.Timestamp("2024-02-29")
    assert df["asset_growth_rate"].iloc[0] == 1.5
