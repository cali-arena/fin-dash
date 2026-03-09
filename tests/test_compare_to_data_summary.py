"""
Unit tests for pipelines.validation.compare_to_data_summary: build_validation_report
(abs_tol pass, rel_tol pass, fail, missing actual).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    DEFAULT_HIGHLIGHTED,
    ExpectedColumnsConfig,
    FailFastConfig,
    HighlightedConfig,
    NormalizationConfig,
    ToleranceConfig,
    ValidationPolicy,
    WorkbookConfig,
)
from legacy.legacy_pipelines.validation.compare_to_data_summary import (
    build_validation_report,
    write_validation_report,
)


def _policy(
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-4,
    highlighted: HighlightedConfig | None = None,
) -> ValidationPolicy:
    return ValidationPolicy(
        workbook=WorkbookConfig(path="", sheet="", month_column="Month", month_format=None),
        expected_columns=ExpectedColumnsConfig(
            asset_growth_rate="a", organic_growth_rate="b", external_market_growth_rate="c"
        ),
        normalization=NormalizationConfig(
            percent_to_decimal=False, percent_scale=100.0, month_align="month_end", timezone_naive=True
        ),
        tolerance=ToleranceConfig(abs_tol=abs_tol, rel_tol=rel_tol),
        fail_fast=FailFastConfig(max_mismatched_months=10, max_deviation=1.0, fail_on_missing_months=True),
        highlighted=highlighted or DEFAULT_HIGHLIGHTED,
    )


def test_one_month_passing_by_abs_tol() -> None:
    """Small absolute diff within abs_tol => pass."""
    policy = _policy(abs_tol=1e-5, rel_tol=1e-4)
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "asset_growth_rate": [0.10],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    actual = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "asset_growth_rate": [0.10 + 1e-6],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    report = build_validation_report(expected, actual, policy)
    assert len(report) == 1
    assert report["pass_asset_growth_rate"].iloc[0] == True
    assert report["pass_organic_growth_rate"].iloc[0] == True
    assert report["pass_external_market_growth_rate"].iloc[0] == True
    assert report["all_pass"].iloc[0] == True
    assert report["any_fail"].iloc[0] == False
    assert report["abs_err_asset_growth_rate"].iloc[0] == pytest.approx(1e-6)


def test_one_month_passing_by_rel_tol() -> None:
    """Larger abs diff but within rel_tol => pass."""
    policy = _policy(abs_tol=1e-9, rel_tol=0.01)
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-02-29")],
        "asset_growth_rate": [1.0],
        "organic_growth_rate": [0.5],
        "external_market_growth_rate": [0.1],
    })
    actual = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-02-29")],
        "asset_growth_rate": [1.005],
        "organic_growth_rate": [0.5],
        "external_market_growth_rate": [0.1],
    })
    report = build_validation_report(expected, actual, policy)
    assert len(report) == 1
    assert report["pass_asset_growth_rate"].iloc[0] == True
    assert report["rel_err_asset_growth_rate"].iloc[0] == pytest.approx(0.005)
    assert report["all_pass"].iloc[0] == True


def test_one_month_failing() -> None:
    """Diff exceeds both abs_tol and rel_tol => fail."""
    policy = _policy(abs_tol=1e-6, rel_tol=1e-6)
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-03-31")],
        "asset_growth_rate": [0.10],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    actual = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-03-31")],
        "asset_growth_rate": [0.20],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    report = build_validation_report(expected, actual, policy)
    assert len(report) == 1
    assert report["pass_asset_growth_rate"].iloc[0] == False
    assert report["pass_organic_growth_rate"].iloc[0] == True
    assert report["all_pass"].iloc[0] == False
    assert report["any_fail"].iloc[0] == True
    assert report["diff_asset_growth_rate"].iloc[0] == pytest.approx(0.1)


def test_one_month_missing_actual() -> None:
    """Expected month not in actual => left join gives NaN actual => pass_* False, reason missing_actual."""
    policy = _policy(abs_tol=1e-6, rel_tol=1e-4)
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "asset_growth_rate": [0.10, 0.11],
        "organic_growth_rate": [0.05, 0.05],
        "external_market_growth_rate": [0.02, 0.02],
    })
    actual = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-02-29")],
        "asset_growth_rate": [0.11],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    report = build_validation_report(expected, actual, policy)
    assert len(report) == 2
    row0 = report[report["month_end"] == pd.Timestamp("2024-01-31")].iloc[0]
    assert row0["reason"] == "missing_actual"
    assert row0["pass_asset_growth_rate"] == False
    assert row0["pass_organic_growth_rate"] == False
    assert row0["pass_external_market_growth_rate"] == False
    assert row0["all_pass"] == False
    assert pd.isna(row0["actual_asset_growth_rate"])
    row1 = report[report["month_end"] == pd.Timestamp("2024-02-29")].iloc[0]
    assert row1["reason"] == ""
    assert row1["all_pass"] == True


def test_report_ordering_and_dtype() -> None:
    """Report sorted by month_end asc; month_end is datetime64[ns]."""
    policy = _policy()
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-01-31")],
        "asset_growth_rate": [0.1, 0.1],
        "organic_growth_rate": [0.05, 0.05],
        "external_market_growth_rate": [0.02, 0.02],
    })
    actual = expected.copy()
    report = build_validation_report(expected, actual, policy)
    assert report["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert report["month_end"].iloc[1] == pd.Timestamp("2024-03-31")
    assert report["month_end"].dtype == "datetime64[ns]"


def test_write_validation_report(tmp_path: Path) -> None:
    """write_validation_report creates qa/validation_report.csv."""
    policy = _policy()
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "asset_growth_rate": [0.1],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.02],
    })
    report = build_validation_report(expected, expected.copy(), policy)
    path = write_validation_report(report, root=tmp_path)
    assert path == tmp_path / "qa" / "validation_report.csv"
    assert path.exists()
    loaded = pd.read_csv(path)
    assert "month_end" in loaded.columns
    assert "all_pass" in loaded.columns
    assert "pass_asset_growth_rate" in loaded.columns


def test_highlighted_via_list_marks_correct_rows() -> None:
    """mode=list: parse months to month-end; highlighted True only for those rows."""
    policy = _policy(
        highlighted=HighlightedConfig(mode="list", column=None, months=["2024-02-29", "2024-01-15"])
    )
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")],
        "asset_growth_rate": [0.1, 0.11, 0.12],
        "organic_growth_rate": [0.05, 0.05, 0.05],
        "external_market_growth_rate": [0.02, 0.02, 0.02],
    })
    actual = expected.copy()
    report = build_validation_report(expected, actual, policy)
    assert list(report["highlighted"].values) == [True, True, False]
    assert list(report["drives_fail_fast"].values) == [True, True, False]
    assert "fail_fast_any_fail" in report.columns


def test_highlighted_via_column_y_n_1_0_true_false() -> None:
    """mode=column: coerce Y/N, 1/0, True/False to bool for highlighted."""
    policy = _policy(
        highlighted=HighlightedConfig(mode="column", column="Highlighted", months=None)
    )
    expected = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")],
        "asset_growth_rate": [0.1, 0.11, 0.12],
        "organic_growth_rate": [0.05, 0.05, 0.05],
        "external_market_growth_rate": [0.02, 0.02, 0.02],
        "Highlighted": ["Y", "N", 1],
    })
    actual = expected[["month_end", "asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]].copy()
    report = build_validation_report(expected, actual, policy)
    assert list(report["highlighted"].values) == [True, False, True]
    report2 = build_validation_report(
        expected.assign(Highlighted=[True, False, 0]),
        actual,
        _policy(highlighted=HighlightedConfig(mode="column", column="Highlighted", months=None)),
    )
    assert list(report2["highlighted"].values) == [True, False, False]
