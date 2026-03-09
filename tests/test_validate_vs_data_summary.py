"""
Unit tests for pipelines.qa.validate_vs_data_summary: evaluate_fail_fast.
Covers each trigger rule and a PASS scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from legacy.legacy_pipelines.qa.validate_vs_data_summary import (
    evaluate_fail_fast,
    format_failure_message,
    format_pass_message,
    write_fail_examples,
)

RATES = ["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]


def _policy(
    max_fail_months: int = 2,
    max_abs_err: float = 0.02,
    max_rel_err: float = 0.05,
    fail_on_missing_months: bool = True,
) -> ValidationPolicy:
    return ValidationPolicy(
        workbook=WorkbookConfig(path="", sheet="", month_column="Month", month_format=None),
        expected_columns=ExpectedColumnsConfig(
            asset_growth_rate="a", organic_growth_rate="b", external_market_growth_rate="c"
        ),
        normalization=NormalizationConfig(
            percent_to_decimal=False, percent_scale=100.0, month_align="month_end", timezone_naive=True
        ),
        tolerance=ToleranceConfig(abs_tol=0.0, rel_tol=0.0),
        fail_fast=FailFastConfig(
            max_mismatched_months=10,
            max_deviation=1.0,
            fail_on_missing_months=fail_on_missing_months,
            max_fail_months=max_fail_months,
            max_abs_err=max_abs_err,
            max_rel_err=max_rel_err,
        ),
        highlighted=DEFAULT_HIGHLIGHTED,
    )


def _minimal_report_df(
    month_ends: list[str],
    drives_fail_fast: list[bool],
    any_fail: list[bool],
    abs_err: float = 0.0,
    rel_err: float = 0.0,
    expected_val: float = 0.1,
    actual_val: float | None = 0.1,
) -> pd.DataFrame:
    """Build a minimal report with one metric (asset_growth_rate) for simplicity; duplicate to all RATES."""
    n = len(month_ends)
    data = {
        "month_end": pd.to_datetime(month_ends),
        "drives_fail_fast": drives_fail_fast,
        "any_fail": any_fail,
    }
    for m in RATES:
        data[f"expected_{m}"] = [expected_val] * n
        data[f"actual_{m}"] = [actual_val if actual_val is not None else np.nan] * n
        data[f"abs_err_{m}"] = [abs_err] * n
        data[f"rel_err_{m}"] = [rel_err] * n
        data[f"pass_{m}"] = [not any_fail[i] for i in range(n)]
    return pd.DataFrame(data)


def _report_df_with_diff_columns(month_ends: list[str], drives_fail_fast: list[bool], abs_err_by_metric: dict[str, list[float]]) -> pd.DataFrame:
    """Build report with diff/abs_err/rel_err per metric (for write_fail_examples tests)."""
    n = len(month_ends)
    data = {
        "month_end": pd.to_datetime(month_ends),
        "drives_fail_fast": drives_fail_fast,
        "any_fail": [True] * n,
    }
    for m in RATES:
        abs_vals = abs_err_by_metric.get(m, [0.0] * n)
        if len(abs_vals) != n:
            abs_vals = abs_vals + [0.0] * (n - len(abs_vals))
        data[f"expected_{m}"] = [0.1] * n
        data[f"actual_{m}"] = [0.1 + abs_vals[i] for i in range(n)]
        data[f"diff_{m}"] = [abs_vals[i] for i in range(n)]
        data[f"abs_err_{m}"] = abs_vals
        data[f"rel_err_{m}"] = [a / 0.1 if 0.1 != 0 else 0.0 for a in abs_vals]
        data[f"pass_{m}"] = [False] * n
    return pd.DataFrame(data)


def test_evaluate_fail_fast_pass_scenario() -> None:
    """Highlighted months all pass => triggered=False."""
    report = _minimal_report_df(
        month_ends=["2024-01-31", "2024-02-29"],
        drives_fail_fast=[True, True],
        any_fail=[False, False],
        abs_err=0.001,
        rel_err=0.01,
    )
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05)
    out = evaluate_fail_fast(report, policy)
    assert out["triggered"] is False
    assert out["reasons"] == []
    assert out["failing_months"] == []
    assert "asset_growth_rate" in out["worst"]
    assert out["worst"]["asset_growth_rate"]["abs_err"] == 0.001


def test_evaluate_fail_fast_trigger_fail_months() -> None:
    """fail_months > max_fail_months => triggered."""
    report = _minimal_report_df(
        month_ends=["2024-01-31", "2024-02-29", "2024-03-31"],
        drives_fail_fast=[True, True, True],
        any_fail=[True, True, True],  # 3 failing months
        abs_err=0.001,
        rel_err=0.01,
    )
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05)
    out = evaluate_fail_fast(report, policy)
    assert out["triggered"] is True
    assert any("fail_months" in r for r in out["reasons"])
    assert len(out["failing_months"]) == 3


def test_evaluate_fail_fast_trigger_worst_abs_err() -> None:
    """worst_abs_err_overall > max_abs_err => triggered."""
    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[False],
        abs_err=0.03,
        rel_err=0.01,
    )
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05)
    out = evaluate_fail_fast(report, policy)
    assert out["triggered"] is True
    assert any("worst_abs_err" in r for r in out["reasons"])


def test_evaluate_fail_fast_trigger_worst_rel_err() -> None:
    """worst_rel_err_overall > max_rel_err => triggered."""
    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[False],
        abs_err=0.001,
        rel_err=0.10,
    )
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05)
    out = evaluate_fail_fast(report, policy)
    assert out["triggered"] is True
    assert any("worst_rel_err" in r for r in out["reasons"])


def test_evaluate_fail_fast_trigger_missing_months() -> None:
    """fail_on_missing_months and missing actual in highlighted => triggered."""
    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[True],
        actual_val=None,
    )
    report["abs_err_asset_growth_rate"] = 0.0
    report["rel_err_asset_growth_rate"] = 0.0
    policy = _policy(fail_on_missing_months=True)
    out = evaluate_fail_fast(report, policy)
    assert out["triggered"] is True
    assert any("missing" in r.lower() for r in out["reasons"])


def test_evaluate_fail_fast_nan_handling_no_trigger_on_abs_rel() -> None:
    """All NaN in abs/rel err (highlighted only) => do not trigger on abs/rel rules; only missing can trigger."""
    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[False],
        abs_err=0.0,
        rel_err=0.0,
    )
    report["abs_err_asset_growth_rate"] = np.nan
    report["rel_err_asset_growth_rate"] = np.nan
    report["abs_err_organic_growth_rate"] = np.nan
    report["rel_err_organic_growth_rate"] = np.nan
    report["abs_err_external_market_growth_rate"] = np.nan
    report["rel_err_external_market_growth_rate"] = np.nan
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05, fail_on_missing_months=False)
    out = evaluate_fail_fast(report, policy)
    # Should not trigger: no fail_months over limit, no numeric worst > threshold, no missing
    assert out["triggered"] is False
    assert not any("worst_abs_err" in r for r in out["reasons"])
    assert not any("worst_rel_err" in r for r in out["reasons"])


def test_evaluate_fail_fast_uses_only_highlighted() -> None:
    """Only highlighted rows (drives_fail_fast == True) count; non-highlighted failures ignored."""
    report = _minimal_report_df(
        month_ends=["2024-01-31", "2024-02-29", "2024-03-31"],
        drives_fail_fast=[True, False, False],
        any_fail=[False, True, True],
        abs_err=0.001,
        rel_err=0.01,
    )
    policy = _policy(max_fail_months=2, max_abs_err=0.02, max_rel_err=0.05)
    out = evaluate_fail_fast(report, policy)
    # Only 1 highlighted row and it passes => no trigger
    assert out["triggered"] is False
    assert len(out["failing_months"]) == 0


def test_evaluate_fail_fast_worst_structure() -> None:
    """worst contains per-metric month_end, abs_err, rel_err, expected, actual."""
    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[False],
        abs_err=0.01,
        rel_err=0.02,
        expected_val=0.10,
        actual_val=0.11,
    )
    policy = _policy()
    out = evaluate_fail_fast(report, policy)
    for m in RATES:
        w = out["worst"][m]
        assert "month_end" in w and "abs_err" in w and "rel_err" in w and "expected" in w and "actual" in w
        assert w["abs_err"] == 0.01
        assert w["rel_err"] == 0.02
        assert w["expected"] == 0.10
        assert w["actual"] == 0.11


def test_evaluate_fail_fast_missing_columns_raises() -> None:
    """Missing required columns => ValueError."""
    report = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-31")], "drives_fail_fast": [True], "any_fail": [False]})
    policy = _policy()
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_fail_fast(report, policy)


# --- format_failure_message / format_pass_message ---


def test_format_failure_message_contains_required_fields() -> None:
    """Failure message includes VALIDATION FAILED, reasons, failing metrics, failing months, worst, QA pointers."""
    result = {
        "triggered": True,
        "reasons": ["fail_months (3) > max_fail_months (2)", "worst_abs_err_overall (0.03) > max_abs_err (0.02)"],
        "failing_months": ["2024-01-31", "2024-02-29", "2024-03-31"],
        "failing_metrics": ["asset_growth_rate", "organic_growth_rate"],
        "worst": {
            "asset_growth_rate": {
                "month_end": "2024-02-29",
                "abs_err": 0.03,
                "rel_err": 0.05,
                "expected": 0.10,
                "actual": 0.13,
            },
            "organic_growth_rate": {"month_end": "2024-01-31", "abs_err": 0.01, "rel_err": 0.02, "expected": 0.05, "actual": 0.06},
            "external_market_growth_rate": {"month_end": None, "abs_err": None, "rel_err": None, "expected": None, "actual": None},
        },
    }
    msg = format_failure_message(result, report_paths=None)
    assert "VALIDATION FAILED" in msg
    assert "fail_months (3) > max_fail_months (2)" in msg
    assert "worst_abs_err_overall" in msg
    assert "Failing metrics:" in msg
    assert "asset_growth_rate" in msg
    assert "organic_growth_rate" in msg
    assert "Failing month_end:" in msg
    assert "2024-01-31" in msg
    assert "2024-02-29" in msg
    assert "Worst deltas per metric:" in msg
    assert "month_end=2024-02-29" in msg
    assert "abs_err=0.03" in msg
    assert "rel_err=0.05" in msg
    assert "expected=0.1" in msg or "expected=0.10" in msg
    assert "actual=0.13" in msg
    assert "qa/validation_report.csv" in msg
    assert "qa/validation_summary.json" in msg
    assert "qa/validation_fail_examples.csv" in msg


def test_format_failure_message_failing_months_truncated_with_more() -> None:
    """When failing_months > 10, message shows first 10 and '+N more'."""
    result = {
        "triggered": True,
        "reasons": ["fail_months (12) > max_fail_months (2)"],
        "failing_months": [f"2024-{m:02d}-01" for m in range(1, 13)],
        "failing_metrics": ["asset_growth_rate"],
        "worst": {
            "asset_growth_rate": {"month_end": "2024-01-01", "abs_err": 0.1, "rel_err": 0.2, "expected": 0.5, "actual": 0.6},
            "organic_growth_rate": {"month_end": None, "abs_err": None, "rel_err": None, "expected": None, "actual": None},
            "external_market_growth_rate": {"month_end": None, "abs_err": None, "rel_err": None, "expected": None, "actual": None},
        },
    }
    msg = format_failure_message(result)
    assert "VALIDATION FAILED" in msg
    assert "+2 more" in msg
    assert "first 10" in msg


def test_format_failure_message_uses_report_paths_override() -> None:
    """Custom report_paths appear in the message."""
    result = {"triggered": True, "reasons": [], "failing_months": [], "failing_metrics": [], "worst": {}}
    paths = {
        "validation_report": "out/qa/validation_report.csv",
        "validation_summary": "out/qa/validation_summary.json",
        "validation_fail_examples": "out/qa/validation_fail_examples.csv",
    }
    msg = format_failure_message(result, report_paths=paths)
    assert "out/qa/validation_report.csv" in msg
    assert "out/qa/validation_summary.json" in msg
    assert "out/qa/validation_fail_examples.csv" in msg


def test_format_pass_message_contains_required_fields() -> None:
    """Pass message includes VALIDATION PASSED, highlighted count, fail months count, worst abs/rel."""
    result = {
        "triggered": False,
        "highlighted_count": 5,
        "fail_months_count": 0,
        "worst_abs_err_overall": 0.001,
        "worst_rel_err_overall": 0.01,
    }
    msg = format_pass_message(result)
    assert "VALIDATION PASSED" in msg
    assert "Highlighted months: 5" in msg
    assert "Fail months count: 0" in msg
    assert "Worst abs_err" in msg
    assert "0.001" in msg
    assert "Worst rel_err" in msg
    assert "0.01" in msg


def test_format_pass_message_from_evaluate_result() -> None:
    """Pass message built from actual evaluate_fail_fast return has all fields."""
    report = _minimal_report_df(
        month_ends=["2024-01-31", "2024-02-29"],
        drives_fail_fast=[True, True],
        any_fail=[False, False],
        abs_err=0.001,
        rel_err=0.002,
    )
    policy = _policy()
    result = evaluate_fail_fast(report, policy)
    msg = format_pass_message(result)
    assert "VALIDATION PASSED" in msg
    assert "Highlighted months: 2" in msg
    assert "Fail months count: 0" in msg
    assert "0.001" in msg
    assert "0.002" in msg


# --- write_fail_examples ---


def test_write_fail_examples_ordering_and_metric_blocks(tmp_path: Path) -> None:
    """write_fail_examples outputs three metric blocks, sorted by abs_err desc (NaNs last), month_end ISO."""
    # 4 highlighted months; abs_err for asset_growth_rate: 0.01, 0.05, 0.10, 0.02 (so order: 0.10, 0.05, 0.02, 0.01)
    report = _report_df_with_diff_columns(
        month_ends=["2024-03-31", "2024-01-31", "2024-04-30", "2024-02-29"],
        drives_fail_fast=[True, True, True, True],
        abs_err_by_metric={
            "asset_growth_rate": [0.01, 0.05, 0.10, 0.02],
            "organic_growth_rate": [0.10, 0.02, 0.01, 0.05],
            "external_market_growth_rate": [0.02, 0.02, 0.02, 0.02],
        },
    )
    out_csv = tmp_path / "validation_fail_examples.csv"
    write_fail_examples(report, out_path=str(out_csv), top_n=2)

    df = pd.read_csv(out_csv)
    assert list(df.columns) == [
        "metric", "month_end", "expected", "actual", "diff", "abs_err", "rel_err", "pass", "drives_fail_fast"
    ]
    # Three metric blocks (order: asset, organic, external)
    metrics = df["metric"].tolist()
    assert metrics == [
        "asset_growth_rate", "asset_growth_rate",
        "organic_growth_rate", "organic_growth_rate",
        "external_market_growth_rate", "external_market_growth_rate",
    ]
    # Within asset_growth_rate block: abs_err descending -> 0.10, 0.05
    asset = df[df["metric"] == "asset_growth_rate"]
    assert list(asset["abs_err"].values) == pytest.approx([0.10, 0.05])
    # Within organic_growth_rate block: abs_err descending -> 0.10, 0.05 (top two)
    organic = df[df["metric"] == "organic_growth_rate"]
    assert list(organic["abs_err"].values) == pytest.approx([0.10, 0.05])
    # month_end ISO format YYYY-MM-DD
    for me in df["month_end"]:
        assert len(me) == 10 and me[4] == "-" and me[7] == "-"


# --- CLI main() smoke test ---

MINIMAL_POLICY_YAML = """
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


def test_cli_main_pass_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI main(): with existing report that passes, exit 0 and print VALIDATION PASSED (no subprocess)."""
    from legacy.legacy_pipelines.qa.validate_vs_data_summary import main

    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "qa").mkdir(parents=True)
    (tmp_path / "configs" / "validation_policy.yml").write_text(MINIMAL_POLICY_YAML, encoding="utf-8")

    report = _minimal_report_df(
        month_ends=["2024-01-31"],
        drives_fail_fast=[True],
        any_fail=[False],
        abs_err=0.0,
        rel_err=0.0,
    )
    report["diff_asset_growth_rate"] = 0.0
    report["diff_organic_growth_rate"] = 0.0
    report["diff_external_market_growth_rate"] = 0.0
    report.to_csv(tmp_path / "qa" / "validation_report.csv", index=False, date_format="%Y-%m-%d")

    import sys
    old_argv = sys.argv
    try:
        sys.argv = [
            "",
            "--root", str(tmp_path),
            "--policy", "configs/validation_policy.yml",
            "--report", "qa/validation_report.csv",
        ]
        exit_code = main()
    finally:
        sys.argv = old_argv

    assert exit_code == 0
    out, err = capsys.readouterr()
    assert "VALIDATION PASSED" in out
    assert "Highlighted months:" in out
