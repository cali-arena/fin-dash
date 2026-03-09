"""
Tests for qa/validate_vs_data_summary: DATA SUMMARY formulas and MISSING_DATA recategorization.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Run from repo root so qa and app are importable
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.validate_vs_data_summary import SKIP_INCOMPLETE_COVERAGE, run as validation_run


def test_validation_uses_data_summary_formulas(tmp_path: Path) -> None:
    """Validation run uses compute_firm_rates_df; firm rates match DATA SUMMARY formulas."""
    curated = tmp_path / "curated"
    curated.mkdir()
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    # One month with valid begin/end/nnb
    metrics = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-02-28")],
        "begin_aum": [1000.0],
        "end_aum": [1100.0],
        "nnb": [50.0],
        "nnf": [10.0],
        "channel": ["A"],
        "product_ticker": ["X"],
        "src_country": ["US"],
        "segment": ["S"],
        "sub_segment": ["S1"],
    })
    metrics.to_parquet(curated / "metrics_monthly.parquet", index=False)
    # Summary expects same formula: asset_growth = 0.1, ogr = 0.05, external = (1100-1000-50)/1000 = 0.05
    summary = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-02-28")],
        "asset_growth_rate": [0.1],
        "organic_growth_rate": [0.05],
        "external_growth_rate": [0.05],
    })
    summary.to_parquet(curated / "data_summary_normalized.parquet", index=False)
    n = validation_run(curated, qa_dir)
    assert n == 0
    report = pd.read_csv(qa_dir / "validation_report.csv")
    assert report["any_fail"].iloc[0] == False
    fr = report["fail_reason"].iloc[0]
    assert fr == "" or (isinstance(fr, float) and pd.isna(fr))


def test_validation_missing_data_when_begin_aum_zero(tmp_path: Path) -> None:
    """When begin_aum is 0 for a month, fail is recategorized to MISSING_DATA."""
    curated = tmp_path / "curated"
    curated.mkdir()
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    # One month with begin_aum=0 (first month)
    metrics = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "begin_aum": [0.0],
        "end_aum": [1000.0],
        "nnb": [20.0],
        "nnf": [5.0],
        "channel": ["A"],
        "product_ticker": ["X"],
        "src_country": ["US"],
        "segment": ["S"],
        "sub_segment": ["S1"],
    })
    metrics.to_parquet(curated / "metrics_monthly.parquet", index=False)
    summary = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "asset_growth_rate": [0.5],
        "organic_growth_rate": [0.1],
        "external_growth_rate": [0.0],
    })
    summary.to_parquet(curated / "data_summary_normalized.parquet", index=False)
    n = validation_run(curated, qa_dir)
    report = pd.read_csv(qa_dir / "validation_report.csv")
    # Our calc will be NaN (begin=0), so abs err may be large and any_fail True
    # fail_reason should be MISSING_DATA because begin_aum <= 0
    fail_reason = str(report["fail_reason"].iloc[0])
    assert "MISSING_DATA" in fail_reason or report["fail_reason"].iloc[0] == "MISSING_DATA"


def test_validation_skip_incomplete_coverage_when_summary_has_month_not_in_firm(tmp_path: Path) -> None:
    """When summary has a month_end that firm (actual) does not have, row is marked SKIP_INCOMPLETE_COVERAGE."""
    curated = tmp_path / "curated"
    curated.mkdir()
    qa_dir = tmp_path / "qa"
    qa_dir.mkdir()
    # Firm has only Jan 2021; summary expects Jan and Feb (gap in actual)
    metrics = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "begin_aum": [0.0],
        "end_aum": [1000.0],
        "nnb": [20.0],
        "channel": ["A"],
        "product_ticker": ["X"],
        "src_country": ["US"],
        "segment": ["S"],
        "sub_segment": ["S1"],
    })
    metrics.to_parquet(curated / "metrics_monthly.parquet", index=False)
    summary = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-28")],
        "asset_growth_rate": [0.0, 0.1],
        "organic_growth_rate": [0.02, 0.05],
        "external_growth_rate": [0.0, 0.05],
    })
    summary.to_parquet(curated / "data_summary_normalized.parquet", index=False)
    n = validation_run(curated, qa_dir)
    report = pd.read_csv(qa_dir / "validation_report.csv")
    # Row for 2021-02-28: firm has no data -> skip_reason / fail_reason SKIP_INCOMPLETE_COVERAGE
    report["month_end_str"] = report["month_end"].astype(str)
    feb_row = report[report["month_end_str"].str.contains("2021-02")]
    assert len(feb_row) == 1, f"Expected one Feb row, got {len(feb_row)}. Report: {report}"
    assert (
        feb_row["skip_reason"].iloc[0] == SKIP_INCOMPLETE_COVERAGE
        or str(feb_row["fail_reason"].iloc[0]) == SKIP_INCOMPLETE_COVERAGE
    )
    assert feb_row["any_fail"].iloc[0] == False
