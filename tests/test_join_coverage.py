"""
Tests for pipelines.agg.join_coverage: warning emission and fail on exceed threshold.
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

from legacy.legacy_pipelines.agg.join_coverage import (
    QA_DIR,
    WARNINGS_FILENAME,
    check_join_coverage,
    compute_join_coverage,
    load_join_coverage_config,
)
from legacy.legacy_pipelines.agg.join_coverage import JoinCoverageError


def test_warning_emission_when_above_warn_threshold(tmp_path: Path) -> None:
    """When pct_missing_channel_l1 > warn_threshold, qa/agg_join_coverage_warnings.json is written (no raise)."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "agg_qa_policy.yml").write_text(
        "join_coverage:\n  warn_threshold: 0.001\n  fail_threshold: 0.05\n",
        encoding="utf-8",
    )
    # 2 of 1000 = 0.2% missing channel_l1 > 0.1% warn
    df = pd.DataFrame({
        "preferred_label": ["a"] * 998 + ["b", "c"],
        "channel_l1": ["Retail"] * 998 + [pd.NA, pd.NA],
        "segment": ["S1"] * 1000,
        "product_ticker": ["X"] * 1000,
    })
    check_join_coverage(df, tmp_path / "configs" / "agg_qa_policy.yml", tmp_path)

    warn_path = tmp_path / QA_DIR / WARNINGS_FILENAME
    assert warn_path.exists()
    data = json.loads(warn_path.read_text())
    assert data["pct_missing_channel_l1"] == pytest.approx(0.002)
    assert data["n_missing_channel_l1"] == 2
    assert "sample_channel_l1_misses" in data
    assert len(data["sample_channel_l1_misses"]) >= 1


def test_fail_when_above_fail_threshold(tmp_path: Path) -> None:
    """When pct_missing_segment > fail_threshold, warnings file is written and JoinCoverageError is raised."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "agg_qa_policy.yml").write_text(
        "join_coverage:\n  warn_threshold: 0.001\n  fail_threshold: 0.05\n",
        encoding="utf-8",
    )
    # 60 of 1000 = 6% missing segment > 5% fail
    df = pd.DataFrame({
        "preferred_label": ["a"] * 1000,
        "channel_l1": ["Retail"] * 1000,
        "segment": ["S1"] * 940 + [pd.NA] * 60,
        "product_ticker": ["X"] * 940 + ["Y"] * 60,
    })
    with pytest.raises(JoinCoverageError) as exc_info:
        check_join_coverage(df, tmp_path / "configs" / "agg_qa_policy.yml", tmp_path)
    assert "fail_threshold" in str(exc_info.value)

    warn_path = tmp_path / QA_DIR / WARNINGS_FILENAME
    assert warn_path.exists()
    data = json.loads(warn_path.read_text())
    assert data["pct_missing_segment"] == pytest.approx(0.06)
    assert data["n_missing_segment"] == 60
    assert "sample_segment_misses" in data
    assert len(data["sample_segment_misses"]) >= 1


def test_compute_join_coverage_no_missing(tmp_path: Path) -> None:
    """When no missing channel_l1 or segment, pct are 0 and samples empty."""
    df = pd.DataFrame({
        "channel_l1": ["A"] * 10,
        "segment": ["S1"] * 10,
    })
    cov = compute_join_coverage(df)
    assert cov["pct_missing_channel_l1"] == 0.0
    assert cov["pct_missing_segment"] == 0.0
    assert cov["sample_channel_l1_misses"] == []
    assert cov["sample_segment_misses"] == []


def test_load_join_coverage_config_from_yml(tmp_path: Path) -> None:
    """Config is read from agg_qa_policy.yml with warn_threshold and fail_threshold."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "agg_qa_policy.yml").write_text(
        "join_coverage:\n  warn_threshold: 0.01\n  fail_threshold: 0.10\n",
        encoding="utf-8",
    )
    warn, fail = load_join_coverage_config(tmp_path, tmp_path / "configs" / "agg_qa_policy.yml")
    assert warn == 0.01
    assert fail == 0.10
