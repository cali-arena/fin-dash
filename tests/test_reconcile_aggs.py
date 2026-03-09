"""
Tests for pipelines.agg.reconcile_aggs: firm_monthly reconciliation (pass and fail).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.agg.reconcile_aggs import (
    QA_DIR,
    RECONCILE_FAIL_CSV,
    ReconcileError,
    load_reconcile_config,
    reconcile_firm_monthly,
)


def test_reconcile_pass_within_tolerance(tmp_path: Path) -> None:
    """When source and agg match per month (within abs_tol/rel_tol), reconciliation passes."""
    time_key = "month_end"
    measure = "end_aum"
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "end_aum": [100.0, 200.0, 500.0],
    })
    agg = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "end_aum": [300.0, 500.0],
    })
    reconcile_firm_monthly(
        source, agg, time_key, measure,
        abs_tol=1e-6, rel_tol=1e-6,
        root=tmp_path,
    )
    assert not (tmp_path / QA_DIR / RECONCILE_FAIL_CSV).exists()


def test_reconcile_fail_writes_csv_and_raises(tmp_path: Path) -> None:
    """When agg differs from source beyond tolerance, ReconcileError is raised and fail CSV is written."""
    time_key = "month_end"
    measure = "end_aum"
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "end_aum": [1000.0, 2000.0],
    })
    agg = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "end_aum": [1000.0 + 1.0, 2000.0],
    })
    with pytest.raises(ReconcileError) as exc_info:
        reconcile_firm_monthly(
            source, agg, time_key, measure,
            abs_tol=1e-6, rel_tol=1e-6,
            root=tmp_path,
        )
    assert "reconciliation failed" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()

    fail_path = tmp_path / QA_DIR / RECONCILE_FAIL_CSV
    assert fail_path.exists()
    df = pd.read_csv(fail_path)
    assert "month_end" in df.columns
    assert "source" in df.columns
    assert "agg" in df.columns
    assert "diff" in df.columns
    assert "abs_err" in df.columns
    assert "rel_err" in df.columns
    assert len(df) >= 1
    row = df.iloc[0]
    assert abs(float(row["diff"])) == 1.0
    assert float(row["abs_err"]) == 1.0


def test_load_reconcile_config_defaults(tmp_path: Path) -> None:
    """When no config file exists, defaults are returned."""
    cfg = load_reconcile_config(tmp_path)
    assert cfg["abs_tol"] == 1e-6
    assert cfg["rel_tol"] == 1e-6
    assert cfg["measure"] == "end_aum"


def test_load_reconcile_config_from_yml(tmp_path: Path) -> None:
    """Reconcile config is read from configs/agg_qa_policy.yml or agg_policy.yml."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "agg_qa_policy.yml").write_text(
        "reconcile:\n  abs_tol: 0.01\n  rel_tol: 0.02\n  measure: begin_aum\n",
        encoding="utf-8",
    )
    cfg = load_reconcile_config(tmp_path)
    assert cfg["abs_tol"] == 0.01
    assert cfg["rel_tol"] == 0.02
    assert cfg["measure"] == "begin_aum"
