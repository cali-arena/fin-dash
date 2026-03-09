"""
Unit tests for pipelines.validation.qa_firm_level: identity check and guard alignment (synthetic DataFrames).
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

from legacy.legacy_pipelines.validation.qa_firm_level import (
    load_qa_policy,
    run_firm_qa,
    run_guard_check,
    run_identity_check,
    run_inf_nan_sweep,
    DEFAULT_QA_CONFIG,
)


def _synthetic_firm_df(
    identity_ok: bool = True,
    guard_ok: bool = True,
) -> pd.DataFrame:
    """Synthetic firm-level df: 2 months. If not identity_ok, one row violates end = begin + nnb + pnl."""
    if identity_ok:
        df = pd.DataFrame({
            "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "begin_aum_firm": [100.0, 200.0],
            "end_aum_firm": [110.0, 220.0],
            "nnb_firm": [5.0, 10.0],
            "market_pnl_firm": [5.0, 10.0],
            "asset_growth_rate": [0.1, 0.1],
            "organic_growth_rate": [0.05, 0.05],
            "external_market_growth_rate": [0.05, 0.05],
            "source": ["recomputed_from_metrics_monthly"] * 2,
            "global_slice_rowcount_per_month": [1, 1],
        })
    else:
        # end_aum_firm != begin + nnb + pnl for first row (e.g. 110 != 100+5+5 = 110, so make it 111)
        df = pd.DataFrame({
            "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "begin_aum_firm": [100.0, 200.0],
            "end_aum_firm": [111.0, 220.0],
            "nnb_firm": [5.0, 10.0],
            "market_pnl_firm": [5.0, 10.0],
            "asset_growth_rate": [0.1, 0.1],
            "organic_growth_rate": [0.05, 0.05],
            "external_market_growth_rate": [0.05, 0.05],
            "source": ["recomputed_from_metrics_monthly"] * 2,
            "global_slice_rowcount_per_month": [1, 1],
        })
    if not guard_ok:
        # begin_aum_firm <= 0 but rates non-NaN
        df = pd.DataFrame({
            "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "begin_aum_firm": [0.0, 200.0],
            "end_aum_firm": [0.0, 220.0],
            "nnb_firm": [0.0, 10.0],
            "market_pnl_firm": [0.0, 10.0],
            "asset_growth_rate": [0.0, 0.1],
            "organic_growth_rate": [0.0, 0.05],
            "external_market_growth_rate": [0.0, 0.05],
            "source": ["recomputed_from_metrics_monthly"] * 2,
            "global_slice_rowcount_per_month": [1, 1],
        })
    return df


def test_load_qa_policy_defaults() -> None:
    """When policy file missing, defaults returned."""
    config = load_qa_policy(Path("nonexistent/firm_qa_policy.yml"))
    assert config["identity"]["atol"] == 1e-6
    assert config["identity"]["tol_pct"] == 1e-6
    assert config["guard"]["threshold"] == 0.0
    assert "asset_growth_rate" in config["rate_columns"]


def test_identity_check_no_violations(tmp_path: Path) -> None:
    """Identity check with exact identity: 0 violations, no CSV."""
    df = _synthetic_firm_df(identity_ok=True, guard_ok=True)
    config = DEFAULT_QA_CONFIG.copy()
    n = run_identity_check(df, config, tmp_path)
    assert n == 0
    assert not (tmp_path / "firm_identity_violations.csv").exists()


def test_identity_check_with_violation(tmp_path: Path) -> None:
    """Identity check with one row violating end != begin+nnb+pnl: 1 violation, CSV written."""
    df = _synthetic_firm_df(identity_ok=False, guard_ok=True)
    config = DEFAULT_QA_CONFIG.copy()
    config["identity"]["atol"] = 1e-6
    config["identity"]["tol_pct"] = 1e-9
    n = run_identity_check(df, config, tmp_path)
    assert n == 1
    csv_path = tmp_path / "firm_identity_violations.csv"
    assert csv_path.exists()
    viol = pd.read_csv(csv_path)
    assert len(viol) == 1
    assert "resid" in viol.columns
    assert "abs_resid" in viol.columns


def test_guard_check_no_violations(tmp_path: Path) -> None:
    """Guard check when all guarded rows have NaN rates: 0 violations."""
    df = _synthetic_firm_df(identity_ok=True, guard_ok=True)
    # Set first row to begin_aum_firm=0 and rates to NaN (compliant)
    df.loc[0, "begin_aum_firm"] = 0.0
    df.loc[0, "asset_growth_rate"] = float("nan")
    df.loc[0, "organic_growth_rate"] = float("nan")
    df.loc[0, "external_market_growth_rate"] = float("nan")
    config = DEFAULT_QA_CONFIG.copy()
    n = run_guard_check(df, config, tmp_path)
    assert n == 0


def test_guard_check_with_violation(tmp_path: Path) -> None:
    """Guard check: begin_aum_firm <= 0 but rate non-NaN -> violation, CSV written."""
    df = _synthetic_firm_df(identity_ok=True, guard_ok=False)
    config = DEFAULT_QA_CONFIG.copy()
    n = run_guard_check(df, config, tmp_path)
    assert n == 1
    csv_path = tmp_path / "firm_guard_violations.csv"
    assert csv_path.exists()
    viol = pd.read_csv(csv_path)
    assert len(viol) == 1
    assert viol["begin_aum_firm"].iloc[0] == 0.0


def test_inf_nan_sweep(tmp_path: Path) -> None:
    """Inf/NaN sweep writes firm_nan_summary.json with nan_counts; has_inf if any inf."""
    df = _synthetic_firm_df(identity_ok=True, guard_ok=True)
    df.loc[0, "organic_growth_rate"] = float("nan")
    config = DEFAULT_QA_CONFIG.copy()
    summary = run_inf_nan_sweep(df, config, tmp_path)
    assert summary["rowcount"] == 2
    assert summary["nan_counts"]["organic_growth_rate"] == 1
    assert summary["nan_counts"]["asset_growth_rate"] == 0
    assert summary["has_inf"] is False
    json_path = tmp_path / "firm_nan_summary.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["nan_counts"]["organic_growth_rate"] == 1


def test_run_firm_qa_with_df(tmp_path: Path) -> None:
    """run_firm_qa with provided df runs all checks and writes artifacts."""
    df = _synthetic_firm_df(identity_ok=True, guard_ok=True)
    result = run_firm_qa(df=df, root=tmp_path)
    assert "identity_violations" in result
    assert "guard_violations" in result
    assert "nan_summary" in result
    assert result["identity_violations"] == 0
    assert result["guard_violations"] == 0
    assert (tmp_path / "curated" / "qa" / "firm_nan_summary.json").exists()
