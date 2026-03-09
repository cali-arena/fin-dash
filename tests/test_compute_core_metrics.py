"""
Unit tests for pipelines.metrics.compute_core_metrics: contract enforcement and metric computation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.metrics.compute_core_metrics import (
    _enforce_contract,
    compute_core_metrics,
    GRAIN_COLS,
    COL_BEGIN_AUM,
    COL_END_AUM,
    COL_NNB,
    COL_NNF,
    COL_MARKET_PNL,
    COL_OGR,
    COL_FEE_YIELD,
    COL_SLICE_KEY,
)


def _minimal_input(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "path_id": ["p1"] * n,
        "slice_id": ["s1"] * n,
        "month_end": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"][:n]),
        "slice_key": ["k1"] * n,
        "begin_aum": [100.0, 110.0, 120.0][:n],
        "end_aum": [110.0, 120.0, 130.0][:n],
        "nnb": [5.0, 5.0, 5.0][:n],
        "nnf": [0.5, 0.6, 0.7][:n],
    })


def _minimal_policy() -> dict:
    return {
        "version": 1,
        "inputs": {"required_columns": ["begin_aum", "end_aum", "nnb", "nnf"], "grain_required": GRAIN_COLS},
        "policies": {
            "begin_aum_guard": {"mode": "nan", "threshold": 0.0, "applies_to_rates": ["ogr", "market_impact_rate", "total_growth_rate"]},
            "fee_yield_guard": {"mode": "nan", "threshold": 0.0, "cap_value": 0.0},
            "inf_handling": {"mode": "nan"},
            "clamp": {"enabled": True, "mode": "warn_only", "caps": {"ogr": {"min": -2.0, "max": 2.0}, "market_impact_rate": {"min": -2.0, "max": 2.0}, "total_growth_rate": {"min": -2.0, "max": 2.0}, "fee_yield": {"min": 0.0, "max": 0.2}}},
        },
        "audit": {},
    }


def test_compute_core_metrics_output_columns() -> None:
    df = _minimal_input()
    policy = _minimal_policy()
    out, qa = compute_core_metrics(df, policy)
    assert list(out.columns) == [
        "path_id", "slice_id", "month_end", "slice_key",
        "begin_aum", "end_aum", "nnb", "nnf", "market_pnl",
        "ogr", "market_impact_rate", "total_growth_rate", "fee_yield",
        "ogr_clamped_flag", "market_impact_rate_clamped_flag", "total_growth_rate_clamped_flag", "fee_yield_clamped_flag",
    ]
    assert len(out) == 3


def test_compute_core_metrics_market_pnl() -> None:
    df = _minimal_input(1)
    df["begin_aum"] = [100.0]
    df["end_aum"] = [110.0]
    df["nnb"] = [5.0]
    policy = _minimal_policy()
    out, _ = compute_core_metrics(df, policy)
    assert out["market_pnl"].iloc[0] == 5.0  # 110 - 100 - 5


def test_compute_core_metrics_qa_effects_structure() -> None:
    df = _minimal_input()
    policy = _minimal_policy()
    _, qa = compute_core_metrics(df, policy)
    assert "guard_nan_counts" in qa
    assert "inf_to_nan_count" in qa
    assert "clamp_counts" in qa
    assert qa["guard_nan_counts"][COL_OGR] >= 0
    assert qa["clamp_counts"][COL_OGR] >= 0


def test_enforce_contract_pass() -> None:
    df = _minimal_input()
    policy = _minimal_policy()
    _enforce_contract(df, policy)


def test_enforce_contract_missing_column() -> None:
    df = _minimal_input().drop(columns=["nnb"])
    policy = _minimal_policy()
    with pytest.raises(ValueError, match="missing required column"):
        _enforce_contract(df, policy)


def test_enforce_contract_duplicate_grain() -> None:
    df = _minimal_input(2)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    policy = _minimal_policy()
    with pytest.raises(ValueError, match="unique"):
        _enforce_contract(df, policy)


def test_metrics_policy_gate_passes_on_computed_metrics() -> None:
    """Run gate on metrics produced by compute_core_metrics; gate should pass."""
    import tempfile
    from legacy.legacy_pipelines.contracts.metrics_policy_contract import load_metrics_policy, validate_metrics_policy
    from legacy.legacy_pipelines.metrics.metrics_policy_gate import run_gate

    policy_path = PROJECT_ROOT / "configs/metrics_policy.yml"
    if not policy_path.exists():
        pytest.skip("configs/metrics_policy.yml not found")
    policy = validate_metrics_policy(load_metrics_policy(policy_path))
    df = _minimal_input()
    out, _ = compute_core_metrics(df, policy)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    try:
        out.to_parquet(tmp, index=False)
        report = run_gate(PROJECT_ROOT, metrics_path=Path(tmp), policy_config_path=policy_path)
        assert report.get("passed") is True, report
    finally:
        Path(tmp).unlink(missing_ok=True)
