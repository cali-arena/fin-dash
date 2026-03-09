"""
Unit tests for pipelines.validation.recompute_firm_level: filter_global_slice (all modes),
recompute_firm_level (aggregation, guards, ordering, float64).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.validation.recompute_firm_level import (
    filter_global_slice,
    recompute_firm_level,
)


def test_filter_global_slice_path_id() -> None:
    df = pd.DataFrame({
        "path_id": ["global", "other", "global"],
        "slice_id": ["s1", "s2", "s3"],
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "begin_aum": [100.0, 200.0, 150.0],
    })
    out = filter_global_slice(df, {"mode": "path_id", "column": "path_id", "value": "global"})
    assert len(out) == 2
    assert out["path_id"].tolist() == ["global", "global"]


def test_filter_global_slice_flag() -> None:
    df = pd.DataFrame({
        "is_global": [True, False, True],
        "month_end": [pd.Timestamp("2024-01-31")] * 3,
        "begin_aum": [10.0, 20.0, 30.0],
    })
    out = filter_global_slice(df, {"mode": "flag", "column": "is_global", "value": True})
    assert len(out) == 2
    assert out["is_global"].all()


def test_filter_global_slice_null_keys() -> None:
    df = pd.DataFrame({
        "path_id": ["", "p1", None],
        "slice_id": [None, "s1", ""],
        "month_end": [pd.Timestamp("2024-01-31")] * 3,
        "begin_aum": [10.0, 20.0, 30.0],
    })
    out = filter_global_slice(df, {"mode": "null_keys", "keys": ["path_id", "slice_id"]})
    assert len(out) == 2
    assert out["begin_aum"].tolist() == [10.0, 30.0]


def test_filter_global_slice_default_selector() -> None:
    df = pd.DataFrame({"path_id": ["global", "other"], "month_end": [pd.Timestamp("2024-01-31")] * 2})
    out = filter_global_slice(df, None)
    assert len(out) == 1
    assert out["path_id"].iloc[0] == "global"


def test_filter_global_slice_empty_df() -> None:
    df = pd.DataFrame(columns=["path_id", "month_end"])
    out = filter_global_slice(df, {"mode": "path_id", "column": "path_id", "value": "global"})
    assert out.empty


FIRM_LEVEL_OUTPUT_COLUMNS = [
    "month_end",
    "begin_aum_firm",
    "end_aum_firm",
    "nnb_firm",
    "market_pnl_firm",
    "asset_growth_rate",
    "organic_growth_rate",
    "external_market_growth_rate",
    "source",
    "global_slice_rowcount_per_month",
]


def test_recompute_firm_level_aggregation_and_rates() -> None:
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "begin_aum": [100.0, 200.0, 150.0],
        "end_aum": [110.0, 220.0, 165.0],
        "nnb": [5.0, 10.0, 7.5],
        "market_pnl": [5.0, 10.0, 7.5],
    })
    policy = {"policies": {"begin_aum_guard": {"mode": "nan", "threshold": 0.0}}}
    out = recompute_firm_level(df, policy)
    assert list(out.columns) == FIRM_LEVEL_OUTPUT_COLUMNS
    assert len(out) == 2
    assert out["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert out["month_end"].iloc[1] == pd.Timestamp("2024-02-29")
    assert out["begin_aum_firm"].iloc[0] == 300.0
    assert out["end_aum_firm"].iloc[0] == 330.0
    assert out["nnb_firm"].iloc[0] == 15.0
    assert out["market_pnl_firm"].iloc[0] == 15.0
    assert out["source"].iloc[0] == "recomputed_from_metrics_monthly"
    assert out["global_slice_rowcount_per_month"].iloc[0] == 2
    assert out["global_slice_rowcount_per_month"].iloc[1] == 1
    # asset_growth_rate = (330-300)/300 = 0.1
    assert out["asset_growth_rate"].iloc[0] == pytest.approx(0.1)
    assert out["organic_growth_rate"].iloc[0] == pytest.approx(15.0 / 300.0)
    assert out["external_market_growth_rate"].iloc[0] == pytest.approx(15.0 / 300.0)
    assert out["asset_growth_rate"].dtype == "float64"
    assert out["month_end"].dtype == "datetime64[ns]"
    assert out["month_end"].is_unique


def test_recompute_firm_level_guard_zero_begin_aum() -> None:
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum": [0.0],
        "end_aum": [0.0],
        "nnb": [0.0],
        "market_pnl": [0.0],
    })
    policy = {"policies": {"begin_aum_guard": {"mode": "nan", "threshold": 0.0}}}
    out = recompute_firm_level(df, policy)
    assert len(out) == 1
    assert pd.isna(out["asset_growth_rate"].iloc[0])
    assert pd.isna(out["organic_growth_rate"].iloc[0])
    assert pd.isna(out["external_market_growth_rate"].iloc[0])


def test_recompute_firm_level_ordering_month_end_asc() -> None:
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-03-31"), pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "begin_aum": [100.0, 100.0, 100.0],
        "end_aum": [100.0, 100.0, 100.0],
        "nnb": [0.0, 0.0, 0.0],
        "market_pnl": [0.0, 0.0, 0.0],
    })
    policy = {"policies": {"begin_aum_guard": {"mode": "nan", "threshold": 0.0}}}
    out = recompute_firm_level(df, policy)
    assert out["month_end"].iloc[0] == pd.Timestamp("2024-01-31")
    assert out["month_end"].iloc[1] == pd.Timestamp("2024-02-29")
    assert out["month_end"].iloc[2] == pd.Timestamp("2024-03-31")


def test_recompute_firm_level_missing_columns_raises() -> None:
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-31")], "begin_aum": [100.0]})
    policy = {}
    with pytest.raises(ValueError, match="required columns missing"):
        recompute_firm_level(df, policy)


def test_load_metrics_monthly_missing_returns_empty(tmp_path: Path) -> None:
    from legacy.legacy_pipelines.validation.recompute_firm_level import load_metrics_monthly
    out = load_metrics_monthly(tmp_path / "nonexistent.parquet")
    assert out.empty


def test_write_firm_level_output(tmp_path: Path) -> None:
    from legacy.legacy_pipelines.validation.recompute_firm_level import write_firm_level_output
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum_firm": [300.0],
        "end_aum_firm": [330.0],
        "nnb_firm": [15.0],
        "market_pnl_firm": [15.0],
        "asset_growth_rate": [0.1],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.05],
        "source": ["recomputed_from_metrics_monthly"],
        "global_slice_rowcount_per_month": [1],
    })
    policy = {"policies": {}}
    write_firm_level_output(df, tmp_path, policy, dataset_version="test-v1")
    parquet_path = tmp_path / "curated" / "qa" / "firm_level_recomputed.parquet"
    meta_path = tmp_path / "curated" / "qa" / "firm_level_recomputed.meta.json"
    assert parquet_path.exists()
    assert meta_path.exists()
    import json
    meta = json.loads(meta_path.read_text())
    assert meta["rowcount"] == 1
    assert meta["dataset_version"] == "test-v1"
    assert "policy_hash" in meta
    assert "min_month_end" in meta
    assert "max_month_end" in meta


def test_cache_hit_skips_recompute(tmp_path: Path) -> None:
    """When data/.version.json exists and cache has matching policy_hash, recompute is skipped and curated/qa updated."""
    import json
    import subprocess
    import sys
    from legacy.legacy_pipelines.validation.recompute_firm_level import (
        _policy_hash,
        _read_dataset_version,
        _write_firm_level_to_dir,
    )
    from legacy.legacy_pipelines.contracts.metrics_policy_contract import load_metrics_policy, validate_metrics_policy

    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / ".version.json").write_text(
        json.dumps({"dataset_version": "test-dv-cache", "pipeline_version": "p1"}), encoding="utf-8"
    )
    assert _read_dataset_version(tmp_path) == "test-dv-cache"

    policy_path = PROJECT_ROOT / "configs" / "metrics_policy.yml"
    if not policy_path.exists():
        pytest.skip("configs/metrics_policy.yml missing")
    policy = validate_metrics_policy(load_metrics_policy(policy_path))
    ph = _policy_hash(policy)
    cache_dir = tmp_path / "data" / "cache" / "test-dv-cache" / "qa"
    cache_dir.mkdir(parents=True)
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum_firm": [100.0],
        "end_aum_firm": [110.0],
        "nnb_firm": [5.0],
        "market_pnl_firm": [5.0],
        "asset_growth_rate": [0.1],
        "organic_growth_rate": [0.05],
        "external_market_growth_rate": [0.05],
        "source": ["recomputed_from_metrics_monthly"],
        "global_slice_rowcount_per_month": [1],
    })
    _write_firm_level_to_dir(df, cache_dir, policy, "test-dv-cache")
    meta_path = cache_dir / "firm_level_recomputed.meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["policy_hash"] == ph

    (tmp_path / "configs").mkdir(parents=True)
    import shutil
    shutil.copy(policy_path, tmp_path / "configs" / "metrics_policy.yml")
    (tmp_path / "curated").mkdir(parents=True)
    metrics_df = pd.DataFrame({
        "path_id": ["global"],
        "slice_id": ["s1"],
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [5.0],
        "market_pnl": [5.0],
    })
    metrics_df.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)
    r = subprocess.run(
        [sys.executable, "-m", "pipelines.validation.recompute_firm_level", "--root", str(tmp_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert (tmp_path / "curated" / "qa" / "firm_level_recomputed.parquet").exists()
    assert (tmp_path / "curated" / "qa" / "firm_level_recomputed.meta.json").exists()
    # Cache was used (no recompute): curated copy matches cache
    assert (tmp_path / "curated" / "qa" / "firm_level_recomputed.parquet").stat().st_size == (cache_dir / "firm_level_recomputed.parquet").stat().st_size
