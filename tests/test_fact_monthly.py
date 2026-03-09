"""
Pytest: fact_monthly build (aggregation, channel derivation, AUM snapshot) and curated persist + gates.
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

from legacy.legacy_src.curate.fact_monthly import (
    GRAIN,
    build_fact_monthly,
    derive_channels,
)
from legacy.legacy_src.persist.curated_store import (
    persist_fact_monthly,
    run_curated_fact_monthly_gates,
)


def _minimal_raw(
    *,
    n_rows: int = 4,
    month_end: str = "2021-01-31",
    product_ticker: str = "AGG",
    channel_raw: str = "RIA",
    channel_standard: str | None = "RIA",
    channel_best: str | None = "RIA",
    src_country: str = "US",
    product_country: str = "US",
    segment: str = "S1",
    sub_segment: str = "SS1",
    nnb: list[float] | None = None,
    nnf: list[float] | None = None,
    aum: list[float] | None = None,
) -> pd.DataFrame:
    """Build minimal df_raw with all required grain columns and measures."""
    if nnb is None:
        nnb = [1.0, 2.0, 3.0, 4.0][:n_rows]
    if nnf is None:
        nnf = [0.1, 0.2, 0.3, 0.4][:n_rows]
    if aum is None:
        aum = [100.0, 200.0, 300.0, 400.0][:n_rows]
    nnb = nnb + [0.0] * (n_rows - len(nnb))
    nnf = nnf + [0.0] * (n_rows - len(nnf))
    aum = aum + [0.0] * (n_rows - len(aum))

    df = pd.DataFrame({
        "month_end": pd.to_datetime([month_end] * n_rows),
        "product_ticker": [product_ticker] * n_rows,
        "channel_raw": [channel_raw] * n_rows,
        "channel_standard": [channel_standard] * n_rows if channel_standard is not None else [pd.NA] * n_rows,
        "channel_best": [channel_best] * n_rows if channel_best is not None else [pd.NA] * n_rows,
        "src_country": [src_country] * n_rows,
        "product_country": [product_country] * n_rows,
        "segment": [segment] * n_rows,
        "sub_segment": [sub_segment] * n_rows,
        "net_new_business": nnb[:n_rows],
        "net_new_base_fees": nnf[:n_rows],
        "asset_under_management": aum[:n_rows],
    })
    return df


def test_duplicates_at_grain_aggregation_nnb_nnf() -> None:
    """Duplicate rows at same grain are aggregated: nnb and nnf summed."""
    # Same grain, 4 rows
    df_raw = _minimal_raw(n_rows=4, nnb=[10.0, 20.0, 30.0, 40.0], nnf=[1.0, 2.0, 3.0, 4.0], aum=[100.0, 200.0, 300.0, 400.0])
    fact, stats = build_fact_monthly(df_raw)
    assert len(fact) == 1
    assert stats["rows_in"] == 4
    assert stats["rows_out"] == 1
    assert fact["net_new_business"].iloc[0] == 100.0
    assert fact["net_new_base_fees"].iloc[0] == 10.0


def test_aum_snapshot_max_fallback_deterministic() -> None:
    """AUM uses max fallback when no date_raw; result is deterministic (max of group)."""
    df_raw = _minimal_raw(n_rows=3, aum=[50.0, 200.0, 150.0])
    fact, stats = build_fact_monthly(df_raw)
    assert stats["aum_rule_used"] == "max"
    assert len(fact) == 1
    assert fact["asset_under_management"].iloc[0] == 200.0


def test_channel_best_derived_from_channel_standard() -> None:
    """When channel_best is null, it is filled from channel_standard (fallback)."""
    df_raw = _minimal_raw(n_rows=2, channel_standard="StdChannel", channel_best=None)
    out = derive_channels(df_raw, mapping_df=None)
    assert out["channel_best"].iloc[0] == "StdChannel"
    assert out["channel_best"].iloc[1] == "StdChannel"


def test_channel_best_kept_when_present() -> None:
    """When channel_best is non-null in input, it is kept."""
    df_raw = _minimal_raw(n_rows=1, channel_standard="Std", channel_best="Best")
    out = derive_channels(df_raw, mapping_df=None)
    assert out["channel_best"].iloc[0] == "Best"


def test_two_grains_two_rows_out() -> None:
    """Two distinct grains yield two output rows."""
    df1 = _minimal_raw(n_rows=1, product_ticker="AGG", segment="S1")
    df2 = _minimal_raw(n_rows=1, product_ticker="BND", segment="S2")
    df_raw = pd.concat([df1, df2], ignore_index=True)
    fact, stats = build_fact_monthly(df_raw)
    assert len(fact) == 2
    assert stats["rows_out"] == 2
    assert set(fact["product_ticker"]) == {"AGG", "BND"}


def test_persist_fact_monthly_and_gates_pass() -> None:
    """persist_fact_monthly runs gates and writes parquet + meta (versioned and latest)."""
    df_raw = _minimal_raw(n_rows=2)
    fact, stats = build_fact_monthly(df_raw)
    tmp = Path(__file__).resolve().parents[1] / "tmp_curated_test"
    tmp.mkdir(exist_ok=True)
    versioned_parquet = tmp / "data" / "cache" / "test_v1" / "curated" / "fact_monthly.parquet"
    versioned_meta = tmp / "data" / "cache" / "test_v1" / "curated" / "fact_monthly.meta.json"
    latest_parquet = tmp / "curated" / "fact_monthly.parquet"
    latest_meta = tmp / "curated" / "fact_monthly.meta.json"
    try:
        persist_fact_monthly(fact, stats, "test_v1", root=tmp, write_latest_copy=True)
        assert versioned_parquet.exists()
        assert versioned_meta.exists()
        assert latest_parquet.exists()
        assert latest_meta.exists()
        with open(latest_meta, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["dataset_version"] == "test_v1"
        assert meta["status"] == "ok"
        assert meta["rows"] == 1
        assert meta["grain"] == GRAIN
        assert "stats_summary" in meta
    finally:
        for p in (versioned_parquet, versioned_meta, latest_parquet, latest_meta):
            if p.exists():
                p.unlink()


def test_gates_fail_null_grain_raises() -> None:
    """run_curated_fact_monthly_gates raises when a grain column has nulls."""
    fact, _ = build_fact_monthly(_minimal_raw(n_rows=1))
    fact.loc[0, "segment"] = pd.NA
    with pytest.raises(ValueError, match="segment.*null"):
        run_curated_fact_monthly_gates(fact, {"rows_out": 1})


def test_gates_fail_zero_rows_raises() -> None:
    """run_curated_fact_monthly_gates raises when rows_out <= 0."""
    fact, stats = build_fact_monthly(_minimal_raw(n_rows=1))
    with pytest.raises(ValueError, match="rows_out"):
        run_curated_fact_monthly_gates(fact, {"rows_out": 0})


def test_gates_fail_duplicate_grain_raises() -> None:
    """run_curated_fact_monthly_gates raises when grain is not unique."""
    fact, _ = build_fact_monthly(_minimal_raw(n_rows=1))
    fact = pd.concat([fact, fact], ignore_index=True)
    with pytest.raises(ValueError, match="unique|duplicate"):
        run_curated_fact_monthly_gates(fact, {"rows_out": 2})


def test_persist_gates_fail_writes_meta_then_raises() -> None:
    """When gates fail, persist_fact_monthly writes failure meta then raises."""
    fact, stats = build_fact_monthly(_minimal_raw(n_rows=1))
    fact.loc[0, "product_country"] = pd.NA
    tmp = Path(__file__).resolve().parents[1] / "tmp_curated_test"
    tmp.mkdir(exist_ok=True)
    latest_meta = tmp / "curated" / "fact_monthly.meta.json"
    try:
        with pytest.raises(ValueError, match="gates failed"):
            persist_fact_monthly(fact, stats, "test_v2", root=tmp, write_latest_copy=True)
        assert latest_meta.exists()
        with open(latest_meta, encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["status"] == "failed"
        assert "gate_errors" in meta
    finally:
        if latest_meta.exists():
            latest_meta.unlink()
        (tmp / "curated").rmdir() if (tmp / "curated").exists() else None
