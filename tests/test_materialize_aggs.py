"""
Tests for pipelines.agg.materialize_aggs: cache hit/miss, rowcounts, runtime.
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

from legacy.legacy_pipelines.agg.manifest import get_table, load_manifest
from legacy.legacy_pipelines.agg.materialize_aggs import (
    aggregate_table,
    apply_null_handling,
    load_metrics_frame,
    materialize_aggs,
    write_agg,
)
from legacy.legacy_pipelines.contracts.agg_policy_contract import load_and_validate_agg_policy

MINIMAL_AGG_YAML = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum", "end_aum", "nnb"]
    rates: []
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "recompute"
    weights: {}
"""


def test_materialize_aggs_writes_then_cache_hit(tmp_path: Path) -> None:
    """First run writes parquet + meta; second run with same dataset_version + policy_hash is cache hit."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(MINIMAL_AGG_YAML, encoding="utf-8")
    (tmp_path / "data" / ".version.json").write_text(
        json.dumps({"dataset_version": "test_v1"}), encoding="utf-8"
    )
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "channel_l1": ["A", "A"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 210.0],
        "nnb": [10.0, 10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)

    results1 = materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)
    assert "firm_monthly" in results1
    assert results1["firm_monthly"] == "written"
    assert (tmp_path / "agg" / "firm_monthly.parquet").exists()
    assert (tmp_path / "agg" / "firm_monthly.meta.json").exists()
    meta = json.loads((tmp_path / "agg" / "firm_monthly.meta.json").read_text())
    assert meta["dataset_version"] == "test_v1"
    assert "policy_hash" in meta
    assert meta["rowcount"] == 2

    results2 = materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)
    assert results2["firm_monthly"] == "cache_hit"
    assert results2["channel_monthly_channel_l1"] == "cache_hit"
    assert results2["ticker_monthly_product_ticker"] == "cache_hit"


def test_verify_source_columns_raises_on_missing(tmp_path: Path) -> None:
    """When source is missing required columns, materialize_aggs raises with missing, available, which grain."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(MINIMAL_AGG_YAML, encoding="utf-8")
    (tmp_path / "data" / ".version.json").write_text(json.dumps({"dataset_version": "v1"}), encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "channel_l1": ["A"],
        "begin_aum": [100.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)
    with pytest.raises(ValueError) as exc_info:
        materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)
    msg = str(exc_info.value)
    assert "missing" in msg.lower()
    assert "available" in msg.lower()
    assert "grain" in msg.lower()


def test_manifest_includes_expected_entries_and_grains(tmp_path: Path) -> None:
    """After materialize_aggs, manifest.json has dataset_version, policy_hash, tables with correct grain and dims_used."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(MINIMAL_AGG_YAML, encoding="utf-8")
    (tmp_path / "data" / ".version.json").write_text(json.dumps({"dataset_version": "test_v1"}), encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "channel_l1": ["A"],
        "product_ticker": ["X"],
        "src_country_canonical": ["US"],
        "segment": ["S1"],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)
    materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    manifest_path = tmp_path / "agg" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "dataset_version" in manifest
    assert manifest["dataset_version"] == "test_v1"
    assert "policy_hash" in manifest
    assert "tables" in manifest
    tables = manifest["tables"]
    names = [t["name"] for t in tables]
    assert "firm_monthly" in names
    assert "channel_monthly_channel_l1" in names
    assert "ticker_monthly_product_ticker" in names
    firm = next(t for t in tables if t["name"] == "firm_monthly")
    assert firm["grain"] == "firm_monthly"
    assert firm["dims_used"] == []
    assert firm["measures"] == ["begin_aum", "end_aum", "nnb"]
    assert firm["rowcount"] == 1
    ch = next(t for t in tables if t["name"] == "channel_monthly_channel_l1")
    assert ch["grain"] == "channel_monthly"
    assert ch["dims_used"] == ["channel_l1"]


def test_load_manifest_and_get_table(tmp_path: Path) -> None:
    """load_manifest returns dict; get_table(name) returns path; get_table(nonexistent) raises KeyError."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(MINIMAL_AGG_YAML, encoding="utf-8")
    (tmp_path / "data" / ".version.json").write_text(json.dumps({"dataset_version": "v1"}), encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "channel_l1": ["A"],
        "product_ticker": ["X"],
        "src_country_canonical": ["US"],
        "segment": ["S1"],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)
    materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    manifest = load_manifest(tmp_path)
    assert isinstance(manifest, dict)
    assert "tables" in manifest
    path = get_table("firm_monthly", root=tmp_path)
    assert path == tmp_path / "agg" / "firm_monthly.parquet"
    assert path.exists()
    with pytest.raises(KeyError) as exc_info:
        get_table("nonexistent_table", root=tmp_path)
    assert "nonexistent_table" in str(exc_info.value)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "other_dir")


# --- Engine: sums, recompute rates, UNKNOWN nulls, firm_monthly no dims ---

POLICY_WITH_RATES_YAML = """
agg:
  source_table: "curated/metrics_monthly.parquet"
  time_key: "month_end"
  measures:
    additive: ["begin_aum", "end_aum", "nnb", "nnf", "market_pnl"]
    rates: ["ogr", "market_impact_rate", "fee_yield"]
  dims:
    channel_l1: "channel_l1"
    product_ticker: "product_ticker"
    src_country_canonical: "src_country_canonical"
    segment: "segment"
  grains:
    firm_monthly: []
    channel_monthly: [["channel_l1"]]
    ticker_monthly: ["product_ticker"]
    geo_monthly: ["src_country_canonical"]
    segment_monthly: [["segment"]]
  null_handling:
    strategy: "UNKNOWN"
    unknown_label: "UNKNOWN"
  rollup:
    additive_method: "sum"
    rates_method: "recompute"
    weights:
      ogr: "begin_aum"
      market_impact_rate: "begin_aum"
      fee_yield: "nnb"
"""


def test_aggregate_table_sums_and_recompute_rates(tmp_path: Path) -> None:
    """Sums are correct; recompute rates = nnb/begin_aum, market_pnl/begin_aum, nnf/nnb with guards."""
    (tmp_path / "p.yml").write_text(POLICY_WITH_RATES_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    # Two rows same month: begin_aum 100+200=300, nnb 10+20=30, nnf 1+2=3, market_pnl 5+10=15
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31")],
        "channel_l1": ["A", "A"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
        "nnf": [1.0, 2.0],
        "market_pnl": [5.0, 10.0],
    })
    out = aggregate_table(df, [], policy)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["begin_aum"] == 300.0
    assert row["end_aum"] == 330.0
    assert row["nnb"] == 30.0
    assert row["nnf"] == 3.0
    assert row["market_pnl"] == 15.0
    assert row["ogr"] == pytest.approx(30.0 / 300.0)
    assert row["market_impact_rate"] == pytest.approx(15.0 / 300.0)
    assert row["fee_yield"] == pytest.approx(3.0 / 30.0)


def test_aggregate_table_recompute_rates_guard_nan(tmp_path: Path) -> None:
    """When begin_aum or nnb is 0, recomputed rate is NaN; inf replaced by NaN."""
    (tmp_path / "p.yml").write_text(POLICY_WITH_RATES_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum": [0.0],
        "end_aum": [0.0],
        "nnb": [0.0],
        "nnf": [0.0],
        "market_pnl": [0.0],
    })
    out = aggregate_table(df, [], policy)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["ogr"])
    assert pd.isna(out.iloc[0]["market_impact_rate"])
    assert pd.isna(out.iloc[0]["fee_yield"])


def test_apply_null_handling_unknown_fills_nulls(tmp_path: Path) -> None:
    """UNKNOWN strategy fills null dims with unknown_label."""
    (tmp_path / "p.yml").write_text(MINIMAL_AGG_YAML, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "channel_l1": ["A", pd.NA],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 210.0],
        "nnb": [10.0, 10.0],
    })
    out = apply_null_handling(df, policy)
    assert out["channel_l1"].iloc[1] == "UNKNOWN"
    assert len(out) == 2


def test_firm_monthly_has_no_dims_and_meta_has_grain(tmp_path: Path) -> None:
    """firm_monthly output has only time_key + additive + rates; meta has grain []."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(POLICY_WITH_RATES_YAML, encoding="utf-8")
    (tmp_path / "data" / ".version.json").write_text(json.dumps({"dataset_version": "v1"}), encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "channel_l1": ["A"],
        "product_ticker": ["X"],
        "src_country_canonical": ["US"],
        "segment": ["S1"],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [10.0],
        "nnf": [1.0],
        "market_pnl": [5.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)
    materialize_aggs(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    firm = pd.read_parquet(tmp_path / "agg" / "firm_monthly.parquet")
    assert "month_end" in firm.columns
    assert "channel_l1" not in firm.columns
    assert "begin_aum" in firm.columns
    assert "ogr" in firm.columns
    meta = json.loads((tmp_path / "agg" / "firm_monthly.meta.json").read_text())
    assert meta["grain"] == []
    assert "schema_hash" in meta
    assert meta["min_month"] is not None
    assert meta["max_month"] is not None
