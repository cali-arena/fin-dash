"""
Minimal unit tests for pipelines.agg.build_aggs: sums, recomputed rates, deterministic sorting.
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

from legacy.legacy_pipelines.agg.build_aggs import (
    FAIL_CONTEXT_FILENAME,
    QA_DIR,
    DimJoinError,
    build_one_agg,
    load_source_frame,
    maybe_join_dims,
    normalize_null_dims,
    preflight,
    run,
    write_agg_fail_context,
    write_meta_json,
    write_parquet_atomic,
)
from legacy.legacy_pipelines.contracts.agg_policy_contract import load_and_validate_agg_policy

POLICY_WITH_RATES = """
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


def test_build_one_agg_sums_correct() -> None:
    """Sums over group are correct (two rows same month)."""
    measures = {
        "additive": ["begin_aum", "end_aum", "nnb", "nnf", "market_pnl"],
        "rates": ["ogr", "market_impact_rate", "fee_yield"],
    }
    rollup = {"rates_method": "recompute", "weights": {}}
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31")],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
        "nnf": [1.0, 2.0],
        "market_pnl": [5.0, 10.0],
    })
    out = build_one_agg(df, "month_end", [], measures, rollup)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["begin_aum"] == 300.0
    assert row["end_aum"] == 330.0
    assert row["nnb"] == 30.0
    assert row["nnf"] == 3.0
    assert row["market_pnl"] == 15.0


def test_build_one_agg_recomputed_rates_correct() -> None:
    """Recomputed rates: ogr = nnb/begin_aum, market_impact_rate = market_pnl/begin_aum, fee_yield = nnf/nnb."""
    measures = {
        "additive": ["begin_aum", "end_aum", "nnb", "nnf", "market_pnl"],
        "rates": ["ogr", "market_impact_rate", "fee_yield"],
    }
    rollup = {"rates_method": "recompute", "weights": {}}
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum": [300.0],
        "end_aum": [330.0],
        "nnb": [30.0],
        "nnf": [3.0],
        "market_pnl": [15.0],
    })
    out = build_one_agg(df, "month_end", [], measures, rollup)
    assert out["ogr"].iloc[0] == pytest.approx(30.0 / 300.0)
    assert out["market_impact_rate"].iloc[0] == pytest.approx(15.0 / 300.0)
    assert out["fee_yield"].iloc[0] == pytest.approx(3.0 / 30.0)


def test_build_one_agg_recompute_guards_nan() -> None:
    """When denominator <= 0, rate is NaN; inf replaced by NaN."""
    measures = {
        "additive": ["begin_aum", "end_aum", "nnb", "nnf", "market_pnl"],
        "rates": ["ogr", "market_impact_rate", "fee_yield"],
    }
    rollup = {"rates_method": "recompute", "weights": {}}
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "begin_aum": [0.0],
        "end_aum": [0.0],
        "nnb": [0.0],
        "nnf": [0.0],
        "market_pnl": [0.0],
    })
    out = build_one_agg(df, "month_end", [], measures, rollup)
    assert pd.isna(out["ogr"].iloc[0])
    assert pd.isna(out["market_impact_rate"].iloc[0])
    assert pd.isna(out["fee_yield"].iloc[0])


def test_build_one_agg_deterministic_sort() -> None:
    """Output is sorted by [time_key] + grain_dims (lexicographic, stable)."""
    measures = {"additive": ["begin_aum"], "rates": []}
    rollup = {"rates_method": "store_numerators", "weights": {}}
    df = pd.DataFrame({
        "month_end": [
            pd.Timestamp("2024-02-29"),
            pd.Timestamp("2024-01-31"),
            pd.Timestamp("2024-02-29"),
            pd.Timestamp("2024-01-31"),
        ],
        "channel_l1": ["B", "A", "A", "B"],
        "begin_aum": [10.0, 20.0, 30.0, 40.0],
    })
    out = build_one_agg(df, "month_end", ["channel_l1"], measures, rollup)
    assert list(out["month_end"]) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-02-29"),
    ]
    assert list(out["channel_l1"]) == ["A", "B", "A", "B"]
    assert list(out["begin_aum"]) == [20.0, 40.0, 30.0, 10.0]


def test_write_parquet_atomic_and_meta(tmp_path: Path) -> None:
    """write_parquet_atomic and write_meta_json produce files atomically."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    write_parquet_atomic(df, tmp_path / "out.parquet")
    assert (tmp_path / "out.parquet").exists()
    back = pd.read_parquet(tmp_path / "out.parquet")
    pd.testing.assert_frame_equal(back, df)

    meta = {"rowcount": 2, "grain": []}
    write_meta_json(meta, tmp_path / "out.meta.json")
    assert (tmp_path / "out.meta.json").exists()
    assert json.loads((tmp_path / "out.meta.json").read_text()) == meta


def test_normalize_null_dims_unknown(tmp_path: Path) -> None:
    """normalize_null_dims with UNKNOWN fills null dims."""
    (tmp_path / "p.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
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
        "nnf": [1.0, 1.0],
        "market_pnl": [5.0, 5.0],
    })
    out = normalize_null_dims(df, policy)
    assert out["channel_l1"].iloc[1] == "UNKNOWN"
    assert len(out) == 2


def test_run_writes_firm_monthly_and_meta(tmp_path: Path) -> None:
    """Full run: load source, normalize, build aggs; firm_monthly has correct sums and meta."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "channel_l1": ["A", "A"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
        "nnf": [1.0, 2.0],
        "market_pnl": [5.0, 10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)

    run(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    assert (tmp_path / "agg" / "firm_monthly.parquet").exists()
    firm = pd.read_parquet(tmp_path / "agg" / "firm_monthly.parquet")
    assert firm["begin_aum"].sum() == 300.0
    assert firm["nnb"].sum() == 30.0
    assert "ogr" in firm.columns
    assert (tmp_path / "agg" / "firm_monthly.meta.json").exists()
    meta = json.loads((tmp_path / "agg" / "firm_monthly.meta.json").read_text())
    assert meta["rowcount"] == 2
    assert meta["grain"] == []
    assert "schema_hash" in meta
    assert "min_month" in meta
    assert "max_month" in meta
    assert "min_month_end" in meta
    assert "max_month_end" in meta
    assert "created_at" in meta
    assert "columns" in meta


def test_load_source_frame_raises_on_missing(tmp_path: Path) -> None:
    """load_source_frame raises FileNotFoundError when source table does not exist."""
    (tmp_path / "p.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    with pytest.raises(FileNotFoundError, match="not found"):
        load_source_frame(policy, root=tmp_path)


def test_missing_column_error_message_and_diagnostics_file(tmp_path: Path) -> None:
    """On missing dim column, error message is actionable and qa/agg_build_fail_context.json is written."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
    # Source missing 'segment' so segment_monthly grain will fail
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "channel_l1": ["A", "A"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
        "nnf": [1.0, 2.0],
        "market_pnl": [5.0, 10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)

    with pytest.raises(ValueError) as exc_info:
        run(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    msg = str(exc_info.value)
    assert "segment" in msg or "needs dim columns" in msg.lower() or "missing" in msg.lower()
    assert "segment_monthly" in msg or "segment" in msg

    diag_path = tmp_path / QA_DIR / FAIL_CONTEXT_FILENAME
    assert diag_path.exists(), "qa/agg_build_fail_context.json should be written on failure"
    ctx = json.loads(diag_path.read_text())
    assert "table_name" in ctx
    assert "missing_columns" in ctx
    assert "segment" in ctx["missing_columns"]
    assert "sample_rows" in ctx
    assert "dtypes" in ctx
    assert "policy_excerpt" in ctx
    assert "time_key" in ctx["policy_excerpt"] or "measures" in str(ctx["policy_excerpt"])


def test_diagnostics_file_creation_on_preflight_failure(tmp_path: Path) -> None:
    """On preflight failure (e.g. missing additive), fail context is written with table_name preflight."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "channel_l1": ["A"],
        "product_ticker": ["X"],
        "src_country_canonical": ["US"],
        "segment": ["S1"],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [10.0],
        # missing nnf, market_pnl
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)

    with pytest.raises(ValueError):
        run(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    diag_path = tmp_path / QA_DIR / FAIL_CONTEXT_FILENAME
    assert diag_path.exists()
    ctx = json.loads(diag_path.read_text())
    assert ctx.get("table_name") == "preflight"
    assert "missing_columns" in ctx
    assert "policy_excerpt" in ctx


def test_manifest_contains_all_required_tables(tmp_path: Path) -> None:
    """After run(), agg/manifest.json exists with dataset_version, policy_hash, and tables listing all outputs."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "agg_policy.yml").write_text(POLICY_WITH_RATES, encoding="utf-8")
    source = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "channel_l1": ["A", "A"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
        "nnf": [1.0, 2.0],
        "market_pnl": [5.0, 10.0],
    })
    source.to_parquet(tmp_path / "curated" / "metrics_monthly.parquet", index=False)

    run(policy_path=tmp_path / "configs" / "agg_policy.yml", root=tmp_path)

    manifest_path = tmp_path / "agg" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert "dataset_version" in manifest
    assert "policy_hash" in manifest
    assert "tables" in manifest
    tables = manifest["tables"]
    expected_names = {"firm_monthly", "channel_monthly", "ticker_monthly", "geo_monthly", "segment_monthly"}
    actual_names = {t["name"] for t in tables}
    assert expected_names == actual_names, f"tables mismatch: expected {expected_names}, got {actual_names}"
    for t in tables:
        assert "name" in t
        assert "path" in t
        assert "grain" in t
        assert "rowcount" in t
        assert "min_month_end" in t
        assert "max_month_end" in t
        assert "columns" in t
        assert t["path"].startswith("agg/") and t["path"].endswith(".parquet")


# ---- Dimension join discipline ----
# Policy with channel_monthly grain so channel_l1 is required by grains.
POLICY_WITH_CHANNEL_GRAIN = """
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
    rates_method: "store_numerators"
    weights: {}
"""


def test_missing_channel_l1_triggers_join(tmp_path: Path) -> None:
    """When a grain requires channel_l1 and it is missing, maybe_join_dims joins dim_channel on preferred_label."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "p.yml").write_text(POLICY_WITH_CHANNEL_GRAIN, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    # Source has preferred_label but no channel_l1
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "preferred_label": ["retail", "inst"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
    })
    dim_channel = pd.DataFrame({
        "preferred_label": ["retail", "inst"],
        "channel_l1": ["Retail", "Institutional"],
    })
    dim_channel.to_parquet(tmp_path / "curated" / "dim_channel.parquet", index=False)

    out = maybe_join_dims(df, policy, root=tmp_path)

    assert len(out) == len(df)
    assert "channel_l1" in out.columns
    assert list(out["channel_l1"]) == ["Retail", "Institutional"]


def test_already_present_channel_l1_skips_join(tmp_path: Path) -> None:
    """When channel_l1 is already in the frame, maybe_join_dims does not join dim_channel (no row change, values preserved)."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "p.yml").write_text(POLICY_WITH_CHANNEL_GRAIN, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
        "preferred_label": ["x", "y"],
        "channel_l1": ["A", "B"],
        "product_ticker": ["X", "Y"],
        "src_country_canonical": ["US", "US"],
        "segment": ["S1", "S1"],
        "begin_aum": [100.0, 200.0],
        "end_aum": [110.0, 220.0],
        "nnb": [10.0, 20.0],
    })
    # dim_channel would map x->X, y->Y; if we joined we'd get different channel_l1
    dim_channel = pd.DataFrame({
        "preferred_label": ["x", "y"],
        "channel_l1": ["X", "Y"],
    })
    dim_channel.to_parquet(tmp_path / "curated" / "dim_channel.parquet", index=False)

    out = maybe_join_dims(df, policy, root=tmp_path)

    assert len(out) == len(df)
    assert list(out["channel_l1"]) == ["A", "B"]


def test_duplicate_dim_keys_fails_cleanly(tmp_path: Path) -> None:
    """When dim_channel has duplicate keys on the join column, maybe_join_dims raises DimJoinError with diagnostics."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "p.yml").write_text(POLICY_WITH_CHANNEL_GRAIN, encoding="utf-8")
    policy = load_and_validate_agg_policy(tmp_path / "p.yml")
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-31")],
        "preferred_label": ["retail"],
        "product_ticker": ["X"],
        "src_country_canonical": ["US"],
        "segment": ["S1"],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [10.0],
    })
    dim_channel = pd.DataFrame({
        "preferred_label": ["retail", "retail"],
        "channel_l1": ["R1", "R2"],
    })
    dim_channel.to_parquet(tmp_path / "curated" / "dim_channel.parquet", index=False)

    with pytest.raises(DimJoinError) as exc_info:
        maybe_join_dims(df, policy, root=tmp_path)

    msg = str(exc_info.value)
    assert "duplicate" in msg.lower()
    assert "dim_channel" in msg
    assert "Sample" in msg or "sample" in msg
