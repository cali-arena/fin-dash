"""
Tests for app.data.duckdb_views: load_views_manifest, query_view (views-only surface).
"""
from pathlib import Path

import pytest

from app.data.duckdb_views import load_views_manifest, query_view


def test_load_views_manifest_returns_none_when_missing(tmp_path: Path) -> None:
    """When duckdb_views_manifest.json is missing, load_views_manifest returns None."""
    assert load_views_manifest(tmp_path) is None


def test_load_views_manifest_returns_dict_when_present(tmp_path: Path) -> None:
    """When manifest exists, load_views_manifest returns parsed dict."""
    (tmp_path / "analytics").mkdir(parents=True)
    (tmp_path / "analytics" / "duckdb_views_manifest.json").write_text(
        '{"db_path": "analytics/db.duckdb", "schema": "dash", "views": []}',
        encoding="utf-8",
    )
    manifest = load_views_manifest(tmp_path)
    assert manifest is not None
    assert manifest.get("schema") == "dash"
    assert manifest.get("views") == []


def test_query_view_requires_manifest(tmp_path: Path) -> None:
    """query_view raises FileNotFoundError when manifest is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        query_view("v_firm_monthly", root=tmp_path)
    assert "duckdb_views_manifest" in str(exc_info.value) or "create_views" in str(exc_info.value)


def test_query_view_with_manifest_and_filters(tmp_path: Path) -> None:
    """query_view with manifest and filters builds parameterized WHERE and returns DataFrame."""
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run as build_run
    from legacy.legacy_pipelines.duckdb.create_views import run as create_views_run
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text("{}", encoding="utf-8")
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    df_channel = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "channel_l1": ["Direct"],
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    df_ticker = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "product_ticker": ["X"],
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    df.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df_ticker.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
    (tmp_path / "duckdb_policy.yml").write_text("""
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "dash"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
    v_channel_monthly: { source: "agg.channel_monthly" }
    v_ticker_monthly: { source: "agg.ticker_monthly" }
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    build_run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    create_views_run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    manifest = load_views_manifest(tmp_path)
    assert manifest is not None
    out = query_view("v_firm_monthly", root=tmp_path, manifest=manifest)
    assert out is not None and len(out) == 1
    out_filtered = query_view("v_firm_monthly", filters={"month_end": pd.Timestamp("2024-01-01")}, root=tmp_path, manifest=manifest)
    assert len(out_filtered) == 1
    with pytest.raises(ValueError, match="not in view"):
        query_view("v_firm_monthly", filters={"invalid_col": 1}, root=tmp_path, manifest=manifest)
