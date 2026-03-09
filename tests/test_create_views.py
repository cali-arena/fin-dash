"""
Tests for pipelines.duckdb.create_views: idempotent v_* layer, health check, actionable errors.
View SQL: thin projection + rate logic; channel_l2 optional; COALESCE stored rates.
"""
from pathlib import Path

import pytest

from legacy.legacy_pipelines.duckdb.create_views import (
    CreateViewsError,
    build_view_sql,
    get_table_columns,
    run,
)


def test_create_views_fails_when_db_missing(tmp_path: Path) -> None:
    """When DuckDB file does not exist, run() raises CreateViewsError with build_analytics_db hint."""
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
    with pytest.raises(CreateViewsError) as exc_info:
        run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    assert "build_analytics_db" in str(exc_info.value)
    assert "not found" in str(exc_info.value).lower()


def test_create_views_fails_when_required_tables_missing(tmp_path: Path) -> None:
    """When DB exists but required agg tables are missing, run() raises with list of missing tables."""
    import duckdb
    (tmp_path / "analytics").mkdir(parents=True)
    duckdb.connect(str(tmp_path / "analytics" / "analytics.duckdb")).close()
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
    with pytest.raises(CreateViewsError) as exc_info:
        run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    err = str(exc_info.value)
    assert "build_analytics_db" in err
    assert "agg_firm_monthly" in err or "missing" in err.lower()


def test_create_views_succeeds_after_build(tmp_path: Path) -> None:
    """After build_analytics_db, create_views creates v_* views and health check passes."""
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run as build_run
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text("{}", encoding="utf-8")
    # Include required view columns: month_end, end_aum, nnb, market_pnl (and begin_aum, nnf for rates)
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
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    import duckdb
    con = duckdb.connect(str(tmp_path / "analytics" / "analytics.duckdb"))
    row = con.execute('SELECT COUNT(*) FROM dash.v_firm_monthly').fetchone()
    assert row[0] == 1
    row2 = con.execute("SELECT MIN(month_end), MAX(month_end) FROM dash.v_firm_monthly").fetchone()
    assert row2[0] is not None and row2[1] is not None
    con.close()

    # Views manifest written (surface contract)
    import json
    views_manifest_path = tmp_path / "analytics" / "duckdb_views_manifest.json"
    assert views_manifest_path.exists()
    views_manifest = json.loads(views_manifest_path.read_text(encoding="utf-8"))
    assert views_manifest.get("db_path") == "analytics/analytics.duckdb"
    assert views_manifest.get("schema") == "dash"
    assert "views" in views_manifest
    views_list = views_manifest["views"]
    assert len(views_list) >= 3
    firm_view = next(v for v in views_list if "v_firm_monthly" in v.get("name", ""))
    assert firm_view.get("source_table") == "dash.agg_firm_monthly"
    assert "month_end" in (firm_view.get("columns") or [])
    assert "created_at" in views_manifest


def test_get_table_columns_returns_column_list() -> None:
    """get_table_columns returns column names from DESCRIBE."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA s")
    con.execute(
        'CREATE TABLE s.t (month_end DATE, channel_l1 VARCHAR, channel_l2 VARCHAR, begin_aum DOUBLE, end_aum DOUBLE)'
    )
    cols = get_table_columns(con, "s", "t")
    con.close()
    assert "month_end" in cols
    assert "channel_l1" in cols
    assert "channel_l2" in cols
    assert "begin_aum" in cols
    assert "end_aum" in cols


def test_build_view_sql_includes_channel_l2_when_present() -> None:
    """build_view_sql for v_channel_monthly includes channel_l2 when table has that column."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA s")
    con.execute(
        'CREATE TABLE s.agg_channel_monthly (month_end DATE, channel_l1 VARCHAR, channel_l2 VARCHAR, begin_aum DOUBLE, end_aum DOUBLE, nnb DOUBLE, nnf DOUBLE, market_pnl DOUBLE)'
    )
    sql = build_view_sql(con, "s", "agg_channel_monthly", "v_channel_monthly")
    con.close()
    assert "channel_l1" in sql
    assert "channel_l2" in sql
    assert "begin_aum" in sql
    assert "ogr" in sql
    assert "market_impact_rate" in sql
    assert "fee_yield" in sql


def test_build_view_sql_omits_channel_l2_when_absent() -> None:
    """build_view_sql for v_channel_monthly omits channel_l2 when table does not have that column."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA s")
    con.execute(
        'CREATE TABLE s.agg_channel_monthly (month_end DATE, channel_l1 VARCHAR, begin_aum DOUBLE, end_aum DOUBLE, nnb DOUBLE, nnf DOUBLE, market_pnl DOUBLE)'
    )
    sql = build_view_sql(con, "s", "agg_channel_monthly", "v_channel_monthly")
    con.close()
    assert "channel_l1" in sql
    assert "channel_l2" not in sql


def test_build_view_sql_uses_coalesce_when_stored_rate_exists() -> None:
    """When table has ogr column, generated SQL uses COALESCE(ogr, computed) for backwards compatibility."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA s")
    con.execute(
        'CREATE TABLE s.agg_firm_monthly (month_end DATE, begin_aum DOUBLE, end_aum DOUBLE, nnb DOUBLE, nnf DOUBLE, market_pnl DOUBLE, ogr DOUBLE, market_impact_rate DOUBLE, fee_yield DOUBLE)'
    )
    sql = build_view_sql(con, "s", "agg_firm_monthly", "v_firm_monthly")
    con.close()
    assert "COALESCE(t.\"ogr\"" in sql
    assert "COALESCE(t.\"market_impact_rate\"" in sql
    assert "COALESCE(t.\"fee_yield\"" in sql


def test_build_view_sql_computed_only_when_no_stored_rates() -> None:
    """When table has no ogr/market_impact_rate/fee_yield, SQL uses only computed expressions (no COALESCE)."""
    import duckdb
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA s")
    con.execute(
        'CREATE TABLE s.agg_firm_monthly (month_end DATE, begin_aum DOUBLE, end_aum DOUBLE, nnb DOUBLE, nnf DOUBLE, market_pnl DOUBLE)'
    )
    sql = build_view_sql(con, "s", "agg_firm_monthly", "v_firm_monthly")
    con.close()
    assert "COALESCE(t.\"ogr\"" not in sql
    assert "AS \"ogr\"" in sql
    assert "CASE WHEN t.\"begin_aum\" > 0 THEN t.\"nnb\" / t.\"begin_aum\"" in sql
