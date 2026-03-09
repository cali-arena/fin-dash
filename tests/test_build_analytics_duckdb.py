"""
Tests for pipelines.duckdb.build_analytics_duckdb: preflight, view health, deterministic errors.
"""
from pathlib import Path

import pytest

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import DuckDBPolicyError
from legacy.legacy_pipelines.duckdb.build_analytics_duckdb import (
    DuckDBBuildError,
    _preflight,
    _view_health_check,
    run,
)


def test_run_raises_when_agg_manifest_missing(tmp_path: Path) -> None:
    """When agg/manifest.json is missing, run() raises FileNotFoundError with clear message."""
    policy_yml = tmp_path / "duckdb_policy.yml"
    policy_yml.write_text("""
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    with pytest.raises(FileNotFoundError) as exc_info:
        run(policy_path=policy_yml, root=tmp_path)
    assert "manifest" in str(exc_info.value).lower() or "agg" in str(exc_info.value).lower()


def test_reads_views_only_guard_blocks_parquet_load(tmp_path: Path) -> None:
    """When duckdb_manifest exists with reads_views_only=true, agg_store.load_table raises."""
    (tmp_path / "analytics").mkdir(parents=True)
    (tmp_path / "analytics" / "duckdb_manifest.json").write_text(
        '{"reads_views_only": true, "dataset_version": "x", "db_path": "analytics/analytics.duckdb"}',
        encoding="utf-8",
    )
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text(
        '{"dataset_version": "x", "policy_hash": "h", "tables": [{"name": "firm_monthly", "path": "agg/firm_monthly.parquet"}]}',
        encoding="utf-8",
    )
    from app.agg_store import load_table
    with pytest.raises(RuntimeError) as exc_info:
        load_table("firm_monthly", tmp_path, dataset_version="x")
    assert "reads views only" in str(exc_info.value).lower()
    assert "build_analytics_duckdb" in str(exc_info.value)


def test_preflight_raises_with_missing_parquet_list(tmp_path: Path) -> None:
    """Preflight fails with DuckDBBuildError listing missing parquet paths and suggests build_aggs."""
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text(
        '{"dataset_version": "test", "policy_hash": "h", "tables": []}',
        encoding="utf-8",
    )
    (tmp_path / "duckdb_policy.yml").write_text("""
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    from legacy.legacy_pipelines.contracts.duckdb_policy_contract import load_and_validate_duckdb_policy
    policy = load_and_validate_duckdb_policy(tmp_path / "duckdb_policy.yml")
    with pytest.raises(DuckDBBuildError) as exc_info:
        _preflight(tmp_path, policy)
    msg = str(exc_info.value)
    assert "missing" in msg.lower() or "Preflight failed" in msg
    assert "build_aggs" in msg
    assert "firm_monthly" in msg or "parquet" in msg.lower()


def test_run_raises_when_agg_parquet_missing(tmp_path: Path) -> None:
    """When an agg parquet is missing, run() raises DuckDBBuildError from preflight with list."""
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text(
        '{"dataset_version": "test", "policy_hash": "h", "tables": []}',
        encoding="utf-8",
    )
    policy_yml = tmp_path / "duckdb_policy.yml"
    policy_yml.write_text("""
duckdb:
  db_path: "analytics/analytics.duckdb"
  schema: "analytics"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    with pytest.raises(DuckDBBuildError) as exc_info:
        run(policy_path=policy_yml, root=tmp_path)
    assert "build_aggs" in str(exc_info.value)


def test_view_health_check_fails_when_count_zero(tmp_path: Path) -> None:
    """View health check raises DuckDBBuildError when v_firm_monthly has 0 rows."""
    import duckdb
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute('CREATE SCHEMA analytics')
    con.execute('CREATE VIEW analytics.v_firm_monthly AS SELECT 1 AS x WHERE 1 = 0')
    with pytest.raises(DuckDBBuildError) as exc_info:
        _view_health_check(con, "analytics", "v_firm_monthly")
    assert "0 rows" in str(exc_info.value) or "failed" in str(exc_info.value).lower()
    con.close()


def test_view_health_check_fails_when_month_end_missing(tmp_path: Path) -> None:
    """View health check raises when view has no month_end column."""
    import duckdb
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute('CREATE SCHEMA analytics')
    con.execute('CREATE VIEW analytics.v_firm_monthly AS SELECT 1 AS id')
    with pytest.raises(DuckDBBuildError) as exc_info:
        _view_health_check(con, "analytics", "v_firm_monthly")
    assert "month_end" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()
    con.close()


def test_run_succeeds_and_writes_schema_hashes(tmp_path: Path) -> None:
    """With minimal parquet + meta.json (schema_hash), run() completes and manifest has schema_hashes."""
    import pandas as pd
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text(
        '{"dataset_version": "dv1", "policy_hash": "ph1", "tables": []}',
        encoding="utf-8",
    )
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]})
    for name in ("firm_monthly", "channel_monthly", "ticker_monthly"):
        (tmp_path / "agg" / f"{name}.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(tmp_path / "agg" / f"{name}.parquet", index=False)
        (tmp_path / "agg" / f"{name}.meta.json").write_text(
            '{"schema_hash": "abc123", "dataset_version": "dv1"}',
            encoding="utf-8",
        )
    policy_yml = tmp_path / "duckdb_policy.yml"
    policy_yml.write_text("""
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
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    run(policy_path=policy_yml, root=tmp_path)
    import json
    manifest_path = tmp_path / "analytics" / "duckdb_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("dataset_version") == "dv1"
    assert "created_sources" in manifest
    assert "firm_monthly" in manifest["created_sources"]
    assert "created_views" in manifest
    assert "v_firm_monthly" in manifest["created_views"]
    assert "schema_hashes" in manifest
    assert manifest["schema_hashes"].get("firm_monthly") == "abc123"


# ---- build_analytics_db.py (materialized tables, idempotent, policy-driven naming) ----

def test_build_analytics_db_name_mapping() -> None:
    """Policy-driven naming: agg_*, metrics_monthly, dim stem -> table name."""
    from legacy.legacy_pipelines.duckdb.build_analytics_db import (
        agg_source_to_table_name,
        curated_dim_stem_to_table_name,
        curated_metrics_to_table_name,
    )
    assert agg_source_to_table_name("firm_monthly") == "agg_firm_monthly"
    assert agg_source_to_table_name("channel_monthly") == "agg_channel_monthly"
    assert agg_source_to_table_name("ticker_monthly") == "agg_ticker_monthly"
    assert agg_source_to_table_name("geo_monthly") == "agg_geo_monthly"
    assert agg_source_to_table_name("segment_monthly") == "agg_segment_monthly"
    assert curated_metrics_to_table_name() == "metrics_monthly"
    assert curated_dim_stem_to_table_name("dim_channel") == "dim_channel"
    assert curated_dim_stem_to_table_name("dim_product") == "dim_product"
    assert curated_dim_stem_to_table_name("dim_geo") == "dim_geo"
    assert curated_dim_stem_to_table_name("dim_time") == "dim_time"


def test_build_analytics_db_optional_missing_warns_only(tmp_path: Path) -> None:
    """Optional sources missing: warn only, run succeeds with required only."""
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run
    (tmp_path / "agg").mkdir(parents=True)
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]})
    df_channel = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0], "channel_l1": ["Direct"]})
    df.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
    # Policy references optional geo_monthly, segment_monthly, metrics_monthly (paths missing)
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
      geo_monthly: "agg/geo_monthly.parquet"
      segment_monthly: "agg/segment_monthly.parquet"
    curated:
      metrics_monthly: "curated/metrics_monthly.parquet"
  views_to_create:
    v_firm_monthly: { source: "agg.firm_monthly" }
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    import duckdb
    con = duckdb.connect(str(tmp_path / "analytics" / "analytics.duckdb"))
    for t in ("agg_firm_monthly", "agg_channel_monthly", "agg_ticker_monthly"):
        r = con.execute(f'SELECT COUNT(*) FROM dash."{t}"').fetchone()
        assert r[0] == 1
    con.close()


def test_build_analytics_db_preflight_raises_when_parquet_missing(tmp_path: Path) -> None:
    """build_analytics_db: preflight raises AnalyticsDBError with missing list."""
    from legacy.legacy_pipelines.duckdb.build_analytics_db import AnalyticsDBError, run
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
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    with pytest.raises(AnalyticsDBError) as exc_info:
        run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    assert "missing" in str(exc_info.value).lower()
    assert "build_aggs" in str(exc_info.value)


def test_build_analytics_db_run_succeeds(tmp_path: Path) -> None:
    """build_analytics_db: run creates materialized tables and validates."""
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run
    (tmp_path / "agg").mkdir(parents=True)
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]})
    df_channel = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0], "channel_l1": ["Direct"]})
    df.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
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
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    import duckdb
    con = duckdb.connect(str(tmp_path / "analytics" / "analytics.duckdb"))
    for t in ("agg_firm_monthly", "agg_channel_monthly", "agg_ticker_monthly"):
        r = con.execute(f'SELECT COUNT(*) FROM dash."{t}"').fetchone()
        assert r[0] == 1
    con.close()


def test_build_analytics_db_manifest_content(tmp_path: Path) -> None:
    """build_analytics_db: duckdb_manifest.json has dataset_version, policy_hash, pipeline_version, loaded_tables, created_at."""
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text('{"dataset_version": "dv-synthetic", "policy_hash": "ph"}', encoding="utf-8")
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]})
    df_channel = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0], "channel_l1": ["Direct"]})
    df.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
    (tmp_path / "agg" / "firm_monthly.meta.json").write_text('{"schema_hash": "abc123"}', encoding="utf-8")
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
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    import json
    manifest = json.loads((tmp_path / "analytics" / "duckdb_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("dataset_version") == "dv-synthetic"
    assert "policy_hash" in manifest and len(manifest["policy_hash"]) == 64
    assert "pipeline_version" in manifest
    assert manifest.get("db_path") == "analytics/analytics.duckdb"
    assert manifest.get("schema") == "dash"
    assert "loaded_tables" in manifest
    tables = manifest["loaded_tables"]
    assert len(tables) >= 3
    firm = next(t for t in tables if "agg_firm_monthly" in t["name"])
    assert firm["name"] == "dash.agg_firm_monthly"
    assert "source_path" in firm and "firm_monthly.parquet" in firm["source_path"]
    assert firm["rowcount"] == 1
    assert firm.get("min_month_end") and firm.get("max_month_end")
    assert firm.get("schema_hash") == "abc123"
    assert "created_at" in manifest and "T" in manifest["created_at"]


def test_build_analytics_db_skip_when_manifest_matches(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """build_analytics_db: second run with force=False skips rebuild when manifest matches."""
    import logging
    import pandas as pd
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run
    caplog.set_level(logging.INFO)
    (tmp_path / "agg").mkdir(parents=True)
    df = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]})
    df_channel = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0], "channel_l1": ["Direct"]})
    df.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
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
  ui_rule:
    reads_views_only: true
""", encoding="utf-8")
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    manifest_path = tmp_path / "analytics" / "duckdb_manifest.json"
    first_created = manifest_path.read_text(encoding="utf-8")
    caplog.clear()
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path, force=False)
    assert "Skip rebuild" in caplog.text
    assert manifest_path.read_text(encoding="utf-8") == first_created

    # --force forces rebuild: manifest is rewritten (e.g. created_at changes)
    run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path, force=True)
    second_created = manifest_path.read_text(encoding="utf-8")
    import json as _json
    first_manifest = _json.loads(first_created)
    second_manifest = _json.loads(second_created)
    assert second_manifest["created_at"] >= first_manifest["created_at"]
