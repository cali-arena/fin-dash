"""
Tests for pipelines.duckdb.rebuild_analytics_layer: one-command entrypoint, exit codes for CI.
"""
from pathlib import Path

import pytest

from legacy.legacy_pipelines.duckdb.rebuild_analytics_layer import EXIT_FAIL, EXIT_OK, run


def test_rebuild_analytics_layer_exit_2_on_invalid_policy(tmp_path: Path) -> None:
    """Invalid or missing policy yields exit 2."""
    (tmp_path / "bad_policy.yml").write_text("duckdb:\n  db_path: not-a-duckdb-file\n", encoding="utf-8")
    code = run(policy_path=tmp_path / "bad_policy.yml", root=tmp_path)
    assert code == EXIT_FAIL

    code_missing = run(policy_path=tmp_path / "nonexistent.yml", root=tmp_path)
    assert code_missing == EXIT_FAIL


def test_rebuild_analytics_layer_exit_0_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline: build + create_views + smoke returns exit 0."""
    import pandas as pd
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
    code = run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    assert code == EXIT_OK
    assert (tmp_path / "analytics" / "analytics.duckdb").exists()
    assert (tmp_path / "analytics" / "duckdb_views_manifest.json").exists()
