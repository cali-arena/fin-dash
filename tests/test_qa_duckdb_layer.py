"""
Tests for pipelines.duckdb.qa_duckdb_layer: rowcount parity and view time-budget checks (mocked).
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from legacy.legacy_pipelines.duckdb.qa_duckdb_layer import (
    EXIT_FAIL,
    EXIT_OK,
    DEFAULT_TIME_BUDGET_MS,
    _load_qa_config,
    _parquet_row_count,
    run,
    run_rowcount_parity,
    run_view_latency_checks,
)


def test_load_qa_config_defaults_when_no_qa(tmp_path: Path) -> None:
    """When policy has no qa section, default time budgets are returned."""
    (tmp_path / "policy.yml").write_text("""
duckdb:
  db_path: "analytics/db.duckdb"
  schema: "s"
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
    cfg = _load_qa_config(tmp_path / "policy.yml")
    assert cfg == DEFAULT_TIME_BUDGET_MS


def test_load_qa_config_uses_policy_thresholds(tmp_path: Path) -> None:
    """When policy has qa.time_budget_ms, those values are used (with defaults for missing)."""
    (tmp_path / "policy.yml").write_text("""
duckdb:
  db_path: "analytics/db.duckdb"
  schema: "s"
  refresh_mode: "rebuild"
  source_paths:
    agg:
      firm_monthly: "agg/firm_monthly.parquet"
      channel_monthly: "agg/channel_monthly.parquet"
      ticker_monthly: "agg/ticker_monthly.parquet"
  views_to_create: {}
  ui_rule:
    reads_views_only: true
qa:
  time_budget_ms:
    firm: 100
    channel: 300
""", encoding="utf-8")
    cfg = _load_qa_config(tmp_path / "policy.yml")
    assert cfg["firm"] == 100
    assert cfg["channel"] == 300
    assert cfg["ticker"] == 500  # default


def test_parquet_row_count_uses_pyarrow_or_pandas(tmp_path: Path) -> None:
    """_parquet_row_count returns number of rows from parquet file."""
    import pandas as pd
    (tmp_path / "agg").mkdir(parents=True)
    df = pd.DataFrame({"a": [1, 2, 3]})
    df.to_parquet(tmp_path / "agg" / "f.parquet", index=False)
    n = _parquet_row_count(tmp_path / "agg" / "f.parquet")
    assert n == 3


@patch("pipelines.duckdb.qa_duckdb_layer.duckdb")
def test_run_rowcount_parity_equal_counts(mock_duckdb: MagicMock, tmp_path: Path) -> None:
    """When parquet and DuckDB counts match, parity rows have pass=True."""
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "firm_monthly.parquet").write_bytes(b"x")
    (tmp_path / "agg" / "channel_monthly.parquet").write_bytes(b"x")
    (tmp_path / "agg" / "ticker_monthly.parquet").write_bytes(b"x")
    con = MagicMock()
    con.execute.return_value.fetchone.side_effect = [(10,), (20,), (30,)]
    mock_duckdb.connect.return_value = con

    with patch("pipelines.duckdb.qa_duckdb_layer._parquet_row_count", side_effect=[10, 20, 30]):
        rows = run_rowcount_parity(
            tmp_path,
            tmp_path / "p.yml",
            str(tmp_path / "analytics" / "db.duckdb"),
            "s",
            {"firm_monthly": "agg/firm_monthly.parquet", "channel_monthly": "agg/channel_monthly.parquet", "ticker_monthly": "agg/ticker_monthly.parquet"},
        )
    assert len(rows) == 3
    assert all(r["pass"] for r in rows)
    assert rows[0]["parquet_count"] == 10 and rows[0]["duckdb_count"] == 10 and rows[0]["diff"] == "0"


@patch("pipelines.duckdb.qa_duckdb_layer.duckdb")
def test_run_rowcount_parity_mismatch_fails(mock_duckdb: MagicMock, tmp_path: Path) -> None:
    """When parquet and DuckDB counts differ, parity row has pass=False."""
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "firm_monthly.parquet").write_bytes(b"x")
    con = MagicMock()
    con.execute.return_value.fetchone.return_value = (5,)
    mock_duckdb.connect.return_value = con

    with patch("pipelines.duckdb.qa_duckdb_layer._parquet_row_count", return_value=10):
        rows = run_rowcount_parity(
            tmp_path,
            tmp_path / "p.yml",
            str(tmp_path / "db.duckdb"),
            "s",
            {"firm_monthly": "agg/firm_monthly.parquet"},
        )
    assert len(rows) == 1
    assert rows[0]["pass"] is False
    assert rows[0]["diff"] == "5"


@patch("pipelines.duckdb.qa_duckdb_layer.duckdb")
def test_run_view_latency_checks_under_budget(mock_duckdb: MagicMock) -> None:
    """When queries complete in under budget, all_passed is True."""
    con = MagicMock()
    con.execute().fetchall.return_value = []
    mock_duckdb.connect.return_value.__enter__ = lambda self: self
    mock_duckdb.connect.return_value.__exit__ = lambda *a: None
    mock_duckdb.connect.return_value = con

    # 3 views × 2 calls each (t0, t1): each query "takes" 1ms
    with patch("pipelines.duckdb.qa_duckdb_layer.time.perf_counter", side_effect=[0.0, 0.001, 0.001, 0.002, 0.002, 0.003]):
        results, all_passed = run_view_latency_checks("/fake/db", "s", DEFAULT_TIME_BUDGET_MS)
    assert all_passed is True
    assert len(results) == 3
    assert all(r["pass"] for r in results)


@patch("pipelines.duckdb.qa_duckdb_layer.duckdb")
def test_run_view_latency_checks_over_budget_fails(mock_duckdb: MagicMock) -> None:
    """When firm query exceeds 200ms, pass is False."""
    con = MagicMock()
    con.execute.return_value.fetchall.return_value = []
    mock_duckdb.connect.return_value = con

    # 3 views × 2 calls each; first query "takes" 0.5s (500ms) -> over 200ms budget for firm
    with patch("pipelines.duckdb.qa_duckdb_layer.time.perf_counter", side_effect=[0.0, 0.5, 0.5, 0.5, 0.5, 0.5]):
        results, all_passed = run_view_latency_checks("/fake/db", "s", {"firm": 200, "channel": 500, "ticker": 500})
    assert all_passed is False
    firm = next(r for r in results if "firm" in r["view"])
    assert firm["pass"] is False
    assert firm["query_ms"] is not None and firm["query_ms"] >= 200


def test_run_exit_0_when_all_pass(tmp_path: Path) -> None:
    """run() returns EXIT_OK when parity and latency pass."""
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "analytics").mkdir(parents=True)
    import pandas as pd
    for name in ("firm_monthly", "channel_monthly", "ticker_monthly"):
        pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "end_aum": [100.0]}).to_parquet(tmp_path / "agg" / f"{name}.parquet", index=False)
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
    from legacy.legacy_pipelines.duckdb.build_analytics_db import run as build_run
    from legacy.legacy_pipelines.duckdb.create_views import run as views_run
    (tmp_path / "agg" / "manifest.json").write_text("{}", encoding="utf-8")
    df_channel = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "channel_l1": ["D"], "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0]})
    df_ticker = pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "product_ticker": ["X"], "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0]})
    pd.DataFrame({"month_end": [pd.Timestamp("2024-01-01")], "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0]}).to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    df_channel.to_parquet(tmp_path / "agg" / "channel_monthly.parquet", index=False)
    df_ticker.to_parquet(tmp_path / "agg" / "ticker_monthly.parquet", index=False)
    build_run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    views_run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    code = run(policy_path=tmp_path / "duckdb_policy.yml", root=tmp_path)
    assert code == EXIT_OK
    assert (tmp_path / "qa" / "duckdb_rowcount_parity.csv").exists()
    assert (tmp_path / "qa" / "duckdb_view_latency.json").exists()
    parity = (tmp_path / "qa" / "duckdb_rowcount_parity.csv").read_text()
    assert "parquet_count" in parity and "duckdb_count" in parity
    latency = (tmp_path / "qa" / "duckdb_view_latency.json").read_text()
    assert "query_ms" in latency and "all_passed" in latency
