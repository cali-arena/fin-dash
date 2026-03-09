"""
Unit tests for data_gateway guardrails: views-only enforcement and STRICT_VIEWS_ONLY groupby/merge ban.
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data_gateway import (
    VIEWS_ONLY_MESSAGE,
    _validate_views_only,
    query_df,
)
from app.guardrails.no_adhoc_agg import install_strict_views_only_guard


def test_views_only_rejects_non_view_table() -> None:
    """query_df with FROM schema.non_view_table raises RuntimeError with the required message."""
    config = {"db_path": "/nonexistent/db.duckdb", "schema": "analytics"}
    sql_bad = 'SELECT * FROM "analytics"."agg_firm_monthly"'
    with pytest.raises(RuntimeError) as exc_info:
        query_df(sql_bad, _config=config)
    assert "v_* views only" in str(exc_info.value)
    assert "create_views.py" in str(exc_info.value)
    assert VIEWS_ONLY_MESSAGE in str(exc_info.value)


def test_views_only_rejects_table_without_v_prefix() -> None:
    """_validate_views_only rejects schema.other_table (no v_ prefix)."""
    sql = 'SELECT * FROM "analytics"."other_table"'
    with pytest.raises(RuntimeError) as exc_info:
        _validate_views_only(sql, "analytics")
    assert VIEWS_ONLY_MESSAGE in str(exc_info.value)


def test_views_only_accepts_view() -> None:
    """query_df with FROM schema.v_* passes validation; returns DataFrame when connection is mocked."""
    config = {
        "db_path": "/fake/path.duckdb",
        "schema": "analytics",
        "dataset_version": "test",
    }
    sql = 'SELECT * FROM "analytics"."v_firm_monthly"'
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchdf.return_value = pd.DataFrame({"a": [1]})

    with patch("app.data_gateway.get_connection", return_value=mock_conn):
        result = query_df(sql, _config=config)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1]
    mock_conn.execute.assert_called_once()


def test_strict_views_only_ban_groupby(monkeypatch: pytest.MonkeyPatch) -> None:
    """When STRICT_VIEWS_ONLY=1 and install_strict_views_only_guard() is applied, groupby raises."""
    monkeypatch.setenv("STRICT_VIEWS_ONLY", "1")
    orig_groupby = pd.DataFrame.groupby
    orig_merge_df = pd.DataFrame.merge
    orig_merge_pd = getattr(pd, "merge", None)
    try:
        install_strict_views_only_guard()
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        with pytest.raises(RuntimeError) as exc_info:
            df.groupby("a").sum()
        assert "agg tables" in str(exc_info.value).lower() or "pipelines" in str(exc_info.value)
    finally:
        pd.DataFrame.groupby = orig_groupby
        pd.DataFrame.merge = orig_merge_df
        if orig_merge_pd is not None:
            pd.merge = orig_merge_pd
        monkeypatch.delenv("STRICT_VIEWS_ONLY", raising=False)
