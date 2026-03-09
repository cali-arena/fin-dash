"""
Tests for app.startup.prewarm: prewarm_enabled, prewarm_common calls gateway (mocked).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_prewarm_enabled_when_prewarm_env_set() -> None:
    """PREWARM=1 -> prewarm_enabled() True."""
    with patch.dict("os.environ", {"PREWARM": "1"}):
        from app.startup.prewarm import prewarm_enabled
        assert prewarm_enabled() is True


def test_prewarm_enabled_false_when_unset() -> None:
    """PREWARM not set -> prewarm_enabled() False."""
    with patch.dict("os.environ", {"PREWARM": "0"}):
        from app.startup.prewarm import prewarm_enabled
        assert prewarm_enabled() is False


def test_prewarm_common_calls_run_query_run_aggregate_run_chart() -> None:
    """prewarm_common calls run_query, run_aggregate, run_chart with expected args."""
    from app.startup.prewarm import prewarm_common

    mock_query = MagicMock()
    mock_agg = MagicMock()
    mock_chart = MagicMock()

    with patch("app.data_gateway.run_query", mock_query), \
         patch("app.data_gateway.run_aggregate", mock_agg), \
         patch("app.data_gateway.run_chart", mock_chart):
        result = prewarm_common(root=Path.cwd())

    assert result.get("ok") is True
    mock_query.assert_called_once()
    mock_agg.assert_called_once()
    mock_chart.assert_called_once()

    q_args, q_kw = mock_query.call_args
    assert q_args[0] == "firm_monthly"
    assert "month_end_range" in q_args[1]
