"""
Tests for app.perf_budget: get_budget_ms, check_perf_budget (pure logic; no Streamlit runtime).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.perf_budget import (
    PerfBudgetConfig,
    check_perf_budget,
    get_budget_ms,
    load_perf_budget,
)


def test_get_budget_ms_level_a_firm() -> None:
    """Level A firm_monthly -> firm_monthly_ms budget."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    assert get_budget_ms("A", "firm_monthly", cfg) == 200


def test_get_budget_ms_level_a_channel() -> None:
    """Level A channel_monthly -> channel_monthly_ms budget."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    assert get_budget_ms("A", "channel_monthly", cfg) == 500


def test_get_budget_ms_level_a_unknown_defaults_channel() -> None:
    """Level A unknown query -> channel budget as default."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    assert get_budget_ms("A", "other_query", cfg) == 500


def test_get_budget_ms_level_c() -> None:
    """Level C -> heavy_chart_ms budget."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    assert get_budget_ms("C", "waterfall", cfg) == 1000


def test_check_perf_budget_under_budget_no_raise() -> None:
    """Elapsed under budget -> no raise."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    check_perf_budget("A", "firm_monthly", 100.0, config=cfg)


def test_check_perf_budget_exceeded_dev_fail_raises() -> None:
    """Elapsed over budget + DEV_MODE + fail_in_dev -> RuntimeError."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    with patch.dict("os.environ", {"DEV_MODE": "1"}):
        with pytest.raises(RuntimeError) as exc_info:
            check_perf_budget("A", "firm_monthly", 250.0, config=cfg)
        assert "Perf budget exceeded" in str(exc_info.value)
        assert "200ms" in str(exc_info.value)


def test_check_perf_budget_exceeded_not_dev_no_raise() -> None:
    """Elapsed over budget but not dev mode -> no raise (just logs)."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=True,
    )
    with patch.dict("os.environ", {"DEV_MODE": "0"}):
        check_perf_budget("A", "firm_monthly", 250.0, config=cfg)


def test_check_perf_budget_fail_in_dev_false_no_raise() -> None:
    """fail_in_dev=false -> no raise even when dev and exceeded."""
    cfg = PerfBudgetConfig(
        firm_monthly_ms=200,
        channel_monthly_ms=500,
        ticker_monthly_ms=500,
        geo_monthly_ms=500,
        segment_monthly_ms=500,
        heavy_chart_ms=1000,
        warn_multiplier=1.25,
        fail_in_dev=False,
    )
    with patch.dict("os.environ", {"DEV_MODE": "1"}):
        check_perf_budget("A", "firm_monthly", 250.0, config=cfg)


def test_load_perf_budget_from_file(tmp_path: Path) -> None:
    """Load config from YAML; validate values."""
    yml = tmp_path / "perf_budget.yml"
    yml.write_text("""
perf_budget:
  firm_monthly_ms: 150
  channel_monthly_ms: 400
  ticker_monthly_ms: 400
  geo_monthly_ms: 400
  segment_monthly_ms: 400
  heavy_chart_ms: 800
  warn_multiplier: 1.5
  fail_in_dev: false
""", encoding="utf-8")
    cfg = load_perf_budget(path=yml, root=None)
    assert cfg.firm_monthly_ms == 150
    assert cfg.channel_monthly_ms == 400
    assert cfg.heavy_chart_ms == 800
    assert cfg.warn_multiplier == 1.5
    assert cfg.fail_in_dev is False


def test_load_perf_budget_missing_uses_defaults(tmp_path: Path) -> None:
    """Missing file -> default config (use _load_from_path to avoid cache)."""
    from app.perf_budget import _load_from_path
    missing = tmp_path / "perf_budget_nonexistent_12345.yml"
    assert not missing.exists()
    cfg = _load_from_path(missing)
    assert cfg.firm_monthly_ms == 200
    assert cfg.heavy_chart_ms == 1000
