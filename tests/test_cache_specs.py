"""
Tests for app.cache.specs: deterministic ordering, validation errors, JSON-serializable outputs.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app.cache.specs import (
    AGG_CHANNEL_MIX,
    AGG_KPI_CARDS,
    AGG_SPECS,
    AGG_TOPN_TICKERS,
    CHART_SPECS,
    build_corr_matrix_payload,
    build_waterfall_payload,
    compute_channel_mix,
    compute_kpi_cards,
    compute_rolling_avg,
    compute_topn_tickers,
    validate_agg_name,
    validate_chart_name,
)


def test_validate_agg_name_raises_for_unknown() -> None:
    """Unknown agg_name raises ValueError with message to add spec."""
    with pytest.raises(ValueError) as exc_info:
        validate_agg_name("unknown_agg")
    assert "Invalid agg_name" in str(exc_info.value)
    assert "Add a spec" in str(exc_info.value) or "AGG_SPECS" in str(exc_info.value)


def test_validate_agg_name_allows_registered() -> None:
    """Registered agg names do not raise."""
    for name in AGG_SPECS:
        validate_agg_name(name)


def test_validate_chart_name_raises_for_unknown() -> None:
    """Unknown chart_name raises ValueError with message to add spec."""
    with pytest.raises(ValueError) as exc_info:
        validate_chart_name("unknown_chart")
    assert "Invalid chart_name" in str(exc_info.value)
    assert "Add a spec" in str(exc_info.value) or "CHART_SPECS" in str(exc_info.value)


def test_validate_chart_name_allows_registered() -> None:
    """Registered chart names do not raise."""
    for name in CHART_SPECS:
        validate_chart_name(name)


def test_topn_tickers_deterministic_ordering() -> None:
    """Ties broken by ticker name (asc); same input -> same output order."""
    df = pd.DataFrame({
        "product_ticker": ["B", "A", "C", "A"],
        "end_aum": [100.0, 50.0, 50.0, 50.0],
        "nnb": [10.0, 5.0, 5.0, 5.0],
    })
    out1 = compute_topn_tickers(df, {"top_n": 5, "by": "end_aum"})
    out2 = compute_topn_tickers(df, {"top_n": 5, "by": "end_aum"})
    assert out1["tickers"] == out2["tickers"]
    assert out1["values"] == out2["values"]
    # After groupby: A=100, B=100, C=50. Sort value desc then ticker asc -> A, B, C
    assert out1["tickers"][0] == "A"
    assert out1["tickers"][1] == "B"
    assert out1["tickers"][2] == "C"
    assert out1["values"][:2] == [100.0, 100.0]
    assert out1["values"][2] == 50.0


def test_channel_mix_deterministic_ordering() -> None:
    """Channels sorted by name; same input -> same output order."""
    base = pd.Timestamp("2024-06-01")
    df = pd.DataFrame({
        "month_end": [base, base, base],
        "channel_l1": ["C", "A", "B"],
        "end_aum": [30.0, 10.0, 20.0],
    })
    out1 = compute_channel_mix(df)
    out2 = compute_channel_mix(df)
    assert out1["channels"] == out2["channels"]
    assert out1["channels"] == ["A", "B", "C"]
    assert out1["shares"] == pytest.approx([10 / 60, 20 / 60, 30 / 60])


def test_rolling_avg_stable_column_order() -> None:
    """Rolling 3m output has stable keys and list order."""
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")],
        "end_aum": [100.0, 110.0, 120.0],
    })
    out = compute_rolling_avg(df)
    assert "month_end" in out and "end_aum" in out and "rolling_3m" in out
    assert len(out["month_end"]) == 3
    assert out["rolling_3m"][0] == 100.0
    assert out["rolling_3m"][1] == 105.0
    assert out["rolling_3m"][2] == 110.0


def test_kpi_cards_stable_key_order() -> None:
    """KPI keys in fixed order; row_count first."""
    df = pd.DataFrame({
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    out = compute_kpi_cards(df)
    assert out["row_count"] == 1
    assert out["begin_aum"] == 90.0
    assert out["end_aum"] == 100.0


def test_chart_payloads_json_serializable() -> None:
    """Chart payloads are JSON-serializable (no Timestamp/numpy)."""
    df = pd.DataFrame({
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    payload = build_waterfall_payload(df)
    json.dumps(payload)

    df2 = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "end_aum": [100.0], "nnb": [10.0],
    })
    corr_payload = build_corr_matrix_payload(df2)
    json.dumps(corr_payload)


def test_corr_matrix_deterministic_column_order() -> None:
    """Correlation matrix columns in sorted order."""
    df = pd.DataFrame({
        "z_col": [1.0, 2.0], "a_col": [3.0, 4.0], "m_col": [5.0, 6.0],
    })
    out = build_corr_matrix_payload(df)
    assert out.get("columns") == ["a_col", "m_col", "z_col"]
    assert list(out.get("data", {}).keys()) == ["a_col", "m_col", "z_col"]


def test_pyramid_unknown_agg_raises() -> None:
    """get_aggregate with unknown agg_name raises ValueError (validation via specs)."""
    from unittest.mock import patch
    from app.cache import pyramid as pyramid_mod
    with patch("app.data_gateway._run_query_uncached", return_value=pd.DataFrame({"end_aum": [100.0]})):
        with pytest.raises(ValueError) as exc_info:
            pyramid_mod.get_aggregate(
                "v1", "nonexistent_agg", "firm_monthly", "h" * 40, "{}", None
            )
    assert "Invalid agg_name" in str(exc_info.value) or "nonexistent_agg" in str(exc_info.value)


def test_pyramid_unknown_chart_raises() -> None:
    """get_chart_payload with unknown chart_name raises ValueError (validation via specs)."""
    from unittest.mock import patch
    from app.cache import pyramid as pyramid_mod
    with patch("app.data_gateway._run_query_uncached", return_value=pd.DataFrame({"end_aum": [100.0]})):
        with pytest.raises(ValueError) as exc_info:
            pyramid_mod.get_chart_payload(
                "v1", "nonexistent_chart", "kpi_cards", "firm_monthly", "h" * 40, "{}", None
            )
    assert "Invalid chart_name" in str(exc_info.value) or "nonexistent_chart" in str(exc_info.value)
