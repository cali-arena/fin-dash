"""
Tests for NLQ executor: QueryResult outputs, safety gates, parameterization, allowlist, row cap.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.nlq.executor import (
    DEFAULT_LIMIT_TOP,
    MAX_ROWS,
    QueryResult,
    execute_queryspec,
)
from app.nlq.governance import GovernanceError
from models.query_spec import ChartSpec, QuerySpec, SortSpec, TimeRange


# Synthetic fact: columns matching metric_reg / dim_reg (canonical keys)
def _synthetic_fact() -> pd.DataFrame:
    return pd.DataFrame({
        "month_end": pd.to_datetime([date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)] * 4),
        "channel": ["Institutional", "Retail", "Institutional", "Retail"] * 3,
        "product_ticker": ["SPY", "SPY", "QQQ", "QQQ"] * 3,
        "src_country": ["US", "US", "EMEA", "EMEA"] * 3,
        "end_aum": [100.0, 110.0, 105.0, 115.0, 102.0, 112.0, 108.0, 118.0, 101.0, 111.0, 106.0, 116.0],
        "begin_aum": [95.0, 105.0, 100.0, 110.0, 98.0, 108.0, 103.0, 113.0, 99.0, 109.0, 104.0, 114.0],
        "nnb": [5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0, 3.0, 2.0, 2.0, 2.0],
        "nnf": [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.3, 0.2, 0.2, 0.2],
    })


# Minimal registries: end_aum (column), nnb (sum), channel/product_ticker/src_country
EXECUTOR_METRIC_REG = {
    "version": "1.0",
    "metrics": [
        {
            "metric_id": "end_aum",
            "label": "End AUM",
            "formula": "column:end_aum",
            "default_agg": "last",
            "grain": "month_end",
            "format": "currency",
            "allowed_dims": ["channel", "product_ticker", "src_country"],
            "default_chart": "line",
        },
        {
            "metric_id": "nnb",
            "label": "Net New Business",
            "formula": "column:nnb",
            "default_agg": "sum",
            "grain": "month_end",
            "format": "currency",
            "allowed_dims": ["channel", "product_ticker", "src_country"],
            "default_chart": "bar",
        },
    ],
}

EXECUTOR_DIM_REG = {
    "version": "1.0",
    "dimensions": {
        "channel": {"label": "Channel", "column": "channel", "type": "string"},
        "product_ticker": {"label": "Ticker", "column": "product_ticker", "type": "string"},
        "src_country": {"label": "Source Country", "column": "src_country", "type": "string"},
        "month_end": {"label": "Month End", "column": "month_end", "type": "date"},
    },
    "aliases": {},
}


def test_headline_query_no_dims_returns_single_row_and_numbers() -> None:
    """Headline query (no dims) returns single-row data and numbers present."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=[],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=50,
        chart=ChartSpec(type="table"),
    )
    allowlist = {"max_rows": MAX_ROWS}
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    assert isinstance(result, QueryResult)
    assert len(result.data) == 1
    assert "metric" in result.data.columns
    assert result.numbers.get("metric_id") == "end_aum"
    assert "value" in result.numbers
    assert "formatted" in result.numbers
    assert result.chart_spec == {"type": "table"}
    assert "metric_id" in result.explain_context
    assert "row_count" in result.explain_context
    assert result.explain_context["row_count"] == 1


def test_breakdown_by_channel_returns_within_default_limit() -> None:
    """Breakdown by channel returns <= DEFAULT_LIMIT_TOP when limit set to that."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="nnb",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=DEFAULT_LIMIT_TOP,
        chart=ChartSpec(type="bar", x="channel", y="metric"),
    )
    allowlist = {"max_rows": MAX_ROWS}
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    assert len(result.data) <= DEFAULT_LIMIT_TOP
    assert list(result.data.columns) == ["channel", "metric"]
    assert result.chart_spec["type"] == "bar"
    assert result.chart_spec["x"] == "channel"
    assert result.chart_spec["y"] == "metric"
    assert "top_entities" in result.explain_context
    assert result.explain_context["dims_used"] == ["channel"]


def test_trend_query_returns_month_end_series_ordered_asc() -> None:
    """Trend query (dimensions including time or series over month_end) returns month_end ordered asc when sort asc."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="asc"),
        limit=50,
        chart=ChartSpec(type="line", x="month_end", y="metric", series="channel"),
    )
    allowlist = {"max_rows": MAX_ROWS}
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    assert "metric" in result.data.columns
    assert result.chart_spec["type"] == "line"
    assert result.chart_spec["x"] == "month_end"
    assert result.chart_spec["y"] == "metric"
    assert result.chart_spec["series"] == "channel"
    # Sort is by metric asc; data should be ordered
    if len(result.data) >= 2:
        assert result.data["metric"].iloc[0] <= result.data["metric"].iloc[-1]


def test_blocked_pii_column_raises_value_error() -> None:
    """Plan that would select a blocked (PII) column raises ValueError before execution."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=20,
        chart=ChartSpec(type="table"),
    )
    allowlist = {"pii_columns": ["channel"], "max_rows": MAX_ROWS}
    with pytest.raises(ValueError, match="blocked.*PII|Plan selects blocked"):
        execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)


def test_unknown_dimension_in_registry_raises_governance_error() -> None:
    """Spec with dimension not in dim_registry triggers GovernanceError before execution."""
    df = _synthetic_fact()
    # segment not in EXECUTOR_DIM_REG dimensions
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["segment"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=20,
        chart=ChartSpec(type="table"),
    )
    with pytest.raises(GovernanceError):
        execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, {})


def test_allowlist_columns_not_in_plan_raises() -> None:
    """Plan that selects a dimension not in allowlist columns raises ValueError."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=20,
        chart=ChartSpec(type="table"),
    )
    allowlist = {"columns": ["metric"], "max_rows": MAX_ROWS}
    with pytest.raises(ValueError, match="not in allowlist|Plan selects columns"):
        execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)


def test_row_cap_enforced_meta_shows_clamped() -> None:
    """Requesting high limit with low max_rows: result capped and meta shows clamped."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=500,
        chart=ChartSpec(type="table"),
    )
    allowlist = {"max_rows": 2}
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    assert result.meta["row_cap"] == 2
    assert result.meta["applied_limit"] == 2
    assert len(result.data) <= 2
    assert any("clamp" in w.lower() for w in result.meta.get("warnings", []))


def test_params_used_no_sql_injection() -> None:
    """Filters are applied via values from spec only; SQL uses parameter placeholders (params)."""
    from app.nlq.executor import _build_where_and_params

    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel", "src_country"],
        filters={"channel": ["Institutional"], "src_country": ["US", "EMEA"]},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 6, 30)),
        sort=SortSpec(by="metric", order="desc"),
        limit=50,
        chart=ChartSpec(type="table"),
    )
    parts, params = _build_where_and_params(qs)
    assert "?" in " ".join(parts)
    assert date(2024, 1, 1) in params
    assert date(2024, 6, 30) in params
    assert "Institutional" in params
    assert "US" in params
    assert "EMEA" in params
    assert len(params) == 5
    # No user string concatenation: placeholders only
    for p in parts:
        assert "Institutional" not in p
        assert "US" not in p


def test_stable_sort_ties_deterministic_order() -> None:
    """Stable sorting: same metric value yields deterministic row order (e.g. by index)."""
    # Craft data: two channels with identical metric so tie-break matters
    df = pd.DataFrame({
        "month_end": pd.to_datetime([date(2024, 3, 31)] * 4),
        "channel": ["A", "B", "C", "D"],
        "end_aum": [100.0, 100.0, 99.0, 98.0],
    })
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=10,
        chart=ChartSpec(type="table"),
    )
    allowlist = {"max_rows": MAX_ROWS}
    r1 = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    r2 = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, allowlist)
    order1 = list(r1.data["channel"].values)
    order2 = list(r2.data["channel"].values)
    assert order1 == order2, "Stable sort: same input must produce same order"
    # First two rows are tie (100.0); order should be stable (A then B from original df)
    assert r1.data["metric"].iloc[0] == r1.data["metric"].iloc[1] == 100.0
    assert order1[0] == "A" and order1[1] == "B"


def test_query_result_always_has_chart_spec_and_explain_context() -> None:
    """QueryResult always includes chart_spec and explain_context."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="end_aum",
        dimensions=[],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=50,
        chart=ChartSpec(type="line", x="month_end", y="metric"),
    )
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, {})
    assert hasattr(result, "chart_spec")
    assert isinstance(result.chart_spec, dict)
    assert "type" in result.chart_spec
    assert hasattr(result, "explain_context")
    assert isinstance(result.explain_context, dict)
    assert "metric_id" in result.explain_context
    assert "filters_applied" in result.explain_context
    assert "row_count" in result.explain_context


def test_table_chart_with_dimensions_auto_falls_back_to_bar() -> None:
    """Dimensional queries should auto-render a chart even if chart type is table."""
    df = _synthetic_fact()
    qs = QuerySpec(
        metric_id="nnb",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 3, 31)),
        sort=SortSpec(by="metric", order="desc"),
        limit=20,
        chart=ChartSpec(type="table"),
    )
    result = execute_queryspec(qs, df, EXECUTOR_METRIC_REG, EXECUTOR_DIM_REG, {"max_rows": MAX_ROWS})
    assert result.chart_spec["type"] == "bar"
    assert result.chart_spec["x"] == "channel"
    assert result.chart_spec["y"] == "metric"
