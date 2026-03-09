"""
Tests for QuerySpec schema and governance validation.
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.nlq.governance import GovernanceError, validate_queryspec
from models.query_spec import ChartSpec, QuerySpec, SortSpec, TimeRange


# Minimal registries for tests
MINIMAL_METRIC_REG = {
    "version": "1.0",
    "metrics": [
        {
            "metric_id": "end_aum",
            "label": "End AUM",
            "formula": "column:end_aum",
            "default_agg": "last",
            "grain": "month_end",
            "format": "currency",
            "allowed_dims": ["channel", "product_ticker", "src_country", "segment"],
            "default_chart": "line",
        },
        {
            "metric_id": "firm_only_metric",
            "label": "Firm Only",
            "formula": "column:ytd_pct",
            "default_agg": "last",
            "grain": "month_end",
            "format": "percent",
            "allowed_dims": [],
            "default_chart": "table",
        },
    ],
}

MINIMAL_DIM_REG = {
    "version": "1.0",
    "dimensions": {
        "channel": {"label": "Channel", "column": "standard_channel", "type": "string", "synonyms": ["channel"], "allowed_grains": ["month_end", "channel"]},
        "product_ticker": {"label": "Ticker", "column": "product_ticker", "type": "string", "synonyms": ["ticker"], "allowed_grains": ["month_end", "ticker"]},
        "month_end": {"label": "Month End", "column": "month_end", "type": "date", "synonyms": ["month"], "allowed_grains": ["month_end"]},
    },
    "aliases": {"ticker": "product_ticker"},
}


def test_valid_spec_passes() -> None:
    spec = QuerySpec(
        metric_id="end_aum",
        dimensions=["channel"],
        filters={},
        time_range=TimeRange(start=date(2024, 1, 1), end=date(2024, 6, 30)),
        sort=SortSpec(by="metric", order="desc"),
        limit=50,
        chart=ChartSpec(type="table"),
    )
    validate_queryspec(spec, MINIMAL_METRIC_REG, MINIMAL_DIM_REG)


def test_unknown_metric_id_fails() -> None:
    spec = QuerySpec(metric_id="unknown_metric", dimensions=[], filters={})
    with pytest.raises(GovernanceError) as exc_info:
        validate_queryspec(spec, MINIMAL_METRIC_REG, MINIMAL_DIM_REG)
    assert "not found in metric_registry" in str(exc_info.value)
    assert "unknown_metric" in str(exc_info.value)


def test_dimension_not_allowed_by_metric_fails() -> None:
    # end_aum allows channel, product_ticker, src_country, segment; not "other_dim"
    spec = QuerySpec(
        metric_id="end_aum",
        dimensions=["other_dim"],
        filters={},
    )
    with pytest.raises(GovernanceError) as exc_info:
        validate_queryspec(spec, MINIMAL_METRIC_REG, MINIMAL_DIM_REG)
    assert "not found in dim_registry" in str(exc_info.value) or "not in" in str(exc_info.value)


def test_dimension_not_in_metric_allowed_dims_fails() -> None:
    # Use a metric that only allows channel; pass product_ticker (exists in dim_reg but not allowed)
    reg_metric_channel_only = {
        "version": "1.0",
        "metrics": [
            {
                "metric_id": "channel_aum",
                "label": "Channel AUM",
                "formula": "column:end_aum",
                "default_agg": "sum",
                "grain": "channel",
                "format": "currency",
                "allowed_dims": ["channel"],
                "default_chart": "bar",
            },
        ],
    }
    spec = QuerySpec(
        metric_id="channel_aum",
        dimensions=["product_ticker"],
        filters={},
    )
    with pytest.raises(GovernanceError) as exc_info:
        validate_queryspec(spec, reg_metric_channel_only, MINIMAL_DIM_REG)
    assert "allowed_dims" in str(exc_info.value) or "not in" in str(exc_info.value)


def test_extra_keys_forbidden_fails() -> None:
    with pytest.raises(ValidationError):
        QuerySpec(
            metric_id="end_aum",
            dimensions=[],
            filters={},
            extra_forbidden_key="not_allowed",
        )


def test_firm_only_metric_rejects_dimensions() -> None:
    spec = QuerySpec(
        metric_id="firm_only_metric",
        dimensions=["channel"],
        filters={},
    )
    with pytest.raises(GovernanceError) as exc_info:
        validate_queryspec(spec, MINIMAL_METRIC_REG, MINIMAL_DIM_REG)
    assert "firm-only" in str(exc_info.value) or "allowed_dims empty" in str(exc_info.value)


def test_time_range_start_after_end_fails() -> None:
    spec = QuerySpec(
        metric_id="end_aum",
        dimensions=[],
        filters={},
        time_range=TimeRange(start=date(2024, 12, 1), end=date(2024, 1, 1)),
    )
    with pytest.raises(GovernanceError) as exc_info:
        validate_queryspec(spec, MINIMAL_METRIC_REG, MINIMAL_DIM_REG)
    assert "start" in str(exc_info.value) and "end" in str(exc_info.value)
