from __future__ import annotations

from datetime import date

from app.nlq.parser import ParseError, parse_nlq


def test_parse_show_nnb_by_channel_returns_bar_chart_spec() -> None:
    metric_reg = {
        "version": "1.0",
        "metrics": [
            {
                "metric_id": "nnb",
                "label": "Net New Business",
                "synonyms": ["net new business", "nnb", "flows"],
                "formula": "column:nnb",
                "default_agg": "sum",
                "grain": "month_end",
                "format": "currency",
                "allowed_dims": ["channel"],
                "default_chart": "bar",
            }
        ],
    }
    dim_reg = {
        "version": "1.0",
        "dimensions": {
            "channel": {"label": "Channel", "column": "channel", "type": "string", "synonyms": ["channel"]},
            "month_end": {"label": "Month End", "column": "month_end", "type": "date", "synonyms": ["month"]},
        },
        "aliases": {"distribution": "channel"},
    }

    spec_or_error = parse_nlq(
        "Show net new business by channel",
        metric_reg,
        dim_reg,
        value_catalog={},
        today=date(2026, 3, 7),
    )

    assert not isinstance(spec_or_error, ParseError)
    assert spec_or_error.metric_id == "nnb"
    assert spec_or_error.dimensions == ["channel"]
    assert spec_or_error.chart.type == "bar"
    assert spec_or_error.chart.x == "channel"
    assert spec_or_error.chart.y == "metric"
