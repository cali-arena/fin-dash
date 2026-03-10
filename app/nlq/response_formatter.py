"""
NLQ response formatter: build a single response object for the Intelligence Desk.
Contains response_text, optional_table, optional_chart_data. No raw data passed to LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.nlq.query_executor import DataQueryResult, ExecutorResult, MarketQueryPayload


@dataclass
class FormattedNLQResponse:
    """Formatted response for the NLQ UI: text, optional table, optional chart payload."""
    response_text: str
    optional_table: pd.DataFrame | None = None
    optional_chart_data: dict[str, Any] | None = None
    intent: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _dataframe_to_chart_payload(df: pd.DataFrame, chart_spec: dict[str, Any]) -> dict[str, Any]:
    """Build chart-ready payload: x, y, series, type, and optional data columns."""
    if df is None or df.empty:
        return {"type": "table", "data": []}
    chart_type = chart_spec.get("type") or "table"
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    series_col = chart_spec.get("series")
    out: dict[str, Any] = {
        "type": chart_type,
        "columns": list(df.columns),
        "row_count": len(df),
    }
    if x_col and x_col in df.columns:
        out["x"] = x_col
        out["x_values"] = df[x_col].astype(str).tolist()
    if y_col and y_col in df.columns:
        out["y"] = y_col
        out["y_values"] = df[y_col].tolist()
    if series_col and series_col in df.columns:
        out["series"] = series_col
        out["series_values"] = df[series_col].astype(str).tolist()
    # Optional: small preview for UI (no full dataset to LLM)
    out["preview_rows"] = df.head(10).to_dict(orient="records") if len(df) > 0 else []
    return out


def format_data_result(
    data_result: DataQueryResult,
    *,
    narrative_text: str = "",
    include_table: bool = True,
    include_chart_data: bool = True,
) -> FormattedNLQResponse:
    """
    Format a DataQueryResult into FormattedNLQResponse.
    response_text = narrative_text or a short summary from metrics; optional_table/optional_chart_data from result.
    """
    text = narrative_text.strip() if narrative_text else ""
    if not text and data_result.metrics:
        parts = [f"{k}: {v}" for k, v in list(data_result.metrics.items())[:5]]
        text = "Verified result. " + ("; ".join(parts) if parts else "No headline metrics.")
    if not text:
        text = "Query completed. See table and chart below."

    optional_table = None
    if include_table and data_result.data is not None and not data_result.data.empty:
        optional_table = data_result.data

    optional_chart_data = None
    if include_chart_data and data_result.data is not None and data_result.chart_spec:
        optional_chart_data = _dataframe_to_chart_payload(data_result.data, data_result.chart_spec)

    return FormattedNLQResponse(
        response_text=text,
        optional_table=optional_table,
        optional_chart_data=optional_chart_data,
        intent="data_question",
        meta=data_result.meta,
    )


def format_market_payload(
    market_payload: MarketQueryPayload,
    *,
    response_text: str = "",
) -> FormattedNLQResponse:
    """
    Format market intelligence path: response_text is the LLM-generated answer; no table/chart from executor.
    """
    text = (response_text or "").strip() or "Market intelligence query ready for LLM. No internal data included."
    return FormattedNLQResponse(
        response_text=text,
        optional_table=None,
        optional_chart_data=None,
        intent="market_intelligence",
        meta={"query_text": market_payload.query_text, "context_hint": market_payload.context_hint},
    )


def format_executor_result(
    result: ExecutorResult,
    *,
    narrative_text: str = "",
    market_response_text: str = "",
) -> FormattedNLQResponse:
    """
    Format ExecutorResult into FormattedNLQResponse.
    - If data_result: use format_data_result with optional narrative_text.
    - If market_payload: use format_market_payload with market_response_text (from LLM layer).
    - If error: response_text = error, optional_* = None.
    """
    if result.error:
        return FormattedNLQResponse(
            response_text=result.error,
            optional_table=None,
            optional_chart_data=None,
            intent=result.intent,
            error=result.error,
        )
    if result.data_result is not None:
        return format_data_result(result.data_result, narrative_text=narrative_text)
    if result.market_payload is not None:
        return format_market_payload(result.market_payload, response_text=market_response_text)
    return FormattedNLQResponse(
        response_text="No result or payload.",
        optional_table=None,
        optional_chart_data=None,
        intent=result.intent,
        error="No data_result or market_payload.",
    )
