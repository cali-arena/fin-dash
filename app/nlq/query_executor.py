"""
NLQ query executor: orchestrate intent-based execution.
- data_question: run deterministic query on processed dataset (Python only); return aggregated table, metrics, chart-ready dataframe.
- market_intelligence: return query payload for LLM reasoning layer (no raw data passed to LLM).
LLM never receives raw dataset; only verified results or the question text for external search.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from app.nlq.intent_classifier import IntentResult, classify_intent, is_data_question
from app.nlq.parameter_extractor import ExtractedParams, extract_parameters

# Optional imports for data path (parser + executor)
try:
    from app.nlq.executor import QueryResult, execute_queryspec
    from app.nlq.governance import GovernanceError, load_dim_registry, load_metric_registry, validate_queryspec
    from app.nlq.parser import ParseError, parse_nlq
    _HAS_DATA_PATH = True
except ImportError:
    _HAS_DATA_PATH = False


@dataclass
class MarketQueryPayload:
    """Payload for market_intelligence: question text and optional context hint for LLM layer. No raw data."""
    query_text: str
    context_hint: str = ""


@dataclass
class DataQueryResult:
    """Result of a deterministic data query: table, metrics, chart spec."""
    data: pd.DataFrame
    metrics: dict[str, Any]
    chart_spec: dict[str, Any]
    meta: dict[str, Any]
    raw_result: Any = None  # QueryResult when from execute_queryspec


@dataclass
class ExecutorResult:
    """Union result: either data query result or market query payload."""
    intent: str  # "data_question" | "market_intelligence"
    data_result: DataQueryResult | None = None
    market_payload: MarketQueryPayload | None = None
    error: str | None = None
    parse_error: Any = None  # ParseError if parse failed


def run_data_query(
    query_text: str,
    *,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    value_catalog: dict[str, set[str]],
    df: pd.DataFrame,
    allowlist: dict[str, Any],
    today: date | None = None,
) -> tuple[DataQueryResult | None, ParseError | None, str | None]:
    """
    Run deterministic data query: parse NLQ -> QuerySpec, execute, return aggregated result.
    Returns (DataQueryResult, None, None) on success; (None, ParseError, None) on parse error; (None, None, error_msg) on execution error.
    """
    if not _HAS_DATA_PATH:
        return None, None, "Data query path unavailable (missing parser/executor)."
    today = today or date.today()
    params = extract_parameters(query_text, value_catalog=value_catalog, today=today)
    spec_or_err = parse_nlq(query_text, metric_reg, dim_reg, value_catalog, today=today)
    if isinstance(spec_or_err, ParseError):
        return None, spec_or_err, None
    qs = spec_or_err
    try:
        validate_queryspec(qs, metric_reg, dim_reg, out_logs=None)
    except GovernanceError as e:
        return None, None, str(e)
    try:
        result = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=False)
    except (ValueError, GovernanceError) as e:
        return None, None, str(e)
    if result is None:
        return None, None, "Executor returned no result."
    data = result.data if hasattr(result, "data") else pd.DataFrame()
    metrics = (result.numbers or {}) if hasattr(result, "numbers") else {}
    chart_spec = (result.chart_spec or {}) if hasattr(result, "chart_spec") else {}
    meta = (result.meta or {}) if hasattr(result, "meta") else {}
    return DataQueryResult(data=data, metrics=metrics, chart_spec=chart_spec, meta=meta, raw_result=result), None, None


def run_intent(
    query_text: str,
    *,
    intent_override: str | None = None,
    prefer_data_mode: bool = True,
    metric_reg: dict[str, Any] | None = None,
    dim_reg: dict[str, Any] | None = None,
    value_catalog: dict[str, set[str]] | None = None,
    df: pd.DataFrame | None = None,
    allowlist: dict[str, Any] | None = None,
    today: date | None = None,
) -> ExecutorResult:
    """
    Classify intent and run the appropriate path.
    - If data_question: run deterministic query; return ExecutorResult with data_result set.
    - If market_intelligence: return ExecutorResult with market_payload set (query for LLM layer; no raw data).
    """
    query_text = (query_text or "").strip()
    if not query_text:
        return ExecutorResult(intent="data_question", error="Empty query.")

    # Classify intent
    if intent_override in ("data_question", "market_intelligence"):
        intent = intent_override
        intent_result = None
    else:
        intent_result = classify_intent(query_text)
        intent = intent_result.intent
        if intent == "ambiguous":
            intent = "data_question" if prefer_data_mode else "market_intelligence"

    # Market intelligence path: no raw data, only payload for LLM
    if intent == "market_intelligence":
        hint = (intent_result.reason if intent_result else "External market question.")[:200]
        return ExecutorResult(
            intent="market_intelligence",
            market_payload=MarketQueryPayload(query_text=query_text, context_hint=hint),
        )

    # Data question path: run deterministic query
    if not _HAS_DATA_PATH or metric_reg is None or dim_reg is None or df is None:
        return ExecutorResult(
            intent="data_question",
            error="Data path not configured (metric_reg, dim_reg, df required).",
        )
    value_catalog = value_catalog or {}
    allowlist = allowlist or {}
    today = today or date.today()
    data_result, parse_error, exec_error = run_data_query(
        query_text,
        metric_reg=metric_reg,
        dim_reg=dim_reg,
        value_catalog=value_catalog,
        df=df,
        allowlist=allowlist,
        today=today,
    )
    if parse_error is not None:
        return ExecutorResult(intent="data_question", parse_error=parse_error, error=parse_error.message)
    if exec_error is not None:
        return ExecutorResult(intent="data_question", error=exec_error)
    if data_result is None:
        return ExecutorResult(intent="data_question", error="No result from data query.")
    return ExecutorResult(intent="data_question", data_result=data_result)
