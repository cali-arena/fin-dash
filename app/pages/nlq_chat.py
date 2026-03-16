"""
Intelligence Desk page (tab "nlq_chat").
Loaded by app/main.py via PAGE_RENDERERS["nlq_chat"] = render_nlq_chat — this is the only implementation.
- Data Questions: governed query, verified result, optional narrative. Chart/table in single response area.
- Market Intelligence: external search + Claude; answer labeled as external.
"""
from __future__ import annotations

import logging
import importlib.util
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import json
import pandas as pd
import plotly.express as px
import streamlit as st

from app.data.data_gateway import Q_CHANNEL_MONTHLY, Q_GEO_MONTHLY, Q_TICKER_MONTHLY, run_query
from app.pages.intelligence_desk_retrieval import retrieve_intelligence_desk_context
from app.nlq.deterministic_summary import build_deterministic_summary
from app.nlq.executor import QueryResult, execute_queryspec
from app.nlq.governance import GovernanceError, load_dim_registry, load_metric_registry, validate_queryspec
from app.nlq.market_search import search_market_context
from app.nlq.parser import ParseError, parse_nlq
from app.state import FilterState, filter_state_to_gateway_dict
from app.nlq.executor import EXECUTOR_TIMEOUT_MS
from app.ui.formatters import fmt_percent, format_df, infer_common_formats
from app.ui.exports import render_export_buttons
from app.ui.guardrails import fallback_note, render_chart_or_fallback, render_empty_state
from app.ui.theme import apply_enterprise_plotly_style, safe_render_plotly
from app.services.intelligence_chat_service import (
    ChatProviderError,
    generate_chat_reply,
    get_provider_status,
)

# Claude client: optional; must not crash app if missing or broken (cloud-safe)
try:
    from app.services.claude_client import (
        ClaudeError,
        anthropic_sdk_available,
        claude_generate,
        claude_generate_grounded,
        has_claude_api_key,
    )
except Exception:
    ClaudeError = Exception  # type: ignore[misc, assignment]
    anthropic_sdk_available = None  # type: ignore[assignment]
    claude_generate = None  # type: ignore[assignment]
    claude_generate_grounded = None  # type: ignore[assignment]
    has_claude_api_key = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
KNOWN_ETF_TICKERS = frozenset({"AGG", "HYG", "TIP", "MUB", "MBB", "IUSB", "SUB"})
CHAT_HISTORY_KEY = "nlq_chat_history"
logger = logging.getLogger(__name__)
BUILD_MARKER = "claude-cloud-deploy-2026-03-13"
CLAUDE_DEBUG_KEY = "nlq_claude_debug"
INTEL_DESK_MODEL = "claude-haiku-4-5"
INTEL_CHAT_HISTORY_KEY = "inteldesk_chat_history_v3"
INTEL_CHAT_INPUT_KEY = "inteldesk_chat_input_v3"
INTEL_LAST_SUBSET_DF_KEY = "inteldesk_last_subset_df"


@dataclass(frozen=True)
class DataIntentExtraction:
    intent: str
    metric_id: str
    dimensions: list[str]
    filters: dict[str, list[str]]
    threshold_op: str | None = None
    threshold_value: float | None = None
    time_start: date | None = None
    time_end: date | None = None


@dataclass(frozen=True)
class QueryRoute:
    route: str  # data_question | market_intelligence | ambiguous
    reason: str


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _classify_query_route(text: str) -> QueryRoute:
    """
    Lightweight routing classifier.
    - data_question: deterministic internal analytics intent
    - market_intelligence: external macro/competitor/sentiment intent
    - ambiguous: mixed signals; fallback to selected mode
    """
    t = _normalize_text(text)
    explicit_market_phrases = (
        "latest outlook",
        "current outlook",
        "major asset classes",
        "rate expectations",
        "market outlook",
    )
    if any(p in t for p in explicit_market_phrases):
        return QueryRoute(route="market_intelligence", reason="question asks for external market outlook")
    market_terms = (
        "fed", "inflation", "macro", "sentiment", "competitor", "competitors",
        "market conditions", "rates", "treasury", "ecb", "boe", "geopolitical",
        "outlook", "news", "external", "policy", "earnings season",
    )
    data_terms = (
        "nnb", "nnf", "ogr", "market impact", "fee yield", "aum", "channel",
        "sub-channel", "sub channel", "country", "segment", "sub-segment",
        "sub segment", "ticker", "etf", "contributors", "flows", "above", "below",
        "ytd", "qoq", "yoy", "1m",
    )
    market_hits = sum(1 for k in market_terms if k in t)
    data_hits = sum(1 for k in data_terms if k in t)
    if market_hits > 0 and data_hits == 0:
        return QueryRoute(route="market_intelligence", reason="question references external market context")
    if data_hits > 0 and market_hits == 0:
        return QueryRoute(route="data_question", reason="question references governed internal metrics/dimensions")
    if market_hits > 0 and data_hits > 0:
        return QueryRoute(route="ambiguous", reason="question mixes internal and external intents")
    return QueryRoute(route="ambiguous", reason="question intent is not explicit")


def _extract_numeric_threshold(text: str) -> tuple[str | None, float | None]:
    """
    Parse simple threshold expressions:
    - above / greater than / over $100k
    - below / less than / under 0.5%
    Returns (op, value in decimal for percent inputs and absolute for currency inputs).
    """
    t = _normalize_text(text)
    m = re.search(
        r"\b(above|greater than|over|below|less than|under)\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([kmb%]?)\s*(?:$|[\s,.;:!?])",
        t,
        re.I,
    )
    if not m:
        return None, None
    op_raw = m.group(1).lower()
    raw_val = float(m.group(2))
    suffix = (m.group(3) or "").lower()
    if suffix == "k":
        raw_val *= 1_000.0
    elif suffix == "m":
        raw_val *= 1_000_000.0
    elif suffix == "b":
        raw_val *= 1_000_000_000.0
    elif suffix == "%":
        raw_val /= 100.0
    op = "gt" if op_raw in ("above", "greater than", "over") else "lt"
    return op, raw_val


def _extract_month_or_quarter_window(text: str, today: date) -> tuple[date | None, date | None]:
    t = _normalize_text(text)
    q = re.search(r"\bq([1-4])\b", t)
    if q:
        qn = int(q.group(1))
        year = today.year
        start_month = (qn - 1) * 3 + 1
        end_month = start_month + 2
        _, last_day = monthrange(year, end_month)
        return date(year, start_month, 1), date(year, end_month, last_day)
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for mname, month_no in month_map.items():
        if re.search(rf"\b{mname}\b", t):
            year = today.year if month_no <= today.month else today.year - 1
            _, last_day = monthrange(year, month_no)
            return date(year, month_no, 1), date(year, month_no, last_day)
    return None, None


def _extract_catalog_filters(text: str, value_catalog: dict[str, set[str]]) -> dict[str, list[str]]:
    t = _normalize_text(text)
    filters: dict[str, list[str]] = {}
    for dim in ("channel", "sub_channel", "product_ticker", "segment", "sub_segment", "src_country", "country"):
        vals = value_catalog.get(dim) or set()
        if not vals:
            continue
        matched: list[str] = []
        for v in sorted(vals):
            sv = str(v).strip()
            if not sv:
                continue
            if _normalize_text(sv) in t:
                matched.append(sv)
        if matched:
            filters[dim] = matched[:5]
    return filters


def _intent_and_queryspec(
    text: str,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    value_catalog: dict[str, set[str]],
    today: date,
):
    """
    Deterministic business-oriented intent extraction for demo prompts.
    Returns either (DataIntentExtraction, None) or (None, QuerySpec via parse_nlq fallback).
    """
    from models.query_spec import ChartSpec, QuerySpec, SortSpec, TimeRange

    t = _normalize_text(text)
    filters = _extract_catalog_filters(text, value_catalog)
    threshold_op, threshold_value = _extract_numeric_threshold(text)
    t_start, t_end = _extract_month_or_quarter_window(text, today)

    # Intent: decomposition question handled outside QuerySpec path
    if "difference between organic growth and aum growth" in t or ("organic growth" in t and "aum growth" in t and "difference" in t):
        return DataIntentExtraction(
            intent="decomposition",
            metric_id="ogr",
            dimensions=[],
            filters=filters,
            threshold_op=threshold_op,
            threshold_value=threshold_value,
            time_start=t_start,
            time_end=t_end,
        ), None
    if ("nnb" in t or "net new business" in t) and ("low fee yield" in t or "fee yield" in t) and ("etf" in t or "ticker" in t or "product" in t):
        return DataIntentExtraction(
            intent="growth_quality_flags",
            metric_id="fee_yield",
            dimensions=["product_ticker"],
            filters=filters,
            threshold_op=threshold_op,
            threshold_value=threshold_value,
            time_start=t_start,
            time_end=t_end,
        ), None

    # Heuristic intents for business prompts
    intent = "contributors"
    metric_id = "nnb"
    dimensions: list[str] = ["channel"]
    chart = ChartSpec(type="bar", x="channel", y="metric")

    if "etf" in t or "ticker" in t or "product" in t:
        intent = "etf_ticker"
        dimensions = ["product_ticker"]
        chart = ChartSpec(type="bar", x="product_ticker", y="metric")
    if "fee yield" in t:
        metric_id = "fee_yield"
    if "market" in t and "organic" in t:
        metric_id = "market_impact_abs"
    elif "market impact" in t:
        metric_id = "market_impact_abs"
    elif "organic growth" in t or "ogr" in t:
        metric_id = "ogr"
    elif "nnf" in t or "net new flow" in t or "fee flow" in t:
        metric_id = "nnf"
    elif "nnb" in t or "net new business" in t or "contributors" in t:
        metric_id = "nnb"

    # Make filter dimensions valid per QuerySpec rules.
    for fk in filters.keys():
        if fk not in dimensions:
            dimensions.append(fk)

    # Ensure dimensions are governed.
    dims_governed = []
    for d in dimensions:
        canon = d
        try:
            from app.nlq.governance import normalize_dim_token
            canon = normalize_dim_token(d, dim_reg) or d
        except Exception:
            pass
        if canon not in dims_governed:
            dims_governed.append(canon)

    tr = TimeRange(start=t_start, end=t_end, granularity="month")
    if t_start is None and t_end is None:
        # fallback deterministic window if prompt implies YTD / recent
        if "ytd" in t:
            tr = TimeRange(start=date(today.year, 1, 1), end=today, granularity="month")
        elif "last 12" in t:
            tr = TimeRange(start=date(today.year - 1, today.month, 1), end=today, granularity="month")

    qs = QuerySpec(
        metric_id=metric_id,
        dimensions=dims_governed,
        filters=filters,
        time_range=tr,
        sort=SortSpec(by="metric", order="desc"),
        limit=50,
        chart=chart if chart.x in dims_governed else ChartSpec(type="table"),
    )
    return DataIntentExtraction(
        intent=intent,
        metric_id=metric_id,
        dimensions=dims_governed,
        filters=filters,
        threshold_op=threshold_op,
        threshold_value=threshold_value,
        time_start=tr.start,
        time_end=tr.end,
    ), qs


def _apply_threshold_to_result(result: QueryResult, op: str | None, value: float | None) -> QueryResult:
    if op is None or value is None or result is None or result.data is None or result.data.empty:
        return result
    df = result.data.copy()
    if "metric" not in df.columns:
        return result
    metric = pd.to_numeric(df["metric"], errors="coerce")
    if op == "gt":
        df = df[metric > float(value)]
    else:
        df = df[metric < float(value)]
    out = QueryResult(
        data=df.reset_index(drop=True),
        numbers=result.numbers,
        chart_spec=result.chart_spec,
        explain_context=result.explain_context,
        meta={**(result.meta or {}), "threshold_applied": {"op": op, "value": value}},
    )
    return out


def _load_value_catalog(gateway_dict: dict[str, Any], root: Path) -> dict[str, set[str]]:
    """
    Build value_catalog (distinct values per dim) from gateway. Cached by dataset_version via run_query.
    Returns dict of dim -> set of values. If data not available, return empty sets.
    """
    try:
        df = run_query(Q_TICKER_MONTHLY, gateway_dict, root=root)
    except Exception:
        try:
            df = run_query(Q_CHANNEL_MONTHLY, gateway_dict, root=root)
        except Exception:
            return {}
    if df is None or df.empty:
        return {}
    catalog: dict[str, set[str]] = {}
    for col in ("channel", "sub_channel", "product_ticker", "src_country", "country", "segment", "sub_segment", "month_end"):
        if col in df.columns:
            catalog[col] = set(df[col].dropna().astype(str).str.strip().unique().tolist())
    return catalog


@st.cache_data(ttl=300, show_spinner=False)
def _load_intelligence_desk_df(gateway_dict_json: str, root_str: str, prefer_geo: bool = False) -> pd.DataFrame:
    """
    Load platform dataset for Intelligence Desk.
    - prefer_geo=True: try Q_GEO_MONTHLY first (country/region questions).
    - Falls back through ticker → channel → empty-filters variants.
    Cached by (gateway_dict_json, root_str, prefer_geo).
    """
    try:
        gateway_dict = json.loads(gateway_dict_json)
    except Exception:
        gateway_dict = {}
    root = Path(root_str) if root_str else ROOT
    query_order = (
        [Q_GEO_MONTHLY, Q_TICKER_MONTHLY, Q_CHANNEL_MONTHLY]
        if prefer_geo
        else [Q_TICKER_MONTHLY, Q_CHANNEL_MONTHLY, Q_GEO_MONTHLY]
    )
    for q_name in query_order:
        try:
            out = run_query(q_name, gateway_dict, root=root)
            if isinstance(out, pd.DataFrame) and not out.empty:
                logger.debug(
                    "[IntelDesk] _load_intelligence_desk_df query=%s prefer_geo=%s rows=%d cols=%s",
                    q_name, prefer_geo, len(out), list(out.columns[:10]),
                )
                return out
        except Exception:
            pass
    # Last resort: try with an empty filter dict to get any available data.
    for q_name in query_order:
        try:
            out = run_query(q_name, {}, root=root)
            if isinstance(out, pd.DataFrame) and not out.empty:
                logger.debug(
                    "[IntelDesk] _load_intelligence_desk_df fallback empty-filter query=%s rows=%d",
                    q_name, len(out),
                )
                return out
        except Exception:
            pass
    logger.warning("[IntelDesk] _load_intelligence_desk_df returned empty for prefer_geo=%s", prefer_geo)
    return pd.DataFrame()


# Refusal phrases Claude returns when it sees no real data context.
_CLAUDE_REFUSAL_FRAGMENTS = (
    "don't have access",
    "do not have access",
    "no access to",
    "not have access",
    "cannot access",
    "can't access",
    "i have no data",
    "no dataset",
    "no data available",
    "not provided",
    "not available in the context",
    "not included in",
    "unable to provide",
    "don't have information",
    "do not have information",
    "cannot provide",
    "can't provide",
    "based on my training",
    "my knowledge",
    "as an ai",
    "as a language model",
)


def _is_inteldesk_subset_relevant(question: str, subset_df: pd.DataFrame) -> bool:
    """
    Lightweight relevance gate: only block when the subset is clearly missing
    the columns required to answer the specific question type.
    """
    if subset_df is None or not isinstance(subset_df, pd.DataFrame) or subset_df.empty:
        return False
    q = _normalize_text(question)
    cols = {str(c).strip().lower() for c in subset_df.columns}

    # Flow/NNB questions need at least one flow metric.
    if any(k in q for k in ("flow", "flows", "inflow", "outflow", "nnb", "nnf", "net new")):
        if not ({"nnb", "nnf"} & cols):
            return False
    # AUM questions need an AUM column.
    if any(k in q for k in ("aum", "assets under management")):
        if not ({"end_aum", "begin_aum"} & cols):
            return False
    # Country questions need a geography column (any variant).
    if any(k in q for k in ("country", "countries", "region", "geographic")):
        if not ({"country", "src_country", "geo"} & cols):
            return False
    # Channel questions need a channel column.
    if "channel" in q and not ({"channel", "sub_channel"} & cols):
        return False
    # Note: "growth" is NOT a hard gate — the subset may still be useful with NNB/AUM.
    return True


def _is_inteldesk_answer_grounded(answer: str, context_markdown: str, subset_df: pd.DataFrame) -> bool:
    """
    Return False if the answer is clearly a generic/refusal response from Claude
    rather than an answer grounded in the dataset context.
    """
    if not (answer or "").strip():
        return False
    if not (context_markdown or "").strip():
        return False
    if subset_df is None or not isinstance(subset_df, pd.DataFrame) or subset_df.empty:
        return False

    # Detect standard Claude refusal / no-data phrases.
    text_lower = (answer or "").lower()
    if any(frag in text_lower for frag in _CLAUDE_REFUSAL_FRAGMENTS):
        logger.debug("[IntelDesk] answer_grounded=False (refusal phrase detected)")
        return False

    return True


def _execute_data_query_service(
    *,
    qs: Any,
    df: pd.DataFrame,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    allowlist: dict[str, Any],
) -> tuple[QueryResult | None, str | None]:
    """
    Deterministic query service for governed internal-data questions.
    Applies validation + execution and returns structured (result, error_message).
    """
    if qs is None:
        return None, "Unable to interpret this as a governed internal-data query."
    try:
        validate_queryspec(qs, metric_reg, dim_reg, out_logs=None)
    except GovernanceError as e:
        return None, f"Validation failed: {e}"
    try:
        result = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=False)
    except ValueError as e:
        return None, f"Execution error: {e}"
    except GovernanceError as e:
        return None, f"Governance error: {e}"
    return result, None


def _metric_label(result: QueryResult) -> str:
    """Label for headline metric from explain_context or numbers."""
    return (
        (result.explain_context or {}).get("metric_label")
        or (result.numbers or {}).get("metric_id", "Metric").replace("_", " ").title()
    )


def render_chart(
    result: QueryResult,
    *,
    full_export_provider: Callable[[], pd.DataFrame] | None = None,
    allow_full: bool = False,
) -> None:
    """
    Deterministic chart from result.chart_spec. Line: px.line (min points guardrail); Bar: px.bar.
    Table: skip chart. Always render table underneath. Display uses format_df; download is raw CSV.
    """
    df = result.data
    if df is None or not isinstance(df, pd.DataFrame):
        render_empty_state("No data returned for this query.", "Adjust filters or time range.")
        return
    if df.empty:
        render_empty_state("No data returned for this query.", "Adjust filters or time range.")
        return

    chart_spec = result.chart_spec or {}
    chart_type = chart_spec.get("type") or "table"
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    series_col = chart_spec.get("series")
    plot_df = df.sort_values(by=x_col, ascending=True).copy() if (x_col and x_col in df.columns) else df.copy()
    fallback_cols = [c for c in (x_col, y_col, series_col) if c and c in df.columns] or list(df.columns)[:8]

    def _draw_line() -> None:
        if not (x_col and y_col and x_col in plot_df.columns and y_col in plot_df.columns):
            return
        if series_col and series_col in plot_df.columns:
            fig = px.line(plot_df, x=x_col, y=y_col, color=series_col)
        else:
            fig = px.line(plot_df, x=x_col, y=y_col)
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=60))
        apply_enterprise_plotly_style(fig)
        safe_render_plotly(fig, user_message="Chart not shown for this selection.")

    def _draw_bar() -> None:
        if not (x_col and y_col and x_col in df.columns and y_col in df.columns):
            return
        plot_df_bar = df.copy()
        if series_col and series_col in plot_df_bar.columns:
            fig = px.bar(plot_df_bar, x=x_col, y=y_col, color=series_col)
        else:
            fig = px.bar(plot_df_bar, x=x_col, y=y_col)
        fig.update_layout(xaxis_tickangle=-45, margin=dict(l=40, r=40, t=40, b=80))
        apply_enterprise_plotly_style(fig)
        safe_render_plotly(fig, user_message="Chart not shown for this selection.")

    if chart_type == "line" and x_col and y_col and x_col in df.columns and y_col in df.columns:
        render_chart_or_fallback(
            _draw_line,
            plot_df,
            fallback_cols,
            fallback_note("chart_insufficient_points", {"min_points": 2}),
            min_points=2,
            empty_reason="No data for chart.",
            empty_hint="Adjust filters or time range.",
        )
    elif chart_type == "bar" and x_col and y_col and x_col in df.columns and y_col in df.columns:
        render_chart_or_fallback(
            _draw_bar,
            df,
            fallback_cols,
            fallback_note("chart_insufficient_points", {"min_points": 1}),
            min_points=1,
            empty_reason="No data for chart.",
            empty_hint="Adjust filters or time range.",
        )

    st.subheader("Results Table")
    df_show = format_df(df, infer_common_formats(df))
    st.dataframe(df_show, height=420, width="stretch", hide_index=True)
    render_export_buttons(
        df,
        full_export_provider,
        "tab3_nlq_result",
        allow_full=allow_full,
    )


def _render_result(
    result: QueryResult,
    *,
    full_export_provider: Callable[[], pd.DataFrame] | None = None,
    allow_full: bool = False,
) -> None:
    """
    Headline (Answer if numbers formatted/value else Result), chart from chart_spec, table + download,
    safety message if rows_capped or limit_clamped.
    """
    numbers = result.numbers or {}
    has_formatted = "formatted" in numbers and numbers.get("formatted") not in (None, "-")
    has_value = "value" in numbers and numbers.get("value") is not None

    if has_formatted or has_value:
        st.subheader("Portfolio Answer")
        label = _metric_label(result)
        display_val = numbers.get("formatted") if has_formatted else str(numbers.get("value", "-"))
        st.metric(label=label, value=display_val)
    else:
        st.subheader("Analytics Result")

    meta = result.meta or {}
    if meta.get("rows_capped") or any("clamp" in str(w).lower() for w in meta.get("warnings", [])):
        st.info("Results capped for safety. Adjust query or use export flow if needed.")

    render_chart(result, full_export_provider=full_export_provider, allow_full=allow_full)


# Spec: 4 prompt presets per mode (client spec)
MARKET_PROMPTS = [
    "What are current Fed rate expectations and how could they affect multi-asset flows?",
    "Summarize this week's global equity and bond market drivers.",
    "What is the latest outlook for ETF inflows across major asset classes?",
    "How are inflation surprises affecting duration and credit positioning?",
]

DATA_PROMPTS = [
    "Which ETFs had high NNB but low fee yield in Q3?",
    "Show products in Wealth channel where fee yield is below 0.5%.",
    "Tell me about contributors in Broker Dealer with NNB above $100k.",
    "What drove the difference between organic growth and AUM growth in June?",
]

PLACEHOLDER_DATA = "Which ETFs had high NNB but low fee yield in Q3?"
PLACEHOLDER_MARKET = "What is the latest outlook for ETF inflows across major asset classes?"

NLQ_RESPONSE_KEY = "nlq_response"

# LLM settings: Claude model in session state; API key from Streamlit secrets (no UI input)
LLM_MODEL_KEY = "nlq_llm_model"

CLAUDE_MODELS = ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]
DEFAULT_CLAUDE_MODEL = "claude-3-7-sonnet-latest"


def _claude_key_configured() -> bool:
    if has_claude_api_key is None:
        try:
            key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
            configured = bool(key and key != "your-key-here")
            logger.info("DEBUG Claude secret_detected=%s (fallback)", "yes" if configured else "no")
            return configured
        except Exception:
            logger.info("DEBUG Claude secret_detected=no (fallback failed)")
            return False
    try:
        configured = bool(has_claude_api_key())
        logger.info("DEBUG Claude secret_detected=%s", "yes" if configured else "no")
        return configured
    except Exception:
        return False


def _claude_sdk_available() -> bool:
    if anthropic_sdk_available is not None:
        try:
            available = bool(anthropic_sdk_available())
            return available
        except Exception:
            return False
    try:
        return importlib.util.find_spec("anthropic") is not None
    except Exception:
        return False


def _set_claude_debug(*, secret_detected: bool, path_selected: bool, request_success: str) -> None:
    st.session_state[CLAUDE_DEBUG_KEY] = {
        "build_marker": BUILD_MARKER,
        "secret_detected": secret_detected,
        "sdk_available": _claude_sdk_available(),
        "path_selected": path_selected,
        "request_success": request_success,
    }


def _get_data_narrative(payload: dict[str, Any]) -> tuple[str, str | None]:
    """Data Questions narrative + fallback warning. Uses Claude when configured."""
    secret_detected = _claude_key_configured()
    sdk_available = _claude_sdk_available()
    path_selected = bool(secret_detected and sdk_available and claude_generate is not None)
    logger.info(
        "[%s] DEBUG Claude data_path secret_detected=%s sdk_available=%s path_selected=%s request_success=n/a",
        BUILD_MARKER,
        "yes" if secret_detected else "no",
        "yes" if sdk_available else "no",
        "yes" if path_selected else "no",
    )
    _set_claude_debug(secret_detected=secret_detected, path_selected=path_selected, request_success="n/a")
    if not secret_detected:
        logger.info("[%s] Data narrative routing: Claude unavailable (secret missing)", BUILD_MARKER)
        _set_claude_debug(secret_detected=False, path_selected=False, request_success="no")
        return "", "Claude secret is not configured. Verified output shown."
    if not sdk_available or claude_generate is None:
        logger.warning("[%s] Data narrative routing: Claude unavailable (SDK/import issue)", BUILD_MARKER)
        _set_claude_debug(secret_detected=True, path_selected=False, request_success="no")
        return "", "Anthropic SDK unavailable in deployment. Verified output shown."
    model = (st.session_state.get(LLM_MODEL_KEY) or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
    prompt = (
        "You are a concise analyst.\n"
        "Use only the numbers and facts in the payload.\n"
        "Do not invent or infer numbers. Do not perform calculations.\n\n"
        "Payload:\n"
        f"{payload}"
    )
    try:
        logger.info("[%s] Data narrative routing selected Claude (model=%s)", BUILD_MARKER, model)
        with st.spinner("Generating narrative..."):
            narrative = claude_generate(prompt=prompt, model=model, max_tokens=512)
        logger.info("[%s] Data narrative Claude request succeeded", BUILD_MARKER)
        logger.info("[%s] DEBUG Claude data_path secret_detected=yes sdk_available=yes path_selected=yes request_success=yes", BUILD_MARKER)
        _set_claude_debug(secret_detected=True, path_selected=True, request_success="yes")
        return narrative, None
    except (ClaudeError, RuntimeError) as e:
        logger.warning("[%s] Data narrative Claude request failed: %s: %s", BUILD_MARKER, type(e).__name__, str(e))
        logger.info("[%s] DEBUG Claude data_path secret_detected=yes sdk_available=yes path_selected=yes request_success=no", BUILD_MARKER)
        _set_claude_debug(secret_detected=True, path_selected=True, request_success="no")
        return "", "Claude request failed. Verified output shown."
    except Exception as e:
        logger.exception("[%s] Data narrative Claude request failed unexpectedly: %s", BUILD_MARKER, type(e).__name__)
        logger.info("[%s] DEBUG Claude data_path secret_detected=yes sdk_available=yes path_selected=yes request_success=no", BUILD_MARKER)
        _set_claude_debug(secret_detected=True, path_selected=True, request_success="no")
        return "", "Claude request failed. Verified output shown."


def _render_prompt_presets(is_data_mode: bool) -> None:
    """Four presets in a 2x2 grid with compact visual weight."""
    prompts = DATA_PROMPTS if is_data_mode else MARKET_PROMPTS
    for row in range(0, 4, 2):
        cols = st.columns(2)
        for col_idx, i in enumerate(range(row, min(row + 2, 4))):
            with cols[col_idx]:
                if st.button(
                    prompts[i],
                    key=f"nlq_preset_{'data' if is_data_mode else 'market'}_{i}",
                    width="content",
                ):
                    st.session_state["nlq_question"] = prompts[i]


def _render_active_scope(state: FilterState) -> None:
    """Single subtle line: portfolio scope and reporting window (compact)."""
    scope = "Enterprise-wide portfolio"
    if getattr(state, "slice_dim", None) and getattr(state, "slice_value", None):
        scope = f"{state.slice_dim}: {state.slice_value}"
    start = getattr(state, "date_start", None)
    end = getattr(state, "date_end", None)
    if start and end:
        line = f"Active portfolio scope: {scope}. Reporting window: {start} to {end}."
    else:
        line = f"Active portfolio scope: {scope}."
    st.markdown(f"<div class='nlq-scope'>{line}</div>", unsafe_allow_html=True)


def _set_nlq_response(
    *,
    intent: str,
    header: str,
    subtitle: str,
    narrative: str = "",
    table_df: pd.DataFrame | None = None,
    chart_result: QueryResult | None = None,
    full_export_provider: Callable[[], pd.DataFrame] | None = None,
    error: str | None = None,
    placeholder_fallback: str | None = None,
    provider_meta: str | None = None,
    response_meta: str | None = None,
) -> None:
    """Store one response for the unified response area. Table/chart rendered from table_df and chart_result."""
    if response_meta is None:
        if intent == "market_intelligence":
            response_meta = "Response type: Market Intelligence (External)"
        elif intent == "data_question":
            response_meta = "Response type: Internal Data (Verified Python query)"
    st.session_state[NLQ_RESPONSE_KEY] = {
        "intent": intent,
        "header": header,
        "subtitle": subtitle,
        "narrative": narrative,
        "table_df": table_df,
        "chart_result": chart_result,
        "full_export_provider": full_export_provider,
        "error": error,
        "placeholder_fallback": placeholder_fallback,
        "provider_meta": provider_meta,
        "response_meta": response_meta,
    }


def _render_response_area(state: FilterState, contract: dict[str, Any]) -> None:
    """Single response container: compact header, narrative, optional table/chart. Visually close to input."""
    resp = st.session_state.get(NLQ_RESPONSE_KEY)
    if resp is None:
        st.markdown("<div class='nlq-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='nlq-empty-state'>"
            "<div class='nlq-empty-title'>Result will appear here</div>"
            "Choose a mode, ask a question, and the result will appear here."
            "</div>",
            unsafe_allow_html=True,
        )
        dbg = st.session_state.get(CLAUDE_DEBUG_KEY) or {}
        if dbg:
            st.caption(
                "Build: {build} | Anthropic secret detected: {secret} | Anthropic SDK import available: {sdk} | "
                "Claude path selected: {path} | Claude request succeeded: {req}".format(
                    build=dbg.get("build_marker", BUILD_MARKER),
                    secret="yes" if dbg.get("secret_detected") else "no",
                    sdk="yes" if dbg.get("sdk_available") else "no",
                    path="yes" if dbg.get("path_selected") else "—",
                    req=dbg.get("request_success", "—"),
                )
            )
        return

    st.markdown("<div class='nlq-divider'></div>", unsafe_allow_html=True)
    header = resp.get("header") or "Response"
    subtitle = resp.get("subtitle") or ""
    st.markdown(f"<p class='nlq-response-header'><strong>{header}</strong></p>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
    response_meta = (resp.get("response_meta") or "").strip()
    if response_meta:
        st.markdown(f"<div class='nlq-response-meta'>{response_meta}</div>", unsafe_allow_html=True)
    dbg = st.session_state.get(CLAUDE_DEBUG_KEY) or {}
    if dbg:
        st.caption(
            "Build marker: {build} | Anthropic secret detected: {secret} | Anthropic SDK import available: {sdk} | "
            "Claude path selected: {path} | Claude request succeeded: {req}".format(
                build=dbg.get("build_marker", BUILD_MARKER),
                secret="yes" if dbg.get("secret_detected") else "no",
                sdk="yes" if dbg.get("sdk_available") else "no",
                path="yes" if dbg.get("path_selected") else "no",
                req=dbg.get("request_success", "n/a"),
            )
        )

    error = resp.get("error")
    if error:
        st.error(error)
        return

    placeholder_fallback = (resp.get("placeholder_fallback") or "").strip()
    if placeholder_fallback:
        st.markdown(
            f"<div class='nlq-fallback-note'>{placeholder_fallback}</div>",
            unsafe_allow_html=True,
        )

    narrative = (resp.get("narrative") or "").strip()
    if narrative:
        intent = (resp.get("intent") or "").strip()
        label = "Claude Narrative" if intent == "data_question" else "Claude Market Brief"
        st.markdown(
            f"<div class='nlq-narrative-section'><span class='nlq-narrative-label'>{label}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='nlq-narrative-anchor' aria-hidden='true'></div>", unsafe_allow_html=True)
        st.markdown(narrative)
    provider_meta = resp.get("provider_meta")
    if provider_meta:
        st.caption(f"_{provider_meta}_")

    chart_result = resp.get("chart_result")
    if chart_result is not None:
        render_chart(
            chart_result,
            full_export_provider=resp.get("full_export_provider"),
            allow_full=False,
        )

    table_df = resp.get("table_df")
    if table_df is not None and isinstance(table_df, pd.DataFrame) and not table_df.empty:
        st.dataframe(table_df, height=420, width="stretch", hide_index=True)
        if resp.get("full_export_provider"):
            render_export_buttons(
                table_df,
                resp["full_export_provider"],
                "tab3_nlq_result",
                allow_full=False,
            )


def _render_compact_history() -> None:
    """Compact, optional conversation history below the response area."""
    hist = st.session_state.get(CHAT_HISTORY_KEY) or []
    if not hist:
        return
    with st.expander("Recent conversation", expanded=False):
        for msg in hist[-8:]:
            role = msg.get("role", "assistant")
            prefix = "You" if role == "user" else "Desk"
            st.markdown(f"**{prefix}:** {msg.get('text', '')}")


def _inject_nlq_page_css() -> None:
    """Page-scoped CSS for Intelligence Desk: tighter, institutional layout. No other pages affected."""
    st.markdown(
        """
        <style>
        /* Compact subtitle under title */
        .nlq-subtitle { color: #b7c5e3; font-size: 0.85rem; margin-top: -0.25rem !important; margin-bottom: 0.4rem !important; line-height: 1.35; }
        /* Light, tight guidance banner — less boxy */
        .nlq-banner { color: #b7c5e3; font-size: 0.82rem; line-height: 1.4; border-left: 3px solid #4c7edb; border-radius: 6px; padding: 0.4rem 0.6rem; background: rgba(17,29,58,0.6); margin: 0.15rem 0 0.35rem 0; }
        .nlq-banner strong { color: #f8fbff; }
        /* Mode label */
        .nlq-mode-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #b7c5e3; margin-bottom: 0.2rem !important; }
        /* Compact mode card — smaller height, less bulk */
        .nlq-mode-card { border: 1px solid #2a3d67; border-radius: 8px; padding: 0.4rem 0.65rem; background: rgba(23,40,77,0.5); font-size: 0.85rem; margin: 0.2rem 0 0.4rem 0; line-height: 1.35; }
        .nlq-mode-card strong { color: #8fb4ff; }
        .nlq-mode-card .nlq-mode-label { margin-bottom: 0.15rem !important; }
        /* Subtle scope line */
        .nlq-scope { color: #7f93bc; font-size: 0.78rem; margin: 0.15rem 0 0.3rem 0 !important; }
        /* Response area: compact, close to input */
        .nlq-divider { height: 1px; background: #2a3d67; margin: 0.35rem 0 0.5rem 0 !important; }
        .nlq-response-header { font-size: 1.1rem !important; margin-bottom: 0.15rem !important; margin-top: 0.25rem !important; }
        .nlq-response-meta { color: #9fb0d3; font-size: 0.78rem; margin: -0.1rem 0 0.35rem 0; }
        /* Narrative: clean section, dark dashboard style, multi-paragraph */
        .nlq-narrative-section { border-left: 3px solid #4c7edb; padding: 0.35rem 0 0.15rem 0.5rem; margin: 0.4rem 0 0.25rem 0; background: rgba(17,29,58,0.35); border-radius: 0 6px 6px 0; }
        .nlq-narrative-label { color: #9fb0d3; font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
        .nlq-narrative-anchor + div[data-testid="stMarkdown"] { color: #e2e8f0; line-height: 1.55; margin-top: 0.2rem !important; padding-left: 0.1rem; }
        .nlq-narrative-anchor + div[data-testid="stMarkdown"] p { margin-bottom: 0.65rem !important; }
        .nlq-narrative-anchor + div[data-testid="stMarkdown"] p:last-child { margin-bottom: 0 !important; }
        .nlq-fallback-note { color: #8b9dc3; font-size: 0.8rem; margin: 0.2rem 0 0.35rem 0; font-style: italic; }
        .nlq-empty-state { border: 1px solid #2a3d67; border-radius: 8px; padding: 0.5rem 0.65rem; background: rgba(17,29,58,0.5); font-size: 0.88rem; color: #b7c5e3; margin: 0.25rem 0 0.4rem 0; }
        .nlq-empty-state .nlq-empty-title { color: #f8fbff; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.15rem; }
        /* Chat message box: ensure typed text readable on dark (reinforces global form layer) */
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextArea"] [data-baseweb="textarea"] textarea {
            color: #f8fbff !important;
            -webkit-text-fill-color: #f8fbff !important;
            background-color: #17284d !important;
        }
        [data-testid="stTextArea"] textarea::placeholder {
            color: #8b9dc3 !important;
        }
        /* Tighten widget spacing for this page */
        div[data-testid="stTextInput"] { margin-bottom: 0.2rem !important; }
        div[data-testid="stRadio"] { margin-bottom: 0.15rem !important; }
        .stButton { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
        /* Preset buttons lighter than primary action */
        .stButton button[kind="secondary"] { opacity: 0.92; padding: 0.25rem 0.6rem !important; }
        /* Chat-only Intelligence Desk (v2): spacing, hierarchy, dark institutional */
        .inteldesk-subtitle { color: #b7c5e3; font-size: 0.85rem; margin-top: -0.2rem !important; margin-bottom: 0.5rem !important; line-height: 1.35; }
        .inteldesk-status { color: #8b9dc3; font-size: 0.78rem; margin-bottom: 0.6rem !important; font-style: normal; }
        .inteldesk-examples-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #b7c5e3; margin-bottom: 0.35rem !important; }
        .inteldesk-examples-row + div [data-testid="column"] .stButton button { background: rgba(23,40,77,0.6) !important; color: #e2e8f0 !important; border: 1px solid #2a3d67 !important; border-radius: 8px !important; padding: 0.5rem 0.65rem !important; font-size: 0.85rem !important; text-align: left !important; min-height: 2.5rem; }
        .inteldesk-examples-row + div [data-testid="column"] .stButton button:hover { border-color: #4c7edb !important; background: rgba(23,40,77,0.85) !important; }
        .inteldesk-divider { height: 1px; background: #2a3d67; margin: 0.6rem 0 0.5rem 0 !important; }
        .inteldesk-response-placeholder { color: #8b9dc3; font-size: 0.85rem; padding: 0.75rem 0; }
        .inteldesk-response-anchor + div[data-testid="stMarkdown"] { color: #e2e8f0; line-height: 1.55; margin-top: 0.25rem !important; }
        .inteldesk-response-anchor + div[data-testid="stMarkdown"] p { margin-bottom: 0.65rem !important; }
        .inteldesk-response-anchor + div[data-testid="stMarkdown"] p:last-child { margin-bottom: 0 !important; }
        @media (max-width: 640px) { .inteldesk-examples-row + div [data-testid="column"] { min-width: 0 !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk entrypoint. Uses simplified v2 flow; legacy renderer kept disconnected."""
    try:
        _render_intelligence_desk_v2(state, contract)
    except Exception as e:
        logging.getLogger(__name__).exception("Intelligence Desk error")
        st.error("Intelligence Desk encountered an error. Other tabs are unaffected.")


def _render_intelligence_desk_v2(state: FilterState, contract: dict[str, Any]) -> None:
    """Chat-only Intelligence Desk."""
    _ = contract
    _inject_nlq_page_css()
    st.title("Intelligence Desk")
    st.markdown(
        "<div class='inteldesk-subtitle'>Ask one focused question and get a concise analyst response.</div>",
        unsafe_allow_html=True,
    )
    provider_status = get_provider_status()
    st.markdown(
        f"<div class='inteldesk-status'>{provider_status.status_text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='inteldesk-examples-label'>Example prompts</div>", unsafe_allow_html=True)
    st.markdown("<div class='inteldesk-examples-row' aria-hidden='true'></div>", unsafe_allow_html=True)
    examples = [
        "Which ETF had the highest inflow this month?",
        "Summarize ETF flow trends from the data.",
        "Which channel or country has the most AUM?",
    ]
    ecols = st.columns(3)
    for idx, ex in enumerate(examples):
        with ecols[idx]:
            if st.button(ex, key=f"inteldesk_ex_{idx}", width="stretch"):
                st.session_state[INTEL_CHAT_INPUT_KEY] = ex

    if INTEL_CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[INTEL_CHAT_HISTORY_KEY] = []
    if INTEL_CHAT_INPUT_KEY not in st.session_state:
        st.session_state[INTEL_CHAT_INPUT_KEY] = ""

    st.markdown("<div class='inteldesk-divider'></div>", unsafe_allow_html=True)
    prompt = st.text_area(
        "Message",
        key=INTEL_CHAT_INPUT_KEY,
        placeholder="Ask a market or portfolio intelligence question...",
        height=100,
    )
    user_text = (prompt or "").strip()
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        run_clicked = st.button(
            "Generate",
            key="inteldesk_send_btn",
            type="primary",
            disabled=not user_text,
        )
    with c2:
        clear_clicked = st.button("Clear chat", key="inteldesk_clear_btn")

    if clear_clicked:
        st.session_state[INTEL_CHAT_HISTORY_KEY] = []
        st.session_state[INTEL_CHAT_INPUT_KEY] = ""
        st.session_state.pop(INTEL_LAST_SUBSET_DF_KEY, None)
        st.rerun()

    history: list[dict[str, str]] = st.session_state.get(INTEL_CHAT_HISTORY_KEY) or []
    response_box = st.container()

    if run_clicked and user_text:
        history.append({"role": "user", "text": user_text})
        st.session_state[INTEL_LAST_SUBSET_DF_KEY] = pd.DataFrame()
        subset_df: pd.DataFrame = pd.DataFrame()
        if not provider_status.enabled or not claude_generate_grounded:
            history.append({"role": "assistant", "text": "Provider unavailable."})
        else:
            # Detect country/geo questions so the right query table is loaded.
            _q_lower = user_text.lower()
            _prefer_geo = any(k in _q_lower for k in ("country", "countries", "region", "geographic", "geo"))
            gateway_dict = filter_state_to_gateway_dict(state)
            gateway_json = json.dumps(gateway_dict, sort_keys=True, default=str)
            df_raw = _load_intelligence_desk_df(gateway_json, str(ROOT), prefer_geo=_prefer_geo)
            logger.debug(
                "[IntelDesk] question=%r prefer_geo=%s df_raw_shape=%s df_raw_cols=%s",
                user_text[:120],
                _prefer_geo,
                getattr(df_raw, "shape", None),
                list(df_raw.columns[:12]) if not df_raw.empty else [],
            )
            subset_df, context_markdown = retrieve_intelligence_desk_context(user_text, df_raw)
            logger.debug(
                "[IntelDesk] subset rows=%d cols=%s context_len=%d",
                len(subset_df),
                list(subset_df.columns),
                len(context_markdown),
            )
            if subset_df.empty or not context_markdown:
                answer = "Data not available in the current dataset."
                st.session_state[INTEL_LAST_SUBSET_DF_KEY] = pd.DataFrame()
            elif not _is_inteldesk_subset_relevant(user_text, subset_df):
                answer = "The available dataset does not contain the columns needed to answer this specific question."
                st.session_state[INTEL_LAST_SUBSET_DF_KEY] = pd.DataFrame()
            else:
                system_prompt = (
                    "You are a financial data analyst operating inside the Intelligence Desk of an ETF analytics platform.\n\n"
                    "Your task is to answer the user's question ONLY using the dataset context provided below.\n\n"
                    "Rules:\n"
                    "- Do not use outside knowledge.\n"
                    "- Do not invent values or extrapolate.\n"
                    "- Do not guess missing fields.\n"
                    "- If the answer is not in the dataset, say exactly: 'Data not available in the current dataset.'\n"
                    "- If the dataset only partially answers the question, explain the limitation.\n"
                    "- Be concise and analyst-style.\n"
                    "- Highlight ranking or key drivers visible in the data.\n\n"
                    "Dataset context (use ONLY this data):\n\n"
                    f"{context_markdown}"
                )
                claude_success = False
                try:
                    with st.spinner("Analysing dataset..."):
                        answer = claude_generate_grounded(
                            system_prompt,
                            user_text,
                            model=INTEL_DESK_MODEL,
                            max_tokens=1000,
                        )
                        claude_success = True
                        logger.debug("[IntelDesk] Claude call success, answer_len=%d", len(answer))
                except (ChatProviderError, ClaudeError) as e:
                    answer = (str(getattr(e, "message", e)) or "Request failed.").strip()
                    logger.debug("[IntelDesk] Claude call error: %s", answer)
                except Exception as _exc:
                    answer = "Request failed."
                    logger.debug("[IntelDesk] Claude call exception: %s", _exc)
                if claude_success and _is_inteldesk_answer_grounded(answer, context_markdown, subset_df):
                    st.session_state[INTEL_LAST_SUBSET_DF_KEY] = subset_df
                elif claude_success:
                    # Claude returned a refusal — tell the user data is insufficient.
                    logger.debug("[IntelDesk] answer_grounded=False, replacing with unavailable message")
                    answer = "The dataset does not contain sufficient information to answer this question reliably."
                    st.session_state[INTEL_LAST_SUBSET_DF_KEY] = pd.DataFrame()
            history.append({"role": "assistant", "text": (answer or "").strip() or "No response returned."})
        st.session_state[INTEL_CHAT_HISTORY_KEY] = history

    history = st.session_state.get(INTEL_CHAT_HISTORY_KEY) or []
    with response_box:
        st.markdown("<div class='inteldesk-divider'></div>", unsafe_allow_html=True)
        if run_clicked and not user_text:
            st.warning("Enter a message before generating.")
        elif not history:
            st.markdown(
                "<div class='inteldesk-response-placeholder'>Ask a question above or click an example to start.</div>",
                unsafe_allow_html=True,
            )
        else:
            last_assistant = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
            if last_assistant:
                st.markdown("<div class='inteldesk-response-anchor' aria-hidden='true'></div>", unsafe_allow_html=True)
                st.markdown(last_assistant.get("text", ""))
                last_subset = st.session_state.get(INTEL_LAST_SUBSET_DF_KEY)
                if last_subset is not None and isinstance(last_subset, pd.DataFrame) and not last_subset.empty:
                    st.markdown("### Data used for analysis")
                    st.dataframe(last_subset, use_container_width=True)
            else:
                st.markdown(
                    "<div class='inteldesk-response-placeholder'>Awaiting response.</div>",
                    unsafe_allow_html=True,
                )

    if history:
        with st.expander("Conversation history", expanded=False):
            for m in history[-12:]:
                prefix = "You" if m.get("role") == "user" else "Desk"
                st.markdown(f"**{prefix}:** {m.get('text', '')}")


def _run_market_intelligence_v2(text: str, state: FilterState, contract: dict[str, Any], claude_ready: bool) -> None:
    if not claude_ready:
        _set_nlq_response(
            intent="market_intelligence",
            header="Market Intelligence - external sources",
            subtitle="This answer reflects external context, not your internal book.",
            error="Claude unavailable. Verified output shown.",
            response_meta="Response type: Market Intelligence (External)",
        )
        _render_response_area(state, contract)
        return

    context = search_market_context(text)
    prompt = (
        "You are a market intelligence analyst.\n"
        "Provide a concise external-market brief with: executive take, observed signals, implications, risks.\n\n"
        f"User question: {text}\n\n"
        f"External context:\n{context}"
    )
    try:
        with st.spinner("Generating market brief..."):
            answer = claude_generate(prompt=prompt, model=INTEL_DESK_MODEL)
        _set_claude_debug(secret_detected=True, path_selected=True, request_success="yes")
        _set_nlq_response(
            intent="market_intelligence",
            header="Market Intelligence - external sources",
            subtitle="This answer reflects external context, not your internal book.",
            narrative=answer,
            provider_meta=f"Claude (Anthropic) | {INTEL_DESK_MODEL}",
            response_meta="Response type: Market Intelligence (External)",
        )
    except Exception:
        _set_claude_debug(secret_detected=_claude_key_configured(), path_selected=True, request_success="no")
        _set_nlq_response(
            intent="market_intelligence",
            header="Market Intelligence - external sources",
            subtitle="This answer reflects external context, not your internal book.",
            error="Claude request failed. Verified output shown.",
            response_meta="Response type: Market Intelligence (External)",
        )
    _render_response_area(state, contract)


def _run_verified_data_v2(text: str, state: FilterState, contract: dict[str, Any]) -> None:
    try:
        metric_reg = load_metric_registry()
        dim_reg = load_dim_registry()
    except (FileNotFoundError, ValueError) as e:
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=f"Registries failed to load: {e}",
        )
        _render_response_area(state, contract)
        return

    gateway_dict = filter_state_to_gateway_dict(state)
    value_catalog = _load_value_catalog(gateway_dict, ROOT)
    parsed = parse_nlq(text, metric_reg, dim_reg, value_catalog, today=date.today())
    if isinstance(parsed, ParseError):
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=parsed.message or "Unable to parse this as a governed query.",
        )
        _render_response_area(state, contract)
        return
    qs = parsed

    try:
        df = run_query(Q_TICKER_MONTHLY, gateway_dict, root=ROOT)
    except Exception:
        df = run_query(Q_CHANNEL_MONTHLY, gateway_dict, root=ROOT)

    allowlist = {
        "columns": {
            "channel",
            "sub_channel",
            "product_ticker",
            "src_country",
            "country",
            "segment",
            "sub_segment",
            "month_end",
            "metric",
            "begin_aum",
            "end_aum",
            "nnb",
            "nnf",
        },
        "pii_columns": set(),
        "max_rows": 5000,
    }
    with st.spinner("Running verified query..."):
        result, svc_error = _execute_data_query_service(
            qs=qs,
            df=df,
            metric_reg=metric_reg,
            dim_reg=dim_reg,
            allowlist=allowlist,
        )
    if svc_error:
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=svc_error,
        )
        _render_response_area(state, contract)
        return

    numbers = result.numbers or {}
    bullets = build_deterministic_summary(qs, result)
    narrative_payload = {
        "query": text,
        "queryspec": qs.model_dump(mode="json"),
        "numbers": numbers,
        "meta": result.meta or {},
        "deterministic_summary": bullets,
        "top_rows_preview": (result.data.head(5).to_dict(orient="records") if isinstance(result.data, pd.DataFrame) and not result.data.empty else []),
    }
    narrative, narrative_warning = _get_data_narrative(narrative_payload)
    narrative_text = ""
    if bullets:
        narrative_text += "\n".join([f"- {b}" for b in bullets]) + "\n\n"
    if (narrative or "").strip():
        narrative_text += narrative.strip()
        placeholder_fallback = None
    else:
        narrative_text += "Verified output is shown below."
        placeholder_fallback = narrative_warning or "Narrative unavailable. Verified output below."

    _set_nlq_response(
        intent="data_question",
        header="Verified Data Result",
        subtitle="Calculated from your internal filtered dataset.",
        narrative=narrative_text,
        chart_result=result,
        placeholder_fallback=placeholder_fallback,
        response_meta="Response type: Internal Data (Verified Python query)",
    )
    _render_response_area(state, contract)


def _render_intelligence_desk(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk UI implementation."""
    _ = contract
    logger.info("[%s] Intelligence Desk render start", BUILD_MARKER)
    _inject_nlq_page_css()

    # Initial deployment debug state so marker is visible before first request (easy to remove later)
    if CLAUDE_DEBUG_KEY not in st.session_state:
        _set_claude_debug(
            secret_detected=_claude_key_configured(),
            path_selected=False,
            request_success="—",
        )

    # --- 1. Page header (compact) ---
    st.title("Intelligence Desk")
    st.markdown(
        "<div class='nlq-subtitle'>Ask data questions over your internal book or market intelligence over external sources. Results are verified; narrative is optional.</div>",
        unsafe_allow_html=True,
    )

    # --- 2. Guidance banner (light, tight) ---
    st.markdown(
        "<div class='nlq-banner'>"
        "<strong>Two ways to ask:</strong> "
        "Data Questions = your internal data, governed query, verified result. "
        "Market Intelligence = external sources (rates, macro, sentiment), clearly labeled as external."
        "</div>",
        unsafe_allow_html=True,
    )

    # --- 3. Mode selector (tighter) ---
    st.markdown("<span class='nlq-mode-label'>Choose mode</span>", unsafe_allow_html=True)
    with st.container():
        mode = st.radio(
            "Choose mode",
            ["Data Questions", "Market Intelligence"],
            key="nlq_mode",
            horizontal=True,
            format_func=lambda x: x,
            label_visibility="collapsed",
        )
    is_data_mode = mode == "Data Questions"

    # --- 3b. LLM settings: model in session state; API key from Streamlit secrets (no UI input) ---
    if LLM_MODEL_KEY not in st.session_state:
        st.session_state[LLM_MODEL_KEY] = "claude-3-7-sonnet-latest"

    with st.expander("LLM settings (for Market Intelligence)", expanded=False):
        has_key = _claude_key_configured()
        st.caption("Claude: Enabled" if has_key else "Claude: Unavailable")
        saved_model = st.session_state.get(LLM_MODEL_KEY) or ""
        model_index = CLAUDE_MODELS.index(saved_model) if saved_model in CLAUDE_MODELS else 0
        model_ui = st.selectbox(
            "Model",
            CLAUDE_MODELS,
            key="nlq_llm_model_claude",
            index=model_index,
        )
        st.session_state[LLM_MODEL_KEY] = model_ui

    active_model = st.session_state.get(LLM_MODEL_KEY) or ""
    active_key = _claude_key_configured()
    market_ready = bool(active_model and active_key)

    # --- 4. Current mode card (compact) ---
    if is_data_mode:
        st.markdown(
            "<div class='nlq-mode-card'>"
            "<div class='nlq-mode-label'>Current mode</div>"
            "<strong>DATA QUESTIONS</strong> — Verified answers from your internal portfolio data; calculations and filtering run before any narrative."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='nlq-mode-card'>"
            "<div class='nlq-mode-label'>Current mode</div>"
            "<strong>MARKET INTELLIGENCE</strong> — External sources; answer labeled as Market Intelligence."
            "</div>",
            unsafe_allow_html=True,
        )

    if is_data_mode:
        st.caption("Mode status: Internal deterministic analytics path.")
        st.caption(f"Claude: {'Enabled' if active_key else 'Unavailable'}")
    else:
        readiness = "Ready" if market_ready else "Setup required"
        st.caption(f"Mode status: {readiness} | Claude model: {active_model or 'Not selected'}")
        st.caption(f"Claude: {'Enabled' if active_key else 'Unavailable'}")

    # --- 5. Active scope line (subtle) ---
    _render_active_scope(state)

    # --- 6. Prompt presets (2x2, supporting role) ---
    st.caption("Suggested prompts")
    _render_prompt_presets(is_data_mode)

    # --- 7. Input section (tighter, primary/secondary buttons) ---
    placeholder = PLACEHOLDER_DATA if is_data_mode else PLACEHOLDER_MARKET
    question = st.text_input(
        "Ask a question",
        key="nlq_question",
        placeholder=placeholder,
        label_visibility="visible",
    )
    col_btn1, col_btn2, _ = st.columns([2, 1, 5])
    with col_btn1:
        run_clicked = st.button("Generate response", key="nlq_run_btn", type="primary")
    with col_btn2:
        clear_clicked = st.button("Clear", key="nlq_clear_btn")

    if clear_clicked:
        if "nlq_question" in st.session_state:
            st.session_state["nlq_question"] = ""
        if NLQ_RESPONSE_KEY in st.session_state:
            del st.session_state[NLQ_RESPONSE_KEY]
        st.rerun()

    if not run_clicked:
        _render_response_area(state, contract)
        _render_compact_history()
        return

    text = (question or "").strip()
    if not text:
        _set_nlq_response(
            intent="data_question" if is_data_mode else "market_intelligence",
            header="Response",
            subtitle="Question required",
            error="Enter a question before generating a response.",
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    route = _classify_query_route(text)
    route_to_market = False
    if route.route == "market_intelligence":
        route_to_market = True
    elif route.route == "data_question":
        route_to_market = False
    else:
        route_to_market = not is_data_mode

    # --- Market Intelligence path: Claude via Streamlit secrets; label answer as external ---
    if route_to_market:
        model = (st.session_state.get(LLM_MODEL_KEY) or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
        has_key = _claude_key_configured()
        sdk_available = _claude_sdk_available()
        logger.info("[%s] Market routing selected Claude (secret_detected=%s, sdk_available=%s, model=%s)", BUILD_MARKER, has_key, sdk_available, model)
        logger.info(
            "[%s] DEBUG Claude market_path secret_detected=%s sdk_available=%s path_selected=%s request_success=n/a",
            BUILD_MARKER,
            "yes" if has_key else "no",
            "yes" if sdk_available else "no",
            "yes" if (has_key and sdk_available and claude_generate is not None) else "no",
        )
        _set_claude_debug(
            secret_detected=has_key,
            path_selected=bool(has_key and sdk_available and claude_generate is not None),
            request_success="n/a",
        )

        if not has_key:
            _set_claude_debug(secret_detected=False, path_selected=False, request_success="no")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="Claude secret is not configured.",
                response_meta="Response type: Market Intelligence (External)",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return

        if not sdk_available or claude_generate is None:
            _set_claude_debug(secret_detected=True, path_selected=False, request_success="no")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="Anthropic SDK unavailable in deployment.",
                response_meta="Response type: Market Intelligence (External)",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return

        context = search_market_context(text)
        prompt = (
            "You are a market intelligence analyst.\n\n"
            "Rules:\n"
            "- Keep the response concise and executive-friendly (3-5 short paragraphs max).\n"
            "- Clearly separate what is observable/cited from what is inference.\n"
            "- Do not express false certainty; hedge when evidence is limited.\n"
            "- This answer is external/market-oriented only; do not reference internal portfolio data.\n"
            "- If context is thin, say so explicitly and label the answer as general-market inference.\n"
            "- Use this structure:\n"
            "  1) Executive take\n"
            "  2) What is observable now\n"
            "  3) Implications for flows/positioning\n"
            "  4) Risks and watch-items\n\n"
            f"User question: {text}\n\n"
            "Return a compact institutional note. Keep the answer explicitly external-market focused."
        )
        if (context or "").strip():
            prompt += f"\n\nContext from external sources:\n{context.strip()}"

        try:
            with st.spinner("Generating market brief..."):
                answer = claude_generate(prompt=prompt, model=model)
            logger.info("[%s] Market Claude request succeeded (model=%s)", BUILD_MARKER, model)
            logger.info("[%s] DEBUG Claude market_path secret_detected=yes sdk_available=yes path_selected=yes request_success=yes", BUILD_MARKER)
            _set_claude_debug(secret_detected=True, path_selected=True, request_success="yes")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                narrative=answer,
                provider_meta=f"Claude (Anthropic) | {model}",
                response_meta=f"Response type: Market Intelligence (External) | Provider: Claude (Anthropic) | Model: {model}",
            )
        except ClaudeError as e:
            logger.warning("[%s] Market Claude request failed: %s: %s", BUILD_MARKER, type(e).__name__, str(e))
            logger.info("[%s] DEBUG Claude market_path secret_detected=yes sdk_available=yes path_selected=yes request_success=no", BUILD_MARKER)
            _set_claude_debug(secret_detected=True, path_selected=True, request_success="no")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="Claude request failed. Verified output shown.",
                response_meta=f"Response type: Market Intelligence (External) | Provider: Claude (Anthropic) | Model: {model}",
            )
        except RuntimeError:
            logger.warning("[%s] Market Claude request failed: RuntimeError", BUILD_MARKER)
            logger.info("[%s] DEBUG Claude market_path secret_detected=yes sdk_available=yes path_selected=yes request_success=no", BUILD_MARKER)
            _set_claude_debug(secret_detected=True, path_selected=True, request_success="no")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="Claude request failed. Verified output shown.",
                response_meta=f"Response type: Market Intelligence (External) | Provider: Claude (Anthropic) | Model: {model}",
            )
        except Exception as e:
            logger.warning("[%s] Market Claude request failed unexpectedly: %s: %s", BUILD_MARKER, type(e).__name__, str(e))
            logger.info("[%s] DEBUG Claude market_path secret_detected=yes sdk_available=yes path_selected=yes request_success=no", BUILD_MARKER)
            _set_claude_debug(secret_detected=True, path_selected=True, request_success="no")
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="An unexpected error occurred. Please try again.",
                response_meta=f"Response type: Market Intelligence (External) | Provider: Claude (Anthropic) | Model: {model}",
            )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    # --- Data Questions path: classify intent, extract params, deterministic query, verified result -> Claude narrative only ---
    try:
        metric_reg = load_metric_registry()
        dim_reg = load_dim_registry()
    except (FileNotFoundError, ValueError) as e:
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=f"Registries failed to load: {e}",
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    gateway_dict = filter_state_to_gateway_dict(state)
    value_catalog = _load_value_catalog(gateway_dict, ROOT)
    today = date.today()
    extracted, qs_from_business = _intent_and_queryspec(text, metric_reg, dim_reg, value_catalog, today)
    special_intent = bool(
        extracted is not None
        and qs_from_business is None
        and extracted.intent in {"decomposition", "growth_quality_flags"}
    )
    qs = None
    if not special_intent:
        if extracted is not None and qs_from_business is not None:
            qs = qs_from_business
        else:
            spec_or_error = parse_nlq(text, metric_reg, dim_reg, value_catalog, today=today)
            if isinstance(spec_or_error, ParseError):
                _set_nlq_response(
                    intent="data_question",
                    header="Verified Data Result",
                    subtitle="Calculated from your internal filtered dataset.",
                    error=spec_or_error.message or "Parse failed.",
                )
                _render_response_area(state, contract)
                _render_compact_history()
                return
            qs = spec_or_error

    try:
        with st.spinner("Generating response..."):
            try:
                df = run_query(Q_TICKER_MONTHLY, gateway_dict, root=ROOT)
            except Exception:
                df = run_query(Q_CHANNEL_MONTHLY, gateway_dict, root=ROOT)
    except Exception as e:
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=f"Data load failed: {e}",
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    allowlist = {
        "columns": {
            "channel",
            "sub_channel",
            "product_ticker",
            "src_country",
            "country",
            "segment",
            "sub_segment",
            "month_end",
            "metric",
            "begin_aum",
            "end_aum",
            "nnb",
            "nnf",
        },
        "pii_columns": set(),
        "max_rows": 5000,
    }

    if extracted is not None and extracted.intent == "decomposition":
        # Deterministic decomposition answer from governed monthly inputs.
        base = df.copy()
        if base is None or base.empty:
            _set_nlq_response(
                intent="data_question",
                header="Verified Data Result",
                subtitle="Calculated from your internal filtered dataset.",
                narrative="",
                error="No data returned for this query.",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return
        base["month_end"] = pd.to_datetime(base.get("month_end"), errors="coerce")
        for c in ("begin_aum", "end_aum", "nnb"):
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")
        if extracted.time_start is not None:
            base = base[base["month_end"].dt.date >= extracted.time_start]
        if extracted.time_end is not None:
            base = base[base["month_end"].dt.date <= extracted.time_end]
        monthly = (
            base.groupby("month_end", as_index=False)[["begin_aum", "end_aum", "nnb"]]
            .sum(min_count=1)
            .sort_values("month_end")
        )
        if monthly.empty:
            _set_nlq_response(
                intent="data_question",
                header="Verified Data Result",
                subtitle="Calculated from your internal filtered dataset.",
                narrative="",
                error="No data returned for this query.",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return
        latest = monthly.iloc[-1]
        begin = float(latest.get("begin_aum")) if pd.notna(latest.get("begin_aum")) else float("nan")
        end = float(latest.get("end_aum")) if pd.notna(latest.get("end_aum")) else float("nan")
        nnb = float(latest.get("nnb")) if pd.notna(latest.get("nnb")) else float("nan")
        ogr = (nnb / begin) if begin and begin == begin else float("nan")
        aum_growth = ((end - begin) / begin) if begin and begin == begin and end == end else float("nan")
        delta = (aum_growth - ogr) if aum_growth == aum_growth and ogr == ogr else float("nan")

        out_df = pd.DataFrame(
            [
                {
                    "Month End": pd.Timestamp(latest["month_end"]).strftime("%Y-%m"),
                    "Beginning AUM": begin,
                    "Ending AUM": end,
                    "NNB": nnb,
                    "Organic Growth Rate": ogr,
                    "AUM Growth Rate": aum_growth,
                    "Difference (AUM - Organic)": delta,
                }
            ]
        )
        summary_md = (
            f"- Month: **{out_df.iloc[0]['Month End']}**\n"
            f"- Organic growth rate: **{fmt_percent(ogr, decimals=2)}**\n"
            f"- AUM growth rate: **{fmt_percent(aum_growth, decimals=2)}**\n"
            f"- Difference: **{fmt_percent(delta, decimals=2)}**"
        )
        narrative_payload = {
            "query": text,
            "numbers": {
                "organic_growth_rate": ogr,
                "aum_growth_rate": aum_growth,
                "difference": delta,
            },
            "top_rows_preview": out_df.to_dict(orient="records"),
        }
        narrative, narrative_warning = _get_data_narrative(narrative_payload)
        if not (narrative or "").strip():
            narrative = (
                "For the selected month, the difference between AUM growth and organic growth is shown in the verified table. "
                "AUM growth reflects total balance movement, while organic growth isolates net new business."
            )
        full_narrative = summary_md + "\n\n" + narrative
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            narrative=full_narrative,
            table_df=format_df(out_df, infer_common_formats(out_df)),
            placeholder_fallback=narrative_warning,
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    if extracted is not None and extracted.intent == "growth_quality_flags":
        base = df.copy()
        if base is None or base.empty:
            _set_nlq_response(
                intent="data_question",
                header="Verified Data Result",
                subtitle="Calculated from your internal filtered dataset.",
                narrative="",
                error="No data returned for this query.",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return
        base["month_end"] = pd.to_datetime(base.get("month_end"), errors="coerce")
        for c in ("nnb", "nnf"):
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")
        if extracted.time_start is not None:
            base = base[base["month_end"].dt.date >= extracted.time_start]
        if extracted.time_end is not None:
            base = base[base["month_end"].dt.date <= extracted.time_end]
        for dim, vals in (extracted.filters or {}).items():
            if dim in base.columns and vals:
                base = base[base[dim].astype(str).isin([str(v) for v in vals])]
        if "product_ticker" not in base.columns:
            _set_nlq_response(
                intent="data_question",
                header="Verified Data Result",
                subtitle="Calculated from your internal filtered dataset.",
                error="Ticker data not available.",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return
        agg = base.groupby("product_ticker", as_index=False).agg(nnb=("nnb", "sum"), nnf=("nnf", "sum"))
        if "etf" in _normalize_text(text):
            up = agg["product_ticker"].astype(str).str.upper()
            agg = agg[up.isin(KNOWN_ETF_TICKERS) | up.str.contains("ETF", na=False)]
        agg["fee_yield"] = agg.apply(
            lambda r: (float(r["nnf"]) / float(r["nnb"])) if pd.notna(r.get("nnb")) and float(r.get("nnb")) > 0 else float("nan"),
            axis=1,
        )
        agg = agg.dropna(subset=["nnb", "fee_yield"])
        if agg.empty:
            _set_nlq_response(
                intent="data_question",
                header="Verified Data Result",
                subtitle="Calculated from your internal filtered dataset.",
                narrative="",
                error="No data returned for this query.",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return
        nnb_med = float(agg["nnb"].median())
        fy_med = float(agg["fee_yield"].median())
        flagged = agg[(agg["nnb"] >= nnb_med) & (agg["fee_yield"] < fy_med)].copy()
        flagged = flagged.sort_values(["nnb", "fee_yield"], ascending=[False, True]).head(20)

        show = flagged.rename(columns={"product_ticker": "Ticker", "nnb": "NNB", "nnf": "NNF", "fee_yield": "Fee Yield"})
        summary_md = (
            f"- Universe rows: **{len(agg)}**\n"
            f"- NNB median: **{nnb_med:,.0f}**\n"
            f"- Fee yield median: **{fmt_percent(fy_med, decimals=2)}**\n"
            f"- Flagged (high NNB + low fee yield): **{len(flagged)}**"
        )
        narrative_payload = {
            "query": text,
            "numbers": {"nnb_median": nnb_med, "fee_yield_median": fy_med, "flagged_count": int(len(flagged))},
            "top_rows_preview": show.head(10).to_dict(orient="records"),
        }
        narrative, narrative_warning = _get_data_narrative(narrative_payload)
        if not (narrative or "").strip():
            narrative = f"Detected {len(flagged)} ticker(s) with high NNB and low fee yield versus peer medians in the selected window."
        full_narrative = summary_md + "\n\n" + narrative
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            narrative=full_narrative,
            table_df=format_df(show, infer_common_formats(show)),
            placeholder_fallback=narrative_warning,
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    with st.spinner("Generating response..."):
        result, svc_error = _execute_data_query_service(
            qs=qs,
            df=df,
            metric_reg=metric_reg,
            dim_reg=dim_reg,
            allowlist=allowlist,
        )
    if svc_error:
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=svc_error,
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    if extracted is not None:
        result = _apply_threshold_to_result(result, extracted.threshold_op, extracted.threshold_value)

    if result.meta.get("status") == "timeout":
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            error=f"NLQ query exceeded time budget ({EXECUTOR_TIMEOUT_MS} ms). Try a narrower time range or limit.",
            response_meta="Response type: Internal Data (Verified Python query)",
        )
        _render_response_area(state, contract)
        _render_compact_history()
        return

    def _full_export_provider() -> pd.DataFrame:
        r = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=True)
        return r.data if r and hasattr(r, "data") and r.data is not None else pd.DataFrame()

    bullets = build_deterministic_summary(qs, result)
    numbers = result.numbers or {}
    has_formatted = "formatted" in numbers and numbers.get("formatted") not in (None, "-")
    has_value = "value" in numbers and numbers.get("value") is not None
    headline = ""
    if has_formatted or has_value:
        label = _metric_label(result)
        display_val = numbers.get("formatted") if has_formatted else str(numbers.get("value", "-"))
        headline = f"**{label}:** {display_val}\n\n"

    narrative_payload = {
        "query": text,
        "queryspec": qs.model_dump(mode="json"),
        "numbers": numbers,
        "meta": result.meta or {},
        "deterministic_summary": bullets,
        "top_rows_preview": (result.data.head(5).to_dict(orient="records") if isinstance(result.data, pd.DataFrame) and not result.data.empty else []),
    }
    narrative, narrative_warning = _get_data_narrative(narrative_payload)
    narrative_text = headline
    if bullets:
        narrative_text += "\n".join([f"- {b}" for b in bullets]) + "\n\n"
    if (narrative or "").strip():
        narrative_text += narrative.strip()
        placeholder_fallback = None
    else:
        narrative_text += "Verified output is shown below."
        placeholder_fallback = narrative_warning or "Narrative unavailable. Verified output below."

    _set_nlq_response(
        intent="data_question",
        header="Verified Data Result",
        subtitle="Calculated from your internal filtered dataset.",
        narrative=narrative_text,
        chart_result=result,
        full_export_provider=_full_export_provider,
        placeholder_fallback=placeholder_fallback,
        response_meta="Response type: Internal Data (Verified Python query)",
    )
    _render_response_area(state, contract)
    _render_compact_history()

