"""
Intelligence Desk page (tab "nlq_chat").
Loaded by app/main.py via PAGE_RENDERERS["nlq_chat"] = render_nlq_chat — this is the only implementation.
- Data Questions: governed query, verified result, optional narrative. Chart/table in single response area.
- Market Intelligence: external search + Claude; answer labeled as external.
"""
from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data.data_gateway import Q_CHANNEL_MONTHLY, Q_TICKER_MONTHLY, run_query
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

# LLM client: optional; must not crash app if missing or broken (cloud-safe)
try:
    from app.services.llm_client import LLMError, generate_data_narrative, generate_market_intelligence
except Exception:
    generate_market_intelligence = None  # type: ignore[assignment]
    generate_data_narrative = None  # type: ignore[assignment]
    LLMError = Exception  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[2]
KNOWN_ETF_TICKERS = frozenset({"AGG", "HYG", "TIP", "MUB", "MBB", "IUSB", "SUB"})
CHAT_HISTORY_KEY = "nlq_chat_history"


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

# LLM settings: session state only; never persist key to disk
LLM_PROVIDER_KEY = "nlq_llm_provider"
LLM_MODEL_KEY = "nlq_llm_model"
LLM_API_KEY_KEY = "nlq_llm_api_key"

CLAUDE_MODELS = ["claude-3-5-sonnet-latest", "claude-3-7-sonnet-latest"]
OPENAI_MODELS = ["gpt-4.1-mini", "gpt-4.1"]


def _get_data_narrative(payload: dict[str, Any]) -> str:
    """Data Questions narrative using only UI session-state key; no env/secrets. Returns '' if no key or not Claude."""
    provider = st.session_state.get(LLM_PROVIDER_KEY) or ""
    if "Claude" not in provider:
        return ""
    api_key = (st.session_state.get(LLM_API_KEY_KEY) or "").strip()
    if not api_key or api_key == "your-key-here":
        return ""
    model = (st.session_state.get(LLM_MODEL_KEY) or "").strip()
    if not model:
        return ""
    if generate_data_narrative is None:
        return ""
    return generate_data_narrative(api_key, model, payload)


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

    error = resp.get("error")
    if error:
        st.error(error)
        return

    placeholder_fallback = resp.get("placeholder_fallback")
    if placeholder_fallback:
        st.info(placeholder_fallback)

    narrative = (resp.get("narrative") or "").strip()
    if narrative:
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
        .nlq-empty-state { border: 1px solid #2a3d67; border-radius: 8px; padding: 0.5rem 0.65rem; background: rgba(17,29,58,0.5); font-size: 0.88rem; color: #b7c5e3; margin: 0.25rem 0 0.4rem 0; }
        .nlq-empty-state .nlq-empty-title { color: #f8fbff; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.15rem; }
        /* Tighten widget spacing for this page */
        div[data-testid="stTextInput"] { margin-bottom: 0.2rem !important; }
        div[data-testid="stRadio"] { margin-bottom: 0.15rem !important; }
        .stButton { margin-top: 0.1rem !important; margin-bottom: 0.1rem !important; }
        /* Preset buttons lighter than primary action */
        .stButton button[kind="secondary"] { opacity: 0.92; padding: 0.25rem 0.6rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk: mode, presets, input, single response area. Never crashes app."""
    try:
        _render_intelligence_desk(state, contract)
    except Exception as e:
        st.error("Intelligence Desk encountered an error. Other tabs are unaffected.")
        st.exception(e)


def _render_intelligence_desk(state: FilterState, contract: dict[str, Any]) -> None:
    """Intelligence Desk UI implementation."""
    _ = contract
    _inject_nlq_page_css()

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

    # --- 3b. LLM settings (session state only; key never persisted to disk) ---
    if LLM_PROVIDER_KEY not in st.session_state:
        st.session_state[LLM_PROVIDER_KEY] = "Claude (Anthropic)"
    if LLM_MODEL_KEY not in st.session_state:
        st.session_state[LLM_MODEL_KEY] = "claude-3-5-sonnet-latest"
    if LLM_API_KEY_KEY not in st.session_state:
        st.session_state[LLM_API_KEY_KEY] = ""

    with st.expander("LLM settings (for Market Intelligence)", expanded=True):
        _provider_default_idx = 1 if st.session_state.get(LLM_PROVIDER_KEY) == "OpenAI" else 0
        provider_ui = st.radio(
            "Provider",
            ["Claude (Anthropic)", "OpenAI"],
            key="nlq_llm_provider_ui",
            horizontal=True,
            index=_provider_default_idx,
        )
        api_key_ui = st.text_input(
            "API key (optional to replace current key)",
            key="nlq_llm_api_key_input",
            type="password",
            placeholder="Enter your API key (not stored on disk)",
            label_visibility="visible",
        )
        has_saved_key = bool((st.session_state.get(LLM_API_KEY_KEY) or "").strip())
        st.caption(f"Credential status: {'Configured in session' if has_saved_key else 'Not configured'}")
        saved_model = st.session_state.get(LLM_MODEL_KEY) or ""
        if provider_ui == "Claude (Anthropic)":
            model_list = CLAUDE_MODELS
            model_index = model_list.index(saved_model) if saved_model in model_list else 0
            model_ui = st.selectbox(
                "Model",
                model_list,
                key="nlq_llm_model_claude",
                index=model_index,
            )
        else:
            model_list = OPENAI_MODELS
            model_index = model_list.index(saved_model) if saved_model in model_list else 0
            model_ui = st.selectbox(
                "Model",
                model_list,
                key="nlq_llm_model_openai",
                index=model_index,
            )
        if st.button("Apply", key="nlq_llm_apply"):
            st.session_state[LLM_PROVIDER_KEY] = provider_ui
            st.session_state[LLM_MODEL_KEY] = model_ui
            typed_key = (api_key_ui or "").strip()
            if typed_key:
                st.session_state[LLM_API_KEY_KEY] = typed_key
            st.success("Settings applied. Key is stored in session only.")
            st.rerun()

    active_provider = st.session_state.get(LLM_PROVIDER_KEY) or "Claude (Anthropic)"
    active_model = st.session_state.get(LLM_MODEL_KEY) or ""
    active_key = (st.session_state.get(LLM_API_KEY_KEY) or "").strip()
    market_ready = bool(active_model and active_key and active_key != "your-key-here")

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
    else:
        readiness = "Ready" if market_ready else "Setup required"
        st.caption(f"Mode status: {readiness} | Provider: {active_provider} | Model: {active_model or 'Not selected'}")

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

    # --- Market Intelligence path: UI-configured LLM (Claude or OpenAI); label answer as external ---
    if route_to_market:
        provider = st.session_state.get(LLM_PROVIDER_KEY) or "Claude (Anthropic)"
        model = st.session_state.get(LLM_MODEL_KEY) or ""
        api_key = st.session_state.get(LLM_API_KEY_KEY) or ""

        if not api_key or api_key == "your-key-here" or not model:
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence - external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error=(
                    "LLM settings are required for Market Intelligence. "
                    "Open LLM settings, enter a valid API key, select a model, and click Apply."
                ),
                response_meta="Response type: Market Intelligence (External)",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return

        if generate_market_intelligence is None:
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence — external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="LLM client is not available. Install app.services.llm_client dependencies.",
                response_meta="Response type: Market Intelligence (External)",
            )
            _render_response_area(state, contract)
            _render_compact_history()
            return

        provider_code = "claude" if "Claude" in provider else "openai"
        context = search_market_context(text)

        try:
            with st.spinner("Generating response..."):
                answer, meta_label = generate_market_intelligence(
                    provider=provider_code,
                    model=model,
                    api_key=api_key,
                    prompt=text,
                    context=context,
                )
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence — external sources",
                subtitle="This answer reflects external context, not your internal book.",
                narrative=answer,
                provider_meta=meta_label,
                response_meta=f"Response type: Market Intelligence (External) | Provider: {provider} | Model: {model}",
            )
        except LLMError as e:
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence — external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error=e.message,
                response_meta=f"Response type: Market Intelligence (External) | Provider: {provider} | Model: {model}",
            )
        except Exception as e:
            _set_nlq_response(
                intent="market_intelligence",
                header="Market Intelligence — external sources",
                subtitle="This answer reflects external context, not your internal book.",
                error="An unexpected error occurred. Please try again.",
                response_meta=f"Response type: Market Intelligence (External) | Provider: {provider} | Model: {model}",
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
        narrative = _get_data_narrative(narrative_payload)
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
        narrative = _get_data_narrative(narrative_payload)
        if not (narrative or "").strip():
            narrative = f"Detected {len(flagged)} ticker(s) with high NNB and low fee yield versus peer medians in the selected window."
        full_narrative = summary_md + "\n\n" + narrative
        _set_nlq_response(
            intent="data_question",
            header="Verified Data Result",
            subtitle="Calculated from your internal filtered dataset.",
            narrative=full_narrative,
            table_df=format_df(show, infer_common_formats(show)),
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
    narrative = _get_data_narrative(narrative_payload)
    narrative_text = headline
    if bullets:
        narrative_text += "\n".join([f"- {b}" for b in bullets]) + "\n\n"
    if (narrative or "").strip():
        narrative_text += narrative.strip()
        placeholder_fallback = None
    else:
        narrative_text += "Verified output is shown below."
        placeholder_fallback = "Add API key in LLM settings (Claude) to include narrative over verified outputs."

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

