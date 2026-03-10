"""
Tab 3: Chat with two explicit modes.
- Data Questions: Python classifies intent, extracts parameters, runs deterministic query; Claude receives verified output only and writes narrative. Chart/table auto-triggered.
- Market Intelligence: external search + Claude; answer clearly labeled as external.
Claude never performs calculations or touches raw internal data.
"""
from __future__ import annotations

import json
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data.data_gateway import Q_CHANNEL_MONTHLY, run_query
from app.nlq.deterministic_summary import build_deterministic_summary
from app.nlq.executor import QueryResult, execute_queryspec
from app.nlq.explain import (
    build_explain_payload,
    llm_explain,
    validate_explanation_numbers,
)
from app.nlq.governance import GovernanceError, load_dim_registry, load_metric_registry, validate_queryspec
from app.nlq.market_search import search_market_context
from app.nlq.parser import ParseError, parse_nlq, to_json
from app.state import FilterState, filter_state_to_gateway_dict, get_filter_state
from app.nlq.executor import EXECUTOR_TIMEOUT_MS
from app.ui.formatters import fmt_percent, format_df, infer_common_formats
from app.ui.exports import render_export_buttons
from app.ui.guardrails import fallback_note, render_chart_or_fallback, render_empty_state, render_timeout_state
from app.ui.theme import apply_enterprise_plotly_style, safe_render_plotly

try:
    from app.llm.claude import claude_market_intelligence, claude_narrative_from_payload
except ImportError:
    def claude_market_intelligence(_q: str, _c: str) -> str:
        return ""
    def claude_narrative_from_payload(_payload: dict[str, Any]) -> str:
        return ""

try:
    from app.observability import render_obs_panel
except ImportError:
    def render_obs_panel(_tab_id: str) -> None:
        pass

from app.ui.observability import render_observability_panel

ROOT = Path(__file__).resolve().parents[2]
CHAT_HISTORY_KEY = "nlq_chat_history"
KNOWN_ETF_TICKERS = frozenset({"AGG", "HYG", "TIP", "MUB", "MBB", "IUSB", "SUB"})


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


def _render_chat_history() -> None:
    hist = st.session_state.get(CHAT_HISTORY_KEY) or []
    if not hist:
        return
    st.markdown("#### Conversation")
    for msg in hist:
        role = msg.get("role", "assistant")
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg.get("text", ""))


def _load_value_catalog(gateway_dict: dict[str, Any], root: Path) -> dict[str, set[str]]:
    """
    Build value_catalog (distinct values per dim) from gateway. Cached by dataset_version via run_query.
    Returns dict of dim -> set of values. If data not available, return empty sets.
    """
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


def _render_parse_error(err: ParseError) -> None:
    """Governed error block: render_empty_state with message + recovery hint listing suggestions."""
    message = err.message or "Parse failed."
    suggestions = err.suggestions or {}
    hint_parts: list[str] = []
    if isinstance(suggestions, dict):
        for key in ("metrics", "dimensions", "values"):
            items = suggestions.get(key)
            if items is None:
                continue
            if isinstance(items, dict):
                hint_parts.append(f"{key}: {', '.join(str(v) for v in list(items.values())[:5])}")
            elif isinstance(items, (list, tuple)):
                hint_parts.append(f"{key}: {', '.join(str(x) for x in items[:5])}")
            else:
                hint_parts.append(f"{key}: {items}")
    recovery_hint = (
        "Try one of: " + "; ".join(hint_parts)
        if hint_parts
        else "Rephrase the question using portfolio metrics, filters, and a time window."
    )
    render_empty_state(message, recovery_hint, icon="!")
    _render_supported_question_examples()
    if err.details:
        with st.expander("Details", expanded=False):
            st.json(err.details)


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
    st.dataframe(df_show, height=420, use_container_width=True, hide_index=True)
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


MARKET_INTELLIGENCE_LABEL = (
    "**Market Intelligence** - this answer draws on external sources, not your internal data."
)
INTERNAL_DATA_LABEL = (
    "**Internal Data Answer** - this result is generated by deterministic governed Python logic over your internal dataset."
)

DATA_PROMPTS = [
    "Show net new business by channel for YTD and highlight the top contributors.",
    "Which ETFs have the highest net new flow this quarter?",
    "Compare fee yield by product ticker for the last 12 months.",
    "Identify channels with negative market impact but positive NNB.",
]

MARKET_PROMPTS = [
    "What are current Fed rate expectations and how could they affect multi-asset flows?",
    "Summarize this week's global equity and bond market drivers.",
    "What is the latest outlook for ETF inflows across major asset classes?",
    "How are inflation surprises affecting duration and credit positioning?",
]

SUPPORTED_DATA_QUESTION_EXAMPLES = [
    "Contributors in Broker Dealer channel with NNB above $100k",
    "ETFs with high NNB but low fee yield in Q3",
    "Difference between organic growth and AUM growth in June",
    "Products in Wealth channel where fee yield is below 0.5%",
]


def _render_supported_question_examples() -> None:
    st.markdown("**Try questions like:**")
    for q in SUPPORTED_DATA_QUESTION_EXAMPLES:
        st.markdown(f"- {q}")


def _render_active_filter_scope(state: FilterState) -> None:
    scope = "Enterprise-wide portfolio"
    slice_dim = getattr(state, "slice_dim", None)
    slice_value = getattr(state, "slice_value", None)
    if slice_dim and slice_value:
        scope = f"{slice_dim}: {slice_value}"
    date_start = getattr(state, "date_start", None)
    date_end = getattr(state, "date_end", None)
    if date_start and date_end:
        st.caption(f"Active portfolio scope: {scope}. Reporting window: {date_start} to {date_end}.")
    else:
        st.caption(f"Active portfolio scope: {scope}.")


def _render_prompt_presets(is_data_mode: bool) -> None:
    prompts = DATA_PROMPTS if is_data_mode else MARKET_PROMPTS
    st.caption("Prompt presets")
    cols = st.columns(2)
    for i, prompt in enumerate(prompts):
        with cols[i % 2]:
            if st.button(prompt, key=f"nlq_preset_{'data' if is_data_mode else 'market'}_{i}", use_container_width=True):
                st.session_state["nlq_question"] = prompt


def render(state: FilterState, contract: dict[str, Any]) -> None:
    """Tab 3: two modes - Data Questions (deterministic query + Claude narrative) or Market Intelligence (external + Claude)."""
    st.title("Intelligence Desk")
    st.markdown(
        "<div class='section-subtitle'>Ask data questions over your internal book or market intelligence over external sources. Results are verified; narrative is optional.</div>",
        unsafe_allow_html=True,
    )
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []
    st.markdown(
        "<div class='section-frame'>"
        "<strong>Two ways to ask:</strong> "
        "<strong>Data Questions</strong> = your internal data, governed query, verified result. "
        "<strong>Market Intelligence</strong> = external sources (rates, macro), answer clearly labeled as external."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Choose mode**")
    mode = st.radio(
        "Mode",
        ["Data Questions", "Market Intelligence"],
        key="nlq_mode",
        horizontal=True,
        format_func=lambda x: x,
        label_visibility="collapsed",
    )
    is_data_mode = mode == "Data Questions"
    mode_label = "Data Questions"
    mode_short = "Internal data - verified numbers - optional narrative"
    if not is_data_mode:
        mode_label = "Market Intelligence"
        mode_short = "External sources - answer labeled as Market Intelligence"
    st.markdown(
        f"<div class='nlq-mode-badge'>"
        f"<div class='nlq-mode-label'>Current mode: {mode_label}</div>"
        f"{mode_short}"
        f"</div>",
        unsafe_allow_html=True,
    )
    _render_chat_history()
    _render_active_filter_scope(state)
    st.markdown("---")
    st.subheader("Ask your question" if is_data_mode else "Ask a market intelligence question")
    if is_data_mode:
        st.caption("We parse your question, run a governed query on internal data, then show verified results and an optional narrative.")
        placeholder = "e.g., Show net new business by channel over the last 12 months"
    else:
        st.caption("Answers use external sources and are labeled as Market Intelligence. Placeholder may appear if external search is not configured.")
        placeholder = "e.g., What are current Fed rate expectations?"
    _render_prompt_presets(is_data_mode)
    with st.expander("Example prompts", expanded=False):
        if is_data_mode:
            st.markdown("- Show net new business by channel over the last 12 months")
            st.markdown("- Compare ETF flows by geography YTD")
            st.markdown("- Which channels had strongest organic growth last quarter?")
            st.markdown("- Which products are driving fee yield improvement?")
        else:
            st.markdown("- What are current Fed rate expectations?")
            st.markdown("- How did equity markets perform last month?")
            st.markdown("- What are current drivers of ETF flows globally?")
    question = st.text_input(
        "Question" if is_data_mode else "Market question",
        key="nlq_question",
        placeholder=placeholder,
    )
    run_clicked = st.button("Generate response", key="nlq_run_btn")

    if not run_clicked:
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    text = (question or "").strip()
    if not text:
        st.warning("Enter a question.")
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return
    st.session_state[CHAT_HISTORY_KEY].append({"role": "user", "text": text, "mode": mode})

    route = _classify_query_route(text)
    route_to_market = False
    if route.route == "market_intelligence":
        route_to_market = True
        if is_data_mode:
            st.caption("Routed to Market Intelligence based on question intent.")
    elif route.route == "data_question":
        route_to_market = False
        if not is_data_mode:
            st.caption("Routed to Data Questions based on question intent.")
    else:
        route_to_market = not is_data_mode
        st.caption(f"Intent is mixed; using selected mode ({mode}).")

    # --- Market Intelligence path: external search + Claude; label answer as external ---
    if route_to_market:
        with st.spinner("Searching external sources..."):
            context = search_market_context(text)
        with st.spinner("Generating answer..."):
            answer = claude_market_intelligence(text, context)
        st.markdown(
            "<div class='section-frame'>"
            "<strong>Market Intelligence</strong> - This answer uses external sources, not your internal data."
            "</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Market Intelligence answer")
        with st.expander("External context used", expanded=False):
            st.code(context or "(no external context retrieved)", language="text")
        if (answer or "").strip():
            st.markdown(answer)
            st.session_state[CHAT_HISTORY_KEY].append(
                {"role": "assistant", "mode": mode, "text": f"**Market Intelligence**\n\n{answer}"}
            )
        else:
            st.caption("External intelligence is currently unavailable. Configure ANTHROPIC_API_KEY and optional search provider integration.")
            st.session_state[CHAT_HISTORY_KEY].append(
                {"role": "assistant", "mode": mode, "text": "Market Intelligence is not available. Configure external search and ANTHROPIC_API_KEY to enable it."}
            )
        render_observability_panel(filters=state, drill_state=None, queryspec=None)
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    # --- Data Questions path: classify intent, extract params, deterministic query, verified result -> Claude narrative only ---
    try:
        metric_reg = load_metric_registry()
        dim_reg = load_dim_registry()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Registries failed to load: {e}")
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    gateway_dict = filter_state_to_gateway_dict(state)
    value_catalog = _load_value_catalog(gateway_dict, ROOT)
    if not value_catalog:
        st.caption("Some filter values are unavailable in this run. Query validation will still proceed with governed metrics and dimensions.")

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
                _render_parse_error(spec_or_error)
                if st.session_state.get("dev_mode"):
                    with st.expander("Debug (dev)", expanded=False):
                        st.json(to_json(spec_or_error))
                render_obs_panel(contract.get("tab_id", "nlq_chat"))
                return
            qs = spec_or_error

    st.markdown(
        "<div class='section-frame'>"
        "<strong>Internal data answer</strong> - This result is from a governed query over your internal dataset. Numbers are verified; narrative is generated from these facts only."
        "</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Internal data answer")

    try:
        with st.spinner("Loading governed dataset..."):
            df = run_query(Q_CHANNEL_MONTHLY, gateway_dict, root=ROOT)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
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
            render_empty_state("No data returned for this query.", "Adjust filters or time range.")
            render_obs_panel(contract.get("tab_id", "nlq_chat"))
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
            render_empty_state("No data returned for this query.", "Adjust filters or time range.")
            render_obs_panel(contract.get("tab_id", "nlq_chat"))
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
        st.subheader("Deterministic Summary")
        st.markdown(
            f"- Month: **{out_df.iloc[0]['Month End']}**\n"
            f"- Organic growth rate: **{fmt_percent(ogr, decimals=2)}**\n"
            f"- AUM growth rate: **{fmt_percent(aum_growth, decimals=2)}**\n"
            f"- Difference: **{fmt_percent(delta, decimals=2)}**"
        )
        st.subheader("Verified result")
        st.dataframe(format_df(out_df, infer_common_formats(out_df)), use_container_width=True, hide_index=True)
        narrative_payload = {
            "query": text,
            "numbers": {
                "organic_growth_rate": ogr,
                "aum_growth_rate": aum_growth,
                "difference": delta,
            },
            "top_rows_preview": out_df.to_dict(orient="records"),
        }
        narrative = claude_narrative_from_payload(narrative_payload)
        st.markdown("#### Narrative Summary")
        if (narrative or "").strip():
            st.markdown(narrative)
            st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": narrative})
        else:
            fallback = (
                "For the selected month, the difference between AUM growth and organic growth is shown in the verified table above. "
                "AUM growth reflects total balance movement, while organic growth isolates net new business."
            )
            st.markdown(fallback)
            st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": fallback})
        render_observability_panel(filters=state, drill_state=None, queryspec=None)
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    if extracted is not None and extracted.intent == "growth_quality_flags":
        base = df.copy()
        if base is None or base.empty:
            render_empty_state("No data returned for this query.", "Adjust filters or time range.")
            render_obs_panel(contract.get("tab_id", "nlq_chat"))
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
            render_empty_state("Ticker data not available.", "Try a different question or data slice.")
            render_obs_panel(contract.get("tab_id", "nlq_chat"))
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
            render_empty_state("No data returned for this query.", "Adjust filters or time range.")
            render_obs_panel(contract.get("tab_id", "nlq_chat"))
            return
        nnb_med = float(agg["nnb"].median())
        fy_med = float(agg["fee_yield"].median())
        flagged = agg[(agg["nnb"] >= nnb_med) & (agg["fee_yield"] < fy_med)].copy()
        flagged = flagged.sort_values(["nnb", "fee_yield"], ascending=[False, True]).head(20)

        st.subheader("Deterministic Summary")
        st.markdown(
            f"- Universe rows: **{len(agg)}**\n"
            f"- NNB median: **{nnb_med:,.0f}**\n"
            f"- Fee yield median: **{fmt_percent(fy_med, decimals=2)}**\n"
            f"- Flagged (high NNB + low fee yield): **{len(flagged)}**"
        )
        st.subheader("Verified result")
        show = flagged.rename(columns={"product_ticker": "Ticker", "nnb": "NNB", "nnf": "NNF", "fee_yield": "Fee Yield"})
        st.dataframe(format_df(show, infer_common_formats(show)), use_container_width=True, hide_index=True)

        narrative_payload = {
            "query": text,
            "numbers": {"nnb_median": nnb_med, "fee_yield_median": fy_med, "flagged_count": int(len(flagged))},
            "top_rows_preview": show.head(10).to_dict(orient="records"),
        }
        narrative = claude_narrative_from_payload(narrative_payload)
        st.markdown("#### Narrative Summary")
        if (narrative or "").strip():
            st.markdown(narrative)
            st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": narrative})
        else:
            fallback = (
                f"Detected {len(flagged)} ticker(s) with high NNB and low fee yield versus peer medians in the selected window."
            )
            st.markdown(fallback)
            st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": fallback})
        render_observability_panel(filters=state, drill_state=None, queryspec=None)
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    with st.spinner("Running deterministic query..."):
        result, svc_error = _execute_data_query_service(
            qs=qs,
            df=df,
            metric_reg=metric_reg,
            dim_reg=dim_reg,
            allowlist=allowlist,
        )
    if svc_error:
        st.error(svc_error)
        if st.session_state.get("dev_mode") and qs is not None:
            with st.expander("Debug (dev)", expanded=False):
                st.json(qs.model_dump(mode="json"))
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    if extracted is not None:
        result = _apply_threshold_to_result(result, extracted.threshold_op, extracted.threshold_value)

    if result.meta.get("status") == "timeout":
        render_timeout_state("NLQ query", EXECUTOR_TIMEOUT_MS, "Try adding a limit or narrower time range.")

    with st.expander("How This Question Was Interpreted", expanded=False):
        try:
            st.json(qs.model_dump(mode="json"))
        except Exception:
            st.caption("Unable to render extracted parameters for this query.")

    def _full_export_provider() -> pd.DataFrame:
        r = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=True)
        return r.data if r and hasattr(r, "data") and r.data is not None else pd.DataFrame()

    st.subheader("Deterministic Summary")
    bullets = build_deterministic_summary(qs, result)
    if bullets:
        st.markdown("\n".join([f"- {b}" for b in bullets]))
    else:
        st.caption("No deterministic explanation available for this result.")

    st.subheader("Verified result")
    _render_result(
        result,
        full_export_provider=_full_export_provider,
        allow_full=st.session_state.get("export_mode_toggle", False),
    )

    with st.expander("Verified Evidence Summary", expanded=True):
        st.markdown("\n".join([f"- {b}" for b in bullets]))

    narrative_payload = {
        "query": text,
        "queryspec": qs.model_dump(mode="json"),
        "numbers": result.numbers or {},
        "meta": result.meta or {},
        "deterministic_summary": bullets,
        "top_rows_preview": (result.data.head(5).to_dict(orient="records") if isinstance(result.data, pd.DataFrame) and not result.data.empty else []),
    }
    with st.spinner("Generating narrative wording from verified output..."):
        narrative = claude_narrative_from_payload(narrative_payload)
    st.markdown("#### Narrative Summary")
    if (narrative or "").strip():
        st.markdown(narrative)
        st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": narrative})
    else:
        st.caption("Narrative wording is not available. Set ANTHROPIC_API_KEY to enable narrative over verified outputs.")
        fallback = "Verified output is shown above. Narrative wording is not available until ANTHROPIC_API_KEY is configured."
        st.session_state[CHAT_HISTORY_KEY].append({"role": "assistant", "mode": mode, "text": fallback})

    dataset_version = st.session_state.get("dataset_version")
    filter_hash = state.filter_state_hash() if hasattr(state, "filter_state_hash") and callable(getattr(state, "filter_state_hash")) else None
    if st.session_state.get("dev_mode"):
        executed_at = datetime.now(timezone.utc).isoformat()
        with st.expander("Audit Trail", expanded=False):
            st.text(f"dataset_version: {dataset_version or '-'}")
            st.text(f"filter_hash: {filter_hash or '-'}")
            st.text(f"executed_at: {executed_at}")
            st.caption("QuerySpec JSON")
            qs_json = json.dumps(qs.model_dump(mode="json"), indent=2)
            st.code(qs_json, language="json")

        explain_enabled = st.toggle("Additional Narrative Check (optional)", value=False, key="nlq_explain_toggle")
        if explain_enabled:
            payload = build_explain_payload(qs, result, dataset_version=dataset_version, filter_hash=filter_hash)
            explanation = llm_explain(payload)
            if (explanation or "").strip():
                ok, _ = validate_explanation_numbers(explanation, payload)
                if ok:
                    st.markdown("#### Additional Explainability Layer")
                    st.markdown(explanation)
                else:
                    st.warning("Additional explainability output was rejected because it introduced unsupported numbers.")

    if st.session_state.get("dev_mode"):
        with st.expander("Debug (dev)", expanded=False):
            st.caption("QuerySpec")
            st.json(qs.model_dump(mode="json"))
            st.caption("Executor meta")
            st.json(result.meta)

    render_observability_panel(filters=state, drill_state=None, queryspec=qs)
    render_obs_panel(contract.get("tab_id", "nlq_chat"))

