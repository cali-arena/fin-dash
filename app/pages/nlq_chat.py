"""
NLQ Chat tab: user question → parse_nlq → QuerySpec/ParseError → execute_queryspec → render.
No LLM. Governed parser and executor only; data via gateway.
Uses app.ui.guardrails and app.ui.formatters for empty state, chart fallback, and table display.
"""
from __future__ import annotations

import json
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
from app.nlq.parser import ParseError, parse_nlq, to_json
from app.state import FilterState, filter_state_to_gateway_dict, get_filter_state
from app.nlq.executor import EXECUTOR_TIMEOUT_MS
from app.ui.formatters import format_df, infer_common_formats
from app.ui.exports import render_export_buttons
from app.ui.guardrails import fallback_note, render_chart_or_fallback, render_empty_state, render_timeout_state
from app.ui.theme import apply_enterprise_plotly_style, safe_render_plotly

try:
    from app.observability import render_obs_panel
except ImportError:
    def render_obs_panel(_tab_id: str) -> None:
        pass

from app.ui.observability import render_observability_panel

ROOT = Path(__file__).resolve().parents[2]


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
    for col in ("channel", "product_ticker", "src_country", "segment", "month_end"):
        if col in df.columns:
            catalog[col] = set(df[col].dropna().astype(str).str.strip().unique().tolist())
    return catalog


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
    recovery_hint = "Try one of: " + "; ".join(hint_parts) if hint_parts else "Rephrase the question or check metric/dimension names."
    render_empty_state(message, recovery_hint, icon="⚠️")
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
        render_empty_state("No rows returned.", "Adjust filters or time range.")
        return
    if df.empty:
        render_empty_state("No rows returned.", "Adjust filters or time range.")
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
        safe_render_plotly(fig, user_message="Line chart unavailable for this selection.")

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
        safe_render_plotly(fig, user_message="Bar chart unavailable for this selection.")

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
    has_formatted = "formatted" in numbers and numbers.get("formatted") not in (None, "—")
    has_value = "value" in numbers and numbers.get("value") is not None

    if has_formatted or has_value:
        st.subheader("Portfolio Answer")
        label = _metric_label(result)
        display_val = numbers.get("formatted") if has_formatted else str(numbers.get("value", "—"))
        st.metric(label=label, value=display_val)
    else:
        st.subheader("Analytics Result")

    meta = result.meta or {}
    if meta.get("rows_capped") or any("clamp" in str(w).lower() for w in meta.get("warnings", [])):
        st.info("Results capped for safety. Adjust query or use export flow if needed.")

    render_chart(result, full_export_provider=full_export_provider, allow_full=allow_full)


def render(state: FilterState, contract: dict[str, Any]) -> None:
    """NLQ Chat: question input → Run → parse → validate → execute → render."""
    st.title("Guided Analytics")
    st.caption("Governed analytics assistant with deterministic execution and controlled metric definitions.")
    with st.expander("Example prompts", expanded=False):
        st.markdown("- Show net new business by channel over the last 12 months")
        st.markdown("- Compare ETF flows by geography YTD")
        st.markdown("- Which channels had strongest organic growth last quarter?")
    question = st.text_input(
        "Ask an analytics question",
        key="nlq_question",
        placeholder="e.g., Show net new business by channel over the last 12 months",
    )
    run_clicked = st.button("Run analysis", key="nlq_run_btn")

    if not run_clicked:
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    text = (question or "").strip()
    if not text:
        st.warning("Enter an analytics question.")
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

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
        st.caption("Value catalog unavailable; governed value checks are limited for this run.")

    spec_or_error = parse_nlq(text, metric_reg, dim_reg, value_catalog, today=date.today())

    if isinstance(spec_or_error, ParseError):
        _render_parse_error(spec_or_error)
        if st.session_state.get("dev_mode"):
            with st.expander("Debug (dev)", expanded=False):
                st.json(to_json(spec_or_error))
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    qs = spec_or_error
    validation_logs: list[str] = []
    try:
        validate_queryspec(qs, metric_reg, dim_reg, out_logs=validation_logs)
    except GovernanceError as e:
        st.error(f"Validation failed: {e}")
        validation_logs.append(str(e))
        if st.session_state.get("dev_mode"):
            with st.expander("Debug (dev)", expanded=False):
                st.json(qs.model_dump(mode="json"))
                st.caption("Validation logs")
                st.text("\n".join(validation_logs))
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    try:
        df = run_query(Q_CHANNEL_MONTHLY, gateway_dict, root=ROOT)
    except Exception as e:
        st.error(f"Data load failed: {e}")
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    allowlist = {
        "columns": {"channel", "product_ticker", "src_country", "segment", "month_end", "metric"},
        "pii_columns": set(),
        "max_rows": 5000,
    }
    try:
        result = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=False)
    except ValueError as e:
        st.error(f"Execution (allowlist): {e}")
        if st.session_state.get("dev_mode"):
            with st.expander("Debug (dev)", expanded=False):
                st.json(qs.model_dump(mode="json"))
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return
    except GovernanceError as e:
        st.error(f"Governance: {e}")
        if st.session_state.get("dev_mode"):
            with st.expander("Debug (dev)", expanded=False):
                st.json(qs.model_dump(mode="json"))
        render_obs_panel(contract.get("tab_id", "nlq_chat"))
        return

    if result.meta.get("status") == "timeout":
        render_timeout_state("NLQ query", EXECUTOR_TIMEOUT_MS, "Try adding a limit or narrower time range.")

    def _full_export_provider() -> pd.DataFrame:
        r = execute_queryspec(qs, df, metric_reg, dim_reg, allowlist, export_mode=True)
        return r.data if r and hasattr(r, "data") and r.data is not None else pd.DataFrame()

    st.subheader("Explanation")
    bullets = build_deterministic_summary(qs, result)
    if bullets:
        st.markdown("\n".join([f"- {b}" for b in bullets]))
    else:
        st.caption("No deterministic explanation available for this result.")

    _render_result(
        result,
        full_export_provider=_full_export_provider,
        allow_full=st.session_state.get("export_mode_toggle", False),
    )

    with st.expander("Guided Summary (facts)", expanded=True):
        st.markdown("\n".join([f"- {b}" for b in bullets]))

    dataset_version = st.session_state.get("dataset_version")
    filter_hash = state.filter_state_hash() if hasattr(state, "filter_state_hash") and callable(getattr(state, "filter_state_hash")) else None
    executed_at = datetime.now(timezone.utc).isoformat()
    with st.expander("Governance Audit", expanded=False):
        st.text(f"dataset_version: {dataset_version or '—'}")
        st.text(f"filter_hash: {filter_hash or '—'}")
        st.text(f"executed_at: {executed_at}")
        st.caption("QuerySpec JSON")
        qs_json = json.dumps(qs.model_dump(mode="json"), indent=2)
        st.code(qs_json, language="json")

    explain_enabled = st.toggle("Explain (optional)", value=False, key="nlq_explain_toggle")
    if explain_enabled:
        payload = build_explain_payload(qs, result, dataset_version=dataset_version, filter_hash=filter_hash)
        explanation = llm_explain(payload)
        if not (explanation or "").strip():
            st.markdown("\n".join([f"- {b}" for b in bullets]))
        else:
            ok, disallowed = validate_explanation_numbers(explanation, payload)
            if not ok:
                st.warning("Explanation rejected (introduced unsupported numbers). Showing deterministic summary.")
                st.markdown("\n".join([f"- {b}" for b in bullets]))
            else:
                st.markdown(explanation)

    if st.session_state.get("dev_mode"):
        with st.expander("Debug (dev)", expanded=False):
            st.caption("QuerySpec")
            st.json(qs.model_dump(mode="json"))
            st.caption("Validation logs")
            st.text("\n".join(validation_logs))
            st.caption("Executor meta")
            st.json(result.meta)

    render_observability_panel(filters=state, drill_state=None, queryspec=qs)
    render_obs_panel(contract.get("tab_id", "nlq_chat"))
