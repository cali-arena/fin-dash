"""Central guardrails for empty states, row caps, and chart fallback."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

DEFAULT_EMPTY_REASON = "No data available."
DEFAULT_EMPTY_HINT = "Adjust filters or date range."


def render_empty_state(reason: str, recovery_hint: str, icon: str = "WARNING") -> None:
    icon_norm = str(icon or "").strip().upper()
    user_changed = bool(st.session_state.get("ui_filters_user_changed", False))
    message = f"{reason} {recovery_hint}"
    if icon_norm == "ERROR":
        st.error(f"[ERROR] {message}")
        return
    if icon_norm == "WARNING" and user_changed:
        st.warning(f"[WARNING] {message}")
        return
    st.markdown(
        (
            "<div class='empty-state-card'>"
            f"<div class='empty-state-title'>{reason}</div>"
            f"<div>{recovery_hint}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_timeout_state(op_name: str, budget_ms: int, hint: str) -> None:
    render_empty_state(f"{op_name} exceeded query budget ({budget_ms} ms).", hint, icon="INFO")


def render_error_state(context: str, error: Exception, hint: str = "Try adjusting filters or reload the page.") -> None:
    render_empty_state(f"{context} unavailable.", hint, icon="INFO")
    if st.session_state.get("dev_mode"):
        with st.expander(f"{context} error details (dev)", expanded=False):
            st.exception(error)


def missing_required_columns(df: pd.DataFrame | None, required_cols: list[str]) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame):
        return list(required_cols)
    return [c for c in required_cols if c not in df.columns]


def ensure_non_empty(
    df: pd.DataFrame | None,
    reason: str = DEFAULT_EMPTY_REASON,
    hint: str = DEFAULT_EMPTY_HINT,
) -> bool:
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        render_empty_state(reason, hint)
        return False
    return True


def ensure_min_points(
    df: pd.DataFrame | None,
    x_col: str | None,
    min_points: int = 2,
    reason: str = DEFAULT_EMPTY_REASON,
    hint: str = DEFAULT_EMPTY_HINT,
) -> bool:
    if not ensure_non_empty(df, reason, hint):
        return False
    assert df is not None
    if len(df) < min_points:
        render_empty_state(reason, hint)
        return False
    if x_col and x_col in df.columns and df[x_col].nunique() < min_points:
        render_empty_state(reason, hint)
        return False
    return True


def apply_max_rows(df: pd.DataFrame, max_rows: int) -> tuple[pd.DataFrame, bool]:
    if df is None or df.empty:
        return pd.DataFrame(), False
    if len(df) <= max_rows:
        return df, False
    return df.head(max_rows).copy(), True


def render_selection_reset(reason: str = "Selection reset because it is invalid under current filters.") -> None:
    st.caption(reason)


def render_chart_or_fallback(
    chart_fn: Callable[[], None],
    df: pd.DataFrame | None,
    fallback_cols: list[str],
    note: str,
    min_points: int = 2,
    empty_reason: str = DEFAULT_EMPTY_REASON,
    empty_hint: str = DEFAULT_EMPTY_HINT,
) -> None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        render_empty_state(empty_reason, empty_hint)
        return
    if len(df) < min_points:
        st.caption(note)
        cols = [c for c in fallback_cols if c in df.columns]
        st.dataframe(df[cols] if cols else df, width="stretch", hide_index=True)
        return
    try:
        chart_fn()
    except Exception as exc:
        render_error_state("Chart rendering", exc, "Chart unavailable for this selection.")


def fallback_note(kind: str, details: dict[str, Any] | None = None) -> str:
    details = details or {}
    minimum = details.get("min_points", 2)
    if kind in {"insufficient_trend", "insufficient_points", "chart_insufficient_points"}:
        return f"Insufficient points to render chart (need {minimum}+). Showing table."
    if kind == "selection_no_rows":
        return "Current selection has no rows under active filters."
    return "Showing table fallback."
