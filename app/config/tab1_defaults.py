"""
Governed default state for Tab 1 (Executive Dashboard).
Single source of truth for period mode, dimension filters, and scope semantics.
Used for deterministic first-load behavior in local and Streamlit Cloud.
All defaults are explicit and documented; no silent fallbacks that change calculation meaning.
"""
from __future__ import annotations

from typing import Any

# Session key: set once when tab1 state is initialized from these defaults.
TAB1_INIT_DONE_KEY = "tab1_governed_init_done"

# --- Governed defaults (intended for first load and parity across environments) ---
# Scope: when all dimension filters are "All", KPIs use firm-wide data. That is "firm" scope.
# When any dimension filter is set, KPIs use the selected slice. That is "slice" scope.
TAB1_DEFAULT_PERIOD = "YTD"
TAB1_DEFAULT_CHANNEL = "All"
TAB1_DEFAULT_SUB_CHANNEL = "All"
TAB1_DEFAULT_COUNTRY = "All"
# NOTE: Segment filter is intentionally absent — source data is always Fixed Income.
TAB1_DEFAULT_SUB_SEGMENT = "All"
TAB1_DEFAULT_SALES_FOCUS = "All"
TAB1_DEFAULT_PRODUCT_TICKER = "All"

# Scope mode: "firm" = firm-wide (all filters All), "slice" = selected slice (any filter set).
# Default scope for KPI calculation when no filter is set is firm-wide.
SCOPE_MODE_FIRM = "firm"
SCOPE_MODE_SLICE = "slice"


def get_tab1_governed_defaults() -> dict[str, str]:
    """Return the single governed default state for Tab 1. Used for one-time session init."""
    return {
        "tab1_period": TAB1_DEFAULT_PERIOD,
        "tab1_filter_channel": TAB1_DEFAULT_CHANNEL,
        "tab1_filter_sub_channel": TAB1_DEFAULT_SUB_CHANNEL,
        "tab1_filter_country": TAB1_DEFAULT_COUNTRY,
        "tab1_filter_sub_segment": TAB1_DEFAULT_SUB_SEGMENT,
        "tab1_filter_sales_focus": TAB1_DEFAULT_SALES_FOCUS,
        "tab1_filter_ticker": TAB1_DEFAULT_PRODUCT_TICKER,
    }


def get_tab1_dimension_keys() -> list[str]:
    """Keys that define scope (when any is not 'All', scope is slice)."""
    return [
        "tab1_filter_channel",
        "tab1_filter_sub_channel",
        "tab1_filter_country",
        "tab1_filter_sub_segment",
        "tab1_filter_sales_focus",
        "tab1_filter_ticker",
    ]


def get_scope_mode_from_state(state_snapshot: dict[str, Any]) -> str:
    """
    Explicit scope mode from current tab1 dimension state.
    Returns SCOPE_MODE_FIRM when all dimension filters are 'All', else SCOPE_MODE_SLICE.
    """
    for key in get_tab1_dimension_keys():
        val = state_snapshot.get(key, "All")
        if val not in (None, "", "All"):
            return SCOPE_MODE_SLICE
    return SCOPE_MODE_FIRM


def get_scope_label_from_state(state_snapshot: dict[str, Any]) -> str:
    """
    Human-readable label for active scope used for KPI calculation.
    Firm-wide when all dimensions are All; otherwise the narrowest set dimension.
    """
    if get_scope_mode_from_state(state_snapshot) == SCOPE_MODE_FIRM:
        return "Firm-wide"
    if state_snapshot.get("tab1_filter_ticker", "All") not in (None, "", "All"):
        return f"Selected slice (Product: {state_snapshot.get('tab1_filter_ticker', 'All')})"
    if state_snapshot.get("tab1_filter_sub_segment", "All") not in (None, "", "All"):
        return f"Selected slice (Sub-segment: {state_snapshot.get('tab1_filter_sub_segment', 'All')})"
    if state_snapshot.get("tab1_filter_sales_focus", "All") not in (None, "", "All"):
        return f"Selected slice (Sales Focus: {state_snapshot.get('tab1_filter_sales_focus', 'All')})"
    if state_snapshot.get("tab1_filter_country", "All") not in (None, "", "All"):
        return f"Selected slice (Geography: {state_snapshot.get('tab1_filter_country', 'All')})"
    if state_snapshot.get("tab1_filter_sub_channel", "All") not in (None, "", "All"):
        return f"Selected slice (Sub-channel: {state_snapshot.get('tab1_filter_sub_channel', 'All')})"
    if state_snapshot.get("tab1_filter_channel", "All") not in (None, "", "All"):
        return f"Selected slice (Channel: {state_snapshot.get('tab1_filter_channel', 'All')})"
    return "Firm-wide"
