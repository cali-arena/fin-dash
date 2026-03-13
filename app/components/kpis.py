"""
KPI card components. Streamlit primitives only; no state or data access.
"""
from __future__ import annotations

import math
from typing import Any

import streamlit as st
from app.ui.formatters import fmt_currency_kpi, fmt_percent


def _as_metric_value(value: Any) -> str:
    """Render-safe metric value: never expose raw nan/None."""
    if value is None:
        return "—"
    if isinstance(value, float) and math.isnan(value):
        return "—"
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", ""}:
        return "—"
    return str(value)


def _as_metric_delta(value: Any) -> str | None:
    """Render-safe delta: hide invalid values."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", ""}:
        return None
    return str(value)


def render_kpi_row(kpis: list[dict[str, Any]]) -> None:
    """
    Render a row/grid of KPI cards using Streamlit primitives.
    Each item: { "label": str, "value": str, "delta": str | None, "help": str | None }
    """
    if not kpis:
        return
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            label = kpi.get("label") or ""
            value = _as_metric_value(kpi.get("value"))
            delta = _as_metric_delta(kpi.get("delta"))
            help_text = kpi.get("help")
            st.metric(
                label=label,
                value=value,
                delta=delta,
                help=help_text,
            )


def build_executive_overview_primary_kpis(kpi_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Canonical 5-card Executive Overview row used by tab 1 and tab 2.
    Keeps formatting and ordering consistent.
    """
    return [
        {"label": "Selected Scope End AUM", "value": fmt_currency_kpi(kpi_snapshot.get("end_aum")), "delta": None, "help": None},
        {"label": "Net New Business", "value": fmt_currency_kpi(kpi_snapshot.get("nnb")), "delta": None, "help": None},
        {"label": "Net New Flow", "value": fmt_currency_kpi(kpi_snapshot.get("nnf")), "delta": None, "help": None},
        {"label": "Organic Growth", "value": fmt_percent(kpi_snapshot.get("ogr"), decimals=2, signed=False), "delta": None, "help": None},
        {"label": "Market Movement", "value": fmt_currency_kpi(kpi_snapshot.get("market_pnl")), "delta": None, "help": None},
    ]


def build_executive_overview_secondary_kpis(kpi_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Optional second row for tab 2, matching formatter policy used in the primary row."""
    return [
        {"label": "Begin AUM", "value": fmt_currency_kpi(kpi_snapshot.get("begin_aum")), "delta": None, "help": None},
        {"label": "Fee Yield", "value": fmt_percent(kpi_snapshot.get("fee_yield"), decimals=2, signed=False), "delta": None, "help": None},
    ]


def example_kpis() -> list[dict[str, Any]]:
    """Return a list of demo KPIs for placeholder or testing."""
    return [
        {"label": "Total AUM", "value": "—", "delta": None, "help": "Sum of end-period AUM"},
        {"label": "NNB", "value": "—", "delta": None, "help": "Net new business"},
        {"label": "NNF", "value": "—", "delta": None, "help": "Net new flows"},
        {"label": "Row count", "value": "0", "delta": None, "help": "Number of records"},
    ]


def kpis_from_gateway_dict(gw: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build KPI list for render_kpi_row from gateway kpi_firm_global() dict.
    Keys: total_aum, total_nnb, growth_rate_pct, row_count.
    """
    growth_raw = gw.get("growth_rate_pct")
    growth_value = "—"
    if isinstance(growth_raw, (int, float)) and not (isinstance(growth_raw, float) and math.isnan(growth_raw)):
        growth_value = f"{growth_raw}%"
    return [
        {"label": "Total AUM", "value": gw.get("total_aum", "—"), "delta": None, "help": "Sum of end-period AUM"},
        {"label": "NNB", "value": gw.get("total_nnb", "—"), "delta": None, "help": "Net new business"},
        {"label": "Growth %", "value": growth_value, "delta": None, "help": "Period growth rate"},
        {"label": "Row count", "value": gw.get("row_count", 0), "delta": None, "help": "Number of records"},
    ]
