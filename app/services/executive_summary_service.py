"""
Executive Summary payload used by the Streamlit dashboard KPI strip.
Metrics originate from the governed dataset agg/firm_monthly and follow the KPI
contract defined in kpi_definitions.py. Formatting is done only in this service
layer; KPI calculations remain unformatted.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from app.data_gateway import get_executive_kpis


def format_currency(value: float | None) -> str:
    """Format a numeric value as currency for display. Handles None/NaN; formatting only, no logic."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def format_percent(value: float | None) -> str:
    """Format a decimal rate as percentage for display (e.g. 0.05 → 5.00%). Handles None/NaN; formatting only."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return f"{float(value) * 100:,.2f}%"
    except (TypeError, ValueError):
        return "—"


def get_executive_summary_payload(root: Path | None = None) -> dict[str, float]:
    """
    Return the Executive Summary KPI payload for the Streamlit KPI strip.
    Uses governed get_executive_kpis(); converts result to a dictionary of floats.
    Use format_currency / format_percent when rendering; do not format inside KPI calculations.
    """
    kpis = get_executive_kpis(root)
    return {
        "end_aum": float(kpis.end_aum) if not _is_nan(kpis.end_aum) else float("nan"),
        "mom_growth": float(kpis.mom_growth) if not _is_nan(kpis.mom_growth) else float("nan"),
        "ytd_growth": float(kpis.ytd_growth) if not _is_nan(kpis.ytd_growth) else float("nan"),
        "nnb": float(kpis.nnb) if not _is_nan(kpis.nnb) else float("nan"),
        "nnf": float(kpis.nnf) if not _is_nan(kpis.nnf) else float("nan"),
        "ogr": float(kpis.ogr) if not _is_nan(kpis.ogr) else float("nan"),
        "market_impact": float(kpis.market_impact) if not _is_nan(kpis.market_impact) else float("nan"),
        "market_pnl": float(kpis.market_pnl) if not _is_nan(kpis.market_pnl) else float("nan"),
    }


def _is_nan(x: Any) -> bool:
    if x is None:
        return True
    try:
        return isinstance(x, float) and math.isnan(x)
    except (TypeError, ValueError):
        return True
