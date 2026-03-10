"""
Governed KPI pipeline: single canonical computation for top-level metrics.
All top-level KPIs (End AUM, NNB, NNF, OGR, Market Movement) must be computed via this module.
"""
from __future__ import annotations

from app.kpi.contract import (
    PERIOD_1M,
    PERIOD_QOQ,
    PERIOD_YTD,
    PERIOD_YOY,
    SCOPE_FIRM,
    SCOPE_SLICE,
)
from app.kpi.service import KPIResult, compute_kpi

__all__ = [
    "compute_kpi",
    "KPIResult",
    "SCOPE_FIRM",
    "SCOPE_SLICE",
    "PERIOD_1M",
    "PERIOD_QOQ",
    "PERIOD_YTD",
    "PERIOD_YOY",
]
