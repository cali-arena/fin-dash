"""
Canonical scope and period contracts for KPI computation.
Single source of truth for firm-wide vs selected slice and for 1M / QoQ / YTD / YoY.
"""
from __future__ import annotations

# Scope: used for labeling and for selecting data source (firm vs slice).
SCOPE_FIRM = "firm"
SCOPE_SLICE = "slice"

# Period: applied to monthly data to get the window; then "latest" = last row in window.
PERIOD_1M = "1M"
PERIOD_QOQ = "QoQ"
PERIOD_YTD = "YTD"
PERIOD_YOY = "YoY"

PERIOD_ALLOWED = (PERIOD_1M, PERIOD_QOQ, PERIOD_YTD, PERIOD_YOY)
SCOPE_ALLOWED = (SCOPE_FIRM, SCOPE_SLICE)
