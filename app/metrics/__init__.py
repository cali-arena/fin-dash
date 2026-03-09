"""
Metrics package: KPI definitions and deterministic calculations for the Executive Summary.
"""
from app.metrics.snapshot import (
    FirmMetricsSnapshot,
    build_canonical_metrics_pack,
    get_metrics_debug_info,
    metrics_ready_for_display,
    validate_snapshot,
    validation_required_metrics,
)

__all__ = [
    "FirmMetricsSnapshot",
    "build_canonical_metrics_pack",
    "get_metrics_debug_info",
    "metrics_ready_for_display",
    "validate_snapshot",
    "validation_required_metrics",
]
