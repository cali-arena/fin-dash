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
from app.metrics.qa_guardrails import (
    check_aum_reconciliation,
    check_fee_yield_consistency,
    check_nnb_nnf_magnitude_ratio,
    run_metric_qa,
)

__all__ = [
    "FirmMetricsSnapshot",
    "build_canonical_metrics_pack",
    "get_metrics_debug_info",
    "metrics_ready_for_display",
    "validate_snapshot",
    "validation_required_metrics",
    "check_aum_reconciliation",
    "check_fee_yield_consistency",
    "check_nnb_nnf_magnitude_ratio",
    "run_metric_qa",
]
