"""Metrics pipeline: compute metrics by drill path; one canonical metrics_monthly table."""
from legacy.legacy_pipelines.metrics.compute_metrics import (
    build_metrics_monthly,
    validate_metrics_slices_contract,
    write_metrics_slice_coverage_qa,
)
from legacy.legacy_pipelines.metrics.rate_policies import (
    safe_divide,
    apply_begin_aum_guard,
    apply_fee_yield_guard,
    apply_clamp,
    coerce_inf_to_nan,
)

__all__ = [
    "build_metrics_monthly",
    "validate_metrics_slices_contract",
    "write_metrics_slice_coverage_qa",
    "safe_divide",
    "apply_begin_aum_guard",
    "apply_fee_yield_guard",
    "apply_clamp",
    "coerce_inf_to_nan",
]
