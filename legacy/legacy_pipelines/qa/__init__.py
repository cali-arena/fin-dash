"""QA and fail-fast gate evaluation (Step 4)."""

from legacy.legacy_pipelines.qa.validate_vs_data_summary import (
    evaluate_fail_fast,
    format_failure_message,
    format_pass_message,
    write_fail_examples,
)

__all__ = [
    "evaluate_fail_fast",
    "format_failure_message",
    "format_pass_message",
    "write_fail_examples",
]
