"""
Quality gate for unmapped channel key combinations. Warn or fail based on thresholds.
"""
from __future__ import annotations

from typing import Any


def gate_unmapped(
    *,
    total_rows: int,
    unmapped_rows: int,
    unmapped_keys: int,
    fail_above_ratio: float = 0.01,
    fail_above_keys: int | None = None,
    mode: str = "warn",
) -> tuple[bool, str, dict[str, Any]]:
    """
    Evaluate unmapped channel metrics against thresholds.

    - ratio = unmapped_rows / max(total_rows, 1)
    - message: "Unmapped channel keys: {unmapped_keys}, rows: {unmapped_rows} ({ratio:.2%})"
      (in warn mode prefixed with "WARNING: ")
    - mode "warn": ok=True always
    - mode "fail": ok=False if ratio > fail_above_ratio or
      (fail_above_keys is not None and unmapped_keys > fail_above_keys)

    Returns (ok, message, stats).
    """
    ratio = unmapped_rows / max(total_rows, 1)
    base_message = (
        f"Unmapped channel keys: {unmapped_keys}, rows: {unmapped_rows} ({ratio:.2%})"
    )

    stats: dict[str, Any] = {
        "total_rows": total_rows,
        "unmapped_rows": unmapped_rows,
        "unmapped_keys": unmapped_keys,
        "ratio": ratio,
        "mode": mode,
        "fail_above_ratio": fail_above_ratio,
        "fail_above_keys": fail_above_keys,
    }

    if mode == "warn":
        message = f"WARNING: {base_message}"
        return (True, message, stats)

    if mode == "fail":
        fail_ratio = ratio > fail_above_ratio
        fail_keys = (
            fail_above_keys is not None and unmapped_keys > fail_above_keys
        )
        ok = not (fail_ratio or fail_keys)
        message = base_message
        return (ok, message, stats)

    raise ValueError(f"gate_unmapped: mode must be 'warn' or 'fail', got {mode!r}")
