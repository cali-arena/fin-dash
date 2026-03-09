"""
Validate a DataFrame against a YAML schema. Column presence only (no heavy transforms).
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _required_column_names(schema: dict[str, Any]) -> set[str]:
    """Exact column names from schema required_columns."""
    names = set()
    for col in schema.get("required_columns") or []:
        if isinstance(col, dict) and "name" in col:
            names.add(col["name"])
    return names


def _allowed_column_names(schema: dict[str, Any]) -> set[str]:
    """Required + optional column names (exact)."""
    names = set(_required_column_names(schema))
    for col in schema.get("optional_columns") or []:
        if isinstance(col, dict) and "name" in col:
            names.add(col["name"])
    return names


def validate_dataframe_against_schema(
    df: pd.DataFrame,
    schema: dict[str, Any],
    *,
    context: str = "",
) -> tuple[bool, list[str], list[str]]:
    """
    Returns (ok, errors, warnings).
    - Hard fail if any required column missing (exact name match).
    - Extra columns: errors if allow_extra_columns false; else warnings if log_extra_columns true.
    """
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"[{context}] " if context else ""

    required = _required_column_names(schema)
    allowed = _allowed_column_names(schema)
    actual = set(df.columns)

    missing = required - actual
    if missing:
        errors.append(f"{prefix}Missing required columns: {sorted(missing)}")

    vr = schema.get("validation_rules") or {}
    allow_extra = vr.get("allow_extra_columns", True)
    log_extra = vr.get("log_extra_columns", False)
    extra = actual - allowed

    if extra:
        extra_sorted = sorted(extra)
        if not allow_extra:
            errors.append(f"{prefix}Extra columns not allowed: {extra_sorted}")
        elif log_extra:
            warnings.append(f"{prefix}Extra columns (allowed): {extra_sorted}")

    ok = len(errors) == 0
    return ok, errors, warnings


def format_validation_errors(errors: list[str], warnings: list[str]) -> str:
    """Single human-readable message from errors and warnings."""
    lines = []
    if errors:
        lines.append("Errors:")
        for e in errors:
            lines.append(f"  • {e}")
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  • {w}")
    return "\n".join(lines) if lines else ""
