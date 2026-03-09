"""
Canonicalization: rename incoming Excel/CSV columns to canonical names (snake_case).
Uses schemas/canonical_columns.yml. Unmatched headers are kept; reported in canonicalization_report.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.schemas.canonical_resolver import (
    load_canonical_columns,
    required_canonicals,
    resolve_headers_to_canonical,
)


def canonicalize_dataframe(
    df: pd.DataFrame,
    canonical_schema_path: str | Path = "schemas/canonical_columns.yml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Rename df columns to canonical names via canonical_columns.yml. Keep unmatched headers.
    Validates required canonicals present; raises ValueError if missing or duplicate source mappings.
    Returns (df_canonical, canonicalization_report).
    """
    canonical_schema = load_canonical_columns(canonical_schema_path)
    mapping_original_to_canonical, unmatched_headers = resolve_headers_to_canonical(
        list(df.columns), canonical_schema
    )

    # Detect duplicate source mappings: two or more originals -> same canonical
    canonical_to_originals: dict[str, list[str]] = {}
    for orig, canon in mapping_original_to_canonical.items():
        canonical_to_originals.setdefault(canon, []).append(orig)
    duplicates = {c: origs for c, origs in canonical_to_originals.items() if len(origs) > 1}
    if duplicates:
        parts = [f"{c!r} <- {origs!r}" for c, origs in duplicates.items()]
        raise ValueError(
            "Duplicate source mappings: multiple source headers map to the same canonical column. "
            "Resolve duplicates in source or schema. " + "; ".join(parts)
        )

    # Build rename map (only resolved)
    rename_map = {orig: canon for orig, canon in mapping_original_to_canonical.items()}
    df_canonical = df.rename(columns=rename_map)

    # Validate required canonical columns present
    required = required_canonicals(canonical_schema)
    resolved_canonicals = set(mapping_original_to_canonical.values())
    missing_required = [c for c in required if c not in resolved_canonicals]
    if missing_required:
        available = sorted(resolved_canonicals) + sorted(unmatched_headers)
        raise ValueError(
            f"Missing required canonical columns: {missing_required}. "
            f"Available after resolve: {available}."
        )

    # Report: resolved canonical_name -> original_header (1:1 after duplicate check)
    resolved: dict[str, str] = {canon: orig for orig, canon in mapping_original_to_canonical.items()}

    report: dict[str, Any] = {
        "resolved": resolved,
        "missing_required": missing_required,
        "unmatched_headers": unmatched_headers,
    }
    return df_canonical, report


def type_enforcement_params_from_canonical(canonical_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Derive date_col, currency_cols, id_cols from canonical schema by dtype.
    For use after canonicalize when type enforcement runs on canonical columns only.
    """
    date_col: str | None = None
    currency_cols: list[str] = []
    id_cols: list[str] = []
    for col in canonical_schema.get("columns") or []:
        if not isinstance(col, dict) or not col.get("canonical_name"):
            continue
        name = col["canonical_name"]
        dtype = (col.get("dtype") or "string").lower()
        if dtype == "datetime":
            date_col = name
        elif dtype == "float64":
            currency_cols.append(name)
        elif dtype == "string":
            id_cols.append(name)
    return {
        "date_col": date_col or "month_end",
        "currency_cols": currency_cols,
        "id_cols": id_cols,
    }
