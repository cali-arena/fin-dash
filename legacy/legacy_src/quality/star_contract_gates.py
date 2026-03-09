"""
Star contract validation gates: fact grain, dimension keys, join coverage.
No I/O; explicit error messages.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _is_null_or_empty(series: pd.Series) -> pd.Series:
    """True where value is null/NA or (if string-like) empty/whitespace."""
    null = series.isna()
    if series.dtype == "string" or (series.dtype == object and series.notna().any()):
        try:
            stripped = series.astype(str).str.strip()
            empty = stripped == ""
            return null | empty
        except (TypeError, AttributeError):
            pass
    return null


def validate_fact_grain(
    df_fact: pd.DataFrame,
    grain_cols: list[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Enforce fact grain contract: no null/empty in grain cols, uniqueness, rowcount > 0.

    Returns (ok, errors, stats).
    stats: rowcount, duplicate_count, null_counts (dict col -> count).
    """
    errors: list[str] = []
    stats: dict[str, Any] = {
        "rowcount": len(df_fact),
        "duplicate_count": 0,
        "null_counts": {},
    }

    if len(df_fact) == 0:
        errors.append("Fact table has zero rows; rowcount must be > 0.")
        return (False, errors, stats)

    missing = [c for c in grain_cols if c not in df_fact.columns]
    if missing:
        errors.append(f"Fact missing grain columns: {missing}. Required: {grain_cols}.")
        return (False, errors, stats)

    # Null/empty in grain columns
    for col in grain_cols:
        bad = _is_null_or_empty(df_fact[col])
        n = int(bad.sum())
        stats["null_counts"][col] = n
        if n > 0:
            errors.append(f"Grain column {col!r} has {n} null/empty value(s).")

    # Uniqueness of grain
    dup = df_fact[grain_cols].duplicated(keep=False)
    duplicate_count = int(dup.sum())
    stats["duplicate_count"] = duplicate_count
    if duplicate_count > 0:
        errors.append(
            f"Fact grain is not unique: {duplicate_count} row(s) are duplicates on {grain_cols}."
        )

    ok = len(errors) == 0
    return (ok, errors, stats)


def validate_dim(
    df_dim: pd.DataFrame,
    key_cols: list[str],
    required_cols: list[str],
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Enforce dimension contract: required columns present, no null/empty in key cols, uniqueness on keys.

    Returns (ok, errors, stats).
    stats: rowcount, duplicate_count, null_counts (per key col), missing_columns.
    """
    errors: list[str] = []
    stats: dict[str, Any] = {
        "rowcount": len(df_dim),
        "duplicate_count": 0,
        "null_counts": {},
        "missing_columns": [],
    }

    missing = [c for c in required_cols if c not in df_dim.columns]
    if missing:
        stats["missing_columns"] = missing
        errors.append(f"Dimension missing required columns: {missing}. Required: {required_cols}.")
        return (False, errors, stats)

    key_missing = [c for c in key_cols if c not in df_dim.columns]
    if key_missing:
        errors.append(f"Dimension missing key columns: {key_missing}. Key cols: {key_cols}.")
        return (False, errors, stats)

    if len(df_dim) == 0:
        errors.append("Dimension has zero rows.")
        return (False, errors, stats)

    # Null/empty in key columns
    for col in key_cols:
        bad = _is_null_or_empty(df_dim[col])
        n = int(bad.sum())
        stats["null_counts"][col] = n
        if n > 0:
            errors.append(f"Key column {col!r} has {n} null/empty value(s).")

    # Uniqueness of key columns
    dup = df_dim[key_cols].duplicated(keep=False)
    duplicate_count = int(dup.sum())
    stats["duplicate_count"] = duplicate_count
    if duplicate_count > 0:
        errors.append(
            f"Dimension key is not unique: {duplicate_count} row(s) are duplicates on {key_cols}."
        )

    ok = len(errors) == 0
    return (ok, errors, stats)


def validate_star_model(
    df_fact: pd.DataFrame,
    dims: dict[str, pd.DataFrame],
    join_specs: list[dict[str, str]],
    min_coverage_ratio: float = 0.999,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Enforce join coverage: fraction of fact rows with a matching dimension key must be >= min_coverage_ratio.

    join_specs: list of {"dim_name": str, "fact_key": str, "dim_key": str}.
    Example: [
        {"dim_name": "dim_time", "fact_key": "month_end", "dim_key": "month_end"},
        {"dim_name": "dim_geo", "fact_key": "src_country", "dim_key": "country_key"},
    ]

    Returns (ok, errors, stats).
    stats: rowcount (fact), join_coverage per spec (and per dim_name), min_coverage_ratio.
    """
    errors: list[str] = []
    stats: dict[str, Any] = {
        "fact_rowcount": len(df_fact),
        "min_coverage_ratio": min_coverage_ratio,
        "join_coverage": [],
        "join_coverage_by_dim": {},
    }

    if len(df_fact) == 0:
        errors.append("Fact table has zero rows; cannot compute join coverage.")
        return (False, errors, stats)

    for spec in join_specs:
        dim_name = spec.get("dim_name")
        fact_key = spec.get("fact_key")
        dim_key = spec.get("dim_key")
        if not dim_name or not fact_key or not dim_key:
            errors.append(f"Invalid join_spec: each spec must have dim_name, fact_key, dim_key. Got: {spec}.")
            continue
        if dim_name not in dims:
            errors.append(f"Dimension {dim_name!r} not in dims. Available: {list(dims.keys())}.")
            continue
        df_dim = dims[dim_name]
        if fact_key not in df_fact.columns:
            errors.append(f"Fact table missing join column {fact_key!r} for {dim_name}.")
            continue
        if dim_key not in df_dim.columns:
            errors.append(f"Dimension {dim_name!r} missing key column {dim_key!r}.")
            continue

        # Merge fact key onto dim key to get coverage (handles dtype alignment)
        dim_sub = df_dim[[dim_key]].dropna().drop_duplicates()
        merged = df_fact[[fact_key]].merge(
            dim_sub, left_on=fact_key, right_on=dim_key, how="left"
        )
        matched_rows = int(merged[dim_key].notna().sum())
        coverage_ratio = matched_rows / len(df_fact)
        coverage_pct = coverage_ratio * 100.0

        spec_label = f"{dim_name}({fact_key}->{dim_key})"
        stats["join_coverage"].append({
            "spec": spec_label,
            "dim_name": dim_name,
            "fact_key": fact_key,
            "dim_key": dim_key,
            "coverage_ratio": float(coverage_ratio),
            "coverage_pct": round(coverage_pct, 4),
            "matched_rows": matched_rows,
            "total_fact_rows": len(df_fact),
        })

        if dim_name not in stats["join_coverage_by_dim"]:
            stats["join_coverage_by_dim"][dim_name] = []
        stats["join_coverage_by_dim"][dim_name].append({
            "fact_key": fact_key,
            "coverage_ratio": float(coverage_ratio),
            "coverage_pct": round(coverage_pct, 4),
        })

        if coverage_ratio < min_coverage_ratio:
            errors.append(
                f"Join coverage {spec_label}: {coverage_pct:.2f}% (matched {matched_rows}/{len(df_fact)} rows) "
                f"below threshold {min_coverage_ratio:.1%}."
            )

    ok = len(errors) == 0
    return (ok, errors, stats)
