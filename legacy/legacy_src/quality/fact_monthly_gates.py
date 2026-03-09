"""
Validation gates for curated fact_monthly before write. Schema-driven; no I/O in validate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_SCHEMA_DTYPES = frozenset([
    "datetime64[ns]",
    "string",
    "float64",
    "int64",
    "category",
    "bool",
])

# Dtype equivalence: schema dtype -> set of accepted pandas dtype strings.
# Int64 (nullable) accepted when schema expects int64. String: strict (string only, not object).
DTYPE_EQUIVALENCES: dict[str, frozenset[str]] = {
    "int64": frozenset({"int64", "Int64"}),
    "float64": frozenset({"float64", "Float64"}),
    "datetime64[ns]": frozenset({"datetime64[ns]", "datetime64[us]", "datetime64[ms]"}),
    "string": frozenset({"string"}),  # strict: do not accept object
    "category": frozenset({"category"}),
    "bool": frozenset({"bool", "boolean"}),
}


def load_fact_schema(schema_path: str | Path) -> dict[str, Any]:
    """
    Load and parse fact_monthly schema YAML. Extract grain, required_canonical_columns, dtypes.
    dtypes values must be one of: datetime64[ns], string, float64, int64, category, bool.
    """
    p = Path(schema_path)
    if not p.is_file():
        raise FileNotFoundError(f"Schema file not found: {p.resolve()}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Schema root must be a YAML mapping, got {type(raw)}")

    grain = raw.get("grain")
    if not isinstance(grain, list) or not all(isinstance(x, str) for x in grain):
        raise ValueError("Schema must have 'grain' as a list of strings")
    required = raw.get("required_canonical_columns")
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
        raise ValueError("Schema must have 'required_canonical_columns' as a list of strings")
    dtypes = raw.get("dtypes")
    if not isinstance(dtypes, dict):
        raise ValueError("Schema must have 'dtypes' as a mapping (dict)")
    for col, dt in dtypes.items():
        if not isinstance(dt, str):
            raise ValueError(f"Schema dtypes[{col!r}] must be a string, got {type(dt)}")
        if dt not in ALLOWED_SCHEMA_DTYPES:
            raise ValueError(
                f"Schema dtypes[{col!r}]={dt!r} is not in allowed set: {sorted(ALLOWED_SCHEMA_DTYPES)}"
            )

    return {
        "grain": list(grain),
        "required_canonical_columns": list(required),
        "dtypes": dict(dtypes),
    }


def _normalize_dtype_for_compare(dtype_str: str) -> str:
    """Return a key for equivalence lookup (e.g. Int64 -> int64 for matching)."""
    s = str(dtype_str).strip().lower()
    if s == "int64" or s == "integer":
        return "int64"
    if s in ("float64", "float"):
        return "float64"
    if s.startswith("datetime64"):
        return "datetime64[ns]"
    if s == "string":
        return "string"
    if s == "category":
        return "category"
    if s in ("bool", "boolean"):
        return "bool"
    return s


def _dtype_matches(schema_dtype: str, df_dtype_str: str) -> bool:
    """True if df column dtype satisfies schema dtype (with allowed equivalences)."""
    allowed = DTYPE_EQUIVALENCES.get(schema_dtype)
    if allowed is not None:
        return df_dtype_str in allowed or _normalize_dtype_for_compare(df_dtype_str) in {
            _normalize_dtype_for_compare(x) for x in allowed
        }
    return schema_dtype == df_dtype_str


def validate_fact_monthly(
    df_fact: pd.DataFrame,
    schema: dict[str, Any],
    *,
    strict_dtypes: bool = True,
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Validate curated fact_monthly DataFrame against loaded schema. No I/O.

    Gates:
      1) Required columns: hard fail if any required_canonical_columns missing.
      2) Dtypes: for each col in schema.dtypes, compare normalized dtype; strict_dtypes=True
         means string schema expects pandas StringDtype (string), not object; Int64 is allowed
         when schema expects int64 (nullable integer).
      3) No nulls in grain: null = isna() OR empty string after strip for string-like cols.
      4) Uniqueness of grain: duplicates reported with count and first 10 sample keys.

    Returns (ok, errors, stats). Errors are explicit and actionable.
    """
    errors: list[str] = []
    stats: dict[str, Any] = {}

    grain = schema.get("grain") or []
    required = schema.get("required_canonical_columns") or []
    dtypes_schema = schema.get("dtypes") or {}

    # --- Stats: rowcount and dtype_signature (ordered by col name) ---
    stats["rowcount"] = len(df_fact)
    cols_sorted = sorted(df_fact.columns, key=str)
    stats["dtype_signature"] = {str(c): str(df_fact[c].dtype) for c in cols_sorted}

    # --- Gate 1: Required columns present ---
    missing = [c for c in required if c not in df_fact.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}. Required: {required}")

    # --- Gate 2: Dtypes match (only for columns present in both) ---
    for col in dtypes_schema:
        if col not in df_fact.columns:
            continue
        expected = dtypes_schema[col]
        actual = str(df_fact[col].dtype)
        if strict_dtypes and expected == "string" and actual == "object":
            errors.append(
                f"Column {col!r}: dtype is object but schema requires string (use pandas StringDtype, strict_dtypes=True)."
            )
        elif not strict_dtypes and expected == "string" and actual == "object":
            # Allow object when schema says string if strict_dtypes=False (documented).
            pass
        elif not _dtype_matches(expected, actual):
            errors.append(f"Column {col!r}: dtype {actual!r} does not match schema {expected!r}.")

    # --- Gate 3: No nulls in grain ---
    null_counts_grain: dict[str, int] = {}
    for g in grain:
        if g not in df_fact.columns:
            continue
        ser = df_fact[g]
        is_na = pd.isna(ser)
        if pd.api.types.is_string_dtype(ser) or ser.dtype == object:
            stripped = ser.astype(str).str.strip()
            empty = stripped.eq("") | stripped.str.lower().eq("nan")
            invalid = is_na | empty
        else:
            invalid = is_na
        n = int(invalid.sum())
        null_counts_grain[g] = n
        if n > 0:
            errors.append(f"Grain column {g!r} has {n} null or empty value(s); no nulls allowed.")
    stats["null_counts_grain"] = null_counts_grain

    # --- Gate 4: Uniqueness of grain ---
    grain_present = [g for g in grain if g in df_fact.columns]
    if grain_present:
        dup = df_fact.duplicated(subset=grain_present, keep=False)
        duplicate_count = int(dup.sum())
        stats["duplicate_count"] = duplicate_count
        if duplicate_count > 0:
            dup_rows = df_fact.loc[dup]
            sample = dup_rows[grain_present].drop_duplicates().head(10)
            sample_keys = [tuple(row) for _, row in sample.iterrows()]
            errors.append(
                f"Grain is not unique: {duplicate_count} row(s) are duplicates. Sample keys (first 10): {sample_keys}"
            )
    else:
        stats["duplicate_count"] = 0

    ok = len(errors) == 0
    return ok, errors, stats
