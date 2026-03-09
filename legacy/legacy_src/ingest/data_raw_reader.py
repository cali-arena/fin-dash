"""
Contract-first typed CSV reader for DATA_RAW using schemas/data_raw.schema.yml.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.schemas.loader import load_yaml_schema, validate_schema_shape
from legacy.legacy_src.validators.schema_validator import (
    format_validation_errors,
    validate_dataframe_against_schema,
)

logger = logging.getLogger(__name__)


def _build_column_maps(schema: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, bool]]:
    """(name -> canonical_name), (canonical_name -> name), (canonical_name -> dtype), (canonical_name -> nullable)."""
    name_to_canon: dict[str, str] = {}
    canon_to_dtype: dict[str, str] = {}
    canon_to_nullable: dict[str, bool] = {}
    for col in (schema.get("required_columns") or []) + (schema.get("optional_columns") or []):
        if not isinstance(col, dict) or "name" not in col:
            continue
        name = col["name"]
        canon = col.get("canonical_name", name)
        name_to_canon[name] = canon
        canon_to_dtype[canon] = col.get("dtype", "string")
        canon_to_nullable[canon] = col.get("nullable", False)
    canon_to_name = {v: k for k, v in name_to_canon.items()}
    return name_to_canon, canon_to_name, canon_to_dtype, canon_to_nullable


def _parse_date_month_end(series: pd.Series) -> pd.Series:
    """Parse as datetime and convert to month-end (same month)."""
    dt = pd.to_datetime(series, errors="coerce")
    return dt + pd.offsets.MonthEnd(0)


def _parse_currency_series(series: pd.Series, rules: dict[str, Any]) -> pd.Series:
    """Strip symbols, remove commas, parentheses -> negative, cast to float64."""
    s = series.astype(str).str.strip()
    if rules.get("strip_currency_symbols", True):
        s = s.str.replace("R$", "", regex=False).str.replace(r"[\$£€¥]", "", regex=True).str.strip()
    if rules.get("allow_commas", True):
        s = s.str.replace(",", "", regex=False)
    if rules.get("allow_parentheses_negative", True):
        # (123.45) -> -123.45
        s = s.str.replace(r"^\(([^)]*)\)\s*$", r"-\1", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _parse_string_series(series: pd.Series, rules: dict[str, Any]) -> pd.Series:
    """Trim and normalize whitespace."""
    s = series.astype(str)
    if rules.get("trim", True):
        s = s.str.strip()
    if rules.get("normalize_whitespace", True):
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _enforce_dtype(series: pd.Series, dtype: str, nullable: bool) -> pd.Series:
    """Enforce schema dtype on series."""
    if dtype == "datetime":
        return series  # already datetime64[ns] from parsing
    if dtype == "float64":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if dtype == "int64":
        num = pd.to_numeric(series, errors="coerce")
        if nullable:
            return num.astype("Int64")
        return num.fillna(0).astype("int64")
    if dtype == "category":
        return series.astype("category")
    if dtype == "string":
        return series.astype(pd.StringDtype())
    if dtype == "bool":
        return series.astype(bool)
    return series.astype(pd.StringDtype())


def apply_data_raw_schema(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """
    Apply schema parsing and dtypes to a DataFrame whose columns are contract names (schema required_columns[].name).
    Returns DataFrame with canonical column names and enforced dtypes. Skips columns not present in df.
    """
    name_to_canon, canon_to_name, canon_to_dtype, canon_to_nullable = _build_column_maps(schema)
    currency_canon = set(schema.get("currency_columns") or [])
    date_rule = (schema.get("parsing_rules") or {}).get("date") or {}
    date_source = date_rule.get("source_column", "Date")
    curr_rule = (schema.get("parsing_rules") or {}).get("currency") or {}
    str_rule = (schema.get("parsing_rules") or {}).get("strings") or {}

    out: dict[str, pd.Series] = {}
    for canon, source in canon_to_name.items():
        if source not in df.columns:
            continue
        ser = df[source].copy()
        dtype = canon_to_dtype.get(canon, "string")

        if date_source == source and date_rule.get("rule") == "month_end":
            ser = _parse_date_month_end(ser)
        elif canon in currency_canon:
            ser = _parse_currency_series(ser, curr_rule)
        elif dtype in ("string", "category"):
            ser = _parse_string_series(ser, str_rule)
        elif dtype == "float64":
            ser = _parse_currency_series(ser, curr_rule)
        else:
            ser = _parse_string_series(ser, str_rule)

        ser = _enforce_dtype(ser, dtype, canon_to_nullable.get(canon, False))
        out[canon] = ser

    result = pd.DataFrame(out)
    result.attrs["source_columns"] = name_to_canon
    return result


def read_data_raw(
    csv_path: str | Path = "data/input/DATA_RAW.csv",
    schema_path: str | Path = "schemas/data_raw.schema.yml",
) -> pd.DataFrame:
    """
    Load DATA_RAW CSV with schema-driven parsing and validation.
    Returns DataFrame with canonical column names and enforced dtypes.
    Hard fails on missing required columns; logs warnings for extra columns.
    """
    csv_path = Path(csv_path)
    schema_path = Path(schema_path)
    context = str(csv_path)

    schema = load_yaml_schema(schema_path)
    validate_schema_shape(schema)

    df = pd.read_csv(csv_path, dtype=str)

    ok, errors, warnings = validate_dataframe_against_schema(df, schema, context=context)
    if not ok:
        raise ValueError(format_validation_errors(errors, warnings))
    for w in warnings:
        logger.warning(w)

    result = apply_data_raw_schema(df, schema)

    allowed = set(_build_column_maps(schema)[0].keys())
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        for c in extra:
            result[c] = df[c]
        logger.warning("Extra columns (unmapped): %s", extra)

    return result
