"""
Robust Excel reader for DATA RAW: load as strings only, no silent coercion.
Uses schema for contract columns; reports missing/extra with original names.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.ingest.load_report import LoadReport
from legacy.legacy_src.schemas.loader import load_yaml_schema, validate_schema_shape

logger = logging.getLogger(__name__)


def normalize_header(h: str, *, match_case_insensitive: bool = True) -> str:
    """
    Strip leading/trailing whitespace; collapse multiple spaces/tabs/newlines to single space.
    If match_case_insensitive, return lowercased result for matching.
    """
    s = str(h).strip()
    s = re.sub(r"[ \t\n\r]+", " ", s)
    if match_case_insensitive:
        s = s.lower()
    return s


def build_header_maps(
    headers: list[str],
    *,
    match_case_insensitive: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns (original -> normalized, normalized -> original).
    Raises ValueError if two originals normalize to the same key (collision).
    """
    original_to_normalized: dict[str, str] = {}
    normalized_to_original: dict[str, str] = {}
    for orig in headers:
        norm = normalize_header(orig, match_case_insensitive=match_case_insensitive)
        original_to_normalized[orig] = norm
        if norm in normalized_to_original and normalized_to_original[norm] != orig:
            raise ValueError(
                f"Header collision: two columns normalize to the same key: "
                f"{normalized_to_original[norm]!r} and {orig!r} -> {norm!r}"
            )
        normalized_to_original[norm] = orig
    return original_to_normalized, normalized_to_original


def match_contract_columns(
    contract_cols: list[str],
    normalized_to_original: dict[str, str],
    *,
    match_case_insensitive: bool = True,
) -> tuple[list[str], list[str]]:
    """
    For each contract col: normalize; if in normalized_to_original pick that original, else add to missing (contract name).
    Returns (selected_original_headers, missing_contract_columns). Deterministic order.
    """
    selected: list[str] = []
    missing: list[str] = []
    for c in contract_cols:
        norm = normalize_header(c, match_case_insensitive=match_case_insensitive)
        if norm in normalized_to_original:
            selected.append(normalized_to_original[norm])
        else:
            missing.append(c)
    return selected, missing


def _get_expected_columns_contract(schema: dict[str, Any]) -> list[str]:
    """Contract column names in order (required then optional)."""
    out: list[str] = []
    for col in (schema.get("required_columns") or []) + (schema.get("optional_columns") or []):
        if isinstance(col, dict) and "name" in col:
            out.append(col["name"])
    return out


def load_data_raw_excel(
    xlsx_path: str | Path = "data/input/source.xlsx",
    sheet_name: str = "DATA RAW",
    schema_path: str | Path = "schemas/data_raw.schema.yml",
    *,
    match_case_insensitive: bool = True,
    hard_fail_missing: bool = True,
    allow_extra_columns: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Load DATA RAW sheet as all-string DataFrame. No dtype inference.
    Returns (df_raw_strings, LoadReport).
    If hard_fail_missing and missing_columns non-empty, raises ValueError with report.pretty().
    If extra columns and not allow_extra_columns, raises ValueError. If allow_extra_columns, logs warning.
    """
    xlsx_path = Path(xlsx_path)
    schema_path = Path(schema_path)

    schema = load_yaml_schema(schema_path)
    validate_schema_shape(schema)
    expected_columns_contract = _get_expected_columns_contract(schema)

    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path.resolve()}")

    # Read sheet: all values as string (no silent coercion)
    df = pd.read_excel(
        xlsx_path,
        sheet_name=sheet_name,
        header=0,
        engine="openpyxl",
        dtype=str,
    )
    df = df.fillna("").astype(str)

    rows_read = len(df)
    columns_detected_original = list(df.columns)

    header_original_to_normalized, header_normalized_to_original = build_header_maps(
        columns_detected_original,
        match_case_insensitive=match_case_insensitive,
    )
    columns_detected_normalized = [header_original_to_normalized[c] for c in columns_detected_original]

    selected_columns_used, missing_columns = match_contract_columns(
        expected_columns_contract,
        header_normalized_to_original,
        match_case_insensitive=match_case_insensitive,
    )
    selected_set = set(selected_columns_used)
    extra_columns = [h for h in columns_detected_original if h not in selected_set]

    report = LoadReport(
        xlsx_path=str(xlsx_path.resolve()),
        sheet_name=sheet_name,
        rows_read=rows_read,
        columns_detected_original=columns_detected_original,
        columns_detected_normalized=columns_detected_normalized,
        expected_columns_contract=expected_columns_contract,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        selected_columns_used=selected_columns_used,
        header_mapping_original_to_normalized=header_original_to_normalized,
        header_mapping_normalized_to_original=header_normalized_to_original,
    )

    if hard_fail_missing and missing_columns:
        msg = f"Input contract validation failed for sheet {sheet_name}\n{report.pretty()}"
        raise ValueError(msg)

    if extra_columns and not allow_extra_columns:
        msg = f"Extra columns not allowed for sheet {sheet_name}\n{report.pretty()}"
        raise ValueError(msg)

    if extra_columns and allow_extra_columns:
        logger.warning("Extra columns in sheet %s: %s", sheet_name, extra_columns)

    df_raw_strings = df[selected_columns_used].copy() if selected_columns_used else pd.DataFrame()
    df_raw_strings = df_raw_strings.fillna("").astype(str)

    return df_raw_strings, report
