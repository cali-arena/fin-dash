"""
Pre-flight input contract validation for the finance dashboard.
Validates presence of required CSV files and required columns (when specified).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# Required CSV filenames (exact). Source: docs/data_contract.md
REQUIRED_FILES = [
    "DATA_RAW.csv",
    "DATA_SUMMARY.csv",
    "DATA_MAPPING.csv",
    "ETF.csv",
    "EXECUTIVE_SUMMARY.csv",
]

# Required columns per file. From docs/data_contract.md; empty = validate file existence only.
# DATA_RAW.csv: explicit required columns (exact names including leading/trailing spaces).
DATA_RAW_REQUIRED = [
    "Date",
    "Channel",
    "src_country",
    " Asset Under Management ",
    "Net new business",
    " net new base fees ",
    "display_firm",
    "product_country",
    "Standard Channel",
    " best_of_source ",
    "product_ticker",
    "Segment",
    "sub_segment",
    "uswa_sales_focus_2020",
    "master_custodian_firm",
]

# DATA_SUMMARY.csv: named required columns; first column is TBD (unnamed) so not validated by name.
DATA_SUMMARY_REQUIRED = [
    "Asset growth Rate",
    "Organic growth rate",
    "External growth rate",
    "Velocity",
]

# DATA_MAPPING.csv: TBD confirm columns — validate file existence only.
DATA_MAPPING_REQUIRED: list[str] = []  # TODO: confirm column names in data_contract.md

# ETF.csv: minimum required set from contract; remaining columns omitted for brevity.
ETF_REQUIRED = [
    "Ticker",
    "Name",
    "Incept. Date",
    "Net Assets (USD)",
    "Net Assets as of",
    "Asset Class",
    "Sub Asset Class",
    "Region",
    "Market",
    "Location",
    "Investment Style",
]

# EXECUTIVE_SUMMARY.csv: TBD confirm header/columns — validate file existence only.
EXECUTIVE_SUMMARY_REQUIRED: list[str] = []  # TODO: confirm column names in data_contract.md

REQUIRED_COLUMNS_MAP = {
    "DATA_RAW.csv": DATA_RAW_REQUIRED,
    "DATA_SUMMARY.csv": DATA_SUMMARY_REQUIRED,
    "DATA_MAPPING.csv": DATA_MAPPING_REQUIRED,
    "ETF.csv": ETF_REQUIRED,
    "EXECUTIVE_SUMMARY.csv": EXECUTIVE_SUMMARY_REQUIRED,
}

# Optional columns per file (empty = none documented).
OPTIONAL_COLUMNS_MAP: dict[str, list[str]] = {
    "DATA_RAW.csv": [],
    "DATA_SUMMARY.csv": ["Standard Deviation", "VAR", "Inflation", "Interest rates", "Currency impact"],
    "DATA_MAPPING.csv": [],
    "ETF.csv": ["SEDOL", "ISIN", "CUSIP", "Gross Expense Ratio (%)", "Net Expense Ratio (%)", "Key Facts"],
    "EXECUTIVE_SUMMARY.csv": [],
}


def load_contract_spec() -> dict[str, Any]:
    """
    Return the input contract spec as a Python dict.
    Keys: required_files, required_columns, optional_columns.
    """
    return {
        "required_files": list(REQUIRED_FILES),
        "required_columns": dict(REQUIRED_COLUMNS_MAP),
        "optional_columns": dict(OPTIONAL_COLUMNS_MAP),
    }


def _get_csv_headers(csv_path: Path) -> list[str] | None:
    """Read CSV header only (nrows=0). Returns list of column names or None on error."""
    try:
        df = pd.read_csv(csv_path, nrows=0)
        return list(df.columns)
    except Exception:
        return None


def validate_inputs(base_dir: str) -> tuple[bool, list[str]]:
    """
    Check that all required files exist and have required columns.
    Uses pandas with nrows=0 for header-only read.
    Returns (ok, errors). ok is True only when there are no errors.
    """
    base = Path(base_dir)
    errors: list[str] = []

    spec = load_contract_spec()
    required_files = spec["required_files"]
    required_columns = spec["required_columns"]

    for filename in required_files:
        path = base / filename
        if not path.is_file():
            errors.append(f"{filename}: file missing")
            continue

        required = required_columns.get(filename, [])
        if not required:
            continue

        headers = _get_csv_headers(path)
        if headers is None:
            errors.append(f"{filename}: could not read CSV header")
            continue

        missing = [c for c in required if c not in headers]
        if missing:
            errors.append(f"{filename}: missing required columns: {missing}")

    ok = len(errors) == 0
    return ok, errors


def format_errors(errors: list[str]) -> str:
    """Turn a list of error strings into a single human-readable message."""
    if not errors:
        return ""
    return "\n".join(f"  • {e}" for e in errors)
