from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest
from openpyxl import Workbook

# Ensure project root (with `src/`) is on sys.path when running tests directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.ingest.excel_reader import load_data_raw_excel, _get_expected_columns_contract
from legacy.legacy_src.schemas.loader import load_yaml_schema, validate_schema_shape


def _build_test_workbook(path: Path, headers: list[str], rows: list[dict[str, str]]) -> Path:
    """
    Create a small Excel file with sheet 'DATA RAW', the provided headers, and row dicts.
    Values are written as literals so pandas with dtype=str must preserve them.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "DATA RAW"

    # Header row
    ws.append(headers)

    # Data rows – map by header to keep order deterministic
    for row in rows:
        ws.append([row.get(h, "") for h in headers])

    wb.save(path)
    return path


def _load_contract_columns(schema_path: Path) -> list[str]:
    schema = load_yaml_schema(schema_path)
    validate_schema_shape(schema)
    return _get_expected_columns_contract(schema)


def test_excel_reader_strings_extra_and_rowcount(tmp_path: Path) -> None:
    """
    Excel is read as strings, extra columns reported, and row counts match.
    """
    schema_path = Path("schemas") / "data_raw.schema.yml"
    required_cols = _load_contract_columns(schema_path)

    # Add one extra column that is not part of the contract
    headers = required_cols + ["EXTRA_COL"]

    # Two rows with string values that must not be coerced
    row1 = {
        "Date": "2021-01-15",
        " Asset Under Management ": "(1,234.50)",
        "Net new business": "00012",
    }
    row2 = {
        "Date": "2021-01-16",
        " Asset Under Management ": "(2,345.60)",
        "Net new business": "00034",
    }

    xlsx_path = tmp_path / "data_raw_test.xlsx"
    _build_test_workbook(xlsx_path, headers=headers, rows=[row1, row2])

    df, report = load_data_raw_excel(
        xlsx_path=xlsx_path,
        sheet_name="DATA RAW",
        schema_path=schema_path,
        hard_fail_missing=True,
        allow_extra_columns=True,
    )

    # 1) All dtypes are object or pandas string, and raw values preserved as strings
    for dtype in df.dtypes:
        assert pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
    assert df.loc[0, "Net new business"] == "00012"

    # 2) Row count matches
    assert report.rows_read == 2

    # 3) Extra column is correctly reported in the LoadReport
    assert "EXTRA_COL" in report.extra_columns


def test_excel_reader_missing_required_column_hard_fails(tmp_path: Path) -> None:
    """
    Removing a required column causes a ValueError that includes the missing name and LoadReport.pretty().
    """
    schema_path = Path("schemas") / "data_raw.schema.yml"
    required_cols = _load_contract_columns(schema_path)

    assert "Net new business" in required_cols  # sanity check for the contract

    # Drop one required column from the header set
    headers = [c for c in required_cols if c != "Net new business"]

    row = {
        "Date": "2021-01-15",
        " Asset Under Management ": "(1,234.50)",
        # intentionally omit "Net new business"
    }

    xlsx_path = tmp_path / "data_raw_missing_col.xlsx"
    _build_test_workbook(xlsx_path, headers=headers, rows=[row])

    with pytest.raises(ValueError) as excinfo:
        load_data_raw_excel(
            xlsx_path=xlsx_path,
            sheet_name="DATA RAW",
            schema_path=schema_path,
            hard_fail_missing=True,
            allow_extra_columns=True,
        )

    msg = str(excinfo.value)
    # 4) Error message includes the missing column name and pretty LoadReport output
    assert "Net new business" in msg
    assert "LoadReport" in msg

