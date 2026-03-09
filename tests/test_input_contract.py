"""
Pytest tests for input contract validation.
Uses a temp directory with small CSV fixtures.
"""
from pathlib import Path

import pytest

from app.validators.input_contract import (
    format_errors,
    load_contract_spec,
    validate_inputs,
)


def _csv_line(values: list[str]) -> str:
    """Escape CSV cells (quote if contains comma or newline)."""
    out = []
    for v in values:
        if "," in v or "\n" in v or '"' in v:
            out.append('"' + v.replace('"', '""') + '"')
        else:
            out.append(v)
    return ",".join(out)


@pytest.fixture
def valid_input_dir(tmp_path: Path) -> Path:
    """Create a directory with all required CSVs and required columns (headers only)."""
    spec = load_contract_spec()
    required_files = spec["required_files"]
    required_columns = spec["required_columns"]

    for filename in required_files:
        path = tmp_path / filename
        cols = required_columns.get(filename, [])
        if cols:
            header = _csv_line(cols)
            dummy_row = _csv_line(["x"] * len(cols))
        else:
            header = "col1,col2"
            dummy_row = "a,b"
        path.write_text(header + "\n" + dummy_row + "\n", encoding="utf-8")

    return tmp_path


def test_load_contract_spec() -> None:
    spec = load_contract_spec()
    assert "required_files" in spec
    assert "required_columns" in spec
    assert "optional_columns" in spec
    assert "DATA_RAW.csv" in spec["required_files"]
    assert "DATA_RAW.csv" in spec["required_columns"]


def test_validate_inputs_missing_file(tmp_path: Path) -> None:
    # Only create one required file; rest missing
    (tmp_path / "DATA_RAW.csv").write_text("Date,Channel\n2020-01-01,BD", encoding="utf-8")
    ok, errors = validate_inputs(str(tmp_path))
    assert ok is False
    assert len(errors) >= 1
    # At least one error should be about a missing file
    assert any("file missing" in e for e in errors)


def test_validate_inputs_missing_column(tmp_path: Path, valid_input_dir: Path) -> None:
    # Overwrite DATA_RAW with header missing a required column (e.g. Date)
    bad_header = "Channel,src_country\nBD,US"
    (valid_input_dir / "DATA_RAW.csv").write_text(bad_header, encoding="utf-8")
    ok, errors = validate_inputs(str(valid_input_dir))
    assert ok is False
    assert any("DATA_RAW" in e for e in errors)
    assert any("missing" in e.lower() or "required" in e.lower() for e in errors)


def test_validate_inputs_ok(valid_input_dir: Path) -> None:
    ok, errors = validate_inputs(str(valid_input_dir))
    assert ok is True, format_errors(errors)
    assert errors == []


def test_format_errors() -> None:
    assert format_errors([]) == ""
    out = format_errors(["a", "b"])
    assert "a" in out and "b" in out
    assert "•" in out or "  " in out
