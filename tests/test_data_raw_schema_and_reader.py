"""
Pytest: schema-first parsing and validation for DATA_RAW reader.
Uses tmp_path; minimal CSV with exact contract column names.
"""
from pathlib import Path

import pytest

from legacy.legacy_src.ingest.data_raw_reader import read_data_raw

# Exact required column headers per contract (including spaces)
REQUIRED_HEADER = (
    "Date,Channel,src_country, Asset Under Management ,Net new business, net new base fees ,"
    "display_firm,product_country,Standard Channel, best_of_source ,product_ticker,Segment,"
    "sub_segment,uswa_sales_focus_2020,master_custodian_firm"
)
MINIMAL_ROW = (
    "2021-01-15,Broker Dealer,US,1000.00,50.00,1.00,firm1,US,BD,1,AGG,FI,Multi,2a,cc1"
)


def _write_csv(path: Path, header: str, *rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header] + list(rows)), encoding="utf-8")


@pytest.fixture
def schema_path() -> Path:
    return Path("schemas/data_raw.schema.yml")


# --- 1) Missing required column -> hard fail ---


def test_missing_required_column_hard_fail(tmp_path: Path, schema_path: Path) -> None:
    """Missing 'Net new business' -> ValueError with message including that column."""
    csv_path = tmp_path / "DATA_RAW.csv"
    # Omit "Net new business" from header (14 cols); row has 14 values
    header_missing_nnb = (
        "Date,Channel,src_country, Asset Under Management , net new base fees ,"
        "display_firm,product_country,Standard Channel, best_of_source ,product_ticker,Segment,"
        "sub_segment,uswa_sales_focus_2020,master_custodian_firm"
    )
    row = "2021-01-15,BD,US,1000.00,1.00,f1,US,BD,1,AGG,FI,Multi,2a,cc1"
    _write_csv(csv_path, header_missing_nnb, row)

    with pytest.raises(ValueError) as exc_info:
        read_data_raw(csv_path, schema_path)

    msg = str(exc_info.value)
    assert "Missing required" in msg or "missing" in msg.lower()
    assert "Net new business" in msg


# --- 2) Extra columns allowed but logged ---


def test_extra_columns_allowed_and_logged(
    tmp_path: Path, schema_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Extra column FOO_EXTRA present -> df includes it and warning mentions FOO_EXTRA."""
    csv_path = tmp_path / "DATA_RAW.csv"
    header_extra = REQUIRED_HEADER + ",FOO_EXTRA"
    row_extra = MINIMAL_ROW + ",extra_val"
    _write_csv(csv_path, header_extra, row_extra)

    with caplog.at_level("WARNING"):
        df = read_data_raw(csv_path, schema_path)

    assert "FOO_EXTRA" in df.columns
    assert "FOO_EXTRA" in caplog.text or "Extra columns" in caplog.text


# --- 3) Currency parsing parentheses ---


def test_currency_parentheses_negative(tmp_path: Path, schema_path: Path) -> None:
    """AUM '(1,234.50)' -> asset_under_management == -1234.50."""
    csv_path = tmp_path / "DATA_RAW.csv"
    # Quote AUM so comma inside is not a CSV separator
    row = '2021-01-15,Broker Dealer,US,"(1,234.50)",50.00,1.00,firm1,US,BD,1,AGG,FI,Multi,2a,cc1'
    _write_csv(csv_path, REQUIRED_HEADER, row)

    df = read_data_raw(csv_path, schema_path)

    assert "asset_under_management" in df.columns
    assert df["asset_under_management"].iloc[0] == pytest.approx(-1234.50)


# --- 4) Date month-end normalization ---


def test_date_month_end_normalization(tmp_path: Path, schema_path: Path) -> None:
    """Input Date '2021-01-15' -> canonical date column equals 2021-01-31."""
    csv_path = tmp_path / "DATA_RAW.csv"
    _write_csv(csv_path, REQUIRED_HEADER, MINIMAL_ROW)

    df = read_data_raw(csv_path, schema_path)

    assert "date" in df.columns
    got = df["date"].iloc[0]
    assert got.year == 2021 and got.month == 1 and got.day == 31
