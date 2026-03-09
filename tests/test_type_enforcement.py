"""
Pytest: type enforcement pipeline — strict date parsing, currency, identifiers, stats.
No file I/O; small in-memory DataFrames only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_src.transform.type_enforcement import (
    enforce_types_data_raw,
    normalize_identifier,
    parse_currency,
)


def test_strict_date_parsing_and_rejects() -> None:
    """Strict date parsing: valid dates kept and converted to month-end; bad_date row rejected."""
    df = pd.DataFrame({
        "date": ["2021-01-15", "01/31/2021", "bad-date"],
        "amount": ["100", "200", "300"],
        "channel": ["A", "B", "C"],
    })
    df_clean, df_rejects, stats = enforce_types_data_raw(
        df,
        date_col="date",
        currency_cols=["amount"],
        id_cols=["channel"],
    )
    assert len(df_clean) == 2
    assert len(df_rejects) == 1
    assert df_rejects["_reject_reason"].iloc[0] == "bad_date"
    assert df_rejects["date"].iloc[0] == "bad-date"

    # Month-end: 2021-01-15 -> 2021-01-31, 01/31/2021 -> 2021-01-31
    jan31 = pd.Timestamp("2021-01-31")
    assert (df_clean["date"].iloc[0] == jan31) and (df_clean["date"].iloc[1] == jan31)


def test_currency_parsing_and_rejects() -> None:
    """Currency parsing: valid values parsed; 'oops' row rejected with bad_currency:<col>."""
    df = pd.DataFrame({
        "date": ["2021-01-15", "2021-01-15", "2021-01-15", "2021-01-15"],
        "amount": ["(1,234.56)", "-1,234.56", "1234.56", "oops"],
        "channel": ["A", "B", "C", "D"],
    })
    df_clean, df_rejects, stats = enforce_types_data_raw(
        df,
        date_col="date",
        currency_cols=["amount"],
        id_cols=["channel"],
    )
    assert len(df_clean) == 3
    assert len(df_rejects) == 1
    assert df_rejects["_reject_reason"].iloc[0] == "bad_currency:amount"
    assert df_rejects["amount"].iloc[0] == "oops"

    # Parsed floats
    assert abs(df_clean.loc[df_clean["channel"] == "A", "amount"].iloc[0] - (-1234.56)) < 1e-2
    assert abs(df_clean.loc[df_clean["channel"] == "B", "amount"].iloc[0] - (-1234.56)) < 1e-2
    assert abs(df_clean.loc[df_clean["channel"] == "C", "amount"].iloc[0] - 1234.56) < 1e-2


def test_identifier_normalization() -> None:
    """Identifier normalization: strip and collapse whitespace; case preserved."""
    df = pd.DataFrame({
        "date": ["2021-01-15", "2021-01-15"],
        "amount": ["0", "0"],
        "channel": ["  Broker   Dealer ", "UNITED   STATES"],
    })
    df_clean, df_rejects, stats = enforce_types_data_raw(
        df,
        date_col="date",
        currency_cols=["amount"],
        id_cols=["channel"],
    )
    assert len(df_rejects) == 0
    assert df_clean["channel"].iloc[0] == "Broker Dealer"
    assert df_clean["channel"].iloc[1] == "UNITED STATES"

    # Direct normalize_identifier
    ser = pd.Series(["  Broker   Dealer ", "UNITED   STATES"])
    out = normalize_identifier(ser)
    assert out.iloc[0] == "Broker Dealer"
    assert out.iloc[1] == "UNITED STATES"


def test_stats_correctness() -> None:
    """Stats: rows_in, rows_clean, rows_rejected and reject_counts match."""
    df = pd.DataFrame({
        "date": ["2021-01-15", "bad", "2021-01-16"],
        "amount": ["1,000", "oops", "2,000"],
        "channel": ["X", "Y", "Z"],
    })
    df_clean, df_rejects, stats = enforce_types_data_raw(
        df,
        date_col="date",
        currency_cols=["amount"],
        id_cols=["channel"],
    )
    assert stats["rows_in"] == 3
    assert stats["rows_clean"] == len(df_clean)
    assert stats["rows_rejected"] == len(df_rejects)
    assert stats["rows_clean"] + stats["rows_rejected"] == stats["rows_in"]

    assert "reject_counts" in stats
    # Row 1 bad_date, row 2 could be bad_currency if date was fixed (here row 1 is bad_date, row 2 is bad_date so only one rejected for bad_date? No: row 0 valid, row 1 bad_date, row 2 valid -> 1 reject. So reject_counts has "bad_date": 1.
    assert stats["reject_counts"].get("bad_date") == 1
    assert "currency_parse_nulls" in stats
