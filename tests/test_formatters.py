"""
Tests for app.ui.formatters: NaN handling, scaling boundaries, negatives, percent/bps.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from app.ui.formatters import (
    NA_STR,
    fmt_bps,
    fmt_currency,
    fmt_date,
    fmt_percent,
    format_df,
    infer_common_formats,
)


def test_fmt_currency_nan_none() -> None:
    assert fmt_currency(None) == NA_STR
    assert fmt_currency(float("nan")) == NA_STR
    assert fmt_currency(np.nan) == NA_STR


def test_fmt_currency_scaling_boundaries() -> None:
    assert fmt_currency(999, unit="auto") == "999.00"
    assert fmt_currency(1000, unit="auto") == "1.00K"
    assert fmt_currency(1234, unit="auto") == "1.23K"
    assert fmt_currency(1e6, unit="auto") == "1.00M"
    assert fmt_currency(1.25e6, unit="auto") == "1.25M"
    assert fmt_currency(1e9, unit="auto") == "1.00B"
    assert fmt_currency(2.5e9, unit="auto") == "2.50B"


def test_fmt_currency_negative() -> None:
    assert fmt_currency(-1234, unit="auto") == "-1.23K"
    assert fmt_currency(-1e6, unit="auto") == "-1.00M"


def test_fmt_currency_thousands_separator() -> None:
    assert fmt_currency(1234.56, unit="full") == "1,234.56"
    assert fmt_currency(1234567.89, unit="full") == "1,234,567.89"


def test_fmt_percent_nan() -> None:
    assert fmt_percent(None) == NA_STR
    assert fmt_percent(float("nan")) == NA_STR


def test_fmt_percent_conversion() -> None:
    assert fmt_percent(0.1234, decimals=2) == "12.34%"
    assert fmt_percent(0.0) == "0.00%"
    assert fmt_percent(-0.05, decimals=2) == "-5.00%"
    assert fmt_percent(0.01, decimals=0) == "1%"


def test_fmt_percent_signed() -> None:
    assert fmt_percent(0.05, signed=True) == "+5.00%"
    assert fmt_percent(-0.05, signed=True) == "-5.00%"


def test_fmt_bps_nan() -> None:
    assert fmt_bps(None) == NA_STR
    assert fmt_bps(float("nan")) == NA_STR


def test_fmt_bps_fraction_input() -> None:
    assert fmt_bps(0.0001, decimals=0) == "1 bps"
    assert fmt_bps(0.01, decimals=0) == "100 bps"
    assert fmt_bps(0.00015, decimals=1) == "1.5 bps"


def test_fmt_date() -> None:
    assert fmt_date(date(2024, 3, 15)) == "2024-03"
    assert fmt_date(datetime(2024, 6, 1, 12, 0)) == "2024-06"
    assert fmt_date("2024-01-15") == "2024-01"
    assert fmt_date(None) == NA_STR
    assert fmt_date("") == NA_STR


def test_format_df() -> None:
    df = pd.DataFrame({"a": [1.5, float("nan"), 3], "b": [0.1, 0.2, 0.3]})
    col_formats = {"a": lambda x: fmt_currency(x), "b": lambda x: fmt_percent(x)}
    out = format_df(df, col_formats)
    assert out["a"].iloc[0] == "1.50"
    assert out["a"].iloc[1] == NA_STR
    assert out["b"].iloc[0] == "10.00%"


def test_infer_common_formats() -> None:
    df = pd.DataFrame({"end_aum": [100], "nnb": [10], "ogr": [0.05], "other": [1]})
    fmts = infer_common_formats(df)
    assert "end_aum" in fmts
    assert "nnb" in fmts
    assert "ogr" in fmts
    assert "other" not in fmts
    assert fmts["end_aum"](1000) == "1.00K"
    assert fmts["ogr"](0.1234) == "12.34%"


def test_fmt_currency_zero() -> None:
    assert fmt_currency(0, unit="auto") == "0.00"


def test_fmt_percent_zero() -> None:
    assert fmt_percent(0) == "0.00%"
