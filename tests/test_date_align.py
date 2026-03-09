"""
Unit tests for app.date_align: get_latest_month_end, get_prior_month_end, get_year_start_month_end.
Uses synthetic month_end sequences with gaps to ensure gap-aware behavior.
"""
from __future__ import annotations

import pytest

import pandas as pd

from app.date_align import (
    get_latest_month_end,
    get_prior_month_end,
    get_year_start_month_end,
    has_month_gaps,
    is_single_month,
)


def _df(month_ends: list[str]) -> pd.DataFrame:
    """DataFrame with single column month_end from ISO date strings."""
    return pd.DataFrame({
        "month_end": pd.to_datetime(month_ends),
    })


def test_get_latest_month_end_empty() -> None:
    assert get_latest_month_end(pd.DataFrame()) is None
    assert get_latest_month_end(pd.DataFrame(columns=["other"])) is None
    assert get_latest_month_end(_df([])) is None


def test_get_latest_month_end_single() -> None:
    df = _df(["2021-06-30"])
    out = get_latest_month_end(df)
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2021-06-30"


def test_get_latest_month_end_with_gaps() -> None:
    # Gaps: Jan, Mar, Jun 2021; Jan 2022
    df = _df(["2021-01-31", "2021-03-31", "2021-06-30", "2022-01-31"])
    out = get_latest_month_end(df)
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2022-01-31"


def test_get_prior_month_end_empty_or_single() -> None:
    assert get_prior_month_end(_df([]), "2022-01-31") is None
    assert get_prior_month_end(_df(["2022-01-31"]), "2022-01-31") is None


def test_get_prior_month_end_with_gaps() -> None:
    # Data: 2021-01, 2021-03, 2021-06, 2022-01 (no Feb, Apr, May, etc.)
    df = _df(["2021-01-31", "2021-03-31", "2021-06-30", "2022-01-31"])
    # Prior to 2022-01-31 is 2021-06-30 (not calendar prev month)
    out = get_prior_month_end(df, "2022-01-31")
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2021-06-30"

    # Prior to 2021-06-30 is 2021-03-31
    out2 = get_prior_month_end(df, "2021-06-30")
    assert out2 is not None
    assert pd.Timestamp(out2).strftime("%Y-%m-%d") == "2021-03-31"

    # Prior to 2021-01-31 is None
    assert get_prior_month_end(df, "2021-01-31") is None


def test_get_prior_month_end_consecutive_months() -> None:
    df = _df(["2021-10-31", "2021-11-30", "2021-12-31"])
    assert get_prior_month_end(df, "2021-12-31") is not None
    assert pd.Timestamp(get_prior_month_end(df, "2021-12-31")).strftime("%Y-%m-%d") == "2021-11-30"
    assert get_prior_month_end(df, "2021-10-31") is None


def test_get_year_start_month_end_empty() -> None:
    assert get_year_start_month_end(_df([]), "2022-06-30") is None


def test_get_year_start_month_end_with_gaps() -> None:
    # Same year: only Mar and Jun 2021
    df = _df(["2021-03-31", "2021-06-30", "2022-01-31"])
    # First month in 2021 is 2021-03-31 (Jan/Feb not in data)
    out = get_year_start_month_end(df, "2021-06-30")
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2021-03-31"

    # For 2022, only Jan exists
    out2 = get_year_start_month_end(df, "2022-01-31")
    assert out2 is not None
    assert pd.Timestamp(out2).strftime("%Y-%m-%d") == "2022-01-31"


def test_get_year_start_month_end_full_year() -> None:
    # Use proper month-end dates for 2021
    df = pd.DataFrame({
        "month_end": pd.to_datetime([
            "2021-01-31", "2021-02-28", "2021-03-31", "2021-04-30", "2021-05-31", "2021-06-30",
            "2021-07-31", "2021-08-31", "2021-09-30", "2021-10-31", "2021-11-30", "2021-12-31",
        ]),
    })
    out = get_year_start_month_end(df, "2021-12-31")
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2021-01-31"


def test_get_year_start_month_end_no_data_in_year() -> None:
    df = _df(["2020-12-31", "2022-01-31"])
    # 2021 has no rows
    assert get_year_start_month_end(df, "2021-06-30") is None


def test_na_dropped() -> None:
    df = pd.DataFrame({
        "month_end": pd.to_datetime(["2021-01-31", "2021-03-31", pd.NaT, "2021-06-30"]),
    })
    assert get_latest_month_end(df) is not None
    assert pd.Timestamp(get_latest_month_end(df)).strftime("%Y-%m-%d") == "2021-06-30"
    assert get_prior_month_end(df, "2021-06-30") is not None
    assert pd.Timestamp(get_prior_month_end(df, "2021-06-30")).strftime("%Y-%m-%d") == "2021-03-31"
    assert get_year_start_month_end(df, "2021-06-30") is not None
    assert pd.Timestamp(get_year_start_month_end(df, "2021-06-30")).strftime("%Y-%m-%d") == "2021-01-31"


def test_duplicate_months_deduped() -> None:
    df = pd.DataFrame({
        "month_end": pd.to_datetime(["2021-01-31", "2021-01-31", "2021-06-30", "2021-06-30"]),
    })
    out = get_latest_month_end(df)
    assert out is not None
    assert pd.Timestamp(out).strftime("%Y-%m-%d") == "2021-06-30"
    out_prior = get_prior_month_end(df, "2021-06-30")
    assert out_prior is not None
    assert pd.Timestamp(out_prior).strftime("%Y-%m-%d") == "2021-01-31"


def test_is_single_month() -> None:
    assert is_single_month(pd.DataFrame()) is False
    assert is_single_month(_df([])) is False
    assert is_single_month(_df(["2021-06-30"])) is True
    assert is_single_month(_df(["2021-01-31", "2021-02-28"])) is False


def test_has_month_gaps_no_gaps() -> None:
    """Consecutive months -> no gaps."""
    df = _df(["2021-01-31", "2021-02-28", "2021-03-31"])
    assert has_month_gaps(df) is False
    df2 = _df(["2021-10-31", "2021-11-30", "2021-12-31"])
    assert has_month_gaps(df2) is False


def test_has_month_gaps_with_gaps() -> None:
    """Non-consecutive months (e.g. Jan, Mar, May) -> gaps."""
    df = _df(["2021-01-31", "2021-03-31", "2021-05-31"])
    assert has_month_gaps(df) is True
    df2 = _df(["2021-01-31", "2021-06-30", "2022-01-31"])
    assert has_month_gaps(df2) is True


def test_has_month_gaps_single_or_empty() -> None:
    """Single month or empty -> no gaps (len < 2)."""
    assert has_month_gaps(_df([])) is False
    assert has_month_gaps(_df(["2021-06-30"])) is False
