"""
Unit tests for app.filters: apply_filters validates columns and applies in deterministic order.
"""
import pandas as pd
import pytest

from app.filters import apply_filters, dimension_columns


def test_dimension_columns_excludes_time_and_measures() -> None:
    """dimension_columns returns sorted list excluding month_end and measure cols."""
    df = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "channel_l1": ["A"],
        "end_aum": [100.0],
    })
    assert dimension_columns(df) == ["channel_l1"]


def test_apply_filters_month_end_range() -> None:
    """month_end_range filter applies boolean mask on month_end."""
    df = pd.DataFrame({
        "month_end": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-01"]),
        "end_aum": [10.0, 20.0, 30.0],
    })
    out = apply_filters(df, {"month_end_range": (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-07-01"))})
    assert len(out) == 1
    assert out["month_end"].iloc[0] == pd.Timestamp("2024-06-01")


def test_apply_filters_in_filter() -> None:
    """Dimension IN filter keeps only rows where col in values."""
    df = pd.DataFrame({"x": ["a", "b", "c"], "v": [1, 2, 3]})
    out = apply_filters(df, {"x": ["a", "c"]})
    assert len(out) == 2
    assert set(out["x"]) == {"a", "c"}


def test_apply_filters_empty_list_no_filter() -> None:
    """Empty list for a dimension key means skip that filter."""
    df = pd.DataFrame({"x": ["a", "b"], "v": [1, 2]})
    out = apply_filters(df, {"x": []})
    assert len(out) == 2


def test_apply_filters_validates_column_exists() -> None:
    """apply_filters raises ValueError if filter column not in df."""
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError) as exc_info:
        apply_filters(df, {"y": [1]})
    assert "not in DataFrame" in str(exc_info.value)
