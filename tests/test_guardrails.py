"""
Tests for app.ui.guardrails: fallback_note, ensure_non_empty and ensure_min_points return values.
Render functions require Streamlit runtime; we patch st.warning/st.info to no-ops and assert booleans.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from app.ui.guardrails import (
    ensure_min_points,
    ensure_non_empty,
    fallback_note,
)


def test_fallback_note_insufficient_trend() -> None:
    assert "Insufficient points" in fallback_note("insufficient_trend")
    assert "2+" in fallback_note("insufficient_trend", {})
    assert "5+" in fallback_note("insufficient_trend", {"min_points": 5})


def test_fallback_note_selection_no_rows() -> None:
    assert "Selection yields no rows" in fallback_note("selection_no_rows")
    assert "Selection cleared" in fallback_note("selection_no_rows")


def test_fallback_note_insufficient_points() -> None:
    assert "Insufficient points" in fallback_note("insufficient_points", {"min_points": 3})
    assert "3+" in fallback_note("insufficient_points", {"min_points": 3})


def test_fallback_note_unknown_kind() -> None:
    assert "Showing table" in fallback_note("unknown")


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_non_empty_none(_info: object, _warning: object) -> None:
    assert ensure_non_empty(None, "r", "h") is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_non_empty_empty_df(_info: object, _warning: object) -> None:
    assert ensure_non_empty(pd.DataFrame(), "r", "h") is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_non_empty_has_rows(_info: object, _warning: object) -> None:
    assert ensure_non_empty(pd.DataFrame({"a": [1]}), "r", "h") is True


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_empty_df(_info: object, _warning: object) -> None:
    assert ensure_min_points(pd.DataFrame(), "x", min_points=2) is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_none(_info: object, _warning: object) -> None:
    assert ensure_min_points(None, "x", min_points=2) is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_single_unique(_info: object, _warning: object) -> None:
    df = pd.DataFrame({"x": [1, 1], "y": [10, 20]})
    assert ensure_min_points(df, "x", min_points=2) is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_enough_unique(_info: object, _warning: object) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    assert ensure_min_points(df, "x", min_points=2) is True


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_len_under_min(_info: object, _warning: object) -> None:
    df = pd.DataFrame({"x": [1], "y": [10]})
    assert ensure_min_points(df, "x", min_points=2) is False


@patch("app.ui.guardrails.st.warning")
@patch("app.ui.guardrails.st.info")
def test_ensure_min_points_zero_min_points(_info: object, _warning: object) -> None:
    assert ensure_min_points(pd.DataFrame({"x": []}), "x", min_points=0) is False
    df = pd.DataFrame({"x": [1]})
    assert ensure_min_points(df, "x", min_points=0) is True
