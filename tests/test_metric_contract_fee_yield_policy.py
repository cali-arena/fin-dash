"""
Tests for fee_yield policy: NNF / Avg AUM and behavior when NNB <= 0.
"""
from __future__ import annotations

import math

import pytest

from app.metrics.metric_contract import compute_fee_yield

NAN = float("nan")


def test_fee_yield_positive_nnb() -> None:
    """Normal: nnf/avg_aum with positive nnb."""
    out = compute_fee_yield(1.0, 100.0, 120.0, nnb=5.0)
    assert out == pytest.approx(1.0 / 110.0)
    assert not math.isnan(out)


def test_fee_yield_nnb_zero_returns_nan() -> None:
    """DATA SUMMARY policy: when NNB <= 0, fee_yield = NaN."""
    out = compute_fee_yield(1.0, 100.0, 120.0, nnb=0.0)
    assert math.isnan(out)


def test_fee_yield_nnb_negative_returns_nan() -> None:
    out = compute_fee_yield(1.0, 100.0, 120.0, nnb=-1.0)
    assert math.isnan(out)


def test_fee_yield_nnb_none_no_policy() -> None:
    """When nnb not passed, no NNB<=0 guard (backward compat)."""
    out = compute_fee_yield(1.0, 100.0, 120.0)
    assert out == pytest.approx(1.0 / 110.0)


def test_fee_yield_avg_aum_zero_returns_nan() -> None:
    out = compute_fee_yield(1.0, 0.0, 0.0, nnb=1.0)
    assert math.isnan(out)
