"""
Tests for app.metrics.data_summary_formulas (DATA SUMMARY checksum formulas).
Covers: negative market month, first month (begin_aum missing), and consistency with validation.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from app.metrics.data_summary_formulas import (
    asset_growth_rate,
    compute_firm_rates_df,
    external_growth_rate,
    market_impact_residual,
    organic_growth_rate,
)


def test_asset_growth_rate_basic() -> None:
    assert asset_growth_rate(100.0, 110.0) == pytest.approx(0.1)
    assert asset_growth_rate(100.0, 90.0) == pytest.approx(-0.1)


def test_organic_growth_rate_basic() -> None:
    assert organic_growth_rate(5.0, 100.0) == pytest.approx(0.05)
    assert organic_growth_rate(-2.0, 100.0) == pytest.approx(-0.02)


def test_market_impact_residual_negative_month() -> None:
    """Negative market month: end < begin + nnb => market_impact negative."""
    begin, end, nnb = 100.0, 95.0, 2.0
    mi = market_impact_residual(begin, end, nnb)
    assert mi == pytest.approx(-7.0)
    rate = external_growth_rate(begin, end, nnb)
    assert rate == pytest.approx(-0.07)


def test_first_month_begin_aum_missing() -> None:
    """First month: begin_aum missing or zero => rates NaN."""
    out = asset_growth_rate(None, 100.0)
    assert math.isnan(out)
    out = organic_growth_rate(5.0, 0.0)
    assert math.isnan(out)
    out = external_growth_rate(0.0, 100.0, 2.0)
    assert math.isnan(out)


def test_compute_firm_rates_df_first_month() -> None:
    """Firm DataFrame with one row where begin_aum=0: calc columns are NaN."""
    firm = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-01-31")],
        "begin_aum": [0.0],
        "end_aum": [100.0],
        "nnb": [5.0],
    })
    out = compute_firm_rates_df(firm)
    assert "asset_growth_rate_calc" in out.columns
    assert pd.isna(out["asset_growth_rate_calc"].iloc[0])
    assert pd.isna(out["organic_growth_rate_calc"].iloc[0])
    assert pd.isna(out["external_growth_rate_calc"].iloc[0])


def test_compute_firm_rates_df_normal() -> None:
    firm = pd.DataFrame({
        "month_end": [pd.Timestamp("2021-02-28")],
        "begin_aum": [100.0],
        "end_aum": [110.0],
        "nnb": [3.0],
    })
    out = compute_firm_rates_df(firm)
    assert out["asset_growth_rate_calc"].iloc[0] == pytest.approx(0.1)
    assert out["organic_growth_rate_calc"].iloc[0] == pytest.approx(0.03)
    assert out["external_growth_rate_calc"].iloc[0] == pytest.approx(0.07)
