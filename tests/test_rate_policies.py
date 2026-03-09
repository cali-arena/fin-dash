"""
Unit tests for pipelines.metrics.rate_policies: safe_divide, guards, clamp, coerce_inf_to_nan.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from legacy.legacy_pipelines.metrics.rate_policies import (
    safe_divide,
    apply_begin_aum_guard,
    apply_fee_yield_guard,
    apply_clamp,
    coerce_inf_to_nan,
)


# --- safe_divide ---


def test_safe_divide_basic() -> None:
    numer = pd.Series([1.0, 4.0, 0.0])
    denom = pd.Series([2.0, 2.0, 1.0])
    out = safe_divide(numer, denom)
    pd.testing.assert_series_equal(out, pd.Series([0.5, 2.0, 0.0]))


def test_safe_divide_nan_propagation() -> None:
    numer = pd.Series([1.0, float("nan"), 3.0])
    denom = pd.Series([1.0, 1.0, float("nan")])
    out = safe_divide(numer, denom)
    assert out.iloc[0] == 1.0
    assert math.isnan(out.iloc[1])
    assert math.isnan(out.iloc[2])


def test_safe_divide_inf_becomes_nan() -> None:
    numer = pd.Series([1.0, -1.0, 1.0])
    denom = pd.Series([0.0, 0.0, 0.0])
    out = safe_divide(numer, denom)
    assert out.iloc[0] != out.iloc[0]  # NaN
    assert out.iloc[1] != out.iloc[1]
    assert out.iloc[2] != out.iloc[2]
    assert not math.isinf(out.iloc[0])
    assert not math.isinf(out.iloc[1])


def test_safe_divide_zero_denom_nan() -> None:
    numer = pd.Series([0.0])
    denom = pd.Series([0.0])
    out = safe_divide(numer, denom)
    assert len(out) == 1
    assert math.isnan(out.iloc[0])


def test_safe_divide_returns_float_dtype() -> None:
    numer = pd.Series([1], dtype="Int64")
    denom = pd.Series([2], dtype="Int64")
    out = safe_divide(numer, denom)
    assert out.dtype == float or (hasattr(out.dtype, "kind") and out.dtype.kind == "f")


# --- apply_begin_aum_guard ---


def test_begin_aum_guard_nan_mode() -> None:
    rate = pd.Series([1.0, 2.0, 3.0])
    begin_aum = pd.Series([0.0, 0.5, 1.0])
    policy = {"mode": "nan", "threshold": 0.0}
    out = apply_begin_aum_guard(rate, begin_aum, policy)
    assert math.isnan(out.iloc[0])  # begin_aum 0.0 <= 0
    assert out.iloc[1] == 2.0       # 0.5 > 0, unchanged
    assert out.iloc[2] == 3.0


def test_begin_aum_guard_zero_mode() -> None:
    rate = pd.Series([1.0, 2.0, 3.0])
    begin_aum = pd.Series([0.0, 0.0, 1.0])
    policy = {"mode": "zero", "threshold": 0.0}
    out = apply_begin_aum_guard(rate, begin_aum, policy)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == 0.0
    assert out.iloc[2] == 3.0


def test_begin_aum_guard_threshold_positive() -> None:
    rate = pd.Series([10.0, 20.0])
    begin_aum = pd.Series([1.0, 5.0])
    policy = {"mode": "nan", "threshold": 3.0}
    out = apply_begin_aum_guard(rate, begin_aum, policy)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == 20.0


def test_begin_aum_guard_does_not_mutate_input() -> None:
    rate = pd.Series([1.0, 2.0])
    begin_aum = pd.Series([0.0, 1.0])
    policy = {"mode": "zero", "threshold": 0.0}
    apply_begin_aum_guard(rate, begin_aum, policy)
    assert rate.iloc[0] == 1.0
    assert begin_aum.iloc[0] == 0.0


# --- apply_fee_yield_guard ---


def test_fee_yield_guard_nan_mode() -> None:
    fee_yield = pd.Series([0.01, 0.02, 0.03])
    nnb = pd.Series([0.0, 10.0, 20.0])
    policy = {"mode": "nan", "threshold": 0.0}
    out = apply_fee_yield_guard(fee_yield, nnb, policy)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == 0.02
    assert out.iloc[2] == 0.03


def test_fee_yield_guard_zero_mode() -> None:
    fee_yield = pd.Series([0.01, 0.02])
    nnb = pd.Series([0.0, 100.0])
    policy = {"mode": "zero", "threshold": 0.0}
    out = apply_fee_yield_guard(fee_yield, nnb, policy)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == 0.02


def test_fee_yield_guard_cap_mode() -> None:
    fee_yield = pd.Series([0.05, 0.10])
    nnb = pd.Series([0.0, 50.0])
    policy = {"mode": "cap", "threshold": 0.0, "cap_value": 0.0}
    out = apply_fee_yield_guard(fee_yield, nnb, policy)
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == 0.10


def test_fee_yield_guard_cap_value_nonzero() -> None:
    fee_yield = pd.Series([0.05])
    nnb = pd.Series([0.0])
    policy = {"mode": "cap", "threshold": 0.0, "cap_value": 0.15}
    out = apply_fee_yield_guard(fee_yield, nnb, policy)
    assert out.iloc[0] == 0.15


# --- apply_clamp ---


def test_clamp_warn_only_does_not_change_values() -> None:
    rate = pd.Series([-5.0, 0.0, 5.0])
    clamp_policy = {
        "enabled": True,
        "mode": "warn_only",
        "caps": {"ogr": {"min": -2.0, "max": 2.0}},
    }
    rate_out, clamped = apply_clamp(rate, "ogr", clamp_policy)
    pd.testing.assert_series_equal(rate_out, rate)
    assert clamped.iloc[0] == True
    assert clamped.iloc[1] == False
    assert clamped.iloc[2] == True


def test_clamp_hard_clamp_clips() -> None:
    rate = pd.Series([-5.0, 0.0, 5.0])
    clamp_policy = {
        "enabled": True,
        "mode": "hard_clamp",
        "caps": {"ogr": {"min": -2.0, "max": 2.0}},
    }
    rate_out, clamped = apply_clamp(rate, "ogr", clamp_policy)
    assert rate_out.iloc[0] == -2.0
    assert rate_out.iloc[1] == 0.0
    assert rate_out.iloc[2] == 2.0
    assert clamped.iloc[0] == True
    assert clamped.iloc[1] == False
    assert clamped.iloc[2] == True


def test_clamp_disabled_returns_unchanged() -> None:
    rate = pd.Series([10.0])
    clamp_policy = {"enabled": False, "mode": "hard_clamp", "caps": {"x": {"min": 0.0, "max": 1.0}}}
    rate_out, clamped = apply_clamp(rate, "x", clamp_policy)
    assert rate_out.iloc[0] == 10.0
    assert clamped.iloc[0] == False


def test_clamp_unknown_metric_returns_unchanged() -> None:
    rate = pd.Series([1.0])
    clamp_policy = {"enabled": True, "mode": "hard_clamp", "caps": {"ogr": {"min": 0.0, "max": 1.0}}}
    rate_out, clamped = apply_clamp(rate, "unknown_metric", clamp_policy)
    assert rate_out.iloc[0] == 1.0
    assert clamped.iloc[0] == False


def test_clamp_nan_not_marked_clamped() -> None:
    rate = pd.Series([float("nan"), 3.0])
    clamp_policy = {"enabled": True, "mode": "warn_only", "caps": {"x": {"min": -1.0, "max": 1.0}}}
    _, clamped = apply_clamp(rate, "x", clamp_policy)
    assert clamped.iloc[0] == False
    assert clamped.iloc[1] == True


def test_clamp_fee_yield_bounds() -> None:
    rate = pd.Series([-0.01, 0.05, 0.25])
    clamp_policy = {"enabled": True, "mode": "hard_clamp", "caps": {"fee_yield": {"min": 0.0, "max": 0.2}}}
    rate_out, clamped = apply_clamp(rate, "fee_yield", clamp_policy)
    assert rate_out.iloc[0] == 0.0
    assert rate_out.iloc[1] == 0.05
    assert rate_out.iloc[2] == 0.2
    assert clamped.iloc[0] == True
    assert clamped.iloc[1] == False
    assert clamped.iloc[2] == True


# --- coerce_inf_to_nan ---


def test_coerce_inf_to_nan_replaces_inf() -> None:
    df = pd.DataFrame({"a": [1.0, math.inf, -math.inf], "b": [0.0, 0.0, 0.0]})
    out = coerce_inf_to_nan(df, ["a"])
    assert not math.isinf(out["a"].iloc[1])
    assert not math.isinf(out["a"].iloc[2])
    assert math.isnan(out["a"].iloc[1])
    assert math.isnan(out["a"].iloc[2])
    assert out["a"].iloc[0] == 1.0
    pd.testing.assert_series_equal(out["b"], df["b"])


def test_coerce_inf_to_nan_skips_missing_cols() -> None:
    df = pd.DataFrame({"a": [math.inf]})
    out = coerce_inf_to_nan(df, ["a", "missing"])
    assert math.isnan(out["a"].iloc[0])
    assert "missing" not in out.columns


def test_coerce_inf_to_nan_does_not_mutate_input() -> None:
    df = pd.DataFrame({"a": [math.inf]})
    coerce_inf_to_nan(df, ["a"])
    assert math.isinf(df["a"].iloc[0])


def test_coerce_inf_to_nan_multiple_cols() -> None:
    df = pd.DataFrame({"x": [1.0, math.inf], "y": [-math.inf, 2.0]})
    out = coerce_inf_to_nan(df, ["x", "y"])
    assert math.isnan(out["x"].iloc[1])
    assert math.isnan(out["y"].iloc[0])
    assert out["x"].iloc[0] == 1.0
    assert out["y"].iloc[1] == 2.0
