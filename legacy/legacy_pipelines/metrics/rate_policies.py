"""
Pure functions for rate policies: safe division, begin_aum guard, fee_yield guard, clamp.
Deterministic and side-effect free. Policy dicts match configs/metrics_policy.yml.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


def safe_divide(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """
    Return numer/denom as float. Converts +/-inf to NaN; preserves NaN propagation.
    """
    out = numer / denom
    out = out.astype(float)
    out = out.replace([math.inf, -math.inf], float("nan"))
    return out


def apply_begin_aum_guard(
    rate: pd.Series,
    begin_aum: pd.Series,
    policy: dict[str, Any],
) -> pd.Series:
    """
    Where begin_aum <= threshold: set rate to NaN (mode "nan") or 0.0 (mode "zero").
    Returns a new series; does not mutate inputs.
    """
    mode = (policy.get("mode") or "nan").strip().lower()
    if mode not in ("nan", "zero"):
        mode = "nan"
    threshold = float(policy.get("threshold", 0.0))
    out = rate.astype(float).copy()
    mask = (begin_aum <= threshold) & begin_aum.notna()
    if mode == "nan":
        out = out.where(~mask, float("nan"))
    else:
        out = out.where(~mask, 0.0)
    return out


def apply_fee_yield_guard(
    fee_yield: pd.Series,
    nnb: pd.Series,
    policy: dict[str, Any],
) -> pd.Series:
    """
    Where nnb <= threshold: set fee_yield to NaN (mode "nan"), 0.0 (mode "zero"), or cap_value (mode "cap").
    Returns a new series; does not mutate inputs.
    """
    mode = (policy.get("mode") or "nan").strip().lower()
    if mode not in ("nan", "zero", "cap"):
        mode = "nan"
    threshold = float(policy.get("threshold", 0.0))
    cap_value = float(policy.get("cap_value", 0.0))
    out = fee_yield.astype(float).copy()
    mask = (nnb <= threshold) & nnb.notna()
    if mode == "nan":
        out = out.where(~mask, float("nan"))
    elif mode == "zero":
        out = out.where(~mask, 0.0)
    else:
        out = out.where(~mask, cap_value)
    return out


def apply_clamp(
    rate: pd.Series,
    metric_name: str,
    clamp_policy: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """
    If enabled: warn_only => do not change values, clamped_flag True where out of bounds;
    hard_clamp => clip to [min, max] and clamped_flag True where clipping occurred.
    Returns (rate_out, clamped_flag). Both series are new; inputs are not mutated.
    """
    if not clamp_policy.get("enabled", True):
        flag = pd.Series(False, index=rate.index)
        return rate.astype(float).copy(), flag

    caps = clamp_policy.get("caps") or {}
    cap_spec = caps.get(metric_name) if isinstance(caps, dict) else None
    if not isinstance(cap_spec, dict):
        return rate.astype(float).copy(), pd.Series(False, index=rate.index)

    mn = float(cap_spec.get("min", -math.inf))
    mx = float(cap_spec.get("max", math.inf))
    mode = (clamp_policy.get("mode") or "warn_only").strip().lower()
    if mode not in ("warn_only", "hard_clamp"):
        mode = "warn_only"

    rate_f = rate.astype(float)
    out_of_bounds = (rate_f < mn) | (rate_f > mx)
    # NaN is not < mn and not > mx, so out_of_bounds is False for NaN
    out_of_bounds = out_of_bounds & rate_f.notna()

    if mode == "warn_only":
        return rate_f.copy(), out_of_bounds
    # hard_clamp
    rate_out = rate_f.clip(lower=mn, upper=mx)
    clamped_flag = out_of_bounds.copy()
    return rate_out, clamped_flag


def coerce_inf_to_nan(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return a new DataFrame with +/-inf replaced by NaN in the listed columns.
    Columns not in df are skipped. Does not mutate the input DataFrame.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = out[c].replace([math.inf, -math.inf], float("nan"))
    return out
