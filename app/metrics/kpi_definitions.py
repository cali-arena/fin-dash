"""
This module defines the deterministic KPI contract used by the Executive Summary
across all dashboard tabs. All KPI calculations are centralized here; the UI must
never compute metrics ad-hoc. Calculations operate on firm-level monthly data
(agg/firm_monthly or analytics.v_firm_monthly).

All metric math must go through the guard functions (apply_metric_guards, safe_divide)
so that behavior is deterministic and aligned with configs/metrics_policy.yml across
the dashboard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Project root for config path
_APP_DIR = Path(__file__).resolve().parent.parent
_ROOT = _APP_DIR.parent


@lru_cache(maxsize=1)
def load_metrics_policy(path: str = "configs/metrics_policy.yml") -> dict[str, Any]:
    """
    Load metrics policy from YAML. Expects structure:
      metrics_policy:
        division_by_zero: "nan"
        infinite_values: "nan"
        missing_values: "nan"
    Falls back to defaults if file or keys are missing.
    """
    full = _ROOT / path
    out = {"division_by_zero": "nan", "infinite_values": "nan", "missing_values": "nan"}
    if not full.exists():
        return out
    try:
        import yaml
        data = yaml.safe_load(full.read_text(encoding="utf-8")) or {}
        policy = data.get("metrics_policy") or data.get("policies") or {}
        if isinstance(policy, dict):
            out["division_by_zero"] = policy.get("division_by_zero") or out["division_by_zero"]
            out["infinite_values"] = policy.get("infinite_values") or policy.get("inf_handling", {}).get("mode") or out["infinite_values"]
            out["missing_values"] = policy.get("missing_values") or out["missing_values"]
    except Exception:
        pass
    return out


def apply_metric_guards(value: float | None) -> float:
    """
    Apply global metrics policy guards to a single value. All KPI outputs must
    pass through this so behavior is deterministic across the dashboard.
    - If value is inf or -inf → return np.nan
    - If value is None → return np.nan
    - If value is NaN → keep NaN
    """
    _policy = load_metrics_policy()
    if value is None:
        # missing_values policy
        return np.nan
    try:
        f = float(value)
    except (TypeError, ValueError):
        return np.nan
    if math.isnan(f):
        return np.nan
    if math.isinf(f):
        # infinite_values policy (e.g. "nan")
        return np.nan
    return f


def safe_divide(a: float | None, b: float | None) -> float:
    """
    Division that respects metrics policy: no division by zero. All divisions
    in KPI calculations must use this for deterministic behavior.
    - If b == 0 or b is None → return NaN
    - Else return a / b (with apply_metric_guards on result).
    """
    _policy = load_metrics_policy()
    if b is None:
        return np.nan
    try:
        bf = float(b)
    except (TypeError, ValueError):
        return np.nan
    if bf == 0 or math.isnan(bf) or math.isinf(bf):
        # division_by_zero / invalid divisor
        return np.nan
    try:
        af = float(a) if a is not None else np.nan
    except (TypeError, ValueError):
        af = np.nan
    return apply_metric_guards(af / bf)


def _safe_float(val: Any) -> float:
    """Coerce to float; return NaN for non-finite or missing. Uses apply_metric_guards for policy consistency."""
    if val is None:
        return np.nan
    if isinstance(val, float) and math.isnan(val):
        return np.nan
    try:
        f = float(val)
        return apply_metric_guards(f)
    except (TypeError, ValueError):
        return np.nan


@dataclass(frozen=True)
class FirmMonthlySnapshot:
    """
    One row of firm-level monthly data. Schema aligns with agg/firm_monthly
    or analytics.v_firm_monthly.
    """
    month_end: pd.Timestamp | None
    begin_aum: float
    end_aum: float
    nnb: float
    nnf: float
    market_pnl: float
    ogr: float
    market_impact_rate: float


@dataclass
class ExecutiveKPIs:
    """
    Executive Summary KPIs derived from the latest month and prior period.
    All growth rates are decimal (e.g. 0.05 = 5%). Missing or invalid values are NaN.
    """
    end_aum: float
    mom_growth: float
    ytd_growth: float
    nnb: float
    nnf: float
    ogr: float
    market_impact: float
    market_pnl: float


def compute_executive_kpis(df: pd.DataFrame) -> ExecutiveKPIs:
    """
    Compute Executive Summary KPIs from firm monthly data.

    - Latest month = max(month_end). End AUM = end_aum of that row.
    - MoM growth = (end_aum_t - end_aum_t-1) / end_aum_t-1; NaN if prior end_aum is 0.
    - YTD growth = (end_aum_latest - end_aum_year_start) / end_aum_year_start;
      year start = first row of current year; NaN if year-start value is 0.
    - NNB, NNF, OGR, market_impact_rate, market_pnl = values from latest row.
    - DataFrame is sorted by month_end ascending. Infinite values are coerced to NaN.
    - Never raises; returns ExecutiveKPIs with NaN where data is incomplete.
    """
    try:
        if df is None or df.empty:
            return ExecutiveKPIs(
                end_aum=np.nan,
                mom_growth=np.nan,
                ytd_growth=np.nan,
                nnb=np.nan,
                nnf=np.nan,
                ogr=np.nan,
                market_impact=np.nan,
                market_pnl=np.nan,
            )
    except Exception:
        return ExecutiveKPIs(
            end_aum=np.nan,
            mom_growth=np.nan,
            ytd_growth=np.nan,
            nnb=np.nan,
            nnf=np.nan,
            ogr=np.nan,
            market_impact=np.nan,
            market_pnl=np.nan,
        )

    try:
        df = df.sort_values("month_end", ascending=True).reset_index(drop=True)
    except Exception:
        return ExecutiveKPIs(
            end_aum=np.nan,
            mom_growth=np.nan,
            ytd_growth=np.nan,
            nnb=np.nan,
            nnf=np.nan,
            ogr=np.nan,
            market_impact=np.nan,
            market_pnl=np.nan,
        )

    cols = set(df.columns)
    def _get(s: pd.Series, default: float = np.nan) -> float:
        if s is None or s.empty:
            return default
        v = s.iloc[-1]
        return _safe_float(v)

    # Latest row index
    latest_idx = len(df) - 1
    latest = df.iloc[latest_idx]
    month_end_col = "month_end"
    if month_end_col not in cols:
        return ExecutiveKPIs(
            end_aum=np.nan,
            mom_growth=np.nan,
            ytd_growth=np.nan,
            nnb=np.nan,
            nnf=np.nan,
            ogr=np.nan,
            market_impact=np.nan,
            market_pnl=np.nan,
        )

    latest_ts = latest.get(month_end_col)
    try:
        latest_dt = pd.Timestamp(latest_ts) if latest_ts is not None else None
    except Exception:
        latest_dt = None

    # End AUM (latest) — all metric math goes through guard functions
    end_aum_series = df["end_aum"] if "end_aum" in cols else pd.Series(dtype=float)
    end_aum_latest = apply_metric_guards(_get(end_aum_series))

    # MoM growth: safe_divide only; no raw division
    if latest_idx >= 1 and "end_aum" in cols:
        end_aum_prev = _safe_float(df["end_aum"].iloc[latest_idx - 1])
        mom_growth = safe_divide(end_aum_latest - end_aum_prev, end_aum_prev)
        mom_growth = apply_metric_guards(mom_growth)
    else:
        mom_growth = np.nan

    # YTD growth: safe_divide only
    ytd_growth = np.nan
    if latest_dt is not None and "end_aum" in cols:
        try:
            year_start = pd.Timestamp(year=latest_dt.year, month=1, day=1)
            year_mask = pd.to_datetime(df["month_end"], errors="coerce") >= year_start
            year_rows = df.loc[year_mask]
            if not year_rows.empty:
                first_year_idx = year_rows.index[0]
                end_aum_year_start = _safe_float(df.loc[first_year_idx, "end_aum"])
                ytd_growth = safe_divide(end_aum_latest - end_aum_year_start, end_aum_year_start)
                ytd_growth = apply_metric_guards(ytd_growth)
        except Exception:
            pass

    # Latest row metrics — all passed through guards
    nnb = apply_metric_guards(_get(df["nnb"], np.nan) if "nnb" in cols else np.nan)
    nnf = apply_metric_guards(_get(df["nnf"], np.nan) if "nnf" in cols else np.nan)
    ogr = apply_metric_guards(_get(df["ogr"], np.nan) if "ogr" in cols else np.nan)
    market_impact = apply_metric_guards(_get(df["market_impact_rate"], np.nan) if "market_impact_rate" in cols else np.nan)
    market_pnl = apply_metric_guards(_get(df["market_pnl"], np.nan) if "market_pnl" in cols else np.nan)

    # Final outputs through guards for deterministic behavior
    return ExecutiveKPIs(
        end_aum=apply_metric_guards(end_aum_latest),
        mom_growth=apply_metric_guards(mom_growth),
        ytd_growth=apply_metric_guards(ytd_growth),
        nnb=apply_metric_guards(nnb),
        nnf=apply_metric_guards(nnf),
        ogr=apply_metric_guards(ogr),
        market_impact=apply_metric_guards(market_impact),
        market_pnl=apply_metric_guards(market_pnl),
    )
