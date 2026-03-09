"""
Reconciliation and validation layer for metric acceptance criteria.
Unit-test friendly; no Streamlit. Encodes:
- Waterfall totals reconcile within tolerance (configs/metric_contract.yml)
- No inf values survive (converted to NaN)
- Empty data → no_data payload for UI (no broken charts)
"""
from __future__ import annotations

import math
from typing import Any

from app.metrics.metric_contract import load_metric_contract

NAN = float("nan")


def reconcile_waterfall(
    begin_aum: float,
    end_aum: float,
    nnb: float,
    market_impact: float,
    tol_abs: float,
    tol_rel: float,
) -> dict[str, Any]:
    """
    Check Begin + NNB + MarketImpact ≈ End within tolerance.
    ok = abs(diff) <= max(tol_abs, tol_rel * max(1.0, abs(end))).
    """
    lhs = begin_aum + nnb + market_impact
    diff = end_aum - lhs
    scale = max(1.0, abs(end_aum))
    threshold = max(tol_abs, tol_rel * scale)
    ok = abs(diff) <= threshold
    return {"ok": ok, "diff": diff, "lhs": lhs, "end": end_aum}


def reconcile_waterfall_from_contract(
    begin_aum: float,
    end_aum: float,
    nnb: float,
    market_impact: float,
) -> dict[str, Any]:
    """Reconcile using tolerances from configs/metric_contract.yml."""
    contract = load_metric_contract()
    tolerances = (contract.get("tolerances") or {})
    tol_abs = float(tolerances.get("waterfall_abs", 1e-6))
    tol_rel = float(tolerances.get("waterfall_rel", 1e-9))
    return reconcile_waterfall(
        begin_aum, end_aum, nnb, market_impact, tol_abs, tol_rel
    )


def validate_no_nan_inf(
    df: Any,
    required_cols: list[str],
) -> dict[str, dict[str, int]]:
    """
    Count NaN and inf per column. Returns {col: {"nan": n, "inf": m}, ...}.
    Only considers required_cols that exist in df; missing cols get 0, 0.
    """
    try:
        import pandas as pd
    except ImportError:
        return {c: {"nan": 0, "inf": 0} for c in required_cols}

    if df is None or (hasattr(df, "empty") and df.empty):
        return {c: {"nan": 0, "inf": 0} for c in required_cols}

    out: dict[str, dict[str, int]] = {}
    for col in required_cols:
        if col not in df.columns:
            out[col] = {"nan": 0, "inf": 0}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(s.isna().sum())
        inf_count = int((s == float("inf")).sum() + (s == float("-inf")).sum())
        out[col] = {"nan": nan_count, "inf": inf_count}
    return out


def validate_empty_selection(df: Any) -> bool:
    """Return True if the dataframe (or selection) is empty."""
    if df is None:
        return True
    if hasattr(df, "empty"):
        return bool(df.empty)
    if hasattr(df, "__len__"):
        return len(df) == 0
    return True


def format_no_data_panel(reason: str) -> dict[str, Any]:
    """Payload for UI when there is no data; avoids broken charts."""
    return {"status": "no_data", "reason": reason}
