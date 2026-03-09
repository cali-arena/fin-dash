"""
Deterministic reconciliation checks from ReportPack only. No metric recomputation.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

# Tolerances (deterministic, traceable)
AUM_WATERFALL_TOLERANCE = 1.0   # abs(end - (begin + nnb + mi)) <= this
SHARES_SUM_TOLERANCE = 0.01     # abs(sum(share) - 1.0) <= this
TS_SNAPSHOT_TOLERANCE = 0.0     # exact match for same-month end_aum


def run_reconciliation(pack: Any) -> list[dict[str, Any]]:
    """
    Run light checks from pack data only. Returns list of
    {"check", "status": "PASS"|"FAIL"|"SKIP", "value", "tolerance", "reason"?}.
    """
    out: list[dict[str, Any]] = []

    # --- AUM waterfall: end_aum ≈ begin_aum + nnb + market_impact_abs ---
    snap = getattr(pack, "firm_snapshot", None)
    if snap is None or (hasattr(snap, "empty") and snap.empty):
        out.append({
            "check": "AUM_WATERFALL",
            "status": "SKIP",
            "value": None,
            "tolerance": AUM_WATERFALL_TOLERANCE,
            "reason": "firm_snapshot empty or missing",
        })
    else:
        row = snap.iloc[0]
        for col in ("begin_aum", "end_aum", "nnb", "market_impact_abs"):
            if col not in snap.columns:
                out.append({
                    "check": "AUM_WATERFALL",
                    "status": "SKIP",
                    "value": None,
                    "tolerance": AUM_WATERFALL_TOLERANCE,
                    "reason": f"missing column {col}",
                })
                break
        else:
            begin = _num(row.get("begin_aum"))
            end = _num(row.get("end_aum"))
            nnb = _num(row.get("nnb"))
            mi = _num(row.get("market_impact_abs"))
            if begin is None and end is None:
                out.append({
                    "check": "AUM_WATERFALL",
                    "status": "SKIP",
                    "value": None,
                    "tolerance": AUM_WATERFALL_TOLERANCE,
                    "reason": "missing numeric values",
                })
            else:
                implied = (begin or 0) + (nnb or 0) + (mi or 0)
                diff = abs((end or 0) - implied)
                status = "PASS" if diff <= AUM_WATERFALL_TOLERANCE else "FAIL"
                out.append({
                    "check": "AUM_WATERFALL",
                    "status": status,
                    "value": round(diff, 4),
                    "tolerance": AUM_WATERFALL_TOLERANCE,
                    "reason": None if status == "PASS" else f"|end - (begin+nnb+mi)| = {diff:.4f}",
                })

    # --- Shares sum: sum(aum_share) or sum(share) ≈ 1.0 ---
    rank = getattr(pack, "channel_rank", None)
    if rank is None or (hasattr(rank, "empty") and rank.empty):
        out.append({
            "check": "SHARES_SUM",
            "status": "SKIP",
            "value": None,
            "tolerance": SHARES_SUM_TOLERANCE,
            "reason": "channel_rank empty or missing",
        })
    else:
        share_col = "aum_share" if "aum_share" in rank.columns else ("share" if "share" in rank.columns else None)
        if share_col is None:
            out.append({
                "check": "SHARES_SUM",
                "status": "SKIP",
                "value": None,
                "tolerance": SHARES_SUM_TOLERANCE,
                "reason": "no aum_share or share column",
            })
        else:
            total = rank[share_col].sum()
            t = _num(total)
            if t is None:
                out.append({
                    "check": "SHARES_SUM",
                    "status": "SKIP",
                    "value": None,
                    "tolerance": SHARES_SUM_TOLERANCE,
                    "reason": "share column non-numeric or all NaN",
                })
            else:
                diff = abs(t - 1.0)
                status = "PASS" if diff <= SHARES_SUM_TOLERANCE else "FAIL"
                out.append({
                    "check": "SHARES_SUM",
                    "status": status,
                    "value": round(t, 6),
                    "tolerance": SHARES_SUM_TOLERANCE,
                    "reason": None if status == "PASS" else f"sum({share_col}) = {t:.6f}",
                })

    # --- time_series last month end_aum == firm_snapshot end_aum ---
    ts = getattr(pack, "time_series", None)
    if ts is None or (hasattr(ts, "empty") and ts.empty) or snap is None or (hasattr(snap, "empty") and snap.empty):
        out.append({
            "check": "TS_SNAPSHOT_MATCH",
            "status": "SKIP",
            "value": None,
            "tolerance": TS_SNAPSHOT_TOLERANCE,
            "reason": "time_series or firm_snapshot empty",
        })
    elif "end_aum" not in ts.columns or "month_end" not in ts.columns:
        out.append({
            "check": "TS_SNAPSHOT_MATCH",
            "status": "SKIP",
            "value": None,
            "tolerance": TS_SNAPSHOT_TOLERANCE,
            "reason": "time_series missing end_aum or month_end",
        })
    else:
        ts_last = ts.sort_values("month_end", ascending=False).iloc[0]
        snap_end = _num(snap.iloc[0].get("end_aum"))
        ts_end = _num(ts_last.get("end_aum"))
        if snap_end is None or ts_end is None:
            out.append({
                "check": "TS_SNAPSHOT_MATCH",
                "status": "SKIP",
                "value": None,
                "tolerance": TS_SNAPSHOT_TOLERANCE,
                "reason": "missing end_aum values",
            })
        else:
            diff = abs(snap_end - ts_end)
            status = "PASS" if diff <= TS_SNAPSHOT_TOLERANCE else "FAIL"
            out.append({
                "check": "TS_SNAPSHOT_MATCH",
                "status": status,
                "value": round(diff, 4),
                "tolerance": TS_SNAPSHOT_TOLERANCE,
                "reason": None if status == "PASS" else f"|snapshot - ts_last| = {diff:.4f}",
            })

    return out


def _num(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and (x != x or not math.isfinite(x))):
        return None
    try:
        v = float(x)
        return v if v == v and math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
