"""
Metric QA guardrails: deterministic checks for NNB, NNF, AUM reconciliation, fee yield, and unit consistency.
No UI; returns structured results for KPI service and optional dev logging. All monetary values in same unit (e.g. dollars).
"""
from __future__ import annotations

import logging
from typing import Any

from app.metrics.metric_contract import coerce_num, compute_fee_yield, compute_market_impact

logger = logging.getLogger(__name__)

# Tolerances
RECON_TOLERANCE = 1.0  # absolute currency tolerance for begin+nnb+market=end
RECON_TOLERANCE_PCT = 1e-4  # relative: variance as share of end_aum
FEE_YIELD_TOLERANCE = 1e-6  # absolute for fee yield comparison
NNF_NNB_RATIO_THRESHOLD = 100.0  # flag when |NNB|/|NNF| or inverse > this


def _is_finite(x: float) -> bool:
    v = coerce_num(x)
    return v == v and abs(v) != float("inf")


def check_aum_reconciliation(
    begin_aum: float,
    end_aum: float,
    nnb: float,
    market_movement: float,
    *,
    tol_abs: float = RECON_TOLERANCE,
    tol_pct: float = RECON_TOLERANCE_PCT,
) -> dict[str, Any]:
    """
    Verify begin_aum + NNB + market_movement = end_aum (within tolerance).
    Returns dict: ok, variance, variance_pct, message.
    """
    out: dict[str, Any] = {"ok": True, "variance": None, "variance_pct": None, "message": None}
    if not all(_is_finite(x) for x in (begin_aum, end_aum, nnb, market_movement)):
        out["ok"] = False
        out["message"] = "AUM reconciliation skipped: one or more inputs missing or non-finite."
        return out
    recon = float(begin_aum) + float(nnb) + float(market_movement) - float(end_aum)
    out["variance"] = recon
    if abs(end_aum) >= 1e-12:
        out["variance_pct"] = recon / float(end_aum)
    if abs(recon) <= tol_abs:
        out["message"] = "Begin AUM + NNB + Market = End AUM (reconciled)."
        return out
    if tol_pct and abs(end_aum) >= 1e-12 and abs(recon / float(end_aum)) <= tol_pct:
        out["ok"] = True
        out["message"] = f"Reconciled within {tol_pct*100:.4f}% of End AUM."
        return out
    out["ok"] = False
    out["message"] = f"AUM reconciliation variance: {recon:.4f} (begin+nnb+market−end)."
    return out


def check_fee_yield_consistency(
    nnf: float,
    begin_aum: float,
    end_aum: float,
    nnb: float,
    fee_yield_expected: float | None = None,
    *,
    tol: float = FEE_YIELD_TOLERANCE,
) -> dict[str, Any]:
    """
    Fee yield = NNF / avg_aum (metric_contract). Optionally compare to fee_yield_expected (e.g. from row).
    Returns dict: ok, fee_yield_implied, fee_yield_expected, message.
    """
    out: dict[str, Any] = {"ok": True, "fee_yield_implied": None, "fee_yield_expected": fee_yield_expected, "message": None}
    implied = compute_fee_yield(nnf, begin_aum, end_aum, nnb=nnb)
    if implied != implied:
        out["ok"] = False
        out["message"] = "Fee yield could not be computed (NNF/avg AUM) for given inputs."
        return out
    out["fee_yield_implied"] = implied
    if fee_yield_expected is not None and _is_finite(fee_yield_expected):
        diff = abs(implied - float(fee_yield_expected))
        if diff > tol:
            out["ok"] = False
            out["message"] = f"Fee yield mismatch: implied {implied:.6f} vs expected {fee_yield_expected:.6f}."
        else:
            out["message"] = "Fee yield consistent with NNF and AUM."
    else:
        out["message"] = "Fee yield implied from NNF and avg AUM (no expected value to compare)."
    return out


def check_nnb_nnf_magnitude_ratio(
    nnb: float,
    nnf: float,
    threshold: float = NNF_NNB_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """
    Flag when NNB and NNF differ by more than threshold (e.g. 100x) — possible unit or scale mismatch.
    Returns dict: ok, ratio_nnb_nnf, unit_consistency, message.
    """
    out: dict[str, Any] = {"ok": True, "ratio_nnb_nnf": None, "unit_consistency": "ok", "message": None}
    a_nnb, a_nnf = abs(coerce_num(nnb)), abs(coerce_num(nnf))
    if a_nnf < 1e-12 and a_nnb < 1e-12:
        out["message"] = "NNB and NNF both zero or missing; no ratio check."
        return out
    if a_nnf < 1e-12:
        out["message"] = "NNF zero or missing; ratio check skipped."
        return out
    ratio = a_nnb / a_nnf
    out["ratio_nnb_nnf"] = round(ratio, 4)
    if ratio > threshold or ratio < (1.0 / threshold):
        out["ok"] = False
        out["unit_consistency"] = "possible_mismatch"
        out["message"] = (
            f"NNF and NNB differ by more than {threshold:.0f}× in this period. "
            "If source data uses the same unit for both, verify ETL/source scaling; otherwise values may be correct (e.g. NNF as fee revenue)."
        )
    else:
        out["message"] = "NNB and NNF magnitude ratio within expected range."
    return out


def run_metric_qa(
    snapshot: dict[str, Any] | Any,
    *,
    tol_recon_abs: float = RECON_TOLERANCE,
    tol_fee_yield: float = FEE_YIELD_TOLERANCE,
    nnf_nnb_threshold: float = NNF_NNB_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """
    Run all metric QA checks on a snapshot-like dict (begin_aum, end_aum, nnb, nnf, market_impact_abs or market_movement).
    For use in KPI service or dev/debug. Returns summary with checks, overall ok, and messages.
    """
    if hasattr(snapshot, "validation"):
        # KPIResult-like: use .begin_aum, .end_aum, .nnb, .nnf, .market_movement
        begin_aum = coerce_num(getattr(snapshot, "begin_aum", None))
        end_aum = coerce_num(getattr(snapshot, "end_aum", None))
        nnb = coerce_num(getattr(snapshot, "nnb", None))
        nnf = coerce_num(getattr(snapshot, "nnf", None))
        market = coerce_num(getattr(snapshot, "market_movement", None))
    elif isinstance(snapshot, dict):
        begin_aum = coerce_num(snapshot.get("begin_aum"))
        end_aum = coerce_num(snapshot.get("end_aum"))
        nnb = coerce_num(snapshot.get("nnb"))
        nnf = coerce_num(snapshot.get("nnf"))
        market = coerce_num(snapshot.get("market_impact_abs") or snapshot.get("market_movement"))
    else:
        return {"ok": False, "checks": {}, "messages": [], "error": "run_metric_qa requires dict or KPIResult-like object."}

    checks: dict[str, Any] = {}
    messages: list[str] = []
    all_ok = True

    recon = check_aum_reconciliation(begin_aum, end_aum, nnb, market, tol_abs=tol_recon_abs)
    checks["aum_reconciliation"] = recon
    if not recon["ok"]:
        all_ok = False
        messages.append(recon["message"] or "AUM reconciliation failed.")

    fee = check_fee_yield_consistency(nnf, begin_aum, end_aum, nnb, fee_yield_expected=snapshot.get("fee_yield") if isinstance(snapshot, dict) else None, tol=tol_fee_yield)
    checks["fee_yield_consistency"] = fee
    if not fee["ok"]:
        all_ok = False
        if fee["message"]:
            messages.append(fee["message"])

    ratio_check = check_nnb_nnf_magnitude_ratio(nnb, nnf, threshold=nnf_nnb_threshold)
    checks["nnb_nnf_magnitude"] = ratio_check
    if not ratio_check["ok"]:
        all_ok = False
        if ratio_check["message"]:
            messages.append(ratio_check["message"])

    return {"ok": all_ok, "checks": checks, "messages": messages}
