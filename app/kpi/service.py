"""
Single canonical KPI service: one path for End AUM, NNB, NNF, OGR, Market Movement.
Uses metric_contract formulas; one prior-period strategy; explicit handling for missing prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.kpi.contract import PERIOD_1M, PERIOD_ALLOWED, PERIOD_QOQ, PERIOD_YTD, PERIOD_YOY
from app.metrics.metric_contract import compute_fee_yield, compute_market_impact, compute_ogr, coerce_num

NAN = float("nan")


@dataclass
class KPIResult:
    """Canonical top-level KPI outputs plus validation metadata."""
    end_aum: float
    nnb: float
    nnf: float
    ogr: float
    market_movement: float
    begin_aum: float
    prior_period_used: bool
    scope_label: str
    scope_mode: str  # "firm" | "slice"
    period: str
    month_end: Any
    validation: dict[str, Any] = field(default_factory=dict)


def apply_period_canonical(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Canonical period filter: 1M, QoQ, YTD, YoY. Returns rows in the selected window."""
    if df.empty or "month_end" not in df.columns:
        return df
    work = df.copy()
    work["month_end"] = pd.to_datetime(work["month_end"], errors="coerce")
    work = work.dropna(subset=["month_end"]).sort_values("month_end")
    last_dt = work["month_end"].max()
    if pd.isna(last_dt):
        return work
    if period == PERIOD_1M:
        return work[work["month_end"] == last_dt]
    if period == PERIOD_QOQ:
        return work[work["month_end"] >= (last_dt - pd.DateOffset(months=2))]
    if period == PERIOD_YTD:
        ystart = pd.Timestamp(year=int(last_dt.year), month=1, day=1)
        return work[work["month_end"] >= ystart]
    if period == PERIOD_YOY:
        return work[work["month_end"] >= (last_dt - pd.DateOffset(months=11))]
    return work


# Threshold: if |NNB|/|NNF| or |NNF|/|NNB| exceeds this, flag possible unit mismatch (e.g. NNF in thousands, NNB in same unit as AUM).
NNF_NNB_RATIO_THRESHOLD = 100.0


def _add_nnf_nnb_unit_check(nnb: float, nnf: float, validation: dict[str, Any]) -> None:
    """
    If NNB and NNF are both non-zero but differ by more than NNF_NNB_RATIO_THRESHOLD,
    set validation["unit_consistency"] = "possible_mismatch" and add a warning.
    Does not change any KPI value; display remains as computed from source.
    """
    if nnb != nnb or nnf != nnf:
        validation["unit_consistency"] = "ok"
        return
    a_nnb, a_nnf = abs(nnb), abs(nnf)
    if a_nnf < 1e-12 and a_nnb < 1e-12:
        validation["unit_consistency"] = "ok"
        return
    if a_nnf < 1e-12:
        validation["unit_consistency"] = "ok"
        return
    ratio = a_nnb / a_nnf
    if ratio > NNF_NNB_RATIO_THRESHOLD or ratio < (1.0 / NNF_NNB_RATIO_THRESHOLD):
        validation["unit_consistency"] = "possible_mismatch"
        validation["nnb_nnf_ratio"] = round(ratio, 4)
        validation["warnings"].append(
            "NNF and NNB differ by more than 100x in this period. "
            "If your source data uses the same unit for both, check ETL/source for a scaling error; "
            "otherwise the displayed values are correct."
        )
    else:
        validation["unit_consistency"] = "ok"


def _resolve_prior_period_begin_aum(monthly: pd.DataFrame, last_row_index: int) -> tuple[float, bool]:
    """
    Canonical prior-period lookup: begin_aum for the latest row = previous row's end_aum.
    Returns (begin_aum, prior_period_used). If no previous row, returns (NaN, False).
    """
    if monthly.empty or last_row_index < 0:
        return NAN, False
    if last_row_index == 0:
        return NAN, False
    prev = monthly.iloc[last_row_index - 1]
    prior_end = prev.get("end_aum")
    begin = coerce_num(prior_end)
    return begin, not (begin != begin)  # prior_period_used = not NaN


def compute_kpi(
    monthly_df: pd.DataFrame,
    period: str,
    scope_label: str,
    scope_mode: str = "slice",
) -> KPIResult:
    """
    Single canonical function for top-level KPI computation.
    - monthly_df: aggregated by month_end with columns begin_aum, end_aum, nnb, nnf (and optionally market_impact, ogr).
    - period: one of 1M, QoQ, YTD, YoY.
    - scope_label: human-readable label (e.g. "Firm-wide" or "Selected slice (Channel: X)").
    - scope_mode: "firm" | "slice" for consistency checks.
    Returns KPIResult with end_aum, nnb, nnf, ogr, market_movement, begin_aum, prior_period_used, and validation.
    """
    validation: dict[str, Any] = {"inputs": {}, "prior_period": {}, "outputs": {}, "warnings": []}
    period = period if period in PERIOD_ALLOWED else PERIOD_YTD

    if monthly_df.empty:
        return KPIResult(
            end_aum=NAN,
            nnb=NAN,
            nnf=NAN,
            ogr=NAN,
            market_movement=NAN,
            begin_aum=NAN,
            prior_period_used=False,
            scope_label=scope_label,
            scope_mode=scope_mode,
            period=period,
            month_end=None,
            validation={
                **validation,
                "warnings": ["No monthly data provided for KPI computation."],
            },
        )

    # Ensure numeric columns
    work = monthly_df.copy()
    for col in ("begin_aum", "end_aum", "nnb", "nnf"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values("month_end").reset_index(drop=True)

    # Apply period filter (canonical)
    window = apply_period_canonical(work, period)
    if window.empty:
        return KPIResult(
            end_aum=NAN,
            nnb=NAN,
            nnf=NAN,
            ogr=NAN,
            market_movement=NAN,
            begin_aum=NAN,
            prior_period_used=False,
            scope_label=scope_label,
            scope_mode=scope_mode,
            period=period,
            month_end=None,
            validation={**validation, "warnings": ["Period filter returned no rows."]},
        )

    # Latest row: last row in window (canonical "latest" for KPIs)
    last_dt = window["month_end"].max()
    last_in_window = window[window["month_end"] == last_dt].iloc[-1]
    # Ordinal position of that row in full sorted work (for prior-period lookup)
    work = work.reset_index(drop=True)
    mask = work["month_end"] == last_dt
    if not mask.any():
        last_pos = -1
    else:
        last_pos = int(work.index[mask][-1])

    end_aum = coerce_num(last_in_window.get("end_aum"))
    nnb = coerce_num(last_in_window.get("nnb"))
    nnf = coerce_num(last_in_window.get("nnf"))
    begin_aum_from_row = coerce_num(last_in_window.get("begin_aum"))

    # Canonical prior-period: use previous month's end_aum if current begin_aum is missing/invalid
    if begin_aum_from_row != begin_aum_from_row or (begin_aum_from_row == 0 and end_aum == end_aum):
        begin_aum, prior_period_used = _resolve_prior_period_begin_aum(work, last_pos)
        if prior_period_used and begin_aum != begin_aum:
            prior_period_used = False
    else:
        begin_aum = begin_aum_from_row
        prior_period_used = True

    validation["inputs"] = {
        "month_end": str(last_dt) if last_dt is not None else None,
        "begin_aum_raw": float(begin_aum_from_row) if begin_aum_from_row == begin_aum_from_row else None,
        "end_aum": float(end_aum) if end_aum == end_aum else None,
        "nnb": float(nnb) if nnb == nnb else None,
        "nnf": float(nnf) if nnf == nnf else None,
    }
    validation["pipeline"] = (
        "Single path: app.kpi.service.compute_kpi. Source: monthly_df (sum by month_end); "
        "period filter; latest row in window. Columns: begin_aum, end_aum, nnb, nnf. No scaling applied."
    )
    validation["prior_period"] = {
        "begin_aum_used": float(begin_aum) if begin_aum == begin_aum else None,
        "prior_period_used": prior_period_used,
    }

    if not prior_period_used and (begin_aum != begin_aum or begin_aum == 0):
        validation["warnings"].append("Prior-period AUM missing; Market Movement and OGR may be NaN.")

    # Canonical formulas (metric_contract only)
    market_movement = compute_market_impact(begin_aum, end_aum, nnb)
    ogr = compute_ogr(nnb, begin_aum)

    # Guard: avoid reporting 0 when it was a fallback bug (we use NaN for missing)
    if market_movement == 0.0 and (begin_aum != begin_aum or nnb != nnb or end_aum != end_aum):
        market_movement = NAN
        validation["warnings"].append("Market Movement set to NaN due to missing inputs (avoid 0 fallback).")

    validation["outputs"] = {
        "end_aum": float(end_aum) if end_aum == end_aum else None,
        "nnb": float(nnb) if nnb == nnb else None,
        "nnf": float(nnf) if nnf == nnf else None,
        "ogr": float(ogr) if ogr == ogr else None,
        "market_movement": float(market_movement) if market_movement == market_movement else None,
    }

    # Window-level source totals (for audit: raw values in selected period)
    window_sum_nnb = float(window["nnb"].sum()) if "nnb" in window.columns else None
    window_sum_nnf = float(window["nnf"].sum()) if "nnf" in window.columns else None
    validation["source_summary"] = {
        "window_row_count": int(len(window)),
        "window_sum_nnb": window_sum_nnb,
        "window_sum_nnf": window_sum_nnf,
        "latest_row_nnb": float(nnb) if nnb == nnb else None,
        "latest_row_nnf": float(nnf) if nnf == nnf else None,
    }

    # Unit consistency: NNF and NNB should be in the same unit (e.g. dollars). If ratio is extreme, flag for review.
    _add_nnf_nnb_unit_check(nnb, nnf, validation)

    # Fee yield consistency: implied fee yield from NNF/avg_aum (for audit; does not change displayed KPIs)
    if end_aum == end_aum and begin_aum == begin_aum and nnf == nnf and nnb == nnb:
        fee_yield_implied = compute_fee_yield(nnf, begin_aum, end_aum, nnb=nnb)
        validation["outputs"]["fee_yield_implied"] = float(fee_yield_implied) if fee_yield_implied == fee_yield_implied else None
        last_fee = coerce_num(last_in_window.get("fee_yield"))
        if last_fee == last_fee and fee_yield_implied == fee_yield_implied and abs(fee_yield_implied - last_fee) > 1e-6:
            validation["fee_yield_consistency"] = "mismatch"
            validation["warnings"].append(
                "Fee yield from row differs from NNF/avg AUM; verify source or derived column."
            )
        else:
            validation["fee_yield_consistency"] = "ok"

    # Reconciliation check: begin + nnb + market should equal end_aum
    if begin_aum == begin_aum and nnb == nnb and market_movement == market_movement and end_aum == end_aum:
        recon = begin_aum + nnb + market_movement - end_aum
        validation["reconciliation_variance"] = float(recon)
        if abs(recon) > 1.0:
            validation["warnings"].append(f"KPI reconciliation variance: {recon:.4f} (begin+nnb+market-end).")

    return KPIResult(
        end_aum=end_aum if end_aum == end_aum else NAN,
        nnb=nnb if nnb == nnb else NAN,
        nnf=nnf if nnf == nnf else NAN,
        ogr=ogr if ogr == ogr else NAN,
        market_movement=market_movement if market_movement == market_movement else NAN,
        begin_aum=begin_aum if begin_aum == begin_aum else NAN,
        prior_period_used=prior_period_used,
        scope_label=scope_label,
        scope_mode=scope_mode,
        period=period,
        month_end=last_dt,
        validation=validation,
    )


def validate_kpi_against_latest_row(
    monthly_df: pd.DataFrame,
    kpi_result: KPIResult,
    *,
    tol_currency: float = 1e-2,
    tol_rate: float = 1e-8,
) -> dict[str, Any]:
    """
    Cross-check canonical KPI output against the latest row in the same monthly input.
    Used for parity diagnostics only (no recalculation side-effects).
    """
    out: dict[str, Any] = {
        "has_data": False,
        "latest_month_end": None,
        "latest_row": {},
        "kpi": {},
        "checks": {},
        "warnings": [],
    }
    if monthly_df is None or not isinstance(monthly_df, pd.DataFrame) or monthly_df.empty:
        out["warnings"].append("No monthly data available for KPI parity validation.")
        return out

    work = monthly_df.copy().sort_values("month_end")
    latest = work.iloc[-1]
    out["has_data"] = True
    out["latest_month_end"] = str(latest.get("month_end"))

    latest_end = coerce_num(latest.get("end_aum"))
    latest_nnb = coerce_num(latest.get("nnb"))
    latest_nnf = coerce_num(latest.get("nnf"))
    latest_ogr = coerce_num(latest.get("ogr"))
    latest_market = coerce_num(latest.get("market_impact"))

    out["latest_row"] = {
        "end_aum": None if latest_end != latest_end else float(latest_end),
        "nnb": None if latest_nnb != latest_nnb else float(latest_nnb),
        "nnf": None if latest_nnf != latest_nnf else float(latest_nnf),
        "ogr": None if latest_ogr != latest_ogr else float(latest_ogr),
        "market_movement": None if latest_market != latest_market else float(latest_market),
    }
    out["kpi"] = {
        "end_aum": None if kpi_result.end_aum != kpi_result.end_aum else float(kpi_result.end_aum),
        "nnb": None if kpi_result.nnb != kpi_result.nnb else float(kpi_result.nnb),
        "nnf": None if kpi_result.nnf != kpi_result.nnf else float(kpi_result.nnf),
        "ogr": None if kpi_result.ogr != kpi_result.ogr else float(kpi_result.ogr),
        "market_movement": None if kpi_result.market_movement != kpi_result.market_movement else float(kpi_result.market_movement),
    }

    def _close(a: float, b: float, tol: float) -> bool:
        if a != a or b != b:
            return False
        return abs(float(a) - float(b)) <= tol

    end_match = _close(latest_end, kpi_result.end_aum, tol_currency)
    nnb_match = _close(latest_nnb, kpi_result.nnb, tol_currency)
    nnf_match = _close(latest_nnf, kpi_result.nnf, tol_currency)
    ogr_match = _close(latest_ogr, kpi_result.ogr, tol_rate)
    market_match = _close(latest_market, kpi_result.market_movement, tol_currency)

    out["checks"] = {
        "end_aum_match": end_match,
        "nnb_match": nnb_match,
        "nnf_match": nnf_match,
        "ogr_match": ogr_match,
        "market_movement_match": market_match,
    }

    # Core divergence symptom observed previously: End AUM matched while NNB/OGR/Market diverged.
    if end_match and (not nnb_match or not ogr_match or not market_match):
        out["warnings"].append(
            "End AUM matches latest row but at least one dependent KPI (NNB/OGR/Market Movement) differs."
        )

    return out
