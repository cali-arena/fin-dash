"""
Canonical metric computation enforcing configs/metric_contract.yml.
Pure Python; no Streamlit. Guards: division-by-zero => NaN, inf => NaN, None => NaN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# In-module cache for contract config
_contract_cache: dict[str, Any] | None = None

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs" / "metric_contract.yml"

NAN = float("nan")


def load_metric_contract() -> dict[str, Any]:
    """Load configs/metric_contract.yml. Cached in-module after first load."""
    global _contract_cache
    if _contract_cache is not None:
        return _contract_cache
    out: dict[str, Any] = {}
    if CONTRACT_PATH.exists():
        try:
            import yaml
            data = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out = data
        except Exception:
            pass
    _contract_cache = out
    return out


def coerce_num(x: Any) -> float:
    """None => NaN; inf/-inf => NaN; non-numeric => NaN; else float."""
    if x is None:
        return NAN
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NAN
    if math.isnan(v) or math.isinf(v):
        return NAN
    return v


def safe_divide(a: Any, b: Any) -> float:
    """Division with guards: division-by-zero => NaN; inf/None => NaN."""
    bf = coerce_num(b)
    if bf == 0 or math.isnan(bf):
        return NAN
    af = coerce_num(a)
    if math.isnan(af):
        return NAN
    result = af / bf
    return coerce_num(result)


def compute_market_impact(begin_aum: Any, end_aum: Any, nnb: Any) -> float:
    """Canonical: end_aum - begin_aum - nnb. Missing/invalid => NaN."""
    end = coerce_num(end_aum)
    begin = coerce_num(begin_aum)
    n = coerce_num(nnb)
    if math.isnan(end) or math.isnan(begin) or math.isnan(n):
        return NAN
    return coerce_num(end - begin - n)


def compute_ogr(nnb: Any, begin_aum: Any) -> float:
    """Canonical: nnb / begin_aum. Guard: begin_aum==0 => NaN."""
    return safe_divide(nnb, begin_aum)


def compute_market_impact_rate(market_impact: Any, begin_aum: Any) -> float:
    """Canonical: market_impact / begin_aum. Guard: begin_aum==0 => NaN."""
    return safe_divide(market_impact, begin_aum)


def _fee_yield_annualize_factor() -> float:
    """Read annualize_factor from contract; default 12."""
    contract = load_metric_contract()
    try:
        fy = (contract.get("formulas") or {}).get("fee_yield") or {}
        ann = fy.get("annualization") or {}
        return float(ann.get("annualize_factor", 12))
    except (TypeError, ValueError):
        return 12.0


def _fee_yield_annualization_mode() -> str:
    """Read annualization.mode from contract; 'monthly' or 'annualized'."""
    contract = load_metric_contract()
    try:
        fy = (contract.get("formulas") or {}).get("fee_yield") or {}
        ann = fy.get("annualization") or {}
        return str(ann.get("mode", "monthly")).strip().lower() or "monthly"
    except (TypeError, ValueError):
        return "monthly"


def compute_fee_yield(
    nnf: Any,
    begin_aum: Any,
    end_aum: Any,
    annualize: bool | None = None,
    nnb: Any = None,
) -> float:
    """
    Fee yield = nnf / average_aum, with average_aum = (begin_aum + end_aum) / 2.
    DATA SUMMARY policy: when NNB <= 0 or missing, return NaN (no fee yield).
    If annualize is None, use contract annualization.mode:
      monthly => base; annualized => base * annualize_factor.
    """
    if nnb is not None:
        n = coerce_num(nnb)
        if math.isnan(n) or n <= 0:
            return NAN
    begin = coerce_num(begin_aum)
    end = coerce_num(end_aum)
    avg = (begin + end) / 2.0
    if avg <= 0 or math.isnan(avg):
        return NAN
    base = safe_divide(nnf, avg)
    if math.isnan(base):
        return NAN
    if annualize is not None:
        if annualize:
            factor = _fee_yield_annualize_factor()
            return coerce_num(base * factor)
        return coerce_num(base)
    mode = _fee_yield_annualization_mode()
    if mode == "annualized":
        factor = _fee_yield_annualize_factor()
        return coerce_num(base * factor)
    return coerce_num(base)


def compute_fee_yield_nnf_nnb(nnf: Any, nnb: Any) -> float:
    """
    Client formula: Fee Yield = NNF / NNB.
    Returns NaN on divide-by-zero (NNB missing or <= 0). No annualization.
    """
    n = coerce_num(nnb)
    if math.isnan(n) or n <= 0:
        return NAN
    return safe_divide(nnf, nnb)


@dataclass
class MetricRow:
    """One row of metrics: inputs plus derived fields from canonical formulas."""

    month_end: Any
    begin_aum: float
    end_aum: float
    nnb: float
    nnf: float
    market_pnl: float
    market_impact: float
    ogr: float
    market_impact_rate: float
    fee_yield: float


def compute_metric_row(row: Mapping[str, Any]) -> MetricRow:
    """
    Compute derived fields from row using only canonical functions.
    Missing inputs => derived fields are NaN; does not crash.
    """
    month_end = row.get("month_end")
    begin_aum = coerce_num(row.get("begin_aum"))
    end_aum = coerce_num(row.get("end_aum"))
    nnb = coerce_num(row.get("nnb"))
    nnf = coerce_num(row.get("nnf"))
    market_pnl = coerce_num(row.get("market_pnl"))

    market_impact = compute_market_impact(begin_aum, end_aum, nnb)
    ogr = compute_ogr(nnb, begin_aum)
    market_impact_rate = compute_market_impact_rate(market_impact, begin_aum)
    fee_yield = compute_fee_yield(nnf, begin_aum, end_aum, annualize=None, nnb=nnb)

    return MetricRow(
        month_end=month_end,
        begin_aum=begin_aum,
        end_aum=end_aum,
        nnb=nnb,
        nnf=nnf,
        market_pnl=market_pnl,
        market_impact=market_impact,
        ogr=ogr,
        market_impact_rate=market_impact_rate,
        fee_yield=fee_yield,
    )
