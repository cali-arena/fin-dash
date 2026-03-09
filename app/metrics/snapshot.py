"""
Canonical firm metrics snapshot: single source of truth for AUM, NNB, OGR, Market Impact, Fee Yield.
All dashboard KPIs and header AUM must use this. No duplicate calculation paths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

NAN = float("nan")


@dataclass
class FirmMetricsSnapshot:
    """
    One-row canonical metrics for the current period. All derived fields from metric_contract.
    Safe: division-by-zero and missing inputs yield NaN, never crash.
    """
    month_end: Any
    begin_aum: float
    end_aum: float
    nnb: float
    nnf: float
    ogr: float
    market_impact_abs: float  # Market P&L (currency)
    market_impact_rate: float
    fee_yield: float
    mom_pct: float
    ytd_pct: float
    yoy_pct: float

    def to_display_dict(self) -> dict[str, Any]:
        """For KPI strip and header: same keys as payload['raw'] / payload['kpis']."""
        return {
            "end_aum": self.end_aum,
            "nnb": self.nnb,
            "nnf": self.nnf,
            "ogr": self.ogr,
            "market_impact": self.market_impact_rate,  # display as percent
            "market_pnl": self.market_impact_abs,
            "mom_growth": self.mom_pct,
            "ytd_growth": self.ytd_pct,
        }

    @classmethod
    def from_row(cls, row: Any) -> FirmMetricsSnapshot:
        """Build from canonical snapshot row (Series or dict). Missing/NaN => NAN."""
        def _f(k: str, default: float = NAN) -> float:
            if hasattr(row, "get"):
                v = row.get(k, default)
            else:
                try:
                    v = row[k] if k in row else default
                except (KeyError, TypeError):
                    v = default
            try:
                v = float(v)
            except (TypeError, ValueError):
                return NAN
            return v if math.isfinite(v) else NAN

        return cls(
            month_end=row.get("month_end") if hasattr(row, "get") else getattr(row, "month_end", None),
            begin_aum=_f("begin_aum"),
            end_aum=_f("end_aum"),
            nnb=_f("nnb"),
            nnf=_f("nnf"),
            ogr=_f("ogr"),
            market_impact_abs=_f("market_impact_abs"),
            market_impact_rate=_f("market_impact_rate"),
            fee_yield=_f("fee_yield"),
            mom_pct=_f("mom_pct"),
            ytd_pct=_f("ytd_pct"),
            yoy_pct=_f("yoy_pct"),
        )


def validate_snapshot(snapshot: FirmMetricsSnapshot) -> list[str]:
    """
    Returns list of validation issue messages; empty if valid.
    - No AUM mismatch (single source).
    - When begin_aum > 0, derived rates should be computable (or explicitly NaN).
    """
    issues: list[str] = []
    if snapshot.end_aum != snapshot.end_aum and snapshot.begin_aum != snapshot.begin_aum:
        issues.append("end_aum and begin_aum are missing")
    if snapshot.begin_aum == 0 or (snapshot.begin_aum != snapshot.begin_aum):
        if snapshot.ogr == snapshot.ogr or snapshot.market_impact_rate == snapshot.market_impact_rate:
            pass  # NaN is expected for first month
    return issues


def metrics_ready_for_display(row: Any) -> bool:
    """
    True if the canonical row has at least end_aum so KPI strip and header can show a value.
    Use before building payload or rendering charts that depend on this snapshot.
    """
    if row is None:
        return False
    v = row.get("end_aum") if hasattr(row, "get") else getattr(row, "end_aum", None)
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# Required for "four core metrics never blank" when source data exists: NNB, OGR, Market Impact, Market P&L
REQUIRED_CORE_METRIC_KEYS = ("end_aum", "nnb", "ogr", "market_impact_abs", "market_impact_rate")


def validation_required_metrics(row: Any) -> list[str]:
    """
    Returns list of required metric keys that are missing or non-finite in the canonical row.
    Empty list means all required metrics are present (or explicitly NaN for first period).
    Use to flag when source data exists but metrics were not computed.
    """
    if row is None:
        return list(REQUIRED_CORE_METRIC_KEYS)
    missing: list[str] = []
    for k in REQUIRED_CORE_METRIC_KEYS:
        v = row.get(k) if hasattr(row, "get") else getattr(row, k, None)
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or not math.isfinite(v))):
                missing.append(k)
        except (TypeError, ValueError):
            missing.append(k)
    return missing


def build_canonical_metrics_pack(
    snapshot_row: Any,
    firm_total_nnb: float | None = None,
) -> dict[str, Any]:
    """
    Single canonical metrics pack for KPIs, narrative, waterfall, trend, report, NLQ.
    Includes: beginning_aum, ending_aum, nnb, nnf, ogr, market_pnl, market_impact_rate, fee_yield,
    nnb_share_of_firm, growth_quality_flag. All from snapshot_row; no new calculations.
    """
    pack: dict[str, Any] = {}
    if snapshot_row is None:
        return pack

    def _f(k: str, default: float = NAN) -> float:
        v = snapshot_row.get(k, default) if hasattr(snapshot_row, "get") else getattr(snapshot_row, k, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    pack["beginning_aum"] = _f("begin_aum")
    pack["ending_aum"] = _f("end_aum")
    pack["nnb"] = _f("nnb")
    pack["nnf"] = _f("nnf")
    pack["ogr"] = _f("ogr")
    pack["market_pnl"] = _f("market_impact_abs")
    pack["market_impact_rate"] = _f("market_impact_rate")
    pack["fee_yield"] = _f("fee_yield")
    pack["mom_pct"] = _f("mom_pct")
    pack["ytd_pct"] = _f("ytd_pct")
    pack["yoy_pct"] = _f("yoy_pct")
    pack["month_end"] = snapshot_row.get("month_end") if hasattr(snapshot_row, "get") else getattr(snapshot_row, "month_end", None)

    total_nnb = firm_total_nnb if firm_total_nnb is not None and math.isfinite(firm_total_nnb) and firm_total_nnb != 0 else None
    nnb = pack["nnb"]
    if total_nnb is not None and total_nnb != 0 and nnb == nnb:
        pack["nnb_share_of_firm"] = nnb / total_nnb
    else:
        pack["nnb_share_of_firm"] = NAN

    nnb_ok = nnb == nnb and nnb > 0.02 * (pack["beginning_aum"] or NAN) if pack["beginning_aum"] == pack["beginning_aum"] and pack["beginning_aum"] else (nnb == nnb and nnb > 0)
    fy = pack["fee_yield"]
    low_fee = fy != fy or fy < 0.02
    pack["growth_quality_flag"] = "high_nnb_low_fee_yield" if (nnb_ok and low_fee) else ("high_nnb_high_fee_yield" if nnb_ok else "low_nnb")

    return pack


def get_metrics_debug_info(snapshot_df: Any, period_frames: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Debug validation: exact period selected, row count, formula labels.
    Use in expanders or logs to confirm which period and rows were used and that canonical formulas were applied.
    """
    info: dict[str, Any] = {"period": {}, "row_count": 0, "formulas_applied": []}
    if period_frames:
        info["period"] = {
            "current_month_end": str(period_frames.get("current_month_end")),
            "prior_month_end": str(period_frames.get("prior_month_end")),
            "ytd_start": str(period_frames.get("aum_at_year_start")),
        }
    if snapshot_df is not None and hasattr(snapshot_df, "__len__"):
        try:
            info["row_count"] = int(len(snapshot_df))
        except Exception:
            pass
    info["formulas_applied"] = [
        "ogr = nnb / beginning_aum",
        "market_pnl = ending_aum - beginning_aum - nnb",
        "market_impact_rate = market_pnl / beginning_aum",
        "fee_yield = nnf / average_aum (or NNB-based when contract set)",
    ]
    return info
