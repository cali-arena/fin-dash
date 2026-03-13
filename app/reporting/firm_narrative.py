"""
Deterministic executive narrative from firm_snapshot payload.

This module generates deterministic executive narrative based strictly on
firm_snapshot facts; no LLM. Safe to import in isolation (no Streamlit, no DuckDB).
"""
from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

# OGR narrative thresholds (decimal: 0.02 = 2%, 0.005 = 0.5%)
OGR_STRONG = 0.02
OGR_MODERATE = 0.005
# NNF: include sentence only if |NNF| >= this or >= 1e-6 * end_aum (when end_aum valid)
NNF_MIN_ABSOLUTE = 1000.0


def is_valid_number(x: Any) -> bool:
    """Return False for None, NaN, inf, -inf; True for finite numeric values."""
    if x is None:
        return False
    try:
        v = float(x)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def fmt_currency(x: Any) -> str:
    """Format as currency (K/M/B). Same as dashboard KPI formatter."""
    from app.ui.formatters import fmt_currency_kpi
    return fmt_currency_kpi(float(x) if is_valid_number(x) else None, decimals=2)


def fmt_percent(x: Any) -> str:
    """Format as percent (decimal in). Uses app.ui.formatters."""
    from app.ui.formatters import fmt_percent as _fmt_percent
    return _fmt_percent(float(x) if is_valid_number(x) else None, decimals=2, signed=False)


def fmt_month_label(iso_date_str: Any) -> str:
    """
    Convert ISO date string (e.g. "2026-03-31") to "Mar 2026".
    If missing or invalid return "latest month".
    """
    if iso_date_str is None or not isinstance(iso_date_str, str):
        return "latest month"
    s = (iso_date_str or "").strip()
    if not s or len(s) < 10:
        return "latest month"
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.strftime("%b %Y")
    except (ValueError, TypeError):
        return "latest month"


def sanitize_sentence(s: str) -> str:
    """
    Replace accidental double spaces with single space and strip trailing/leading spaces.

    >>> sanitize_sentence("  A  B   C  ")
    'A B C'
    """
    if not isinstance(s, str):
        return ""
    return re.sub(r"  +", " ", s.strip())


# Tokens that must not appear in review-safe narrative (case-sensitive substrings)
_LEAK_TOKENS = ("nan", "NaN", "inf", "Infinity", "None")
_FALLBACK_SENTENCE = "Certain metrics were not available for the selected period."


def assert_no_leaks(sentences: list[str]) -> list[str]:
    """
    Deterministic check: no sentence may contain 'nan', 'NaN', 'inf', 'Infinity', 'None'.
    Any sentence containing a leak is replaced with a safe fallback.

    >>> assert_no_leaks(["Good.", "Bad: nan value"])
    ['Good.', 'Certain metrics were not available for the selected period.']
    """
    result: list[str] = []
    for sent in sentences:
        if not isinstance(sent, str):
            result.append(_FALLBACK_SENTENCE)
            continue
        if any(token in sent for token in _LEAK_TOKENS):
            result.append(_FALLBACK_SENTENCE)
        else:
            result.append(sent)
    return result


def _get_raw(payload: dict[str, Any], key: str) -> Any:
    """Safe read from payload["raw"]; no KPI computation."""
    raw = payload.get("raw") or {}
    return raw.get(key)


def build_firm_narrative(snapshot_payload: dict[str, Any]) -> list[str]:
    """
    Build 3–6 bullet-ready sentences from firm_snapshot payload.
    Deterministic and review-safe; uses only snapshot facts (no LLM).
    Uses only payload["raw"] and payload["context"]; never computes KPIs.
    All numeric insertions go through fmt_currency / fmt_percent.
    """
    if not snapshot_payload:
        return []
    raw = snapshot_payload.get("raw") or {}
    context = snapshot_payload.get("context") or {}
    latest_label = fmt_month_label(context.get("latest_month_end"))
    ytd_start_label = fmt_month_label(context.get("ytd_start_month_end"))
    out: list[str] = []

    # 1) End AUM in latest month
    end_aum_str = fmt_currency(_get_raw(snapshot_payload, "end_aum"))
    out.append(f"End AUM closed at {end_aum_str} in {latest_label}.")

    # 2) Month-over-month AUM
    mom = _get_raw(snapshot_payload, "mom_growth")
    if not is_valid_number(mom):
        out.append("Month-over-month change was not available.")
    else:
        mom_f = float(mom)
        if mom_f > 0:
            out.append(f"Month-over-month AUM rose by {fmt_percent(mom)}.")
        elif mom_f < 0:
            out.append(f"Month-over-month AUM declined by {fmt_percent(mom)}.")
        else:
            out.append("Month-over-month AUM was flat.")

    # 3) MoM drivers: NNB + market impact
    nnb_valid = is_valid_number(_get_raw(snapshot_payload, "nnb"))
    mi_valid = is_valid_number(_get_raw(snapshot_payload, "market_impact"))
    if not nnb_valid and not mi_valid:
        out.append("Drivers were not available for the selected period.")
    elif nnb_valid and mi_valid:
        mi_f = float(_get_raw(snapshot_payload, "market_impact"))
        tailwind_headwind = "market tailwind" if mi_f > 0 else "market headwind"
        out.append(
            f"MoM movement was driven by NNB of {fmt_currency(_get_raw(snapshot_payload, 'nnb'))} "
            f"and market impact of {fmt_percent(_get_raw(snapshot_payload, 'market_impact'))} ({tailwind_headwind})."
        )
    elif nnb_valid:
        out.append(
            f"MoM movement was driven by NNB of {fmt_currency(_get_raw(snapshot_payload, 'nnb'))}; "
            "market impact was not available."
        )
    else:
        mi_f = float(_get_raw(snapshot_payload, "market_impact"))
        tailwind_headwind = "market tailwind" if mi_f > 0 else "market headwind"
        out.append(
            f"MoM movement was driven by market impact of {fmt_percent(_get_raw(snapshot_payload, 'market_impact'))} "
            f"({tailwind_headwind}); NNB was not available."
        )

    # 4) Year-to-date growth
    ytd = _get_raw(snapshot_payload, "ytd_growth")
    if not is_valid_number(ytd):
        out.append("Year-to-date growth was not available for the selected range.")
    else:
        out.append(f"Year-to-date growth stands at {fmt_percent(ytd)} versus {ytd_start_label}.")

    # 5) OGR
    ogr = _get_raw(snapshot_payload, "ogr")
    if not is_valid_number(ogr):
        out.append("OGR was not available.")
    else:
        ogr_f = float(ogr)
        if ogr_f >= OGR_STRONG:
            flow_label = "strong net inflows"
        elif ogr_f >= OGR_MODERATE:
            flow_label = "moderate inflows"
        else:
            flow_label = "soft inflows"
        out.append(f"OGR is {fmt_percent(ogr)}, indicating {flow_label}.")

    # Optional 6: NNF if valid and materially non-zero
    nnf = _get_raw(snapshot_payload, "nnf")
    end_aum_val = _get_raw(snapshot_payload, "end_aum")
    if is_valid_number(nnf):
        nnf_f = float(nnf)
        threshold = NNF_MIN_ABSOLUTE
        if is_valid_number(end_aum_val):
            threshold = max(threshold, 1e-6 * float(end_aum_val))
        if abs(nnf_f) >= threshold:
            out.append(f"NNF closed at {fmt_currency(nnf)} for {latest_label}.")

    # QA: sanitize and ensure no invalid tokens leak
    out = [sanitize_sentence(s) for s in out]
    return assert_no_leaks(out)


def build_firm_narrative_text(snapshot_payload: dict[str, Any]) -> str:
    """
    Single string with bullet formatting for the firm narrative.
    Uses build_firm_narrative() internally; output format: "• sentence1\\n• sentence2\\n..."

    Example structure (content depends on payload):
        • End AUM closed at $12.3B in Mar 2026.
        • Month-over-month AUM rose by 0.50%.
        • MoM movement was driven by NNB of $100.0M and market impact of 0.20% (market tailwind).
        • Year-to-date growth stands at 3.25% versus Jan 2026.
        • OGR is 1.20%, indicating moderate inflows.
    """
    sentences = build_firm_narrative(snapshot_payload)
    if not sentences:
        return ""
    return "• " + "\n• ".join(sentences)
