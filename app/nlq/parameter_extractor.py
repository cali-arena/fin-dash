"""
Parameter extractor for NLQ: extract dimensions, date range, and thresholds from natural language.
Rule-based (regex + keyword). No LLM; no raw data access.
Used to build filters and time range for deterministic queries.
"""
from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class ThresholdSpec:
    """Single threshold: above/below a value (currency or percent)."""
    op: str  # "gt" | "lt"
    value: float
    unit: str  # "currency" | "percent"


@dataclass
class ExtractedParams:
    """Structured parameters extracted from user question."""
    tickers: list[str] = field(default_factory=list)
    channel: list[str] = field(default_factory=list)
    segment: list[str] = field(default_factory=list)
    country: list[str] = field(default_factory=list)
    date_start: date | None = None
    date_end: date | None = None
    thresholds: list[ThresholdSpec] = field(default_factory=list)
    filters: dict[str, list[str]] = field(default_factory=dict)

    def to_filter_dict(self) -> dict[str, list[str]]:
        """Merge dimension extractions into a single filter dict for QuerySpec."""
        out: dict[str, list[str]] = dict(self.filters)
        if self.channel:
            out["channel"] = self.channel
        if self.segment:
            out["segment"] = self.segment
        if self.country:
            out["country"] = list(set(out.get("country", []) + self.country))
        if self.tickers:
            out["product_ticker"] = self.tickers
        return out


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# Regex for threshold: above/below/greater than/over/under + number + optional K/M/B/%
_THRESHOLD_PATTERN = re.compile(
    r"\b(above|greater than|over|below|less than|under)\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\s*([kmb%]?)\s*(?:$|[\s,.;:!?])",
    re.I,
)

# Month names for date extraction
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def extract_thresholds(query: str) -> list[ThresholdSpec]:
    """
    Extract threshold phrases like "above $100k", "below 0.5%".
    Returns list of ThresholdSpec (op, value, unit). Percent values in decimal (0.5% -> 0.005).
    """
    t = _normalize(query)
    out: list[ThresholdSpec] = []
    for m in _THRESHOLD_PATTERN.finditer(t):
        op_raw = (m.group(1) or "").strip().lower()
        op = "gt" if op_raw in ("above", "greater than", "over") else "lt"
        raw_val = float(m.group(2))
        suffix = (m.group(3) or "").lower()
        if suffix == "k":
            raw_val *= 1_000.0
            unit = "currency"
        elif suffix == "m":
            raw_val *= 1_000_000.0
            unit = "currency"
        elif suffix == "b":
            raw_val *= 1_000_000_000.0
            unit = "currency"
        elif suffix == "%":
            raw_val /= 100.0
            unit = "percent"
        else:
            unit = "currency"
        out.append(ThresholdSpec(op=op, value=raw_val, unit=unit))
    return out


def extract_date_range(query: str, today: date | None = None) -> tuple[date | None, date | None]:
    """
    Extract date range from query: Q1–Q4, month names, or implicit YTD/last N months.
    Returns (date_start, date_end) or (None, None).
    """
    today = today or date.today()
    t = _normalize(query)

    # Quarter: Q1, Q2, Q3, Q4
    q = re.search(r"\bq([1-4])\b", t)
    if q:
        qn = int(q.group(1))
        start_month = (qn - 1) * 3 + 1
        end_month = start_month + 2
        _, last_day = monthrange(today.year, end_month)
        return date(today.year, start_month, 1), date(today.year, end_month, last_day)

    # Single month by name
    for mname, month_no in _MONTH_NAMES.items():
        if re.search(rf"\b{mname}\b", t):
            year = today.year if month_no <= today.month else today.year - 1
            _, last_day = monthrange(year, month_no)
            return date(year, month_no, 1), date(year, month_no, last_day)

    # YTD / last N months (caller can map to concrete dates if needed)
    if "ytd" in t or "year to date" in t:
        return date(today.year, 1, 1), today
    if "last 12" in t or "last 12 months" in t:
        from datetime import timedelta
        end_d = today
        start_d = end_d - timedelta(days=365)
        return start_d, end_d
    if "last 6" in t:
        from datetime import timedelta
        end_d = today
        start_d = end_d - timedelta(days=180)
        return start_d, end_d
    if "last 3" in t or "q3" in t or "q4" in t:
        from datetime import timedelta
        end_d = today
        start_d = end_d - timedelta(days=90)
        return start_d, end_d

    return None, None


def extract_dimension_values(
    query: str,
    value_catalog: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Extract dimension filter values by matching query text to a catalog of allowed values.
    value_catalog: e.g. {"channel": {"Wealth", "Institutional"}, "product_ticker": {"AGG", "HYG"}}.
    Returns dict dimension -> list of matched values (max 5 per dimension).
    """
    catalog = value_catalog or {}
    t = _normalize(query)
    filters: dict[str, list[str]] = {}
    for dim in ("channel", "sub_channel", "product_ticker", "segment", "sub_segment", "country", "src_country"):
        vals = catalog.get(dim) or set()
        if not vals:
            continue
        matched: list[str] = []
        for v in sorted(vals):
            sv = str(v).strip()
            if not sv:
                continue
            if _normalize(sv) in t or (len(sv) >= 3 and sv.lower() in t):
                matched.append(sv)
        if matched:
            filters[dim] = matched[:5]
    return filters


def extract_parameters(
    query: str,
    value_catalog: dict[str, set[str]] | None = None,
    today: date | None = None,
) -> ExtractedParams:
    """
    Extract all parameters from a natural language question.
    Returns ExtractedParams (tickers, channel, segment, country, date range, thresholds, filters).
    """
    today = today or date.today()
    thresholds = extract_thresholds(query)
    date_start, date_end = extract_date_range(query, today)
    filters = extract_dimension_values(query, value_catalog)

    tickers: list[str] = list(filters.get("product_ticker", []))
    channel: list[str] = list(filters.get("channel", []))
    segment: list[str] = list(filters.get("segment", []))
    country: list[str] = list(filters.get("country", []) + filters.get("src_country", []))

    return ExtractedParams(
        tickers=tickers,
        channel=channel,
        segment=segment,
        country=country,
        date_start=date_start,
        date_end=date_end,
        thresholds=thresholds,
        filters={k: v for k, v in filters.items() if k not in ("channel", "segment", "country", "src_country", "product_ticker")},
    )
