"""
Deterministic NLQ parser. Returns QuerySpec (validated) or ParseError only.
This module NEVER executes SQL and NEVER reads raw user input into a query.
It only produces structured QuerySpec or ParseError. Extraction is regex + registry + catalog.
"""
from __future__ import annotations

import difflib
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from pydantic import ValidationError

from app.nlq.governance import GovernanceError, normalize_dim_token, validate_queryspec
from models.query_spec import ChartSpec, QuerySpec, SortSpec, TimeRange

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

TOP_K_SUGGESTIONS = 5


class ParseErrorCode(Enum):
    UNKNOWN_METRIC = "unknown_metric"
    UNKNOWN_DIMENSION = "unknown_dimension"
    DIM_NOT_ALLOWED = "dim_not_allowed"
    UNKNOWN_FILTER_VALUE = "unknown_filter_value"
    AMBIGUOUS = "ambiguous"
    INVALID_TIME_RANGE = "invalid_time_range"
    EMPTY_QUERY = "empty_query"


@dataclass(frozen=True)
class ParseError:
    code: ParseErrorCode
    message: str
    details: dict[str, Any]
    suggestions: dict[str, Any] | None = None


# Intent keywords (rule-first, no ML). Same input always yields same intent set.
INTENT_TREND_KEYWORDS = [
    "over time", "trend", "monthly", "by month", "since",
    "last 12 months", "ytd", "qtd", "rolling", "last 6 months",
    "last 3 months", "last 24 months",
]
INTENT_COMPARE_KEYWORDS = [
    "vs", "versus", "compare", "top", "bottom", "best", "worst", "rank", "movers",
]
INTENT_SNAPSHOT_KEYWORDS = [
    "latest", "current", "this month", "current month", "as of",
    "end of month", "today",
]
# Breakdown: pattern \bby\s+(channel|ticker|country|geo|segment|product)\b
INTENT_BREAKDOWN_PATTERN = re.compile(
    r"\bby\s+(channel|ticker|country|geo|segment|product|distribution|region|fund)\b",
    re.I,
)

# Tie-break precedence when multiple intents (highest first): trend > breakdown > compare > snapshot.
# Defaults (chart type, sort, limit) are chosen by the single effective intent.
INTENT_PRECEDENCE = ("trend", "breakdown", "compare", "snapshot")


def classify_intent(text: str) -> set[str]:
    """
    Rule-first intent classification from keywords. No ML.
    Returns a set of intent labels: trend, compare, breakdown, snapshot.
    Same input text always yields the same intent set.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    intents: set[str] = set()
    for phrase in INTENT_TREND_KEYWORDS:
        if phrase in normalized:
            intents.add("trend")
            break
    for phrase in INTENT_COMPARE_KEYWORDS:
        if phrase in normalized:
            intents.add("compare")
            break
    if INTENT_BREAKDOWN_PATTERN.search(normalized):
        intents.add("breakdown")
    for phrase in INTENT_SNAPSHOT_KEYWORDS:
        if phrase in normalized:
            intents.add("snapshot")
            break
    return intents


def _effective_intent(intents: set[str]) -> str | None:
    """Single intent for defaults: first in INTENT_PRECEDENCE that appears in intents."""
    for intent in INTENT_PRECEDENCE:
        if intent in intents:
            return intent
    return None


def _defaults_from_intent(
    effective: str | None,
    dimensions: list[str],
) -> tuple[str, SortSpec, int, str | None, str | None, str | None]:
    """Returns (chart_type, sort, limit, chart_x, chart_y, chart_series). Deterministic."""
    if effective == "trend":
        return (
            "line",
            SortSpec(by=None, order="desc"),
            50,
            "month_end",
            "metric",
            dimensions[0] if dimensions else None,
        )
    if effective == "breakdown":
        return (
            "bar",
            SortSpec(by="metric", order="desc"),
            50,
            dimensions[0] if dimensions else None,
            "metric",
            None,
        )
    if effective == "compare":
        return (
            "bar" if dimensions else "table",
            SortSpec(by="metric", order="desc"),
            50,
            dimensions[0] if dimensions else None,
            "metric",
            None,
        )
    if effective == "snapshot":
        return ("table", SortSpec(by="metric", order="desc"), 10, None, None, None)
    # No intent or unknown
    return ("table", SortSpec(by="metric", order="desc"), 50, None, None, None)


def _normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _build_metric_index(metric_reg: dict[str, Any]) -> tuple[dict[str, dict], dict[str, list[tuple[str, int]]]]:
    """
    Inverted index: phrase (lower) -> [(metric_id, phrase_len)].
    Uses metric_id, label tokens, and optional synonyms. Collisions stored as list.
    Returns (by_id, phrase_to_candidates).
    """
    metrics = metric_reg.get("metrics") or []
    if not isinstance(metrics, list):
        return {}, {}
    by_id: dict[str, dict] = {}
    phrase_to_candidates: dict[str, list[tuple[str, int]]] = {}
    for m in metrics:
        if not isinstance(m, dict):
            continue
        mid = m.get("metric_id")
        if mid is None:
            continue
        mid_lower = str(mid).strip().lower()
        by_id[mid_lower] = m
        phrases: list[str] = [mid_lower]
        label = m.get("label") or ""
        for word in re.split(r"[\s/()]+", str(label)):
            w = word.strip().lower()
            if w and len(w) > 1:
                phrases.append(w)
        for syn in m.get("synonyms") or []:
            if isinstance(syn, str) and syn.strip():
                phrases.append(syn.strip().lower())
        for p in phrases:
            if not p:
                continue
            entry = (mid_lower, len(p))
            if p not in phrase_to_candidates:
                phrase_to_candidates[p] = []
            if entry not in phrase_to_candidates[p]:
                phrase_to_candidates[p].append(entry)
    return by_id, phrase_to_candidates


def extract_metric(text: str, metric_reg: dict[str, Any]) -> str | ParseError:
    """
    Extract metric_id from text using registry index. Longer phrase matches win.
    Multiple metrics with same score -> AMBIGUOUS. No match -> UNKNOWN_METRIC with fuzzy suggestions.
    """
    normalized = _normalize_text(text)
    by_id, phrase_to_candidates = _build_metric_index(metric_reg)
    if not by_id:
        return ParseError(
            code=ParseErrorCode.UNKNOWN_METRIC,
            message="No metrics in registry",
            details={},
            suggestions=None,
        )
    matches: list[tuple[str, int, list[tuple[str, int]]]] = []
    for phrase, candidates in phrase_to_candidates.items():
        if phrase in normalized:
            plen = len(phrase)
            matches.append((phrase, plen, candidates))
    if not matches:
        best = difflib.get_close_matches(
            normalized[:60].replace(" ", "_"),
            list(by_id.keys()),
            n=5,
            cutoff=0.25,
        )
        if not best:
            best = sorted(by_id.keys())[:5]
        return ParseError(
            code=ParseErrorCode.UNKNOWN_METRIC,
            message=f"Unknown metric; suggested: {best}",
            details={"input": normalized},
            suggestions={"metrics": best},
        )
    matches.sort(key=lambda x: -x[1])
    best_len = matches[0][1]
    best_matches = [m for m in matches if m[1] == best_len]
    metric_ids: set[str] = set()
    for _, _, candidates in best_matches:
        for mid, _ in candidates:
            metric_ids.add(mid)
    if len(metric_ids) > 1:
        return ParseError(
            code=ParseErrorCode.AMBIGUOUS,
            message=f"Multiple metrics matched: {sorted(metric_ids)}",
            details={"input": normalized, "matched": sorted(metric_ids)},
            suggestions={"metrics": sorted(metric_ids)},
        )
    return next(iter(metric_ids))


# Plural/variant -> canonical dimension token for "by X" extraction
DIM_PLURAL_VARIANTS: list[tuple[str, str]] = [
    ("channels", "channel"),
    ("tickers", "product_ticker"),
    ("products", "product_ticker"),
    ("countries", "src_country"),
    ("geos", "src_country"),
    ("regions", "src_country"),
    ("segments", "segment"),
    ("funds", "product_ticker"),
]


def extract_dimensions(text: str, dim_reg: dict[str, Any]) -> list[str]:
    """
    Detect 'by <dim>' / 'breakdown by <dim>' and synonyms; normalize via normalize_dim_token.
    Handles plurals and common variants.
    """
    normalized = _normalize_text(text)
    dims: list[str] = []
    by_match = re.search(
        r"\b(?:by|breakdown\s+by)\s+(.+?)(?:\s+where|\s+for\s+last|\s+last\s+\d+|\s+ytd|\s+this\s+month|\s+in\s+|\s+since\s+|$)",
        normalized,
        re.I,
    )
    if not by_match:
        return []
    rest = by_match.group(1).strip()
    for part in re.split(r"\s+and\s+|\s*,\s*", rest):
        token = part.strip().lower()
        if not token:
            continue
        canon = normalize_dim_token(token, dim_reg)
        if not canon:
            for plural, canon_hint in DIM_PLURAL_VARIANTS:
                if token == plural:
                    canon = normalize_dim_token(canon_hint, dim_reg)
                    break
        if not canon:
            canon = normalize_dim_token(token, dim_reg)
        if canon and canon not in dims:
            dims.append(canon)
    return dims


def extract_time_range(text: str, today: date) -> dict[str, Any] | ParseError:
    """
    Recognize ytd, last N months, explicit 2025-01 to 2025-12, Jan 2025, since <date>.
    Returns {"start": date|None, "end": date|None, "granularity": "month"} or ParseError.
    """
    normalized = _normalize_text(text)
    start, end = None, None
    if re.search(r"\bytd\b", normalized):
        start = date(today.year, 1, 1)
        end = today
        return {"start": start, "end": end, "granularity": "month"}
    if re.search(r"\bthis\s+month\b", normalized):
        start = date(today.year, today.month, 1)
        end = today
        return {"start": start, "end": end, "granularity": "month"}
    last_n = re.search(r"\blast\s+(\d+)\s+months?\b", normalized)
    if last_n:
        n = int(last_n.group(1))
        if n < 1 or n > 120:
            return ParseError(
                code=ParseErrorCode.INVALID_TIME_RANGE,
                message="Invalid month count (use 1-120)",
                details={"input": normalized},
                suggestions=None,
            )
        end = today
        year, month = today.year, today.month
        for _ in range(n - 1):
            month -= 1
            if month < 1:
                month += 12
                year -= 1
        start = date(year, month, 1)
        return {"start": start, "end": end, "granularity": "month"}
    since = re.search(r"\bsince\s+(\d{4})-(\d{2})(?:-(\d{2}))?", normalized)
    if since:
        y, m, d = int(since.group(1)), int(since.group(2)), int(since.group(3)) if since.group(3) else 1
        try:
            start = date(y, m, d)
            end = today
            if start > end:
                return ParseError(
                    code=ParseErrorCode.INVALID_TIME_RANGE,
                    message="since date must be before today",
                    details={"input": normalized},
                    suggestions=None,
                )
            return {"start": start, "end": end, "granularity": "month"}
        except ValueError:
            return ParseError(
                code=ParseErrorCode.INVALID_TIME_RANGE,
                message="Invalid since date",
                details={"input": normalized},
                suggestions=None,
            )
    range_m = re.search(r"(\d{4})-(\d{2})(?:\s+to\s+|\s*-\s*)(\d{4})-(\d{2})", normalized)
    if range_m:
        y1, m1, y2, m2 = int(range_m.group(1)), int(range_m.group(2)), int(range_m.group(3)), int(range_m.group(4))
        try:
            start = date(y1, m1, 1)
            _, last = monthrange(y2, m2)
            end = date(y2, m2, last)
            if start > end:
                return ParseError(
                    code=ParseErrorCode.INVALID_TIME_RANGE,
                    message="start must be <= end",
                    details={"input": normalized},
                    suggestions=None,
                )
            return {"start": start, "end": end, "granularity": "month"}
        except ValueError:
            return ParseError(
                code=ParseErrorCode.INVALID_TIME_RANGE,
                message="Invalid date range",
                details={"input": normalized},
                suggestions=None,
            )
    single_month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})\b", normalized)
    if single_month:
        months = "jan feb mar apr may jun jul aug sep oct nov dec".split()
        mo = single_month.group(1).lower()[:3]
        yr = int(single_month.group(2))
        if mo in months:
            mi = months.index(mo) + 1
            try:
                _, last = monthrange(yr, mi)
                start = date(yr, mi, 1)
                end = date(yr, mi, last)
                return {"start": start, "end": end, "granularity": "month"}
            except ValueError:
                pass
    return {"start": start, "end": end, "granularity": "month"}


def extract_filters(
    text: str,
    dim_reg: dict[str, Any],
    value_catalog: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Parse patterns: in US, for US, country US, geo EMEA, ticker SPY, channel institutional.
    Only include filters for known dimensions; validate against value_catalog when present.
    Do not guess if not confident.
    """
    normalized = _normalize_text(text)
    out: dict[str, list[str]] = {}
    value_catalog = value_catalog or {}
    dimensions = dim_reg.get("dimensions") or {}
    if not isinstance(dimensions, dict):
        return out
    canon_keys = {str(k).strip().lower() for k in dimensions.keys()}
    aliases = dim_reg.get("aliases") or {}
    if isinstance(aliases, dict):
        for al, canon in aliases.items():
            if str(canon).strip().lower() in canon_keys:
                canon_keys.add(str(al).strip().lower())

    def catalog_match(dim_canon: str, val: str) -> bool:
        allowed = value_catalog.get(dim_canon)
        if allowed is None:
            return False
        if not isinstance(allowed, (list, tuple, set)):
            allowed = [str(allowed)]
        return val.strip().lower() in {str(v).strip().lower() for v in allowed}

    def add_filter(dim_canon: str, val: str) -> None:
        if dim_canon not in out:
            out[dim_canon] = []
        if val not in out[dim_canon]:
            out[dim_canon].append(val)

    where = re.search(r"\bwhere\s+(\w+)\s+in\s+(.+?)(?:\s+by|\s+where|$)", normalized)
    if where:
        dim_tok = where.group(1).strip().lower()
        canon = normalize_dim_token(dim_tok, dim_reg)
        if canon and canon in canon_keys:
            vals = [v.strip() for v in re.split(r"\s*,\s*", where.group(2).strip()) if v.strip()]
            for v in vals:
                if value_catalog.get(canon) is None or catalog_match(canon, v):
                    add_filter(canon, v)

    for dim_canon in ["src_country", "channel", "product_ticker", "segment"]:
        if dim_canon not in canon_keys:
            continue
        raw_vals = value_catalog.get(dim_canon) or []
        if not isinstance(raw_vals, (list, tuple, set)):
            raw_vals = [str(raw_vals)]
        catalog_vals = list(raw_vals)
        catalog_lower = {str(x).strip().lower(): str(x).strip() for x in catalog_vals}
        if dim_canon == "src_country":
            pats = [r"\b(?:country|geo|region)\s+([A-Za-z0-9]+)\b", r"\bin\s+([A-Za-z0-9]+)\b", r"\bfor\s+([A-Za-z0-9]+)\b"]
        elif dim_canon == "product_ticker":
            pats = [r"\bticker\s+([A-Za-z0-9]{2,6})\b", r"\b(?:fund|etf)\s+([A-Za-z0-9]{2,6})\b"]
        elif dim_canon == "channel":
            pats = [r"\bchannel\s+(\w+)\b"]
        else:
            pats = [rf"\b{dim_canon}\s+(\w+)\b"]
        for pat in pats:
            for m in re.finditer(pat, normalized, re.I):
                val = m.group(1).strip()
                if dim_canon == "product_ticker" and len(val) >= 2 and len(val) <= 6 and val.isupper():
                    add_filter(dim_canon, val)
                    break
                if catalog_lower and val.lower() in catalog_lower:
                    add_filter(dim_canon, catalog_lower[val.lower()])
                    break
                if not catalog_vals and dim_canon == "product_ticker" and 2 <= len(val) <= 6 and val.isalpha():
                    add_filter(dim_canon, val.upper())
                    break
    return out


def _extract_dimensions(normalized: str, dim_reg: dict[str, Any]) -> tuple[list[str], str | None]:
    """Extract dimension keys after 'by' or 'breakdown by'. Returns (list of canonical dims, error)."""
    dims: list[str] = []
    by_match = re.search(r"\b(?:by|breakdown\s+by)\s+(.+?)(?:\s+where|\s+for\s+last|\s+ytd|\s+this\s+month|$)", normalized, re.I)
    if not by_match:
        return [], None
    rest = by_match.group(1).strip()
    for part in re.split(r"\s+and\s+|\s*,\s*", rest):
        token = part.strip().lower()
        if not token:
            continue
        canon = normalize_dim_token(token, dim_reg)
        if not canon:
            return [], f"Unknown dimension: '{token}'"
        if canon not in dims:
            dims.append(canon)
    return dims, None


def _extract_time_range(normalized: str, today: date) -> tuple[date | None, date | None, str | None]:
    """Returns (start, end, error). Uses calendar month for month_end semantics."""
    start, end = None, None
    if re.search(r"\bytd\b", normalized):
        start = date(today.year, 1, 1)
        end = today
        return start, end, None
    if re.search(r"\bthis\s+month\b", normalized):
        start = date(today.year, today.month, 1)
        end = today
        return start, end, None
    last_n = re.search(r"\blast\s+(\d+)\s+months?\b", normalized)
    if last_n:
        n = int(last_n.group(1))
        if n < 1 or n > 120:
            return None, None, "Invalid month count (use 1-120)"
        end = today
        year, month = today.year, today.month
        for _ in range(n - 1):
            month -= 1
            if month < 1:
                month += 12
                year -= 1
        start = date(year, month, 1)
        return start, end, None
    return start, end, None


def _extract_filters(normalized: str, dim_reg: dict[str, Any]) -> tuple[dict[str, list[str]], str | None]:
    """Extract filters from 'where dim in v1, v2' or 'filter dim: v1, v2'. Returns (filters, error)."""
    out: dict[str, list[str]] = {}
    where = re.search(r"\bwhere\s+(.+)", normalized)
    if not where:
        return out, None
    clause = where.group(1).strip()
    in_match = re.match(r"(\w+)\s+in\s+(.+)", clause, re.I)
    if in_match:
        dim_token = in_match.group(1).strip().lower()
        vals_str = in_match.group(2).strip()
        canon = normalize_dim_token(dim_token, dim_reg)
        if not canon:
            return {}, f"Unknown filter dimension: '{dim_token}'"
        vals = [v.strip() for v in re.split(r"\s*,\s*", vals_str) if v.strip()]
        if not vals:
            return {}, "Filter must have at least one value"
        out[canon] = vals
        return out, None
    filter_match = re.match(r"(\w+)\s*:\s*(.+)", clause, re.I)
    if filter_match:
        dim_token = filter_match.group(1).strip().lower()
        vals_str = filter_match.group(2).strip()
        canon = normalize_dim_token(dim_token, dim_reg)
        if not canon:
            return {}, f"Unknown filter dimension: '{dim_token}'"
        vals = [v.strip() for v in re.split(r"\s*,\s*", vals_str) if v.strip()]
        if not vals:
            return {}, "Filter must have at least one value"
        out[canon] = vals
        return out, None
    return out, None


def _top_k_similar(value: str, catalog_values: list[str], k: int = TOP_K_SUGGESTIONS) -> list[str]:
    """Return top k catalog values most similar to value. Uses rapidfuzz if installed else difflib."""
    if not catalog_values:
        return []
    catalog_list = [str(x).strip() for x in catalog_values if x is not None]
    if not catalog_list:
        return []
    value = value.strip()
    if _HAS_RAPIDFUZZ:
        scored = [(c, fuzz.ratio(value.lower(), c.lower())) for c in catalog_list]
        scored.sort(key=lambda x: -x[1])
        return [c for c, _ in scored[:k]]
    matches = difflib.get_close_matches(value, catalog_list, n=k, cutoff=0.2)
    return list(matches)[:k]


def _validate_filter_values(
    spec: QuerySpec,
    value_catalog: dict[str, Any],
) -> ParseError | None:
    """
    value_catalog: dict like {"channel": set([...]), "product_ticker": set([...]), ...}.
    For each filter dim and each value, if value not in catalog (case-insensitive) return ParseError
    with UNKNOWN_FILTER_VALUE and suggestions = top_k_similar.
    """
    if not value_catalog or not isinstance(value_catalog, dict):
        return None
    for dim, values in spec.filters.items():
        raw = value_catalog.get(dim)
        if raw is None:
            continue
        if isinstance(raw, set):
            allowed_set = {str(v).strip().lower() for v in raw}
            catalog_list = list(raw)
        else:
            catalog_list = list(raw) if isinstance(raw, (list, tuple)) else [str(raw)]
            allowed_set = {str(v).strip().lower() for v in catalog_list}
        for v in values:
            if not v or v.strip().lower() in allowed_set:
                continue
            top = _top_k_similar(v, catalog_list, k=TOP_K_SUGGESTIONS)
            return ParseError(
                code=ParseErrorCode.UNKNOWN_FILTER_VALUE,
                message=f"Unknown value '{v}' for {dim}.",
                details={"dim": dim, "value": v},
                suggestions={"values": {dim: top}},
            )
    return None


def parse_nlq(
    text: str,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    value_catalog: dict[str, Any],
    today: date,
) -> QuerySpec | ParseError:
    """
    Parse natural language into a validated QuerySpec or a ParseError.
    Deterministic, rule-first. No SQL or raw user input ever reaches execution.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return ParseError(
            code=ParseErrorCode.EMPTY_QUERY,
            message="Query text is empty",
            details={},
            suggestions=None,
        )

    metric_result = extract_metric(normalized, metric_reg)
    if isinstance(metric_result, ParseError):
        return metric_result
    metric_id = metric_result

    dimensions = extract_dimensions(normalized, dim_reg)

    tr_result = extract_time_range(normalized, today)
    if isinstance(tr_result, ParseError):
        return tr_result
    time_range = TimeRange(
        start=tr_result.get("start"),
        end=tr_result.get("end"),
        granularity=tr_result.get("granularity", "month"),
    )

    filters = extract_filters(normalized, dim_reg, value_catalog)

    intents = classify_intent(normalized)
    effective = _effective_intent(intents)
    chart_type, sort_spec, limit_val, chart_x, chart_y, chart_series = _defaults_from_intent(effective, dimensions)
    if "chart" in normalized or "graph" in normalized:
        if "line" in normalized:
            chart_type, chart_x, chart_y, chart_series = "line", "month_end", "metric", (dimensions[0] if dimensions else None)
        elif "bar" in normalized and dimensions:
            chart_type, chart_x, chart_y = "bar", dimensions[0], "metric"
    if chart_type == "line":
        chart = ChartSpec(type="line", x="month_end", y="metric", series=chart_series)
    elif chart_type == "bar" and chart_x and dimensions:
        chart = ChartSpec(type="bar", x=chart_x, y="metric")
    else:
        chart = ChartSpec(type="table")

    try:
        spec = QuerySpec(
            metric_id=metric_id,
            dimensions=dimensions,
            filters=filters,
            time_range=time_range,
            sort=sort_spec,
            limit=limit_val,
            chart=chart,
        )
    except ValidationError as e:
        err_msg = str(e)
        if "time_range" in err_msg or "start" in err_msg or "end" in err_msg:
            code = ParseErrorCode.INVALID_TIME_RANGE
        else:
            code = ParseErrorCode.AMBIGUOUS
        return ParseError(
            code=code,
            message=f"Invalid query spec: {err_msg}",
            details={"validation_errors": e.errors()},
            suggestions=None,
        )

    try:
        validate_queryspec(spec, metric_reg, dim_reg)
    except GovernanceError as gov_err:
        msg = str(gov_err)
        if "metric_id" in msg or "not found in metric_registry" in msg:
            code = ParseErrorCode.UNKNOWN_METRIC
        elif "allowed_dims" in msg or "firm-only" in msg:
            code = ParseErrorCode.DIM_NOT_ALLOWED
        elif "dimension" in msg and "not found" in msg:
            code = ParseErrorCode.UNKNOWN_DIMENSION
        elif "filter" in msg:
            code = ParseErrorCode.UNKNOWN_FILTER_VALUE
        else:
            code = ParseErrorCode.AMBIGUOUS
        return ParseError(
            code=code,
            message=msg,
            details={"governance_error": msg},
            suggestions=None,
        )

    value_err = _validate_filter_values(spec, value_catalog)
    if value_err is not None:
        return value_err

    return spec


def to_json(spec_or_error: QuerySpec | ParseError) -> dict[str, Any]:
    """
    Deterministic JSON-serializable output.
    QuerySpec -> spec.model_dump() (with date/time serialized).
    ParseError -> {"error": {"code": ..., "message": ..., "details": ..., "suggestions": ...}}.
    """
    if isinstance(spec_or_error, ParseError):
        err = spec_or_error
        return {
            "error": {
                "code": err.code.value,
                "message": err.message,
                "details": err.details,
                "suggestions": err.suggestions,
            }
        }
    return spec_or_error.model_dump(mode="json")


def rewrite_question(text: str) -> str:
    """
    Optional LLM hook: rewrite user question for clarity only.
    NOT called by default. Rewrite can only help clarify phrasing; it cannot set metric/dims
    unless registry validation passes. Caller must still pass result through parse_nlq and
    governance; no free-form SQL or execution path exists.
    """
    return text.strip()
