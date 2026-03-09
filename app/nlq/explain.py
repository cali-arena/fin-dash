"""
Optional LLM explanation for NLQ results. Fact-locked: may explain only using payload;
cannot introduce new numbers or facts. Validation rejects any number not in allowed set.
"""
from __future__ import annotations

import re
from typing import Any

from app.nlq.executor import QueryResult
from models.query_spec import QuerySpec


def build_explain_payload(
    qs: QuerySpec,
    result: QueryResult,
    dataset_version: str | None = None,
    filter_hash: str | None = None,
) -> dict[str, Any]:
    """
    Build payload for LLM: queryspec, numbers, table_summary (row_count, columns, head_rows top 5), meta.
    No raw data beyond top 5 rows.
    """
    df = result.data
    row_count = len(df) if df is not None else 0
    columns = list(df.columns) if df is not None and not df.empty else []
    head_rows: list[dict[str, Any]] = []
    if df is not None and not df.empty:
        head = df.head(5)
        for _, row in head.iterrows():
            head_rows.append(row.astype(str).to_dict())
    table_summary = {
        "row_count": row_count,
        "columns": columns,
        "head_rows": head_rows,
    }
    meta = result.meta or {}
    payload_meta: dict[str, Any] = {
        "applied_limit": meta.get("applied_limit"),
    }
    if dataset_version is not None:
        payload_meta["dataset_version"] = dataset_version
    if filter_hash is not None:
        payload_meta["filter_hash"] = filter_hash
    return {
        "queryspec": qs.model_dump(mode="json"),
        "numbers": result.numbers or {},
        "table_summary": table_summary,
        "meta": payload_meta,
    }


def llm_explain(payload: dict[str, Any]) -> str:
    """
    Stub: call LLM if configured; otherwise return "".
    When implementing: instruct "Use ONLY the provided payload. Do NOT add new numbers or facts. If unsure, say you cannot infer."
    """
    return ""


# Regex: ints, floats, percentages (12%, -0.5%), currency-like (1,234.56 or $1.23)
_NUM_PAT = re.compile(
    r"(?:^|[\s\(\[])"  # start or after space/paren
    r"([-+]?"
    r"(?:\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.?\d*)"  # 1,234.56 or 12 or .5
    r"(?:\s*%)?)"  # optional %
    r"(?=[\s\)\]\.,;:?]|$)"
)
_CURRENCY_PAT = re.compile(r"[$€£]\s*[\d,]+\.?\d*|[\d,]+\.?\d*\s*(?:USD|EUR|GBP|%)")


def extract_numbers(text: str) -> set[str]:
    """Extract numeric tokens: ints, floats, percentages (e.g. 1.23, 12%, -0.5), currency-like."""
    if not text:
        return set()
    found: set[str] = set()
    for m in _NUM_PAT.finditer(text):
        t = m.group(1).strip()
        if t and t not in ("+", "-"):
            found.add(t)
    for m in _CURRENCY_PAT.finditer(text):
        found.add(m.group(0).strip())
    # Also catch bare numbers in text (e.g. "value 42" or "42.5")
    for m in re.finditer(r"\b(-?\d+(?:,\d{3})*(?:\.\d+)?|-?\d*\.\d+)\s*%?\b", text):
        found.add(m.group(0).strip())
    return found


def _normalize_num(s: str) -> str:
    """Normalize for comparison: strip commas, percent as decimal."""
    s = str(s).strip().replace(",", "")
    if s.endswith("%"):
        s = s[:-1].strip()
        try:
            return str(float(s) / 100.0)
        except ValueError:
            return s
    return s


def allowed_numbers(payload: dict[str, Any]) -> set[str]:
    """
    Collect all numeric tokens from numbers values, table_summary head_rows, row_count, applied_limit.
    Normalized (strip commas, percent handling).
    """
    out: set[str] = set()
    numbers = payload.get("numbers") or {}
    for v in numbers.values():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            out.add(_normalize_num(str(v)))
        else:
            out.add(_normalize_num(str(v)))
    ts = payload.get("table_summary") or {}
    for row in ts.get("head_rows") or []:
        for v in (row or {}).values():
            out.add(_normalize_num(str(v)))
    rc = ts.get("row_count")
    if rc is not None:
        out.add(_normalize_num(str(rc)))
    meta = payload.get("meta") or {}
    al = meta.get("applied_limit")
    if al is not None:
        out.add(_normalize_num(str(al)))
    return out


def validate_explanation_numbers(explanation: str, payload: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    If explanation contains any numeric token not in allowed_numbers -> reject.
    Returns (passed, list of disallowed tokens).
    """
    allowed = allowed_numbers(payload)
    found = extract_numbers(explanation)
    disallowed: list[str] = []
    for token in found:
        norm = _normalize_num(token)
        if norm and norm not in allowed:
            disallowed.append(token)
    return (len(disallowed) == 0, disallowed)


def deterministic_summary(payload: dict[str, Any]) -> str:
    """Fact-only summary from payload when LLM is off or explanation rejected."""
    parts = []
    qs = payload.get("queryspec") or {}
    metric_id = qs.get("metric_id", "metric")
    dims = qs.get("dimensions") or []
    numbers = payload.get("numbers") or {}
    ts = payload.get("table_summary") or {}
    row_count = ts.get("row_count", 0)
    if numbers.get("formatted") not in (None, "", "—"):
        parts.append(f"**{metric_id.replace('_', ' ').title()}**: {numbers.get('formatted')}.")
    elif numbers.get("value") is not None:
        parts.append(f"**{metric_id.replace('_', ' ').title()}**: {numbers.get('value')}.")
    if dims:
        parts.append(f"Breakdown by: {', '.join(dims)}.")
    parts.append(f"Result: {row_count} row(s).")
    filters = (qs.get("filters") or {})
    if filters:
        parts.append(f"Filters applied: {list(filters.keys())}.")
    return " ".join(parts)
