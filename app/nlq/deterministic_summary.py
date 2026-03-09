"""
Deterministic factual summary for NLQ results. No adjectives; facts only.
Used when LLM is disabled or explanation is rejected.
"""
from __future__ import annotations

from app.nlq.executor import QueryResult
from models.query_spec import QuerySpec


def build_deterministic_summary(qs: QuerySpec, result: QueryResult) -> list[str]:
    """
    Output 2–5 bullet lines, facts only:
    - Metric + agg + time range
    - Dimensions used
    - Filters applied
    - Headline value if present
    - Top 1–3 entities (if dimension query), from first rows of result.data
    Neutral; no adjectives like strong/weak.
    """
    bullets: list[str] = []
    ctx = result.explain_context or {}
    numbers = result.numbers or {}
    metric_label = ctx.get("metric_label") or qs.metric_id.replace("_", " ").title()
    agg = ctx.get("agg_used") or "—"
    tr = qs.time_range
    time_str = "—"
    if tr.start or tr.end:
        if tr.start and tr.end:
            time_str = f"{tr.start.isoformat()} to {tr.end.isoformat()}"
        elif tr.start:
            time_str = f"from {tr.start.isoformat()}"
        else:
            time_str = f"to {tr.end.isoformat()}"
    bullets.append(f"Metric: {metric_label} (aggregation: {agg}); time range: {time_str}.")

    dims = qs.dimensions or []
    if dims:
        bullets.append(f"Dimensions: {', '.join(dims)}.")
    else:
        bullets.append("Dimensions: none (headline).")

    filters = qs.filters or {}
    if filters:
        parts = [f"{k}={v}" for k, v in filters.items()]
        bullets.append(f"Filters applied: {', '.join(parts)}.")
    else:
        bullets.append("Filters applied: none.")

    if numbers.get("formatted") not in (None, "", "—"):
        bullets.append(f"Headline value: {numbers.get('formatted')}.")
    elif numbers.get("value") is not None:
        bullets.append(f"Headline value: {numbers.get('value')}.")

    df = result.data
    if df is not None and not df.empty and dims and dims[0] in df.columns:
        top = df[dims[0]].head(3).astype(str).tolist()
        if top:
            bullets.append(f"Top entities ({dims[0]}): {', '.join(top)}.")

    return bullets[:5]
