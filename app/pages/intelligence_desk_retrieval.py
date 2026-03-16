"""
Retrieval layer for Intelligence Desk: question → relevant dataset slice.
Used to ground Claude answers in the platform's internal financial data only.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


# Keywords → preferred column sets (only columns that exist will be used).
_QUESTION_KEYWORD_COLUMNS: dict[str, list[str]] = {
    "aum": ["product_ticker", "ticker", "end_aum", "channel", "country", "month_end"],
    "flow": ["product_ticker", "ticker", "nnb", "nnf", "channel", "country"],
    "flows": ["product_ticker", "ticker", "nnb", "nnf", "channel", "country"],
    "inflow": ["product_ticker", "ticker", "nnb", "channel", "country"],
    "outflow": ["product_ticker", "ticker", "nnb", "nnf", "channel", "country"],
    "nnb": ["product_ticker", "ticker", "nnb", "channel", "country", "month_end"],
    "nnf": ["product_ticker", "ticker", "nnf", "channel", "country"],
    "ticker": ["product_ticker", "ticker", "nnb", "end_aum", "channel", "country"],
    "etf": ["product_ticker", "ticker", "nnb", "end_aum", "channel", "country"],
    "channel": ["channel", "product_ticker", "ticker", "nnb", "end_aum", "country"],
    "country": ["country", "src_country", "product_ticker", "ticker", "nnb", "end_aum"],
    "growth": ["product_ticker", "ticker", "nnb", "end_aum", "channel", "country", "ogr"],
    "performance": ["product_ticker", "ticker", "end_aum", "nnb", "channel", "country"],
    "return": ["product_ticker", "ticker", "end_aum", "nnb", "channel", "country"],
    "risk": ["product_ticker", "ticker", "end_aum", "nnb", "channel", "country"],
    "segment": ["segment", "sub_segment", "product_ticker", "ticker", "nnb", "end_aum"],
    "asset class": ["segment", "channel", "product_ticker", "ticker", "nnb", "end_aum"],
}

# Fallback when no keyword matches.
_DEFAULT_COLUMNS = [
    "product_ticker", "ticker", "channel", "country", "nnb", "end_aum", "month_end",
]

_MAX_ROWS = 100


def _normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def retrieve_intelligence_desk_context(
    question: str,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Inspect the question, detect key financial terms, extract a relevant slice of df,
    and return (subset_df, context_markdown). Safe against missing columns and empty data.

    - Only selects columns that exist in df.
    - Limits to _MAX_ROWS rows, drops all-null rows, prefers rows with real values.
    - Returns context_markdown as subset_df.to_markdown(index=False) for the prompt.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame(), ""
    if df.empty:
        return pd.DataFrame(), ""

    work = df.copy()
    available = set(work.columns)
    q = _normalize_question(question)

    # Choose columns by keyword (first match wins); only keep columns that exist.
    chosen: list[str] = []
    for keyword, cols in _QUESTION_KEYWORD_COLUMNS.items():
        if keyword in q:
            for c in cols:
                if c in available and c not in chosen:
                    chosen.append(c)
            break
    if not chosen:
        chosen = [c for c in _DEFAULT_COLUMNS if c in available]
    if not chosen:
        chosen = [c for c in work.columns[:10]]

    subset = work[[c for c in chosen if c in work.columns]]
    if subset.empty:
        return pd.DataFrame(), ""

    subset = subset.dropna(how="all", axis=0)
    metric_cols = [c for c in ("nnb", "nnf", "end_aum", "ogr") if c in subset.columns]
    if metric_cols:
        subset = subset.dropna(subset=metric_cols, how="all")
    if subset.empty:
        return pd.DataFrame(), ""

    # Prefer rows with more non-null values (optional: sort by relevance)
    sort_col = None
    sort_ascending = False
    if "nnb" in subset.columns and any(k in q for k in ("outflow", "redemption", "withdrawal", "withdraw")):
        sort_col = "nnb"
        sort_ascending = True
    elif "nnb" in subset.columns and any(k in q for k in ("nnb", "flow", "flows", "inflow")):
        sort_col = "nnb"
        sort_ascending = False
    elif "end_aum" in subset.columns and ("aum" in q or "asset" in q):
        sort_col = "end_aum"
        sort_ascending = False
    elif "ogr" in subset.columns and "growth" in q:
        sort_col = "ogr"
        sort_ascending = False
    if sort_col and sort_col in subset.columns:
        try:
            subset = subset.copy()
            subset["_sort_"] = pd.to_numeric(subset[sort_col], errors="coerce")
            subset = subset.sort_values("_sort_", ascending=sort_ascending, na_position="last").drop(columns=["_sort_"])
        except Exception:
            pass

    subset = subset.head(_MAX_ROWS).reset_index(drop=True)
    if subset.empty:
        return pd.DataFrame(), ""

    try:
        context_markdown = subset.to_markdown(index=False)
    except Exception:
        context_markdown = subset.to_string(index=False)

    return subset, context_markdown or ""
