"""
Retrieval layer for Intelligence Desk: question → relevant dataset slice.
Used to ground Claude answers in the platform's internal financial data only.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd


# Keywords → preferred column sets (only columns that exist will be used).
# Order matters: first matching keyword wins. Include both canonical and alias
# column names so the function works regardless of which query was loaded.
_QUESTION_KEYWORD_COLUMNS: dict[str, list[str]] = {
    "net new business": ["product_ticker", "ticker", "nnb", "channel", "geo", "src_country", "country", "month_end"],
    "aum":              ["product_ticker", "ticker", "end_aum", "channel", "geo", "src_country", "country", "month_end"],
    "inflow":           ["product_ticker", "ticker", "nnb", "channel", "geo", "src_country", "country"],
    "outflow":          ["product_ticker", "ticker", "nnb", "nnf", "channel", "geo", "src_country", "country"],
    "flow":             ["product_ticker", "ticker", "nnb", "nnf", "channel", "geo", "src_country", "country"],
    "flows":            ["product_ticker", "ticker", "nnb", "nnf", "channel", "geo", "src_country", "country"],
    "nnb":              ["product_ticker", "ticker", "nnb", "channel", "geo", "src_country", "country", "month_end"],
    "nnf":              ["product_ticker", "ticker", "nnf", "channel", "geo", "src_country", "country"],
    "etf":              ["product_ticker", "ticker", "nnb", "end_aum", "channel", "geo", "src_country", "country"],
    "ticker":           ["product_ticker", "ticker", "nnb", "end_aum", "channel", "geo", "src_country", "country"],
    "country":          ["geo", "src_country", "country", "product_ticker", "ticker", "nnb", "end_aum"],
    "region":           ["geo", "src_country", "country", "product_ticker", "ticker", "nnb", "end_aum"],
    "channel":          ["channel", "sub_channel", "product_ticker", "ticker", "nnb", "end_aum"],
    "growth":           ["product_ticker", "ticker", "nnb", "end_aum", "ogr", "channel", "geo", "src_country", "country"],
    "performance":      ["product_ticker", "ticker", "end_aum", "nnb", "channel"],
    "return":           ["product_ticker", "ticker", "end_aum", "nnb", "channel"],
    "risk":             ["product_ticker", "ticker", "end_aum", "nnb", "channel"],
    "segment":          ["segment", "sub_segment", "product_ticker", "ticker", "nnb", "end_aum"],
    "asset class":      ["segment", "channel", "product_ticker", "ticker", "nnb", "end_aum"],
}

# Fallback when no keyword matches.
_DEFAULT_COLUMNS = [
    "product_ticker", "ticker", "channel", "geo", "src_country", "country",
    "nnb", "end_aum", "month_end",
]

_MAX_ROWS = 30


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
        # last-resort: first 10 available columns
        chosen = list(work.columns[:10])

    subset = work[[c for c in chosen if c in work.columns]].copy()
    if subset.empty:
        return pd.DataFrame(), ""

    # Drop rows that are entirely null.
    subset = subset.dropna(how="all", axis=0)
    # Drop rows where ALL metric columns are null (keeps rows with at least one metric).
    metric_cols = [c for c in ("nnb", "nnf", "end_aum", "ogr") if c in subset.columns]
    if metric_cols:
        subset = subset.dropna(subset=metric_cols, how="all")
    if subset.empty:
        return pd.DataFrame(), ""

    # Sort by the most relevant metric, descending by default.
    sort_col: str | None = None
    sort_ascending = False
    q_has_outflow = any(k in q for k in ("outflow", "redemption", "withdrawal", "worst", "lowest"))
    q_has_nnb = any(k in q for k in ("nnb", "flow", "flows", "inflow", "net new business"))
    q_has_aum = any(k in q for k in ("aum", "asset", "assets", "largest", "biggest"))
    q_has_growth = "growth" in q

    if "nnb" in subset.columns and q_has_outflow:
        sort_col, sort_ascending = "nnb", True
    elif "nnb" in subset.columns and q_has_nnb:
        sort_col, sort_ascending = "nnb", False
    elif "end_aum" in subset.columns and q_has_aum:
        sort_col, sort_ascending = "end_aum", False
    elif "ogr" in subset.columns and q_has_growth:
        sort_col, sort_ascending = "ogr", False

    if sort_col and sort_col in subset.columns:
        try:
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


# Model used for general (non-dataset) answers; same as Intelligence Desk tab.
_INTEL_DESK_MODEL = "claude-haiku-4-5"


def claude_generate_general_answer(question: str) -> str:
    """
    Fallback when dataset retrieval is empty: call Claude for a general
    analyst-style answer. Uses the same Claude client as the rest of the app.
    """
    question = (question or "").strip()
    if not question:
        return "Please ask a question."
    try:
        from app.services.claude_client import claude_generate
    except Exception:
        return "Claude is not available. Try a question that can be answered from the dataset."
    prompt = (
        "You are a financial markets analyst.\n\n"
        "Answer the user question clearly and concisely.\n\n"
        f"Question:\n{question}"
    )
    try:
        return claude_generate(prompt=prompt, model=_INTEL_DESK_MODEL, max_tokens=1000)
    except Exception as e:
        msg = getattr(e, "message", str(e)) or "Request failed."
        return f"Unable to get a general answer: {msg}."
