"""
Type enforcement pipeline for DATA RAW: strict date parsing, currency parsing,
identifier normalization, and rejection of unparseable rows.
Uses pandas + numpy only; no I/O. Raises only for programmer errors (missing columns).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

# Editable list of date formats tried in order for strict parsing.
DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%b %d, %Y",
    "%B %d, %Y",
]


def parse_dates_strict(
    series: pd.Series,
    formats: list[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Parse string series with multiple date formats. Tries each format only on
    entries that are still null. Returns (parsed_dt, parse_ok) where parse_ok
    is a boolean series True where the value was successfully parsed.
    """
    if formats is None:
        formats = DATE_FORMATS
    s = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        still_null = result.isna()
        if not still_null.any():
            break
        try_parse = pd.to_datetime(s.loc[still_null], format=fmt, errors="coerce")
        result.loc[still_null] = try_parse.values
    parse_ok = result.notna()
    return result, parse_ok


def to_month_end(dt_series: pd.Series) -> pd.Series:
    """Convert datetime series to month-end (same month)."""
    return dt_series + pd.offsets.MonthEnd(0)


def parse_currency(series: pd.Series) -> pd.Series:
    """
    Parse currency strings to float64. Handles:
    - (1,234.56) -> -1234.56
    - -1,234.56 -> -1234.56
    - " $ 1 234,56 " (EU style): best-effort. Limitation: we treat comma with
      no dot as decimal separator (e.g. 1234,56 -> 1234.56); if both comma
      and dot exist we assume dot is decimal (US). Space as thousands sep
      is removed; ambiguous mixes of EU/US may be misparsed.
    Does not fill NaN; caller decides reject policy.
    """
    s = series.astype(str).str.strip()
    # Detect parentheses negative and remove parens
    s = s.str.replace(r"^\s*\(([^)]*)\)\s*$", r"-\1", regex=True).str.strip()
    # Remove currency symbols and spaces
    s = s.str.replace("R$", "", regex=False)
    s = s.str.replace(r"[\$£€¥]", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    # Thousands vs decimal: if comma present and no dot -> decimal comma (EU)
    has_dot = s.str.contains(r"\.", na=False)
    has_comma = s.str.contains(",", na=False)
    eu_style = has_comma & ~has_dot
    s = s.where(~eu_style, s.str.replace(",", ".", regex=False))
    # Remove remaining commas (thousands)
    s = s.str.replace(",", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    return out.astype("float64")


def normalize_identifier(series: pd.Series) -> pd.Series:
    """
    Normalize identifier strings: strip, collapse internal whitespace to
    single space. Case is preserved unless caller applies additional transform.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.astype(pd.StringDtype())


def enforce_types_data_raw(
    df_raw: pd.DataFrame,
    *,
    date_col: str,
    currency_cols: list[str],
    id_cols: list[str],
    date_formats: list[str] | None = None,
    reject_reason_col: str = "_reject_reason",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Run type enforcement pipeline: strict date parse -> month-end, currency
    parse, identifier normalization. Unparseable rows go to df_rejects with
    original raw values and a reason. Returns (df_clean, df_rejects, stats).
    Raises only for missing columns (programmer error).
    """
    if date_formats is None:
        date_formats = DATE_FORMATS
    required = [date_col] + list(currency_cols) + list(id_cols)
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns for type enforcement: {missing}")

    df = df_raw.copy()
    n = len(df)
    reject_reason: dict[int, str] = {}
    # (a) Date strict parse -> month-end; reject bad_date
    date_series = df[date_col].astype(str)
    parsed_dt, parse_ok = parse_dates_strict(date_series, formats=date_formats)
    month_end = to_month_end(parsed_dt)
    bad_date_idx = parsed_dt.isna()
    for i in df.index[bad_date_idx]:
        reject_reason[int(i)] = "bad_date"

    # (b) Currency parse; reject rows with NaN in any currency (only if not already rejected)
    currency_parsed: dict[str, pd.Series] = {}
    currency_parse_nulls: dict[str, int] = {}
    for col in currency_cols:
        parsed = parse_currency(df[col])
        currency_parsed[col] = parsed
        currency_parse_nulls[col] = int(parsed.isna().sum())
    for idx in df.index:
        if reject_reason.get(int(idx)):
            continue
        for col in currency_cols:
            if pd.isna(currency_parsed[col].loc[idx]):
                reject_reason[int(idx)] = f"bad_currency:{col}"
                break

    rejected_idx = sorted(reject_reason.keys())
    clean_idx = [i for i in df.index if i not in reject_reason]

    # Build df_clean: non-rejected, date_col -> month_end, currency cols parsed, id_cols normalized
    df_clean = df.loc[clean_idx].copy()
    if len(df_clean) > 0:
        df_clean[date_col] = month_end.loc[clean_idx].values
        for col in currency_cols:
            df_clean[col] = currency_parsed[col].loc[clean_idx].values
        for col in id_cols:
            df_clean[col] = normalize_identifier(df_clean[col])

    # Build df_rejects: original raw values + reject reason
    if rejected_idx:
        df_rejects = df_raw.loc[rejected_idx].copy()
        df_rejects[reject_reason_col] = [reject_reason[i] for i in rejected_idx]
    else:
        df_rejects = pd.DataFrame()

    reject_counts = dict(Counter(reject_reason.values()))
    stats: dict[str, Any] = {
        "rows_in": n,
        "rows_clean": len(df_clean),
        "rows_rejected": len(df_rejects),
        "reject_counts": reject_counts,
        "currency_parse_nulls": currency_parse_nulls,
    }
    return df_clean, df_rejects, stats
