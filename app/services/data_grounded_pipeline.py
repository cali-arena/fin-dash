from __future__ import annotations

import difflib
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from datetime import date, datetime
from numbers import Number
from typing import Any

import pandas as pd

try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "show",
    "suppose",
    "scenario",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "those",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
}

FORECAST_TERMS = {
    "forecast",
    "future",
    "likelihood",
    "next",
    "probability",
    "projection",
    "trend",
}
WHAT_IF_TERMS = {"if", "scenario", "suppose", "assuming", "assume"}
METRIC_TERMS = {
    "average",
    "count",
    "decrease",
    "direction",
    "down",
    "growth",
    "increase",
    "max",
    "mean",
    "median",
    "min",
    "rate",
    "slope",
    "std",
    "trend",
    "up",
    "volatility",
}
MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
DATE_NAME_HINTS = ("date", "time", "month", "year", "day", "period", "quarter")
MAX_PROMPT_CHARS = 80_000
DEFAULT_MODEL = "claude-3-7-sonnet-latest"
DEFAULT_MAX_TOKENS = 900
DEFAULT_TIMEOUT_SECONDS = 30


def _empty_context() -> dict[str, Any]:
    return {
        "columns_used": [],
        "filters_applied": {},
        "time_range": {"start": "", "end": ""},
        "aggregations": {},
        "trend_analysis": {},
        "sample_rows": [],
        "row_count": 0,
    }


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9%$<>=.\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _round_float(value: Any, digits: int = 6) -> float | None:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return round(value_float, digits)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        if isinstance(value, int):
            return int(value)
        rounded = _round_float(value)
        return rounded if rounded is not None else None
    return str(value)


def _series_to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not any(token in _normalize_text(series.name) for token in DATE_NAME_HINTS):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    coerced = pd.to_datetime(series, errors="coerce", format="mixed")
    valid_ratio = float(coerced.notna().mean()) if len(series) else 0.0
    name_hint = _normalize_text(series.name)
    if any(token in name_hint for token in DATE_NAME_HINTS):
        return coerced
    return coerced if valid_ratio >= 0.6 else pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")


def _detect_date_column(df: pd.DataFrame) -> str | None:
    candidates: list[tuple[float, int, str]] = []
    for index, column in enumerate(df.columns):
        series = _series_to_datetime(df[column])
        valid_ratio = float(series.notna().mean()) if len(series) else 0.0
        if valid_ratio <= 0.0:
            continue
        name_hint = _normalize_text(column)
        score = valid_ratio * 100.0
        if any(token in name_hint for token in DATE_NAME_HINTS):
            score += 25.0
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            score += 25.0
        candidates.append((score, index, str(column)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def _ensure_timestamp(value: pd.Timestamp | datetime | date | str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _quarter_range(year: int, quarter: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_month = ((quarter - 1) * 3) + 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    end = (start + pd.offsets.QuarterEnd()).normalize()
    return start.normalize(), end


def _month_range(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = (start + pd.offsets.MonthEnd()).normalize()
    return start.normalize(), end


def _year_range(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year, month=12, day=31)


def _parse_numeric_token(raw_value: str) -> float | None:
    token = str(raw_value or "").strip().lower().replace(",", "")
    multiplier = 1.0
    if token.endswith("%"):
        token = token[:-1]
    if token.endswith("k"):
        multiplier = 1_000.0
        token = token[:-1]
    elif token.endswith("m"):
        multiplier = 1_000_000.0
        token = token[:-1]
    elif token.endswith("b"):
        multiplier = 1_000_000_000.0
        token = token[:-1]
    token = token.replace("$", "").strip()
    try:
        return float(token) * multiplier
    except ValueError:
        return None


def _scaled_numeric_value(raw_value: str, series: pd.Series) -> float | None:
    parsed = _parse_numeric_token(raw_value)
    if parsed is None:
        return None
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    normalized_raw = str(raw_value or "").strip().lower()
    if "%" in normalized_raw and not numeric_series.empty and numeric_series.abs().median() <= 1:
        return parsed / 100.0
    return parsed


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    coerced = pd.to_numeric(series, errors="coerce")
    return bool(len(series) and float(coerced.notna().mean()) >= 0.8)


def _match_score(keyword: str, column: str) -> float:
    normalized_keyword = _normalize_text(keyword)
    normalized_column = _normalize_text(column)
    if not normalized_keyword or not normalized_column:
        return 0.0
    keyword_tokens = set(_tokenize(normalized_keyword))
    column_tokens = set(_tokenize(normalized_column))
    overlap = len(keyword_tokens & column_tokens)
    score = 0.0
    if normalized_keyword == normalized_column:
        score += 100.0
    if normalized_keyword in normalized_column or normalized_column in normalized_keyword:
        score += 35.0
    if keyword_tokens and column_tokens:
        score += (overlap / max(len(keyword_tokens), len(column_tokens))) * 40.0
    if _HAS_RAPIDFUZZ:
        score += float(fuzz.ratio(normalized_keyword, normalized_column)) * 0.4
    else:
        score += difflib.SequenceMatcher(None, normalized_keyword, normalized_column).ratio() * 40.0
    return score


def extract_keywords(question: str) -> list[str]:
    tokens = _tokenize(question)
    keywords = [token for token in tokens if token not in STOPWORDS and (len(token) > 1 or token.isdigit())]
    return _dedupe_preserve_order(keywords)


def match_columns(keywords: list[str], df_columns: list[str]) -> list[str]:
    if not df_columns:
        return []
    if not keywords:
        return [str(column) for column in df_columns[: min(3, len(df_columns))]]

    ranked: list[tuple[float, int, str]] = []
    for index, column in enumerate(df_columns):
        column_name = str(column)
        best = max((_match_score(keyword, column_name) for keyword in keywords), default=0.0)
        coverage = sum(1 for keyword in keywords if _normalize_text(keyword) in _normalize_text(column_name))
        score = best + (coverage * 5.0)
        ranked.append((score, index, column_name))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    strong_matches = [column for score, _, column in ranked if score >= 30.0]
    target = min(8, max(3, len(strong_matches))) if df_columns else 0
    selected = strong_matches[:target]
    if len(selected) < min(3, len(df_columns)):
        for _, _, column in ranked:
            if column not in selected:
                selected.append(column)
            if len(selected) >= min(3, len(df_columns)):
                break
    return selected[: min(8, len(df_columns))]


def _find_explicit_date_range(question: str, anchor_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    normalized_question = _normalize_text(question)
    years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", normalized_question)]
    if len(years) >= 2 and ("between" in normalized_question or "from" in normalized_question):
        start_year, end_year = min(years[0], years[1]), max(years[0], years[1])
        start, _ = _year_range(start_year)
        _, end = _year_range(end_year)
        return start, end
    month_match = re.search(
        r"\b("
        + "|".join(MONTH_NAME_TO_NUMBER.keys())
        + r")\s+(19\d{2}|20\d{2}|21\d{2})\b",
        normalized_question,
    )
    if month_match:
        month = MONTH_NAME_TO_NUMBER[month_match.group(1)]
        year = int(month_match.group(2))
        return _month_range(year, month)
    if years:
        return _year_range(years[0])
    month_only_match = re.search(r"\b(" + "|".join(MONTH_NAME_TO_NUMBER.keys()) + r")\b", normalized_question)
    if month_only_match:
        month = MONTH_NAME_TO_NUMBER[month_only_match.group(1)]
        year = anchor_date.year
        return _month_range(year, month)
    return None


def _find_relative_date_range(question: str, anchor_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    normalized_question = _normalize_text(question)
    if "ytd" in normalized_question or "year to date" in normalized_question or "this year" in normalized_question:
        return pd.Timestamp(year=anchor_date.year, month=1, day=1), anchor_date
    if "mtd" in normalized_question or "month to date" in normalized_question or "this month" in normalized_question:
        start, _ = _month_range(anchor_date.year, anchor_date.month)
        return start, anchor_date
    if "qtd" in normalized_question or "quarter to date" in normalized_question or "this quarter" in normalized_question:
        quarter = ((anchor_date.month - 1) // 3) + 1
        start, _ = _quarter_range(anchor_date.year, quarter)
        return start, anchor_date
    if "last year" in normalized_question or "previous year" in normalized_question:
        return _year_range(anchor_date.year - 1)
    if "last month" in normalized_question or "previous month" in normalized_question:
        previous_month = (anchor_date - pd.DateOffset(months=1)).normalize()
        return _month_range(previous_month.year, previous_month.month)
    if "last quarter" in normalized_question or "previous quarter" in normalized_question:
        previous_quarter_anchor = (anchor_date - pd.DateOffset(months=3)).normalize()
        previous_quarter = ((previous_quarter_anchor.month - 1) // 3) + 1
        return _quarter_range(previous_quarter_anchor.year, previous_quarter)

    match = re.search(r"\blast\s+(\d+)\s+(day|days|month|months|year|years)\b", normalized_question)
    if match:
        quantity = int(match.group(1))
        unit = match.group(2)
        end = anchor_date
        if "day" in unit:
            start = (anchor_date - pd.Timedelta(days=quantity)).normalize()
        elif "month" in unit:
            start = (anchor_date - pd.DateOffset(months=quantity)).normalize()
        else:
            start = (anchor_date - pd.DateOffset(years=quantity)).normalize()
        return start, end
    return None


def _categorical_filter_candidates(question: str, df: pd.DataFrame) -> dict[str, list[str]]:
    normalized_question = f" {_normalize_text(question)} "
    filters: dict[str, list[str]] = {}
    for column in df.columns:
        series = df[column]
        if _is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        non_null = series.dropna()
        unique_values = sorted({str(value).strip() for value in non_null.tolist() if str(value).strip()})
        if not unique_values or len(unique_values) > 100:
            continue
        matches: list[tuple[int, int, str]] = []
        for value in unique_values:
            normalized_value = _normalize_text(value)
            if len(normalized_value) < 2:
                continue
            needle = f" {normalized_value} "
            position = normalized_question.find(needle)
            if position >= 0:
                matches.append((position, -len(normalized_value), value))
        if matches:
            matches.sort(key=lambda item: (item[0], item[1], item[2]))
            filters[str(column)] = [value for _, _, value in matches[:5]]
    return filters


def _numeric_filter_candidates(question: str, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    numeric_columns = [str(column) for column in df.columns if _is_numeric_series(df[column])]
    if not numeric_columns:
        return {}

    filters: dict[str, list[dict[str, Any]]] = {}
    patterns = [
        (
            "between",
            re.compile(
                r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s+between\s+(?P<low>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)\s+and\s+(?P<high>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)",
                re.IGNORECASE,
            ),
        ),
        (
            "gt",
            re.compile(
                r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s*(?:>=|greater than or equal to|at least|more than|greater than|above|over)\s*(?P<value>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)",
                re.IGNORECASE,
            ),
        ),
        (
            "lt",
            re.compile(
                r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s*(?:<=|less than or equal to|at most|less than|below|under)\s*(?P<value>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)",
                re.IGNORECASE,
            ),
        ),
    ]

    for operation, pattern in patterns:
        for match in pattern.finditer(question):
            hint = (match.groupdict().get("hint") or "").strip()
            hint_keywords = extract_keywords(hint) or extract_keywords(question)
            matched_columns = [
                column
                for column in match_columns(hint_keywords, numeric_columns)
                if column in numeric_columns
            ]
            if not matched_columns:
                continue
            column = matched_columns[0]
            series = df[column]
            if operation == "between":
                low = _scaled_numeric_value(match.group("low"), series)
                high = _scaled_numeric_value(match.group("high"), series)
                if low is None or high is None:
                    continue
                if 1900 <= int(low) <= 2100 and 1900 <= int(high) <= 2100 and _detect_date_column(df):
                    continue
                record = {"op": "between", "value": [min(low, high), max(low, high)]}
            elif operation == "gt":
                value = _scaled_numeric_value(match.group("value"), series)
                if value is None:
                    continue
                record = {"op": ">", "value": value}
            else:
                value = _scaled_numeric_value(match.group("value"), series)
                if value is None:
                    continue
                record = {"op": "<", "value": value}
            filters.setdefault(column, []).append(record)

    return filters


def detect_filters(question: str, df: pd.DataFrame) -> dict[str, Any]:
    try:
        if df.empty:
            return {}
        filters: dict[str, Any] = {}
        date_column = _detect_date_column(df)
        if date_column:
            date_series = _series_to_datetime(df[date_column]).dropna()
            if not date_series.empty:
                anchor_date = _ensure_timestamp(date_series.max())
                explicit_range = _find_explicit_date_range(question, anchor_date)
                relative_range = _find_relative_date_range(question, anchor_date)
                selected_range = explicit_range or relative_range
                if selected_range:
                    start, end = selected_range
                    filters["date"] = {
                        "column": date_column,
                        "start": start.date().isoformat(),
                        "end": end.date().isoformat(),
                    }

        categorical = _categorical_filter_candidates(question, df)
        if categorical:
            filters["categorical"] = categorical

        numeric = _numeric_filter_candidates(question, df)
        if numeric:
            filters["numeric"] = numeric

        return filters
    except Exception:
        return {}


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df.copy(), {}

    filtered_df = df.copy()
    applied: dict[str, Any] = {}
    mask = pd.Series(True, index=filtered_df.index)

    date_filter = filters.get("date")
    if isinstance(date_filter, dict):
        column = date_filter.get("column")
        if column in filtered_df.columns:
            date_series = _series_to_datetime(filtered_df[column])
            start = pd.Timestamp(date_filter["start"])
            end = pd.Timestamp(date_filter["end"])
            date_mask = date_series.between(start, end, inclusive="both")
            mask &= date_mask.fillna(False)
            applied[str(column)] = {"type": "date", "start": start.date().isoformat(), "end": end.date().isoformat()}

    for column, values in sorted(filters.get("categorical", {}).items()):
        if column not in filtered_df.columns or not values:
            continue
        normalized_values = {_normalize_text(value) for value in values}
        series = filtered_df[column].map(_normalize_text)
        category_mask = series.isin(normalized_values)
        mask &= category_mask.fillna(False)
        applied[str(column)] = {"type": "categorical", "values": [str(value) for value in values]}

    for column, conditions in sorted(filters.get("numeric", {}).items()):
        if column not in filtered_df.columns or not conditions:
            continue
        numeric_series = pd.to_numeric(filtered_df[column], errors="coerce")
        column_mask = pd.Series(True, index=filtered_df.index)
        serialized_conditions: list[dict[str, Any]] = []
        for condition in conditions:
            op = condition.get("op")
            if op == "between":
                low, high = condition["value"]
                column_mask &= numeric_series.between(low, high, inclusive="both").fillna(False)
                serialized_conditions.append({"op": "between", "value": [_json_safe(low), _json_safe(high)]})
            elif op == ">":
                value = float(condition["value"])
                column_mask &= (numeric_series > value).fillna(False)
                serialized_conditions.append({"op": ">", "value": _json_safe(value)})
            elif op == "<":
                value = float(condition["value"])
                column_mask &= (numeric_series < value).fillna(False)
                serialized_conditions.append({"op": "<", "value": _json_safe(value)})
        mask &= column_mask
        if serialized_conditions:
            applied[str(column)] = {"type": "numeric", "conditions": serialized_conditions}

    return filtered_df.loc[mask].copy(), applied


def _numeric_aggregations(series: pd.Series) -> dict[str, Any]:
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return {}
    return {
        "mean": _round_float(numeric_series.mean()),
        "median": _round_float(numeric_series.median()),
        "std": _round_float(numeric_series.std(ddof=0)),
        "min": _round_float(numeric_series.min()),
        "max": _round_float(numeric_series.max()),
    }


def _categorical_aggregations(series: pd.Series) -> dict[str, int]:
    values = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
    if not values:
        return {}
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {key: int(value) for key, value in ordered[:5]}


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(values) / len(values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _trend_analysis(filtered_df: pd.DataFrame, date_column: str | None, selected_columns: list[str]) -> tuple[dict[str, Any], dict[str, str]]:
    if not date_column or date_column not in filtered_df.columns or filtered_df.empty:
        return {}, {"start": "", "end": ""}

    date_series = _series_to_datetime(filtered_df[date_column])
    valid_mask = date_series.notna()
    if not valid_mask.any():
        return {}, {"start": "", "end": ""}

    time_range = {
        "start": date_series[valid_mask].min().date().isoformat(),
        "end": date_series[valid_mask].max().date().isoformat(),
    }

    numeric_candidates = [
        column
        for column in selected_columns
        if column in filtered_df.columns and _is_numeric_series(filtered_df[column])
    ]
    basis_column = numeric_candidates[0] if numeric_candidates else "__row_count__"

    working = pd.DataFrame({"__date__": date_series})
    if basis_column == "__row_count__":
        working["__value__"] = 1.0
    else:
        working["__value__"] = pd.to_numeric(filtered_df[basis_column], errors="coerce")

    working = working.dropna(subset=["__date__", "__value__"]).sort_values("__date__")
    if working.empty:
        return {}, time_range

    grouped = working.groupby("__date__", as_index=False)["__value__"].sum()
    values = grouped["__value__"].astype(float).tolist()
    slope = _linear_slope(values)
    first_value = values[0]
    last_value = values[-1]
    if first_value == 0:
        growth_rate = 0.0 if last_value == 0 else (1.0 if last_value > 0 else -1.0)
    else:
        growth_rate = (last_value - first_value) / abs(first_value)
    pct_change = grouped["__value__"].pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    volatility = float(pct_change.std(ddof=0)) if not pct_change.empty else 0.0
    average_level = max(abs(sum(values) / len(values)), 1.0)
    if abs(slope) / average_level < 0.01 and abs(growth_rate) < 0.02:
        direction = "flat"
    elif slope > 0 or growth_rate > 0:
        direction = "up"
    else:
        direction = "down"

    return (
        {
            "direction": direction,
            "slope": _round_float(slope),
            "volatility": _round_float(volatility),
            "growth_rate": _round_float(growth_rate),
            "basis": basis_column,
        },
        time_range,
    )


def _compact_sample_rows(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if df.empty:
        return []
    chosen_columns = [column for column in columns if column in df.columns]
    if not chosen_columns:
        chosen_columns = [str(column) for column in df.columns[: min(5, len(df.columns))]]
    records = df[chosen_columns].head(10).to_dict(orient="records")
    return [{key: _json_safe(value) for key, value in record.items()} for record in records]


def retrieve_context(question: str, df: pd.DataFrame) -> dict[str, Any]:
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return _empty_context()

        keywords = extract_keywords(question)
        matched_columns = match_columns(keywords, [str(column) for column in df.columns])
        detected_filters = detect_filters(question, df)
        filtered_df, applied_filters = _apply_filters(df, detected_filters)
        if not applied_filters:
            filtered_df = df.head(20).copy()

        date_column = _detect_date_column(filtered_df if not filtered_df.empty else df)
        trend_analysis, time_range = _trend_analysis(filtered_df, date_column, matched_columns)

        aggregations: dict[str, Any] = {}
        columns_used = _dedupe_preserve_order(
            matched_columns
            + list(applied_filters.keys())
            + ([date_column] if date_column else [])
            + ([trend_analysis.get("basis")] if trend_analysis.get("basis") and trend_analysis.get("basis") != "__row_count__" else [])
        )
        if len(columns_used) < min(3, len(df.columns)):
            for column in [str(value) for value in df.columns]:
                if column not in columns_used:
                    columns_used.append(column)
                if len(columns_used) >= min(3, len(df.columns)):
                    break
        columns_used = columns_used[: min(8, len(df.columns))]

        for column in columns_used:
            if column not in filtered_df.columns:
                continue
            series = filtered_df[column]
            if _is_numeric_series(series):
                stats = _numeric_aggregations(series)
                if stats:
                    aggregations[column] = stats
            elif column != date_column:
                counts = _categorical_aggregations(series)
                if counts:
                    aggregations[column] = counts

        if trend_analysis:
            aggregations["time_series"] = {
                "basis": trend_analysis["basis"],
                "growth_rate": trend_analysis["growth_rate"],
            }

        return {
            "columns_used": columns_used,
            "filters_applied": applied_filters,
            "time_range": time_range,
            "aggregations": aggregations,
            "trend_analysis": trend_analysis,
            "sample_rows": _compact_sample_rows(filtered_df, columns_used),
            "row_count": int(len(filtered_df)),
        }
    except Exception:
        return _empty_context()


def detect_question_type(question: str) -> str:
    normalized_question = _normalize_text(question)
    tokens = set(_tokenize(normalized_question))
    if WHAT_IF_TERMS & tokens:
        return "what_if"
    if FORECAST_TERMS & tokens:
        return "forecast"
    return "descriptive"


def build_prompt(question: str, context: dict[str, Any], q_type: str) -> str:
    context_json = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=_json_safe)
    instructions = [
        "Use only the DATA CONTEXT. Do not use prior knowledge or external facts.",
        "If the context does not support the question, reply exactly: Dataset does not support this query.",
        "Cite concrete metrics, counts, dates, or trend fields from the context.",
        "Do not invent columns, filters, rows, or causal claims.",
    ]
    if q_type == "forecast":
        instructions.append("Frame the answer as a data-grounded forecast using only observed trends in the context.")
        instructions.append("Explain uncertainty and do not make deterministic claims.")
    elif q_type == "what_if":
        instructions.append("Treat the answer as a conditional scenario grounded only in the observed data context.")
        instructions.append("State assumptions explicitly and avoid unsupported certainty.")

    body = "\n".join(instructions)
    prompt = f"[USER QUESTION]\n{question.strip()}\n\n[DATA CONTEXT]\n{context_json}\n\n[INSTRUCTIONS]\n{body}"
    return prompt[:MAX_PROMPT_CHARS]


def validate_context(context: dict[str, Any]) -> bool:
    try:
        required_keys = {
            "columns_used",
            "filters_applied",
            "time_range",
            "aggregations",
            "trend_analysis",
            "sample_rows",
            "row_count",
        }
        if not isinstance(context, dict) or set(context.keys()) != required_keys:
            return False
        columns_used = context.get("columns_used")
        if not isinstance(columns_used, list) or not columns_used or not all(isinstance(value, str) and value for value in columns_used):
            return False
        if int(context.get("row_count", 0)) <= 0:
            return False
        if not context.get("aggregations") and not context.get("sample_rows"):
            return False
        if len(context.get("sample_rows", [])) > 10:
            return False
        time_range = context.get("time_range", {})
        if not isinstance(time_range, dict) or set(time_range.keys()) != {"start", "end"}:
            return False
        return True
    except Exception:
        return False


def evaluate_response(response: str) -> dict[str, Any]:
    normalized_response = _normalize_text(response)
    has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", normalized_response))
    has_metric_term = any(term in normalized_response for term in METRIC_TERMS)
    has_trend_term = any(term in normalized_response for term in {"trend", "slope", "growth", "volatility", "up", "down", "flat"})
    grounded = has_number or has_metric_term or has_trend_term
    confidence = 0.0
    confidence += 0.45 if has_number else 0.0
    confidence += 0.3 if has_metric_term else 0.0
    confidence += 0.25 if has_trend_term else 0.0
    return {"grounded": grounded, "confidence": round(min(confidence, 1.0), 2)}


def _resolve_api_key() -> str | None:
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if api_key:
        return api_key
    try:
        import streamlit as st

        secret = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
        return secret or None
    except Exception:
        return None


def _call_claude_sync(prompt: str, model: str, max_tokens: int) -> str:
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError("Claude API key is not configured.")
    try:
        from anthropic import Anthropic
    except Exception as exc:
        raise RuntimeError("Anthropic SDK is unavailable.") from exc

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = (getattr(block, "text", "") or "").strip()
        if text:
            parts.append(text)
    message = "\n".join(parts).strip()
    if not message:
        raise RuntimeError("Claude returned an empty response.")
    return message


def call_claude(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            result["text"] = _call_claude_sync(prompt=prompt, model=model, max_tokens=max_tokens)
        except BaseException as exc:
            error["exception"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError("Claude call timed out.")
    if "exception" in error:
        raise RuntimeError(str(error["exception"])) from error["exception"]
    return result.get("text", "")


def _insufficient_data_payload() -> dict[str, str]:
    return {
        "error": "insufficient_data_context",
        "message": "Dataset does not support this query",
    }


def _log_pipeline(question: str, context: dict[str, Any], question_type: str, execution_time_ms: int) -> None:
    payload = {
        "question": question,
        "columns_used": context.get("columns_used", []),
        "row_count": int(context.get("row_count", 0)),
        "question_type": question_type,
        "execution_time_ms": execution_time_ms,
    }
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True))


def chat_handler(question: str, df: pd.DataFrame) -> dict[str, Any]:
    start_time = time.perf_counter()
    question_type = "descriptive"
    context = _empty_context()
    try:
        question_type = detect_question_type(question)
        context = retrieve_context(question, df)
        if not validate_context(context):
            return _insufficient_data_payload()

        prompt = build_prompt(question, context, question_type)
        response = call_claude(prompt)
        response_check = evaluate_response(response)
        if not response_check["grounded"]:
            return {
                "error": "ungrounded_response",
                "message": "Generated response was not sufficiently grounded in dataset context",
            }

        return {
            "question_type": question_type,
            "context": context,
            "prompt": prompt,
            "response": response,
            "response_check": response_check,
        }
    except TimeoutError:
        return {
            "error": "llm_timeout",
            "message": "Timed out while generating a grounded response",
        }
    except Exception:
        return {
            "error": "pipeline_execution_failed",
            "message": "Unable to process the request safely",
        }
    finally:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        _log_pipeline(question=question, context=context, question_type=question_type, execution_time_ms=execution_time_ms)


__all__ = [
    "build_prompt",
    "call_claude",
    "chat_handler",
    "detect_filters",
    "detect_question_type",
    "evaluate_response",
    "extract_keywords",
    "match_columns",
    "retrieve_context",
    "validate_context",
]
