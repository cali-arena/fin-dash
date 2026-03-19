from __future__ import annotations

import difflib
import json
import logging
import math
import re
import time
import unicodedata
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

CLAUDE_GENERATION_FAILURE_RESPONSE = "Unable to generate a grounded response safely."
GROUNDED_VALIDATION_FALLBACK = "I could not produce a sufficiently grounded answer from the available dataset context."
ANALYTICAL_VALIDATION_FALLBACK = "Based on the available signals, there is not enough reliable evidence to make a stronger directional inference."
INSUFFICIENT_EVIDENCE_RESPONSE = "There is not enough evidence in the current dataset to answer this reliably."

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
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "show",
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
    "with",
}

TIME_NAME_PRIORITY = (
    "date",
    "month",
    "year",
    "period",
    "timestamp",
    "created_at",
    "ds",
    "time",
    "quarter",
)

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

METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "net_inflows": ("net inflow", "net inflows", "net flow", "net flows", "net subscriptions", "nnb", "net sales"),
    "inflow": ("inflow", "inflows", "gross inflow", "gross inflows", "subscriptions", "sales"),
    "outflow": ("outflow", "outflows", "gross outflow", "gross outflows", "redemptions", "withdrawals"),
    "revenue": ("revenue", "revenues", "sales", "income"),
    "volume": ("volume", "turnover", "activity"),
    "count": ("count", "counts", "transactions", "clients", "customers"),
    "balance": ("balance", "balances", "aum", "assets", "holdings"),
}

QUESTION_META_TOKENS = {
    "above",
    "acceleration",
    "after",
    "analytical",
    "analysis",
    "answer",
    "average",
    "based",
    "before",
    "below",
    "between",
    "bottom",
    "change",
    "changes",
    "chance",
    "compare",
    "comparison",
    "confidence",
    "continue",
    "continuing",
    "could",
    "decline",
    "decrease",
    "decreasing",
    "definitely",
    "descriptive",
    "direction",
    "directional",
    "down",
    "drop",
    "exact",
    "forecast",
    "future",
    "grew",
    "grow",
    "growth",
    "happens",
    "highest",
    "how",
    "if",
    "increase",
    "increasing",
    "inference",
    "largest",
    "least",
    "likelihood",
    "limited",
    "long",
    "low",
    "lowest",
    "may",
    "medium",
    "momentum",
    "month",
    "months",
    "most",
    "next",
    "observed",
    "overall",
    "period",
    "previous",
    "probability",
    "quantified",
    "quarter",
    "quarters",
    "rank",
    "ranking",
    "rate",
    "reliable",
    "report",
    "result",
    "results",
    "risk",
    "rise",
    "rising",
    "scenario",
    "short",
    "show",
    "signal",
    "signals",
    "smallest",
    "slope",
    "strong",
    "stronger",
    "suppose",
    "sustainable",
    "this",
    "top",
    "trend",
    "trajectory",
    "uncertain",
    "uncertainty",
    "under",
    "up",
    "value",
    "values",
    "volatility",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "window",
    "year",
    "years",
}


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9%$<>=.\-\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return round(number, 6)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _serialize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in record.items()} for record in records]


def _string_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(fuzz.ratio(left, right)) / 100.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def _is_numeric_series(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return True
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(len(series) and float(numeric.notna().mean()) >= 0.8)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if _is_numeric_series(df[column])]


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    result: list[str] = []
    for column in df.columns:
        series = df[column]
        if _is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        result.append(str(column))
    return result


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _series_to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if _is_numeric_series(series) and not any(token in _normalize_text(series.name) for token in TIME_NAME_PRIORITY):
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _empty_signals() -> dict[str, Any]:
    return {
        "n_points": 0,
        "latest_value": None,
        "previous_value": None,
        "pct_change_latest": None,
        "trend_direction": "unknown",
        "trend_strength": "unknown",
        "slope": None,
        "regression_r2": None,
        "volatility": None,
        "moving_average_short": None,
        "moving_average_long": None,
        "acceleration": None,
        "window_comparison": {
            "recent_window_mean": None,
            "prior_window_mean": None,
            "delta": None,
            "pct_delta": None,
        },
        "signal_confidence": "low",
        "signal_confidence_score": 0.0,
        "seasonality_hint": "unknown",
        "noise_level": "unknown",
    }


def _empty_analytics() -> dict[str, Any]:
    return {
        "analytics_used": False,
        "metric_info": {
            "primary_metric": None,
            "secondary_metrics": [],
            "matched_by": "none",
        },
        "signals": _empty_signals(),
        "likelihood": {
            "likelihood_label": "unclear",
            "likelihood_score": 0.5,
            "reasoning_factors": [],
        },
        "analytical_context": {},
        "analytics_error": None,
    }


def _log_event(payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_safe))


def _resolve_final_mode(
    question_type: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
) -> str:
    if evidence_decision.get("not_enough_evidence"):
        return "not_enough_evidence"
    if question_type.get("needs_analytics") and analytics.get("analytics_used"):
        return "analytical_inference"
    return "grounded_data"


def _format_evidence_record(record: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in record.items() if value is not None]
    return ", ".join(parts[:6]).strip()


def _validation_error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "valid": False,
        "response_mode_detected": "unclear",
        "grounded": False,
        "confidence": 0.0,
        "violations": ["validator_error"],
        "error": str(exc),
    }


def _safe_validation_payload(mode: str, confidence: float) -> dict[str, Any]:
    return {
        "valid": True,
        "response_mode_detected": mode,
        "grounded": True,
        "confidence": round(confidence, 2),
        "violations": [],
    }


def _expected_response_mode(
    question_type: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
) -> str:
    if evidence_decision.get("not_enough_evidence"):
        return "not_enough_evidence"
    if question_type.get("needs_analytics") and analytics.get("analytics_used"):
        return "analytical_inference"
    return "grounded_data"


def _is_safe_fallback_response(response: str, expected_mode: str) -> bool:
    return (
        (expected_mode == "grounded_data" and response == GROUNDED_VALIDATION_FALLBACK)
        or (expected_mode == "analytical_inference" and response == ANALYTICAL_VALIDATION_FALLBACK)
        or (expected_mode == "not_enough_evidence" and response == INSUFFICIENT_EVIDENCE_RESPONSE)
    )


def _dataset_reference_tokens(df: pd.DataFrame) -> set[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return set()

    tokens: set[str] = set()
    for column in df.columns:
        tokens.update(_tokenize(column))
        series = df[column]
        if _is_numeric_series(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        unique_values = {str(value).strip() for value in series.dropna().tolist() if str(value).strip()}
        if 0 < len(unique_values) <= 50:
            for value in unique_values:
                tokens.update(_tokenize(value))
    return tokens


def _find_unresolved_question_tokens(question: str, df: pd.DataFrame) -> list[str]:
    dataset_tokens = _dataset_reference_tokens(df)
    unresolved: list[str] = []
    for token in _tokenize(question):
        if token.isdigit() or token in STOPWORDS or token in QUESTION_META_TOKENS:
            continue
        if token not in dataset_tokens:
            unresolved.append(token)
    return _dedupe_preserve_order(unresolved)


def _validate_response_safely(
    response: str,
    question_type: dict[str, Any],
    context: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
) -> dict[str, Any]:
    try:
        return response_validator(response, question_type, context, analytics, evidence_decision)
    except Exception as exc:
        logger.exception("response_validator failed")
        return _validation_error_payload(exc)


def _build_validation_fallback(
    question_type: dict[str, Any],
    context: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[str, str]:
    expected_mode = _resolve_final_mode(question_type, analytics, evidence_decision)

    if expected_mode == "not_enough_evidence":
        return INSUFFICIENT_EVIDENCE_RESPONSE, "not_enough_evidence"

    if expected_mode == "analytical_inference":
        if evidence_decision.get("not_enough_evidence"):
            return INSUFFICIENT_EVIDENCE_RESPONSE, "not_enough_evidence"
        return ANALYTICAL_VALIDATION_FALLBACK, "analytical_inference"

    return GROUNDED_VALIDATION_FALLBACK, "grounded_data"


def classify_question_type(question: str, df: pd.DataFrame) -> dict[str, Any]:
    normalized_question = _normalize_text(question)
    tokens = set(_tokenize(question))
    numeric_columns = _numeric_columns(df) if isinstance(df, pd.DataFrame) else []
    df_columns = [str(column) for column in df.columns] if isinstance(df, pd.DataFrame) else []

    grounded_terms = {
        "highest",
        "lowest",
        "top",
        "most",
        "least",
        "largest",
        "smallest",
        "rank",
        "compare",
        "comparison",
        "which",
    }
    analytical_terms = {
        "trend",
        "trajectory",
        "chance",
        "likely",
        "likelihood",
        "probability",
        "sustainable",
        "sustainability",
        "continuation",
        "continue",
        "continuing",
        "risk",
        "momentum",
        "acceleration",
        "deceleration",
        "slowdown",
        "slow",
        "scenario",
    }
    causal_patterns = (
        "why did",
        "what caused",
        "what caused this",
        "why did this happen",
        "what explains exactly",
        "cause this exactly",
    )
    strong_forecast_patterns = (
        "will definitely",
        "will certainly",
        "guaranteed",
        "exactly what will happen",
    )

    risk_flags: list[str] = []
    if any(pattern in normalized_question for pattern in causal_patterns):
        risk_flags.append("causal_request")
    if any(pattern in normalized_question for pattern in strong_forecast_patterns):
        risk_flags.append("strong_forecast_request")

    if "next year" in normalized_question or "over the next year" in normalized_question or "next 12 months" in normalized_question:
        time_col = infer_time_column(df) if isinstance(df, pd.DataFrame) else None
        if time_col is None:
            risk_flags.append("insufficient_history_possible")
        elif len(prepare_time_series(df, time_col, numeric_columns[0]).index) < 12 if numeric_columns else True:
            risk_flags.append("insufficient_history_possible")

    semantic_hits = _semantic_metric_hits(question)
    metric_requested = bool(semantic_hits)
    unresolved_question_tokens = _find_unresolved_question_tokens(question, df) if isinstance(df, pd.DataFrame) else []
    if metric_requested and not numeric_columns:
        risk_flags.append("unsupported_variable")
    elif metric_requested:
        inferred_metric = infer_target_metric(question, df, {"numeric_columns_in_scope": numeric_columns})
        if inferred_metric.get("primary_metric") is None:
            risk_flags.append("unsupported_variable")
    elif unresolved_question_tokens:
        risk_flags.append("unsupported_variable")

    if any(pattern in normalized_question for pattern in causal_patterns) or "definitely" in normalized_question:
        question_mode = "not_enough_evidence_candidate"
        intent_type = "causal" if any(pattern in normalized_question for pattern in causal_patterns) else "forecast_like"
    elif analytical_terms & tokens or "if this continues" in normalized_question or "what happens if" in normalized_question:
        question_mode = "analytical_inference"
        if "what happens if" in normalized_question or "if this continues" in normalized_question or "suppose" in normalized_question:
            intent_type = "what_if"
        elif "chance" in tokens or "likelihood" in tokens or "probability" in tokens or "risk" in tokens:
            intent_type = "forecast_like"
        else:
            intent_type = "trend"
    elif grounded_terms & tokens or "by channel" in normalized_question or "by ticker" in normalized_question or "by country" in normalized_question:
        question_mode = "grounded_data"
        intent_type = "comparison" if "compare" in tokens or "comparison" in tokens else "descriptive"
    else:
        question_mode = "grounded_data"
        intent_type = "descriptive"

    target_direction = "unknown"
    if {"decline", "decrease", "decreasing", "drop", "down", "fall", "falling", "reduce", "reducing"} & tokens:
        target_direction = "down"
    elif {"increase", "increasing", "growth", "grow", "up", "rise", "rising"} & tokens:
        target_direction = "up"

    return {
        "question_mode": question_mode,
        "intent_type": intent_type,
        "needs_analytics": question_mode == "analytical_inference",
        "needs_direct_answer": question_mode == "grounded_data",
        "risk_flags": _dedupe_preserve_order(risk_flags),
        "target_direction": target_direction,
        "question_text": question,
    }


def extract_keywords(question: str) -> list[str]:
    tokens = _tokenize(question)
    keywords = [token for token in tokens if token not in STOPWORDS and (len(token) > 1 or token.isdigit())]
    return _dedupe_preserve_order(keywords)


def match_columns(keywords: list[str], df_columns: list[str]) -> list[str]:
    if not df_columns:
        return []
    if not keywords:
        return df_columns[: min(3, len(df_columns))]

    ranked: list[tuple[float, int, str]] = []
    for index, column in enumerate(df_columns):
        normalized_column = _normalize_text(column)
        column_tokens = set(_tokenize(column))
        score = 0.0
        for keyword in keywords:
            normalized_keyword = _normalize_text(keyword)
            keyword_tokens = set(_tokenize(keyword))
            if normalized_keyword == normalized_column:
                score = max(score, 100.0)
            elif normalized_keyword in normalized_column or normalized_column in normalized_keyword:
                score = max(score, 70.0)
            else:
                overlap = len(keyword_tokens & column_tokens)
                token_score = (overlap / max(len(column_tokens), 1)) * 40.0
                similarity_score = _string_similarity(normalized_keyword, normalized_column) * 50.0
                score = max(score, token_score + similarity_score)
        ranked.append((score, index, column))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched = [column for score, _, column in ranked if score >= 20.0]
    if len(matched) < min(3, len(df_columns)):
        matched = [column for _, _, column in ranked[: min(3, len(df_columns))]]
    return matched[: min(8, len(df_columns))]


def infer_time_column(df: pd.DataFrame) -> str | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    candidates: list[tuple[float, int, str]] = []
    for index, column in enumerate(df.columns):
        series = df[column]
        name = _normalize_text(column)
        score = 0.0
        if pd.api.types.is_datetime64_any_dtype(series):
            score += 100.0
        else:
            if _is_numeric_series(series) and not any(token in name for token in TIME_NAME_PRIORITY):
                continue
            converted = _series_to_datetime(series)
            valid_ratio = float(converted.notna().mean()) if len(series) else 0.0
            score += valid_ratio * 50.0
            if valid_ratio < 0.6 and not any(token in name for token in TIME_NAME_PRIORITY):
                continue

        for priority_index, token in enumerate(TIME_NAME_PRIORITY):
            if token in name:
                score += 35.0 - priority_index

        if score > 0:
            candidates.append((score, index, str(column)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def _find_explicit_date_range(question: str, anchor_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    normalized_question = _normalize_text(question)
    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", normalized_question)]
    if len(years) >= 2 and ("between" in normalized_question or "from" in normalized_question):
        start_year, end_year = min(years[0], years[1]), max(years[0], years[1])
        return pd.Timestamp(start_year, 1, 1), pd.Timestamp(end_year, 12, 31)

    month_match = re.search(
        r"\b(" + "|".join(MONTH_NAME_TO_NUMBER.keys()) + r")\s+(19\d{2}|20\d{2}|21\d{2})\b",
        normalized_question,
    )
    if month_match:
        month = MONTH_NAME_TO_NUMBER[month_match.group(1)]
        year = int(month_match.group(2))
        start = pd.Timestamp(year, month, 1)
        return start, (start + pd.offsets.MonthEnd()).normalize()

    if years:
        return pd.Timestamp(years[0], 1, 1), pd.Timestamp(years[0], 12, 31)

    month_only = re.search(r"\b(" + "|".join(MONTH_NAME_TO_NUMBER.keys()) + r")\b", normalized_question)
    if month_only:
        month = MONTH_NAME_TO_NUMBER[month_only.group(1)]
        start = pd.Timestamp(anchor_date.year, month, 1)
        return start, (start + pd.offsets.MonthEnd()).normalize()
    return None


def _find_relative_date_range(question: str, anchor_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    normalized_question = _normalize_text(question)
    if "last year" in normalized_question or "previous year" in normalized_question:
        return pd.Timestamp(anchor_date.year - 1, 1, 1), pd.Timestamp(anchor_date.year - 1, 12, 31)
    if "last month" in normalized_question or "previous month" in normalized_question:
        previous_month = (anchor_date - pd.DateOffset(months=1)).normalize()
        start = pd.Timestamp(previous_month.year, previous_month.month, 1)
        return start, (start + pd.offsets.MonthEnd()).normalize()
    if "this year" in normalized_question or "year to date" in normalized_question or "ytd" in normalized_question:
        return pd.Timestamp(anchor_date.year, 1, 1), anchor_date
    if "this month" in normalized_question or "month to date" in normalized_question or "mtd" in normalized_question:
        return pd.Timestamp(anchor_date.year, anchor_date.month, 1), anchor_date

    match = re.search(r"\blast\s+(\d+)\s+(day|days|month|months|year|years)\b", normalized_question)
    if match:
        quantity = int(match.group(1))
        unit = match.group(2)
        if "day" in unit:
            return (anchor_date - pd.Timedelta(days=quantity)).normalize(), anchor_date
        if "month" in unit:
            return (anchor_date - pd.DateOffset(months=quantity)).normalize(), anchor_date
        return (anchor_date - pd.DateOffset(years=quantity)).normalize(), anchor_date
    return None


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


def detect_filters(question: str, df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    filters: dict[str, Any] = {}
    time_col = infer_time_column(df)
    if time_col is not None:
        time_series = _series_to_datetime(df[time_col]).dropna()
        if not time_series.empty:
            anchor_date = pd.Timestamp(time_series.max()).normalize()
            date_range = _find_explicit_date_range(question, anchor_date) or _find_relative_date_range(question, anchor_date)
            if date_range is not None:
                start, end = date_range
                filters["date"] = {
                    "column": time_col,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                }

    categorical_filters: dict[str, list[str]] = {}
    normalized_question = f" {_normalize_text(question)} "
    for column in _categorical_columns(df):
        unique_values = sorted({str(value).strip() for value in df[column].dropna().tolist() if str(value).strip()})
        if not unique_values or len(unique_values) > 100:
            continue
        matched_values = [value for value in unique_values if f" {_normalize_text(value)} " in normalized_question]
        if matched_values:
            categorical_filters[column] = matched_values[:5]
    if categorical_filters:
        filters["categorical"] = categorical_filters

    numeric_filters: dict[str, list[dict[str, Any]]] = {}
    numeric_columns = _numeric_columns(df)
    patterns = [
        ("between", re.compile(r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s+between\s+(?P<low>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)\s+and\s+(?P<high>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)", re.IGNORECASE)),
        ("gt", re.compile(r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s*(?:>=|greater than or equal to|at least|more than|greater than|above|over)\s*(?P<value>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)", re.IGNORECASE)),
        ("lt", re.compile(r"(?P<hint>[a-zA-Z_][a-zA-Z0-9_ ]{0,40}?)\s*(?:<=|less than or equal to|at most|less than|below|under)\s*(?P<value>[$]?-?\d[\d,]*(?:\.\d+)?%?[kmb]?)", re.IGNORECASE)),
    ]
    for operation, pattern in patterns:
        for match in pattern.finditer(question):
            hint_keywords = extract_keywords(match.groupdict().get("hint") or question)
            matched_cols = [column for column in match_columns(hint_keywords, numeric_columns) if column in numeric_columns]
            if not matched_cols:
                continue
            target_col = matched_cols[0]
            if operation == "between":
                low = _parse_numeric_token(match.group("low"))
                high = _parse_numeric_token(match.group("high"))
                if low is None or high is None:
                    continue
                numeric_filters.setdefault(target_col, []).append({"op": "between", "value": [min(low, high), max(low, high)]})
            elif operation == "gt":
                value = _parse_numeric_token(match.group("value"))
                if value is None:
                    continue
                numeric_filters.setdefault(target_col, []).append({"op": ">", "value": value})
            else:
                value = _parse_numeric_token(match.group("value"))
                if value is None:
                    continue
                numeric_filters.setdefault(target_col, []).append({"op": "<", "value": value})
    if numeric_filters:
        filters["numeric"] = numeric_filters
    return filters


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
            applied[column] = {
                "type": "date",
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
            }

    for column, values in sorted(filters.get("categorical", {}).items()):
        if column not in filtered_df.columns or not values:
            continue
        normalized_values = {_normalize_text(value) for value in values}
        series = filtered_df[column].map(_normalize_text)
        category_mask = series.isin(normalized_values)
        mask &= category_mask.fillna(False)
        applied[column] = {"type": "categorical", "values": [str(value) for value in values]}

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
            applied[column] = {"type": "numeric", "conditions": serialized_conditions}

    return filtered_df.loc[mask].copy(), applied


def _reconstruct_scoped_subset(df: pd.DataFrame, context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str] | None]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy(), {"code": "empty_dataframe", "message": "Analytical inference requires a non-empty scoped dataset"}

    filters_applied = context.get("filters_applied", {}) if isinstance(context, dict) else {}
    if not isinstance(filters_applied, dict) or not filters_applied:
        return df.copy(), None

    scoped_df = df.copy()
    mask = pd.Series(True, index=scoped_df.index)

    for column, specification in sorted(filters_applied.items()):
        if column not in scoped_df.columns or not isinstance(specification, dict):
            return (
                pd.DataFrame(columns=df.columns),
                {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
            )

        filter_type = specification.get("type")
        if filter_type == "date":
            date_series = _series_to_datetime(scoped_df[column])
            start = specification.get("start")
            end = specification.get("end")
            if start is None or end is None:
                return (
                    pd.DataFrame(columns=df.columns),
                    {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                )
            date_mask = date_series.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
            mask &= date_mask.fillna(False)
        elif filter_type == "categorical":
            values = specification.get("values") or []
            normalized_values = {_normalize_text(value) for value in values}
            if not normalized_values:
                return (
                    pd.DataFrame(columns=df.columns),
                    {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                )
            category_mask = scoped_df[column].map(_normalize_text).isin(normalized_values)
            mask &= category_mask.fillna(False)
        elif filter_type == "numeric":
            numeric_series = pd.to_numeric(scoped_df[column], errors="coerce")
            numeric_mask = pd.Series(True, index=scoped_df.index)
            conditions = specification.get("conditions") or []
            if not conditions:
                return (
                    pd.DataFrame(columns=df.columns),
                    {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                )
            for condition in conditions:
                operator = condition.get("op")
                if operator == "between":
                    low, high = condition.get("value", [None, None])
                    if low is None or high is None:
                        return (
                            pd.DataFrame(columns=df.columns),
                            {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                        )
                    numeric_mask &= numeric_series.between(float(low), float(high), inclusive="both").fillna(False)
                elif operator == ">":
                    value = condition.get("value")
                    if value is None:
                        return (
                            pd.DataFrame(columns=df.columns),
                            {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                        )
                    numeric_mask &= (numeric_series > float(value)).fillna(False)
                elif operator == "<":
                    value = condition.get("value")
                    if value is None:
                        return (
                            pd.DataFrame(columns=df.columns),
                            {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                        )
                    numeric_mask &= (numeric_series < float(value)).fillna(False)
                else:
                    return (
                        pd.DataFrame(columns=df.columns),
                        {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
                    )
            mask &= numeric_mask
        else:
            return (
                pd.DataFrame(columns=df.columns),
                {"code": "invalid_scoped_subset", "message": "Retrieved scope could not be reconstructed for analytics"},
            )

    scoped_df = scoped_df.loc[mask].copy()
    if scoped_df.empty:
        return scoped_df, {"code": "empty_scoped_subset", "message": "No scoped rows were available for analytics"}

    return scoped_df, None


def _numeric_aggregations(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "mean": _json_safe(numeric.mean()),
        "median": _json_safe(numeric.median()),
        "min": _json_safe(numeric.min()),
        "max": _json_safe(numeric.max()),
    }


def _categorical_counts(series: pd.Series) -> dict[str, int]:
    values = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
    if not values:
        return {}
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {key: int(value) for key, value in ordered[:5]}


def _grouped_metric_summary(filtered_df: pd.DataFrame, categorical_columns: list[str], numeric_columns: list[str], question: str) -> list[dict[str, Any]]:
    if filtered_df.empty or not categorical_columns or not numeric_columns:
        return []
    category_col = categorical_columns[0]
    metric_col = numeric_columns[0]
    normalized_question = _normalize_text(question)
    ascending = any(term in normalized_question for term in ("lowest", "least", "smallest", "bottom"))
    grouped = (
        filtered_df.groupby(category_col, dropna=True)[metric_col]
        .sum()
        .reset_index()
        .sort_values(metric_col, ascending=ascending)
        .head(5)
    )
    return _serialize_records(grouped.to_dict(orient="records"))


def retrieve_context(question: str, df: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "columns_used": [],
            "filters_applied": {},
            "time_column": None,
            "time_range": {"start": None, "end": None},
            "aggregations": {},
            "sample_rows": [],
            "row_count": 0,
            "numeric_columns_in_scope": [],
            "categorical_columns_in_scope": [],
        }

    keywords = extract_keywords(question)
    matched_columns = match_columns(keywords, [str(column) for column in df.columns])
    filters = detect_filters(question, df)
    filtered_df, filters_applied = _apply_filters(df, filters)
    evidence_df = filtered_df if filters_applied else df.copy()

    time_col = infer_time_column(evidence_df if not evidence_df.empty else df)
    time_range = {"start": None, "end": None}
    if time_col is not None and time_col in evidence_df.columns:
        time_series = _series_to_datetime(evidence_df[time_col]).dropna()
        if not time_series.empty:
            time_range = {
                "start": _json_safe(time_series.min()),
                "end": _json_safe(time_series.max()),
            }

    columns_used = _dedupe_preserve_order(matched_columns + list(filters_applied.keys()) + ([time_col] if time_col else []))
    if len(columns_used) < min(3, len(df.columns)):
        for column in [str(column) for column in df.columns]:
            if column not in columns_used:
                columns_used.append(column)
            if len(columns_used) >= min(3, len(df.columns)):
                break
    columns_used = columns_used[: min(8, len(df.columns))]

    numeric_columns_in_scope = [column for column in columns_used if column in evidence_df.columns and _is_numeric_series(evidence_df[column])]
    categorical_columns_in_scope = [column for column in columns_used if column in evidence_df.columns and column not in numeric_columns_in_scope and column != time_col]

    aggregations: dict[str, Any] = {}
    for column in numeric_columns_in_scope:
        stats = _numeric_aggregations(evidence_df[column])
        if stats:
            aggregations[column] = stats
    for column in categorical_columns_in_scope:
        counts = _categorical_counts(evidence_df[column])
        if counts:
            aggregations[column] = counts

    grouped_summary = _grouped_metric_summary(evidence_df, categorical_columns_in_scope, numeric_columns_in_scope, question)
    if grouped_summary:
        aggregations["grouped_metric_summary"] = grouped_summary

    sample_columns = [column for column in columns_used if column in evidence_df.columns]
    sample_rows = _serialize_records(evidence_df[sample_columns].head(10).to_dict(orient="records")) if sample_columns else []

    return {
        "columns_used": columns_used,
        "filters_applied": filters_applied,
        "time_column": time_col,
        "time_range": time_range,
        "aggregations": aggregations,
        "sample_rows": sample_rows,
        "row_count": int(len(evidence_df)),
        "numeric_columns_in_scope": numeric_columns_in_scope,
        "categorical_columns_in_scope": categorical_columns_in_scope,
    }


def _semantic_metric_hits(question: str) -> dict[str, list[str]]:
    normalized_question = _normalize_text(question)
    hits: dict[str, list[str]] = {}
    for canonical_name, phrases in METRIC_SYNONYMS.items():
        matched = [phrase for phrase in phrases if phrase in normalized_question]
        if matched:
            hits[canonical_name] = matched
    return hits


def _find_best_semantic_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    numeric_columns = _numeric_columns(df)
    if not numeric_columns:
        return None

    phrases = METRIC_SYNONYMS.get(canonical_name, ())
    best_match: tuple[float, str] | None = None
    for column in numeric_columns:
        normalized_column = _normalize_text(column)
        if canonical_name == "net_inflows" and "net" not in normalized_column and "nnb" not in normalized_column:
            continue
        score = 0.0
        for phrase in phrases:
            normalized_phrase = _normalize_text(phrase)
            if normalized_phrase == normalized_column:
                score = max(score, 100.0)
            elif normalized_phrase in normalized_column or normalized_column in normalized_phrase:
                score = max(score, 70.0)
            else:
                score = max(score, _string_similarity(normalized_phrase, normalized_column) * 60.0)
        if best_match is None or score > best_match[0]:
            best_match = (score, column)
    if best_match and best_match[0] >= 35.0:
        return best_match[1]
    return None


def infer_target_metric(question: str, df: pd.DataFrame, context: dict[str, Any]) -> dict[str, Any]:
    scoped_numeric = [str(column) for column in context.get("numeric_columns_in_scope", []) if column in df.columns]
    numeric_columns = scoped_numeric or _numeric_columns(df)
    if not numeric_columns:
        return {
            "primary_metric": None,
            "secondary_metrics": [],
            "matched_by": "none",
        }

    normalized_question = _normalize_text(question)
    semantic_hits = _semantic_metric_hits(question)

    if "net inflow" in normalized_question or "net inflows" in normalized_question or "net flow" in normalized_question or "net flows" in normalized_question:
        direct_net = _find_best_semantic_column(df[numeric_columns], "net_inflows")
        if direct_net is not None:
            secondary = [column for column in numeric_columns if column != direct_net][:3]
            return {
                "primary_metric": direct_net,
                "secondary_metrics": secondary,
                "matched_by": "exact" if "net" in _normalize_text(direct_net) else "fuzzy",
            }
        inflow_col = _find_best_semantic_column(df[numeric_columns], "inflow")
        outflow_col = _find_best_semantic_column(df[numeric_columns], "outflow")
        if inflow_col and outflow_col:
            return {
                "primary_metric": "net_inflows",
                "secondary_metrics": [inflow_col, outflow_col],
                "matched_by": "derived",
                "derived_from": {
                    "operation": "subtract",
                    "left": inflow_col,
                    "right": outflow_col,
                },
            }

    ranked: list[tuple[float, int, str, str]] = []
    for index, column in enumerate(numeric_columns):
        normalized_column = _normalize_text(column)
        score = 0.0
        matched_by = "none"
        if normalized_column in normalized_question:
            score += 100.0
            matched_by = "exact"
        else:
            token_overlap = len(set(_tokenize(column)) & set(_tokenize(question)))
            score += token_overlap * 20.0
            for phrases in semantic_hits.values():
                for phrase in phrases:
                    normalized_phrase = _normalize_text(phrase)
                    if normalized_phrase == normalized_column:
                        score += 90.0
                        matched_by = "exact"
                    elif normalized_phrase in normalized_column or normalized_column in normalized_phrase:
                        score += 50.0
                        matched_by = "fuzzy" if matched_by == "none" else matched_by
                    else:
                        similarity = _string_similarity(normalized_phrase, normalized_column)
                        score += similarity * 35.0
                        if similarity >= 0.8 and matched_by == "none":
                            matched_by = "fuzzy"
        ranked.append((score, index, column, matched_by))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_score, _, top_column, matched_by = ranked[0]
    if top_score >= 20.0:
        return {
            "primary_metric": top_column,
            "secondary_metrics": [column for _, _, column, _ in ranked if column != top_column][:3],
            "matched_by": matched_by if matched_by != "none" else "fuzzy",
        }

    if len(numeric_columns) == 1:
        return {
            "primary_metric": numeric_columns[0],
            "secondary_metrics": [],
            "matched_by": "fallback",
        }

    return {
        "primary_metric": None,
        "secondary_metrics": [],
        "matched_by": "none",
    }


def _resolve_metric_frame(metric_info: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    primary_metric = metric_info.get("primary_metric")
    if not primary_metric:
        return df.copy(), None
    if primary_metric in df.columns:
        return df.copy(), str(primary_metric)

    derived_from = metric_info.get("derived_from")
    if isinstance(derived_from, dict) and derived_from.get("operation") == "subtract":
        left = derived_from.get("left")
        right = derived_from.get("right")
        if left in df.columns and right in df.columns:
            working_df = df.copy()
            working_df["net_inflows"] = (
                pd.to_numeric(working_df[left], errors="coerce").fillna(0.0)
                - pd.to_numeric(working_df[right], errors="coerce").fillna(0.0)
            )
            return working_df, "net_inflows"
    return df.copy(), None


def prepare_time_series(df: pd.DataFrame, time_col: str, metric_col: str) -> pd.DataFrame:
    if df.empty or time_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame(columns=[time_col, metric_col])

    ts_df = pd.DataFrame(
        {
            time_col: pd.to_datetime(df[time_col], errors="coerce", format="mixed"),
            metric_col: pd.to_numeric(df[metric_col], errors="coerce"),
        }
    ).dropna(subset=[time_col, metric_col])

    if ts_df.empty:
        return pd.DataFrame(columns=[time_col, metric_col])

    ts_df = (
        ts_df.groupby(time_col, as_index=False)[metric_col]
        .sum()
        .sort_values(time_col, ascending=True)
        .reset_index(drop=True)
    )
    return ts_df[[time_col, metric_col]]


def _compute_regression(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size < 2:
        return None, None
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    predicted = (slope * x) + intercept
    ss_res = float(np.sum((values - predicted) ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    if ss_tot <= 0:
        r2 = 1.0 if ss_res <= 1e-12 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    return float(slope), max(0.0, min(float(r2), 1.0))


def _compute_window_comparison(values: np.ndarray) -> dict[str, Any]:
    result = {
        "recent_window_mean": None,
        "prior_window_mean": None,
        "delta": None,
        "pct_delta": None,
    }
    if values.size == 0:
        return result
    if values.size == 1:
        result["recent_window_mean"] = _json_safe(values[-1])
        return result

    window_size = max(1, values.size // 2)
    if (window_size * 2) <= values.size:
        recent_window = values[-window_size:]
        prior_window = values[-(window_size * 2) : -window_size]
    else:
        recent_window = values[-1:]
        prior_window = values[-2:-1]

    if prior_window.size == 0:
        return result

    recent_mean = float(np.mean(recent_window))
    prior_mean = float(np.mean(prior_window))
    delta = recent_mean - prior_mean
    pct_delta = None if abs(prior_mean) < 1e-12 else (delta / abs(prior_mean)) * 100.0
    return {
        "recent_window_mean": _json_safe(recent_mean),
        "prior_window_mean": _json_safe(prior_mean),
        "delta": _json_safe(delta),
        "pct_delta": _json_safe(pct_delta),
    }


def _directional_consistency(differences: np.ndarray) -> float:
    if differences.size == 0:
        return 0.0
    signs = np.sign(differences)
    non_zero = signs[signs != 0]
    if non_zero.size == 0:
        return 0.0
    return max(0.0, min(abs(float(np.mean(non_zero))), 1.0))


def _noise_level(volatility: float | None) -> str:
    if volatility is None:
        return "unknown"
    if volatility <= 0.05:
        return "low"
    if volatility <= 0.15:
        return "medium"
    return "high"


def _seasonality_hint(values: np.ndarray) -> str:
    if values.size < 8:
        return "unknown"
    differences = np.diff(values)
    if differences.size < 6:
        return "unknown"
    centered = differences - np.mean(differences)
    denominator = float(np.sum(centered**2))
    if denominator <= 1e-12:
        return "none"

    best_autocorr = -1.0
    max_lag = min(12, centered.size // 2)
    for lag in range(2, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        if left.size < 2 or right.size < 2:
            continue
        lag_denom = math.sqrt(float(np.sum(left**2)) * float(np.sum(right**2)))
        if lag_denom <= 1e-12:
            continue
        autocorr = float(np.sum(left * right)) / lag_denom
        best_autocorr = max(best_autocorr, autocorr)

    if best_autocorr >= 0.6:
        return "possible"
    if best_autocorr >= 0.35:
        return "weak"
    return "none"


def _signal_confidence(n_points: int, regression_r2: float | None, volatility: float | None, consistency: float, trend_strength: str) -> tuple[str, float]:
    score = 0.0
    if n_points >= 4:
        score += 0.2
    if n_points >= 8:
        score += 0.2
    if n_points >= 12:
        score += 0.1

    score += (regression_r2 or 0.0) * 0.25

    noise = _noise_level(volatility)
    if noise == "low":
        score += 0.15
    elif noise == "medium":
        score += 0.05
    elif noise == "high":
        score -= 0.1

    score += consistency * 0.15
    if trend_strength == "strong":
        score += 0.1
    elif trend_strength == "moderate":
        score += 0.05

    score = max(0.0, min(score, 1.0))
    if score >= 0.7:
        return "high", round(score, 2)
    if score >= 0.45:
        return "medium", round(score, 2)
    return "low", round(score, 2)


def compute_analytical_signals(ts_df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    signals = _empty_signals()
    if ts_df.empty or metric_col not in ts_df.columns:
        return signals

    values = pd.to_numeric(ts_df[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
    signals["n_points"] = int(values.size)
    if values.size == 0:
        return signals

    signals["latest_value"] = _json_safe(values[-1])
    if values.size >= 2:
        previous_value = values[-2]
        signals["previous_value"] = _json_safe(previous_value)
        if abs(previous_value) >= 1e-12:
            signals["pct_change_latest"] = _json_safe(((values[-1] - previous_value) / abs(previous_value)) * 100.0)
        elif abs(values[-1]) < 1e-12:
            signals["pct_change_latest"] = 0.0

    short_window = max(1, min(3, values.size))
    long_window = max(1, min(6, values.size))
    moving_average_short = float(np.mean(values[-short_window:]))
    moving_average_long = float(np.mean(values[-long_window:]))
    signals["moving_average_short"] = _json_safe(moving_average_short)
    signals["moving_average_long"] = _json_safe(moving_average_long)

    slope, regression_r2 = _compute_regression(values)
    signals["slope"] = _json_safe(slope)
    signals["regression_r2"] = _json_safe(regression_r2)

    pct_changes = pd.Series(values).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = float(pct_changes.std(ddof=0)) if not pct_changes.empty else None
    signals["volatility"] = _json_safe(volatility)
    signals["noise_level"] = _noise_level(volatility)

    differences = np.diff(values)
    if differences.size >= 1:
        momentum_window = max(1, min(3, differences.size))
        recent_momentum = float(np.mean(differences[-momentum_window:]))
        prior_slice = differences[-(momentum_window * 2) : -momentum_window]
        prior_momentum = float(np.mean(prior_slice)) if prior_slice.size else 0.0
        signals["acceleration"] = _json_safe(recent_momentum - prior_momentum)

    window_comparison = _compute_window_comparison(values)
    signals["window_comparison"] = window_comparison

    directional_components: list[float] = []
    if slope is not None:
        mean_abs = max(float(np.mean(np.abs(values))), 1e-9)
        directional_components.append(slope / mean_abs)
    if moving_average_long:
        directional_components.append((moving_average_short - moving_average_long) / max(abs(moving_average_long), 1e-9))
    if signals["pct_change_latest"] is not None:
        directional_components.append(float(signals["pct_change_latest"]) / 100.0)
    if window_comparison["pct_delta"] is not None:
        directional_components.append(float(window_comparison["pct_delta"]) / 100.0)

    directional_score = float(np.mean(directional_components)) if directional_components else 0.0
    if directional_score > 0.01:
        trend_direction = "up"
    elif directional_score < -0.01:
        trend_direction = "down"
    else:
        trend_direction = "flat"
    signals["trend_direction"] = trend_direction if values.size >= 2 else "unknown"

    consistency = _directional_consistency(differences)
    if abs(directional_score) >= 0.08 and consistency >= 0.75 and (regression_r2 or 0.0) >= 0.55:
        trend_strength = "strong"
    elif abs(directional_score) >= 0.03 and consistency >= 0.45:
        trend_strength = "moderate"
    else:
        trend_strength = "weak" if values.size >= 2 else "unknown"
    signals["trend_strength"] = trend_strength

    confidence_label, confidence_score = _signal_confidence(int(values.size), regression_r2, volatility, consistency, trend_strength)
    signals["signal_confidence"] = confidence_label
    signals["signal_confidence_score"] = confidence_score
    signals["seasonality_hint"] = _seasonality_hint(values)
    return signals


def derive_likelihood_assessment(signals: dict[str, Any], question_type: dict[str, Any]) -> dict[str, Any]:
    target_direction = question_type.get("target_direction", "unknown")
    intent_type = question_type.get("intent_type", "descriptive")
    if target_direction not in {"up", "down"} and intent_type not in {"trend", "forecast_like", "what_if"}:
        return {
            "likelihood_label": "unclear",
            "likelihood_score": 0.5,
            "reasoning_factors": ["question does not specify a directional continuation"],
        }

    if target_direction not in {"up", "down"}:
        target_direction = signals.get("trend_direction", "unknown")
    if target_direction not in {"up", "down"}:
        return {
            "likelihood_label": "unclear",
            "likelihood_score": 0.5,
            "reasoning_factors": ["signals do not show a clear directional basis"],
        }

    direction_multiplier = 1.0 if target_direction == "up" else -1.0
    score = 0.5
    reasoning_factors: list[str] = []

    slope = signals.get("slope")
    pct_change_latest = signals.get("pct_change_latest")
    acceleration = signals.get("acceleration")
    volatility = signals.get("volatility")
    regression_r2 = signals.get("regression_r2")
    trend_direction = signals.get("trend_direction")
    window_pct_delta = (signals.get("window_comparison") or {}).get("pct_delta")

    if slope is not None:
        if float(slope) * direction_multiplier > 0:
            score += 0.14
            reasoning_factors.append("slope supports the observed direction")
        elif float(slope) * direction_multiplier < 0:
            score -= 0.14
            reasoning_factors.append("slope runs against the observed direction")

    if pct_change_latest is not None:
        if float(pct_change_latest) * direction_multiplier > 0:
            score += 0.12
            reasoning_factors.append("latest percentage change supports continuation")
        elif float(pct_change_latest) * direction_multiplier < 0:
            score -= 0.12
            reasoning_factors.append("latest percentage change runs against continuation")

    if window_pct_delta is not None:
        if float(window_pct_delta) * direction_multiplier > 0:
            score += 0.14
            if target_direction == "down":
                reasoning_factors.append("recent window mean is below the prior window mean")
            else:
                reasoning_factors.append("recent window mean is above the prior window mean")
        elif float(window_pct_delta) * direction_multiplier < 0:
            score -= 0.14
            if target_direction == "down":
                reasoning_factors.append("recent window mean is above the prior window mean")
            else:
                reasoning_factors.append("recent window mean is below the prior window mean")

    if trend_direction == target_direction:
        score += 0.1
        reasoning_factors.append("trend direction aligns with the continuation being asked about")
    elif trend_direction in {"up", "down"} and trend_direction != target_direction:
        score -= 0.1
        reasoning_factors.append("trend direction does not align with the continuation being asked about")

    if acceleration is not None:
        if float(acceleration) * direction_multiplier > 0:
            score += 0.08
            reasoning_factors.append("acceleration points toward a strengthening move")
        elif float(acceleration) * direction_multiplier < 0:
            score -= 0.08
            reasoning_factors.append("acceleration points toward weakening momentum")

    if volatility is not None and float(volatility) > 0.15:
        score -= 0.08
        reasoning_factors.append("volatility is elevated, which limits certainty")
    elif volatility is not None and float(volatility) <= 0.05:
        score += 0.04
        reasoning_factors.append("volatility is contained, which supports a cleaner signal")

    if regression_r2 is not None and float(regression_r2) < 0.3:
        score -= 0.06
        reasoning_factors.append("regression fit is weak")
    elif regression_r2 is not None and float(regression_r2) >= 0.6:
        score += 0.05
        reasoning_factors.append("regression fit is reasonably strong")

    confidence_score = float(signals.get("signal_confidence_score") or 0.0)
    score += (confidence_score - 0.5) * 0.18
    score = max(0.05, min(score, 0.95))

    if confidence_score < 0.2 and signals.get("trend_direction") == "unknown":
        label = "unclear"
    elif confidence_score < 0.45:
        label = "low"
    elif volatility is not None and float(volatility) > 1.0:
        label = "moderate" if score >= 0.52 else "low"
    elif score >= 0.7 and confidence_score >= 0.7:
        label = "high"
    elif score >= 0.52:
        label = "moderate"
    else:
        label = "low"

    return {
        "likelihood_label": label,
        "likelihood_score": round(score, 2),
        "reasoning_factors": _dedupe_preserve_order(reasoning_factors)[:5] or ["signals are mixed and do not strongly support one direction"],
    }


def build_analytical_context(question: str, metric_info: dict[str, Any], signals: dict[str, Any], likelihood: dict[str, Any], ts_df: pd.DataFrame, time_col: str | None) -> dict[str, Any]:
    if ts_df.empty or time_col is None:
        return {
            "metric_selected": metric_info.get("primary_metric"),
            "series_range": {"start": None, "end": None},
            "observations": 0,
            "signals_summary": {
                "trend_direction": signals.get("trend_direction", "unknown"),
                "trend_strength": signals.get("trend_strength", "unknown"),
                "pct_change_latest": _json_safe(signals.get("pct_change_latest")),
                "slope": _json_safe(signals.get("slope")),
                "volatility": _json_safe(signals.get("volatility")),
                "acceleration": _json_safe(signals.get("acceleration")),
                "signal_confidence": signals.get("signal_confidence", "low"),
            },
            "likelihood_assessment": {
                "label": likelihood.get("likelihood_label", "unclear"),
                "score": _json_safe(likelihood.get("likelihood_score")),
                "reasoning_factors": [str(item) for item in likelihood.get("reasoning_factors", [])],
            },
            "window_comparison": {key: _json_safe(value) for key, value in (signals.get("window_comparison") or {}).items()},
            "sample_points": [],
        }

    series_range = {
        "start": _json_safe(ts_df[time_col].min()),
        "end": _json_safe(ts_df[time_col].max()),
    }
    sample_points = _serialize_records(
        (
            ts_df[[time_col, metric_info.get("primary_metric")]]
            .iloc[np.unique(np.linspace(0, len(ts_df) - 1, num=min(8, len(ts_df)), dtype=int))]
            .to_dict(orient="records")
        )
    )
    return {
        "metric_selected": metric_info.get("primary_metric"),
        "series_range": series_range,
        "observations": int(len(ts_df)),
        "signals_summary": {
            "trend_direction": signals.get("trend_direction", "unknown"),
            "trend_strength": signals.get("trend_strength", "unknown"),
            "pct_change_latest": _json_safe(signals.get("pct_change_latest")),
            "slope": _json_safe(signals.get("slope")),
            "volatility": _json_safe(signals.get("volatility")),
            "acceleration": _json_safe(signals.get("acceleration")),
            "signal_confidence": signals.get("signal_confidence", "low"),
        },
        "likelihood_assessment": {
            "label": likelihood.get("likelihood_label", "unclear"),
            "score": _json_safe(likelihood.get("likelihood_score")),
            "reasoning_factors": [str(item) for item in likelihood.get("reasoning_factors", [])],
        },
        "window_comparison": {key: _json_safe(value) for key, value in (signals.get("window_comparison") or {}).items()},
        "sample_points": sample_points,
    }


def compute_analytics_if_needed(question_type: dict[str, Any], context: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    analytics = _empty_analytics()
    if not question_type.get("needs_analytics"):
        return analytics

    question = str(question_type.get("question_text", "") or "")
    scoped_df, scoped_error = _reconstruct_scoped_subset(df, context)
    if scoped_error is not None:
        analytics["analytics_error"] = scoped_error
        return analytics

    if scoped_df.empty:
        analytics["analytics_error"] = {"code": "empty_scoped_subset", "message": "No scoped rows were available for analytics"}
        return analytics

    time_col = context.get("time_column")
    if not time_col or time_col not in scoped_df.columns:
        analytics["analytics_error"] = {"code": "missing_time_column", "message": "Analytical inference requires a usable time column"}
        return analytics

    metric_info = infer_target_metric(question, scoped_df, context)
    analytics["metric_info"] = metric_info
    working_df, metric_col = _resolve_metric_frame(metric_info, scoped_df)
    if metric_col is None:
        analytics["analytics_error"] = {"code": "missing_metric", "message": "No suitable metric could be inferred for analytics"}
        return analytics

    ts_df = prepare_time_series(working_df, time_col, metric_col)
    if ts_df.empty:
        analytics["analytics_error"] = {"code": "insufficient_series", "message": "No usable time series was available for analytics"}
        return analytics

    metric_info = dict(metric_info)
    metric_info["primary_metric"] = metric_col
    signals = compute_analytical_signals(ts_df, metric_col)
    likelihood = derive_likelihood_assessment(signals, question_type)
    analytical_context = build_analytical_context(question, metric_info, signals, likelihood, ts_df, time_col)

    analytics["analytics_used"] = True
    analytics["metric_info"] = metric_info
    analytics["signals"] = signals
    analytics["likelihood"] = likelihood
    analytics["analytical_context"] = analytical_context
    return analytics


def determine_not_enough_evidence(question_type: dict[str, Any], context: dict[str, Any], analytics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []

    if int(context.get("row_count", 0)) == 0:
        reasons.append("no rows matched the question")
    if not context.get("columns_used"):
        reasons.append("no relevant columns were matched")
    if "unsupported_variable" in question_type.get("risk_flags", []):
        reasons.append("requested variable is not supported by the dataset")
    if question_type.get("needs_analytics") and not context.get("time_column"):
        reasons.append("time-based inference requested but no usable time column exists")
    if analytics.get("analytics_error"):
        reasons.append(str(analytics["analytics_error"].get("message", "analytics could not be computed")))
    if question_type.get("needs_analytics") and int((analytics.get("signals") or {}).get("n_points", 0)) < 4:
        reasons.append("too few observations for a reliable trend inference")
    if question_type.get("intent_type") == "causal":
        reasons.append("causal claims are not supported by observational data alone")
    if "strong_forecast_request" in question_type.get("risk_flags", []) and (analytics.get("signals") or {}).get("signal_confidence") == "low":
        reasons.append("confidence is too low for a strong forward-looking claim")
    if question_type.get("needs_analytics") and (analytics.get("signals") or {}).get("signal_confidence") == "low" and question_type.get("intent_type") in {"forecast_like", "what_if"}:
        reasons.append("signal confidence is low for the requested forward-looking inference")
    if question_type.get("needs_direct_answer") and not context.get("aggregations") and not context.get("sample_rows"):
        reasons.append("granularity is insufficient for a grounded direct answer")

    return {
        "not_enough_evidence": bool(reasons),
        "reasons": _dedupe_preserve_order(reasons),
    }


def build_prompt_with_data_and_signals(
    question: str,
    question_type: dict[str, Any],
    context: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
) -> str:
    if evidence_decision.get("not_enough_evidence"):
        mode = "Not Enough Evidence"
        response_contract = (
            "Claude must:\n"
            "- clearly state there is not enough evidence\n"
            "- explain why using the listed evidence gaps\n"
            "- suggest what additional data would be needed\n"
            "- do not force an answer"
        )
    elif question_type.get("needs_analytics") and analytics.get("analytics_used"):
        mode = "Analytical Inference Answer"
        response_contract = (
            "Claude must:\n"
            "- interpret ONLY the provided analytical signals\n"
            "- use wording such as 'based on the current observed trend'\n"
            "- state that this is directional inference, not a forecast\n"
            "- mention signal confidence explicitly\n"
            "- avoid deterministic future claims\n"
            "- avoid inventing external causes"
        )
    else:
        mode = "Grounded Data Answer"
        response_contract = (
            "Claude must:\n"
            "- answer directly from the dataset context\n"
            "- prefer exact comparisons, rankings, and quantified statements\n"
            "- avoid speculative interpretation\n"
            "- do not invent missing values\n"
            "- if exact support is unavailable, fall back to Not Enough Evidence"
        )

    context_payload = json.dumps(context, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=_json_safe)
    analytical_context = analytics.get("analytical_context") if analytics.get("analytics_used") else {}
    analytical_payload = json.dumps(analytical_context, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=_json_safe)

    return (
        f"[USER QUESTION]\n{question.strip()}\n\n"
        f"[QUESTION MODE]\n{mode}\n\n"
        f"[DATA CONTEXT]\n{context_payload}\n\n"
        f"[ANALYTICAL CONTEXT]\n{analytical_payload}\n\n"
        f"[RESPONSE CONTRACT]\n{response_contract}\n\n"
        "[INSTRUCTIONS]\n"
        "Use only the provided data and analytical context.\n"
        "Do not use external knowledge or generic latent knowledge.\n"
        "Do not hallucinate causes, values, or confidence levels.\n"
        "If evidence is limited, say so plainly.\n"
        "Use professional, quantified wording.\n"
        "Do not produce strong certainty unless the contract explicitly permits it."
    )


def claude_generate(prompt: str, claude_callable: Callable[[str], str]) -> str:
    try:
        response = claude_callable(prompt)
        text = str(response or "").strip()
        return text or CLAUDE_GENERATION_FAILURE_RESPONSE
    except Exception:
        return CLAUDE_GENERATION_FAILURE_RESPONSE


def response_validator(
    response: str,
    question_type: dict[str, Any],
    context: dict[str, Any],
    analytics: dict[str, Any],
    evidence_decision: dict[str, Any],
) -> dict[str, Any]:
    expected_mode = _expected_response_mode(question_type, analytics, evidence_decision)
    if _is_safe_fallback_response(response, expected_mode):
        confidence_by_mode = {
            "grounded_data": 0.7,
            "analytical_inference": 0.72,
            "not_enough_evidence": 0.85,
        }
        return _safe_validation_payload(expected_mode, confidence_by_mode[expected_mode])

    normalized_response = _normalize_text(response)
    has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", normalized_response))
    uncertainty_markers = (
        "based on",
        "directional inference",
        "not a forecast",
        "confidence",
        "uncertain",
        "limited",
        "may",
        "could",
    )
    insufficiency_markers = (
        "not enough evidence",
        "insufficient",
        "dataset does not support",
        "not enough history",
        "missing",
    )
    strong_forecast_markers = ("will definitely", "certainly will", "guaranteed", "will happen")
    analytical_markers = ("trend", "slope", "volatility", "acceleration", "confidence", "observed", "direction")

    if any(marker in normalized_response for marker in insufficiency_markers):
        detected_mode = "not_enough_evidence"
    elif any(marker in normalized_response for marker in analytical_markers):
        detected_mode = "analytical_inference"
    elif has_number:
        detected_mode = "grounded_data"
    else:
        detected_mode = "unclear"

    grounded = False
    confidence = 0.0
    violations: list[str] = []

    if expected_mode == "grounded_data":
        grounded = has_number or bool(context.get("aggregations")) or bool(context.get("sample_rows"))
        confidence = 0.75 if grounded else 0.25
        if not has_number:
            violations.append("missing_quantification")
        if any(marker in normalized_response for marker in strong_forecast_markers):
            violations.append("ungrounded_claim")
        if detected_mode != "grounded_data":
            violations.append("mode_mismatch")
    elif expected_mode == "analytical_inference":
        has_signal_reference = any(marker in normalized_response for marker in analytical_markers)
        has_uncertainty = any(marker in normalized_response for marker in uncertainty_markers)
        grounded = has_signal_reference
        confidence = 0.8 if grounded and has_uncertainty else 0.4
        if not has_signal_reference:
            violations.append("generic_answer")
        if not has_uncertainty:
            violations.append("missing_uncertainty")
        if any(marker in normalized_response for marker in strong_forecast_markers):
            violations.append("strong_forecast_language")
        if detected_mode not in {"analytical_inference", "grounded_data"}:
            violations.append("mode_mismatch")
    else:
        has_insufficiency = any(marker in normalized_response for marker in insufficiency_markers)
        grounded = has_insufficiency
        confidence = 0.85 if grounded else 0.2
        if not has_insufficiency:
            violations.append("generic_answer")
        if any(marker in normalized_response for marker in strong_forecast_markers):
            violations.append("ungrounded_claim")
        if detected_mode != "not_enough_evidence":
            violations.append("mode_mismatch")

    return {
        "valid": not violations,
        "response_mode_detected": detected_mode,
        "grounded": grounded,
        "confidence": round(confidence, 2),
        "violations": _dedupe_preserve_order(violations),
    }


def chat_handler(question: str, df: pd.DataFrame, claude_callable: Callable[[str], str]) -> dict[str, Any]:
    start_time = time.perf_counter()
    question_type = classify_question_type(question, df)
    context = retrieve_context(question, df)
    try:
        analytics = compute_analytics_if_needed(question_type, context, df)
    except Exception as exc:
        logger.exception("compute_analytics_if_needed failed")
        analytics = _empty_analytics()
        analytics["analytics_error"] = {"message": str(exc)}
    evidence_decision = determine_not_enough_evidence(question_type, context, analytics)
    prompt = build_prompt_with_data_and_signals(question, question_type, context, analytics, evidence_decision)
    raw_response = claude_generate(prompt, claude_callable)
    response = raw_response
    validation = _validate_response_safely(response, question_type, context, analytics, evidence_decision)
    final_validation = validation
    final_mode = _resolve_final_mode(question_type, analytics, evidence_decision)
    validation_gate_triggered = False

    if not validation.get("valid", False) and response != CLAUDE_GENERATION_FAILURE_RESPONSE:
        validation_gate_triggered = True
        response, final_mode = _build_validation_fallback(
            question_type=question_type,
            context=context,
            analytics=analytics,
            evidence_decision=evidence_decision,
            validation=validation,
        )
        fallback_validation = _validate_response_safely(
            response,
            question_type,
            context,
            analytics,
            {"not_enough_evidence": final_mode == "not_enough_evidence", "reasons": evidence_decision.get("reasons", [])},
        )
        if not fallback_validation.get("valid", False):
            response = INSUFFICIENT_EVIDENCE_RESPONSE
            final_mode = "not_enough_evidence"
            fallback_validation = _validate_response_safely(
                response,
                question_type,
                context,
                analytics,
                {"not_enough_evidence": True, "reasons": evidence_decision.get("reasons", [])},
            )
        final_validation = fallback_validation

    execution_time_ms = int((time.perf_counter() - start_time) * 1000)
    _log_event(
        {
            "question": question,
            "classified_mode": question_type.get("question_mode"),
            "matched_columns": context.get("columns_used", []),
            "row_count": context.get("row_count", 0),
            "analytics_used": analytics.get("analytics_used", False),
            "selected_metric": (analytics.get("metric_info") or {}).get("primary_metric"),
            "signal_confidence": (analytics.get("signals") or {}).get("signal_confidence"),
            "final_mode": final_mode,
            "execution_time_ms": execution_time_ms,
            "validation_outcome": validation.get("valid"),
            "returned_validation_outcome": final_validation.get("valid"),
            "validation_gate_triggered": validation_gate_triggered,
        }
    )

    return {
        "question": question,
        "question_type": question_type,
        "context": context,
        "analytics": analytics,
        "evidence_decision": evidence_decision,
        "prompt": prompt,
        "response": response,
        "validation": validation,
        "final_validation": final_validation,
        "final_mode": final_mode,
    }


__all__ = [
    "chat_handler",
    "claude_generate",
    "classify_question_type",
    "build_prompt_with_data_and_signals",
    "compute_analytics_if_needed",
    "compute_analytical_signals",
    "derive_likelihood_assessment",
    "detect_filters",
    "determine_not_enough_evidence",
    "extract_keywords",
    "infer_target_metric",
    "infer_time_column",
    "match_columns",
    "prepare_time_series",
    "response_validator",
    "retrieve_context",
]
