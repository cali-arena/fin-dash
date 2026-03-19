from __future__ import annotations

import difflib
import json
import logging
import math
import re
import time
import unicodedata
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

logger = logging.getLogger(__name__)

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

METRIC_SYNONYMS: dict[str, tuple[str, ...]] = {
    "net_inflows": (
        "net inflow",
        "net inflows",
        "net flow",
        "net flows",
        "net subscriptions",
        "nnb",
        "net sales",
    ),
    "inflow": (
        "inflow",
        "inflows",
        "gross inflow",
        "gross inflows",
        "subscriptions",
    ),
    "outflow": (
        "outflow",
        "outflows",
        "redemptions",
        "withdrawals",
        "gross outflow",
        "gross outflows",
    ),
    "revenue": ("revenue", "revenues", "sales", "income"),
    "volume": ("volume", "turnover", "activity"),
    "count": ("count", "counts", "transactions", "clients", "customers"),
    "balance": ("balance", "balances", "aum", "assets", "holdings"),
}


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(value))


def _round_float(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.floating, float)):
        return _round_float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


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


def _error_payload(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


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


def _serialize_dataframe_points(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    if len(df) <= limit:
        sample_df = df.copy()
    else:
        indices = np.linspace(0, len(df) - 1, num=limit, dtype=int)
        sample_df = df.iloc[np.unique(indices)].copy()
    return [
        {key: _json_safe(value) for key, value in record.items()}
        for record in sample_df.to_dict(orient="records")
    ]


def detect_analytical_intent(question: str) -> dict[str, Any]:
    normalized_question = _normalize_text(question)
    tokens = set(_tokenize(question))

    what_if_patterns = (
        "what happens if",
        "if this continues",
        "suppose",
        "assuming",
        "assume",
    )
    risk_patterns = (
        "risk of decline",
        "risk of drop",
        "risk of falling",
        "chance of reducing",
        "chance of drop",
        "chance of decline",
        "likelihood of falling",
        "likelihood of decline",
        "likelihood of reducing",
    )
    trend_terms = {
        "trend",
        "direction",
        "trajectory",
        "trajectory",
        "momentum",
    }
    forecast_terms = {
        "likely",
        "probability",
        "probable",
        "chance",
        "scenario",
        "forecast",
        "outlook",
    }
    down_terms = {
        "decline",
        "decrease",
        "decreasing",
        "drop",
        "down",
        "fall",
        "falling",
        "reduce",
        "reducing",
        "weakening",
    }
    up_terms = {
        "increase",
        "increasing",
        "growth",
        "grow",
        "up",
        "rise",
        "rising",
        "improve",
        "improving",
    }

    if any(pattern in normalized_question for pattern in what_if_patterns):
        intent_type = "what_if"
    elif any(pattern in normalized_question for pattern in risk_patterns):
        intent_type = "risk"
    elif forecast_terms & tokens or "over next" in normalized_question or "next year" in normalized_question:
        intent_type = "forecast_like"
    elif trend_terms & tokens or "going up" in normalized_question or "going down" in normalized_question:
        intent_type = "trend"
    else:
        intent_type = "descriptive"

    if down_terms & tokens:
        target_direction = "down"
    elif up_terms & tokens:
        target_direction = "up"
    else:
        target_direction = "unknown"

    if any(phrase in normalized_question for phrase in ("next month", "next 30 days", "next 90 days", "near term")):
        time_horizon = "short"
    elif any(phrase in normalized_question for phrase in ("next quarter", "next 6 months", "medium term", "next half")):
        time_horizon = "medium"
    elif any(phrase in normalized_question for phrase in ("next year", "over the next year", "long term", "next 12 months")):
        time_horizon = "long"
    else:
        time_horizon = "unknown"

    return {
        "needs_analytics": intent_type != "descriptive",
        "intent_type": intent_type,
        "target_direction": target_direction,
        "time_horizon": time_horizon,
    }


def infer_time_column(df: pd.DataFrame) -> str | None:
    if df.empty:
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
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
            valid_ratio = float(converted.notna().mean()) if len(series) else 0.0
            score += valid_ratio * 50.0
            if valid_ratio < 0.6 and not any(token in name for token in TIME_NAME_PRIORITY):
                continue

        for priority_index, token in enumerate(TIME_NAME_PRIORITY):
            if token in name:
                score += 40.0 - priority_index

        if score > 0:
            candidates.append((score, index, str(column)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return candidates[0][2]


def _semantic_metric_hits(question: str) -> dict[str, list[str]]:
    normalized_question = _normalize_text(question)
    hits: dict[str, list[str]] = {}
    for canonical_name, phrases in METRIC_SYNONYMS.items():
        matched_phrases = [phrase for phrase in phrases if phrase in normalized_question]
        if matched_phrases:
            hits[canonical_name] = matched_phrases
    return hits


def _score_metric_column(column_name: str, question: str, semantic_hits: dict[str, list[str]]) -> tuple[float, str]:
    normalized_column = _normalize_text(column_name)
    normalized_question = _normalize_text(question)
    question_tokens = set(_tokenize(question))
    column_tokens = set(_tokenize(column_name))
    score = 0.0
    matched_by = "none"

    if normalized_column and normalized_column in normalized_question:
        score += 100.0
        matched_by = "exact"

    overlap = len(question_tokens & column_tokens)
    if question_tokens and column_tokens:
        score += (overlap / max(len(column_tokens), 1)) * 40.0

    for phrases in semantic_hits.values():
        for phrase in phrases:
            normalized_phrase = _normalize_text(phrase)
            if normalized_phrase == normalized_column:
                score += 95.0
                matched_by = "exact" if matched_by == "none" else matched_by
            elif normalized_phrase in normalized_column or normalized_column in normalized_phrase:
                score += 55.0
                if matched_by == "none":
                    matched_by = "semantic_rules"
            else:
                similarity = _string_similarity(normalized_phrase, normalized_column)
                score += similarity * 35.0
                if similarity >= 0.8 and matched_by == "none":
                    matched_by = "fuzzy"

    similarity_to_question = _string_similarity(normalized_column, normalized_question)
    score += similarity_to_question * 10.0
    if similarity_to_question >= 0.9 and matched_by == "none":
        matched_by = "fuzzy"

    return score, matched_by


def _find_best_semantic_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    numeric_columns = _numeric_columns(df)
    if not numeric_columns:
        return None

    best_match: tuple[float, str] | None = None
    phrases = METRIC_SYNONYMS.get(canonical_name, ())
    for column in numeric_columns:
        normalized_column = _normalize_text(column)
        if canonical_name == "net_inflows" and "nnb" not in normalized_column and "net" not in normalized_column:
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


def infer_target_metric(question: str, df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = _numeric_columns(df)
    if not numeric_columns:
        return {
            "primary_metric": None,
            "secondary_metrics": [],
            "matched_by": "none",
        }

    semantic_hits = _semantic_metric_hits(question)
    ranked: list[tuple[float, int, str, str]] = []
    for index, column in enumerate(numeric_columns):
        score, matched_by = _score_metric_column(column, question, semantic_hits)
        ranked.append((score, index, column, matched_by))

    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_score, _, top_column, top_match_type = ranked[0]

    normalized_question = _normalize_text(question)
    asks_for_net_inflows = (
        "net inflow" in normalized_question
        or "net inflows" in normalized_question
        or "net flow" in normalized_question
        or "net flows" in normalized_question
    )

    if asks_for_net_inflows:
        direct_net_column = _find_best_semantic_column(df, "net_inflows")
        if direct_net_column is not None:
            secondary = [column for _, _, column, _ in ranked if column != direct_net_column][:3]
            return {
                "primary_metric": direct_net_column,
                "secondary_metrics": secondary,
                "matched_by": "semantic_rules",
            }

        inflow_column = _find_best_semantic_column(df, "inflow")
        outflow_column = _find_best_semantic_column(df, "outflow")
        if inflow_column and outflow_column:
            return {
                "primary_metric": "net_inflows",
                "secondary_metrics": [inflow_column, outflow_column],
                "matched_by": "semantic_rules",
                "derived_from": {
                    "operation": "subtract",
                    "left": inflow_column,
                    "right": outflow_column,
                },
            }

    if top_score < 20.0:
        return {
            "primary_metric": None,
            "secondary_metrics": [],
            "matched_by": "none",
        }

    secondary_metrics = [column for _, _, column, _ in ranked if column != top_column][:3]
    matched_by = top_match_type if top_match_type != "none" else ("fuzzy" if top_score >= 35.0 else "none")
    return {
        "primary_metric": top_column,
        "secondary_metrics": secondary_metrics,
        "matched_by": matched_by,
    }


def _resolve_metric_frame(metric_info: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    primary_metric = metric_info.get("primary_metric")
    if not primary_metric:
        return df.copy(), None, None

    if primary_metric in df.columns:
        return df.copy(), str(primary_metric), str(primary_metric)

    derived_from = metric_info.get("derived_from")
    if isinstance(derived_from, dict) and derived_from.get("operation") == "subtract":
        left_col = derived_from.get("left")
        right_col = derived_from.get("right")
        if left_col in df.columns and right_col in df.columns:
            working_df = df.copy()
            metric_name = "net_inflows"
            working_df[metric_name] = (
                pd.to_numeric(working_df[left_col], errors="coerce").fillna(0.0)
                - pd.to_numeric(working_df[right_col], errors="coerce").fillna(0.0)
            )
            return working_df, metric_name, "net_inflows"

    return df.copy(), None, None


def prepare_time_series(df: pd.DataFrame, time_col: str, metric_col: str) -> pd.DataFrame:
    if df.empty or time_col not in df.columns or metric_col not in df.columns:
        return pd.DataFrame(columns=[time_col, metric_col])

    time_series = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    metric_series = pd.to_numeric(df[metric_col], errors="coerce")

    ts_df = pd.DataFrame({time_col: time_series, metric_col: metric_series}).dropna(subset=[time_col, metric_col])
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
    n_points = int(values.size)
    if n_points == 0:
        return result
    if n_points == 1:
        result["recent_window_mean"] = _round_float(values[-1])
        return result

    window_size = max(1, n_points // 2)
    if (window_size * 2) <= n_points:
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
        "recent_window_mean": _round_float(recent_mean),
        "prior_window_mean": _round_float(prior_mean),
        "delta": _round_float(delta),
        "pct_delta": _round_float(pct_delta),
    }


def _classify_noise(volatility: float | None) -> str:
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
    baseline = np.diff(values)
    if baseline.size < 6:
        return "unknown"
    centered = baseline - np.mean(baseline)
    denominator = float(np.sum(centered**2))
    if denominator <= 1e-12:
        return "none"

    max_lag = min(12, centered.size // 2)
    best_autocorr = -1.0
    for lag in range(2, max_lag + 1):
        left = centered[:-lag]
        right = centered[lag:]
        if left.size < 2 or right.size < 2:
            continue
        numerator = float(np.sum(left * right))
        lag_denom = math.sqrt(float(np.sum(left**2)) * float(np.sum(right**2)))
        if lag_denom <= 1e-12:
            continue
        best_autocorr = max(best_autocorr, numerator / lag_denom)

    if best_autocorr >= 0.6:
        return "possible"
    if best_autocorr >= 0.35:
        return "weak"
    return "none"


def _directional_consistency(differences: np.ndarray) -> float:
    if differences.size == 0:
        return 0.0
    signs = np.sign(differences)
    non_zero = signs[signs != 0]
    if non_zero.size == 0:
        return 0.0
    dominant = abs(float(np.mean(non_zero)))
    return max(0.0, min(dominant, 1.0))


def _trend_direction_and_strength(
    slope: float | None,
    moving_average_short: float | None,
    moving_average_long: float | None,
    pct_change_latest: float | None,
    window_pct_delta: float | None,
    values: np.ndarray,
    regression_r2: float | None,
) -> tuple[str, str]:
    if values.size < 2 or slope is None:
        return "unknown", "unknown"

    mean_abs = max(float(np.mean(np.abs(values))), 1e-9)
    normalized_slope = slope / mean_abs
    score_components: list[float] = [normalized_slope]

    if moving_average_short is not None and moving_average_long is not None:
        score_components.append((moving_average_short - moving_average_long) / max(abs(moving_average_long), 1e-9))
    if pct_change_latest is not None:
        score_components.append(pct_change_latest / 100.0)
    if window_pct_delta is not None:
        score_components.append(window_pct_delta / 100.0)

    directional_score = float(np.mean(score_components)) if score_components else 0.0
    if directional_score > 0.01:
        direction = "up"
    elif directional_score < -0.01:
        direction = "down"
    else:
        direction = "flat"

    consistency = _directional_consistency(np.diff(values))
    absolute_signal = abs(directional_score)
    r2 = regression_r2 or 0.0
    if absolute_signal >= 0.08 and consistency >= 0.75 and r2 >= 0.55:
        strength = "strong"
    elif absolute_signal >= 0.03 and consistency >= 0.45:
        strength = "moderate"
    else:
        strength = "weak"

    return direction, strength


def _signal_confidence(
    n_points: int,
    regression_r2: float | None,
    volatility: float | None,
    consistency: float,
    trend_strength: str,
) -> tuple[str, float]:
    score = 0.0
    if n_points >= 4:
        score += 0.2
    if n_points >= 8:
        score += 0.2
    if n_points >= 12:
        score += 0.1

    score += (regression_r2 or 0.0) * 0.25

    noise_level = _classify_noise(volatility)
    if noise_level == "low":
        score += 0.15
    elif noise_level == "medium":
        score += 0.05
    elif noise_level == "high":
        score -= 0.1

    score += consistency * 0.15
    if trend_strength == "strong":
        score += 0.1
    elif trend_strength == "moderate":
        score += 0.05

    score = max(0.0, min(score, 1.0))
    if score >= 0.7:
        label = "high"
    elif score >= 0.45:
        label = "medium"
    else:
        label = "low"
    return label, round(score, 2)


def compute_analytical_signals(ts_df: pd.DataFrame, metric_col: str) -> dict[str, Any]:
    signals = _empty_signals()
    if ts_df.empty or metric_col not in ts_df.columns:
        return signals

    values = pd.to_numeric(ts_df[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
    n_points = int(values.size)
    signals["n_points"] = n_points
    if n_points == 0:
        return signals

    signals["latest_value"] = _round_float(values[-1])
    if n_points >= 2:
        previous_value = values[-2]
        signals["previous_value"] = _round_float(previous_value)
        if abs(previous_value) >= 1e-12:
            signals["pct_change_latest"] = _round_float(((values[-1] - previous_value) / abs(previous_value)) * 100.0)
        elif abs(values[-1]) < 1e-12:
            signals["pct_change_latest"] = 0.0

    short_window = max(1, min(3, n_points))
    long_window = max(1, min(6, n_points))
    signals["moving_average_short"] = _round_float(float(np.mean(values[-short_window:])))
    signals["moving_average_long"] = _round_float(float(np.mean(values[-long_window:])))

    slope, regression_r2 = _compute_regression(values)
    signals["slope"] = _round_float(slope)
    signals["regression_r2"] = _round_float(regression_r2)

    pct_changes = pd.Series(values).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = float(pct_changes.std(ddof=0)) if not pct_changes.empty else None
    signals["volatility"] = _round_float(volatility)
    signals["noise_level"] = _classify_noise(volatility)

    differences = np.diff(values)
    if differences.size >= 1:
        momentum_window = max(1, min(3, differences.size))
        recent_momentum = float(np.mean(differences[-momentum_window:]))
        prior_slice = differences[-(momentum_window * 2) : -momentum_window]
        if prior_slice.size == 0:
            prior_momentum = 0.0
        else:
            prior_momentum = float(np.mean(prior_slice))
        signals["acceleration"] = _round_float(recent_momentum - prior_momentum)

    window_comparison = _compute_window_comparison(values)
    signals["window_comparison"] = window_comparison

    trend_direction, trend_strength = _trend_direction_and_strength(
        slope=slope,
        moving_average_short=signals["moving_average_short"],
        moving_average_long=signals["moving_average_long"],
        pct_change_latest=signals["pct_change_latest"],
        window_pct_delta=window_comparison["pct_delta"],
        values=values,
        regression_r2=regression_r2,
    )
    signals["trend_direction"] = trend_direction
    signals["trend_strength"] = trend_strength

    consistency = _directional_consistency(differences)
    confidence_label, confidence_score = _signal_confidence(
        n_points=n_points,
        regression_r2=regression_r2,
        volatility=volatility,
        consistency=consistency,
        trend_strength=trend_strength,
    )
    signals["signal_confidence"] = confidence_label
    signals["signal_confidence_score"] = confidence_score
    signals["seasonality_hint"] = _seasonality_hint(values)

    return signals


def derive_risk_or_likelihood(signals: dict[str, Any], target_direction: str) -> dict[str, Any]:
    if target_direction not in {"up", "down"}:
        return {
            "likelihood_label": "unclear",
            "likelihood_score": 0.5,
            "reasoning_factors": ["target direction is not explicit in the question"],
        }

    direction_multiplier = 1.0 if target_direction == "up" else -1.0
    slope = signals.get("slope")
    pct_change_latest = signals.get("pct_change_latest")
    acceleration = signals.get("acceleration")
    window_pct_delta = (signals.get("window_comparison") or {}).get("pct_delta")
    trend_direction = signals.get("trend_direction")
    regression_r2 = signals.get("regression_r2")
    volatility = signals.get("volatility")
    confidence_score = float(signals.get("signal_confidence_score") or 0.0)

    score = 0.5
    reasoning_factors: list[str] = []

    if slope is not None:
        if slope * direction_multiplier > 0:
            score += 0.14
            reasoning_factors.append("slope supports the requested direction")
        elif slope * direction_multiplier < 0:
            score -= 0.14
            reasoning_factors.append("slope runs against the requested direction")

    if pct_change_latest is not None:
        if pct_change_latest * direction_multiplier > 0:
            score += 0.12
            reasoning_factors.append("latest percentage change supports the requested direction")
        elif pct_change_latest * direction_multiplier < 0:
            score -= 0.12
            reasoning_factors.append("latest percentage change runs against the requested direction")

    if window_pct_delta is not None:
        if window_pct_delta * direction_multiplier > 0:
            score += 0.14
            if target_direction == "down":
                reasoning_factors.append("recent window mean is below the prior window mean")
            else:
                reasoning_factors.append("recent window mean is above the prior window mean")
        elif window_pct_delta * direction_multiplier < 0:
            score -= 0.14
            if target_direction == "down":
                reasoning_factors.append("recent window mean is above the prior window mean")
            else:
                reasoning_factors.append("recent window mean is below the prior window mean")

    if trend_direction == target_direction:
        score += 0.12
        reasoning_factors.append("trend direction aligns with the requested direction")
    elif trend_direction in {"up", "down"} and trend_direction != target_direction:
        score -= 0.12
        reasoning_factors.append("trend direction points away from the requested direction")

    if acceleration is not None:
        if acceleration * direction_multiplier > 0:
            score += 0.08
            reasoning_factors.append("acceleration indicates strengthening movement in the requested direction")
        elif acceleration * direction_multiplier < 0:
            score -= 0.08
            reasoning_factors.append("acceleration indicates weakening movement in the requested direction")

    if volatility is not None and volatility > 0.15:
        score -= 0.08
        reasoning_factors.append("volatility is elevated, which reduces certainty")
    elif volatility is not None and volatility <= 0.05:
        score += 0.04
        reasoning_factors.append("volatility is contained, which supports stability of the signal")

    if regression_r2 is not None and regression_r2 < 0.3:
        score -= 0.06
        reasoning_factors.append("regression fit is weak")
    elif regression_r2 is not None and regression_r2 >= 0.6:
        score += 0.05
        reasoning_factors.append("regression fit is reasonably strong")

    score += (confidence_score - 0.5) * 0.18
    score = max(0.05, min(score, 0.95))

    if confidence_score < 0.2 and trend_direction == "unknown":
        label = "unclear"
    elif score >= 0.7:
        label = "high"
    elif score >= 0.52:
        label = "moderate"
    else:
        label = "low"

    unique_reasons = list(dict.fromkeys(reasoning_factors))
    if not unique_reasons:
        unique_reasons = ["signals are mixed and do not strongly support one direction"]

    return {
        "likelihood_label": label,
        "likelihood_score": round(score, 2),
        "reasoning_factors": unique_reasons[:5],
    }


def build_analytical_context_package(
    question: str,
    metric_info: dict,
    signals: dict,
    likelihood: dict,
    time_col: str | None,
    metric_col: str | None,
    ts_df: pd.DataFrame,
) -> dict[str, Any]:
    if ts_df.empty or time_col is None or metric_col is None:
        series_range = {"start": None, "end": None}
        sample_points: list[dict[str, Any]] = []
        observations = 0
    else:
        series_range = {
            "start": _json_safe(ts_df[time_col].min()),
            "end": _json_safe(ts_df[time_col].max()),
        }
        sample_points = _serialize_dataframe_points(ts_df[[time_col, metric_col]], limit=8)
        observations = int(len(ts_df))

    return {
        "question": question,
        "metric_selected": metric_col,
        "time_column": time_col,
        "series_range": series_range,
        "observations": observations,
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
            "reasoning_factors": [
                str(item) for item in (likelihood.get("reasoning_factors") or [])
            ],
        },
        "window_comparison": {
            key: _json_safe(value)
            for key, value in (signals.get("window_comparison") or {}).items()
        },
        "sample_points": sample_points,
    }


def build_analytical_prompt(question: str, analytical_context: dict) -> str:
    serialized_context = json.dumps(
        analytical_context,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_safe,
    )
    return (
        f"[USER QUESTION]\n{question.strip()}\n\n"
        f"[ANALYTICAL CONTEXT]\n{serialized_context}\n\n"
        "[INSTRUCTIONS]\n"
        "You must:\n"
        "- answer only from the analytical context\n"
        "- explain the observed direction\n"
        "- explain the likelihood carefully\n"
        "- mention uncertainty explicitly\n"
        "- avoid deterministic claims\n"
        "- avoid hallucinating causes not present in the data\n"
        "- prefer quantified wording\n"
        "- distinguish observed signal from forward-looking interpretation\n"
        "- if confidence is low, say so clearly\n"
        "- if the question is probabilistic, explain the likelihood as signal-based, not guaranteed forecast"
    )


def _log_event(payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=_json_safe))


def analytical_preprocessor(question: str, df: pd.DataFrame) -> dict[str, Any]:
    start_time = time.perf_counter()
    intent = detect_analytical_intent(question)
    metric_info: dict[str, Any] = {"primary_metric": None, "secondary_metrics": [], "matched_by": "none"}
    signals = _empty_signals()
    likelihood = {"likelihood_label": "unclear", "likelihood_score": 0.5, "reasoning_factors": []}
    analytical_context: dict[str, Any] = {}
    prompt = ""
    time_col: str | None = None
    metric_col: str | None = None
    observations = 0
    error: dict[str, str] | None = None

    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            error = _error_payload("empty_dataframe", "Dataframe is empty")
            return {
                "intent": intent,
                "metric_info": metric_info,
                "signals": signals,
                "likelihood": likelihood,
                "analytical_context": analytical_context,
                "prompt": prompt,
                "error": error,
            }

        time_col = infer_time_column(df)
        if time_col is None:
            error = _error_payload("missing_time_column", "No suitable time column was found")
            return {
                "intent": intent,
                "metric_info": metric_info,
                "signals": signals,
                "likelihood": likelihood,
                "analytical_context": analytical_context,
                "prompt": prompt,
                "error": error,
            }

        metric_info = infer_target_metric(question, df)
        working_df, metric_col, canonical_metric = _resolve_metric_frame(metric_info, df)
        if metric_col is None or canonical_metric is None:
            error = _error_payload("missing_metric", "No suitable numeric metric could be inferred")
            return {
                "intent": intent,
                "metric_info": metric_info,
                "signals": signals,
                "likelihood": likelihood,
                "analytical_context": analytical_context,
                "prompt": prompt,
                "error": error,
            }

        ts_df = prepare_time_series(working_df, time_col, metric_col)
        observations = int(len(ts_df))
        if ts_df.empty:
            error = _error_payload("insufficient_series", "No usable time series could be prepared from the metric")
            return {
                "intent": intent,
                "metric_info": metric_info,
                "signals": signals,
                "likelihood": likelihood,
                "analytical_context": analytical_context,
                "prompt": prompt,
                "error": error,
            }

        signals = compute_analytical_signals(ts_df, metric_col)
        likelihood = derive_risk_or_likelihood(signals, intent["target_direction"])
        analytical_context = build_analytical_context_package(
            question=question,
            metric_info=metric_info,
            signals=signals,
            likelihood=likelihood,
            time_col=time_col,
            metric_col=canonical_metric,
            ts_df=ts_df.rename(columns={metric_col: canonical_metric}),
        )
        prompt = build_analytical_prompt(question, analytical_context)

        return {
            "intent": intent,
            "metric_info": metric_info,
            "signals": signals,
            "likelihood": likelihood,
            "analytical_context": analytical_context,
            "prompt": prompt,
            "error": None,
        }
    finally:
        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        _log_event(
            {
                "question": question,
                "detected_intent": intent.get("intent_type"),
                "metric_selected": metric_col,
                "time_column": time_col,
                "number_of_observations": observations,
                "confidence": signals.get("signal_confidence_score"),
                "execution_time_ms": execution_time_ms,
            }
        )


__all__ = [
    "analytical_preprocessor",
    "build_analytical_context_package",
    "build_analytical_prompt",
    "compute_analytical_signals",
    "derive_risk_or_likelihood",
    "detect_analytical_intent",
    "infer_target_metric",
    "infer_time_column",
    "prepare_time_series",
]
