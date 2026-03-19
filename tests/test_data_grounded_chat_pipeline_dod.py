from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import data_grounded_chat_pipeline as pipeline


def _basic_monthly_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_end": pd.date_range("2024-01-31", periods=6, freq="ME"),
            "region": ["EU", "EU", "US", "US", "EU", "US"],
            "net_flow": [10, 12, 14, 13, 11, 15],
            "fees": [1, 1, 2, 2, 1, 2],
        }
    )


def _flat_combined_region_df() -> pd.DataFrame:
    months = pd.date_range("2024-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "month_end": list(months) * 2,
            "region": ["EU"] * 6 + ["US"] * 6,
            "net_flow": [100, 90, 80, 70, 60, 50, 50, 60, 70, 80, 90, 100],
        }
    )


def _direct_rank_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel": [f"C{i}" for i in range(30)],
            "nnb": list(range(30)),
        }
    )


def _direct_growth_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(30)],
            "growth": list(range(30)),
        }
    )


def _make_ts(values: list[float], metric: str = "metric") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month_end": pd.date_range("2024-01-31", periods=len(values), freq="ME"),
            metric: values,
        }
    )


def test_filter_scoped_analytics_use_filtered_rows_not_global_dataset() -> None:
    df = _flat_combined_region_df()
    question = "What is the likelihood of decline continuing in EU net flow?"

    question_type = pipeline.classify_question_type(question, df)
    context = pipeline.retrieve_context(question, df)
    analytics = pipeline.compute_analytics_if_needed(question_type, context, df)

    assert context["filters_applied"]["region"]["values"] == ["EU"]
    assert context["row_count"] == 6
    assert analytics["analytics_used"] is True
    assert analytics["signals"]["trend_direction"] == "down"
    assert float(analytics["signals"]["slope"]) < 0


@pytest.mark.parametrize(
    ("question", "df", "winner_key", "winner_value"),
    [
        ("Which channel had the highest NNB?", _direct_rank_df(), "channel", "C29"),
        ("Which ticker grew the most?", _direct_growth_df(), "ticker", "T29"),
    ],
)
def test_direct_answers_use_full_evidence_not_preview_rows(
    question: str,
    df: pd.DataFrame,
    winner_key: str,
    winner_value: str,
) -> None:
    context = pipeline.retrieve_context(question, df)

    assert context["row_count"] == len(df)
    grouped = context["aggregations"]["grouped_metric_summary"]
    assert grouped[0][winner_key] == winner_value


def test_invalid_grounded_output_is_blocked_or_replaced() -> None:
    df = _direct_rank_df()
    bad_response = "This seems positive overall."

    result = pipeline.chat_handler(question="Which channel had the highest NNB?", df=df, claude_callable=lambda _: bad_response)

    assert result["validation"]["valid"] is False
    assert result["final_validation"]["valid"] is True
    assert result["response"] == pipeline.GROUNDED_VALIDATION_FALLBACK
    assert result["response"] != bad_response


def test_invalid_analytical_output_is_blocked_or_replaced() -> None:
    df = _flat_combined_region_df()
    bad_response = "It will definitely keep falling."

    result = pipeline.chat_handler(
        question="What is the likelihood of decline continuing in EU net flow?",
        df=df,
        claude_callable=lambda _: bad_response,
    )

    assert result["validation"]["valid"] is False
    assert result["final_validation"]["valid"] is True
    assert result["response"] == pipeline.ANALYTICAL_VALIDATION_FALLBACK
    assert result["response"] != bad_response


def test_invalid_not_enough_evidence_output_is_blocked_or_replaced() -> None:
    df = _direct_rank_df().iloc[0:0].copy()
    bad_response = "Channel C29 will definitely remain on top."

    result = pipeline.chat_handler(
        question="Which channel had the highest NNB?",
        df=df,
        claude_callable=lambda _: bad_response,
    )

    assert result["evidence_decision"]["not_enough_evidence"] is True
    assert result["validation"]["valid"] is False
    assert result["final_validation"]["valid"] is True
    assert result["response"] == "There is not enough evidence in the current dataset to answer this reliably."
    assert result["response"] != bad_response
    assert result["final_mode"] == "not_enough_evidence"


@pytest.mark.parametrize(
    "question",
    [
        "What is profit margin by region?",
        "What is customer churn volatility?",
    ],
)
def test_unsupported_variables_trigger_not_enough_evidence(question: str) -> None:
    df = _basic_monthly_df()

    result = pipeline.chat_handler(question=question, df=df, claude_callable=lambda _: "Not enough evidence.")

    assert result["final_mode"] == "not_enough_evidence"
    assert result["evidence_decision"]["not_enough_evidence"] is True


def test_low_signal_confidence_caps_likelihood_label() -> None:
    ts_df = _make_ts([0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02])
    signals = pipeline.compute_analytical_signals(ts_df, "metric")
    likelihood = pipeline.derive_likelihood_assessment(signals, {"target_direction": "down", "intent_type": "forecast_like"})

    assert signals["signal_confidence"] == "low"
    assert likelihood["likelihood_label"] != "high"


def test_near_zero_instability_is_penalized_in_likelihood_label() -> None:
    ts_df = _make_ts([0.001, 0.002, 0.0005, -0.001, -0.002, -0.003])
    signals = pipeline.compute_analytical_signals(ts_df, "metric")
    likelihood = pipeline.derive_likelihood_assessment(signals, {"target_direction": "down", "intent_type": "forecast_like"})

    assert signals["volatility"] is not None
    assert float(signals["volatility"]) > 1.0
    assert likelihood["likelihood_label"] != "high"


def test_clean_supported_signal_can_reach_high_likelihood() -> None:
    ts_df = _make_ts([100, 95, 90, 85, 80, 75, 70, 65])
    signals = pipeline.compute_analytical_signals(ts_df, "metric")
    likelihood = pipeline.derive_likelihood_assessment(signals, {"target_direction": "down", "intent_type": "forecast_like"})

    assert signals["signal_confidence"] == "high"
    assert likelihood["likelihood_label"] == "high"


def test_claude_callable_exception_is_contained() -> None:
    df = _direct_rank_df()

    result = pipeline.chat_handler(
        question="Which channel had the highest NNB?",
        df=df,
        claude_callable=lambda _: (_ for _ in ()).throw(RuntimeError("claude failed")),
    )

    assert result["response"] == "Unable to generate a grounded response safely."


def test_top_level_analytics_failure_does_not_crash_request(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _flat_combined_region_df()

    def boom(*args, **kwargs):
        raise RuntimeError("analytics blew up")

    monkeypatch.setattr(pipeline, "compute_analytics_if_needed", boom)

    result = pipeline.chat_handler(
        question="What is the likelihood of decline continuing in EU net flow?",
        df=df,
        claude_callable=lambda _: "Not enough evidence.",
    )

    assert isinstance(result, dict)
    assert result.get("final_mode") == "not_enough_evidence"


def test_top_level_validation_failure_does_not_crash_request(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _direct_rank_df()

    def boom(*args, **kwargs):
        raise RuntimeError("validator blew up")

    monkeypatch.setattr(pipeline, "response_validator", boom)

    result = pipeline.chat_handler(
        question="Which channel had the highest NNB?",
        df=df,
        claude_callable=lambda _: "C29 had the highest NNB at 29.",
    )

    assert isinstance(result, dict)
    assert result.get("response")
