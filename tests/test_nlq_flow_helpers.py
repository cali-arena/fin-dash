from __future__ import annotations

from datetime import date

from app.config.llm_config import get_anthropic_api_key
from app.llm.claude_client import (
    generate_data_narrative,
    generate_market_intelligence_response,
)
from app.pages.nlq_chat import (
    _classify_query_route,
    _extract_numeric_threshold,
    _intent_and_queryspec,
)


def _sample_catalog() -> dict[str, set[str]]:
    return {
        "channel": {"Broker Dealer", "Wealth"},
        "sub_channel": {"RIA", "Private Bank"},
        "country": {"United States"},
        "src_country": {"United States"},
        "segment": {"Institutional"},
        "sub_segment": {"Pension"},
        "product_ticker": {"AGG", "HYG", "TIP"},
    }


def test_route_data_question() -> None:
    q = "Tell me about contributors in Broker Dealer channel with NNB above $100k"
    route = _classify_query_route(q)
    assert route.route == "data_question"


def test_route_market_intelligence_outlook() -> None:
    q = "What is the latest outlook for ETF inflows across major asset classes?"
    route = _classify_query_route(q)
    assert route.route == "market_intelligence"


def test_threshold_parser_currency_and_percent() -> None:
    op1, v1 = _extract_numeric_threshold("NNB above $100k")
    op2, v2 = _extract_numeric_threshold("fee yield below 0.5%")
    assert op1 == "gt" and v1 == 100_000.0
    assert op2 == "lt" and abs(v2 - 0.005) < 1e-12


def test_intent_special_cases() -> None:
    metric_reg: dict[str, object] = {}
    dim_reg: dict[str, object] = {}
    catalog = _sample_catalog()
    ex1, qs1 = _intent_and_queryspec(
        "Which ETFs had high NNB but low fee yield in Q3?",
        metric_reg,
        dim_reg,
        catalog,
        date(2026, 3, 10),
    )
    ex2, qs2 = _intent_and_queryspec(
        "What drove the difference between organic growth and AUM growth in June?",
        metric_reg,
        dim_reg,
        catalog,
        date(2026, 3, 10),
    )
    assert ex1 is not None and ex1.intent == "growth_quality_flags" and qs1 is None
    assert ex2 is not None and ex2.intent == "decomposition" and qs2 is None


def test_placeholder_key_treated_as_unconfigured(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your-key-here")
    assert get_anthropic_api_key() is None


def test_claude_client_placeholder_fallback_labels(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your-key-here")
    data_out = generate_data_narrative({"numbers": {"nnb": "$1.0M"}})
    market_out = generate_market_intelligence_response("ETF inflow outlook")
    assert data_out.startswith("Internal Data Answer -")
    assert market_out.startswith("Market Intelligence -")

