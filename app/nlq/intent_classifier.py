"""
Intent classifier for NLQ: detect data_question vs market_intelligence.
Rule-based keyword signals only. No LLM; no raw data access.
Same input always yields the same intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

IntentType = Literal["data_question", "market_intelligence", "ambiguous"]


# Internal dataset / portfolio keywords (data questions)
DATA_QUESTION_KEYWORDS = frozenset({
    "nnb", "nnf", "net new business", "net new flow", "fee yield", "fee flow",
    "aum", "end aum", "begin aum", "organic growth", "ogr", "market impact",
    "channel", "sub-channel", "sub channel", "segment", "sub-segment", "sub segment",
    "ticker", "etf", "product", "contributors", "flows", "versus", "vs",
    "above", "below", "ytd", "qoq", "yoy", "1m", "last 12", "last 6", "last 3",
    "country", "region", "geo", "distribution", "wealth", "broker", "institutional",
})

# Market intelligence / external keywords
MARKET_INTELLIGENCE_KEYWORDS = frozenset({
    "competitors", "competitor", "macro", "sentiment", "rates", "inflation",
    "fed", "treasury", "ecb", "boe", "geopolitical", "market outlook", "outlook",
    "news", "external", "policy", "earnings season", "market conditions",
    "expectations", "drivers", "multi-asset", "equity", "bond", "duration",
    "credit positioning", "etf inflows", "asset classes",
})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@dataclass(frozen=True)
class IntentResult:
    """Result of intent classification."""
    intent: IntentType
    reason: str
    data_score: int
    market_score: int


def classify_intent(query: str) -> IntentResult:
    """
    Classify user question as data_question, market_intelligence, or ambiguous.
    Uses keyword signals only. LLM never invoked; no raw data access.
    """
    t = _normalize(query)
    if not t:
        return IntentResult(
            intent="ambiguous",
            reason="Empty or whitespace-only query.",
            data_score=0,
            market_score=0,
        )

    data_score = sum(1 for k in DATA_QUESTION_KEYWORDS if k in t)
    market_score = sum(1 for k in MARKET_INTELLIGENCE_KEYWORDS if k in t)

    if market_score > 0 and data_score == 0:
        return IntentResult(
            intent="market_intelligence",
            reason="Question references external market context (macro, rates, sentiment, competitors).",
            data_score=data_score,
            market_score=market_score,
        )
    if data_score > 0 and market_score == 0:
        return IntentResult(
            intent="data_question",
            reason="Question references governed internal metrics or dimensions.",
            data_score=data_score,
            market_score=market_score,
        )
    if data_score > 0 and market_score > 0:
        return IntentResult(
            intent="ambiguous",
            reason="Question mixes internal data and external market intents; use selected mode.",
            data_score=data_score,
            market_score=market_score,
        )
    return IntentResult(
        intent="ambiguous",
        reason="No strong keyword signals for data or market intent.",
        data_score=data_score,
        market_score=market_score,
    )


def is_data_question(intent: IntentResult, *, prefer_data: bool = True) -> bool:
    """
    Resolve ambiguous intent using preferred mode.
    When intent is data_question -> True; market_intelligence -> False; ambiguous -> prefer_data.
    """
    if intent.intent == "data_question":
        return True
    if intent.intent == "market_intelligence":
        return False
    return prefer_data
