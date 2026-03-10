"""LLM integration: Claude for narrative only. Lazy imports so no API key or secrets are read at app startup."""
from __future__ import annotations

__all__ = [
    "claude_narrative_from_payload",
    "claude_market_intelligence",
    "generate_data_narrative",
    "generate_market_intelligence_response",
]


def __getattr__(name: str):
    """Lazy load so importing app.llm does not load claude/claude_client or read env/secrets until first use."""
    if name == "claude_narrative_from_payload" or name == "claude_market_intelligence":
        from app.llm.claude import claude_narrative_from_payload, claude_market_intelligence
        return claude_narrative_from_payload if name == "claude_narrative_from_payload" else claude_market_intelligence
    if name == "generate_data_narrative":
        from app.llm.claude_client import generate_data_narrative
        return generate_data_narrative
    if name == "generate_market_intelligence_response":
        from app.llm.claude_client import generate_market_intelligence_response
        return generate_market_intelligence_response
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
