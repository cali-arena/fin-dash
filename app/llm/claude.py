"""
Claude integration: narrative only. Claude never performs calculations or sees raw internal data.
- Data Questions: receives verified payload (numbers + table summary); writes narrative only.
- Market Intelligence: receives external search context; writes answer; caller must label as external.
"""
from __future__ import annotations

import json
from typing import Any

from app.config.llm_config import get_anthropic_api_key

SYSTEM_DATA_NARRATIVE = """You are a concise analyst. You will receive a JSON payload containing verified query results and summary statistics. Your task is to write a short narrative (2-4 sentences) that describes the result in plain language. You must use ONLY the numbers and facts present in the payload. Do NOT add, infer, or invent any new numbers or metrics. If the payload does not contain enough information to answer, say "I can only describe what is in the result: [summary]." Do not perform any calculations."""

SYSTEM_MARKET_INTELLIGENCE = """You are a market commentator. You will receive a user question and optional context from external sources. Answer based on the provided context. If context is missing or thin, say so. Do not claim internal or proprietary data. Keep the answer concise and cite that it is based on external sources when relevant."""


def _call_claude(system: str, user_content: str, max_tokens: int = 512) -> str:
    """Call Claude API if key is configured; return empty string otherwise or on error."""
    api_key = get_anthropic_api_key()
    if not api_key:
        return ""
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        if not msg.content or not isinstance(msg.content, list):
            return ""
        text = msg.content[0]
        if getattr(text, "text", None):
            return (text.text or "").strip()
        return ""
    except Exception:
        return ""


def claude_narrative_from_payload(payload: dict[str, Any]) -> str:
    """
    Data Questions path: Claude receives only the verified payload (numbers + table summary).
    Returns a short narrative. Claude never sees raw rows beyond the top 5 summary; never calculates.
    """
    user_content = "Payload (use only these facts and numbers):\n" + json.dumps(payload, indent=2)
    return _call_claude(SYSTEM_DATA_NARRATIVE, user_content)


def claude_market_intelligence(question: str, context: str) -> str:
    """
    Market Intelligence path: Claude receives the user question and external search context.
    Caller must display the answer with the label: "Market Intelligence — this answer draws on external sources, not your internal data."
    """
    user_content = f"User question: {question}\n\nContext from external sources:\n{context}"
    return _call_claude(SYSTEM_MARKET_INTELLIGENCE, user_content, max_tokens=1024)
