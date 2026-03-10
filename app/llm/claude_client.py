"""
Claude client for Intelligence Desk narrative generation.

Important:
- Python computes and verifies all metrics before calling these functions.
- Claude receives only verified summaries/tables for Data mode.
- Claude never calculates from raw data.
"""
from __future__ import annotations

import json
import os
from typing import Any

# Placeholder key for local wiring; production should override via env/secrets.
ANTHROPIC_API_KEY = "your-key-here"
DEFAULT_MODEL = "claude-3-5-sonnet"

DATA_SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "You receive verified data results and explain them clearly. "
    "Never invent numbers."
)

MARKET_SYSTEM_PROMPT = (
    "You are a macro market intelligence analyst."
)

MARKET_LABEL = (
    "Market Intelligence - this answer draws on external sources, not your internal data."
)

DATA_LABEL = (
    "Internal Data Answer - narrative generated from verified Python outputs only."
)


def _resolve_api_key() -> str | None:
    """Resolve API key from env/secrets/placeholder; placeholder is treated as not configured."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        try:
            import streamlit as st
            key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
        except Exception:
            pass
    if not key:
        key = ANTHROPIC_API_KEY
    if not key or key == "your-key-here":
        return None
    return key


def _call_claude(*, system_prompt: str, user_prompt: str, max_tokens: int = 900, model: str = DEFAULT_MODEL) -> str:
    """Low-level Claude call; returns empty string on unavailable key or request failure."""
    api_key = _resolve_api_key()
    if not api_key:
        return ""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if not getattr(resp, "content", None):
            return ""
        parts = resp.content
        if isinstance(parts, list) and parts:
            first = parts[0]
            txt = getattr(first, "text", "")
            return (txt or "").strip()
    except Exception:
        return ""
    return ""


def generate_data_narrative(summary_data: dict[str, Any]) -> str:
    """
    Data-mode narrative.
    Input must be verified Python outputs only (summary metrics + table snippets).
    """
    if not isinstance(summary_data, dict):
        summary_data = {"summary": str(summary_data)}
    user_prompt = (
        "Verified internal data payload (JSON). "
        "Use only these values and facts. Do not calculate or invent values.\n\n"
        + json.dumps(summary_data, indent=2, default=str)
    )
    answer = _call_claude(
        system_prompt=DATA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=500,
    )
    if answer:
        return f"{DATA_LABEL}\n\n{answer}"
    return (
        f"{DATA_LABEL}\n\n"
        "Narrative generation is unavailable until ANTHROPIC_API_KEY is configured. "
        "Verified results from Python should still be displayed."
    )


def generate_market_intelligence_response(query: str) -> str:
    """
    Market-intelligence narrative.
    This function does not use internal data. It responds from market-oriented prompting.
    """
    q = (query or "").strip()
    if not q:
        return (
            f"{MARKET_LABEL}\n\n"
            "Please enter a market intelligence question."
        )
    user_prompt = (
        "Question:\n"
        f"{q}\n\n"
        "Provide a concise response covering:\n"
        "- macro environment\n"
        "- rates\n"
        "- flows\n"
        "- competitor positioning\n"
        "- sentiment\n"
        "State clearly when evidence is limited."
    )
    answer = _call_claude(
        system_prompt=MARKET_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=900,
    )
    if answer:
        return f"{MARKET_LABEL}\n\n{answer}"
    return (
        f"{MARKET_LABEL}\n\n"
        "External intelligence is not connected. Configure ANTHROPIC_API_KEY "
        "and your external context retrieval path."
    )


def example_usage() -> dict[str, str]:
    """Small usage examples for integration points."""
    sample_data_payload = {
        "summary_metrics": {
            "end_aum": "$17.9B",
            "nnb": "$245.3M",
            "ogr": "1.42%",
            "fee_yield": "0.58%",
        },
        "table_preview": [
            {"channel": "Broker Dealer", "nnb": "$102.4M", "nnf": "$5.9M"},
            {"channel": "Wealth", "nnb": "$88.1M", "nnf": "$4.8M"},
        ],
    }
    return {
        "data_mode": generate_data_narrative(sample_data_payload),
        "market_mode": generate_market_intelligence_response(
            "What is the current rates and ETF flow outlook for multi-asset portfolios?"
        ),
    }

