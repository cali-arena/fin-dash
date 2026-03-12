"""
Claude integration for Intelligence Desk. API key from Streamlit secrets only.
"""
from __future__ import annotations

import logging
import time

from anthropic import Anthropic
import streamlit as st

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 100_000
MAX_TOKENS_CAP = 4096
DEFAULT_MAX_TOKENS = 1200


class ClaudeError(Exception):
    """User-facing error; message is safe to show in UI."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def has_claude_api_key() -> bool:
    """True when ANTHROPIC_API_KEY exists in Streamlit secrets."""
    try:
        key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
        configured = bool(key and key != "your-key-here")
        logger.info("Claude secret detected=%s", configured)
        return configured
    except Exception:
        logger.info("Claude secret detected=False (secrets unavailable)")
        return False


def get_claude_client() -> Anthropic:
    api_key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key or api_key == "your-key-here":
        raise ClaudeError("Claude is not configured.")
    return Anthropic(api_key=api_key)


def _truncate_prompt(prompt: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    if not prompt or len(prompt) <= max_chars:
        return prompt or ""
    return prompt[:max_chars] + "\n\n[Prompt truncated for length.]"


def _cap_max_tokens(max_tokens: int) -> int:
    try:
        n = int(max_tokens)
        return max(1, min(n, MAX_TOKENS_CAP))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOKENS


def claude_generate(
    prompt: str,
    model: str = "claude-3-7-sonnet-latest",
    max_tokens: int = 1200,
) -> str:
    """Generate a response from Claude with safe diagnostics."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ClaudeError("No prompt provided.")
    prompt = _truncate_prompt(prompt)
    max_tokens = _cap_max_tokens(max_tokens)
    start = time.perf_counter()

    try:
        client = get_claude_client()
    except ClaudeError:
        raise
    except Exception as e:
        logger.warning("Claude client init failed: %s", type(e).__name__)
        raise ClaudeError("Unable to connect to Claude. Check app configuration.") from e

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        duration_s = time.perf_counter() - start
        logger.info(
            "Claude request completed: prompt_len=%d, model=%s, duration_sec=%.2f, max_tokens=%d",
            len(prompt),
            model,
            duration_s,
            max_tokens,
        )
        if not response.content:
            raise ClaudeError("Claude returned an empty response. Please try again.")

        text_chunks: list[str] = []
        for block in response.content:
            piece = (getattr(block, "text", "") or "").strip()
            if piece:
                text_chunks.append(piece)
        text = "\n".join(text_chunks).strip()
        if not text:
            raise ClaudeError("Claude returned no text. Please try again.")
        return text
    except ClaudeError:
        raise
    except Exception as e:
        duration_s = time.perf_counter() - start
        err_msg = (getattr(e, "message", None) or str(e) or "").strip().lower()
        logger.info(
            "Claude request failed: prompt_len=%d, model=%s, duration_sec=%.2f, error=%s",
            len(prompt),
            model,
            duration_s,
            type(e).__name__,
        )
        if any(tok in err_msg for tok in ("auth", "invalid", "401", "403", "api_key", "api key", "authentication")):
            raise ClaudeError("Claude authentication failed. Check app configuration.") from e
        if "overloaded" in err_msg or "rate" in err_msg or "429" in err_msg:
            raise ClaudeError("Claude is temporarily busy. Please try again in a moment.") from e
        if "context" in err_msg or "length" in err_msg or "token" in err_msg:
            raise ClaudeError("The request was too long. Try a shorter question or narrower scope.") from e
        raise ClaudeError("Claude could not generate a response. Please try again.") from e
