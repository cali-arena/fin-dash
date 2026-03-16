"""
Claude integration for Intelligence Desk. API key from Streamlit secrets only.
"""
from __future__ import annotations

import logging
import time

import streamlit as st

logger = logging.getLogger(__name__)
BUILD_MARKER = "claude-cloud-deploy-2026-03-13"

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
        logger.info("[%s] Claude secret detected=%s", BUILD_MARKER, configured)
        return configured
    except Exception:
        logger.info("[%s] Claude secret detected=False (secrets unavailable)", BUILD_MARKER)
        return False


def anthropic_sdk_available() -> bool:
    try:
        import anthropic  # noqa: F401
        logger.info("[%s] Anthropic SDK import available=True", BUILD_MARKER)
        return True
    except Exception:
        logger.info("[%s] Anthropic SDK import available=False", BUILD_MARKER)
        return False


def get_claude_client():
    try:
        from anthropic import Anthropic
        logger.info("[%s] Anthropic SDK import available=True (client init)", BUILD_MARKER)
    except Exception as e:
        logger.warning("[%s] Anthropic SDK import available=False (client init): %s", BUILD_MARKER, type(e).__name__)
        raise ClaudeError("Claude SDK is unavailable in this deployment.") from e
    api_key = (st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key or api_key == "your-key-here":
        raise ClaudeError("Claude is not configured.")
    client = Anthropic(api_key=api_key)
    logger.info("[%s] Claude client init success=True", BUILD_MARKER)
    return client


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
            "[%s] Claude request success=True prompt_len=%d model=%s duration_sec=%.2f max_tokens=%d",
            BUILD_MARKER,
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
            "[%s] Claude request success=False prompt_len=%d model=%s duration_sec=%.2f error=%s",
            BUILD_MARKER,
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


def claude_generate_grounded(
    system_prompt: str,
    user_message: str,
    model: str = "claude-3-7-sonnet-latest",
    max_tokens: int = 1200,
) -> str:
    """
    Generate a response with system + user messages. Used for data-grounded Intelligence Desk:
    system = dataset rules + context, user = question.
    """
    system_prompt = _truncate_prompt((system_prompt or "").strip())
    user_message = (user_message or "").strip()
    if not user_message:
        raise ClaudeError("No user message provided.")
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
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_message}],
        )
        duration_s = time.perf_counter() - start
        logger.info(
            "[%s] Claude grounded request success=True system_len=%d user_len=%d model=%s duration_sec=%.2f",
            BUILD_MARKER,
            len(system_prompt),
            len(user_message),
            model,
            duration_s,
        )
        if not response.content:
            raise ClaudeError("Claude returned an empty response. Please try again.")
        text_chunks = []
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
        if any(tok in err_msg for tok in ("auth", "invalid", "401", "403", "api_key", "authentication")):
            raise ClaudeError("Claude authentication failed. Check app configuration.") from e
        if "overloaded" in err_msg or "rate" in err_msg or "429" in err_msg:
            raise ClaudeError("Claude is temporarily busy. Please try again in a moment.") from e
        if "context" in err_msg or "length" in err_msg or "token" in err_msg:
            raise ClaudeError("The request was too long. Try a shorter question or narrower scope.") from e
        raise ClaudeError("Claude could not generate a response. Please try again.") from e
