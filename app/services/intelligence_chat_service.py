from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import streamlit as st

from app.services.claude_client import (
    ClaudeError,
    anthropic_sdk_available,
    claude_generate,
    has_claude_api_key,
)

logger = logging.getLogger(__name__)

CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5"
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"


class ChatProviderError(Exception):
    """User-safe provider error for Intelligence Desk chat."""


@dataclass(frozen=True)
class ProviderStatus:
    provider: str | None
    enabled: bool
    status_text: str


def _openai_sdk_available() -> bool:
    try:
        from openai import OpenAI  # noqa: F401
        return True
    except Exception:
        return False


def _openai_key_configured() -> bool:
    try:
        key = (st.secrets.get("OPENAI_API_KEY") or "").strip()
        return bool(key and key != "your-key-here")
    except Exception:
        return False


def get_provider_status() -> ProviderStatus:
    claude_ready = bool(has_claude_api_key() and anthropic_sdk_available())
    if claude_ready:
        return ProviderStatus(provider="claude", enabled=True, status_text="Provider: Claude enabled")
    openai_ready = bool(_openai_key_configured() and _openai_sdk_available())
    if openai_ready:
        return ProviderStatus(provider="openai", enabled=True, status_text="Provider: OpenAI enabled (Claude unavailable)")
    return ProviderStatus(provider=None, enabled=False, status_text="Provider unavailable")


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    trimmed = messages[-8:]
    convo = "\n".join([f"{m.get('role', 'user')}: {m.get('text', '')}" for m in trimmed])
    return (
        "You are an institutional investment analyst.\n"
        "Provide concise, actionable answers.\n"
        "Clearly separate observable facts from inference when relevant.\n\n"
        f"Conversation:\n{convo}\n\n"
        "Respond to the latest user message."
    )


def _call_openai(messages: list[dict[str, str]], model: str = OPENAI_DEFAULT_MODEL) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise ChatProviderError("OpenAI SDK unavailable in deployment.") from e
    key = (st.secrets.get("OPENAI_API_KEY") or "").strip()
    if not key or key == "your-key-here":
        raise ChatProviderError("OpenAI is not configured.")
    client = OpenAI(api_key=key)
    payload: list[dict[str, Any]] = [
        {"role": "system", "content": "You are an institutional investment analyst. Be concise and clear."}
    ]
    payload.extend(
        {"role": "assistant" if m.get("role") == "assistant" else "user", "content": m.get("text", "")}
        for m in messages[-8:]
    )
    try:
        resp = client.chat.completions.create(model=model, messages=payload, max_tokens=900)
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not text:
            raise ChatProviderError("OpenAI returned no response.")
        return text
    except ChatProviderError:
        raise
    except Exception as e:
        logger.warning("Intelligence chat OpenAI request failed: %s", type(e).__name__)
        raise ChatProviderError("OpenAI request failed.") from e


def generate_chat_reply(messages: list[dict[str, str]]) -> str:
    status = get_provider_status()
    logger.info("Intelligence chat provider selected=%s enabled=%s", status.provider, status.enabled)
    if not status.enabled or not status.provider:
        raise ChatProviderError("Provider unavailable.")
    if status.provider == "claude":
        prompt = _messages_to_prompt(messages)
        try:
            return claude_generate(prompt=prompt, model=CLAUDE_DEFAULT_MODEL, max_tokens=1000)
        except ClaudeError as e:
            raise ChatProviderError(getattr(e, "message", str(e)) or "Claude request failed.") from e
        except Exception as e:
            logger.warning("Intelligence chat Claude request failed: %s", type(e).__name__)
            raise ChatProviderError("Claude request failed.") from e
    if status.provider == "openai":
        return _call_openai(messages)
    raise ChatProviderError("Provider unavailable.")
