"""
LLM adapter for Intelligence Desk Market Intelligence mode.
Supports Claude (Anthropic) and OpenAI. API key and model are passed by the caller.
No secrets are stored in this module.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LLM_ERROR_AUTH = "API request failed: check your API key and try again."
LLM_ERROR_GENERIC = "The provider could not generate a response. Please try again or check your key."

SYSTEM_PROMPT_MARKET = """You are a market intelligence analyst.

Rules:
- Keep the response concise and executive-friendly (3-5 short paragraphs max).
- Clearly separate what is observable/cited from what is inference.
- Do not express false certainty; hedge when evidence is limited.
- This answer is external/market-oriented only; do not reference internal portfolio data.
- If context is thin, say so explicitly and label the answer as general-market inference.
- Use this structure:
  1) Executive take
  2) What is observable now
  3) Implications for flows/positioning
  4) Risks and watch-items
"""


class LLMError(Exception):
    """Raised when the LLM call fails with a user-safe message."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _mask_key(key: str | None) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "..." + key[-2:]


def _call_claude(model: str, api_key: str, system: str, user_content: str, max_tokens: int = 1024) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        raise LLMError("Anthropic SDK not installed. Add 'anthropic' to requirements.")
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    if not msg.content or not isinstance(msg.content, list):
        raise LLMError(LLM_ERROR_GENERIC)
    text = msg.content[0]
    if getattr(text, "text", None):
        return (text.text or "").strip()
    raise LLMError(LLM_ERROR_GENERIC)


def _call_openai(model: str, api_key: str, system: str, user_content: str, max_tokens: int = 1024) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("OpenAI SDK not installed. Add 'openai' to requirements.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    if not resp.choices:
        raise LLMError(LLM_ERROR_GENERIC)
    content = resp.choices[0].message.content
    return (content or "").strip()


def generate_market_intelligence(
    provider: str,
    model: str,
    api_key: str,
    prompt: str,
    context: str | None = None,
) -> tuple[str, str]:
    """
    Generate a market-intelligence answer using the selected provider.

    Returns:
        (answer_text, meta_label)
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise LLMError("Please enter a question.")
    api_key = (api_key or "").strip()
    if not api_key or api_key == "your-key-here":
        raise LLMError("API key is required. Enter your key in LLM settings and click Apply.")
    model = (model or "").strip()
    if not model:
        raise LLMError("Model is required. Select a model in LLM settings and click Apply.")

    user_content = (
        f"User question: {prompt}\n\n"
        "Return a compact institutional note. Keep the answer explicitly external-market focused."
    )
    if context and context.strip():
        user_content += f"\n\nContext from external sources:\n{context.strip()}"

    provider_lower = (provider or "").strip().lower()
    meta_label: str
    try:
        if provider_lower in ("claude", "anthropic"):
            answer = _call_claude(model, api_key, SYSTEM_PROMPT_MARKET, user_content)
            meta_label = f"Claude (Anthropic) | {model}"
        elif provider_lower == "openai":
            answer = _call_openai(model, api_key, SYSTEM_PROMPT_MARKET, user_content)
            meta_label = f"OpenAI | {model}"
        else:
            raise LLMError(f"Unknown provider: {provider}. Use Claude (Anthropic) or OpenAI.")
    except LLMError:
        raise
    except Exception as e:
        msg = str(e).strip().lower()
        if any(tok in msg for tok in ("auth", "invalid", "401", "403", "api_key", "api key")):
            logger.warning("LLM auth-related error (key masked): %s", _mask_key(api_key))
            raise LLMError(LLM_ERROR_AUTH) from e
        logger.exception("LLM call failed")
        raise LLMError(LLM_ERROR_GENERIC) from e

    return answer, meta_label
