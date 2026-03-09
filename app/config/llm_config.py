"""
LLM / Claude configuration. API key from env or placeholder.
Do not commit real keys; use environment variables in production.
"""
from __future__ import annotations

import os

# Placeholder: set ANTHROPIC_API_KEY in environment or replace for local dev only.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-key-here")


def get_anthropic_api_key() -> str | None:
    """Return API key if configured (non-placeholder); otherwise None."""
    key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
    if not key or key.strip() in ("", "your-key-here"):
        return None
    return key.strip()
