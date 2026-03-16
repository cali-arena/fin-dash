"""
LLM / Claude configuration. API key from environment or Streamlit secrets.

CLOUD BOOT: Do not import this module from main.py or from page modules loaded at startup.
Intelligence Desk uses app.services.llm_client only (UI session-state key). This module
is used only by app.llm.claude/claude_client, which are lazy-loaded via app.llm.__getattr__.
get_anthropic_api_key() must never raise (wrapped in try/except).
"""
from __future__ import annotations

import os


def get_anthropic_api_key() -> str | None:
    """Return API key if configured; otherwise None. Never raises (safe for Cloud with no secrets)."""
    try:
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            try:
                import streamlit as _st
                if hasattr(_st, "secrets") and _st.secrets is not None:
                    key = str((_st.secrets.get("ANTHROPIC_API_KEY") or "")).strip()
            except Exception:
                pass
        if not key or key == "your-key-here":
            return None
        return key
    except Exception:
        return None
