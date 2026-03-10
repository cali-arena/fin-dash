"""
LLM / Claude configuration. API key from environment or Streamlit secrets.
Intelligence Desk does not use this; it uses UI session-state only (app.services.llm_client).
This module must never raise at import or when get_anthropic_api_key() is called.
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
                    key = (str((_st.secrets.get("ANTHROPIC_API_KEY") or "")).strip()
            except Exception:
                pass
        if not key or key == "your-key-here":
            return None
        return key
    except Exception:
        return None
