"""
LLM / Claude configuration. API key from environment or Streamlit secrets.
Do not commit real keys; use env vars or Streamlit Cloud Secrets in production.
"""
from __future__ import annotations

import os

# No default placeholder: key must come from env or Streamlit secrets.


def get_anthropic_api_key() -> str | None:
    """Return API key if configured; otherwise None. Safe for Cloud (env + st.secrets)."""
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        try:
            import streamlit as _st
            key = (_st.secrets.get("ANTHROPIC_API_KEY") or "").strip()
        except Exception:
            pass
    if not key:
        return None
    return key
