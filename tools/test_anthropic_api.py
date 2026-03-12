#!/usr/bin/env python3
"""
Test the Anthropic API key with a single short request.

Usage:
  Set your key in the environment, then run from repo root:

    set ANTHROPIC_API_KEY=sk-ant-...   # Windows
    export ANTHROPIC_API_KEY=sk-ant-... # Linux / macOS

    python tools/test_anthropic_api.py

  Or use .streamlit/secrets.toml locally (same key name); the script
  will try to load it if the env var is not set.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _get_api_key() -> str | None:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if key and key != "your-key-here":
        return key
    # Optional: load from .streamlit/secrets.toml
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with open(secrets_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY"):
                        # TOML: ANTHROPIC_API_KEY = "sk-ant-..."
                        if "=" in line:
                            value = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if value and value != "your-key-here":
                                return value
                        break
        except Exception:
            pass
    return None


def main() -> int:
    api_key = _get_api_key()
    if not api_key:
        print("ANTHROPIC_API_KEY is not set or is placeholder.")
        print("Set it in your environment or in .streamlit/secrets.toml (see script docstring).")
        return 1

    try:
        from anthropic import Anthropic
    except ImportError:
        print("anthropic package not installed. Run: pip install anthropic>=0.39")
        return 1

    client = Anthropic(api_key=api_key)
    prompt = "Reply with exactly: OK"
    print("Calling Anthropic API (one short message)...")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        print("Request failed:", type(e).__name__, str(e))
        return 1

    if not response.content:
        print("Unexpected: empty response.")
        return 1

    text = (getattr(response.content[0], "text", "") or "").strip()
    print("Success. Model replied:", repr(text[:200]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
