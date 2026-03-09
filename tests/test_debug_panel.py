"""
Minimal tests for app.observability.debug_panel: is_dev_mode, build_cache_key, hit/miss convention.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


def test_is_dev_mode_from_env() -> None:
    """DEV_MODE=1 in env -> is_dev_mode() True."""
    with patch.dict("os.environ", {"DEV_MODE": "1"}):
        from app.observability.debug_panel import is_dev_mode
        assert is_dev_mode() is True


def test_is_dev_mode_false_when_env_empty() -> None:
    """DEV_MODE=0 or unset -> is_dev_mode() False (when st.secrets doesn't override)."""
    with patch.dict("os.environ", {"DEV_MODE": "0"}):
        from app.observability.debug_panel import is_dev_mode
        assert is_dev_mode() is False


def test_build_cache_key_deterministic() -> None:
    """Same inputs -> same cache key."""
    from app.observability.debug_panel import build_cache_key
    k1 = build_cache_key("A", "firm_monthly", "v1", "abc123", "")
    k2 = build_cache_key("A", "firm_monthly", "v1", "abc123", "")
    assert k1 == k2
    assert "A" in k1 and "firm_monthly" in k1 and "v1" in k1 and "abc123" in k1


def test_build_cache_key_different_extra() -> None:
    """Different extra_key -> different cache key."""
    from app.observability.debug_panel import build_cache_key
    k1 = build_cache_key("B", "kpi_cards", "v1", "abc", "")
    k2 = build_cache_key("B", "kpi_cards", "v1", "abc", "top_n=10")
    assert k1 != k2
