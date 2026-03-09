"""
Tests for TTL routing: get_level_a_ttl_class selects fast/medium/heavy by query_name and policy.
_get_ttl_seconds returns policy TTL or fallback defaults.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.cache.pyramid import _get_ttl_seconds, get_level_a_ttl_class


def test_ttl_class_fast_for_monthly_queries() -> None:
    """Policy maps firm_monthly, channel_monthly, etc. to fast."""
    policy = SimpleNamespace(query_classes={"firm_monthly": "fast", "channel_monthly": "fast", "ticker_monthly": "fast"})
    assert get_level_a_ttl_class("firm_monthly", policy) == "fast"
    assert get_level_a_ttl_class("channel_monthly", policy) == "fast"
    assert get_level_a_ttl_class("ticker_monthly", policy) == "fast"


def test_ttl_class_medium_default() -> None:
    """Unknown query_name or missing policy -> medium."""
    policy = SimpleNamespace(query_classes={"firm_monthly": "fast"})
    assert get_level_a_ttl_class("unknown_query", policy) == "medium"
    assert get_level_a_ttl_class("firm_monthly", None) == "medium"
    assert get_level_a_ttl_class("any", None) == "medium"


def test_ttl_class_heavy_for_heavy_queries() -> None:
    """Policy can map some queries to heavy."""
    policy = SimpleNamespace(query_classes={"correlation_matrix": "heavy", "waterfall_inputs": "heavy"})
    assert get_level_a_ttl_class("correlation_matrix", policy) == "heavy"
    assert get_level_a_ttl_class("waterfall_inputs", policy) == "heavy"


def test_ttl_class_medium_explicit() -> None:
    """Policy can map to medium."""
    policy = SimpleNamespace(query_classes={"kpi_cards": "medium"})
    assert get_level_a_ttl_class("kpi_cards", policy) == "medium"


def test_ttl_class_independent_per_name() -> None:
    """Changing query_name changes selected class; changing policy changes class."""
    policy = SimpleNamespace(query_classes={"a": "fast", "b": "medium", "c": "heavy"})
    assert get_level_a_ttl_class("a", policy) == "fast"
    assert get_level_a_ttl_class("b", policy) == "medium"
    assert get_level_a_ttl_class("c", policy) == "heavy"
    policy2 = SimpleNamespace(query_classes={"a": "heavy"})
    assert get_level_a_ttl_class("a", policy2) == "heavy"


def test_get_ttl_seconds_uses_policy() -> None:
    """When policy has ttl_seconds, _get_ttl_seconds returns policy values."""
    policy = SimpleNamespace(ttl_seconds={"fast": 90, "medium": 400, "heavy": 2000})
    assert _get_ttl_seconds("fast", policy) == 90
    assert _get_ttl_seconds("medium", policy) == 400
    assert _get_ttl_seconds("heavy", policy) == 2000


def test_get_ttl_seconds_fallback_when_no_policy() -> None:
    """When policy is None, _get_ttl_seconds returns default TTL values."""
    assert _get_ttl_seconds("fast", None) == 60
    assert _get_ttl_seconds("medium", None) == 300
    assert _get_ttl_seconds("heavy", None) == 1800


def test_get_ttl_seconds_unknown_class_defaults_medium() -> None:
    """Unknown TTL class returns medium default."""
    policy = SimpleNamespace(ttl_seconds={"fast": 60, "medium": 300, "heavy": 1800})
    assert _get_ttl_seconds("unknown", policy) == 300
