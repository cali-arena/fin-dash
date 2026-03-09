"""
Pytest: cache_key is deterministic and stable for same params (key order irrelevant).
Canonical filter hashing: key order, list order, dates normalize to same hash.
"""
from datetime import date, datetime

import pandas as pd
import pytest

from app.cache.cache_keys import (
    build_cache_key,
    cache_key,
    canonicalize_filters,
    filter_state_hash,
)


def test_same_params_same_key() -> None:
    """Same dataset_version, view, and params -> same key."""
    k1 = cache_key("abc123", "summary", {"a": 1, "b": 2})
    k2 = cache_key("abc123", "summary", {"a": 1, "b": 2})
    assert k1 == k2


def test_params_key_order_stable() -> None:
    """Different dict key order, same values -> same key."""
    k1 = cache_key("dv", "view", {"z": 3, "a": 1, "m": 2})
    k2 = cache_key("dv", "view", {"a": 1, "m": 2, "z": 3})
    assert k1 == k2


def test_different_dataset_version_different_key() -> None:
    """Different dataset_version -> different key."""
    k1 = cache_key("v1", "view", {})
    k2 = cache_key("v2", "view", {})
    assert k1 != k2


def test_different_view_different_key() -> None:
    """Different view -> different key."""
    k1 = cache_key("dv", "view_a", {})
    k2 = cache_key("dv", "view_b", {})
    assert k1 != k2


def test_different_params_different_key() -> None:
    """Different params -> different key."""
    k1 = cache_key("dv", "view", {"x": 1})
    k2 = cache_key("dv", "view", {"x": 2})
    assert k1 != k2


def test_key_format() -> None:
    """Key has shape dataset_version:view:sha1hex (40 chars)."""
    k = cache_key("dsver", "myview", {"p": 1})
    parts = k.split(":")
    assert len(parts) == 3
    assert parts[0] == "dsver"
    assert parts[1] == "myview"
    assert len(parts[2]) == 40
    assert all(c in "0123456789abcdef" for c in parts[2])


# --- Canonical filter hashing ---


def test_same_logical_filters_different_key_order_same_hash() -> None:
    """Same logical filters with different dict key order → same hash."""
    f1 = {"z": 1, "a": 2, "m": 3}
    f2 = {"a": 2, "m": 3, "z": 1}
    assert filter_state_hash(f1) == filter_state_hash(f2)
    assert build_cache_key("v1", "q", f1) == build_cache_key("v1", "q", f2)


def test_list_order_insensitive_same_hash() -> None:
    """Selector list order differences → same hash (lists sorted in canonical form)."""
    f1 = {"channel": ["B", "A", "C"]}
    f2 = {"channel": ["C", "A", "B"]}
    assert filter_state_hash(f1) == filter_state_hash(f2)
    assert canonicalize_filters(f1)["channel"] == ["A", "B", "C"]
    assert canonicalize_filters(f2)["channel"] == ["A", "B", "C"]


def test_date_objects_normalize_to_iso() -> None:
    """date/datetime objects normalize to ISO YYYY-MM-DD."""
    d = date(2024, 3, 1)
    dt = datetime(2024, 3, 1, 12, 30, 0)
    ts = pd.Timestamp("2024-03-01")
    assert canonicalize_filters({"d": d})["d"] == "2024-03-01"
    assert canonicalize_filters({"dt": dt})["dt"] == "2024-03-01"
    assert canonicalize_filters({"ts": ts})["ts"] == "2024-03-01"
    # Same logical date → same hash
    assert filter_state_hash({"x": d}) == filter_state_hash({"x": "2024-03-01"})
    assert filter_state_hash({"x": dt}) == filter_state_hash({"x": "2024-03-01"})


def test_empty_filters_removed() -> None:
    """Null/empty filters (None, [], "", {}) are removed in canonical form."""
    out = canonicalize_filters({"a": 1, "b": None, "c": [], "d": "", "e": {}})
    assert out == {"a": 1}


def test_build_cache_key_format() -> None:
    """build_cache_key returns dataset_version:query_name:sha1hex."""
    k = build_cache_key("dv1", "firm_monthly", {"month_end_range": ("2024-01-01", "2024-12-31")})
    parts = k.split(":")
    assert len(parts) == 3
    assert parts[0] == "dv1"
    assert parts[1] == "firm_monthly"
    assert len(parts[2]) == 40
    assert all(c in "0123456789abcdef" for c in parts[2])
