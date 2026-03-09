"""
Tests for app.cache.cache_gateway: load_cache_policy, get_dataset_version, cached_query.
Cache hit: identical filters → same result, loader called once.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cache import cache_gateway


def test_cached_query_without_streamlit_calls_fn_every_time(tmp_path: Path) -> None:
    """When st is None, cached_query just runs fn() each time (no cache)."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "cache_policy.yml").write_text("""
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys: ["dataset_version", "filter_state_hash", "query_name"]
  ttl_seconds: { fast: 60, medium: 300, heavy: 1800 }
  query_classes: { test_q: "fast" }
  max_entries: { fast: 10, medium: 10, heavy: 10 }
""", encoding="utf-8")
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"dataset_version": "v1"}', encoding="utf-8"
    )
    counter = [0]

    def fn() -> int:
        counter[0] += 1
        return counter[0]

    with patch.object(cache_gateway, "st", None):
        r1 = cache_gateway.cached_query("test_q", {}, fn, root=tmp_path)
        r2 = cache_gateway.cached_query("test_q", {}, fn, root=tmp_path)
    assert r1 == 1
    assert r2 == 2
    assert counter[0] == 2


def test_cached_query_identical_filters_cache_hit(tmp_path: Path) -> None:
    """With st.cache_data mocked, same (dataset_version, query_name, filters) → cache hit; loader called once."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "cache_policy.yml").write_text("""
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys: ["dataset_version", "filter_state_hash", "query_name"]
  ttl_seconds: { fast: 60, medium: 300, heavy: 1800 }
  query_classes: { test_q: "fast" }
  max_entries: { fast: 10, medium: 10, heavy: 10 }
""", encoding="utf-8")
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"dataset_version": "v1"}', encoding="utf-8"
    )

    storage: dict[tuple, int] = {}

    def mock_cache_data(ttl: int | None = None, max_entries: int | None = None):  # noqa: ARG001
        def decorator(f):
            def wrapped(*args: object) -> int:
                key = args
                if key not in storage:
                    storage[key] = f(*args)
                return storage[key]
            return wrapped
        return decorator

    counter = [0]

    def fn() -> int:
        counter[0] += 1
        return counter[0]

    with patch("streamlit.cache_data", mock_cache_data):
        cache_gateway._cached_fns.clear()
        r1 = cache_gateway.cached_query("test_q", {"a": 1}, fn, root=tmp_path)
        r2 = cache_gateway.cached_query("test_q", {"a": 1}, fn, root=tmp_path)
    assert r1 == 1
    assert r2 == 1
    assert counter[0] == 1


def test_load_cache_policy_and_get_dataset_version(tmp_path: Path) -> None:
    """load_cache_policy loads from path; get_dataset_version reads meta.json by policy."""
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "configs" / "cache_policy.yml").write_text("""
cache:
  dataset_version_source:
    path: "curated/metrics_monthly.meta.json"
    key: "dataset_version"
  cache_keys: ["dataset_version", "filter_state_hash", "query_name"]
  ttl_seconds: { fast: 60, medium: 300, heavy: 1800 }
  query_classes: {}
  max_entries: null
""", encoding="utf-8")
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"dataset_version": "abc123"}', encoding="utf-8"
    )
    policy = cache_gateway.load_cache_policy(root=tmp_path)
    assert policy.dataset_version_source_path == "curated/metrics_monthly.meta.json"
    assert policy.dataset_version_source_key == "dataset_version"
    version = cache_gateway.get_dataset_version(root=tmp_path)
    assert version == "abc123"
