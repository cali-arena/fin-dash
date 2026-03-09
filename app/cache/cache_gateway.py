"""
Streamlit cache gateway: single entrypoint for all cached data access.

- load_cache_policy() -> CachePolicy from configs/cache_policy.yml
- get_dataset_version() -> str from policy's dataset_version_source (e.g. curated/metrics_monthly.meta.json)
- cached_query(query_name, filters, fn) -> result of fn(), cached by dataset_version + query_name + filter_state_hash
- cache_policy_hash(policy) -> sha1 of canonical JSON for invalidation/debug

Observability: dev-only logs (dataset_version, query_name, filter_hash, ttl, elapsed_ms) in st.session_state["cache_debug"] (last 50).
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from app.cache.cache_keys import filter_state_hash

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)
CACHE_DEBUG_MAX = 50
DEV_MODE = os.environ.get("DEV_MODE") == "1"

# Context var so the cached function can call the loader on cache miss
_current_loader: contextvars.ContextVar[Callable[[], Any] | None] = contextvars.ContextVar(
    "cache_gateway_loader", default=None
)

T = TypeVar("T")

# Module-level policy cache (path -> policy) to avoid re-reading file every time
_policy_cache: dict[str, Any] = {}

# One cached function per class so TTL/max_entries can differ (Streamlit keys by function + args)
def _make_cached_function(ttl: int, max_entries: int | None) -> Callable:
    import streamlit as _st
    @_st.cache_data(ttl=ttl, max_entries=max_entries)
    def _cached_impl(_dataset_version: str, _query_name: str, _filter_hash: str) -> Any:
        loader = _current_loader.get()
        if loader is None:
            raise RuntimeError("cache_gateway: no loader set; use cached_query(query_name, filters, fn)")
        return loader()
    return _cached_impl

# Lazy cache of (ttl, max_entries) -> cached function so we don't create hundreds of functions
_cached_fns: dict[tuple[int, int | None], Callable] = {}


def _load_policy_impl(path_key: str) -> Any:
    """Inner: load policy from path (for st.cache_resource)."""
    from app.contracts.cache_policy_contract import (
        CachePolicyError,
        load_and_validate_cache_policy,
    )
    path = Path(path_key)
    if not path.exists():
        raise CachePolicyError(f"Cache policy config not found: {path}")
    return load_and_validate_cache_policy(path)


if st is not None:
    @st.cache_resource
    def _load_cache_policy_cached(_path_key: str) -> Any:
        """Load policy once per path (Streamlit resource cache)."""
        return _load_policy_impl(_path_key)
else:
    def _load_cache_policy_cached(_path_key: str) -> Any:
        return _load_policy_impl(_path_key)


def load_cache_policy(path: str | Path | None = None, root: Path | None = None) -> Any:
    """
    Load and validate CachePolicy from configs/cache_policy.yml (once per path when in Streamlit).
    Uses root to resolve path if path is relative. Raises on missing/invalid config.
    When Streamlit is available, uses st.cache_resource so the policy is loaded once.
    """
    from app.contracts.cache_policy_contract import CachePolicyError

    if path is None:
        path = "configs/cache_policy.yml"
    path = Path(path)
    if not path.is_absolute() and root is not None:
        path = Path(root) / path
    if not path.is_absolute():
        path = Path.cwd() / path
    path_key = str(path.resolve())

    try:
        return _load_cache_policy_cached(path_key)
    except Exception:
        if path_key not in _policy_cache:
            try:
                _policy_cache[path_key] = _load_policy_impl(path_key)
            except CachePolicyError:
                raise
        return _policy_cache[path_key]


def get_dataset_version(root: Path | None = None) -> str:
    """
    Read dataset_version from the source defined in cache policy (e.g. curated/metrics_monthly.meta.json).
    Top-level invalidator: when this changes, cache keys change and Streamlit caches miss naturally.
    """
    root = Path(root) if root is not None else Path.cwd()
    policy = load_cache_policy(root=root)
    meta_path = root / policy.dataset_version_source_path
    if not meta_path.exists():
        return "unknown"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(data.get(policy.dataset_version_source_key) or "unknown")
    except Exception:
        return "unknown"


def cache_policy_hash(policy: Any) -> str:
    """
    SHA1 of canonical JSON of the policy for debugging and invalidation semantics.
    Store in st.session_state and show in UI footer when desired.
    """
    try:
        d = asdict(policy) if hasattr(policy, "__dataclass_fields__") else vars(policy)
    except Exception:
        d = {"raw": str(policy)}
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _append_cache_debug(entry: dict[str, Any]) -> None:
    """Append to st.session_state['cache_debug'], keep last CACHE_DEBUG_MAX (dev-only)."""
    if st is None or not DEV_MODE:
        return
    if "cache_debug" not in st.session_state:
        st.session_state["cache_debug"] = []
    st.session_state["cache_debug"] = (
        st.session_state["cache_debug"] + [entry]
    )[-CACHE_DEBUG_MAX:]


def cached_query(
    query_name: str,
    filters: dict[str, Any],
    fn: Callable[[], T],
    root: Path | None = None,
) -> T:
    """
    Run fn() and cache result by dataset_version + query_name + filter_state_hash(filters).
    Uses policy for ttl and max_entries by query class (default "medium" if query not in policy).
    Returns only through this path so no page can bypass caching.
    """
    if st is None:
        return fn()
    policy = load_cache_policy(root=root)
    dataset_version = get_dataset_version(root=root)
    query_class = policy.query_classes.get(query_name, "medium")
    ttl = policy.ttl_seconds.get(query_class, 300)
    max_entries = (policy.max_entries or {}).get(query_class)
    key = (ttl, max_entries)
    if key not in _cached_fns:
        _cached_fns[key] = _make_cached_function(ttl, max_entries)
    _cached_impl = _cached_fns[key]

    fhash = filter_state_hash(filters)
    t0 = time.perf_counter()
    _current_loader.set(fn)
    try:
        result = _cached_impl(dataset_version, query_name, fhash)
        return result
    finally:
        _current_loader.set(None)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if DEV_MODE:
            logger.debug(
                "cache_query dataset_version=%s query_name=%s filter_hash=%s ttl=%s elapsed_ms=%.0f",
                dataset_version, query_name, fhash[:12], ttl, elapsed_ms,
            )
        _append_cache_debug({
            "dataset_version": dataset_version,
            "query_name": query_name,
            "filter_state_hash": fhash[:16],
            "ttl": ttl,
            "elapsed_ms": round(elapsed_ms, 2),
        })


def require_cache_gateway(f: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for data access functions: marks that callers must use the cache gateway.
    Convention: Streamlit pages must only call gateway functions (e.g. data_gateway get_*),
    not duckdb_store / query_df directly. This decorator is a no-op; enforcement is by convention.
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return f(*args, **kwargs)
    return wrapper  # type: ignore[return-value]
