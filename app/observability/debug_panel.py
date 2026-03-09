"""
Dev-only Cache/Perf Debug Panel: dataset_version, hashes, hit/miss, timings.
Proves caching is working via hits/misses and query log.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None

QUERIES_CAP = 200
QUERIES_DISPLAY = 30
SLOW_MS_DEFAULT = 200
OBS_KEY = "obs"
SEEN_CACHE_KEYS_KEY = "obs_seen_keys"


def is_dev_mode() -> bool:
    """True when DEV_MODE=1 or st.secrets["DEV_MODE"] is truthy."""
    if os.environ.get("DEV_MODE") == "1":
        return True
    if st is not None:
        try:
            return bool(st.secrets.get("DEV_MODE"))
        except Exception:
            pass
    return False


def init_metrics_state() -> None:
    """Initialize st.session_state['obs'] and seen cache keys if not present."""
    if st is None:
        return
    st.session_state.setdefault(OBS_KEY, {
        "dataset_version": None,
        "last_filter_hash": None,
        "hits": 0,
        "misses": 0,
        "queries": [],
        "slow_queries": [],
    })
    st.session_state.setdefault(SEEN_CACHE_KEYS_KEY, set())


def build_cache_key(
    level: str,
    name: str,
    dataset_version: str,
    filter_hash: str,
    extra: str = "",
) -> str:
    """Build a stable cache key for hit/miss tracking (convention)."""
    parts = [level, name, dataset_version, filter_hash or ""]
    if extra:
        parts.append(extra)
    return "|".join(parts)


def _check_seen_and_mark(cache_key: str) -> str:
    """If cache_key in seen -> 'hit', else 'miss' and add to seen. Returns 'hit' or 'miss'."""
    if st is None:
        return "miss"
    init_metrics_state()
    seen = st.session_state[SEEN_CACHE_KEYS_KEY]
    if cache_key in seen:
        return "hit"
    seen.add(cache_key)
    return "miss"


def log_query(event: dict[str, Any]) -> None:
    """
    Append a query event to st.session_state['obs']['queries'] (cap QUERIES_CAP).
    event: ts, level (A/B/C), name, dataset_version, filter_hash, elapsed_ms,
           rowcount (optional), cache_status ("hit"|"miss"), backend ("duckdb"|"parquet"|"").
    """
    if st is None or not is_dev_mode():
        return
    init_metrics_state()
    obs = st.session_state[OBS_KEY]
    if event.get("cache_status") == "hit":
        obs["hits"] = obs.get("hits", 0) + 1
    else:
        obs["misses"] = obs.get("misses", 0) + 1
    if event.get("dataset_version") is not None:
        obs["dataset_version"] = event["dataset_version"]
    if event.get("filter_hash") is not None:
        obs["last_filter_hash"] = event.get("filter_hash", "")[:16]

    q = dict(event)
    if "ts" not in q:
        q["ts"] = datetime.now(timezone.utc).isoformat()
    obs["queries"] = (obs["queries"] + [q])[-QUERIES_CAP:]

    elapsed = q.get("elapsed_ms") or 0
    if elapsed > SLOW_MS_DEFAULT:
        obs["slow_queries"] = (obs["slow_queries"] + [q])[-QUERIES_CAP:]


def render_debug_panel() -> None:
    """
    Show dataset_version, last_filter_hash, hits/misses + hit_rate,
    table of last 30 queries (ts desc), and slow queries list.
    Dev-only: call only when is_dev_mode().
    """
    if st is None or not is_dev_mode():
        return
    init_metrics_state()
    obs = st.session_state[OBS_KEY]
    st.caption("Cache / Perf Debug")

    dv = obs.get("dataset_version") or "—"
    fh = obs.get("last_filter_hash") or "—"
    st.text(f"dataset_version: {dv}")
    st.text(f"last_filter_hash: {fh}")

    hits = obs.get("hits", 0)
    misses = obs.get("misses", 0)
    total = hits + misses
    hit_rate = (hits / total * 100) if total else 0
    st.metric("Cache hits", hits)
    st.metric("Cache misses", misses)
    st.metric("Hit rate %", f"{hit_rate:.1f}")

    queries = obs.get("queries") or []
    if queries:
        # Last 30, sorted by ts desc
        import pandas as pd
        df = pd.DataFrame(queries[-QUERIES_DISPLAY:][::-1])
        if not df.empty and "ts" in df.columns:
            st.subheader("Last 30 queries")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No query logs yet.")

    slow = obs.get("slow_queries") or []
    if slow:
        st.subheader("Slow queries")
        for s in slow[-20:]:
            st.text(f"{s.get('elapsed_ms', 0):.0f} ms | {s.get('level', '')} | {s.get('name', '')} | {s.get('cache_status', '')}")
    else:
        st.caption("No slow queries.")

    prewarm = st.session_state.get("prewarm_timings") or []
    if prewarm:
        st.subheader("Prewarm")
        for p in prewarm[-3:]:
            st.text(f"Total: {p.get('total_ms', 0):.0f} ms")
            for c in p.get("calls", []):
                err = c.get("error", "")
                st.text(f"  {c.get('name', '')}: {c.get('elapsed_ms', 0):.0f} ms" + (f" ({err})" if err else ""))


def register_cache_call(
    level: str,
    name: str,
    dataset_version: str,
    filter_hash: str,
    elapsed_ms: float,
    rowcount: int | None,
    extra_key: str = "",
    backend: str = "",
) -> None:
    """
    Build cache_key, determine hit/miss from seen set, update obs and log one query event.
    Call from pyramid (or gateway) after each cached path.
    """
    if st is None or not is_dev_mode():
        return
    cache_key = build_cache_key(level, name, dataset_version, filter_hash, extra_key)
    cache_status = _check_seen_and_mark(cache_key)
    log_query({
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "name": name,
        "dataset_version": dataset_version,
        "filter_hash": filter_hash[:16] if filter_hash else "",
        "elapsed_ms": round(elapsed_ms, 2),
        "rowcount": rowcount,
        "cache_status": cache_status,
        "backend": backend or "",
    })
