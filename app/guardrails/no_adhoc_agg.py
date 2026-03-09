"""
Guardrail: ban DataFrame.groupby and merge in the Streamlit layer when STRICT_AGG_ONLY=1.
Use agg tables; add a new materialized agg in pipelines/agg if you need a new grain.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

ADHOC_AGG_MESSAGE = (
    "Use agg tables; add a new materialized agg in pipelines/agg if you need a new grain."
)


def is_strict_agg_only() -> bool:
    """True when env STRICT_AGG_ONLY=1 or st.secrets['STRICT_AGG_ONLY'] is true."""
    if os.environ.get("STRICT_AGG_ONLY", "").strip() == "1":
        return True
    if st is not None:
        try:
            v = st.secrets.get("STRICT_AGG_ONLY")
            if v is True:
                return True
            if isinstance(v, str) and v.strip().lower() in ("true", "1"):
                return True
        except Exception:
            pass
    return False


def _raise_groupby(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(ADHOC_AGG_MESSAGE)


def _raise_merge(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(ADHOC_AGG_MESSAGE)


@contextmanager
def ban_adhoc_agg() -> Generator[None, None, None]:
    """
    Context manager: monkeypatch pandas DataFrame.groupby and merge (and pd.merge)
    to raise RuntimeError with ADHOC_AGG_MESSAGE. Restores originals on exit.
    Only use when is_strict_agg_only() is True; otherwise this is a no-op for clarity
    you can still use the context when strict is off (no patches applied if we check inside).
    Here we always patch when entering; caller should use only when is_strict_agg_only().
    """
    orig_groupby = pd.DataFrame.groupby
    orig_merge_df = pd.DataFrame.merge
    orig_merge_pd = getattr(pd, "merge", None)

    try:
        pd.DataFrame.groupby = _raise_groupby  # type: ignore[assignment]
        pd.DataFrame.merge = _raise_merge  # type: ignore[assignment]
        if orig_merge_pd is not None:
            pd.merge = _raise_merge  # type: ignore[assignment]
        yield
    finally:
        pd.DataFrame.groupby = orig_groupby  # type: ignore[assignment]
        pd.DataFrame.merge = orig_merge_df  # type: ignore[assignment]
        if orig_merge_pd is not None:
            pd.merge = orig_merge_pd  # type: ignore[assignment]


def install_strict_views_only_guard() -> None:
    """
    When STRICT_VIEWS_ONLY=1, permanently monkeypatch pd.DataFrame.groupby and merge/pd.merge
    to raise RuntimeError (no context manager). Call once at app startup.
    """
    if os.environ.get("STRICT_VIEWS_ONLY", "").strip() != "1":
        return
    pd.DataFrame.groupby = _raise_groupby  # type: ignore[assignment]
    pd.DataFrame.merge = _raise_merge  # type: ignore[assignment]
    if getattr(pd, "merge", None) is not None:
        pd.merge = _raise_merge  # type: ignore[assignment]


def ban_adhoc_agg_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: run f inside ban_adhoc_agg() so groupby/merge raise during f()."""

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with ban_adhoc_agg():
            return f(*args, **kwargs)

    return wrapper
