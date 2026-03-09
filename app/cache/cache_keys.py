"""
Canonical filter hashing and cache key building.

Single source of truth for "how keys are built":
- canonicalize_filters() → stable dict (sorted keys, normalized values)
- filter_state_hash() → sha1 of canonical JSON
- build_cache_key() → dataset_version:query_name:filter_state_hash

Decimals: normalized to string for stable representation (no float drift).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

# Optional numpy for scalar normalization
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False


def _to_serializable(value: Any) -> Any:
    """Convert a single value to a JSON-serializable, stable form. Dates → ISO YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "isoformat") and hasattr(value, "year"):
        try:
            return value.isoformat()[:10]
        except Exception:
            pass
    if _NUMPY_AVAILABLE and hasattr(np, "datetime64") and isinstance(value, np.datetime64):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, Decimal):
        return str(value)
    if _NUMPY_AVAILABLE and hasattr(np, "ndarray") and isinstance(value, (np.integer, np.floating)):
        return int(value) if isinstance(value, np.integer) else float(value)
    if isinstance(value, dict):
        return canonicalize_filters(value)
    if isinstance(value, (list, tuple)):
        return sorted((_to_serializable(x) for x in value), key=_sort_key)
    if isinstance(value, (int, float, str, bool)):
        return value
    return value


def _sort_key(x: Any) -> tuple[int, str]:
    """Stable sort key for list elements: type order then string repr."""
    if x is None:
        return (0, "")
    if isinstance(x, bool):
        return (1, str(x))
    if isinstance(x, int):
        return (2, str(x))
    if isinstance(x, float):
        return (3, str(x))
    if isinstance(x, str):
        return (4, x)
    return (5, json.dumps(x, sort_keys=True, default=str))


def canonicalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Produce a stable, JSON-serializable representation of filters.

    - Removes null/empty: None, [], "", {} (keys dropped)
    - Sorts dict keys recursively
    - Lists: sorted stably (order-insensitive for selectors); elements normalized
    - Dates/datetime: converted to ISO "YYYY-MM-DD" (date) or "YYYY-MM-DD" for Timestamp
    - Decimals: converted to string (stable; no float drift)
    - Numpy/pandas scalars: converted to Python int/float/str
    """
    if not isinstance(filters, dict):
        return filters
    out: dict[str, Any] = {}
    for k in sorted(filters.keys()):
        v = filters[k]
        if v is None:
            continue
        if v == "" or (isinstance(v, (list, tuple)) and len(v) == 0):
            continue
        if isinstance(v, dict) and len(v) == 0:
            continue
        normalized = _to_serializable(v)
        if isinstance(normalized, (list, tuple)):
            normalized = sorted(
                (_to_serializable(x) for x in normalized),
                key=_sort_key,
            )
        elif isinstance(normalized, dict):
            normalized = canonicalize_filters(normalized)
        out[k] = normalized
    return out


def filter_state_hash(filters: dict[str, Any]) -> str:
    """
    SHA1 hex digest of canonical filter state (stable JSON, no whitespace).
    """
    canonical = canonicalize_filters(filters)
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_cache_key(dataset_version: str, query_name: str, filters: dict[str, Any]) -> str:
    """
    Single canonical cache key: dataset_version:query_name:filter_state_hash(filters).
    """
    return f"{dataset_version}:{query_name}:{filter_state_hash(filters)}"


def cache_key(dataset_version: str, view: str, params: dict) -> str:
    """
    Stable cache key (legacy alias): uses build_cache_key with view as query_name.
    """
    return build_cache_key(dataset_version, view, params)
