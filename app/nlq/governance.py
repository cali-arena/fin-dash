"""
Registry loading and QuerySpec validation against governance rules.
Governance validation is the gate before any query execution; failures are explicit and safe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from models.query_spec import QuerySpec

# Default paths relative to project root (canonical app/nlq registries)
DEFAULT_METRIC_REGISTRY_PATH = "app/nlq/metric_registry.yml"
DEFAULT_DIM_REGISTRY_PATH = "app/nlq/dim_registry.yml"


class GovernanceError(Exception):
    """Raised when a QuerySpec fails governance validation."""

    pass


def load_metric_registry(path: str | Path = DEFAULT_METRIC_REGISTRY_PATH) -> dict[str, Any]:
    """Load metric registry YAML. Returns dict with 'version' and 'metrics' (list of metric entries)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
        if not p.exists():
            p = Path(__file__).resolve().parent / "metric_registry.yml"
    if not p.exists():
        p = Path(__file__).resolve().parent.parent.parent / "models" / "registries" / "metric_registry.yml"
    if not p.exists():
        raise FileNotFoundError(f"metric_registry not found: {path}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("metric_registry must be a YAML object")
    return data


def load_dim_registry(path: str | Path = DEFAULT_DIM_REGISTRY_PATH) -> dict[str, Any]:
    """Load dimension registry YAML. Returns dict with 'version', 'dimensions' (keyed by canonical key), and 'aliases'."""
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / p
        if not p.exists():
            p = Path(__file__).resolve().parent / "dim_registry.yml"
    if not p.exists():
        p = Path(__file__).resolve().parent.parent.parent / "models" / "registries" / "dim_registry.yml"
    if not p.exists():
        raise FileNotFoundError(f"dim_registry not found: {path}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("dim_registry must be a YAML object")
    return data


def _metrics_by_id(metric_reg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build metric_id -> entry map from registry."""
    metrics = metric_reg.get("metrics") or []
    if not isinstance(metrics, list):
        raise ValueError("metric_registry.metrics must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    for m in metrics:
        if not isinstance(m, dict):
            continue
        mid = m.get("metric_id")
        if mid is not None:
            by_id[str(mid).strip().lower()] = m
    return by_id


def _allowed_dims_set(metric_entry: dict[str, Any]) -> set[str] | None:
    """Return set of allowed dim keys, or None if '*' (any), or empty set if firm-only."""
    allowed = metric_entry.get("allowed_dims")
    if allowed is None:
        return set()
    if isinstance(allowed, list):
        if "*" in allowed or (len(allowed) == 1 and str(allowed[0]).strip() == "*"):
            return None
        return {str(d).strip().lower() for d in allowed if d}
    if str(allowed).strip() == "*":
        return None
    return set()


def validate_queryspec(
    spec: QuerySpec,
    metric_reg: dict[str, Any],
    dim_reg: dict[str, Any],
    out_logs: list[str] | None = None,
) -> None:
    """
    Validate QuerySpec against metric and dimension registries.
    Raises GovernanceError with a clear message if invalid.
    If out_logs is provided, append log strings for each check (for audit/debug).
    """
    def log(msg: str) -> None:
        if out_logs is not None:
            out_logs.append(msg)

    log("validate_queryspec: start")
    metrics_by_id = _metrics_by_id(metric_reg)
    dimensions = dim_reg.get("dimensions") or {}
    if not isinstance(dimensions, dict):
        log("dim_registry.dimensions is not an object")
        raise GovernanceError("dim_registry.dimensions must be an object")
    log("dim_registry.dimensions OK")

    mid_key = spec.metric_id.strip().lower()
    if mid_key not in metrics_by_id:
        log(f"metric_id '{spec.metric_id}' not in registry")
        raise GovernanceError(
            f"metric_id '{spec.metric_id}' not found in metric_registry; known: {sorted(metrics_by_id.keys())}"
        )
    metric_entry = metrics_by_id[mid_key]
    allowed = _allowed_dims_set(metric_entry)
    log(f"metric '{spec.metric_id}' found; allowed_dims checked")

    if allowed is not None and len(allowed) == 0:
        if spec.dimensions:
            log("metric is firm-only but dimensions non-empty")
            raise GovernanceError(
                f"metric '{spec.metric_id}' is firm-only (allowed_dims empty); dimensions must be empty, got: {spec.dimensions}"
            )
    elif allowed is not None:
        for d in spec.dimensions:
            if d not in allowed:
                log(f"dimension '{d}' not in allowed_dims")
                raise GovernanceError(
                    f"dimension '{d}' is not in metric '{spec.metric_id}' allowed_dims: {sorted(allowed)}"
                )
    log("allowed_dims OK")

    dim_keys = {str(k).strip().lower() for k in dimensions.keys()}
    for d in spec.dimensions:
        if d not in dim_keys:
            canon = normalize_dim_token(d, dim_reg)
            if not canon or canon not in dim_keys:
                log(f"dimension '{d}' not in dim_registry")
                raise GovernanceError(
                    f"dimension '{d}' not found in dim_registry; known: {sorted(dim_keys)}"
                )
    log("dim_registry dims OK")

    time_dim_keys = {"month_end"}
    allowed_filter_dims = set(spec.dimensions) | time_dim_keys
    for fkey in spec.filters:
        fk = fkey.strip().lower()
        canon = normalize_dim_token(fkey, dim_reg)
        if not canon:
            log(f"filter key '{fkey}' not in dim_registry")
            raise GovernanceError(f"filter key '{fkey}' not found in dim_registry")
        if canon not in allowed_filter_dims and canon not in time_dim_keys:
            log(f"filter key '{fkey}' not allowed for this metric")
            raise GovernanceError(
                f"filter key '{fkey}' (canonical: {canon}) must be one of spec.dimensions or time dim (e.g. month_end); allowed: {sorted(allowed_filter_dims)}"
            )
    log("filters OK")

    tr = spec.time_range
    if tr.start is not None and tr.end is not None and tr.start > tr.end:
        log("time_range start > end")
        raise GovernanceError(f"time_range start ({tr.start}) must be <= end ({tr.end})")
    log("validate_queryspec: passed")


def normalize_dim_token(token: str, dim_reg: dict[str, Any]) -> str | None:
    """
    Resolve a user dimension token to a canonical dimension key using synonyms and aliases.
    Returns the canonical key or None if not found.
    """
    if not token or not str(token).strip():
        return None
    t = str(token).strip().lower()
    dimensions = dim_reg.get("dimensions") or {}
    aliases = dim_reg.get("aliases") or {}
    if not isinstance(dimensions, dict):
        return None

    # canonical keys
    for canon, entry in dimensions.items():
        if str(canon).strip().lower() == t:
            return str(canon).strip().lower()

    # synonyms (per dimension)
    for canon, entry in dimensions.items():
        syns = entry.get("synonyms") if isinstance(entry, dict) else []
        if isinstance(syns, list):
            for s in syns:
                if str(s).strip().lower() == t:
                    return str(canon).strip().lower()

    # aliases
    if isinstance(aliases, dict):
        for alias, target in aliases.items():
            if str(alias).strip().lower() == t and target:
                return str(target).strip().lower()

    return None
