"""
Cache policy contract: strict validation of configs/cache_policy.yml.

Validates cache policy and exposes:
- load_and_validate_cache_policy(path) -> CachePolicy

Validations:
- dataset_version_source.path non-empty and endswith .json
- dataset_version_source.key non-empty
- cache_keys includes exactly dataset_version, filter_state_hash, query_name
- ttl_seconds has fast/medium/heavy and all > 0
- query_classes values in {"fast", "medium", "heavy"}
- max_entries optional; if present, fast/medium/heavy and values > 0
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_CACHE_POLICY_PATH = "configs/cache_policy.yml"

REQUIRED_CACHE_KEYS = frozenset({"dataset_version", "filter_state_hash", "query_name"})
TTL_CLASSES = frozenset({"fast", "medium", "heavy"})


class CachePolicyError(Exception):
    """Raised when cache_policy.yml is missing or invalid."""


@dataclass
class CachePolicy:
    dataset_version_source_path: str
    dataset_version_source_key: str
    cache_keys: list[str]
    ttl_seconds: dict[str, int]
    query_classes: dict[str, str]
    max_entries: dict[str, int] | None


def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise CachePolicyError(f"{ctx} must be a mapping/object; got {type(obj).__name__}")
    return obj


def _require_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CachePolicyError(f"{ctx} must be a non-empty string")
    return value.strip()


def _require_list(value: Any, ctx: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise CachePolicyError(f"{ctx} must be a list; got {type(value).__name__}")
    return list(value)


def _validate_dataset_version_source(cfg: Mapping[str, Any], ctx: str) -> tuple[str, str]:
    raw = cfg.get("dataset_version_source")
    if raw is None or not isinstance(raw, Mapping):
        raise CachePolicyError(f"{ctx}.dataset_version_source must be a mapping")
    path = _require_str(raw.get("path"), f"{ctx}.dataset_version_source.path")
    if not path.endswith(".json"):
        raise CachePolicyError(
            f"{ctx}.dataset_version_source.path must end with .json; got {path!r}"
        )
    key = _require_str(raw.get("key"), f"{ctx}.dataset_version_source.key")
    return path, key


def _validate_cache_keys(cfg: Mapping[str, Any], ctx: str) -> list[str]:
    raw = cfg.get("cache_keys")
    keys_list = _require_list(raw, f"{ctx}.cache_keys")
    keys_str = []
    for i, k in enumerate(keys_list):
        if not isinstance(k, str) or not k.strip():
            raise CachePolicyError(f"{ctx}.cache_keys[{i}] must be a non-empty string; got {k!r}")
        keys_str.append(k.strip())
    present = set(keys_str)
    missing = REQUIRED_CACHE_KEYS - present
    if missing:
        raise CachePolicyError(
            f"{ctx}.cache_keys must include {sorted(REQUIRED_CACHE_KEYS)}; missing: {sorted(missing)}"
        )
    return keys_str


def _validate_ttl_seconds(cfg: Mapping[str, Any], ctx: str) -> dict[str, int]:
    raw = cfg.get("ttl_seconds")
    if raw is None or not isinstance(raw, Mapping):
        raise CachePolicyError(f"{ctx}.ttl_seconds must be a mapping")
    out: dict[str, int] = {}
    for cls in TTL_CLASSES:
        val = raw.get(cls)
        if val is None:
            raise CachePolicyError(f"{ctx}.ttl_seconds must have key {cls!r}")
        if not isinstance(val, int) or val <= 0:
            raise CachePolicyError(
                f"{ctx}.ttl_seconds.{cls} must be a positive integer; got {val!r}"
            )
        out[cls] = val
    return out


def _validate_query_classes(cfg: Mapping[str, Any], ctx: str) -> dict[str, str]:
    raw = cfg.get("query_classes")
    if raw is None or not isinstance(raw, Mapping):
        raise CachePolicyError(f"{ctx}.query_classes must be a mapping")
    out: dict[str, str] = {}
    for name, cls in raw.items():
        n = _require_str(name, f"{ctx}.query_classes key")
        c = _require_str(cls, f"{ctx}.query_classes.{name}")
        if c not in TTL_CLASSES:
            raise CachePolicyError(
                f"{ctx}.query_classes.{name} must be one of {sorted(TTL_CLASSES)}; got {c!r}"
            )
        out[n] = c
    return out


def _validate_max_entries(cfg: Mapping[str, Any], ctx: str) -> dict[str, int] | None:
    raw = cfg.get("max_entries")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise CachePolicyError(f"{ctx}.max_entries must be a mapping; got {type(raw).__name__}")
    out: dict[str, int] = {}
    for cls in TTL_CLASSES:
        val = raw.get(cls)
        if val is None:
            raise CachePolicyError(f"{ctx}.max_entries must have key {cls!r} when present")
        if not isinstance(val, int) or val <= 0:
            raise CachePolicyError(
                f"{ctx}.max_entries.{cls} must be a positive integer; got {val!r}"
            )
        out[cls] = val
    return out


def load_and_validate_cache_policy(
    path: str | Path = DEFAULT_CACHE_POLICY_PATH,
) -> CachePolicy:
    """
    Load configs/cache_policy.yml and return a validated CachePolicy.
    Raises CachePolicyError on any schema or validation issue.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise CachePolicyError(f"Cache policy config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise CachePolicyError(
            "PyYAML is required to load cache policy. Install with: pip install pyyaml"
        ) from e
    raw_text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text)
    except Exception as e:
        raise CachePolicyError(f"Failed to parse YAML {path}: {e}") from e

    cfg = _require_mapping(raw.get("cache") or raw, "cache")

    path_val, key_val = _validate_dataset_version_source(cfg, "cache")
    cache_keys = _validate_cache_keys(cfg, "cache")
    ttl_seconds = _validate_ttl_seconds(cfg, "cache")
    query_classes = _validate_query_classes(cfg, "cache")
    max_entries = _validate_max_entries(cfg, "cache")

    return CachePolicy(
        dataset_version_source_path=path_val,
        dataset_version_source_key=key_val,
        cache_keys=cache_keys,
        ttl_seconds=ttl_seconds,
        query_classes=query_classes,
        max_entries=max_entries,
    )
