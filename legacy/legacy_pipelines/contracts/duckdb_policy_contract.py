"""
DuckDB analytics-layer contract: strict config validation (no drift).

Validates configs/duckdb_policy.yml and exposes:
- load_and_validate_duckdb_policy(path) -> DuckDBPolicy

Validations:
- db_path non-empty, endswith .duckdb
- schema non-empty and valid identifier
- refresh_mode in {"rebuild", "incremental"}
- source_paths.agg must include at least firm_monthly, channel_monthly, ticker_monthly
- views_to_create keys start with "v_" and each has valid source reference (e.g. agg.firm_monthly)
- ui_rule.reads_views_only must be true
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_DUCKDB_POLICY_PATH = "configs/duckdb_policy.yml"

REQUIRED_AGG_TABLES = frozenset({"firm_monthly", "channel_monthly", "ticker_monthly"})
RECOMMENDED_AGG_TABLES = frozenset({"geo_monthly", "segment_monthly"})
REFRESH_MODES = frozenset({"rebuild", "incremental"})
VIEW_PREFIX = "v_"
VALID_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
SOURCE_REFERENCE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)$")


class DuckDBPolicyError(Exception):
    """Raised when duckdb_policy.yml is missing or invalid."""


@dataclass
class DuckDBPolicy:
    db_path: str
    schema: str
    refresh_mode: str
    source_paths: dict[str, Any]  # curated: {...}, agg: {...}
    views_to_create: dict[str, dict[str, str]]  # view_name -> {source: "agg.firm_monthly"}
    reads_views_only: bool


def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise DuckDBPolicyError(f"{ctx} must be a mapping/object; got {type(obj).__name__}")
    return obj


def _require_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DuckDBPolicyError(f"{ctx} must be a non-empty string")
    return value.strip()


def _validate_db_path(value: str, ctx: str) -> str:
    s = _require_str(value, ctx)
    if not s.endswith(".duckdb"):
        raise DuckDBPolicyError(f"{ctx} must end with .duckdb; got {s!r}")
    return s


def _validate_schema(value: Any, ctx: str) -> str:
    s = _require_str(value, ctx)
    if not VALID_IDENTIFIER.match(s):
        raise DuckDBPolicyError(f"{ctx} must be a valid identifier (alphanumeric + underscore); got {s!r}")
    return s


def _validate_refresh_mode(value: Any, ctx: str) -> str:
    s = _require_str(value, ctx).lower()
    if s not in REFRESH_MODES:
        raise DuckDBPolicyError(f"{ctx} must be one of {sorted(REFRESH_MODES)}; got {s!r}")
    return s


def _validate_source_paths(cfg: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    raw = cfg.get("source_paths")
    if raw is None or not isinstance(raw, Mapping):
        raise DuckDBPolicyError(f"{ctx}.source_paths must be a mapping")
    out: dict[str, Any] = {"curated": {}, "agg": {}}
    curated = raw.get("curated")
    if curated is not None and isinstance(curated, Mapping):
        out["curated"] = dict(curated)
    agg = raw.get("agg")
    if agg is None or not isinstance(agg, Mapping):
        raise DuckDBPolicyError(f"{ctx}.source_paths.agg must be a mapping")
    out["agg"] = {_require_str(k, f"source_paths.agg key {k!r}"): _require_str(v, f"source_paths.agg[{k!r}]") for k, v in agg.items()}
    missing = REQUIRED_AGG_TABLES - set(out["agg"].keys())
    if missing:
        raise DuckDBPolicyError(
            f"{ctx}.source_paths.agg must include at least {sorted(REQUIRED_AGG_TABLES)}; missing: {sorted(missing)}"
        )
    recommended_missing = RECOMMENDED_AGG_TABLES - set(out["agg"].keys())
    if recommended_missing:
        # Optional but recommended: log or raise with recommendation
        pass  # contract says "optional but recommended"; we only require firm/channel/ticker
    return out


def _parse_view_source(source_val: Any, view_name: str, agg_tables: set[str], ctx: str) -> str:
    """Validate and return source string (e.g. agg.firm_monthly). Table must be in agg_tables."""
    s = _require_str(source_val, f"{ctx} source")
    m = SOURCE_REFERENCE.match(s)
    if not m:
        raise DuckDBPolicyError(
            f"{ctx} source must be a valid reference like 'agg.firm_monthly' (schema.table); got {s!r}"
        )
    _schema, table = m.group(1), m.group(2)
    if table not in agg_tables:
        raise DuckDBPolicyError(
            f"{ctx} source {s!r} references table {table!r} which is not in source_paths.agg. "
            f"Available: {sorted(agg_tables)}"
        )
    return s


def _validate_views_to_create(cfg: Mapping[str, Any], agg_tables: set[str], ctx: str) -> dict[str, dict[str, str]]:
    raw = cfg.get("views_to_create")
    if raw is None or not isinstance(raw, Mapping):
        raise DuckDBPolicyError(f"{ctx}.views_to_create must be a mapping")
    out: dict[str, dict[str, str]] = {}
    for view_name, view_cfg in raw.items():
        if not isinstance(view_name, str) or not view_name.startswith(VIEW_PREFIX):
            raise DuckDBPolicyError(
                f"{ctx}.views_to_create keys must start with '{VIEW_PREFIX}'; got {view_name!r}"
            )
        vc = _require_mapping(view_cfg, f"{ctx}.views_to_create.{view_name}")
        source = vc.get("source")
        if source is None:
            raise DuckDBPolicyError(f"{ctx}.views_to_create.{view_name} must have 'source' key")
        parsed_source = _parse_view_source(source, view_name, agg_tables, f"views_to_create.{view_name}")
        out[view_name] = {"source": parsed_source}
    return out


def _validate_ui_rule(cfg: Mapping[str, Any], ctx: str) -> bool:
    raw = cfg.get("ui_rule")
    if raw is None or not isinstance(raw, Mapping):
        raise DuckDBPolicyError(f"{ctx}.ui_rule must be a mapping")
    reads = raw.get("reads_views_only")
    if reads is not True:
        raise DuckDBPolicyError(f"{ctx}.ui_rule.reads_views_only must be true (baseline rule); got {reads!r}")
    return True


def load_and_validate_duckdb_policy(path: str | Path = DEFAULT_DUCKDB_POLICY_PATH) -> DuckDBPolicy:
    """
    Load configs/duckdb_policy.yml and return a validated DuckDBPolicy.
    Raises DuckDBPolicyError on any schema or validation issue.
    """
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise DuckDBPolicyError(f"DuckDB policy config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise DuckDBPolicyError(
            "PyYAML is required to load duckdb policy. Install with: pip install pyyaml"
        ) from e
    raw_text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text)
    except Exception as e:
        raise DuckDBPolicyError(f"Failed to parse YAML {path}: {e}") from e

    cfg = _require_mapping(raw.get("duckdb") or raw, "duckdb")

    db_path = _validate_db_path(cfg.get("db_path"), "duckdb.db_path")
    schema = _validate_schema(cfg.get("schema"), "duckdb.schema")
    refresh_mode = _validate_refresh_mode(cfg.get("refresh_mode"), "duckdb.refresh_mode")
    source_paths = _validate_source_paths(cfg, "duckdb")
    agg_tables = set(source_paths["agg"].keys())
    views_to_create = _validate_views_to_create(cfg, agg_tables, "duckdb")
    reads_views_only = _validate_ui_rule(cfg, "duckdb")

    return DuckDBPolicy(
        db_path=db_path,
        schema=schema,
        refresh_mode=refresh_mode,
        source_paths=source_paths,
        views_to_create=views_to_create,
        reads_views_only=reads_views_only,
    )
