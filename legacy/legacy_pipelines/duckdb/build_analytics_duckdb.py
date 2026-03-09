"""
Build DuckDB analytics layer: create schema, register parquet tables as views, create UI views.
Refresh mode "rebuild": VIEW-over-parquet for fast builds and light storage.
Preflight: verify all agg parquets + agg manifest. Schema drift: store schema_hashes from meta.json. View health: COUNT + month_end on v_firm_monthly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicy,
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "configs/duckdb_policy.yml"
AGG_MANIFEST_REL = "agg/manifest.json"


class DuckDBBuildError(Exception):
    """Raised when preflight or view health check fails."""


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _load_agg_manifest(root: Path) -> dict:
    """Load agg/manifest.json for dataset_version and policy_hash. Raises FileNotFoundError if missing."""
    path = root / AGG_MANIFEST_REL
    if not path.exists():
        raise FileNotFoundError(
            f"Agg manifest not found: {path}. Run: python -m pipelines.agg.build_aggs"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight(root: Path, policy: DuckDBPolicy) -> tuple[dict, dict[str, str]]:
    """
    Verify agg manifest exists and all parquet files in source_paths.agg exist.
    Returns (agg_manifest, schema_hashes). Raises DuckDBBuildError with missing list if any parquet missing.
    """
    # 1) agg/manifest.json must exist; read dataset_version
    agg_manifest = _load_agg_manifest(root)
    dataset_version = (agg_manifest.get("dataset_version") or "").strip() or "unknown"
    logger.info("Preflight: dataset_version=%s", dataset_version)

    # 2) Every parquet in source_paths.agg must exist
    missing: list[str] = []
    for name, rel_path in (policy.source_paths.get("agg") or {}).items():
        full = root / rel_path
        if not full.exists():
            missing.append(str(full))
    if missing:
        msg = (
            f"Preflight failed: {len(missing)} parquet(s) missing. Run agg builder first.\n"
            f"Missing: {missing}\n"
            f"Run: python -m pipelines.agg.build_aggs"
        )
        raise DuckDBBuildError(msg)

    # 3) Read schema_hash from each source's meta.json (if exists)
    schema_hashes: dict[str, str] = {}
    for name, rel_path in (policy.source_paths.get("agg") or {}).items():
        meta_path = (root / rel_path).with_suffix(".meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                h = meta.get("schema_hash")
                if isinstance(h, str) and h.strip():
                    schema_hashes[name] = h.strip()
            except Exception:
                pass
    return agg_manifest, schema_hashes


def _duckdb_policy_hash(policy: DuckDBPolicy) -> str:
    """Stable SHA-256 of canonical duckdb policy (for manifest and cache)."""
    canonical = {
        "db_path": policy.db_path,
        "schema": policy.schema,
        "refresh_mode": policy.refresh_mode,
        "source_paths": policy.source_paths,
        "views_to_create": policy.views_to_create,
        "reads_views_only": policy.reads_views_only,
    }
    payload = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_parent(path: Path) -> None:
    path = Path(path)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", parent)


def _register_parquet_view(con, schema: str, name: str, parquet_path: Path) -> None:
    """CREATE OR REPLACE VIEW schema.name AS SELECT * FROM read_parquet(path)."""
    path_str = str(parquet_path.resolve().as_posix())
    # Escape single quotes in path for SQL
    path_escaped = path_str.replace("\\", "/").replace("'", "''")
    sql = f'CREATE OR REPLACE VIEW "{schema}"."{name}" AS SELECT * FROM read_parquet(\'{path_escaped}\')'
    con.execute(sql)
    logger.info("View: %s.%s <- %s", schema, name, parquet_path)


def _create_ui_view(con, schema: str, view_name: str, source_table: str) -> None:
    """CREATE OR REPLACE VIEW schema.view_name AS SELECT * FROM schema.source_table."""
    sql = f'CREATE OR REPLACE VIEW "{schema}"."{view_name}" AS SELECT * FROM "{schema}"."{source_table}"'
    con.execute(sql)
    logger.info("UI view: %s.%s <- %s.%s", schema, view_name, schema, source_table)


def _glob_parquet(root: Path, pattern: str) -> list[Path]:
    """Return sorted list of existing parquet paths matching pattern (relative to root)."""
    if "*" not in pattern:
        single = root / pattern
        return [single] if single.exists() else []
    full_pattern = root / pattern
    parent = full_pattern.parent
    if not parent.exists():
        return []
    stem = full_pattern.name
    found = list(parent.glob(stem))
    return sorted(f for f in found if f.suffix.lower() == ".parquet")


def _view_health_check(con, schema: str, view_name: str = "v_firm_monthly") -> None:
    """Run COUNT(*) and MIN/MAX(month_end) on view; raise DuckDBBuildError if count=0 or month_end missing."""
    quoted = f'"{schema}"."{view_name}"'
    try:
        count_row = con.execute(f"SELECT COUNT(*) AS c FROM {quoted}").fetchone()
        count = count_row[0] if count_row else 0
        if count == 0:
            raise DuckDBBuildError(
                f"View health check failed: {quoted} returned 0 rows. "
                "Ensure agg tables have data (run: python -m pipelines.agg.build_aggs)."
            )
    except DuckDBBuildError:
        raise
    except Exception as e:
        raise DuckDBBuildError(f"View health check failed for {quoted}: {e}") from e

    try:
        range_row = con.execute(f"SELECT MIN(month_end) AS mn, MAX(month_end) AS mx FROM {quoted}").fetchone()
        if range_row is None or (range_row[0] is None and range_row[1] is None):
            raise DuckDBBuildError(
                f"View health check failed: {quoted} has no month_end or values. "
                "Ensure source has month_end column and data."
            )
    except DuckDBBuildError:
        raise
    except Exception as e:
        raise DuckDBBuildError(
            f"View health check failed: month_end missing or invalid in {quoted}. {e}"
        ) from e


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
) -> None:
    """
    Rebuild DuckDB analytics layer: preflight (parquet + manifest), schema, views, view health, manifest write.
    Writes analytics/duckdb_manifest.json with dataset_version, policy_hash, schema_hashes, created_views, created_sources.
    """
    root = _root_path(root)
    policy = load_and_validate_duckdb_policy(policy_path)

    if policy.refresh_mode != "rebuild":
        logger.warning("Only refresh_mode=rebuild is implemented; got %s", policy.refresh_mode)

    # Preflight: verify agg manifest + all agg parquets; collect schema_hashes from meta.json
    agg_manifest, schema_hashes = _preflight(root, policy)
    dataset_version = (agg_manifest.get("dataset_version") or "").strip() or "unknown"
    agg_policy_hash = (agg_manifest.get("policy_hash") or "").strip() or ""
    duckdb_policy_hash = _duckdb_policy_hash(policy)

    db_full = root / policy.db_path
    _ensure_parent(db_full)
    analytics_dir = db_full.parent

    try:
        import duckdb
    except ImportError as e:
        raise SystemExit(1) from e

    con = duckdb.connect(str(db_full))
    created_sources: list[str] = []
    created_views: list[str] = []

    try:
        # 1) CREATE SCHEMA IF NOT EXISTS
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{policy.schema}"')
        logger.info("Schema: %s", policy.schema)

        # 2) Register source_paths.agg as views (preflight already verified files exist)
        for name, rel_path in sorted(policy.source_paths["agg"].items()):
            full = root / rel_path
            _register_parquet_view(con, policy.schema, name, full)
            created_sources.append(name)

        # 3) Curated: metrics_monthly -> curated_metrics_monthly
        curated = policy.source_paths.get("curated") or {}
        metrics_rel = curated.get("metrics_monthly")
        if metrics_rel and isinstance(metrics_rel, str):
            full = root / metrics_rel
            if not full.exists():
                raise FileNotFoundError(f"Curated parquet not found: {full}")
            _register_parquet_view(con, policy.schema, "curated_metrics_monthly", full)
            created_sources.append("curated_metrics_monthly")

        # 4) Curated dims_glob -> one view per file
        dims_glob = curated.get("dims_glob")
        if dims_glob and isinstance(dims_glob, str):
            for parquet_path in _glob_parquet(root, dims_glob):
                name = parquet_path.stem
                _register_parquet_view(con, policy.schema, name, parquet_path)
                created_sources.append(name)

        # 5) UI views
        for view_name, view_cfg in sorted(policy.views_to_create.items()):
            source_ref = view_cfg.get("source") or ""
            if "." in source_ref:
                _schema_part, table_name = source_ref.split(".", 1)
                _create_ui_view(con, policy.schema, view_name, table_name)
                created_views.append(view_name)
            else:
                raise ValueError(f"Invalid view source {source_ref!r} for {view_name}")

        # 6) View health check: v_firm_monthly must have rows and month_end
        if "v_firm_monthly" in created_views:
            _view_health_check(con, policy.schema, "v_firm_monthly")

    finally:
        con.close()

    # 7) Write duckdb_manifest.json (schema_hashes for incremental/drift detection)
    manifest = {
        "dataset_version": dataset_version,
        "policy_hash": agg_policy_hash,
        "duckdb_policy_hash": duckdb_policy_hash,
        "db_path": policy.db_path,
        "schema": policy.schema,
        "created_views": created_views,
        "created_sources": created_sources,
        "reads_views_only": policy.reads_views_only,
    }
    if schema_hashes:
        manifest["schema_hashes"] = schema_hashes
    manifest_path = analytics_dir / "duckdb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Clear logs
    logger.info("dataset_version=%s", dataset_version)
    logger.info("created_sources: %s", created_sources)
    logger.info("created_views: %s", created_views)
    logger.info("Wrote %s", manifest_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Build DuckDB analytics layer (schema, parquet views, UI views)."
    )
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY_PATH,
        help="Path to duckdb_policy.yml (default: configs/duckdb_policy.yml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: cwd)",
    )
    args = parser.parse_args()
    try:
        run(policy_path=args.policy, root=args.root)
    except DuckDBPolicyError as e:
        logger.error("Policy error: %s", e)
        raise SystemExit(1) from e
    except DuckDBBuildError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        logger.error("Missing file: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
