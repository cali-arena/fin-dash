"""
Idempotent DuckDB loader: rebuild analytics DB from Parquet (materialized tables, deterministic).
Reproducibility: dataset_version (agg manifest or data/.version.json), policy_hash, duckdb_manifest.json.
Skip rebuild when manifest exists with same dataset_version + policy_hash + pipeline_version unless --force.
Performance: CTAS ORDER BY month_end (+ grain dims), optional pragmas, smoke query suite after load.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicy,
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "configs/duckdb_policy.yml"
AGG_MANIFEST_REL = "agg/manifest.json"
VERSION_JSON_REL = "data/.version.json"

# Naming: parquet source -> DuckDB table name (policy-driven, consistent).
REQUIRED_AGG_SOURCES = frozenset({"firm_monthly", "channel_monthly", "ticker_monthly"})
OPTIONAL_AGG_SOURCES = frozenset({"geo_monthly", "segment_monthly"})

# Optional grain dims for CTAS ORDER BY (zone-map pruning). Key = source name (e.g. channel_monthly).
ORDER_BY_GRAIN_DIMS: dict[str, list[str]] = {
    "channel_monthly": ["channel_l1"],
    "ticker_monthly": ["ticker"],
}


def agg_source_to_table_name(source_name: str) -> str:
    """Agg parquet source name -> DuckDB table: analytics.agg_firm_monthly, agg_channel_monthly, etc."""
    return f"agg_{source_name}"


def curated_metrics_to_table_name() -> str:
    """Curated metrics_monthly -> analytics.metrics_monthly."""
    return "metrics_monthly"


def curated_dim_stem_to_table_name(parquet_stem: str) -> str:
    """Curated dim parquet stem -> DuckDB table: e.g. dim_channel.parquet -> dim_channel, dim_product -> dim_product."""
    return parquet_stem


def _owned_table_names(policy: DuckDBPolicy, dim_stems: list[str]) -> list[str]:
    """Deterministic list of table names we may create (for DROP TABLE IF EXISTS)."""
    out: list[str] = []
    for name in sorted((policy.source_paths.get("agg") or {}).keys()):
        out.append(agg_source_to_table_name(name))
    if (policy.source_paths.get("curated") or {}).get("metrics_monthly"):
        out.append(curated_metrics_to_table_name())
    out.extend(sorted(dim_stems))
    return out


class AnalyticsDBError(Exception):
    """Raised when required parquet missing or validation fails."""


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _dataset_version(root: Path) -> str:
    """Read dataset_version from agg/manifest.json (preferred) or data/.version.json (fallback)."""
    agg_path = root / AGG_MANIFEST_REL
    if agg_path.exists():
        try:
            data = json.loads(agg_path.read_text(encoding="utf-8"))
            v = (data.get("dataset_version") or "").strip()
            if v:
                return v
        except Exception:
            pass
    version_path = root / VERSION_JSON_REL
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
            v = (data.get("dataset_version") or "").strip()
            if v:
                return v
        except Exception:
            pass
    return "unknown"


def _policy_hash(policy: DuckDBPolicy) -> str:
    """SHA-256 of normalized duckdb policy JSON (sort keys)."""
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


def _schema_hash_from_meta(parquet_path: Path) -> str | None:
    """Read schema_hash from parquet's meta.json if present (e.g. agg/firm_monthly.parquet -> agg/firm_monthly.meta.json)."""
    meta_path = parquet_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        h = data.get("schema_hash")
        return str(h).strip() if h else None
    except Exception:
        return None


def _preflight(root: Path, policy: DuckDBPolicy) -> None:
    """
    Required vs optional preflight. Fail if any required source missing; warn only for optional.
    Report: list required vs optional sources; missing optional = warn only.
    """
    agg = policy.source_paths.get("agg") or {}
    curated = policy.source_paths.get("curated") or {}
    required_missing: list[str] = []
    optional_missing: list[str] = []

    for name, rel_path in agg.items():
        full = root / rel_path
        if not full.exists():
            if name in REQUIRED_AGG_SOURCES:
                required_missing.append(str(full))
            else:
                optional_missing.append(f"{name}: {full}")

    if curated.get("metrics_monthly"):
        full = root / curated["metrics_monthly"]
        if not full.exists():
            optional_missing.append(f"metrics_monthly: {full}")

    dims_glob = curated.get("dims_glob")
    if dims_glob:
        dim_files = _glob_parquet(root, dims_glob)
        if not dim_files:
            logger.warning("Preflight: optional dims_glob matched 0 files: %s", dims_glob)

    required_ok = [n for n in sorted(REQUIRED_AGG_SOURCES) if n in agg and (root / agg[n]).exists()]
    optional_agg = [n for n in sorted(agg.keys()) if n not in REQUIRED_AGG_SOURCES]
    logger.info(
        "Preflight: required=%s optional_agg=%s metrics_monthly=%s dims_glob=%s",
        required_ok,
        optional_agg,
        "yes" if curated.get("metrics_monthly") else "no",
        dims_glob or "no",
    )
    for path in optional_missing:
        logger.warning("Preflight: optional source missing (skipped): %s", path)

    if required_missing:
        raise AnalyticsDBError(
            f"Required parquet file(s) missing. Run agg builder first.\nMissing: {required_missing}\nRun: python -m pipelines.agg.build_aggs"
        )


def _drop_owned_tables(con, schema: str, owned: list[str]) -> None:
    """DROP TABLE IF EXISTS for each owned table (agg_*, metrics_monthly, dim_*)."""
    for name in owned:
        con.execute(f'DROP TABLE IF EXISTS "{schema}"."{name}"')


def _parquet_column_names(con, parquet_path: Path) -> list[str]:
    """Return column names of parquet file (for ORDER BY grain dims)."""
    path_str = str(parquet_path.resolve().as_posix()).replace("\\", "/").replace("'", "''")
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path_str}')").fetchall()
        return [r[0] for r in rows] if rows else []
    except Exception:
        return []


def _order_by_clause(has_month_end: bool, extra_columns: list[str] | None = None) -> str:
    """ORDER BY month_end first; then optional grain dims (for zone-map pruning)."""
    if not has_month_end:
        return ""
    parts = ["month_end"]
    if extra_columns:
        parts.extend(extra_columns)
    # Quote identifiers for safety
    quoted = ", ".join(f'"{c}"' for c in parts)
    return f" ORDER BY {quoted}"


def _create_table_from_parquet(
    con,
    schema: str,
    table_name: str,
    parquet_path: Path,
    order_by_month_end: bool = True,
    source_name: str | None = None,
) -> None:
    """CREATE TABLE ... AS SELECT * FROM read_parquet(path) [ORDER BY month_end, <grain_dims>]."""
    path_str = str(parquet_path.resolve().as_posix()).replace("\\", "/").replace("'", "''")
    extra: list[str] = []
    if order_by_month_end and source_name and source_name in ORDER_BY_GRAIN_DIMS:
        cols = _parquet_column_names(con, parquet_path)
        wanted = ORDER_BY_GRAIN_DIMS[source_name]
        extra = [c for c in wanted if c in cols]
    order_sql = _order_by_clause(order_by_month_end, extra if extra else None)
    sql = f'CREATE TABLE "{schema}"."{table_name}" AS SELECT * FROM read_parquet(\'{path_str}\'){order_sql}'
    con.execute(sql)


def _validate_table(con, schema: str, table_name: str, required: bool) -> tuple[int, str | None, str | None]:
    """Return (count, min_month_end, max_month_end). Raises AnalyticsDBError if required and count==0 or month_end missing when expected."""
    quoted = f'"{schema}"."{table_name}"'
    row = con.execute(f"SELECT COUNT(*) AS c FROM {quoted}").fetchone()
    count = int(row[0]) if row else 0
    if required and count == 0:
        raise AnalyticsDBError(f"Table {quoted} has 0 rows. Required tables must have data. Run: python -m pipelines.agg.build_aggs")
    min_me, max_me = None, None
    try:
        r = con.execute(f"SELECT MIN(month_end) AS mn, MAX(month_end) AS mx FROM {quoted}").fetchone()
        if r and (r[0] is not None or r[1] is not None):
            min_me = str(r[0]) if r[0] is not None else None
            max_me = str(r[1]) if r[1] is not None else None
    except Exception:
        pass
    return count, min_me, max_me


def _run_smoke_queries(con, schema: str) -> None:
    """
    Run smoke query suite after load; log runtime and rowcounts.
    Fail loudly (raise AnalyticsDBError) if any query errors.
    """
    qs = [
        ("count_agg_firm_monthly", f'SELECT COUNT(*) AS c FROM "{schema}"."agg_firm_monthly"'),
        ("latest_5_firm", f'SELECT month_end, end_aum FROM "{schema}"."agg_firm_monthly" ORDER BY month_end DESC LIMIT 5'),
        ("top10_channel_by_aum", f'''SELECT channel_l1, SUM(end_aum) AS total_aum FROM "{schema}"."agg_channel_monthly"
WHERE month_end = (SELECT MAX(month_end) FROM "{schema}"."agg_channel_monthly")
GROUP BY 1 ORDER BY 2 DESC LIMIT 10'''),
    ]
    for name, sql in qs:
        t0 = time.perf_counter()
        try:
            rows = con.execute(sql).fetchall()
        except Exception as e:
            raise AnalyticsDBError(f"Smoke query {name!r} failed: {e}\nSQL: {sql}") from e
        elapsed = time.perf_counter() - t0
        rowcount = len(rows)
        logger.info("Smoke %s | rows=%s | %.3fs", name, rowcount, elapsed)


def _glob_parquet(root: Path, pattern: str) -> list[Path]:
    if "*" not in pattern:
        single = root / pattern
        return [single] if single.exists() else []
    full_pattern = root / pattern
    parent = full_pattern.parent
    if not parent.exists():
        return []
    found = list(parent.glob(full_pattern.name))
    return sorted(f for f in found if f.suffix.lower() == ".parquet")


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
    force: bool = False,
) -> None:
    """
    Rebuild analytics DB: drop owned tables, create materialized tables from parquet, validate each.
    Writes analytics/duckdb_manifest.json. If manifest exists with same dataset_version+policy_hash+pipeline_version, skip unless --force.
    """
    root = _root_path(root)
    policy = load_and_validate_duckdb_policy(policy_path)

    if policy.refresh_mode != "rebuild":
        logger.warning("Only refresh_mode=rebuild is implemented; got %s", policy.refresh_mode)

    dataset_version = _dataset_version(root)
    policy_hash_val = _policy_hash(policy)
    try:
        from legacy.legacy_pipelines.agg.meta_utils import get_pipeline_version
        pipeline_version = get_pipeline_version(root)
    except ImportError:
        pipeline_version = __import__("os").environ.get("PIPELINE_VERSION", "unknown").strip() or "unknown"

    db_full = root / policy.db_path
    analytics_dir = db_full.parent
    db_full.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = analytics_dir / "duckdb_manifest.json"

    if manifest_path.exists() and not force:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("dataset_version") == dataset_version
                and existing.get("policy_hash") == policy_hash_val
                and existing.get("pipeline_version") == pipeline_version
            ):
                logger.info("Skip rebuild (manifest match). Use --force to rebuild.")
                return
        except Exception:
            pass

    _preflight(root, policy)
    schema = policy.schema

    import duckdb
    con = duckdb.connect(str(db_full))

    # Optional pragmas (safe defaults for performance)
    try:
        n = os.cpu_count() or 4
        con.execute(f"PRAGMA threads={n}")
        logger.info("PRAGMA threads=%s", n)
    except Exception as e:
        logger.warning("PRAGMA threads failed: %s", e)
    mem = os.environ.get("DUCKDB_MEMORY_LIMIT", "").strip()
    if mem:
        # Safe default pattern: e.g. 2GB, 512MB (no quotes in value)
        mem_safe = mem.replace("'", "").replace('"', "").strip()
        if mem_safe:
            try:
                con.execute(f"PRAGMA memory_limit='{mem_safe}'")
                logger.info("PRAGMA memory_limit=%s", mem_safe)
            except Exception as e:
                logger.warning("PRAGMA memory_limit failed: %s", e)
    tmp_dir = analytics_dir / "tmp"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_str = str(tmp_dir.resolve().as_posix()).replace("\\", "/").replace("'", "''")
        con.execute(f"PRAGMA temp_directory='{tmp_str}'")
        logger.info("PRAGMA temp_directory=%s", tmp_str)
    except Exception as e:
        logger.warning("PRAGMA temp_directory failed: %s", e)

    curated = policy.source_paths.get("curated") or {}
    dim_stems = sorted(set(p.stem for p in _glob_parquet(root, curated.get("dims_glob") or "")))
    owned = _owned_table_names(policy, dim_stems)
    loaded_tables: list[dict] = []

    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        _drop_owned_tables(con, schema, owned)

        agg = policy.source_paths.get("agg") or {}
        required_names = REQUIRED_AGG_SOURCES

        for name, rel_path in sorted(agg.items()):
            full = root / rel_path
            if not full.exists():
                continue
            table_name = agg_source_to_table_name(name)
            _create_table_from_parquet(con, schema, table_name, full, order_by_month_end=True, source_name=name)
            count, min_me, max_me = _validate_table(con, schema, table_name, required=name in required_names)
            logger.info("%s | rows=%s | month_end=%s .. %s", table_name, count, min_me or "?", max_me or "?")
            loaded_tables.append({
                "name": f"{schema}.{table_name}",
                "source_path": str(full.resolve()),
                "rowcount": count,
                "min_month_end": min_me,
                "max_month_end": max_me,
                "schema_hash": _schema_hash_from_meta(full),
            })

        if curated.get("metrics_monthly"):
            full = root / curated["metrics_monthly"]
            if full.exists():
                table_name = curated_metrics_to_table_name()
                _create_table_from_parquet(con, schema, table_name, full, order_by_month_end=True)
                count, min_me, max_me = _validate_table(con, schema, table_name, required=False)
                logger.info("%s | rows=%s | month_end=%s .. %s", table_name, count, min_me or "?", max_me or "?")
                loaded_tables.append({
                    "name": f"{schema}.{table_name}",
                    "source_path": str(full.resolve()),
                    "rowcount": count,
                    "min_month_end": min_me,
                    "max_month_end": max_me,
                    "schema_hash": _schema_hash_from_meta(full),
                })

        if curated.get("dims_glob"):
            for parquet_path in _glob_parquet(root, curated["dims_glob"]):
                table_name = curated_dim_stem_to_table_name(parquet_path.stem)
                _create_table_from_parquet(con, schema, table_name, parquet_path, order_by_month_end=False)
                count, min_me, max_me = _validate_table(con, schema, table_name, required=False)
                logger.info("%s | rows=%s | month_end=%s .. %s", table_name, count, min_me or "?", max_me or "?")
                loaded_tables.append({
                    "name": f"{schema}.{table_name}",
                    "source_path": str(parquet_path.resolve()),
                    "rowcount": count,
                    "min_month_end": min_me,
                    "max_month_end": max_me,
                    "schema_hash": _schema_hash_from_meta(parquet_path),
                })

        _run_smoke_queries(con, schema)
    finally:
        con.close()

    manifest = {
        "dataset_version": dataset_version,
        "policy_hash": policy_hash_val,
        "pipeline_version": pipeline_version,
        "db_path": policy.db_path,
        "schema": schema,
        "loaded_tables": loaded_tables,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Rebuild DuckDB analytics DB from Parquet (idempotent, deterministic).")
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to duckdb_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: cwd)")
    parser.add_argument("--force", action="store_true", help="Rebuild even when manifest matches (skip idempotency)")
    args = parser.parse_args()
    try:
        run(policy_path=args.policy, root=args.root, force=args.force)
    except DuckDBPolicyError as e:
        logger.error("Policy error: %s", e)
        raise SystemExit(1) from e
    except AnalyticsDBError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    except FileNotFoundError as e:
        logger.error("Missing file: %s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
