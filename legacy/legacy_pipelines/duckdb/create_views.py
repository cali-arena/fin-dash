"""
Step 3: Create stable DuckDB views (v_*) as the ONLY UI query surface.
Thin projection + centralized rate logic (recompute if not stored). Idempotent: CREATE OR REPLACE VIEW.
Writes analytics/duckdb_views_manifest.json (views-only contract for Streamlit).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from legacy.legacy_pipelines.contracts.duckdb_policy_contract import (
    DuckDBPolicy,
    DuckDBPolicyError,
    load_and_validate_duckdb_policy,
)

logger = logging.getLogger(__name__)

DEFAULT_POLICY_PATH = "configs/duckdb_policy.yml"
DUCKDB_MANIFEST_REL = "analytics/duckdb_manifest.json"
DUCKDB_VIEWS_MANIFEST_REL = "analytics/duckdb_views_manifest.json"

# Required agg tables that must exist before creating views.
REQUIRED_AGG_TABLES = ("agg_firm_monthly", "agg_channel_monthly", "agg_ticker_monthly")

# Minimum columns each view must expose; fail if any missing.
VIEW_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "v_firm_monthly": ["month_end", "end_aum", "nnb", "market_pnl"],
    "v_channel_monthly": ["month_end", "channel_l1"],
    "v_ticker_monthly": ["month_end", "product_ticker"],
    "v_geo_monthly": ["month_end"],
    "v_segment_monthly": ["month_end"],
}

# Base measure columns (order preserved).
BASE_COLS = ("month_end", "begin_aum", "end_aum", "nnb", "nnf", "market_pnl")
# Optional stored rate columns; we recompute and COALESCE with these when present.
RATE_COLS = ("ogr", "market_impact_rate", "fee_yield")
# Grain columns per view (only include if column exists in table).
VIEW_GRAIN_COLUMNS: dict[str, list[str]] = {
    "v_firm_monthly": [],
    "v_channel_monthly": ["channel_l1", "channel_l2"],
    "v_ticker_monthly": ["product_ticker"],
    "v_geo_monthly": ["region", "geo"],
    "v_segment_monthly": ["segment", "sub_segment"],
}


class CreateViewsError(Exception):
    """Raised when required tables are missing or health check fails."""


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _source_to_table_name(source_ref: str) -> str:
    """Map policy source 'agg.firm_monthly' -> table name 'agg_firm_monthly'."""
    if "." in source_ref:
        _prefix, name = source_ref.split(".", 1)
        return f"agg_{name}"
    return f"agg_{source_ref}"


def _table_exists(con, schema: str, table_name: str) -> bool:
    """Return True if schema.table_name exists."""
    try:
        row = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? LIMIT 1",
            [schema, table_name],
        ).fetchone()
        return row is not None
    except Exception:
        return False


def get_table_columns(con, schema: str, table_name: str) -> list[str]:
    """
    Return column names of the table (DESCRIBE). Use to detect e.g. channel_l2 or stored rate columns.
    """
    try:
        rows = con.execute(
            f'DESCRIBE "{schema}"."{table_name}"'
        ).fetchall()
        # DuckDB DESCRIBE returns (column_name, type, null, key, default, extra)
        return [r[0] for r in rows] if rows else []
    except Exception:
        return []


def build_view_sql(
    con,
    schema: str,
    table_name: str,
    view_name: str,
) -> str:
    """
    Build SELECT SQL for the view: thin projection + computed rates.
    - Grain columns: only those in VIEW_GRAIN_COLUMNS[view_name] that exist in the table.
    - Rates: ogr, market_impact_rate, fee_yield with COALESCE(stored, computed) when stored exists.
    """
    cols = get_table_columns(con, schema, table_name)
    col_set = set(cols)
    quoted_t = f'"{schema}"."{table_name}"'

    # Grain columns (in order, only if present)
    grain_candidates = VIEW_GRAIN_COLUMNS.get(view_name) or []
    grain_selected = [c for c in grain_candidates if c in col_set]

    # Base columns (only those present)
    base_selected = [c for c in BASE_COLS if c in col_set]
    select_parts: list[str] = []

    # month_end first (if present)
    if "month_end" in col_set:
        select_parts.append('t."month_end"')
    for c in grain_selected:
        select_parts.append(f't."{c}"')
    for c in base_selected:
        if c != "month_end":
            select_parts.append(f't."{c}"')

    # Computed rate expressions (same logic for all views); only add when denominator/numerator exist
    if "begin_aum" in col_set and "nnb" in col_set:
        computed_ogr = "CASE WHEN t.\"begin_aum\" > 0 THEN t.\"nnb\" / t.\"begin_aum\" ELSE NULL END"
        select_parts.append(
            f'COALESCE(t."ogr", {computed_ogr}) AS "ogr"' if "ogr" in col_set else f"{computed_ogr} AS \"ogr\""
        )
    if "begin_aum" in col_set and "market_pnl" in col_set:
        computed_mir = "CASE WHEN t.\"begin_aum\" > 0 THEN t.\"market_pnl\" / t.\"begin_aum\" ELSE NULL END"
        select_parts.append(
            f'COALESCE(t."market_impact_rate", {computed_mir}) AS "market_impact_rate"'
            if "market_impact_rate" in col_set
            else f"{computed_mir} AS \"market_impact_rate\""
        )
    if "nnb" in col_set and "nnf" in col_set:
        computed_fy = "CASE WHEN t.\"nnb\" > 0 THEN t.\"nnf\" / t.\"nnb\" ELSE NULL END"
        select_parts.append(
            f'COALESCE(t."fee_yield", {computed_fy}) AS "fee_yield"' if "fee_yield" in col_set else f"{computed_fy} AS \"fee_yield\""
        )

    return f"SELECT {', '.join(select_parts)} FROM {quoted_t} t"


def _create_or_replace_view_with_sql(
    con, schema: str, view_name: str, table_name: str, view_sql: str
) -> None:
    """CREATE OR REPLACE VIEW schema.view_name AS <view_sql>."""
    quoted_view = f'"{schema}"."{view_name}"'
    con.execute(f"CREATE OR REPLACE VIEW {quoted_view} AS {view_sql}")


def _verify_view_columns(con, schema: str, view_name: str) -> list[str]:
    """
    Return column names of the view. Raise CreateViewsError if view is missing or lacks required columns.
    """
    cols = get_table_columns(con, schema, view_name)
    if not cols:
        raise CreateViewsError(
            f"View {schema}.{view_name} has no columns or does not exist. "
            "Run: python -m pipelines.duckdb.build_analytics_db && python -m pipelines.duckdb.create_views"
        )
    required = VIEW_REQUIRED_COLUMNS.get(view_name)
    if required:
        missing = [c for c in required if c not in cols]
        if missing:
            raise CreateViewsError(
                f"View {schema}.{view_name} missing required columns: {missing}. "
                f"Expected at least: {required}. Run create_views after rebuilding analytics DB."
            )
    return cols


def _health_check(con, schema: str) -> None:
    """Run post-create health checks on v_firm_monthly; fail loudly on error."""
    view = f'"{schema}"."v_firm_monthly"'
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()
        count = int(row[0]) if row else 0
        logger.info("Health check: %s rowcount=%s", view, count)
    except Exception as e:
        raise CreateViewsError(f"Health check COUNT(*) failed on {view}: {e}") from e
    try:
        row = con.execute(f"SELECT MIN(month_end), MAX(month_end) FROM {view}").fetchone()
        logger.info("Health check: %s month_end range=%s .. %s", view, row[0], row[1])
    except Exception as e:
        raise CreateViewsError(f"Health check MIN/MAX(month_end) failed on {view}: {e}") from e


def _load_duckdb_manifest_repr(analytics_dir: Path) -> tuple[str, str]:
    """Read analytics/duckdb_manifest.json if present; return (dataset_version, policy_hash)."""
    path = analytics_dir / "duckdb_manifest.json"
    if not path.exists():
        return "unknown", "unknown"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            (data.get("dataset_version") or "unknown").strip(),
            (data.get("policy_hash") or "unknown").strip(),
        )
    except Exception:
        return "unknown", "unknown"


def _write_views_manifest(
    analytics_dir: Path,
    db_path: str,
    schema: str,
    dataset_version: str,
    policy_hash: str,
    views: list[dict],
) -> None:
    """Write analytics/duckdb_views_manifest.json."""
    manifest = {
        "db_path": db_path,
        "schema": schema,
        "dataset_version": dataset_version,
        "policy_hash": policy_hash,
        "views": views,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = analytics_dir / "duckdb_views_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", out_path)


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
) -> None:
    """
    Connect to DuckDB, ensure schema exists, create or replace v_* views from policy.views_to_create.
    Only creates a view if the underlying agg table exists. Fails if required agg tables are missing.
    """
    root = _root_path(root)
    policy = load_and_validate_duckdb_policy(policy_path)
    db_full = root / policy.db_path
    if not db_full.exists():
        raise CreateViewsError(
            f"DuckDB file not found: {db_full}. "
            "Run: python -m pipelines.duckdb.build_analytics_db"
        )

    import duckdb
    con = duckdb.connect(str(db_full))
    schema = policy.schema

    try:
        # Require base tables
        missing = [t for t in REQUIRED_AGG_TABLES if not _table_exists(con, schema, t)]
        if missing:
            raise CreateViewsError(
                f"Required agg tables missing in {schema}: {missing}. "
                "Run: python -m pipelines.duckdb.build_analytics_db"
            )

        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

        # Create or replace views (thin projection + centralized rate logic)
        built_views: list[tuple[str, str]] = []  # (view_name, table_name)
        for view_name, view_cfg in sorted(policy.views_to_create.items()):
            source_ref = view_cfg.get("source", "")
            table_name = _source_to_table_name(source_ref)
            if not _table_exists(con, schema, table_name):
                logger.debug("Skip view %s: table %s does not exist", view_name, table_name)
                continue
            view_sql = build_view_sql(con, schema, table_name, view_name)
            _create_or_replace_view_with_sql(con, schema, view_name, table_name, view_sql)
            built_views.append((view_name, table_name))
            logger.info("Created view %s.%s", schema, view_name)

        _health_check(con, schema)

        # Guardrails: verify each view exists and has required columns
        views_payload: list[dict] = []
        for view_name, table_name in built_views:
            columns = _verify_view_columns(con, schema, view_name)
            views_payload.append({
                "name": f"{schema}.{view_name}",
                "source_table": f"{schema}.{table_name}",
                "columns": columns,
            })

        # Write views-only manifest (surface contract for Streamlit)
        analytics_dir = db_full.parent
        dataset_version, policy_hash = _load_duckdb_manifest_repr(analytics_dir)
        _write_views_manifest(
            analytics_dir,
            db_path=policy.db_path,
            schema=schema,
            dataset_version=dataset_version,
            policy_hash=policy_hash,
            views=views_payload,
        )
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Create or replace DuckDB v_* views (idempotent). Requires build_analytics_db first."
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH, help="Path to duckdb_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default: cwd)")
    args = parser.parse_args()
    try:
        run(policy_path=args.policy, root=args.root)
    except DuckDBPolicyError as e:
        logger.error("Policy error: %s", e)
        raise SystemExit(1) from e
    except CreateViewsError as e:
        logger.error("%s", e)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
