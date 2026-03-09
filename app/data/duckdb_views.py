"""
Views-only DuckDB surface for Streamlit: load duckdb_views_manifest.json, query views with parameterized WHERE.
UI must call views only (no direct table reads).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

VIEWS_MANIFEST_REL = "analytics/duckdb_views_manifest.json"


def load_views_manifest(root: Path | None = None) -> dict[str, Any] | None:
    """
    Load analytics/duckdb_views_manifest.json. Returns None if missing (e.g. create_views not run yet).
    """
    root = Path(root) if root is not None else Path.cwd()
    path = root / VIEWS_MANIFEST_REL
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def query_view(
    view_name: str,
    filters: dict[str, Any] | None = None,
    root: Path | None = None,
    manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Query a view by name with optional filters. Builds parameterized WHERE clauses only (no string interpolation).
    view_name: e.g. "v_firm_monthly" or "analytics.v_firm_monthly" (schema prefix optional).
    filters: dict of column -> value, e.g. {"month_end": "2024-01-01", "channel_l1": "Direct"}. Only columns
             listed in the manifest for that view are allowed.
    """
    root = Path(root) if root is not None else Path.cwd()
    if manifest is None:
        manifest = load_views_manifest(root)
    if not manifest:
        raise FileNotFoundError(
            f"Views manifest not found at {root / VIEWS_MANIFEST_REL}. "
            "Run: python -m pipelines.duckdb.build_analytics_db && python -m pipelines.duckdb.create_views"
        )
    schema = (manifest.get("schema") or "analytics").strip()
    db_path_rel = (manifest.get("db_path") or "").strip()
    if not db_path_rel:
        raise ValueError("Views manifest db_path is empty")
    db_path = str((root / db_path_rel).resolve())
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    # Normalize view name (strip schema prefix if present)
    if "." in view_name:
        _schema_prefix, view_name = view_name.split(".", 1)
    view_full = f"{schema}.{view_name}"
    views_list = manifest.get("views") or []
    view_info = next((v for v in views_list if v.get("name") == view_full), None)
    if not view_info:
        allowed = [v.get("name") for v in views_list if v.get("name")]
        raise ValueError(f"View {view_full!r} not in manifest. Allowed: {allowed}")

    allowed_columns = set(view_info.get("columns") or [])
    where_parts: list[str] = []
    param_values: list[Any] = []
    if filters:
        for col, val in filters.items():
            if col not in allowed_columns:
                raise ValueError(f"Filter column {col!r} not in view {view_full} columns: {sorted(allowed_columns)}")
            where_parts.append(f'"{col}" = ?')
            param_values.append(val)

    sql = f'SELECT * FROM "{schema}"."{view_name}"'
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        if param_values:
            return con.execute(sql, param_values).fetchdf()
        return con.execute(sql).fetchdf()
    finally:
        con.close()
