"""
DuckDB analytics layer for Streamlit: read from views only (no path-based parquet).
Shared resolver: load_duckdb_manifest, query_df, table(view_name).
Caching: st.cache_data keyed by dataset_version + SQL + params.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

DUCKDB_MANIFEST_REL = "analytics/duckdb_manifest.json"

# View name per tab (UI reads views only)
TAB_VIEW = {
    "Overview": "v_firm_monthly",
    "Channel": "v_channel_monthly",
    "Ticker": "v_ticker_monthly",
    "Geo": "v_geo_monthly",
    "Segment": "v_segment_monthly",
}


def load_duckdb_manifest(root: Path, path: str = DUCKDB_MANIFEST_REL) -> dict[str, Any]:
    """Load analytics/duckdb_manifest.json. Raises FileNotFoundError if missing."""
    root = Path(root)
    full = root / path
    if not full.exists():
        raise FileNotFoundError(
            f"DuckDB manifest not found: {full}. Run: python -m pipelines.duckdb.build_analytics_duckdb"
        )
    return json.loads(full.read_text(encoding="utf-8"))


def get_db_path(manifest: dict[str, Any], root: Path) -> str:
    """Return absolute path to the DuckDB file (root / manifest db_path)."""
    root = Path(root)
    rel = (manifest.get("db_path") or "").strip()
    if not rel:
        raise ValueError("duckdb_manifest.db_path is empty")
    return str((root / rel).resolve())


def _query_impl(
    db_path: str,
    sql: str,
    params: list[Any] | dict[str, Any] | None,
    dataset_version: str,
) -> pd.DataFrame:
    """Execute SQL and return DataFrame. Uses DuckDB parameterization (? or $name)."""
    import duckdb
    con = duckdb.connect(db_path, read_only=True)
    try:
        if params is None:
            out = con.execute(sql).fetchdf()
        elif isinstance(params, dict):
            out = con.execute(sql, params).fetchdf()
        else:
            out = con.execute(sql, params).fetchdf()
        return out
    finally:
        con.close()


def query_df(
    sql: str,
    root: Path,
    manifest: dict[str, Any] | None = None,
    params: list[Any] | dict[str, Any] | None = None,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """
    Run parameterized query against the analytics DuckDB. Safe: uses DuckDB params, no string interpolation.
    Cached via st.cache_data with key (dataset_version, sql, tuple(params)).
    """
    root = Path(root)
    if manifest is None:
        manifest = load_duckdb_manifest(root)
    db_path = get_db_path(manifest, root)
    version = (dataset_version or (manifest.get("dataset_version") or "").strip() or "unknown").strip()

    if st is not None:
        @st.cache_data(ttl=3600)
        def _cached(_version: str, _sql: str, _params: tuple[Any, ...] | None, _db: str) -> pd.DataFrame:
            p = list(_params) if _params else None
            return _query_impl(_db, _sql, p, _version)
        param_key: tuple[Any, ...] | None = None
        if isinstance(params, (list, tuple)):
            param_key = tuple(params)
        elif isinstance(params, dict):
            param_key = tuple(sorted(params.items()))
        return _cached(version, sql, param_key, db_path)
    return _query_impl(db_path, sql, params, version)


def table(
    view_name: str,
    root: Path,
    manifest: dict[str, Any] | None = None,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """SELECT * FROM <schema>.<view_name>. Uses manifest schema; cached by dataset_version + view_name."""
    import re
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", view_name):
        raise ValueError(f"Invalid view name: {view_name!r}")
    root = Path(root)
    if manifest is None:
        manifest = load_duckdb_manifest(root)
    schema = (manifest.get("schema") or "analytics").strip()
    sql = f'SELECT * FROM "{schema}"."{view_name}"'
    return query_df(sql, root, manifest=manifest, params=None, dataset_version=dataset_version)


def list_view_names(manifest: dict[str, Any]) -> list[str]:
    """Return created_views from manifest (for UI)."""
    return list(manifest.get("created_views") or [])
