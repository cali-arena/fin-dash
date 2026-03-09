"""
Strict agg-only data access layer for Streamlit. UI reads ONLY agg/*.parquet via manifest.
No scanning of curated/metrics_monthly or raw tables.
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

MANIFEST_REL = "data/agg/manifest.json"
DEFAULT_TIME_COL = "month_end"
MEASURE_COLS = frozenset({
    "begin_aum", "end_aum", "nnb", "nnf", "market_pnl",
    "ogr", "market_impact_rate", "fee_yield",
})


def _load_manifest_impl(root: Path, path: str) -> dict[str, Any]:
    """Internal: read manifest JSON. Tries data/agg/manifest.json then agg/manifest.json."""
    root = Path(root)
    for candidate in (path, "agg/manifest.json"):
        full = root / candidate
        if full.exists():
            return json.loads(full.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Agg manifest not found: {root / path}. Run: python -m pipelines.agg.build_aggs")


def load_manifest(root: Path, path: str = MANIFEST_REL) -> dict[str, Any]:
    """Load agg manifest from root/path (agg/manifest.json). Raises FileNotFoundError if missing.
    When Streamlit is present, cached by (root, path) so dataset_version-driven cache keys stay consistent."""
    root = Path(root)
    if st is not None:
        @st.cache_data(ttl=3600)
        def _cached(_root: Path, _path: str) -> dict[str, Any]:
            return _load_manifest_impl(_root, _path)
        return _cached(root, path)
    return _load_manifest_impl(root, path)


def get_table_path(table_name: str, manifest: dict[str, Any]) -> str:
    """Return relative path (e.g. agg/firm_monthly.parquet) for table. Raises KeyError if not in manifest."""
    tables = manifest.get("tables") or []
    for t in tables:
        if t.get("name") == table_name:
            return t.get("path") or f"data/agg/{table_name}.parquet"
    available = [t.get("name") for t in tables if t.get("name")]
    raise KeyError(f"Table {table_name!r} not in agg manifest. Available: {available!r}")


def get_dataset_version(manifest: dict[str, Any]) -> str:
    """Return dataset_version from agg/manifest.json (for all st.cache_data keys)."""
    return (manifest.get("dataset_version") or "").strip() or "unknown"


def get_firm_monthly_month_range(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Min/max month_end for firm_monthly from manifest table entry (no table read). Returns (min_str, max_str) or (None, None)."""
    for t in manifest.get("tables") or []:
        if t.get("name") == "firm_monthly":
            return (t.get("min_month_end") or t.get("min_month")), (t.get("max_month_end") or t.get("max_month"))
    return (None, None)


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce deterministic dtypes: month_end datetime, measure columns float."""
    out = df.copy()
    if DEFAULT_TIME_COL in out.columns:
        out[DEFAULT_TIME_COL] = pd.to_datetime(out[DEFAULT_TIME_COL], utc=False)
    for c in MEASURE_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def _load_table_impl(
    table_name: str,
    root: Path,
    columns: list[str] | None,
    dataset_version: str,
) -> pd.DataFrame:
    """Internal: resolve path from manifest, read parquet, enforce dtypes."""
    manifest = load_manifest(root)
    path_str = get_table_path(table_name, manifest)
    full_path = root / path_str
    if not full_path.exists():
        raise FileNotFoundError(
            f"Agg table not found: {full_path}. Run: python -m pipelines.agg.build_aggs"
        )
    df = pd.read_parquet(full_path, columns=columns)
    return _enforce_dtypes(df)


def _reads_views_only_guard(root: Path) -> None:
    """If duckdb manifest has reads_views_only=true, raise to block path-based parquet read."""
    try:
        from app.duckdb_store import load_duckdb_manifest
        duck_manifest = load_duckdb_manifest(root)
    except FileNotFoundError:
        return
    if duck_manifest.get("reads_views_only") is True:
        raise RuntimeError(
            "UI reads views only. Path-based parquet reads are disabled. "
            "Run: python -m pipelines.duckdb.build_analytics_duckdb"
        )


def load_table(
    table_name: str,
    root: Path,
    columns: list[str] | None = None,
    dataset_version: str | None = None,
) -> pd.DataFrame:
    """
    Load agg table by name. Path resolved via manifest only (no direct paths).
    Blocked when duckdb_manifest.reads_views_only=true (use duckdb_store.table(view) instead).
    Optional column subset for speed. Enforces month_end datetime, measures float.
    Cached via st.cache_data with key (dataset_version, table_name, columns).
    """
    root = Path(root)
    _reads_views_only_guard(root)
    manifest = load_manifest(root)
    version = (dataset_version or get_dataset_version(manifest)).strip() or "unknown"

    if st is not None:
        @st.cache_data(ttl=3600)
        def _cached(_version: str, _name: str, _cols: tuple[str] | None, _root: Path) -> pd.DataFrame:
            return _load_table_impl(_name, _root, list(_cols) if _cols else None, _version)
        return _cached(version, table_name, tuple(columns) if columns else None, root)
    return _load_table_impl(table_name, root, columns, version)


def list_table_names(root: Path) -> list[str]:
    """Return table names from manifest (for UI selector)."""
    try:
        manifest = load_manifest(root)
        return [t.get("name") for t in (manifest.get("tables") or []) if t.get("name")]
    except FileNotFoundError:
        return []
