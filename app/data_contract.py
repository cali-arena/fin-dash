"""
Single canonical data loader and contract for deterministic, auditable dataset loading.
All pages should rely on this module (or the gateway that uses the same resolved path) for dataset identity.
Exposes: resolved path, DATA_VERSION, fingerprint, row count, date range, key totals; fails loudly if missing.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
_GATEWAY_STATS_WARNED = False

# Default dataset version when meta is missing (overridden by meta or APP_DATA_VERSION).
DATA_VERSION_DEFAULT = "unknown"

# Canonical firm view/table used for contract stats.
CONTRACT_VIEW_NAME = "v_firm_monthly"
CONTRACT_PARQUET_REL = "data/agg/firm_monthly.parquet"


@dataclass(frozen=True)
class DataContractResult:
    """Immutable result of canonical dataset load: path, version, fingerprint, stats, environment."""
    resolved_path: str
    dataset_version: str
    fingerprint: str
    row_count: int
    min_date: str | None
    max_date: str | None
    sum_end_aum: float
    sum_nnb: float
    sum_nnf: float
    environment: str
    backend: str  # "duckdb" | "parquet"


def detect_environment() -> str:
    """Explicit environment label: local, cloud, or unknown."""
    env = (os.environ.get("APP_ENV") or "").strip().lower()
    if env in ("local", "cloud"):
        return env
    # Streamlit Cloud runs with headless server.
    if os.environ.get("STREAMLIT_SERVER_HEADLESS", "").strip().lower() == "true":
        return "cloud"
    # Default to local when running in dev (no headless).
    return "local"


def _resolve_canonical_path_duckdb(root: Path) -> str | None:
    """Return resolved absolute path of DuckDB file if config exists and file is present."""
    try:
        from app.data.data_gateway import get_config
        config = get_config(root)
        db_path = config.get("db_path")
        if not db_path:
            return None
        p = Path(db_path)
        if p.exists():
            return str(p.resolve())
        return None
    except Exception:
        return None


def _resolve_canonical_path_parquet(root: Path) -> str | None:
    """Return resolved path of canonical parquet if it exists."""
    p = root / CONTRACT_PARQUET_REL.replace("/", os.sep)
    if p.exists():
        return str(p.resolve())
    return None


def resolve_canonical_dataset_path(root: Path | None = None) -> str:
    """
    Single canonical resolved file path for the raw source dataset.
    When APP_DATA_BACKEND=parquet: only data/agg/firm_monthly.parquet (parity).
    Otherwise: DuckDB from gateway config first, then data/agg/firm_monthly.parquet.
    Fails loudly (raises FileNotFoundError) if not found.
    """
    root = Path(root) if root is not None else Path.cwd()
    if (os.environ.get("APP_DATA_BACKEND", "").strip().lower() == "parquet"):
        path = _resolve_canonical_path_parquet(root)
        if path:
            return path
        raise FileNotFoundError(
            f"Parquet dataset required but missing: {root / CONTRACT_PARQUET_REL}. "
            "Set APP_DATA_BACKEND=parquet only when data/agg/firm_monthly.parquet is committed. "
            "Run ETL or copy firm_monthly.parquet to data/agg/."
        )
    path = _resolve_canonical_path_duckdb(root)
    if path:
        return path
    path = _resolve_canonical_path_parquet(root)
    if path:
        return path
    raise FileNotFoundError(
        "Expected dataset file is missing. "
        "Tried: (1) DuckDB from analytics/duckdb_views_manifest.json or DUCKDB_PATH env, "
        f"(2) {CONTRACT_PARQUET_REL}. "
        "Run ETL or place analytics.duckdb / firm_monthly.parquet and retry."
    )


def _load_dataset_version(root: Path) -> str:
    """Dataset version from meta or env; never silent fallback for production."""
    env_dv = (os.environ.get("APP_DATA_VERSION") or "").strip()
    if env_dv:
        return env_dv
    try:
        from app.data.data_gateway import load_dataset_version
        return load_dataset_version(root)
    except Exception:
        return DATA_VERSION_DEFAULT


def _stats_from_gateway(root: Path) -> dict[str, Any]:
    """
    Compute contract stats through the canonical data gateway only.
    No direct DuckDB session/query path is allowed here.
    Returns empty/default stats on any failure so the app does not crash.
    """
    empty = {"row_count": 0, "min_date": None, "max_date": None, "sum_end_aum": 0.0, "sum_nnb": 0.0, "sum_nnf": 0.0}
    global _GATEWAY_STATS_WARNED
    try:
        from app.data.data_gateway import Q_FIRM_MONTHLY, run_query

        if root is None:
            return empty
        df = run_query(Q_FIRM_MONTHLY, {}, root=root)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return empty
        logger.debug("Gateway contract stats input: rows=%s cols=%s", len(df), list(df.columns)[:10])

        work = df.copy()
        if "month_end" in work.columns:
            work["month_end"] = pd.to_datetime(work["month_end"], errors="coerce")
        for c in ("end_aum", "nnb", "nnf"):
            if c not in work.columns:
                work[c] = 0.0
            work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)

        min_d = work["month_end"].min() if "month_end" in work.columns and not work["month_end"].isna().all() else None
        max_d = work["month_end"].max() if "month_end" in work.columns and not work["month_end"].isna().all() else None
        return {
            "row_count": int(len(work)),
            "min_date": pd_ts_to_iso(min_d),
            "max_date": pd_ts_to_iso(max_d),
            "sum_end_aum": float(work["end_aum"].sum()) if "end_aum" in work.columns else 0.0,
            "sum_nnb": float(work["nnb"].sum()) if "nnb" in work.columns else 0.0,
            "sum_nnf": float(work["nnf"].sum()) if "nnf" in work.columns else 0.0,
        }
    except Exception as e:
        if not _GATEWAY_STATS_WARNED:
            logger.warning("Gateway contract stats failed: %s", e, exc_info=False)
            _GATEWAY_STATS_WARNED = True
        else:
            logger.debug("Gateway contract stats failed again: %s", e, exc_info=False)
        return empty


def pd_ts_to_iso(ts: Any) -> str | None:
    """Convert pandas/datetime to YYYY-MM-DD."""
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)


def _compute_fingerprint(dataset_version: str, stats: dict[str, Any]) -> str:
    """Deterministic fingerprint from version + row_count + date range + key sums."""
    payload = (
        f"{dataset_version}|{stats.get('row_count', 0)}|"
        f"{stats.get('min_date', '')}|{stats.get('max_date', '')}|"
        f"{stats.get('sum_end_aum', 0)}|{stats.get('sum_nnb', 0)}|{stats.get('sum_nnf', 0)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def get_data_contract(root: Path | None = None) -> DataContractResult:
    """
    Single canonical loader: resolve path, validate file exists, load version and stats, compute fingerprint.
    Logs the exact file path at runtime. Fails loudly if expected dataset is missing.
    """
    root = Path(root) if root is not None else Path.cwd()
    resolved_path = resolve_canonical_dataset_path(root)
    logger.info("Canonical dataset path: %s", resolved_path)

    dataset_version = _load_dataset_version(root)
    environment = detect_environment()

    # Determine backend and compute stats from canonical path
    if Path(resolved_path).suffix.lower() == ".duckdb":
        stats = _stats_from_gateway(root)
        backend = "duckdb"
    else:
        stats = _stats_from_gateway(root)
        backend = "parquet"

    fingerprint = _compute_fingerprint(dataset_version, stats)
    return DataContractResult(
        resolved_path=resolved_path,
        dataset_version=dataset_version,
        fingerprint=fingerprint,
        row_count=stats.get("row_count", 0),
        min_date=stats.get("min_date"),
        max_date=stats.get("max_date"),
        sum_end_aum=stats.get("sum_end_aum", 0.0),
        sum_nnb=stats.get("sum_nnb", 0.0),
        sum_nnf=stats.get("sum_nnf", 0.0),
        environment=environment,
        backend=backend,
    )


def get_data_contract_cached(root: Path | None = None) -> DataContractResult:
    """
    Cached wrapper for get_data_contract (use in Streamlit to avoid recompute every rerun).
    When Streamlit is available, uses st.cache_data(ttl=300).
    """
    root = Path(root) if root is not None else Path.cwd()
    try:
        import streamlit as st
        root_str = str(root)

        @st.cache_data(ttl=300, show_spinner=False)
        def _cached(_root_str: str) -> DataContractResult:
            return get_data_contract(Path(_root_str))

        return _cached(root_str)
    except Exception:
        return get_data_contract(root)
