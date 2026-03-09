"""
Persist curated fact_monthly with quality gates. Writes versioned and optional latest copy + meta.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.curate.fact_monthly import GRAIN

# Paths relative to root (data/cache/{version}/curated/ and curated/ for latest).
CACHE_DIR = "data/cache"
CURATED_DIR = "curated"
FACT_MONTHLY_PARQUET = "fact_monthly.parquet"
FACT_MONTHLY_META_JSON = "fact_monthly.meta.json"


def _atomic_write_json_safe(obj: dict[str, Any], path: Path) -> None:
    """Write JSON atomically; fsync wrapped for Windows."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)
    try:
        content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(temp_path, p)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def run_curated_fact_monthly_gates(
    df_fact: pd.DataFrame,
    stats: dict[str, Any],
) -> None:
    """
    Run quality gates before persist. Raises ValueError on failure.
    - No nulls in grain fields.
    - rows_out > 0 (from stats).
    - Uniqueness of grain.
    """
    errors: list[str] = []

    if stats.get("rows_out", 0) <= 0:
        errors.append("rows_out must be > 0")

    for g in GRAIN:
        if g not in df_fact.columns:
            errors.append(f"Grain column '{g}' missing from df_fact")
            continue
        n = df_fact[g].isna().sum()
        if n > 0:
            errors.append(f"Grain column '{g}' has {int(n)} null(s); no nulls allowed.")

    dup = df_fact.duplicated(subset=GRAIN, keep="first")
    if dup.any():
        errors.append(f"Grain is not unique: {int(dup.sum())} duplicate row(s).")

    if errors:
        raise ValueError("Curated fact_monthly gates failed: " + "; ".join(errors))


def _build_fact_monthly_meta(
    dataset_version: str,
    df_fact: pd.DataFrame,
    stats: dict[str, Any],
    *,
    status: str = "ok",
    gate_errors: list[str] | None = None,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, Any] = {
        "dataset_version": dataset_version,
        "created_at": created_at,
        "grain": list(GRAIN),
        "rows": len(df_fact),
        "status": status,
        "stats_summary": {
            "rows_in": stats.get("rows_in"),
            "rows_out": stats.get("rows_out"),
            "duplicate_groups_count": stats.get("duplicate_groups_count"),
            "aum_rule_used": stats.get("aum_rule_used"),
            "channel_mapping_used": stats.get("channel_mapping_used"),
            "null_counts_grain": stats.get("null_counts_grain"),
        },
    }
    if gate_errors:
        meta["gate_errors"] = gate_errors
    return meta


def persist_fact_monthly(
    df_fact: pd.DataFrame,
    stats: dict[str, Any],
    dataset_version: str,
    *,
    root: Path | None = None,
    write_latest_copy: bool = True,
) -> None:
    """
    Run quality gates, then persist fact_monthly to:
    - data/cache/{dataset_version}/curated/fact_monthly.parquet
    - data/cache/{dataset_version}/curated/fact_monthly.meta.json
    and optionally (write_latest_copy=True):
    - curated/fact_monthly.parquet
    - curated/fact_monthly.meta.json

    If gates fail, writes a failure meta report (curated/fact_monthly.meta.json with status=failed)
    then raises ValueError.
    """
    root = Path(root) if root is not None else Path.cwd()

    try:
        run_curated_fact_monthly_gates(df_fact, stats)
    except ValueError as e:
        gate_errors = [str(e)]
        failure_meta = _build_fact_monthly_meta(
            dataset_version,
            df_fact,
            stats,
            status="failed",
            gate_errors=gate_errors,
        )
        latest_meta_path = root / CURATED_DIR / FACT_MONTHLY_META_JSON
        latest_meta_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json_safe(failure_meta, latest_meta_path)
        raise ValueError(f"Curated fact_monthly gates failed: {e}") from e

    # Versioned paths
    versioned_dir = root / CACHE_DIR / dataset_version / CURATED_DIR
    versioned_dir.mkdir(parents=True, exist_ok=True)
    versioned_parquet = versioned_dir / FACT_MONTHLY_PARQUET
    versioned_meta_path = versioned_dir / FACT_MONTHLY_META_JSON

    from legacy.legacy_src.persist.raw_store import atomic_write_parquet

    atomic_write_parquet(df_fact, versioned_parquet)
    meta = _build_fact_monthly_meta(dataset_version, df_fact, stats)
    _atomic_write_json_safe(meta, versioned_meta_path)

    if write_latest_copy:
        latest_dir = root / CURATED_DIR
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_parquet = latest_dir / FACT_MONTHLY_PARQUET
        latest_meta_path = latest_dir / FACT_MONTHLY_META_JSON
        atomic_write_parquet(df_fact, latest_parquet)
        _atomic_write_json_safe(meta, latest_meta_path)
