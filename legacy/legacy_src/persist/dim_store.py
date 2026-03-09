"""
Persist star dimensions to curated/ (SCD Type 1 overwrite). Atomic parquet + meta JSON.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.persist.fact_monthly_store import atomic_write_parquet


def _atomic_write_meta(meta: dict[str, Any], path: Path) -> None:
    """Write meta JSON atomically (sort_keys, indent=2)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)
    try:
        content = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False)
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


def persist_dim(
    df_dim: pd.DataFrame,
    *,
    path_parquet: Path,
    path_meta: Path,
    dataset_version: str,
    schema_hash: str,
) -> dict[str, Any]:
    """
    Overwrite dimension parquet and meta (SCD Type 1). No validation here; caller validates first.

    Meta includes: dataset_version, schema_hash, rowcount, created_at_utc.
    Returns the meta dict.
    """
    path_parquet = Path(path_parquet)
    path_meta = Path(path_meta)
    path_parquet.parent.mkdir(parents=True, exist_ok=True)

    path_written, used_csv_fallback = atomic_write_parquet(df_dim, path_parquet)
    rowcount = len(df_dim)
    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, Any] = {
        "created_at_utc": created_at_utc,
        "dataset_version": dataset_version,
        "schema_hash": schema_hash,
        "rowcount": rowcount,
    }
    if used_csv_fallback:
        meta["parquet_fallback"] = "csv"
    _atomic_write_meta(meta, path_meta)
    return meta
