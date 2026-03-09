"""
Persist QA artifacts related to channel mapping (e.g., unmapped channel combinations).
Atomic writes; deterministic CSV and JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_csv(df: pd.DataFrame, path: Path) -> bytes:
    """
    Write DataFrame to CSV atomically (temp -> os.replace).

    - No index.
    - Deterministic line endings ("\n").
    - Returns the raw bytes written (for hashing).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}"
    tmp = p.parent / (p.name + suffix)

    # Deterministic column order: use existing order in df
    csv_bytes: bytes
    try:
        # Write to temp path first
        df.to_csv(
            tmp,
            index=False,
            lineterminator="\n",
        )
        with open(tmp, "rb") as f:
            csv_bytes = f.read()
        try:
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise

    return csv_bytes


def _atomic_write_meta(meta: dict[str, Any], path: Path) -> None:
    """Write meta JSON atomically (sort_keys, indent=2, deterministic)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}"
    tmp = p.parent / (p.name + suffix)
    try:
        content = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def persist_unmapped_channels(
    df_unmapped: pd.DataFrame,
    *,
    dataset_version: str,
    path_csv: str | Path = "qa/unmapped_channels.csv",
    path_meta: str | Path = "qa/unmapped_channels.meta.json",
) -> dict[str, Any]:
    """
    Persist unmapped channel key combinations as a QA artifact.

    - Writes qa/unmapped_channels.csv atomically (no index, deterministic).
    - Writes qa/unmapped_channels.meta.json with:
        - dataset_version
        - rowcount (distinct keys: len(df_unmapped))
        - total_unmapped_rows (sum of column 'row_count', if present; else 0)
        - created_at_utc (ISO, second precision, UTC)
        - file_sha256 (SHA-256 hex digest of CSV bytes)
    - Returns the meta dict.
    """
    csv_path = Path(path_csv)
    meta_path = Path(path_meta)

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_bytes = atomic_write_csv(df_unmapped, csv_path)
    sha256 = hashlib.sha256(csv_bytes).hexdigest()

    rowcount = int(len(df_unmapped))
    if "row_count" in df_unmapped.columns and rowcount > 0:
        total_unmapped_rows = int(df_unmapped["row_count"].sum())
    else:
        total_unmapped_rows = 0

    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, Any] = {
        "created_at_utc": created_at_utc,
        "dataset_version": dataset_version,
        "rowcount": rowcount,
        "total_unmapped_rows": total_unmapped_rows,
        "file_sha256": sha256,
        "path_csv": str(csv_path),
    }

    _atomic_write_meta(meta, meta_path)
    return meta

