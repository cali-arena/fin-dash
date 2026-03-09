"""
Deterministic persistence for DATA RAW: parquet (or CSV fallback), sidecar meta, ingest report.
All writes are atomic (temp -> fsync -> os.replace). Sidecar meta always carries provenance.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Default output directory and filenames (paths relative to root/cwd unless overridden).
RAW_DIR_NAME = "raw"
DATA_RAW_PARQUET = "data_raw.parquet"
DATA_RAW_REJECTS_PARQUET = "data_raw_rejects.parquet"
DATA_RAW_META_JSON = "data_raw.meta.json"
INGEST_REPORT_JSON = "ingest_report.json"


def atomic_write_bytes(data: bytes, path: Path) -> None:
    """Write bytes atomically: temp file in same dir, flush+fsync, then os.replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
            f.flush()
            if hasattr(os, "fsync"):
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


def atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    """Write JSON atomically: json.dumps(indent=2, sort_keys=True, ensure_ascii=False) then atomic write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)
    try:
        content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, p)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def _parquet_available() -> bool:
    """True if pandas can write parquet (pyarrow or fastparquet)."""
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"_parquet_check_{os.getpid()}.parquet"
    try:
        pd.DataFrame({"x": [1]}).to_parquet(tmp, index=False)
        return True
    except Exception:
        return False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> tuple[Path, bool]:
    """
    Write DataFrame to path atomically (temp -> os.replace). Prefer parquet; if parquet
    engine unavailable, write CSV to same stem with .csv extension. Returns (path_written, used_csv_fallback).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)

    if _parquet_available():
        try:
            df.to_parquet(temp_path, index=False)
            try:
                with open(temp_path, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass
            os.replace(temp_path, p)
            return (p, False)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            # Fall through to CSV fallback
            pass

    # Fallback: CSV at same path stem with .csv extension
    csv_path = p.with_suffix(".csv")
    temp_csv = csv_path.parent / (csv_path.name + suffix)
    try:
        df.to_csv(temp_csv, index=False)
        try:
            with open(temp_csv, "rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        os.replace(temp_csv, csv_path)
    except Exception:
        if temp_csv.exists():
            try:
                temp_csv.unlink()
            except OSError:
                pass
        raise
    return (csv_path, True)


def build_provenance_meta(
    dataset_version: str,
    pipeline_version: str,
    source_sha256: str,
    created_at_utc: str,
    *,
    output_paths: list[str] | None = None,
    row_counts: dict[str, int] | None = None,
    schema_summary: dict[str, str] | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    """
    Build compact sidecar meta dict. Include output_paths, row_counts, schema_summary (columns + dtypes).
    """
    meta: dict[str, Any] = {
        "created_at_utc": created_at_utc,
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "source_sha256": source_sha256,
    }
    if output_paths is not None:
        meta["output_paths"] = output_paths
    if row_counts is not None:
        meta["row_counts"] = row_counts
    if schema_summary is not None:
        meta["schema_summary"] = schema_summary
    if warning:
        meta["warning"] = warning
    return meta


def _schema_summary_from_df(df: pd.DataFrame) -> dict[str, str]:
    """Compact columns + dtypes for meta."""
    if df.empty:
        return {}
    return {str(c): str(df.dtypes[c]) for c in df.columns}


def persist_raw_outputs(
    df_clean: pd.DataFrame,
    df_rejects: pd.DataFrame,
    ingest_report: dict[str, Any],
    version_manifest: dict[str, Any],
    *,
    raw_dir: Path | None = None,
    skip_clean: bool = False,
) -> None:
    """
    Ensure raw/ exists; write data_raw.parquet (unless skip_clean), data_raw_rejects.parquet,
    data_raw.meta.json, ingest_report.json atomically. Meta always includes dataset_version and provenance.
    When skip_clean=True, only rejects parquet + meta + ingest_report are written; data_raw.parquet is not written.
    """
    root = Path(raw_dir) if raw_dir is not None else Path(RAW_DIR_NAME)
    root.mkdir(parents=True, exist_ok=True)

    path_parquet = root / DATA_RAW_PARQUET
    path_rejects = root / DATA_RAW_REJECTS_PARQUET
    path_meta = root / DATA_RAW_META_JSON
    path_ingest_report = root / INGEST_REPORT_JSON

    if skip_clean:
        path_clean = None
        fallback_clean = False
        output_paths = []
        row_counts = {"data_raw": 0, "data_raw_rejects": len(df_rejects)}
        schema_summary = {}
    else:
        path_clean, fallback_clean = atomic_write_parquet(df_clean, path_parquet)
        output_paths = [str(path_clean)]

    path_rej, fallback_rej = atomic_write_parquet(df_rejects, path_rejects)
    output_paths.append(str(path_rej))
    if not skip_clean:
        row_counts = {"data_raw": len(df_clean), "data_raw_rejects": len(df_rejects)}
        schema_summary = _schema_summary_from_df(df_clean)

    dataset_version = version_manifest.get("dataset_version", "")
    pipeline_version = version_manifest.get("pipeline_version", "")
    source_sha256 = version_manifest.get("source_sha256", "")
    created_at_utc = version_manifest.get("created_at") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    warning = None
    if fallback_clean or fallback_rej:
        warning = "Parquet engine unavailable; one or more outputs written as CSV."

    meta = build_provenance_meta(
        dataset_version,
        pipeline_version,
        source_sha256,
        created_at_utc,
        output_paths=output_paths,
        row_counts=row_counts,
        schema_summary=schema_summary,
        warning=warning,
    )
    atomic_write_json(meta, path_meta)
    atomic_write_json(ingest_report, path_ingest_report)
