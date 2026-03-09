"""
Persist curated fact_monthly.parquet with companion metadata. Atomic writes; validation before write.
"""
from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CURATED_DIR = "curated"
FACT_MONTHLY_PARQUET = "fact_monthly.parquet"
FACT_MONTHLY_META_JSON = "fact_monthly.meta.json"
COL_MONTH_END = "month_end"


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
    Write DataFrame to path atomically (temp -> os.replace).
    Fallback to CSV if parquet engine missing (writes .csv; caller should record in meta).
    Returns (path_written, used_csv_fallback).
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

    # Fallback: CSV at same stem with .csv extension
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
        return (csv_path, True)
    except Exception:
        if temp_csv.exists():
            try:
                temp_csv.unlink()
            except OSError:
                pass
        raise


def build_fact_monthly_meta(
    *,
    dataset_version: str,
    df: pd.DataFrame,
    grain: list[str],
    schema_hash: str,
    schema_signature: dict[str, str],
    created_at_utc: str,
    validation_errors: list[str] | None = None,
    used_csv_fallback: bool = False,
    pipeline_version: str | None = None,
    mapping_stats: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    unmapped_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build deterministic meta dict for curated fact_monthly.
    Includes dataset_version, rowcount, grain, month_end_min/max (ISO), schema_hash, dtypes, created_at_utc.
    month_end must be present in df and datetime; optional validation_errors / used_csv_fallback.
    """
    meta: dict[str, Any] = {
        "created_at_utc": created_at_utc,
        "dataset_version": dataset_version,
        "grain": list(grain),
        "rowcount": len(df),
        "schema_hash": schema_hash,
        "dtypes": dict(schema_signature),
    }
    if pipeline_version is not None:
        meta["pipeline_version"] = pipeline_version
    if validation_errors is not None:
        meta["validation_errors"] = list(validation_errors)
    if used_csv_fallback:
        meta["parquet_fallback"] = "csv"

    if COL_MONTH_END in df.columns and len(df) > 0:
        ser = pd.to_datetime(df[COL_MONTH_END], utc=False)
        meta["month_end_min"] = ser.min().strftime("%Y-%m-%d")
        meta["month_end_max"] = ser.max().strftime("%Y-%m-%d")
    else:
        meta["month_end_min"] = None
        meta["month_end_max"] = None

    if mapping_stats is not None:
        meta["mapping_stats"] = mapping_stats
    if coverage is not None:
        meta["coverage"] = coverage
    if unmapped_gate is not None:
        meta["unmapped_gate"] = unmapped_gate

    return meta


def _atomic_write_meta(meta: dict[str, Any], path: Path) -> None:
    """Write meta JSON atomically (sort_keys, indent for deterministic output)."""
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


def persist_fact_monthly(
    df_fact: pd.DataFrame,
    *,
    dataset_version: str,
    pipeline_version: str,
    schema_path: str | Path = "schemas/fact_monthly.schema.yml",
    root: Path | None = None,
    mapping_stats: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate fact_monthly, then persist curated/fact_monthly.parquet and curated/fact_monthly.meta.json.

    - Runs validate_fact_monthly(...) before write.
    - If validation fails: does NOT write parquet; writes curated/fact_monthly.meta.json with
      dataset_version, created_at_utc, schema_hash, validation_errors; then raises ValueError.
    - If validation passes: writes parquet and meta atomically; returns meta.

    Output paths (relative to root): curated/fact_monthly.parquet, curated/fact_monthly.meta.json.
    """
    root = Path(root) if root is not None else Path.cwd()
    curated_dir = root / CURATED_DIR
    path_parquet = curated_dir / FACT_MONTHLY_PARQUET
    path_meta = curated_dir / FACT_MONTHLY_META_JSON

    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    schema_path = Path(schema_path)

    from legacy.legacy_src.quality.fact_monthly_gates import load_fact_schema, validate_fact_monthly
    from legacy.legacy_src.schemas.schema_hash import schema_hash as compute_schema_hash, dataframe_schema_signature

    schema = load_fact_schema(schema_path)
    sch_hash = compute_schema_hash(schema_path)
    schema_sig = dataframe_schema_signature(df_fact)
    grain = schema["grain"]

    ok, errors, stats = validate_fact_monthly(df_fact, schema, strict_dtypes=False)
    if not ok:
        failure_meta = build_fact_monthly_meta(
            dataset_version=dataset_version,
            df=df_fact,
            grain=grain,
            schema_hash=sch_hash,
            schema_signature=schema_sig,
            created_at_utc=created_at_utc,
            validation_errors=errors,
        )
        _atomic_write_meta(failure_meta, path_meta)
        raise ValueError("fact_monthly validation failed: " + "; ".join(errors)) from None

    path_written, used_csv_fallback = atomic_write_parquet(df_fact, path_parquet)
    meta = build_fact_monthly_meta(
        dataset_version=dataset_version,
        df=df_fact,
        grain=grain,
        schema_hash=sch_hash,
        schema_signature=stats.get("dtype_signature", schema_sig),
        created_at_utc=created_at_utc,
        used_csv_fallback=used_csv_fallback,
        pipeline_version=pipeline_version,
        mapping_stats=mapping_stats,
        coverage=coverage,
        unmapped_gate=unmapped_gate,
    )
    _atomic_write_meta(meta, path_meta)
    return meta
