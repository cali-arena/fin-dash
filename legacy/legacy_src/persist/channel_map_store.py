"""
Persist channel_map for reproducibility: parquet + meta with rowcount and content hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.mapping.channel_map_builder import KEY_COLS, OUT_COLS

CURATED_DIR = "curated"
CHANNEL_MAP_PARQUET = "channel_map.parquet"
CHANNEL_MAP_META_JSON = "channel_map.meta.json"


def dataframe_content_hash(df: pd.DataFrame, key_cols: list[str]) -> str:
    """
    Deterministic content hash: sort by key_cols, select key_cols + output cols in fixed order,
    serialize to UTF-8 CSV with \\n line endings (no index), sha256.
    """
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    out_cols = [c for c in OUT_COLS if c in df.columns]
    cols = key_cols + out_cols
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe_content_hash: missing key columns {missing}")
    ordered = df.sort_values(key_cols)[cols]
    csv_bytes = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _parquet_available() -> bool:
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"_parquet_chk_{os.getpid()}.parquet"
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


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write DataFrame to path atomically (temp -> os.replace).
    Fallback to CSV if parquet engine missing (same basename with .csv).
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
            return
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    csv_path = p.with_suffix(".csv")
    temp_csv = csv_path.parent / (csv_path.name + suffix)
    df.to_csv(temp_csv, index=False)
    try:
        with open(temp_csv, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(temp_csv, csv_path)


def _atomic_write_json(obj: dict[str, Any], path: Path) -> None:
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


def persist_channel_map(
    df_channel_map: pd.DataFrame,
    *,
    dataset_version: str,
    schema_version: str = "2026-03-03",
    root: Path | None = None,
) -> dict[str, Any]:
    """
    Persist curated/channel_map.parquet and curated/channel_map.meta.json.
    Meta: dataset_version, schema_version, rowcount, content_hash, key_cols, out_cols, created_at_utc.
    Returns meta dict.
    """
    root = Path(root) if root is not None else Path.cwd()
    out_dir = root / CURATED_DIR
    path_parquet = out_dir / CHANNEL_MAP_PARQUET
    path_meta = out_dir / CHANNEL_MAP_META_JSON

    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rowcount = len(df_channel_map)
    content_hash = dataframe_content_hash(df_channel_map, KEY_COLS)

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df_channel_map, path_parquet)
    used_csv = not path_parquet.exists()

    meta: dict[str, Any] = {
        "dataset_version": dataset_version,
        "schema_version": schema_version,
        "rowcount": rowcount,
        "content_hash": content_hash,
        "key_cols": list(KEY_COLS),
        "out_cols": list(OUT_COLS),
        "created_at_utc": created_at_utc,
    }
    if used_csv:
        meta["parquet_fallback"] = "csv"

    _atomic_write_json(meta, path_meta)
    return meta
