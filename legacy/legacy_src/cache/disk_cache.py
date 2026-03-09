"""
Disk cache: load or compute DataFrame, persist under dataset_version namespace.
Parquet preferred; CSV fallback if parquet engine missing. Atomic writes.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable

import pandas as pd

from .cache_paths import ensure_dir

_PARQUET_AVAILABLE: bool | None = None


def _parquet_available() -> bool:
    global _PARQUET_AVAILABLE
    if _PARQUET_AVAILABLE is not None:
        return _PARQUET_AVAILABLE
    try:
        import io
        df = pd.DataFrame({"a": [1]})
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        pd.read_parquet(buf)
        _PARQUET_AVAILABLE = True
    except Exception:
        _PARQUET_AVAILABLE = False
    return _PARQUET_AVAILABLE


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_frame_atomic(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp = path.parent / (path.name + suffix)
    try:
        if path.suffix.lower() == ".parquet":
            df.to_parquet(temp, index=False)
        else:
            df.to_csv(temp, index=False)
        os.replace(temp, path)
    except Exception:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
        raise


def load_or_compute_parquet(
    path: Path,
    compute_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """
    If path exists: read (parquet or csv by suffix) and return.
    Else: compute via compute_fn(), write atomically (temp -> replace), return.
    Ensures parent dirs exist before write. Uses CSV fallback if parquet engine missing.
    """
    path = Path(path)
    use_parquet = _parquet_available()
    read_path = path if path.exists() else path.with_suffix(".csv") if path.with_suffix(".csv").exists() else None
    if read_path is not None:
        return _read_frame(read_path)

    df = compute_fn()
    write_path = path if use_parquet else path.with_suffix(".csv")
    _write_frame_atomic(df, write_path)
    return df
