"""
Dataset-version-scoped cache path utilities.
All cached artifacts must live under data/cache/{dataset_version}/.
Stdlib only; typed.
"""
from __future__ import annotations

import re
from pathlib import Path

CACHE_ROOT = Path("data/cache")


def cache_root(dataset_version: str) -> Path:
    """Cache namespace root for this dataset_version: data/cache/{dataset_version}/"""
    return CACHE_ROOT / dataset_version


def cache_path(dataset_version: str, *parts: str) -> Path:
    """
    Path under cache root: CACHE_ROOT / dataset_version / parts...
    Use ensure_dir(path.parent) before writing.

    Examples:
        cache_path(v, "extracts", "data_raw.parquet")
        cache_path(v, "pivots", "aum_by_channel.parquet")
    """
    return cache_root(dataset_version).joinpath(*parts)


def ensure_dir(p: Path) -> None:
    """Ensure directory exists (and parents). Idempotent."""
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(s: str) -> str:
    """Replace unsafe filename characters with '_'. Use for segment names if needed."""
    # Unsafe on common filesystems: / \ : * ? " < > | and control chars
    unsafe = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
    return unsafe.sub("_", s)
