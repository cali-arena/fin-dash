"""
Audit-grade version manifest with atomic writes.
Schema is fixed; JSON uses sort_keys=True, indent=2, ensure_ascii=False.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def build_manifest(
    source_path: str | Path,
    pipeline_version: str,
    dataset_version: str,
    source_bytes: bytes,
    *,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    """
    Build manifest dict with required schema.
    source_file is stored as normalized string path (e.g. relative).
    git_dirty: optional; pass from is_git_dirty() or None when not determined.
    """
    source_file = str(Path(source_path).as_posix())
    source_size_bytes = len(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "created_at": created_at,
        "dataset_version": dataset_version,
        "git_dirty": git_dirty,
        "pipeline_version": pipeline_version,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "source_size_bytes": source_size_bytes,
    }


def atomic_write_json(obj: dict[str, Any], path: str | Path) -> None:
    """
    Write JSON atomically: temp file in same dir, fsync, then os.replace.
    Temp name: <path>.tmp.<pid>.<random>
    Parent directory is created if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.{random.randint(0, 2**31 - 1)}"
    temp_path = p.parent / (p.name + suffix)
    try:
        content = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
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


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    """Load manifest from path. Return None if file does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Manifest is not a JSON object: {type(data)}")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid manifest JSON at {p}: {e}") from e
    except OSError as e:
        raise ValueError(f"Cannot read manifest at {p}: {e}") from e
