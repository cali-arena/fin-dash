"""
Persist dataset fingerprint to a manifest file for reproducibility and caching.
Deterministic serialization (sorted keys). No pandas.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fingerprint import get_dataset_version

DEFAULT_MANIFEST_PATH = "data/.version.json"
DEFAULT_XLSX_PATH = "data/input/source.xlsx"


def load_manifest(path: str = DEFAULT_MANIFEST_PATH) -> dict[str, Any] | None:
    """Load manifest from path. Return None if file does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(manifest: dict[str, Any], path: str = DEFAULT_MANIFEST_PATH) -> None:
    """Write manifest as pretty JSON (sorted keys, indent=2). Ensure parent dir exists."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def ensure_manifest(
    xlsx_path: str = DEFAULT_XLSX_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Compute current fingerprint; if existing manifest has same dataset_version, return it.
    Else overwrite with new manifest (including created_at_utc ISO8601) and return it.
    Clear error if workbook missing (get_dataset_version raises).
    """
    current = get_dataset_version(xlsx_path)
    current["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    existing = load_manifest(manifest_path)
    if (
        existing is not None
        and existing.get("dataset_version") == current["dataset_version"]
    ):
        return existing

    save_manifest(current, manifest_path)
    return current
