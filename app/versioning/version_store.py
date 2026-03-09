"""
Load/save version manifest and ensure it is up to date.
Persists to data/.version.json. Raw bytes only for fingerprinting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.versioning.dataset_fingerprint import build_version_manifest


def short_id(s: str, n: int = 8) -> str:
    """First n characters of s; or full string if len(s) <= n."""
    return s[:n] if s else ""


def load_version_manifest(path: str = "data/.version.json") -> dict[str, Any] | None:
    """Load manifest from path. Returns None if file missing or invalid."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_version_manifest(manifest: dict[str, Any], path: str = "data/.version.json") -> None:
    """Write manifest to path. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_version_manifest(
    base_dir: str = "data/input",
    path: str = "data/.version.json",
) -> dict[str, Any]:
    """
    Compute current manifest; if .version.json exists and dataset_version matches, return it.
    Otherwise overwrite with new manifest and return it.
    Fails fast if required inputs are missing (build_version_manifest raises).
    """
    current = build_version_manifest(base_dir)
    existing = load_version_manifest(path)
    if (
        existing is not None
        and existing.get("dataset_version") == current["dataset_version"]
    ):
        return existing
    save_version_manifest(current, path)
    return current
