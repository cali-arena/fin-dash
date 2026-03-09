"""
Single entrypoint: generate and persist data/.version.json from source workbook.
Uses raw bytes only (no xlsx parsing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .fingerprint import compute_dataset_version, read_workbook_bytes, resolve_pipeline_version
from .git_info import is_git_dirty
from .version_manifest import atomic_write_json, build_manifest, load_manifest

DEFAULT_SOURCE = "data/input/source.xlsx"
DEFAULT_VERSION_PATH = "data/.version.json"


def _source_file_relative(source_path: str | Path, version_path: str | Path) -> str:
    """Normalize source_path to a relative posix path (no absolute in manifest)."""
    vp = Path(version_path).resolve()
    # project root = parent of directory containing .version.json (e.g. data/)
    project_root = vp.parent.parent
    sp = Path(source_path).resolve()
    try:
        rel = sp.relative_to(project_root)
    except ValueError:
        rel = sp
    return rel.as_posix()


def ensure_version_file(
    source_path: str | Path = DEFAULT_SOURCE,
    version_path: str | Path = DEFAULT_VERSION_PATH,
) -> dict[str, Any]:
    """
    Read workbook bytes, compute fingerprint, build manifest.
    If existing manifest has same dataset_version, pipeline_version, and source_sha256, return it.
    Else write atomically and return new manifest.
    """
    excel_bytes = read_workbook_bytes(source_path)
    pipeline_version = resolve_pipeline_version()
    dataset_version = compute_dataset_version(excel_bytes, pipeline_version)
    git_dirty = is_git_dirty()

    source_file_rel = _source_file_relative(source_path, version_path)
    manifest = build_manifest(
        source_file_rel,
        pipeline_version,
        dataset_version,
        excel_bytes,
        git_dirty=git_dirty,
    )

    existing = load_manifest(version_path)
    if existing is not None:
        if (
            existing.get("dataset_version") == manifest["dataset_version"]
            and existing.get("pipeline_version") == manifest["pipeline_version"]
            and existing.get("source_sha256") == manifest["source_sha256"]
        ):
            return existing

    atomic_write_json(manifest, version_path)
    return manifest
