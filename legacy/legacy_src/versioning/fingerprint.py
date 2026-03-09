"""
Deterministic, portable dataset fingerprint for a single XLSX workbook.
Uses raw bytes only. See docs/dataset_version.md for policy.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .git_info import get_git_sha

DEFAULT_XLSX_PATH = Path("data/input/source.xlsx")


def resolve_pipeline_version() -> str:
    """
    1) env PIPELINE_VERSION if set and non-empty
    2) get_git_sha() if available
    3) "unknown"
    """
    env = os.environ.get("PIPELINE_VERSION", "").strip()
    if env:
        return env
    sha = get_git_sha()
    if sha is not None:
        return sha
    return "unknown"


def read_workbook_bytes(path: str | Path) -> bytes:
    """Read workbook raw bytes. Raises FileNotFoundError if file missing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workbook not found: {p.resolve()!s}")
    return p.read_bytes()


def compute_dataset_version(excel_bytes: bytes, pipeline_version: str) -> str:
    """dataset_version = sha256(excel_bytes + b'\\n' + pipeline_version.encode('utf-8')).hexdigest()"""
    payload = excel_bytes + b"\n" + pipeline_version.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_dataset_version(path: str | Path = DEFAULT_XLSX_PATH) -> dict[str, Any]:
    """
    Full fingerprint dict. Fails with clear error if workbook missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Workbook not found: {p.resolve()!s}")

    excel_bytes = p.read_bytes()
    pipeline_version = resolve_pipeline_version()
    dataset_version = compute_dataset_version(excel_bytes, pipeline_version)
    workbook_sha256 = hashlib.sha256(excel_bytes).hexdigest()

    return {
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "workbook_path": str(p.resolve()),
        "workbook_size_bytes": len(excel_bytes),
        "workbook_sha256": workbook_sha256,
    }
