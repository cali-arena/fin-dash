"""
Deterministic dataset fingerprint for caching and reproducibility.
Raw bytes only; no pandas. See docs/dataset_version.md.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Required input filenames in canonical order. Must match docs/data_contract.md.
REQUIRED_INPUT_FILES = [
    "DATA_RAW.csv",
    "DATA_SUMMARY.csv",
    "DATA_MAPPING.csv",
    "ETF.csv",
    "EXECUTIVE_SUMMARY.csv",
]


def get_pipeline_version() -> str:
    """
    Pipeline version string: git SHA if available, else PIPELINE_VERSION env, else "dev".
    Deterministic in CI/deploy when PIPELINE_VERSION is set or repo is present.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        if out.returncode == 0 and out.stdout and len(out.stdout.strip()) == 40:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    env = os.environ.get("PIPELINE_VERSION", "").strip()
    if env:
        return env
    return "dev"


def read_input_bytes(base_dir: str) -> bytes:
    """
    Read raw bytes of required CSV files in fixed order, concatenated with
    separators: file1_bytes + b"\\n---FILE:name---\\n" + file2_bytes + ...
    Raises if any required file is missing.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Input base directory does not exist: {base_dir!r}")

    chunks: list[bytes] = []
    missing: list[str] = []

    for i, filename in enumerate(REQUIRED_INPUT_FILES):
        path = base / filename
        if not path.is_file():
            missing.append(filename)
            continue
        raw = path.read_bytes()
        if i > 0:
            chunks.append(b"\n---FILE:" + filename.encode("utf-8") + b"---\n")
        chunks.append(raw)

    if missing:
        raise FileNotFoundError(
            f"Missing required input file(s) under {base_dir!r}: {missing}. "
            f"Required (in order): {REQUIRED_INPUT_FILES}"
        )

    return b"".join(chunks)


def compute_dataset_version(input_bytes: bytes, pipeline_version: str) -> str:
    """
    Deterministic 64-char hex fingerprint.
    dataset_version = sha256(input_bytes + b"\\n" + pipeline_version.encode("utf-8")).hexdigest()
    """
    payload = input_bytes + b"\n" + pipeline_version.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_version_manifest(base_dir: str) -> dict[str, Any]:
    """
    Build full version manifest: dataset_version, pipeline_version, per-file hashes/sizes, created_at_utc.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Input base directory does not exist: {base_dir!r}")

    pipeline_version = get_pipeline_version()
    inputs_meta: list[dict[str, Any]] = []
    chunks: list[bytes] = []

    for i, filename in enumerate(REQUIRED_INPUT_FILES):
        path = base / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required input file: {path}. Required (in order): {REQUIRED_INPUT_FILES}"
            )
        raw = path.read_bytes()
        inputs_meta.append({
            "name": filename,
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        })
        if i > 0:
            chunks.append(b"\n---FILE:" + filename.encode("utf-8") + b"---\n")
        chunks.append(raw)

    input_bytes = b"".join(chunks)
    dataset_version = compute_dataset_version(input_bytes, pipeline_version)

    return {
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "inputs": inputs_meta,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
