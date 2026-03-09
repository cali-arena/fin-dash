"""
Per-table meta.json utilities: stable dataset_version hashing and schema hashing.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.agg_policy_contract import AggPolicy, policy_hash


def get_pipeline_version(root: Path | None = None) -> str:
    """
    PIPELINE_VERSION from env, else git rev-parse HEAD (short), else "unknown".
    root: repo root for git (default cwd).
    """
    v = os.environ.get("PIPELINE_VERSION", "").strip()
    if v:
        return v
    root = root or Path.cwd()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def load_source_metrics_version(root: Path, source_table: str = "curated/metrics_monthly.parquet") -> str:
    """
    Prefer curated/metrics_monthly.meta.json (key dataset_version or version),
    else data/.version.json dataset_version. Returns empty string if none found.
    """
    root = Path(root)
    # meta next to source table: e.g. curated/metrics_monthly.meta.json
    meta_path = root / Path(source_table).with_suffix(".meta.json")
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            v = (data.get("dataset_version") or data.get("version") or "").strip()
            if v:
                return v
        except Exception:
            pass
    version_path = root / "data" / ".version.json"
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
            return (data.get("dataset_version") or "").strip()
        except Exception:
            pass
    return ""


def hash_policy(policy: AggPolicy) -> str:
    """Stable SHA-256 of normalized agg policy JSON (sorted keys)."""
    return policy_hash(policy)


def hash_schema(df: pd.DataFrame) -> str:
    """SHA-1 of column_name + dtype pairs in column order (not sorted)."""
    key = "|".join(f"{c}:{df.dtypes[c]}" for c in df.columns)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def compute_agg_dataset_version(
    source_metrics_version: str,
    policy_hash_val: str,
    pipeline_version: str,
) -> str:
    """SHA-1 of (source_metrics_version + policy_hash + pipeline_version) for stable dataset_version."""
    payload = f"{source_metrics_version}{policy_hash_val}{pipeline_version}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def write_meta(path_meta: str | Path, meta: dict[str, Any]) -> None:
    """Write meta dict to JSON atomically (.tmp then replace, with fsync)."""
    path_meta = Path(path_meta)
    path_meta.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_meta.with_suffix(path_meta.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        try:
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
        os.replace(tmp, path_meta)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
