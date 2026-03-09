"""
Manifest helper for Streamlit: load manifest and resolve table paths only from manifest (no ad-hoc groupbys).
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"
AGG_DIR = "agg"


def load_manifest(root: Path | None = None) -> dict:
    """
    Load agg/manifest.json. Returns dict with dataset_version, policy_hash, tables (list of table entries).
    Raises FileNotFoundError if manifest does not exist.
    """
    root = root or Path.cwd()
    root = Path(root)
    path = root / AGG_DIR / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Agg manifest not found: {path}. Run materialize_aggs first.")
    return json.loads(path.read_text(encoding="utf-8"))


def get_table(name: str, root: Path | None = None) -> Path:
    """
    Return path to agg/<name>.parquet for a table that exists in the manifest.
    Streamlit should use only this to resolve tables (no dynamic groupbys).
    Raises KeyError if name is not in manifest tables.
    """
    root = root or Path.cwd()
    root = Path(root)
    manifest = load_manifest(root)
    tables = manifest.get("tables") or []
    for t in tables:
        if t.get("name") == name:
            rel = t.get("path") or f"{AGG_DIR}/{name}.parquet"
            return root / rel
    available = [t.get("name") for t in tables if t.get("name")]
    raise KeyError(f"Table {name!r} not in agg manifest. Available: {available!r}")
