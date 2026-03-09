"""
Schema hash utilities for curated tables.
Deterministic: hashes raw file bytes; dataframe schema signature ordered by column name.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def load_text(path: str | Path) -> str:
    """Load file contents as UTF-8 text."""
    p = Path(path)
    return p.read_text(encoding="utf-8")


def schema_hash(path: str | Path) -> str:
    """
    Compute deterministic SHA-256 hash of raw schema file bytes.
    No parsing; the hash is over the exact bytes on disk.
    """
    p = Path(path)
    data = p.read_bytes()
    return hashlib.sha256(data).hexdigest()


def dataframe_schema_signature(df: pd.DataFrame) -> dict[str, str]:
    """
    Deterministic schema signature for a DataFrame:
    {column_name: dtype_as_str}, ordered by column name.
    """
    cols = sorted(str(c) for c in df.columns)
    return {col: str(df[col].dtype) for col in cols}

