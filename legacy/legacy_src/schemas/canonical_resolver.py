"""
Canonical column resolver: map messy Excel/CSV headers to canonical names (snake_case).
Uses schemas/canonical_columns.yml. Matching is case-insensitive and whitespace-normalized.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def normalize_name(s: str) -> str:
    """Strip, collapse internal whitespace to single space, lower-case."""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def load_canonical_columns(path: str | Path = "schemas/canonical_columns.yml") -> dict[str, Any]:
    """Load canonical column schema from YAML. Raises on missing file or invalid structure."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Canonical schema not found: {p.resolve()}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Canonical schema root must be a mapping, got {type(data)}")
    if "columns" not in data:
        raise ValueError("Canonical schema missing key: 'columns'")
    if not isinstance(data["columns"], list):
        raise ValueError(f"Canonical schema 'columns' must be a list, got {type(data['columns'])}")
    return data


def build_alias_index(canonical_schema: dict[str, Any]) -> dict[str, str]:
    """
    Build normalized_alias -> canonical_name. If two canonicals share the same
    normalized alias (collision), raise ValueError with a clear message.
    """
    index: dict[str, str] = {}
    for col in canonical_schema.get("columns") or []:
        if not isinstance(col, dict):
            continue
        canonical = col.get("canonical_name")
        aliases = col.get("accepted_source_names") or []
        if not canonical:
            continue
        for alias in aliases:
            norm = normalize_name(alias)
            if norm in index and index[norm] != canonical:
                raise ValueError(
                    f"Canonical alias collision: normalized alias {norm!r} maps to both "
                    f"{index[norm]!r} and {canonical!r}"
                )
            index[norm] = canonical
    return index


def resolve_headers_to_canonical(
    headers: list[str],
    canonical_schema: dict[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """
    Map original headers to canonical names via normalized alias matching.
    Returns (mapping_original_to_canonical, unmatched_headers). Matching is
    case-insensitive and whitespace-normalized.
    """
    index = build_alias_index(canonical_schema)
    mapping_original_to_canonical: dict[str, str] = {}
    unmatched_headers: list[str] = []
    for h in headers:
        norm = normalize_name(h)
        if norm in index:
            mapping_original_to_canonical[h] = index[norm]
        else:
            unmatched_headers.append(h)
    return mapping_original_to_canonical, unmatched_headers


def required_canonicals(canonical_schema: dict[str, Any]) -> list[str]:
    """Return list of canonical_name for columns with required=true, in schema order."""
    out: list[str] = []
    for col in canonical_schema.get("columns") or []:
        if isinstance(col, dict) and col.get("required") is True and col.get("canonical_name"):
            out.append(col["canonical_name"])
    return out


def validate_canonical_presence(
    df: pd.DataFrame,
    mapping_original_to_canonical: dict[str, str],
    canonical_schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Determine which required canonical columns are present in df via the mapping.
    Returns (ok, missing) where missing is the list of required canonical_name not present.
    """
    required = required_canonicals(canonical_schema)
    # Canonical names we have: values of mapping that correspond to df columns (original headers in df)
    present_canonicals = set()
    for orig, canon in mapping_original_to_canonical.items():
        if orig in df.columns:
            present_canonicals.add(canon)
    missing = [c for c in required if c not in present_canonicals]
    return (len(missing) == 0, missing)
