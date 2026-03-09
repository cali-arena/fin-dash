"""
Deterministic column standardization + alias resolution transformer.
Maps messy headers to canonical names using schemas/canonical_columns.yml.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as e:  # pragma: no cover - environment-specific
    raise ImportError("PyYAML is required for standardize_columns. Install with: pip install pyyaml") from e


def strip_weird_chars(s: str) -> str:
    """
    Remove non-printable / weird characters from a header.

    - Drops:
      - \\u00A0 (non-breaking space)
      - \\u200B-\\u200D (zero-width)
      - all control chars (Unicode category Cc)
    - Keeps printable letters, numbers, punctuation, and regular spaces.
    """
    if s is None:
        s = ""
    out_chars: list[str] = []
    for ch in str(s):
        code = ord(ch)
        cat = unicodedata.category(ch)
        if cat == "Cc":
            continue
        if code == 0x00A0:  # NBSP (usually replaced earlier)
            continue
        if 0x200B <= code <= 0x200D:  # zero-width: treat as space to avoid token merge
            out_chars.append(" ")
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def normalize_header(s: str) -> tuple[str, str]:
    """
    Normalize a header for display and matching.

    Steps:
    1) Unicode normalize NFKC
    2) Replace NBSP with space
    3) strip_weird_chars
    4) Strip leading/trailing whitespace
    5) Collapse all whitespace runs to single space
    6) Lowercase for match key

    Examples:
    - " Asset Under Management " -> display: "Asset Under Management", key: "asset under management"
    - "net\\u00A0new\\u200Bbase fees " -> display: "net new base fees", key: "net new base fees"
    """
    if s is None:
        s = ""
    # 1) NFKC
    s_norm = unicodedata.normalize("NFKC", str(s))
    # 2) NBSP -> space
    s_norm = s_norm.replace("\u00A0", " ")
    # 3) strip weird chars
    s_clean = strip_weird_chars(s_norm)
    # 4) strip
    s_clean = s_clean.strip()
    # 5) collapse whitespace
    s_display = re.sub(r"\s+", " ", s_clean)
    # 6) lower for match key
    key = s_display.lower()
    return s_display, key


def make_match_key(s: str) -> str:
    """Return the normalized matching key for a header."""
    _, key = normalize_header(s)
    return key


def _load_canonical_schema(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Canonical schema not found at {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Canonical schema root must be a mapping, got {type(data)}")
    if "columns" not in data or not isinstance(data["columns"], list):
        raise ValueError("Canonical schema missing 'columns' list")
    return data


def standardize_columns(
    df: pd.DataFrame,
    canonical_schema_path: str | Path = "schemas/canonical_columns.yml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize column headers and resolve aliases to canonical names.

    Returns (df_out, rename_audit).
    """
    schema = _load_canonical_schema(canonical_schema_path)

    # 1) Normalize headers (for output + matching)
    normalized_headers: dict[str, str] = {}
    match_keys: dict[str, str] = {}
    for col in df.columns:
        key = str(col)
        norm_display, match_key = normalize_header(col)
        normalized_headers[key] = norm_display
        match_keys[key] = match_key

    # 2) Build alias index from schema and detect alias collisions
    alias_index: dict[str, str] = {}
    alias_collisions: list[dict[str, Any]] = []
    for col_def in schema.get("columns") or []:
        if not isinstance(col_def, dict):
            continue
        canon = col_def.get("canonical_name")
        aliases = col_def.get("accepted_source_names") or []
        if not canon:
            continue
        for alias in aliases:
            key = make_match_key(alias)
            if key in alias_index and alias_index[key] != canon:
                alias_collisions.append(
                    {
                        "alias": alias,
                        "normalized_alias": key,
                        "canonical_existing": alias_index[key],
                        "canonical_new": canon,
                    }
                )
            else:
                alias_index[key] = canon
    if alias_collisions:
        parts = [
            f"alias {c['alias']!r} ({c['normalized_alias']!r}) -> {c['canonical_existing']!r} vs {c['canonical_new']!r}"
            for c in alias_collisions
        ]
        msg = "Canonical alias collision(s) in schema: " + "; ".join(parts)
        raise ValueError(msg)

    # 3) Resolve each source column to at most one canonical
    resolved_to_canonical: dict[str, str] = {}
    unmatched_columns: list[str] = []
    for orig in df.columns:
        key = match_keys[str(orig)]
        if key in alias_index:
            resolved_to_canonical[str(orig)] = alias_index[key]
        else:
            unmatched_columns.append(str(orig))

    # 4) Collision blocking: two source columns -> same canonical
    canonical_to_source: dict[str, str] = {}
    collisions: list[dict[str, Any]] = []
    for orig, canon in resolved_to_canonical.items():
        if canon in canonical_to_source and canonical_to_source[canon] != orig:
            prev = canonical_to_source[canon]
            collisions.append(
                {
                    "canonical": canon,
                    "sources": sorted([prev, orig]),
                    "normalized_sources": {
                        prev: match_keys[prev],
                        orig: match_keys[orig],
                    },
                }
            )
        else:
            canonical_to_source[canon] = orig

    if collisions:
        parts = [
            f"canonical {c['canonical']!r} from sources {c['sources']} (normalized: {c['normalized_sources']})"
            for c in collisions
        ]
        msg = "Duplicate source columns for canonical: " + "; ".join(parts)
        raise ValueError(msg)

    # 5) Build final rename map: mapped -> canonical, others stay as original
    final_rename_map: dict[str, str] = {}
    for orig in df.columns:
        s_orig = str(orig)
        if s_orig in resolved_to_canonical:
            final_rename_map[s_orig] = resolved_to_canonical[s_orig]
        else:
            final_rename_map[s_orig] = s_orig

    df_out = df.rename(columns=final_rename_map)

    rename_audit: dict[str, Any] = {
        "normalized_headers": {k: normalized_headers[k] for k in sorted(normalized_headers)},
        "resolved_to_canonical": {k: resolved_to_canonical[k] for k in sorted(resolved_to_canonical)},
        "final_rename_map": {k: final_rename_map[k] for k in sorted(final_rename_map)},
        "collisions": collisions,
        "ignored_columns": [],
        "unmatched_columns": sorted(unmatched_columns),
        "canonical_to_source": {k: canonical_to_source[k] for k in sorted(canonical_to_source)},
        "applied_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    return df_out, rename_audit

