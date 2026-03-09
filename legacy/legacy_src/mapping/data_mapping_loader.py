"""
Strict loader for DATA_MAPPING: alias resolution from schema, value normalization, composite key uniqueness.
No guessing: only resolve via schema accepted_source_names.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from e

from legacy.legacy_src.transform.standardize_columns import make_match_key, normalize_header


def _load_schema(schema_path: str | Path) -> dict[str, Any]:
    p = Path(schema_path)
    if not p.is_file():
        raise FileNotFoundError(f"Schema not found: {p.resolve()}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Schema root must be a mapping, got {type(data)}")
    return data


def _is_empty_or_unnamed(name: str) -> bool:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return True
    s = str(name).strip()
    return s == "" or bool(re.match(r"^Unnamed\s*:\s*\d+$", s, re.IGNORECASE))


def _resolve_headers_to_canonical(
    df_columns: list[str],
    accepted_source_names: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Resolve source column names to canonical using schema accepted_source_names.
    Returns (rename_map: source -> canonical, canonical_to_source).
    First match wins; two source columns -> same canonical = collision.
    """
    # alias_index: match_key(alias) -> canonical
    alias_index: dict[str, str] = {}
    for canonical, aliases in (accepted_source_names or {}).items():
        for alias in aliases:
            key = make_match_key(alias)
            if key in alias_index and alias_index[key] != canonical:
                raise ValueError(
                    f"Schema alias collision: match_key {key!r} maps to both {alias_index[key]!r} and {canonical!r}"
                )
            alias_index[key] = canonical

    rename_map: dict[str, str] = {}
    canonical_to_source: dict[str, str] = {}
    for orig in df_columns:
        key = make_match_key(str(orig))
        if key in alias_index:
            canon = alias_index[key]
            if canon in canonical_to_source and canonical_to_source[canon] != orig:
                raise ValueError(
                    f"Duplicate source columns for canonical {canon!r}: "
                    f"both {canonical_to_source[canon]!r} and {orig!r} resolve to it."
                )
            canonical_to_source[canon] = orig
            rename_map[orig] = canon
        else:
            rename_map[orig] = orig  # keep unmapped as-is for extra_columns reporting

    return rename_map, canonical_to_source


def _normalize_cell(value: str, trim: bool, collapse: bool, nfkc: bool, empty_as_null: bool) -> str | pd.NA:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    s = str(value)
    if nfkc:
        s = unicodedata.normalize("NFKC", s)
    if trim:
        s = s.strip()
    if collapse:
        s = re.sub(r"\s+", " ", s)
    if empty_as_null and s == "":
        return pd.NA
    return s


def _normalize_values(
    df: pd.DataFrame,
    rules: dict[str, Any],
) -> pd.DataFrame:
    trim = rules.get("trim_whitespace", True)
    collapse = rules.get("collapse_internal_whitespace", True)
    nfkc = (rules.get("unicode_normalize") or "").upper() == "NFKC"
    empty_as_null = rules.get("empty_string_as_null", True)
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(
            lambda v: _normalize_cell(v, trim, collapse, nfkc, empty_as_null)
        )
    return out


def _enforce_string_dtype(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].astype("string")
    return out


def load_data_mapping(
    path: str | Path = "data/input/DATA_MAPPING.csv",
    schema_path: str | Path = "schemas/data_mapping.schema.yml",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load DATA_MAPPING CSV with schema-driven alias resolution, value normalization, and key uniqueness.
    Returns (df_mapping_canonical, report).
    """
    path = Path(path)
    schema_path = Path(schema_path)
    schema = _load_schema(schema_path)

    required = schema.get("required_columns") or []
    key_def = schema.get("key_definition") or {}
    key_cols = key_def.get("key") or required
    optional = schema.get("optional_enrichment_columns") or []
    accepted = schema.get("accepted_source_names") or {}
    norm_rules = schema.get("normalization_rules") or {}
    uniq_policy = schema.get("uniqueness_policy") or {}
    mode = (uniq_policy.get("mode") or "hard_fail").strip().lower()
    log_duplicates = uniq_policy.get("log_duplicates", True)

    # 1) Read CSV
    if not path.is_file():
        raise FileNotFoundError(f"DATA_MAPPING file not found: {path.resolve()}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows_in = len(df)

    # 2) Empty/unnamed headers -> placeholders
    cols = [str(c).strip() for c in df.columns]
    if all(_is_empty_or_unnamed(c) or c == "" for c in cols):
        df.columns = [f"col{i + 1}" for i in range(len(df.columns))]

    # 3) Resolve headers -> canonical
    rename_map, canonical_to_source = _resolve_headers_to_canonical(list(df.columns), accepted)
    df = df.rename(columns=rename_map)

    # 4) Drop columns that did not resolve to a canonical (keep only canonical-named columns)
    all_canonical = set(required) | set(optional) | set(accepted.keys())
    df = df[[c for c in df.columns if c in all_canonical]].copy()

    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"DATA_MAPPING missing required columns after alias resolution: {missing_required}. "
            f"Required: {required}. Resolved: {list(df.columns)}."
        )

    extra_columns = [c for c in df.columns if c not in required and c not in optional]

    # 5) Normalize values
    df = _normalize_values(df, norm_rules)

    # 6) Enforce string dtype
    df = _enforce_string_dtype(df)

    # 7) Composite key uniqueness
    key_cols_present = [k for k in key_cols if k in df.columns]
    if len(key_cols_present) != len(key_cols):
        missing_key = [k for k in key_cols if k not in df.columns]
        raise ValueError(f"Key columns missing for uniqueness check: {missing_key}")

    duplicates = df.duplicated(subset=key_cols_present, keep=False)
    duplicate_count = int(duplicates.sum())
    sample_duplicate_keys: list[tuple[Any, ...]] = []

    if duplicate_count > 0:
        dup_df = df.loc[duplicates]
        sample_keys_df = dup_df[key_cols_present].drop_duplicates().head(10)
        sample_duplicate_keys = [tuple(row) for _, row in sample_keys_df.iterrows()]

        if mode == "hard_fail":
            sample_with_rows: list[dict[str, Any]] = []
            for i, key_tuple in enumerate(sample_duplicate_keys[:10]):
                mask = (dup_df[key_cols_present[0]] == key_tuple[0])
                for j, k in enumerate(key_cols_present[1:], 1):
                    mask = mask & (dup_df[k] == key_tuple[j])
                sample_with_rows.append({
                    "key": key_tuple,
                    "rows": dup_df.loc[mask].to_dict("records"),
                })
            msg = (
                f"DATA_MAPPING has {duplicate_count} duplicate row(s) by key {key_cols_present}. "
                f"Uniqueness policy is hard_fail. Sample duplicate keys (first 10) with rows: {sample_with_rows}."
            )
            raise ValueError(msg)

        if mode == "dedupe_keep_first":
            df = df.drop_duplicates(subset=key_cols_present, keep="first").reset_index(drop=True)
        elif mode == "dedupe_prefer_non_null":
            enrich_cols = [c for c in (optional or []) if c in df.columns]
            def pick_best(group: pd.DataFrame) -> pd.DataFrame:
                if len(group) == 1:
                    return group
                if not enrich_cols:
                    return group.head(1)
                non_null_count = group[enrich_cols].notna().sum(axis=1)
                best_idx = non_null_count.idxmax()
                return group.loc[[best_idx]]
            df = df.groupby(key_cols_present, as_index=False, dropna=False).apply(
                pick_best
            ).reset_index(drop=True)
        else:
            df = df.drop_duplicates(subset=key_cols_present, keep="first").reset_index(drop=True)

    # 8) Deterministic ordering
    df = df.sort_values(key_cols_present).reset_index(drop=True)
    rows_out = len(df)

    report = build_mapping_report(
        rows_in=rows_in,
        rows_out=rows_out,
        missing_required=missing_required,
        extra_columns=extra_columns,
        duplicates_count=duplicate_count,
        dedupe_mode=mode if duplicate_count > 0 else None,
        sample_duplicate_keys=sample_duplicate_keys if duplicate_count > 0 else None,
        log_duplicates=log_duplicates,
    )
    return df, report


def build_mapping_report(
    *,
    rows_in: int,
    rows_out: int,
    missing_required: list[str],
    extra_columns: list[str],
    duplicates_count: int,
    dedupe_mode: str | None = None,
    sample_duplicate_keys: list[tuple[Any, ...]] | None = None,
    log_duplicates: bool = True,
) -> dict[str, Any]:
    """Build report dict for DATA_MAPPING load."""
    report: dict[str, Any] = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "missing_required": list(missing_required),
        "extra_columns": list(extra_columns),
        "duplicates_count": duplicates_count,
        "dedupe_mode": dedupe_mode,
        "sample_duplicate_keys": list(sample_duplicate_keys) if sample_duplicate_keys else [],
        "log_duplicates": log_duplicates,
    }
    return report
