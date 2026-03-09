"""
Deterministic slice key generation from configs/drill_paths.yml.
Builds slices_index: one row per (path_id, key combination) with slice_id and human slice_key.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

MAX_SLICES_GUARDRAIL = 200_000
DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
CURATED_DIR = "curated"
SLICES_INDEX_NAME = "slices_index"


def normalize_slice_value(v: Any) -> str:
    """
    Normalize a single value for slice key.
    - None/NaN -> "__NULL__"
    - strings: strip + collapse spaces (no uppercase)
    - numbers: stringify with no scientific notation
    """
    if v is None:
        return "__NULL__"
    if getattr(pd, "isna", lambda x: False)(v) if hasattr(v, "__float__") else False:
        try:
            if pd.isna(v):
                return "__NULL__"
        except (TypeError, ValueError):
            pass
    if isinstance(v, str):
        s = v.strip()
        s = re.sub(r"\s+", " ", s)
        return s
    if isinstance(v, (int, float)) or (
        hasattr(v, "dtype")
        and getattr(v.dtype, "kind", None) in ("i", "u", "f")
    ):
        try:
            if pd.isna(v):
                return "__NULL__"
        except (TypeError, ValueError):
            pass
        if isinstance(v, (int, pd.Int64Dtype().type if hasattr(pd, "Int64Dtype") else int)) or (
            hasattr(v, "dtype") and getattr(v.dtype, "kind", None) in ("i", "u")
        ):
            return str(int(v))
        # float: avoid scientific notation
        return format(float(v), "f").rstrip("0").rstrip(".") or "0"
    return str(v).strip()


def compute_slice_id(path_id: str, key_values: dict[str, str]) -> str:
    """Stable sha1 of path_id + '|' + sorted 'k=v' pairs."""
    parts = [path_id]
    for k in sorted(key_values.keys()):
        parts.append(f"{k}={key_values[k]}")
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _slice_key_human(path_id: str, key_values: dict[str, str]) -> str:
    """Human-readable slice_key: global -> 'global'; else 'k1=v1 | k2=v2' sorted by key."""
    if not key_values:
        return path_id if path_id == "global" else path_id
    parts = [f"{k}={key_values[k]}" for k in sorted(key_values.keys())]
    return " | ".join(parts)


def build_slices_index(
    fact_df: pd.DataFrame,
    drill_paths_config: dict[str, Any],
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Build slices index from fact and drill_paths config (enabled paths only).
    Output columns: path_id, path_label, keys, slice_id, slice_key, row_count.
    Deterministic sort: path_id, then slice_key (lexicographic).
    Raises if slice_id not unique, global row_count <= 0, or total slices > MAX_SLICES_GUARDRAIL.
    """
    from legacy.legacy_pipelines.contracts.drill_paths_contract import validate_drill_paths

    validate_drill_paths(drill_paths_config, list(fact_df.columns))
    paths = [
        p
        for p in drill_paths_config.get("drill_paths", [])
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    if not paths:
        raise ValueError("No enabled drill_paths in config")

    rows: list[dict[str, Any]] = []
    for p in paths:
        path_id = str(p.get("id", ""))
        path_label = str(p.get("label", ""))
        keys = list(p.get("keys") or [])
        keys_str = ",".join(keys)

        if not keys:
            # global: one slice, row_count = len(fact_df)
            n = len(fact_df)
            if n <= 0:
                raise ValueError("slices_index: global path must have row_count > 0")
            key_values: dict[str, str] = {}
            slice_id = compute_slice_id(path_id, key_values)
            slice_key = "global" if path_id == "global" else path_id
            rows.append({
                "path_id": path_id,
                "path_label": path_label,
                "keys": keys_str,
                "slice_id": slice_id,
                "slice_key": slice_key,
                "row_count": n,
            })
            continue

        # Ensure all keys exist in fact
        missing = [k for k in keys if k not in fact_df.columns]
        if missing:
            raise ValueError(f"slices_index: path id='{path_id}' keys missing in fact: {missing}")

        # Groupby keys, normalize values, build slice_id and slice_key
        g = fact_df.groupby(keys, dropna=False, sort=False)
        for name, grp in g:
            if len(keys) == 1:
                name = (name,) if not isinstance(name, tuple) else name
            key_values = {}
            for i, k in enumerate(keys):
                val = name[i] if isinstance(name, tuple) else name
                key_values[k] = normalize_slice_value(val)
            slice_id = compute_slice_id(path_id, key_values)
            slice_key = _slice_key_human(path_id, key_values)
            rows.append({
                "path_id": path_id,
                "path_label": path_label,
                "keys": keys_str,
                "slice_id": slice_id,
                "slice_key": slice_key,
                "row_count": len(grp),
            })

    df = pd.DataFrame(rows)
    if len(df) > MAX_SLICES_GUARDRAIL:
        raise ValueError(
            f"slices_index: total slices {len(df)} exceeds guardrail {MAX_SLICES_GUARDRAIL}"
        )
    if not df["slice_id"].is_unique:
        dup = df["slice_id"][df["slice_id"].duplicated(keep=False)].unique().tolist()
        raise ValueError(f"slices_index: slice_id must be unique; duplicates: {dup[:10]}")

    df = df.sort_values(["path_id", "slice_key"], kind="mergesort").reset_index(drop=True)
    return df


def _write_slices_index_and_meta(
    df: pd.DataFrame,
    fact_df: pd.DataFrame,
    curated_dir: Path,
    code_version_string: str,
) -> None:
    """Write curated/slices_index.parquet and curated/slices_index.meta.json (atomic)."""
    # Fact hashes for dataset_version (like other dims)
    cols = sorted(fact_df.columns)
    schema_parts = [f"{c}:{str(fact_df[c].dtype)}" for c in cols]
    fact_schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    sample = fact_df.head(1000)
    buf = io.StringIO()
    sample.to_csv(buf, index=False, lineterminator="\n")
    fact_data_hash_sample = hashlib.sha1(buf.getvalue().encode("utf-8")).hexdigest()
    payload = f"{fact_schema_hash}{fact_data_hash_sample}{code_version_string}"
    dataset_version = hashlib.sha1(payload.encode("utf-8")).hexdigest()

    row_count = len(df)
    out_cols = list(df.columns)
    schema_parts_out = [f"{c}:{str(df[c].dtype)}" for c in sorted(out_cols)]
    schema_hash = hashlib.sha1("|".join(schema_parts_out).encode("utf-8")).hexdigest()
    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "name": SLICES_INDEX_NAME,
        "dataset_version": dataset_version,
        "row_count": row_count,
        "schema_hash": schema_hash,
        "created_at_utc": created_at_utc,
    }

    curated_dir.mkdir(parents=True, exist_ok=True)
    pq_path = curated_dir / f"{SLICES_INDEX_NAME}.parquet"
    meta_path = curated_dir / f"{SLICES_INDEX_NAME}.meta.json"

    fd_pq, tmp_pq = tempfile.mkstemp(suffix=".parquet", dir=curated_dir, prefix="slices_")
    try:
        os.close(fd_pq)
        df.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, pq_path)
    except Exception:
        if Path(tmp_pq).exists():
            Path(tmp_pq).unlink(missing_ok=True)
        raise

    fd_meta, tmp_meta = tempfile.mkstemp(suffix=".meta.json", dir=curated_dir, prefix="slices_")
    try:
        os.close(fd_meta)
        Path(tmp_meta).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    except Exception:
        if Path(tmp_meta).exists():
            Path(tmp_meta).unlink(missing_ok=True)
        raise

    logger.info("Wrote %s (%d rows) and %s", pq_path, row_count, meta_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build slices index from fact + drill_paths config.")
    parser.add_argument("--build", action="store_true", help="Build and write curated/slices_index.parquet + meta.json")
    parser.add_argument("--config", default=DEFAULT_DRILL_PATHS_CONFIG, help="Path to drill_paths.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = Path(args.root) if args.root is not None else Path.cwd()
    if not args.build:
        logger.info("Use --build to build slices index.")
        return 0

    from legacy.legacy_pipelines.contracts.drill_paths_contract import load_drill_paths_config

    config_path = root / args.config
    config = load_drill_paths_config(config_path)
    fact_path = root / config.get("fact_table", "curated/fact_monthly.parquet")
    if not fact_path.exists():
        logger.error("Fact not found: %s", fact_path)
        return 1
    fact_df = pd.read_parquet(fact_path)

    try:
        index_df = build_slices_index(fact_df, config, root=root)
        code_version_string = os.environ.get("PIPELINE_VERSION", "dev")
        _write_slices_index_and_meta(
            index_df,
            fact_df,
            root / CURATED_DIR,
            code_version_string,
        )
    except ValueError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
