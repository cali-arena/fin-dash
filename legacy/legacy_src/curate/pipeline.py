"""
Curate pipeline orchestration: build channel_map and fact_monthly with persistence.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_src.curate.fact_monthly import build_fact_monthly
from legacy.legacy_src.mapping.channel_map_builder import KEY_COLS, OUT_COLS, build_channel_map
from legacy.legacy_src.mapping.data_mapping_loader import load_data_mapping
from legacy.legacy_src.persist.channel_map_store import persist_channel_map
from legacy.legacy_src.persist.dim_store import persist_dim
from legacy.legacy_src.persist.fact_monthly_store import persist_fact_monthly
from legacy.legacy_src.persist.qa_store import persist_unmapped_channels
from legacy.legacy_src.qa.unmapped_channels import extract_unmapped_channel_keys
from legacy.legacy_src.quality.channel_map_coverage import compute_channel_map_coverage
from legacy.legacy_src.quality.star_contract_gates import validate_dim, validate_star_model
from legacy.legacy_src.quality.unmapped_gate import gate_unmapped
from legacy.legacy_src.schemas.schema_hash import schema_hash as compute_schema_hash
from legacy.legacy_src.star.dim_channel import build_dim_channel
from legacy.legacy_src.star.dim_geo import build_dim_geo
from legacy.legacy_src.star.dim_product import build_dim_product
from legacy.legacy_src.star.dim_time import build_dim_time


# Star schema: dimension natural keys and required columns for validation (from schemas/dimensions.schema.yml).
STAR_DIM_CONFIG: dict[str, dict[str, list[str]]] = {
    "dim_time": {"key_cols": ["month_end"], "required_cols": ["month_end", "year", "quarter", "month", "month_name", "is_q_end", "is_y_end"]},
    "dim_channel": {"key_cols": ["preferred_label"], "required_cols": ["preferred_label", "channel_l1", "channel_l2", "channel_id"]},
    "dim_product": {"key_cols": ["product_ticker"], "required_cols": ["product_ticker", "segment", "sub_segment", "product_id"]},
    "dim_geo": {"key_cols": ["country_key"], "required_cols": ["country_key", "country_raw", "country_norm", "iso2", "iso3", "region"]},
}
STAR_JOIN_SPECS: list[dict[str, str]] = [
    {"dim_name": "dim_time", "fact_key": "month_end", "dim_key": "month_end"},
    {"dim_name": "dim_channel", "fact_key": "preferred_label", "dim_key": "preferred_label"},
    {"dim_name": "dim_product", "fact_key": "product_ticker", "dim_key": "product_ticker"},
    {"dim_name": "dim_geo", "fact_key": "src_country", "dim_key": "country_key"},
    {"dim_name": "dim_geo", "fact_key": "product_country", "dim_key": "country_key"},
]


def _atomic_write_json(obj: dict[str, Any], path: Path) -> None:
    """Atomic JSON writer for small coverage reports."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp.{os.getpid()}.cov"
    tmp = p.parent / (p.name + suffix)
    try:
        content = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
        os.replace(tmp, p)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def run_curate_fact_monthly(
    df_raw: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    mapping_path: str | Path = "data/input/DATA_MAPPING.csv",
    mapping_schema_path: str | Path = "schemas/data_mapping.schema.yml",
    fact_schema_path: str | Path = "schemas/fact_monthly.schema.yml",
    root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Orchestration helper:
      - Load DATA_MAPPING via strict loader.
      - Build channel_map and persist it.
      - Build fact_monthly using channel_map as mapping source and persist it.
      - Compute channel mapping coverage and persist to curated/channel_map_coverage.json.

    Returns (df_fact, curated_report).
    """
    root = Path(root) if root is not None else Path.cwd()
    dataset_version = manifest.get("dataset_version", "")
    pipeline_version = manifest.get("pipeline_version", "")

    # 1) Load DATA_MAPPING + build strict channel_map
    df_mapping, mapping_report = load_data_mapping(mapping_path, mapping_schema_path)
    df_channel_map, channel_map_qa = build_channel_map(df_mapping)
    channel_map_meta = persist_channel_map(
        df_channel_map,
        dataset_version=dataset_version,
    )

    # 2) Build fact_monthly using df_channel_map as mapping_df (keys+enrichment)
    df_fact, fact_stats = build_fact_monthly(df_raw, mapping_df=df_channel_map)
    coverage = compute_channel_map_coverage(df_fact, df_channel_map)
    coverage_with_version = {**coverage, "dataset_version": dataset_version}

    # Unmapped channel QA: extract, persist, gate
    df_unmapped = extract_unmapped_channel_keys(df_fact)
    path_csv = root / "qa" / "unmapped_channels.csv"
    path_meta_qa = root / "qa" / "unmapped_channels.meta.json"
    persist_unmapped_channels(
        df_unmapped,
        dataset_version=dataset_version,
        path_csv=path_csv,
        path_meta=path_meta_qa,
    )
    unmapped_rows = (
        int(df_unmapped["row_count"].sum())
        if "row_count" in df_unmapped.columns and len(df_unmapped) > 0
        else 0
    )
    unmapped_keys = len(df_unmapped)
    gate_mode = os.environ.get("UNMAPPED_GATE_MODE", "warn").strip().lower()
    gate_fail_ratio = float(os.environ.get("UNMAPPED_FAIL_ABOVE_RATIO", "0.01"))
    gate_ok, gate_message, gate_stats = gate_unmapped(
        total_rows=len(df_fact),
        unmapped_rows=unmapped_rows,
        unmapped_keys=unmapped_keys,
        mode=gate_mode,
        fail_above_ratio=gate_fail_ratio,
    )
    unmapped_gate = {"ok": gate_ok, "message": gate_message, "stats": gate_stats}
    if not gate_ok:
        raise ValueError(gate_message) from None
    logging.warning(gate_message)

    fact_meta = persist_fact_monthly(
        df_fact,
        dataset_version=dataset_version,
        pipeline_version=pipeline_version,
        schema_path=fact_schema_path,
        root=root,
        mapping_stats=fact_stats.get("mapping_stats"),
        coverage=coverage,
        unmapped_gate=unmapped_gate,
    )

    # 3) Standalone coverage file (same summary as in fact_monthly.meta.json)
    coverage_path = root / "curated" / "channel_map_coverage.json"
    _atomic_write_json(coverage_with_version, coverage_path)

    # 4) Star dimensions (SCD Type 1): build, validate, persist
    curated_dir = root / "curated"
    dim_schema_path = root / "schemas" / "dimensions.schema.yml"
    dim_schema_hash = compute_schema_hash(dim_schema_path) if dim_schema_path.exists() else ""

    df_dim_time = build_dim_time(df_fact)
    df_dim_channel = build_dim_channel(df_fact)
    df_dim_product = build_dim_product(df_fact)
    df_dim_geo = build_dim_geo(df_fact)
    dims = {
        "dim_time": df_dim_time,
        "dim_channel": df_dim_channel,
        "dim_product": df_dim_product,
        "dim_geo": df_dim_geo,
    }

    for dim_name, df_dim in dims.items():
        cfg = STAR_DIM_CONFIG[dim_name]
        ok_dim, err_dim, _ = validate_dim(df_dim, cfg["key_cols"], cfg["required_cols"])
        if not ok_dim:
            raise ValueError(f"Dimension {dim_name} validation failed: " + "; ".join(err_dim)) from None

    ok_star, err_star, star_stats = validate_star_model(
        df_fact, dims, STAR_JOIN_SPECS, min_coverage_ratio=0.999
    )
    if not ok_star:
        raise ValueError("Star model join coverage failed: " + "; ".join(err_star)) from None

    dim_meta: dict[str, dict[str, Any]] = {}
    for dim_name, df_dim in dims.items():
        path_pq = curated_dir / f"{dim_name}.parquet"
        path_m = curated_dir / f"{dim_name}.meta.json"
        dim_meta[dim_name] = persist_dim(
            df_dim,
            path_parquet=path_pq,
            path_meta=path_m,
            dataset_version=dataset_version,
            schema_hash=dim_schema_hash,
        )

    curated_report: dict[str, Any] = {
        "channel_mapping": mapping_report,
        "channel_map_qa": channel_map_qa,
        "channel_map_meta": channel_map_meta,
        "fact_monthly_stats": fact_stats,
        "fact_monthly_meta": fact_meta,
        "channel_map_coverage": coverage_with_version,
        "star_dims_meta": dim_meta,
        "star_validation_stats": star_stats,
    }
    return df_fact, curated_report

