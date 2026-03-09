"""
Build one canonical metric frame: one row per (path_id, slice_id, month_end).
Base: aum_begin_end_by_slice. Fact nnb/nnf aggregated per path via rollup (default sum), then left-joined.
Output: curated/intermediate/metric_frame.parquet + QA missing coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.drill_paths_contract import load_drill_paths_config
from legacy.legacy_pipelines.contracts.rollup_rules_contract import load_rollup_rules
from legacy.legacy_pipelines.slices.slice_keys import compute_slice_id, normalize_slice_value

logger = logging.getLogger(__name__)

AUM_ARTIFACT = "curated/aum_begin_end_by_slice.parquet"
FACT_TABLE = "curated/fact_monthly.parquet"
DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
ROLLUP_RULES_CONFIG = "configs/rollup_rules.yml"
INTERMEDIATE_DIR = "curated/intermediate"
OUTPUT_ARTIFACT = "metric_frame.parquet"
QA_DIR = "qa"
QA_MISSING_COVERAGE_JSON = "metric_frame_missing_coverage.json"
QA_JOIN_COVERAGE_JSON = "metric_frame_join_coverage.json"
QA_MISSING_SLICES_TOP_CSV = "metric_frame_missing_slices_top.csv"

REQUIRED_BASE_COLS = ["path_id", "slice_id", "month_end", "begin_aum", "end_aum", "slice_key"]
OUTPUT_COLUMNS_ORDER = [
    "path_id",
    "slice_id",
    "slice_key",
    "month_end",
    "preferred_label",
    "product_ticker",
    "src_country_canonical",
    "product_country_canonical",
    "begin_aum",
    "end_aum",
    "nnb",
    "nnf",
]

COL_MONTH_END = "month_end"
COL_NNB = "nnb"
COL_NNF = "nnf"

RECON_TOL = 1e-9


def _fact_nnb_nnf_columns(fact: pd.DataFrame) -> tuple[str, str]:
    if COL_NNB in fact.columns and COL_NNF in fact.columns:
        return COL_NNB, COL_NNF
    if "net_new_business" in fact.columns and "net_new_base_fees" in fact.columns:
        return "net_new_business", "net_new_base_fees"
    return "", ""


def _rollup_nnb_nnf(rollup_rules: dict[str, Any]) -> str:
    """Return rollup for nnb/nnf: from measures.nnb / measures.nnf or default 'sum'."""
    measures = rollup_rules.get("measures") or {}
    for name in ("nnb", "nnf"):
        m = measures.get(name)
        if isinstance(m, dict) and m.get("rollup") in ("sum", "max", "last_non_null", "weighted_avg"):
            return str(m["rollup"])
    return "sum"


def _load_base_df(root: Path) -> pd.DataFrame:
    path = root / AUM_ARTIFACT.replace("\\", "/").lstrip("/")
    if not path.exists():
        raise FileNotFoundError(f"AUM artifact not found: {path}")
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Base table missing required columns: {missing}")
    # Hard uniqueness gate on base AUM table
    key_cols = ["path_id", "slice_id", COL_MONTH_END]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, key_cols]
        dup_summary = (
            dup_rows.groupby(key_cols, dropna=False, sort=False)
            .size()
            .reset_index(name="row_count")
        )
        dup_summary = dup_summary.sort_values(
            ["row_count", "path_id", "slice_id", COL_MONTH_END],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(200)
        qa_dir = root / QA_DIR
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / "dup_aum_begin_end_keys.csv"
        dup_summary.to_csv(qa_path, index=False, date_format="%Y-%m-%d")
        total_dup = int(dup_mask.sum())
        raise ValueError(
            f"Base AUM table must be unique at (path_id, slice_id, month_end); "
            f"found {total_dup} duplicate row(s). See {qa_path}"
        )
    return df


def _aggregate_fact_for_path(
    fact: pd.DataFrame,
    path_id: str,
    keys: list[str],
    nnb_col: str,
    nnf_col: str,
    rollup: str,
    qa_dir: Path,
    allow_missing_slice_keys: bool,
) -> pd.DataFrame:
    """Aggregate fact to grain [month_end] + keys; add path_id and slice_id. Returns path_id, slice_id, month_end, nnb, nnf."""
    # C) Missing key columns (hard fail)
    missing_key_cols = [k for k in keys if k not in fact.columns]
    if missing_key_cols:
        raise ValueError(f"Fact missing key columns for path {path_id}: {missing_key_cols}")

    use_cols = [COL_MONTH_END] + keys + [nnb_col, nnf_col]
    missing = [c for c in use_cols if c not in fact.columns]
    if missing:
        raise ValueError(f"Fact missing columns for path {path_id}: {missing}")
    sub = fact[use_cols].copy()

    # D) Missing key values (null/empty)
    if keys:
        missing_mask_any = pd.Series(False, index=sub.index)
        per_key_counts: list[dict[str, Any]] = []
        for k in keys:
            s = sub[k]
            is_null = s.isna()
            is_empty = pd.Series(False, index=sub.index)
            if pd.api.types.is_string_dtype(s) or s.dtype == object:
                is_empty = (~is_null) & (s.astype(str).str.strip() == "")
            key_missing = is_null | is_empty
            cnt = int(key_missing.sum())
            per_key_counts.append({"key": k, "missing_count": cnt})
            missing_mask_any |= key_missing

        missing_rows = int(missing_mask_any.sum())
        if missing_rows > 0:
            qa_dir.mkdir(parents=True, exist_ok=True)
            qa_path = qa_dir / f"missing_slice_keys__{path_id}.csv"
            summary_df = pd.DataFrame(per_key_counts)
            summary_df.insert(0, "row_type", "summary")
            sample = sub.loc[missing_mask_any, [COL_MONTH_END] + keys + [nnb_col, nnf_col]].copy()
            sample.insert(0, "row_type", "sample")
            sample = sample.head(200)
            out_df = pd.concat([summary_df, sample], ignore_index=True)
            out_df.to_csv(qa_path, index=False, date_format="%Y-%m-%d")
            msg = (
                f"Path {path_id}: fact has {missing_rows} row(s) with missing slice key values; "
                f"see {qa_path}"
            )
            if not allow_missing_slice_keys:
                raise ValueError(msg)
            logger.warning("ALLOW_MISSING_SLICE_KEYS=true: %s (proceeding)", msg)
            # When allowed, drop rows with missing keys before aggregation and reconciliation
            sub = sub.loc[~missing_mask_any].copy()

    grain = [COL_MONTH_END] + keys
    agg_df = sub.groupby(grain, dropna=False, sort=False).agg(
        nnb=(nnb_col, "sum"),
        nnf=(nnf_col, "sum"),
    ).reset_index()
    agg_df["path_id"] = path_id
    slice_ids = []
    for _, row in agg_df.iterrows():
        key_values = {k: normalize_slice_value(row[k]) for k in keys}
        slice_ids.append(compute_slice_id(path_id, key_values))
    agg_df["slice_id"] = slice_ids

    # B) Fact aggregation grain uniqueness checks
    # 1) Unique [month_end] + keys after aggregation
    dup_grain = agg_df.duplicated(subset=grain, keep=False)
    # 2) Unique (path_id, slice_id, month_end)
    dup_slice = agg_df.duplicated(subset=["path_id", "slice_id", COL_MONTH_END], keep=False)
    if dup_grain.any() or dup_slice.any():
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / f"dup_fact_agg_after_rollup__{path_id}.csv"
        key_cols = ["path_id", "slice_id", COL_MONTH_END]
        dup_rows = agg_df.loc[dup_grain | dup_slice, key_cols]
        dup_summary = (
            dup_rows.groupby(key_cols, dropna=False, sort=False)
            .size()
            .reset_index(name="row_count")
        )
        dup_summary = dup_summary.sort_values(
            ["row_count", "path_id", "slice_id", COL_MONTH_END],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(200)
        dup_summary.to_csv(qa_path, index=False, date_format="%Y-%m-%d")
        total_dup = int((dup_grain | dup_slice).sum())
        raise ValueError(
            f"Path {path_id}: fact aggregation must be unique at [month_end]+keys and (path_id, slice_id, month_end); "
            f"found {total_dup} duplicate row(s) after rollup. See {qa_path}"
        )

    # 2) Reconciliation check (warn-only): raw fact vs aggregated totals per month_end
    if not sub.empty:
        raw_totals = (
            sub.groupby(COL_MONTH_END, dropna=False, sort=False)
            .agg(
                raw_nnb=(nnb_col, "sum"),
                raw_nnf=(nnf_col, "sum"),
            )
            .reset_index()
        )
        agg_totals = (
            agg_df.groupby(COL_MONTH_END, dropna=False, sort=False)
            .agg(
                agg_nnb=(COL_NNB, "sum"),
                agg_nnf=(COL_NNF, "sum"),
            )
            .reset_index()
        )
        recon = raw_totals.merge(agg_totals, on=COL_MONTH_END, how="outer", sort=False)
        recon = recon.fillna(0.0)
        recon["diff_nnb"] = recon["raw_nnb"] - recon["agg_nnb"]
        recon["diff_nnf"] = recon["raw_nnf"] - recon["agg_nnf"]
        mism_mask = (recon["diff_nnb"].abs() > RECON_TOL) | (recon["diff_nnf"].abs() > RECON_TOL)
        if mism_mask.any():
            qa_dir.mkdir(parents=True, exist_ok=True)
            qa_path = qa_dir / f"nnb_reconciliation__{path_id}.csv"
            out = recon.loc[
                mism_mask,
                [COL_MONTH_END, "raw_nnb", "agg_nnb", "diff_nnb", "raw_nnf", "agg_nnf", "diff_nnf"],
            ]
            out.sort_values(COL_MONTH_END, kind="mergesort").to_csv(
                qa_path, index=False, date_format="%Y-%m-%d"
            )
            logger.warning(
                "NnB/NNF reconciliation mismatch for path %s at %d month_end(s); see %s",
                path_id,
                int(mism_mask.sum()),
                qa_path,
            )

    return agg_df[["path_id", "slice_id", COL_MONTH_END, COL_NNB, COL_NNF]]


def build_canonical_metric_frame(
    root: Path,
    drill_paths_config: dict[str, Any],
    rollup_rules: dict[str, Any],
    fact_df: pd.DataFrame,
    fill_missing_with_zero: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build one canonical table at (path_id, slice_id, month_end) with begin_aum, end_aum, nnb, nnf.
    Returns (metric_frame_df, qa_info) where qa_info has missing_coverage_row_count and optional details.
    """
    base_df = _load_base_df(root)
    nnb_col, nnf_col = _fact_nnb_nnf_columns(fact_df)
    if not nnb_col or not nnf_col:
        raise ValueError("Fact must contain nnb/nnf or net_new_business/net_new_base_fees")

    rollup = _rollup_nnb_nnf(rollup_rules)
    paths = [
        p for p in drill_paths_config.get("drill_paths", [])
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    if not paths:
        raise ValueError("No enabled drill paths in config")

    qa_dir = root / QA_DIR
    allow_missing_slice_keys = os.environ.get("ALLOW_MISSING_SLICE_KEYS", "false").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    parts = []
    for p in paths:
        path_id = str(p.get("id", ""))
        keys = list(p.get("keys") or [])
        part = _aggregate_fact_for_path(
            fact_df,
            path_id,
            keys,
            nnb_col,
            nnf_col,
            rollup,
            qa_dir=qa_dir,
            allow_missing_slice_keys=allow_missing_slice_keys,
        )
        parts.append(part)

    if not parts:
        raise ValueError("No fact aggregation produced for any enabled path")
    fact_agg_all = pd.concat(parts, ignore_index=True)

    base_df = base_df.copy()
    merged = base_df.merge(
        fact_agg_all,
        on=["path_id", "slice_id", COL_MONTH_END],
        how="left",
        sort=False,
        indicator="_merge_fact",
    )
    matched_mask = merged["_merge_fact"] == "both"
    missing_mask = merged["_merge_fact"] == "left_only"

    total_rows = int(len(merged))
    matched_rows = int(matched_mask.sum())
    missing_rows = int(missing_mask.sum())
    matched_rate = float(matched_rows / total_rows) if total_rows > 0 else 0.0

    by_path: dict[str, Any] = {}
    if total_rows > 0 and "path_id" in merged.columns:
        grp = merged.groupby("path_id", sort=False)
        for pid, g in grp:
            t = int(len(g))
            m = int((g["_merge_fact"] == "both").sum())
            miss = int((g["_merge_fact"] == "left_only").sum())
            by_path[str(pid)] = {
                "total_rows": t,
                "matched_rows": m,
                "missing_rows": miss,
                "matched_rate": float(m / t) if t > 0 else 0.0,
            }

    join_coverage = {
        "total_rows": total_rows,
        "matched_rows": matched_rows,
        "missing_rows": missing_rows,
        "matched_rate": matched_rate,
        "by_path_id": by_path,
    }

    # Missing slices top 200 by path_id + slice_key
    missing_df = merged.loc[missing_mask].copy()
    if "months_in_slice" not in missing_df.columns:
        missing_df["months_in_slice"] = pd.NA
    missing_cols = ["path_id", "slice_id", "slice_key", "months_in_slice", COL_MONTH_END]
    missing_slices_top_df = (
        missing_df[missing_cols]
        .sort_values(
            ["path_id", "months_in_slice", "slice_id", COL_MONTH_END],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .head(200)
    )
    missing_slices_top = missing_slices_top_df.to_dict("records")

    merged = merged.drop(columns=["_merge_fact"])
    if fill_missing_with_zero:
        merged.loc[missing_mask, COL_NNB] = 0.0
        merged.loc[missing_mask, COL_NNF] = 0.0
    else:
        merged.loc[missing_mask, COL_NNB] = pd.NA
        merged.loc[missing_mask, COL_NNF] = pd.NA

    missing_count = missing_rows
    qa_info: dict[str, Any] = {
        "missing_coverage_row_count": missing_count,
        "missing_coverage_note": "Rows in base (aum_begin_end) with no matching fact aggregate; nnb/nnf filled per config."
        if missing_count > 0
        else "",
        "join_coverage": join_coverage,
        "missing_slices_top": missing_slices_top,
    }

    for c in OUTPUT_COLUMNS_ORDER:
        if c not in merged.columns:
            if c in base_df.columns:
                merged[c] = base_df[c].values
            else:
                merged[c] = pd.NA
    return merged[OUTPUT_COLUMNS_ORDER], qa_info


def run(root: Path) -> pd.DataFrame:
    """Load configs and fact, build canonical frame, write parquet and QA. Returns the frame."""
    drill_paths_config = load_drill_paths_config(root / DRILL_PATHS_CONFIG)
    rollup_rules = load_rollup_rules(root / ROLLUP_RULES_CONFIG)
    fact_path = root / FACT_TABLE.replace("\\", "/").lstrip("/")
    if not fact_path.exists():
        raise FileNotFoundError(f"Fact table not found: {fact_path}")
    fact_df = pd.read_parquet(fact_path)

    fill_missing = os.environ.get("METRIC_FRAME_FILL_MISSING", "1").strip().lower() in ("1", "true", "yes")
    frame_df, qa_info = build_canonical_metric_frame(
        root, drill_paths_config, rollup_rules, fact_df, fill_missing_with_zero=fill_missing
    )

    out_dir = root / INTERMEDIATE_DIR.replace("\\", "/").lstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_ARTIFACT
    frame_df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(frame_df))

    # Meta: schema_hash, dataset_version, row_count, key_unique
    schema_parts = [f"{c}:{str(frame_df[c].dtype)}" for c in frame_df.columns]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    key_cols = ["path_id", "slice_id", COL_MONTH_END]
    key_unique = not frame_df.duplicated(subset=key_cols, keep=False).any()
    dataset_version = f"metric_frame_{schema_hash[:16]}"
    meta = {
        "row_count": int(len(frame_df)),
        "schema_hash": schema_hash,
        "dataset_version": dataset_version,
        "primary_key": key_cols,
        "key_unique": bool(key_unique),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", meta_path)

    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    # Existing coverage QA (includes join_coverage and missing_slices_top)
    qa_path = qa_dir / QA_MISSING_COVERAGE_JSON
    qa_path.write_text(json.dumps(qa_info, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote QA %s", qa_path)

    # Join coverage JSON
    join_coverage = qa_info.get("join_coverage") or {}
    join_path = qa_dir / QA_JOIN_COVERAGE_JSON
    join_path.write_text(json.dumps(join_coverage, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote QA %s", join_path)

    # Missing slices top CSV
    missing_slices = qa_info.get("missing_slices_top") or []
    if missing_slices:
        missing_df = pd.DataFrame(missing_slices)
        missing_path = qa_dir / QA_MISSING_SLICES_TOP_CSV
        missing_df.to_csv(missing_path, index=False, date_format="%Y-%m-%d")
        logger.info("Wrote QA %s", missing_path)

    return frame_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical metric frame: one row per (path_id, slice_id, month_end).")
    parser.add_argument("--run", action="store_true", help="Run build and write outputs")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    root = args.root or Path(__file__).resolve().parents[2]
    if not args.run:
        logger.info("Use --run to build and write %s/%s", INTERMEDIATE_DIR, OUTPUT_ARTIFACT)
        return 0
    try:
        run(root)
        return 0
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
