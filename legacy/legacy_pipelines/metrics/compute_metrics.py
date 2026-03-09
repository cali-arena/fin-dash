"""
Compute metrics by drill path only. One canonical table: curated/metrics_monthly.parquet.
Contract: AUM-related metrics (end_aum, begin_aum) come ONLY from curated/intermediate/aum_series_all.parquet
(or per-path aum_series__{path_id}.parquet). Never recompute begin_aum in the metrics layer.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
CURATED_DIR = "curated"
INTERMEDIATE_DIR = "curated/intermediate"
QA_DIR = "qa"
AUM_SERIES_ALL = "aum_series_all.parquet"
AUM_SERIES_PREFIX = "aum_series__"
FACT_TABLE = "curated/fact_monthly.parquet"
SLICES_INDEX_TABLE = "curated/slices_index.parquet"
METRICS_TABLE = "metrics_monthly"
METRICS_SLICE_COVERAGE_QA = "metrics_slice_coverage.json"
QA_FIRST_MONTH_ANOMALY_PREFIX = "begin_aum_first_month_anomaly__"
QA_MISMATCH_SAMPLES = "begin_aum_mismatch_samples.csv"

COL_MONTH_END = "month_end"
COL_AUM = "asset_under_management"
COL_NNB = "net_new_business"
COL_NNF = "net_new_base_fees"

OUT_END_AUM = "end_aum"
OUT_NNB = "nnb"
OUT_NNF = "nnf"
OUT_BEGIN_AUM = "begin_aum"
OUT_AUM_GROWTH_RATE = "aum_growth_rate"
IS_FIRST_MONTH_IN_SLICE = "is_first_month_in_slice"

AUM_SERIES_BUILD_CMD = "python -m pipelines.slices.begin_aum_series --build"
LAG_SPOT_CHECK_SEED = 42
LAG_SPOT_CHECK_SAMPLE_SIZE = 50


def _load_aum_series_authoritative(root: Path) -> pd.DataFrame:
    """Load aum_series_all.parquet or per-path aum_series__{path_id}.parquet as authoritative. Never recompute begin_aum."""
    intermediate = root / INTERMEDIATE_DIR
    all_path = intermediate / AUM_SERIES_ALL
    if all_path.exists():
        return pd.read_parquet(all_path)
    raise ValueError(
        f"AUM series not found: {all_path}. Run: {AUM_SERIES_BUILD_CMD}"
    )


def _metrics_for_path_from_aum_series(
    aum_df: pd.DataFrame,
    path_id: str,
    path_label: str,
    keys: list[str],
    fact_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build metrics for one path from aum_series (end_aum + begin_aum authoritative). No shift/lag in metrics layer.
    Optionally merge nnb/nnf from fact at same grain. Compute aum_growth_rate from end_aum and begin_aum only.
    """
    df = aum_df[aum_df["path_id"] == path_id].copy()
    if df.empty:
        return pd.DataFrame(columns=["path_id", "path_label", "slice_id", "slice_key", COL_MONTH_END, OUT_END_AUM, OUT_NNB, OUT_NNF, OUT_BEGIN_AUM, OUT_AUM_GROWTH_RATE])
    df["path_label"] = path_label

    df[OUT_AUM_GROWTH_RATE] = pd.NA
    mask = (df[OUT_BEGIN_AUM].notna()) & (df[OUT_BEGIN_AUM] > 0)
    df.loc[mask, OUT_AUM_GROWTH_RATE] = (
        (df.loc[mask, OUT_END_AUM] - df.loc[mask, OUT_BEGIN_AUM]) / df.loc[mask, OUT_BEGIN_AUM]
    )

    if fact_df is not None and not fact_df.empty and COL_NNB in fact_df.columns and COL_NNF in fact_df.columns:
        group_cols = [COL_MONTH_END] + [k for k in keys if k in df.columns and k in fact_df.columns]
        missing_fact = [c for c in group_cols + [COL_NNB, COL_NNF] if c not in fact_df.columns]
        if not missing_fact and group_cols:
            fact_agg = fact_df.groupby(group_cols, dropna=False, sort=False).agg(
                nnb=(COL_NNB, "sum"),
                nnf=(COL_NNF, "sum"),
            ).reset_index()
            df = df.merge(fact_agg, on=group_cols, how="left", sort=False)
        else:
            df[OUT_NNB] = pd.NA
            df[OUT_NNF] = pd.NA
    else:
        df[OUT_NNB] = pd.NA
        df[OUT_NNF] = pd.NA

    out_cols = ["path_id", "path_label", "slice_id", "slice_key", COL_MONTH_END, OUT_END_AUM]
    if OUT_NNB in df.columns:
        out_cols.extend([OUT_NNB, OUT_NNF])
    out_cols.extend([OUT_BEGIN_AUM, OUT_AUM_GROWTH_RATE])
    return df[[c for c in out_cols if c in df.columns]].copy()


def _validate_metrics_grain(metrics_df: pd.DataFrame) -> None:
    """Ensure output grain is unique: (path_id, slice_id, month_end)."""
    if metrics_df.empty:
        return
    dup = metrics_df.duplicated(subset=["path_id", "slice_id", COL_MONTH_END], keep=False)
    if dup.any():
        n = dup.sum()
        raise ValueError(
            f"metrics_monthly grain must be unique (path_id, slice_id, month_end); found {n} duplicate row(s)"
        )


def _reconciliation_first_month_per_slice(aum_df: pd.DataFrame, qa_dir: Path) -> None:
    """
    For each path_id: sum(is_first_month_in_slice) must equal number of unique slices (each slice has exactly one first month).
    If not, hard fail and dump qa/begin_aum_first_month_anomaly__{path_id}.csv with per-slice n_first_month and expected 1.
    """
    if IS_FIRST_MONTH_IN_SLICE not in aum_df.columns:
        return
    qa_dir.mkdir(parents=True, exist_ok=True)
    for path_id in aum_df["path_id"].unique():
        sub = aum_df[aum_df["path_id"] == path_id]
        count_first = sub[IS_FIRST_MONTH_IN_SLICE].sum()
        count_first = 0 if pd.isna(count_first) else int(count_first)
        unique_slices = sub["slice_id"].nunique()
        if count_first != unique_slices:
            per_slice = (
                sub.groupby("slice_id", sort=False)[IS_FIRST_MONTH_IN_SLICE]
                .sum()
                .reset_index(name="n_first_month")
            )
            per_slice["expected"] = 1
            per_slice["path_id"] = path_id
            anomaly_path = qa_dir / f"{QA_FIRST_MONTH_ANOMALY_PREFIX}{path_id}.csv"
            per_slice[["path_id", "slice_id", "n_first_month", "expected"]].to_csv(anomaly_path, index=False)
            raise ValueError(
                f"begin_aum reconciliation: path_id={path_id} count(is_first_month_in_slice)={count_first} "
                f"!= unique slices={unique_slices}. See {anomaly_path}"
            )


def _reconciliation_lag_spot_check(aum_df: pd.DataFrame, qa_dir: Path) -> None:
    """
    Random sample of 50 slices (seeded); verify begin_aum[t] == end_aum[t-1] for non-first months.
    If any mismatch, hard fail and write qa/begin_aum_mismatch_samples.csv.
    """
    rng = random.Random(LAG_SPOT_CHECK_SEED)
    slice_ids = aum_df["slice_id"].unique().tolist()
    if len(slice_ids) == 0:
        return
    sample_size = min(LAG_SPOT_CHECK_SAMPLE_SIZE, len(slice_ids))
    chosen = rng.sample(slice_ids, sample_size)

    mismatches = []
    for sid in chosen:
        sub = aum_df[aum_df["slice_id"] == sid].sort_values(COL_MONTH_END, kind="mergesort")
        if sub[IS_FIRST_MONTH_IN_SLICE].all():
            continue
        prev_end = None
        for _, row in sub.iterrows():
            if row[IS_FIRST_MONTH_IN_SLICE]:
                prev_end = row[OUT_END_AUM]
                continue
            expected = prev_end
            got = row[OUT_BEGIN_AUM]
            if pd.isna(expected) and pd.isna(got):
                prev_end = row[OUT_END_AUM]
                continue
            if pd.isna(expected) or pd.isna(got) or abs(float(expected) - float(got)) > 1e-9:
                mismatches.append({
                    "slice_id": sid,
                    "month_end": row[COL_MONTH_END],
                    "expected": expected,
                    "got": got,
                })
            prev_end = row[OUT_END_AUM]

    if mismatches:
        qa_dir.mkdir(parents=True, exist_ok=True)
        mismatch_path = qa_dir / QA_MISMATCH_SAMPLES
        pd.DataFrame(mismatches).to_csv(mismatch_path, index=False, date_format="%Y-%m-%d")
        raise ValueError(
            f"begin_aum lag spot-check: {len(mismatches)} mismatch(es). See {mismatch_path}"
        )


def build_metrics_monthly(
    drill_paths_config: dict[str, Any],
    root: Path,
    fact_df: pd.DataFrame | None = None,
    aum_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build one canonical metrics table from aum_series (authoritative end_aum + begin_aum). No begin_aum recomputation.
    Runs reconciliation QA: first-month count vs unique slices; lag spot-check. nnb/nnf optionally from fact at same grain.
    """
    paths = [
        p
        for p in drill_paths_config.get("drill_paths", [])
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    if not paths:
        raise ValueError("No enabled drill_paths in config")

    if aum_df is None:
        aum_df = _load_aum_series_authoritative(root)

    qa_dir = root / QA_DIR
    _reconciliation_first_month_per_slice(aum_df, qa_dir)
    _reconciliation_lag_spot_check(aum_df, qa_dir)

    parts = []
    for p in paths:
        path_id = str(p.get("id", ""))
        path_label = str(p.get("label", ""))
        keys = list(p.get("keys") or [])
        part = _metrics_for_path_from_aum_series(aum_df, path_id, path_label, keys, fact_df)
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["path_id", "slice_id", COL_MONTH_END], kind="mergesort").reset_index(drop=True)
    _validate_metrics_grain(out)
    return out


def validate_metrics_slices_contract(metrics_df: pd.DataFrame, slices_index_df: pd.DataFrame) -> None:
    """
    Validate that every (path_id, slice_id) in metrics_df exists in slices_index.
    If any metric row has a slice not in the index, raise ValueError (contract breach).
    """
    if metrics_df.empty:
        return
    needed = metrics_df[["path_id", "slice_id"]].drop_duplicates()
    index_slices = slices_index_df[["path_id", "slice_id"]].drop_duplicates()
    merged = needed.merge(index_slices, on=["path_id", "slice_id"], how="left", indicator=True)
    missing = merged[merged["_merge"] != "both"]
    if len(missing) > 0:
        n = len(missing)
        sample = missing.head(5)[["path_id", "slice_id"]].to_dict("records")
        raise ValueError(
            f"metrics_slice_contract: {n} (path_id, slice_id) in metrics_monthly not found in slices_index. "
            f"Sample: {sample}"
        )


def write_metrics_slice_coverage_qa(metrics_df: pd.DataFrame, qa_path: Path) -> None:
    """
    Write qa/metrics_slice_coverage.json:
    - enabled_path_count (from metrics path_id.nunique())
    - slices_per_path: { path_id: slice_count }
    - max_months_per_slice, min_months_per_slice
    - slices_with_one_month: list of (path_id, slice_id) or count; warn in structure
    """
    if metrics_df.empty:
        coverage = {
            "enabled_path_count": 0,
            "slices_per_path": {},
            "max_months_per_slice": 0,
            "min_months_per_slice": 0,
            "slices_with_one_month": [],
            "slices_with_one_month_count": 0,
            "warn_slices_with_one_month": False,
        }
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
        return

    months_per_slice = (
        metrics_df.groupby(["path_id", "slice_id"], sort=False)[COL_MONTH_END]
        .nunique()
        .reset_index(name="month_count")
    )
    slices_per_path = (
        metrics_df.groupby("path_id", sort=False)["slice_id"]
        .nunique()
        .to_dict()
    )
    max_months = int(months_per_slice["month_count"].max())
    min_months = int(months_per_slice["month_count"].min())
    one_month = months_per_slice[months_per_slice["month_count"] == 1]
    slices_with_one_month = one_month[["path_id", "slice_id"]].to_dict("records")
    slices_with_one_month_count = len(slices_with_one_month)

    coverage = {
        "enabled_path_count": int(metrics_df["path_id"].nunique()),
        "slices_per_path": {str(k): int(v) for k, v in slices_per_path.items()},
        "max_months_per_slice": max_months,
        "min_months_per_slice": min_months,
        "slices_with_one_month": slices_with_one_month,
        "slices_with_one_month_count": slices_with_one_month_count,
        "warn_slices_with_one_month": slices_with_one_month_count > 0,
    }
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    if coverage["warn_slices_with_one_month"]:
        logger.warning(
            "QA: %d slice(s) have only 1 month of data (see qa/metrics_slice_coverage.json)",
            slices_with_one_month_count,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute metrics by drill path; write metrics_monthly + QA.")
    parser.add_argument("--build", action="store_true", help="Build curated/metrics_monthly.parquet and QA")
    parser.add_argument("--config", default=DEFAULT_DRILL_PATHS_CONFIG, help="Path to drill_paths.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = Path(args.root) if args.root is not None else Path.cwd()
    if not args.build:
        logger.info("Use --build to compute metrics and write outputs.")
        return 0

    from legacy.legacy_pipelines.contracts.drill_paths_contract import load_drill_paths_config

    config_path = root / args.config
    drill_config = load_drill_paths_config(config_path)
    fact_path = root / drill_config.get("fact_table", "curated/fact_monthly.parquet")
    slices_path = root / SLICES_INDEX_TABLE
    if not slices_path.exists():
        logger.error("slices_index not found: %s (run pipelines.slices.slice_keys --build first)", slices_path)
        return 1
    fact_df = pd.read_parquet(fact_path) if fact_path.exists() else None
    if fact_df is None:
        logger.info("Fact not loaded (missing); nnb/nnf will be null in metrics.")
    slices_index_df = pd.read_parquet(slices_path)

    try:
        metrics_df = build_metrics_monthly(drill_config, root, fact_df=fact_df)
        validate_metrics_slices_contract(metrics_df, slices_index_df)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    curated_dir = root / CURATED_DIR
    curated_dir.mkdir(parents=True, exist_ok=True)
    out_path = curated_dir / f"{METRICS_TABLE}.parquet"
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=curated_dir, prefix="metrics_")
    try:
        os.close(fd)
        metrics_df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    except Exception:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%d rows)", out_path, len(metrics_df))

    qa_path = root / QA_DIR / METRICS_SLICE_COVERAGE_QA
    write_metrics_slice_coverage_qa(metrics_df, qa_path)
    logger.info("Wrote QA %s", qa_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
