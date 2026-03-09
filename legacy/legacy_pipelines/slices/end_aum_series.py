"""
Intermediate table builder: one end_aum per (month_end + slice keys) per enabled drill path.
Uses drill_paths.yml for slice keys and rollup_rules.yml for duplicate resolution.
QA: duplicate reports per path, stats JSON, and reconciliation check for sum rollup.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.drill_paths_contract import load_drill_paths_config, validate_drill_paths
from legacy.legacy_pipelines.contracts.rollup_rules_contract import (
    load_rollup_rules,
    validate_rollup_rules,
)

logger = logging.getLogger(__name__)

DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
DEFAULT_ROLLUP_RULES_CONFIG = "configs/rollup_rules.yml"
FACT_TABLE_REL = "curated/fact_monthly.parquet"
INTERMEDIATE_DIR = "curated/intermediate"
QA_DIR = "qa"
END_AUM_SERIES_PREFIX = "end_aum_series__"
END_AUM_SERIES_ALL = "end_aum_series_all.parquet"
QA_DUPLICATES_PREFIX = "end_aum_duplicates__"
QA_STATS_PREFIX = "end_aum_series_stats__"
QA_DUPLICATES_TOP_N = 500
RECONCILIATION_TOLERANCE = 1e-6

# Fact column for AUM (output as end_aum)
AUM_SOURCE_COL = "asset_under_management"
OUT_END_AUM = "end_aum"
COL_MONTH_END = "month_end"

# Optional fact columns for QA sample row identifiers
SOURCE_ROW_ID_CANDIDATES = ["source_row_id", "row_id"]

# All potential key columns for combined dataset (order for output)
ALL_KEY_COLS = [
    "preferred_label",
    "product_ticker",
    "src_country_canonical",
    "product_country_canonical",
]


def _normalize_month_end(ser: pd.Series) -> pd.Series:
    """Normalize month_end to datetime (timezone-naive, normalized)."""
    return pd.to_datetime(ser, utc=False).dt.normalize()


def _normalize_keys_to_string(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Ensure key columns are string-safe; leave canonical as-is (no case/space change)."""
    out = df.copy()
    for k in keys:
        if k not in out.columns:
            continue
        out[k] = out[k].astype(object).fillna("").astype(str).str.strip()
    return out


def _apply_rollup(
    df: pd.DataFrame,
    grain_cols: list[str],
    end_aum_col: str,
    rollup: str,
    snapshot_order_by: list[str] | None,
    snapshot_direction: str,
) -> pd.DataFrame:
    """
    Aggregate to one end_aum per grain. rollup in: sum | max | last_non_null.
    """
    if rollup == "sum":
        agg = df.groupby(grain_cols, dropna=False, sort=False)[end_aum_col].sum(min_count=1)
        return agg.reset_index().rename(columns={end_aum_col: OUT_END_AUM})
    if rollup == "max":
        agg = df.groupby(grain_cols, dropna=False, sort=False)[end_aum_col].max()
        return agg.reset_index().rename(columns={end_aum_col: OUT_END_AUM})
    if rollup == "last_non_null":
        if not snapshot_order_by:
            raise ValueError("rollup last_non_null requires snapshot_tiebreak.order_by in config")
        missing = [c for c in snapshot_order_by if c not in df.columns]
        if missing:
            raise ValueError(
                f"rollup last_non_null: order columns {snapshot_order_by} required; missing in fact: {missing}"
            )
        desc = (snapshot_direction or "desc").lower() == "desc"
        df_sorted = df.sort_values(snapshot_order_by, ascending=not desc, na_position="last")
        # Within each grain, take first row with non-null end_aum
        out_rows = []
        for name, grp in df_sorted.groupby(grain_cols, dropna=False, sort=False):
            if len(grain_cols) == 1:
                name = (name,)
            subset = grp[grp[end_aum_col].notna()]
            if len(subset) == 0:
                val = None
            else:
                val = subset[end_aum_col].iloc[0]
            row = {grain_cols[i]: name[i] for i in range(len(grain_cols))}
            row[OUT_END_AUM] = val
            out_rows.append(row)
        out = pd.DataFrame(out_rows)
        out = out[grain_cols + [OUT_END_AUM]]
        return out
    raise ValueError(f"Unsupported rollup: {rollup}")


def _compute_dup_report_and_stats(
    df: pd.DataFrame,
    grain_cols: list[str],
    aum_col: str,
    row_id_col: str | None,
) -> tuple[pd.DataFrame, int, int, float]:
    """
    Before aggregation: compute duplicate report and stats.
    Returns (dup_report_df with month_end, keys..., rows_in_group, end_aum_min, end_aum_max, end_aum_sum, sample_row_ids), dup_groups, dup_rows_total, dup_rate.
    dup_report limited to top QA_DUPLICATES_TOP_N by rows_in_group desc.
    """
    total_rows = len(df)
    if total_rows == 0:
        return pd.DataFrame(), 0, 0, 0.0

    grp = df.groupby(grain_cols, dropna=False)
    size = grp.size()
    dup_mask = size > 1
    dup_groups = int(dup_mask.sum())
    dup_rows_total = int(size[dup_mask].sum()) if dup_mask.any() else 0
    dup_rate = dup_rows_total / total_rows if total_rows else 0.0

    report_rows = []
    for name, g in grp:
        n = len(g)
        if n <= 1:
            continue
        if len(grain_cols) == 1:
            name = (name,)
        row = {grain_cols[i]: name[i] for i in range(len(grain_cols))}
        row["rows_in_group"] = n
        row["end_aum_min"] = g[aum_col].min()
        row["end_aum_max"] = g[aum_col].max()
        row["end_aum_sum"] = g[aum_col].sum()
        if row_id_col and row_id_col in g.columns:
            row["sample_row_ids"] = ",".join(g[row_id_col].astype(str).head(5).tolist())
        else:
            row["sample_row_ids"] = ""
        report_rows.append(row)
    if not report_rows:
        report = pd.DataFrame(columns=grain_cols + ["rows_in_group", "end_aum_min", "end_aum_max", "end_aum_sum", "sample_row_ids"])
    else:
        report = pd.DataFrame(report_rows)
        report = report.sort_values("rows_in_group", ascending=False, kind="mergesort").head(QA_DUPLICATES_TOP_N).reset_index(drop=True)
    return report, dup_groups, dup_rows_total, dup_rate


def _build_series_for_path(
    fact_df: pd.DataFrame,
    path_id: str,
    keys: list[str],
    rollup_rules: dict[str, Any],
    root: Path,
    build_timestamp: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    Build end_aum series for one drill path: grain = [month_end] + keys, one end_aum per grain.
    Returns (out_df, stats_dict, dup_report_df).
    out_df: month_end + keys + end_aum, sorted. stats_dict: input_rows, output_rows, dup_groups, dup_rows_total, dup_rate, rollup_used, build_timestamp.
    """
    aum_col = AUM_SOURCE_COL if AUM_SOURCE_COL in fact_df.columns else OUT_END_AUM
    if aum_col not in fact_df.columns:
        raise ValueError(f"Fact missing AUM column (need {AUM_SOURCE_COL} or {OUT_END_AUM})")

    grain_cols = [COL_MONTH_END] + keys
    required = [COL_MONTH_END, aum_col] + keys
    row_id_col = None
    for c in SOURCE_ROW_ID_CANDIDATES:
        if c in fact_df.columns:
            row_id_col = c
            break
    missing = [c for c in required if c not in fact_df.columns]
    if missing:
        raise ValueError(f"end_aum_series path {path_id}: fact missing columns: {missing}")

    select_cols = required + ([row_id_col] if row_id_col else [])
    df = fact_df[select_cols].copy()
    if row_id_col is None:
        df["_source_index"] = range(len(df))
        row_id_col = "_source_index"
    df[COL_MONTH_END] = _normalize_month_end(df[COL_MONTH_END])
    df = _normalize_keys_to_string(df, keys)

    if df[COL_MONTH_END].isna().any():
        raise ValueError(f"end_aum_series path {path_id}: month_end must not be null")
    if not pd.api.types.is_numeric_dtype(df[aum_col]):
        raise ValueError(f"end_aum_series path {path_id}: end_aum (source {aum_col}) must be numeric")

    defaults = rollup_rules.get("defaults") or {}
    if defaults.get("null_end_aum") == "drop":
        df = df[df[aum_col].notna()].copy()

    input_rows = len(df)
    dup_report, dup_groups, dup_rows_total, dup_rate = _compute_dup_report_and_stats(df, grain_cols, aum_col, row_id_col)

    df_agg = df[grain_cols + [aum_col]].copy()

    measures = rollup_rules.get("measures") or {}
    end_aum_cfg = measures.get("end_aum") or {}
    rollup = end_aum_cfg.get("rollup", "sum")
    snapshot = end_aum_cfg.get("snapshot_tiebreak") or {}
    order_by = snapshot.get("order_by") if isinstance(snapshot.get("order_by"), list) else None
    direction = snapshot.get("direction", "desc")

    out = _apply_rollup(df_agg, grain_cols, aum_col, rollup, order_by, direction)
    if out.duplicated(subset=grain_cols).any():
        raise ValueError(f"end_aum_series path {path_id}: grain must be unique after aggregation")
    out = out.sort_values(grain_cols, kind="mergesort").reset_index(drop=True)
    output_rows = len(out)

    # Reconciliation: for sum rollup, aggregated total by month must match raw total by month (within tolerance)
    if rollup == "sum":
        raw_by_month = df.groupby(COL_MONTH_END, dropna=False)[aum_col].sum()
        agg_by_month = out.groupby(COL_MONTH_END, dropna=False)[OUT_END_AUM].sum()
        common = raw_by_month.index.union(agg_by_month.index)
        raw_aligned = raw_by_month.reindex(common).fillna(0)
        agg_aligned = agg_by_month.reindex(common).fillna(0)
        diff = (agg_aligned - raw_aligned).abs()
        if diff.max() > RECONCILIATION_TOLERANCE:
            raise ValueError(
                f"end_aum_series path {path_id}: sum rollup reconciliation failed; max |agg - raw| by month = {diff.max()}"
            )
    else:
        raw_by_month = df.groupby(COL_MONTH_END, dropna=False)[aum_col].sum()
        agg_by_month = out.groupby(COL_MONTH_END, dropna=False)[OUT_END_AUM].sum()
        logger.info(
            "end_aum_series path %s (rollup=%s): raw_month_total sample %s, aggregated_month_total sample %s",
            path_id, rollup, raw_by_month.sum(), agg_by_month.sum(),
        )

    stats = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "dup_groups": dup_groups,
        "dup_rows_total": dup_rows_total,
        "dup_rate": dup_rate,
        "rollup_used": rollup,
        "build_timestamp": build_timestamp,
    }
    return out, stats, dup_report


def build_all_end_aum_series(
    fact_df: pd.DataFrame,
    drill_paths_config: dict[str, Any],
    rollup_rules: dict[str, Any],
    root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Build per-path series and combined dataset.
    Returns (per_path, combined_df, qa_data). qa_data[path_id] = {"stats": {...}, "dup_df": DataFrame}.
    """
    from legacy.legacy_pipelines.contracts.drill_paths_contract import validate_drill_paths

    validate_drill_paths(drill_paths_config, list(fact_df.columns))
    validate_rollup_rules(rollup_rules, list(fact_df.columns))

    paths = [
        p
        for p in drill_paths_config.get("drill_paths", [])
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    if not paths:
        raise ValueError("No enabled drill_paths")

    build_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    per_path: dict[str, pd.DataFrame] = {}
    combined_parts = []
    qa_data: dict[str, dict[str, Any]] = {}

    for p in paths:
        path_id = str(p.get("id", ""))
        keys = list(p.get("keys") or [])
        df, stats, dup_report = _build_series_for_path(fact_df, path_id, keys, rollup_rules, root, build_timestamp)
        per_path[path_id] = df
        qa_data[path_id] = {"stats": stats, "dup_df": dup_report}

        combined = df.copy()
        combined["path_id"] = path_id
        for k in ALL_KEY_COLS:
            if k not in combined.columns:
                combined[k] = pd.NA
        combined = combined[["path_id", COL_MONTH_END] + ALL_KEY_COLS + [OUT_END_AUM]]
        combined_parts.append(combined)

    combined_df = pd.concat(combined_parts, ignore_index=True)
    combined_df = combined_df.sort_values(["path_id", COL_MONTH_END] + ALL_KEY_COLS, kind="mergesort").reset_index(drop=True)
    return per_path, combined_df, qa_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Build intermediate end_aum_series per path and combined.")
    parser.add_argument("--build", action="store_true", help="Build and write curated/intermediate/end_aum_series_*.parquet")
    parser.add_argument("--drill-config", default=DEFAULT_DRILL_PATHS_CONFIG, help="Path to drill_paths.yml")
    parser.add_argument("--rollup-config", default=DEFAULT_ROLLUP_RULES_CONFIG, help="Path to rollup_rules.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = Path(args.root) if args.root is not None else Path.cwd()
    if not args.build:
        logger.info("Use --build to build end_aum_series outputs.")
        return 0

    drill_config = load_drill_paths_config(root / args.drill_config)
    rollup_rules = load_rollup_rules(root / args.rollup_config)
    fact_path = root / (drill_config.get("fact_table") or FACT_TABLE_REL)
    if not fact_path.exists():
        logger.error("Fact not found: %s", fact_path)
        return 1

    fact_df = pd.read_parquet(fact_path)

    try:
        per_path, combined_df, qa_data = build_all_end_aum_series(fact_df, drill_config, rollup_rules, root)
    except ValueError as e:
        logger.error("%s", e)
        return 1

    out_dir = root / INTERMEDIATE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)

    for path_id, df in per_path.items():
        out_path = out_dir / f"{END_AUM_SERIES_PREFIX}{path_id}.parquet"
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=out_dir, prefix="end_aum_")
        try:
            os.close(fd)
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out_path)
        except Exception:
            if Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)
            raise
        logger.info("Wrote %s (%d rows)", out_path, len(df))

        qa = qa_data.get(path_id, {})
        dup_df = qa.get("dup_df")
        if dup_df is not None:
            dup_path = qa_dir / f"{QA_DUPLICATES_PREFIX}{path_id}.csv"
            dup_df.to_csv(dup_path, index=False, date_format="%Y-%m-%d")
            logger.info("Wrote QA %s (%d rows)", dup_path, len(dup_df))
        stats = qa.get("stats", {})
        if stats:
            stats_path = qa_dir / f"{QA_STATS_PREFIX}{path_id}.json"
            stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
            logger.info("Wrote QA %s", stats_path)

    all_path = out_dir / END_AUM_SERIES_ALL
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=out_dir, prefix="end_aum_all_")
    try:
        os.close(fd)
        combined_df.to_parquet(tmp, index=False)
        os.replace(tmp, all_path)
    except Exception:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%d rows)", all_path, len(combined_df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
