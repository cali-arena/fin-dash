"""
Build aum_series from end_aum_series: add begin_aum (lag within slice), is_first_month_in_slice,
and slice diagnostics. Hard/soft QA gates; gap and single-month-slice detail in QA files.
Unified combined output: aum_series_all.parquet + meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.drill_paths_contract import load_drill_paths_config
from legacy.legacy_pipelines.slices.slice_keys import compute_slice_id, normalize_slice_value

logger = logging.getLogger(__name__)

DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"
INTERMEDIATE_DIR = "curated/intermediate"
QA_DIR = "qa"
END_AUM_SERIES_PREFIX = "end_aum_series__"
AUM_SERIES_PREFIX = "aum_series__"
AUM_SERIES_ALL = "aum_series_all.parquet"
AUM_SERIES_ALL_META = "aum_series_all.meta.json"
QA_SINGLE_MONTH_PREFIX = "single_month_slices__"
QA_MONTH_GAPS_PREFIX = "month_gaps__"

# Unified key columns for combined output (non-used keys null)
ALL_KEY_COLS = [
    "preferred_label",
    "product_ticker",
    "src_country_canonical",
    "product_country_canonical",
]

COL_MONTH_END = "month_end"
OUT_END_AUM = "end_aum"
OUT_BEGIN_AUM = "begin_aum"
IS_FIRST_MONTH_IN_SLICE = "is_first_month_in_slice"
MONTHS_IN_SLICE = "months_in_slice"
SLICE_MIN_MONTH_END = "slice_min_month_end"
SLICE_MAX_MONTH_END = "slice_max_month_end"


def _slice_key_human(path_id: str, key_values: dict[str, str]) -> str:
    """Human-readable slice_key: 'global' for no keys; else 'k=v | k=v' sorted by key."""
    if not key_values:
        return "global" if path_id == "global" else path_id
    return " | ".join(f"{k}={key_values[k]}" for k in sorted(key_values.keys()))


def _add_path_slice_ids(df: pd.DataFrame, path_id: str, keys: list[str]) -> pd.DataFrame:
    """Add path_id, slice_id, slice_key to df (which has key columns). Mutates and returns df."""
    df = df.copy()
    df["path_id"] = path_id
    slice_ids = []
    slice_keys = []
    for _, row in df.iterrows():
        key_values = {k: normalize_slice_value(row[k]) for k in keys}
        slice_ids.append(compute_slice_id(path_id, key_values))
        slice_keys.append(_slice_key_human(path_id, key_values))
    df["slice_id"] = slice_ids
    df["slice_key"] = slice_keys
    return df


def _allow_negative_aum() -> bool:
    """If True, only check end_aum numeric; else also require >= 0."""
    return os.environ.get("ALLOW_NEGATIVE_AUM", "").strip().lower() in ("1", "true", "yes")


def _month_end_add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """Add n calendar months; return month-end date."""
    return (ts.to_period("M") + n).to_timestamp(how="end")


def _gap_months_between(prev: pd.Timestamp, next_ts: pd.Timestamp) -> list[pd.Timestamp]:
    """Return list of month-end dates strictly between prev and next_ts (expected step 1 month)."""
    out = []
    cur = _month_end_add_months(prev, 1)
    while cur < next_ts:
        out.append(cur)
        cur = _month_end_add_months(cur, 1)
    return out


def _build_aum_series_for_path(
    series_path: Path,
    path_id: str,
    keys: list[str],
    qa_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load end_aum_series, run hard/soft gates, sort, compute begin_aum and slice diagnostics.
    Returns (out_df, single_month_df, gaps_df). Parquet output is lean; gap/single-month detail in QA.
    """
    if not series_path.exists():
        raise FileNotFoundError(
            f"end_aum_series not found: {series_path}. Run: python -m pipelines.slices.end_aum_series --build"
        )
    df = pd.read_parquet(series_path)

    if COL_MONTH_END not in df.columns or OUT_END_AUM not in df.columns:
        raise ValueError(
            f"end_aum_series {path_id} must have month_end and end_aum; got {list(df.columns)}"
        )
    grain_cols = [COL_MONTH_END] + keys
    for c in grain_cols:
        if c not in df.columns:
            raise ValueError(f"end_aum_series {path_id} missing column: {c}")

    # --- HARD: grain uniqueness ---
    if df.duplicated(subset=grain_cols).any():
        raise ValueError(
            f"end_aum_series {path_id}: grain (month_end + keys) must be unique"
        )

    # --- HARD: month_end non-null ---
    if df[COL_MONTH_END].isna().any():
        raise ValueError(f"end_aum_series {path_id}: month_end must be non-null")

    # --- HARD: end_aum numeric and (optionally) >= 0 ---
    if not pd.api.types.is_numeric_dtype(df[OUT_END_AUM]):
        raise ValueError(f"end_aum_series {path_id}: end_aum must be numeric")
    if not _allow_negative_aum() and (df[OUT_END_AUM] < 0).any():
        raise ValueError(f"end_aum_series {path_id}: end_aum must be >= 0 (set ALLOW_NEGATIVE_AUM=1 to allow)")

    df[COL_MONTH_END] = pd.to_datetime(df[COL_MONTH_END], utc=False)
    if hasattr(df[COL_MONTH_END].dtype, "tz") and df[COL_MONTH_END].dtype.tz is not None:
        df[COL_MONTH_END] = df[COL_MONTH_END].dt.tz_localize(None)
    df[COL_MONTH_END] = df[COL_MONTH_END].dt.normalize()

    sort_cols = keys + [COL_MONTH_END] if keys else [COL_MONTH_END]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # --- HARD: within each slice, month_end strictly increasing (no duplicates) ---
    if keys:
        diff = df.groupby(keys, sort=False)[COL_MONTH_END].diff()
    else:
        diff = df[COL_MONTH_END].diff()
    invalid = (diff.notna()) & (diff <= pd.Timedelta(0))
    if invalid.any():
        raise ValueError(
            f"end_aum_series {path_id}: within each slice month_end must be strictly increasing (duplicates or wrong order)"
        )

    if keys:
        df[OUT_BEGIN_AUM] = df.groupby(keys, sort=False)[OUT_END_AUM].shift(1)
    else:
        df[OUT_BEGIN_AUM] = df[OUT_END_AUM].shift(1)
    df[IS_FIRST_MONTH_IN_SLICE] = df[OUT_BEGIN_AUM].isna()

    # Slice diagnostics (repeated per row)
    if keys:
        size_df = df.groupby(keys, sort=False).size().reset_index(name=MONTHS_IN_SLICE)
        df = df.merge(size_df, on=keys, how="left", sort=False)
        min_max = df.groupby(keys, sort=False)[COL_MONTH_END].agg(["min", "max"]).reset_index()
        min_max = min_max.rename(columns={"min": SLICE_MIN_MONTH_END, "max": SLICE_MAX_MONTH_END})
        df = df.merge(min_max, on=keys, how="left", sort=False)
    else:
        n = len(df)
        df[MONTHS_IN_SLICE] = n
        df[SLICE_MIN_MONTH_END] = df[COL_MONTH_END].min()
        df[SLICE_MAX_MONTH_END] = df[COL_MONTH_END].max()

    # --- SOFT: single-month slices ---
    single_month = df[df[MONTHS_IN_SLICE] == 1]
    if not single_month.empty:
        logger.warning(
            "path %s: %d slice(s) have only one month (see qa/single_month_slices__%s.csv)",
            path_id, len(single_month), path_id,
        )
    single_month_df = single_month[keys + [COL_MONTH_END]].drop_duplicates().sort_values(keys + [COL_MONTH_END], kind="mergesort").reset_index(drop=True) if not single_month.empty else pd.DataFrame(columns=keys + [COL_MONTH_END])

    # --- SOFT: gap months ---
    gap_rows: list[dict[str, Any]] = []
    if keys:
        for _, grp in df.groupby(keys, sort=False):
            me = grp[COL_MONTH_END].sort_values().values
            for i in range(len(me) - 1):
                prev_ts = pd.Timestamp(me[i])
                next_ts = pd.Timestamp(me[i + 1])
                gap_list = _gap_months_between(prev_ts, next_ts)
                if not gap_list:
                    continue
                row = {k: grp[k].iloc[0] for k in keys}
                row["prev_month_end"] = prev_ts
                row["next_month_end"] = next_ts
                row["gap_count"] = len(gap_list)
                row["gap_months"] = ",".join(t.strftime("%Y-%m-%d") for t in gap_list)
                gap_rows.append(row)
    else:
        me = df[COL_MONTH_END].sort_values().values
        for i in range(len(me) - 1):
            prev_ts = pd.Timestamp(me[i])
            next_ts = pd.Timestamp(me[i + 1])
            gap_list = _gap_months_between(prev_ts, next_ts)
            if not gap_list:
                continue
            gap_rows.append({
                "prev_month_end": prev_ts,
                "next_month_end": next_ts,
                "gap_count": len(gap_list),
                "gap_months": ",".join(t.strftime("%Y-%m-%d") for t in gap_list),
            })
    gaps_df = pd.DataFrame(gap_rows)
    if not gaps_df.empty:
        logger.warning(
            "path %s: %d month gap(s) detected (see qa/month_gaps__%s.csv)",
            path_id, len(gaps_df), path_id,
        )

    out_cols = [COL_MONTH_END] + keys + [OUT_END_AUM, OUT_BEGIN_AUM, IS_FIRST_MONTH_IN_SLICE, MONTHS_IN_SLICE, SLICE_MIN_MONTH_END, SLICE_MAX_MONTH_END]
    out_df = df[out_cols].copy()
    return out_df, single_month_df, gaps_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build aum_series (begin_aum from end_aum_series) per path; write curated/intermediate/aum_series__{path_id}.parquet"
    )
    parser.add_argument("--build", action="store_true", help="Build and write aum_series parquet per path")
    parser.add_argument("--config", default=DEFAULT_DRILL_PATHS_CONFIG, help="Path to drill_paths.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = Path(args.root) if args.root is not None else Path.cwd()
    if not args.build:
        logger.info("Use --build to build aum_series outputs.")
        return 0

    drill_config = load_drill_paths_config(root / args.config)
    paths = [
        p
        for p in drill_config.get("drill_paths", [])
        if isinstance(p, dict) and p.get("enabled", True)
    ]
    if not paths:
        logger.error("No enabled drill_paths in config")
        return 1

    intermediate = root / INTERMEDIATE_DIR
    qa_dir = root / QA_DIR
    intermediate.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    path_dfs: list[pd.DataFrame] = []
    per_path_rowcounts = 0

    for p in paths:
        path_id = str(p.get("id", ""))
        keys = list(p.get("keys") or [])
        series_path = intermediate / f"{END_AUM_SERIES_PREFIX}{path_id}.parquet"
        try:
            out_df, single_month_df, gaps_df = _build_aum_series_for_path(series_path, path_id, keys, qa_dir)
        except (FileNotFoundError, ValueError) as e:
            logger.error("%s", e)
            return 1

        out_df = _add_path_slice_ids(out_df, path_id, keys)
        per_path_rowcounts += len(out_df)
        path_dfs.append((path_id, keys, out_df))

        out_path = intermediate / f"{AUM_SERIES_PREFIX}{path_id}.parquet"
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=intermediate, prefix="aum_series_")
        try:
            os.close(fd)
            out_df.to_parquet(tmp, index=False)
            os.replace(tmp, out_path)
        except Exception:
            if Path(tmp).exists():
                Path(tmp).unlink(missing_ok=True)
            raise
        logger.info("Wrote %s (%d rows)", out_path, len(out_df))

        if not single_month_df.empty:
            single_path = qa_dir / f"{QA_SINGLE_MONTH_PREFIX}{path_id}.csv"
            single_month_df.to_csv(single_path, index=False, date_format="%Y-%m-%d")
            logger.info("Wrote QA %s (%d rows)", single_path, len(single_month_df))
        if not gaps_df.empty:
            gaps_path = qa_dir / f"{QA_MONTH_GAPS_PREFIX}{path_id}.csv"
            gaps_df.to_csv(gaps_path, index=False, date_format="%Y-%m-%d")
            logger.info("Wrote QA %s (%d rows)", gaps_path, len(gaps_df))

    try:
        # Combined: unified key columns; non-used keys null
        combined_cols = (
            ["path_id", "slice_id", "slice_key", COL_MONTH_END]
            + ALL_KEY_COLS
            + [OUT_END_AUM, OUT_BEGIN_AUM, IS_FIRST_MONTH_IN_SLICE, MONTHS_IN_SLICE]
        )
        combined_parts = []
        for path_id, keys, out_df in path_dfs:
            part = out_df.copy()
            for k in ALL_KEY_COLS:
                if k not in part.columns:
                    part[k] = pd.NA
            combined_parts.append(part[combined_cols])
        combined_df = pd.concat(combined_parts, ignore_index=True)

        # Hard validation: (path_id, slice_id, month_end) unique; rowcount matches sum of per-path
        if combined_df.duplicated(subset=["path_id", "slice_id", COL_MONTH_END], keep=False).any():
            raise ValueError("aum_series_all: (path_id, slice_id, month_end) must be unique")
        if len(combined_df) != per_path_rowcounts:
            raise ValueError(
                f"aum_series_all: rowcount {len(combined_df)} must equal sum of per-path rowcounts {per_path_rowcounts}"
            )
    except ValueError as e:
        logger.error("%s", e)
        return 1

    all_path = intermediate / AUM_SERIES_ALL
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=intermediate, prefix="aum_series_all_")
    try:
        os.close(fd)
        combined_df.to_parquet(tmp, index=False)
        os.replace(tmp, all_path)
    except Exception:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%d rows)", all_path, len(combined_df))

    # meta.json: rowcount, schema_hash, dataset_version
    schema_parts = [f"{c}:{str(combined_df[c].dtype)}" for c in combined_df.columns]
    schema_hash = hashlib.sha1("|".join(sorted(schema_parts)).encode("utf-8")).hexdigest()
    dataset_version = hashlib.sha1(
        f"{schema_hash}{len(combined_df)}{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
    ).hexdigest()
    meta = {
        "row_count": len(combined_df),
        "schema_hash": schema_hash,
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path = intermediate / AUM_SERIES_ALL_META
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", meta_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
