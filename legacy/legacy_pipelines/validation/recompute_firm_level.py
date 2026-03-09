"""
Firm-level recomputation from metrics_monthly (global slice only).

Step 2: aggregate global slice by month_end; recompute firm-level rates using same
guard policy as metrics. Reads curated/metrics_monthly.parquet and configs/metrics_policy.yml.

Output: curated/qa/firm_level_recomputed.parquet + .meta.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.metrics.rate_policies import apply_begin_aum_guard, safe_divide

logger = logging.getLogger(__name__)

DEFAULT_METRICS_PATH = "curated/metrics_monthly.parquet"
OUTPUT_PARQUET = "curated/qa/firm_level_recomputed.parquet"
OUTPUT_META = "curated/qa/firm_level_recomputed.meta.json"
VERSION_MANIFEST_PATH = "data/.version.json"
CACHE_QA_SUBDIR = "qa"

# Default selector: path_id == "global"
DEFAULT_SELECTOR = {
    "mode": "path_id",
    "column": "path_id",
    "value": "global",
}


def load_metrics_monthly(path: str | Path = DEFAULT_METRICS_PATH) -> pd.DataFrame:
    """Load metrics_monthly from parquet. Returns empty DataFrame if file missing."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def filter_global_slice(df: pd.DataFrame, selector: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Filter to rows that represent the global/firm-level slice.

    selector:
      mode: "path_id"  -> column == value (default: column="path_id", value="global")
      mode: "flag"     -> column == value (e.g. column="is_global", value=True)
      mode: "null_keys" -> all keys in keys list are null/empty for global
    """
    if df.empty:
        return df
    sel = selector or DEFAULT_SELECTOR
    mode = (sel.get("mode") or "path_id").strip().lower()
    if mode == "path_id":
        col = sel.get("column") or "path_id"
        val = sel.get("value") if sel.get("value") is not None else "global"
        if col not in df.columns:
            raise ValueError(f"filter_global_slice: column {col!r} not in DataFrame")
        return df.loc[df[col] == val].copy()
    if mode == "flag":
        col = sel.get("column")
        if not col:
            raise ValueError("filter_global_slice: mode=flag requires selector.column")
        val = sel.get("value")
        if val is None:
            raise ValueError("filter_global_slice: mode=flag requires selector.value")
        if col not in df.columns:
            raise ValueError(f"filter_global_slice: column {col!r} not in DataFrame")
        return df.loc[df[col] == val].copy()
    if mode == "null_keys":
        keys = sel.get("keys")
        if not keys:
            raise ValueError("filter_global_slice: mode=null_keys requires selector.keys (list of column names)")
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise ValueError(f"filter_global_slice: keys {missing} not in DataFrame")
        # All listed keys null or empty string
        mask = pd.Series(True, index=df.index)
        for k in keys:
            mask &= df[k].isna() | (df[k].astype(str).str.strip() == "")
        return df.loc[mask].copy()
    raise ValueError(f"filter_global_slice: unknown mode {mode!r}. Use path_id, flag, or null_keys.")


def _coerce_inf_to_nan(series: pd.Series) -> pd.Series:
    """Replace +/-inf with NaN. Returns float64."""
    out = series.astype("float64")
    return out.replace([math.inf, -math.inf], float("nan"))


def recompute_firm_level(df_global: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """
    Aggregate by month_end and recompute firm-level rates.

    Per month_end:
      begin_aum_firm = sum(begin_aum), end_aum_firm = sum(end_aum),
      nnb_firm = sum(nnb), market_pnl_firm = sum(market_pnl)
    Rates (with guard: begin_aum_firm <= 0 => NaN):
      asset_growth_rate = (end_aum_firm - begin_aum_firm) / begin_aum_firm
      organic_growth_rate = nnb_firm / begin_aum_firm
      external_market_growth_rate = market_pnl_firm / begin_aum_firm
    inf/-inf replaced by NaN. Output sorted by month_end asc; rate columns float64.
    """
    required = ["month_end", "begin_aum", "end_aum", "nnb", "market_pnl"]
    missing = [c for c in required if c not in df_global.columns]
    if missing:
        raise ValueError(f"recompute_firm_level: required columns missing: {missing}")

    agg = df_global.groupby("month_end", as_index=False).agg(
        begin_aum_firm=("begin_aum", "sum"),
        end_aum_firm=("end_aum", "sum"),
        nnb_firm=("nnb", "sum"),
        market_pnl_firm=("market_pnl", "sum"),
        global_slice_rowcount_per_month=("begin_aum", "count"),
    )
    b = agg["begin_aum_firm"].astype("float64")
    e = agg["end_aum_firm"].astype("float64")
    nnb = agg["nnb_firm"].astype("float64")
    pnl = agg["market_pnl_firm"].astype("float64")

    begin_guard = (policy.get("policies") or {}).get("begin_aum_guard") or {}
    if not begin_guard:
        begin_guard = {"mode": "nan", "threshold": 0.0}

    asset_raw = safe_divide(e - b, b)
    ogr_raw = safe_divide(nnb, b)
    external_raw = safe_divide(pnl, b)

    asset = apply_begin_aum_guard(asset_raw, b, begin_guard)
    ogr = apply_begin_aum_guard(ogr_raw, b, begin_guard)
    external = apply_begin_aum_guard(external_raw, b, begin_guard)

    agg["asset_growth_rate"] = _coerce_inf_to_nan(asset)
    agg["organic_growth_rate"] = _coerce_inf_to_nan(ogr)
    agg["external_market_growth_rate"] = _coerce_inf_to_nan(external)
    agg["source"] = "recomputed_from_metrics_monthly"

    out = agg[
        [
            "month_end",
            "begin_aum_firm",
            "end_aum_firm",
            "nnb_firm",
            "market_pnl_firm",
            "asset_growth_rate",
            "organic_growth_rate",
            "external_market_growth_rate",
            "source",
            "global_slice_rowcount_per_month",
        ]
    ].copy()
    # month_end: timezone-naive datetime64[ns], aligned to month-end
    me = pd.to_datetime(out["month_end"])
    if me.dt.tz is not None:
        me = me.dt.tz_localize(None)
    out["month_end"] = me.dt.to_period("M").dt.to_timestamp("M").astype("datetime64[ns]")
    out = out.sort_values("month_end", kind="mergesort").reset_index(drop=True)
    for col in ["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]:
        out[col] = out[col].astype("float64")
    out["global_slice_rowcount_per_month"] = out["global_slice_rowcount_per_month"].astype("int64")
    # Uniqueness on month_end
    dupes = out["month_end"].duplicated(keep=False)
    if dupes.any():
        dup_months = out.loc[dupes, "month_end"].unique().tolist()
        raise ValueError(
            f"Firm-level output must be unique on month_end; found duplicates: {dup_months}. "
            f"Diagnostics: duplicated rows = {int(dupes.sum())}, unique month_end count = {out['month_end'].nunique()}."
        )
    return out


def _policy_hash(policy: dict[str, Any]) -> str:
    """Stable SHA-256 of canonical JSON of policy (for meta)."""
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_dataset_version(root: Path) -> str:
    """Read dataset_version from data/.version.json. Returns empty string if missing or invalid."""
    path = root / VERSION_MANIFEST_PATH
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("dataset_version") or "").strip()
    except Exception:
        return ""


def _write_firm_level_to_dir(
    df: pd.DataFrame,
    dir_path: Path,
    policy: dict[str, Any],
    dataset_version: str,
) -> None:
    """Write firm_level_recomputed.parquet and .meta.json into dir_path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    pq = dir_path / "firm_level_recomputed.parquet"
    meta_path = dir_path / "firm_level_recomputed.meta.json"
    df.to_parquet(pq, index=False)
    rowcount = len(df)
    min_month = str(pd.Timestamp(df["month_end"].min())) if not df.empty and "month_end" in df.columns else None
    max_month = str(pd.Timestamp(df["month_end"].max())) if not df.empty and "month_end" in df.columns else None
    meta = {
        "dataset_version": dataset_version,
        "policy_hash": _policy_hash(policy),
        "rowcount": rowcount,
        "min_month_end": min_month,
        "max_month_end": max_month,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d rows) and %s", pq, rowcount, meta_path)


def _copy_to_curated_qa(root: Path, parquet_src: Path, meta_src: Path) -> None:
    """Copy parquet and meta from src paths to curated/qa/ (latest convenience copy)."""
    qa_dir = root / "curated" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_src, qa_dir / "firm_level_recomputed.parquet")
    shutil.copy2(meta_src, qa_dir / "firm_level_recomputed.meta.json")
    logger.info("Copied firm_level_recomputed.* to %s", qa_dir)


def write_firm_level_output(
    df: pd.DataFrame,
    root: Path,
    policy: dict[str, Any],
    dataset_version: str = "",
) -> None:
    """Write curated/qa/firm_level_recomputed.parquet and .meta.json. Creates dirs if needed."""
    qa_dir = root / "curated" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = qa_dir / "firm_level_recomputed.parquet"
    meta_path = qa_dir / "firm_level_recomputed.meta.json"

    df.to_parquet(parquet_path, index=False)
    rowcount = len(df)
    if df.empty:
        min_month = None
        max_month = None
    else:
        min_month = str(pd.Timestamp(df["month_end"].min()))
        max_month = str(pd.Timestamp(df["month_end"].max()))
    meta = {
        "dataset_version": dataset_version,
        "policy_hash": _policy_hash(policy),
        "rowcount": rowcount,
        "min_month_end": min_month,
        "max_month_end": max_month,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d rows) and %s", parquet_path, rowcount, meta_path)


def main() -> int:
    import argparse
    from legacy.legacy_pipelines.contracts.metrics_policy_contract import load_metrics_policy, validate_metrics_policy

    parser = argparse.ArgumentParser(
        description="Recompute firm-level metrics from metrics_monthly (global slice). Writes curated/qa/firm_level_recomputed.*"
    )
    parser.add_argument("--metrics", default=DEFAULT_METRICS_PATH, help="Path to metrics_monthly.parquet")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--policy", default="configs/metrics_policy.yml", help="Path to metrics policy YAML")
    parser.add_argument(
        "--selector-mode",
        choices=["path_id", "flag", "null_keys"],
        default="path_id",
        help="Global slice selector mode",
    )
    parser.add_argument("--selector-column", default="path_id", help="Column for path_id or flag mode")
    parser.add_argument("--selector-value", default="global", help="Value for path_id or flag mode (flag: use true/false)")
    parser.add_argument("--selector-keys", default="", help="Comma-separated keys for null_keys mode")
    args = parser.parse_args()

    root = args.root
    metrics_path = root / args.metrics
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}", file=sys.stderr)
        return 1

    try:
        raw = load_metrics_policy(root / args.policy)
        policy = validate_metrics_policy(raw)
    except Exception as e:
        print(f"Failed to load policy: {e}", file=sys.stderr)
        return 1

    selector: dict[str, Any] = {"mode": args.selector_mode, "column": args.selector_column}
    if args.selector_mode == "path_id":
        selector["value"] = args.selector_value
    elif args.selector_mode == "flag":
        selector["value"] = args.selector_value.strip().lower() in ("true", "1", "yes")
    elif args.selector_mode == "null_keys":
        selector["keys"] = [k.strip() for k in args.selector_keys.split(",") if k.strip()]
        if not selector["keys"]:
            print("--selector-keys required when --selector-mode=null_keys", file=sys.stderr)
            return 1
    else:
        selector["value"] = args.selector_value

    current_policy_hash = _policy_hash(policy)
    dataset_version = _read_dataset_version(root)
    if dataset_version:
        cache_dir = root / "data" / "cache" / dataset_version / CACHE_QA_SUBDIR
        cache_pq = cache_dir / "firm_level_recomputed.parquet"
        cache_meta = cache_dir / "firm_level_recomputed.meta.json"
        if cache_pq.exists() and cache_meta.exists():
            try:
                existing_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                if existing_meta.get("policy_hash") == current_policy_hash:
                    _copy_to_curated_qa(root, cache_pq, cache_meta)
                    logger.info("cache hit firm_level_recomputed")
                    return 0
            except Exception:
                pass

    df = load_metrics_monthly(metrics_path)
    if df.empty:
        print("Metrics DataFrame is empty.", file=sys.stderr)
        return 1
    try:
        df_global = filter_global_slice(df, selector)
    except ValueError as e:
        print(f"filter_global_slice: {e}", file=sys.stderr)
        return 1
    if df_global.empty:
        print("No rows after filter_global_slice. Check selector (e.g. path_id=='global').", file=sys.stderr)
        return 1

    out = recompute_firm_level(df_global, policy)
    if not dataset_version:
        meta_path = root / "curated" / "metrics_monthly.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dataset_version = (meta.get("dataset_version") or "").strip()
            except Exception:
                pass
        if not dataset_version:
            dataset_version = _read_dataset_version(root)
    if dataset_version:
        cache_dir = root / "data" / "cache" / dataset_version / CACHE_QA_SUBDIR
        _write_firm_level_to_dir(out, cache_dir, policy, dataset_version)
        _copy_to_curated_qa(root, cache_dir / "firm_level_recomputed.parquet", cache_dir / "firm_level_recomputed.meta.json")
    else:
        write_firm_level_output(out, root, policy, dataset_version=dataset_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
