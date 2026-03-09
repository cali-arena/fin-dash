"""
Materialize agg tables from source (e.g. metrics_monthly) with cache discipline.

Engine: load_metrics_frame, apply_null_handling, aggregate_table (sum additive + recompute/weighted_avg rates),
write_agg. Cache: agg/<name>.parquet + agg/<name>.meta.json. Skip when meta matches dataset_version + policy_hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

import pandas as pd

from legacy.legacy_pipelines.contracts.agg_policy_contract import (
    AggPolicy,
    AggPolicyError,
    load_and_validate_agg_policy,
    policy_hash,
)

logger = logging.getLogger(__name__)

VERSION_PATH = "data/.version.json"
AGG_DIR = "agg"

# Recompute rate formulas: rate_name -> (numerator_col, denominator_col). Guard: denom <= 0 => NaN; inf => NaN.
RATE_RECOMPUTE_FORMULAS: dict[str, tuple[str, str]] = {
    "ogr": ("nnb", "begin_aum"),
    "market_impact_rate": ("market_pnl", "begin_aum"),
    "fee_yield": ("nnf", "nnb"),
}


def load_metrics_frame(path: str | Path) -> pd.DataFrame:
    """Load metrics source as DataFrame (parquet)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics frame not found: {path}")
    return pd.read_parquet(path)


def apply_null_handling(df: pd.DataFrame, policy: AggPolicy) -> pd.DataFrame:
    """
    Apply null_handling to dimension columns. DROP: drop rows with null in any dim col.
    UNKNOWN: fill null dims with unknown_label. Returns a copy.
    """
    out = df.copy()
    dim_cols = [c for c in policy.dims.values() if c in out.columns]
    if not dim_cols:
        return out
    if policy.null_handling.strategy == "DROP":
        out = out.dropna(subset=dim_cols)
    elif policy.null_handling.strategy == "UNKNOWN":
        label = policy.null_handling.unknown_label
        for c in dim_cols:
            out[c] = out[c].fillna(label)
    return out


def aggregate_table(
    df: pd.DataFrame,
    grain_dims: list[str],
    policy: AggPolicy,
) -> pd.DataFrame:
    """
    Group by [time_key] + grain_dims; sum additive; then compute rates (recompute or weighted_avg).
    Returns columns: time_key, grain dims, additive measures, rate columns (if any). Vectorized.
    """
    time_key = policy.time_key
    dim_cols = [policy.dims[k] for k in grain_dims]
    for c in dim_cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame missing dimension column {c!r}. grain_dims={grain_dims}")
    group_cols = [time_key] + dim_cols
    additive = [m for m in policy.measures.additive if m in df.columns]
    if not additive:
        raise ValueError(f"DataFrame has none of additive measures {policy.measures.additive!r}")
    use = group_cols + additive
    missing = [c for c in use if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    agg_df = df[use].groupby(group_cols, dropna=False).sum().reset_index()

    if policy.measures.rates and policy.rollup.rates_method == "recompute":
        for rate_name in policy.measures.rates:
            if rate_name not in RATE_RECOMPUTE_FORMULAS:
                continue
            num_col, denom_col = RATE_RECOMPUTE_FORMULAS[rate_name]
            if num_col not in agg_df.columns or denom_col not in agg_df.columns:
                continue
            num = agg_df[num_col].astype("float64")
            denom = agg_df[denom_col].astype("float64")
            out_rate = num / denom.replace(0, float("nan"))
            out_rate = out_rate.where(denom > 0, float("nan"))
            out_rate = out_rate.replace([math.inf, -math.inf], float("nan"))
            agg_df[rate_name] = out_rate

    elif policy.measures.rates and policy.rollup.rates_method == "weighted_avg":
        work = df.copy()
        for rate_name in policy.measures.rates:
            weight_col = policy.rollup.weights.get(rate_name)
            if not weight_col or rate_name not in df.columns or weight_col not in df.columns:
                continue
            work["_rw"] = work[rate_name].astype("float64") * work[weight_col].astype("float64")
            g = work.groupby(group_cols, dropna=False)
            sum_rw = g["_rw"].sum()
            sum_w = g[weight_col].sum()
            ser = (sum_rw / sum_w.replace(0, float("nan"))).replace([math.inf, -math.inf], float("nan"))
            wavg = ser.reset_index(name=rate_name)
            agg_df = agg_df.merge(wavg, on=group_cols, how="left")
            work = work.drop(columns=["_rw"])

    # Column order: time_key, grain dims, additive, rates
    rate_cols = [r for r in policy.measures.rates if r in agg_df.columns]
    final_cols = group_cols + additive + rate_cols
    agg_df = agg_df[[c for c in final_cols if c in agg_df.columns]]
    return agg_df


def _schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of column names for meta.json."""
    key = "|".join(sorted(df.columns))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def write_agg(
    df_agg: pd.DataFrame,
    name: str,
    dataset_version: str,
    policy_hash_val: str,
    out_dir: Path,
    grain_dims: list[str],
) -> None:
    """
    Write agg table to out_dir/<name>.parquet and out_dir/<name>.meta.json atomically.
    meta.json: dataset_version, policy_hash, rowcount, schema_hash, grain, min_month, max_month.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_key = df_agg.columns[0] if len(df_agg.columns) > 0 else None
    parquet_path = out_dir / f"{name}.parquet"
    meta_path = out_dir / f"{name}.meta.json"
    pq_tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    try:
        df_agg.to_parquet(pq_tmp, index=False)
        os.replace(pq_tmp, parquet_path)
    finally:
        if pq_tmp.exists():
            pq_tmp.unlink(missing_ok=True)
    rowcount = len(df_agg)
    min_month = str(df_agg[time_key].min()) if time_key and not df_agg.empty else None
    max_month = str(df_agg[time_key].max()) if time_key and not df_agg.empty else None
    meta = {
        "dataset_version": dataset_version,
        "policy_hash": policy_hash_val,
        "rowcount": rowcount,
        "schema_hash": _schema_hash(df_agg),
        "grain": grain_dims,
        "min_month": min_month,
        "max_month": max_month,
    }
    if time_key:
        meta["min_month_end"] = min_month
        meta["max_month_end"] = max_month
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, meta_path)
    logger.info("Wrote %s (%d rows) and %s", parquet_path, rowcount, meta_path)


def _read_dataset_version(root: Path) -> str:
    """Read dataset_version from data/.version.json. Returns empty string if missing or invalid."""
    path = root / VERSION_PATH
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("dataset_version") or "").strip()
    except Exception:
        return ""


def _outputs_from_policy(policy: AggPolicy) -> list[tuple[str, str, list[str]]]:
    """Return list of (output_name, grain_name, dim_keys) for each materialization."""
    out: list[tuple[str, str, list[str]]] = []
    for grain_name, dim_groups in policy.grains.items():
        for dim_list in dim_groups:
            if not dim_list:
                name = grain_name
            else:
                name = f"{grain_name}_{'_'.join(dim_list)}"
            out.append((name, grain_name, dim_list))
    return out


def _verify_source_columns(df: pd.DataFrame, policy: AggPolicy) -> None:
    """
    Verify source has time_key, additive measures, and all dim columns referenced by grains.
    Raise ValueError with missing columns, available columns, and which grain needs them.
    """
    available = set(df.columns)
    required_cols = {policy.time_key}
    required_cols.update(policy.measures.additive)
    dim_keys_used: set[str] = set()
    for _name, _grain, dim_list in _outputs_from_policy(policy):
        dim_keys_used.update(dim_list)
    for k in dim_keys_used:
        required_cols.add(policy.dims[k])
    missing = required_cols - available
    if not missing:
        return
    by_grain: list[str] = []
    for name, grain_name, dim_list in _outputs_from_policy(policy):
        need = [policy.time_key] + [policy.dims[k] for k in dim_list] + policy.measures.additive
        miss = [c for c in need if c not in available]
        if miss:
            by_grain.append(f"  grain {grain_name!r} (output {name!r}) needs: {sorted(miss)}")
    msg = (
        f"Source table missing required columns. missing={sorted(missing)}; "
        f"available={sorted(available)}. "
        + ((" Per grain: " + "; ".join(by_grain)) if by_grain else "")
    )
    raise ValueError(msg)


def _month_range_str(df: pd.DataFrame, time_key: str) -> str:
    if df.empty or time_key not in df.columns:
        return "N/A"
    mn = pd.Timestamp(df[time_key].min())
    mx = pd.Timestamp(df[time_key].max())
    return f"{mn.strftime('%Y-%m-%d')} .. {mx.strftime('%Y-%m-%d')}"


def materialize_aggs(
    policy_path: str | Path = "configs/agg_policy.yml",
    root: Path | None = None,
    *,
    source_path: str | Path | None = None,
) -> dict[str, str]:
    """
    Materialize all agg outputs. Uses cache when agg/<name>.parquet and agg/<name>.meta.json
    match dataset_version + policy_hash. Returns dict of output_name -> "cache_hit" | "written".
    """
    root = root or Path.cwd()
    root = Path(root)
    policy_path = root / policy_path if not Path(policy_path).is_absolute() else Path(policy_path)
    policy = load_and_validate_agg_policy(policy_path)
    current_policy_hash = policy_hash(policy)
    dataset_version = _read_dataset_version(root)
    if not dataset_version:
        logger.warning("No dataset_version in %s; cache will not be used", root / VERSION_PATH)

    source = source_path or (root / policy.source_table)
    source = root / source if not Path(str(source)).is_absolute() else Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source table not found: {source}")

    df_source = load_metrics_frame(source)
    _verify_source_columns(df_source, policy)
    df_clean = apply_null_handling(df_source, policy)

    agg_dir = root / AGG_DIR
    agg_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    table_infos: list[dict] = []
    time_key = policy.time_key

    for name, grain_name, dim_list in _outputs_from_policy(policy):
        parquet_path = agg_dir / f"{name}.parquet"
        meta_path = agg_dir / f"{name}.meta.json"
        cache_hit = False
        rowcount = 0
        if parquet_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("dataset_version") == dataset_version and meta.get("policy_hash") == current_policy_hash:
                    cache_hit = True
                    results[name] = "cache_hit"
                    rowcount = meta.get("rowcount", 0)
                    mn = meta.get("min_month_end") or meta.get("min_month") or ""
                    mx = meta.get("max_month_end") or meta.get("max_month") or ""
                    month_range = f"{mn} .. {mx}" if mn and mx else "N/A"
                    logger.info("cache hit %s: rows=%s month_range=%s", name, rowcount, month_range)
            except Exception:
                pass
        if not cache_hit:
            out_df = aggregate_table(df_clean, dim_list, policy)
            rowcount = len(out_df)
            month_range = _month_range_str(out_df, time_key)
            write_agg(out_df, name, dataset_version, current_policy_hash, agg_dir, dim_list)
            results[name] = "written"
            logger.info("written %s: rows=%s month_range=%s", name, rowcount, month_range)

        rel_path = f"{AGG_DIR}/{name}.parquet"
        table_infos.append({
            "name": name,
            "path": rel_path,
            "grain": grain_name,
            "measures": policy.measures.additive,
            "dims_used": dim_list,
            "rowcount": rowcount,
        })

    manifest = {
        "dataset_version": dataset_version,
        "policy_hash": current_policy_hash,
        "tables": table_infos,
    }
    manifest_path = agg_dir / "manifest.json"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    logger.info("Wrote %s", manifest_path)

    return results


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Materialize agg tables from source with cache (dataset_version + policy_hash). Skip if up-to-date."
    )
    parser.add_argument(
        "--policy",
        default="configs/agg_policy.yml",
        help="Path to agg policy YAML (default: configs/agg_policy.yml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Override source table path (default: from policy)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    try:
        results = materialize_aggs(
            policy_path=args.policy,
            root=args.root,
            source_path=args.source,
        )
    except (AggPolicyError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    hits = sum(1 for v in results.values() if v == "cache_hit")
    writes = sum(1 for v in results.values() if v == "written")
    elapsed = time.perf_counter() - t0
    print("--- materialize_aggs ---")
    print(f"  cache hit: {hits}  written: {writes}  total: {len(results)}")
    print(f"  runtime: {elapsed:.2f}s")
    print("---")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
