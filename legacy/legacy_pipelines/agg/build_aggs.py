"""
Step 2: Single deterministic aggregation builder. Materializes all agg tables from
curated metrics + optional dim joins, driven by configs/agg_policy.yml.

Usage:
  python -m pipelines.agg.build_aggs --policy configs/agg_policy.yml
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from legacy.legacy_pipelines.agg import meta_utils as agg_meta
from legacy.legacy_pipelines.agg.join_coverage import JoinCoverageError, check_join_coverage
from legacy.legacy_pipelines.agg.qa_aggs import AggQAError, validate_agg_qa
from legacy.legacy_pipelines.contracts.agg_policy_contract import (
    AggPolicy,
    load_and_validate_agg_policy,
    policy_hash,
    summarize_agg_policy,
)

logger = logging.getLogger(__name__)


class DimJoinError(Exception):
    """Raised when a dimension join would change row count or dim has duplicate keys."""

AGG_DIR = "agg"
QA_DIR = "qa"
FAIL_CONTEXT_FILENAME = "agg_build_fail_context.json"
DEFAULT_POLICY_PATH = "configs/agg_policy.yml"

# Recompute formulas: rate_name -> (numerator_col, denominator_col). Guards: denom <= 0 => NaN; inf => NaN.
RATE_RECOMPUTE_FORMULAS: dict[str, tuple[str, str]] = {
    "ogr": ("nnb", "begin_aum"),
    "market_impact_rate": ("market_pnl", "begin_aum"),
    "fee_yield": ("nnf", "nnb"),
}


def _output_name(grain_name: str, dim_keys: list[str]) -> str:
    """Map (grain_name, dim_keys) to output table basename (no extension)."""
    if grain_name == "firm_monthly" and not dim_keys:
        return "firm_monthly"
    if grain_name == "channel_monthly":
        return "channel_monthly_l2" if len(dim_keys) > 1 else "channel_monthly"
    if grain_name == "segment_monthly":
        return "segment_sub_segment" if len(dim_keys) > 1 else "segment_monthly"
    if grain_name == "ticker_monthly":
        return "ticker_monthly"
    if grain_name == "geo_monthly":
        return "geo_monthly"
    # fallback: grain_name + dims
    return f"{grain_name}_{'_'.join(dim_keys)}" if dim_keys else grain_name


def load_source_frame(policy: AggPolicy, root: Path | None = None) -> pd.DataFrame:
    """Load the source metrics frame from policy.source_table (relative to root or cwd)."""
    root = root or Path.cwd()
    path = root / policy.source_table
    if not path.exists():
        raise FileNotFoundError(f"Source table not found: {path}")
    return pd.read_parquet(path)


def preflight(df: pd.DataFrame, policy: AggPolicy) -> pd.DataFrame:
    """
    Verify source is valid for aggregation: time_key exists and is datetime;
    additive measures exist and are numeric (coerce safely or raise).
    Returns df with additive columns coerced to numeric; raises ValueError with clear message on failure.
    """
    time_key = policy.time_key
    if time_key not in df.columns:
        raise ValueError(f"Preflight: time_key {time_key!r} missing. Available: {sorted(df.columns)}")
    if not pd.api.types.is_datetime64_any_dtype(df[time_key]):
        raise ValueError(
            f"Preflight: time_key {time_key!r} must be datetime; got {df[time_key].dtype}. "
            "Convert the column to datetime before building aggs."
        )
    additive = policy.measures.additive
    missing = [c for c in additive if c not in df.columns]
    if missing:
        raise ValueError(f"Preflight: additive measures missing: {missing}. Available: {sorted(df.columns)}")
    out = df.copy()
    for col in additive:
        if not pd.api.types.is_numeric_dtype(out[col]):
            try:
                out[col] = pd.to_numeric(out[col], errors="raise")
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Preflight: additive column {col!r} is not numeric and could not be coerced: {e}"
                ) from e
    return out


def _dim_columns_required_by_grains(policy: AggPolicy) -> tuple[list[str], list[str]]:
    """Return (channel_cols, segment_cols) required by any grain. Column names from policy.dims."""
    dim_keys_used: set[str] = set()
    for _name, _grain, dim_list in _outputs_from_policy(policy):
        dim_keys_used.update(dim_list)
    channel_cols = [policy.dims[k] for k in ("channel_l1", "channel_l2") if k in policy.dims]
    segment_cols = [policy.dims[k] for k in ("segment", "sub_segment") if k in policy.dims]
    need_channel = any(k in dim_keys_used for k in ("channel_l1", "channel_l2"))
    need_segment = any(k in dim_keys_used for k in ("segment", "sub_segment"))
    return (
        channel_cols if need_channel else [],
        segment_cols if need_segment else [],
    )


def _channel_join_key(policy: AggPolicy) -> str:
    """Join key for dim_channel (e.g. preferred_label). Could be extended from policy later."""
    return "preferred_label"


def maybe_join_dims(
    df: pd.DataFrame,
    policy: AggPolicy,
    root: Path | None = None,
) -> pd.DataFrame:
    """
    Join dim tables only when a grain requires channel_l1/channel_l2 or segment/sub_segment and they are missing.
    - Channel: load curated/dim_channel.parquet, join on preferred_label (or configured key), bring only channel_l1/channel_l2.
    - Segment: load curated/dim_product.parquet, join on product_ticker, bring only segment/sub_segment.
    Fails if dim has duplicate keys (with sample) or if join would change row count.
    Logs which dims were joined.
    """
    root = root or Path.cwd()
    out = df.copy()
    n_before = len(out)
    channel_cols, segment_cols = _dim_columns_required_by_grains(policy)

    # ---- Channel: only if grains need channel and at least one channel col is missing ----
    need_channel_join = bool(channel_cols) and any(c not in out.columns for c in channel_cols)
    dim_channel_path = root / "curated" / "dim_channel.parquet"
    join_key_channel = _channel_join_key(policy)

    if need_channel_join and dim_channel_path.exists() and join_key_channel in out.columns:
        dim_channel = pd.read_parquet(dim_channel_path)
        bring = [c for c in channel_cols if c in dim_channel.columns]
        if bring and join_key_channel in dim_channel.columns:
            # Fail if dim has duplicate keys
            dupes = dim_channel[dim_channel.duplicated(subset=[join_key_channel], keep=False)]
            if not dupes.empty:
                sample = dupes.head(5)[[join_key_channel] + bring].to_dict() if bring else dupes.head(5).to_dict()
                raise DimJoinError(
                    f"dim_channel has duplicate keys on {join_key_channel!r}. "
                    f"Row count in dim: {len(dim_channel)}; duplicates: {len(dupes)}. "
                    f"Sample duplicate keys: {sample}"
                )
            use_dim = dim_channel[[join_key_channel] + bring].copy()
            out = out.merge(use_dim, on=join_key_channel, how="left")
            if len(out) != n_before:
                raise DimJoinError(
                    f"dim_channel join changed row count from {n_before} to {len(out)}. "
                    "Left join must preserve rows; dim may have duplicate keys or merge key mismatch."
                )
            logger.info("Joined dim_channel (preferred_label -> %s)", bring)
            n_before = len(out)

    # ---- Segment: only if grains need segment/sub_segment and at least one is missing ----
    need_segment_join = bool(segment_cols) and any(c not in out.columns for c in segment_cols)
    dim_product_path = root / "curated" / "dim_product.parquet"
    join_key_product = "product_ticker"

    if need_segment_join and dim_product_path.exists() and join_key_product in out.columns:
        dim_product = pd.read_parquet(dim_product_path)
        bring = [c for c in segment_cols if c in dim_product.columns]
        if bring and join_key_product in dim_product.columns:
            dupes = dim_product[dim_product.duplicated(subset=[join_key_product], keep=False)]
            if not dupes.empty:
                sample = dupes.head(5)[[join_key_product] + bring].to_dict() if bring else dupes.head(5).to_dict()
                raise DimJoinError(
                    f"dim_product has duplicate keys on {join_key_product!r}. "
                    f"Row count in dim: {len(dim_product)}; duplicates: {len(dupes)}. "
                    f"Sample duplicate keys: {sample}"
                )
            use_dim = dim_product[[join_key_product] + bring].copy()
            out = out.merge(use_dim, on=join_key_product, how="left")
            if len(out) != n_before:
                raise DimJoinError(
                    f"dim_product join changed row count from {n_before} to {len(out)}. "
                    "Left join must preserve rows; dim may have duplicate keys or merge key mismatch."
                )
            logger.info("Joined dim_product (product_ticker -> %s)", bring)
            n_before = len(out)

    return out


def normalize_null_dims(df: pd.DataFrame, policy: AggPolicy) -> pd.DataFrame:
    """Apply null_handling to dimension columns: DROP (dropna) or UNKNOWN (fill with unknown_label). Returns copy."""
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


def build_one_agg(
    df: pd.DataFrame,
    time_key: str,
    grain_dims: list[str],
    measures: dict,
    rollup: dict,
) -> pd.DataFrame:
    """
    Build one aggregation: group by [time_key] + grain_dims, sum additive, then apply rates (recompute or store_numerators).
    Strict column check before grouping. Result sorted by [time_key] + grain_dims (lexicographic, stable).
    measures: {"additive": list[str], "rates": list[str]}
    rollup: {"rates_method": "recompute"|"store_numerators"|..., "weights": dict}
    """
    additive = list(measures.get("additive") or [])
    rates = list(measures.get("rates") or [])
    rates_method = (rollup.get("rates_method") or "recompute").lower()
    group_cols = [time_key] + list(grain_dims)

    missing = [c for c in group_cols + additive if c not in df.columns]
    if missing:
        raise ValueError(f"build_one_agg: DataFrame missing required columns: {missing}")

    use = group_cols + additive
    agg_df = df[use].groupby(group_cols, dropna=False).sum().reset_index()

    if rates_method == "recompute" and rates:
        for rate_name in rates:
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

    # Column order: time_key, grain_dims, additive, then rate columns present
    rate_cols = [r for r in rates if r in agg_df.columns]
    final_cols = group_cols + additive + rate_cols
    agg_df = agg_df[[c for c in final_cols if c in agg_df.columns]]

    # Deterministic sort (lexicographic, stable)
    agg_df = agg_df.sort_values(by=group_cols, kind="stable").reset_index(drop=True)
    return agg_df


def write_parquet_atomic(df: pd.DataFrame, path: str | Path) -> None:
    """Write DataFrame to parquet atomically: write to <name>.parquet.tmp, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        try:
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_meta_json(meta: dict, path: str | Path) -> None:
    """Write meta dict to JSON atomically (write .tmp then replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        try:
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of column names + dtypes for meta (delegate to meta_utils for consistency)."""
    return agg_meta.hash_schema(df)


def write_agg_fail_context(
    root: Path,
    *,
    table_name: str | None = None,
    missing_columns: list[str] | None = None,
    sample_rows: list[dict] | None = None,
    dtypes: dict | None = None,
    policy_excerpt: dict | None = None,
) -> Path:
    """
    Write qa/agg_build_fail_context.json with failure diagnostics. Returns path written.
    """
    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / FAIL_CONTEXT_FILENAME
    ctx = {
        "table_name": table_name,
        "missing_columns": missing_columns or [],
        "sample_rows": sample_rows or [],
        "dtypes": dtypes or {},
        "policy_excerpt": policy_excerpt or {},
    }
    path.write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    logger.warning("Wrote fail context to %s", path)
    return path


def _missing_from_error(err: Exception) -> list[str]:
    """Extract missing column names from a typical ValueError message if possible."""
    msg = str(err).lower()
    if "missing" not in msg:
        return []
    out: list[str] = []
    for part in str(err).split(":"):
        if "missing" in part.lower() and "[" in part:
            m = re.search(r"\[([^\]]+)\]", part)
            if m:
                out.extend([x.strip().strip("'\"") for x in m.group(1).split(",")])
    return out


def _write_fail_context_and_reraise(
    root: Path,
    policy: AggPolicy,
    df: pd.DataFrame | None,
    *,
    table_name: str | None = None,
    missing_columns: list[str] | None = None,
    err: Exception | None = None,
) -> None:
    """Build fail context from df and policy, write to qa/, then re-raise the error."""
    missing_columns = missing_columns or (_missing_from_error(err) if err else [])
    sample_rows: list[dict] = []
    dtypes: dict = {}
    if df is not None and not df.empty:
        try:
            sample_rows = df.head(5).to_dict(orient="records")
            for row in sample_rows:
                for k, v in list(row.items()):
                    if hasattr(v, "isoformat"):
                        row[k] = v.isoformat() if v is not None else None
        except Exception:
            pass
        try:
            dtypes = {str(c): str(df.dtypes[c]) for c in df.columns}
        except Exception:
            pass
    try:
        policy_excerpt = summarize_agg_policy(policy)
    except Exception:
        policy_excerpt = {}
    write_agg_fail_context(
        root,
        table_name=table_name,
        missing_columns=missing_columns,
        sample_rows=sample_rows,
        dtypes=dtypes,
        policy_excerpt=policy_excerpt,
    )
    if err is not None:
        raise err
    raise RuntimeError("_write_fail_context_and_reraise called without err")


def _outputs_from_policy(policy: AggPolicy) -> list[tuple[str, str, list[str]]]:
    """Return list of (output_basename, grain_name, dim_keys) for each table."""
    out: list[tuple[str, str, list[str]]] = []
    for grain_name, dim_groups in policy.grains.items():
        for dim_list in dim_groups:
            name = _output_name(grain_name, dim_list)
            out.append((name, grain_name, dim_list))
    return out


def run(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    root: Path | None = None,
) -> None:
    """
    Load policy, load source, optionally join dims, normalize nulls, build each agg table, write parquet + meta.
    """
    root = root or Path.cwd()
    policy_path = Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    policy = load_and_validate_agg_policy(policy_path)
    current_policy_hash = policy_hash(policy)

    df = load_source_frame(policy, root)
    df = maybe_join_dims(df, policy, root)
    join_coverage_stats = check_join_coverage(df, policy_path, root)
    df = normalize_null_dims(df, policy)

    try:
        df = preflight(df, policy)
    except (ValueError, FileNotFoundError) as e:
        _write_fail_context_and_reraise(
            root, policy, df, table_name="preflight", missing_columns=_missing_from_error(e), err=e
        )

    time_key = policy.time_key
    additive = policy.measures.additive

    measures_dict = {
        "additive": policy.measures.additive,
        "rates": policy.measures.rates,
    }
    rollup_dict = {
        "rates_method": policy.rollup.rates_method,
        "weights": getattr(policy.rollup, "weights", {}),
    }

    agg_dir = root / AGG_DIR
    agg_dir.mkdir(parents=True, exist_ok=True)

    dataset_version = ""
    version_path = root / "data" / ".version.json"
    if version_path.exists():
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
            dataset_version = (data.get("dataset_version") or "").strip()
        except Exception:
            pass

    table_infos: list[dict] = []
    current_table: str | None = None
    try:
        for output_name, grain_name, dim_keys in _outputs_from_policy(policy):
            current_table = output_name
            parquet_path = agg_dir / f"{output_name}.parquet"
            meta_path = agg_dir / f"{output_name}.meta.json"
            rel_path = f"{AGG_DIR}/{output_name}.parquet"

            # Cache: skip recompute if meta exists and matches dataset_version + policy_hash
            cache_hit = False
            if meta_path.exists() and parquet_path.exists():
                try:
                    existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if (
                        existing_meta.get("dataset_version") == dataset_version
                        and existing_meta.get("policy_hash") == current_policy_hash
                    ):
                        cache_hit = True
                        rowcount = existing_meta.get("rowcount", 0)
                        min_month_end = existing_meta.get("min_month_end") or existing_meta.get("min_month")
                        max_month_end = existing_meta.get("max_month_end") or existing_meta.get("max_month")
                        columns = existing_meta.get("columns") or []
                        table_infos.append({
                            "name": output_name,
                            "path": rel_path,
                            "grain": grain_name,
                            "rowcount": rowcount,
                            "min_month_end": min_month_end,
                            "max_month_end": max_month_end,
                            "columns": columns,
                        })
                        month_range = f"{min_month_end or '?'} .. {max_month_end or '?'}"
                        logger.info("%s | %s | rows=%s | %s | cache_hit", output_name, grain_name, rowcount, month_range)
                except Exception:
                    pass

            if not cache_hit:
                dim_cols = [policy.dims[k] for k in dim_keys]
                missing_dims = [c for c in dim_cols if c not in df.columns]
                if missing_dims:
                    e = ValueError(f"Grain {grain_name!r} (output {output_name!r}) needs dim columns: {missing_dims}")
                    _write_fail_context_and_reraise(root, policy, df, table_name=output_name, missing_columns=missing_dims, err=e)

                agg_df = build_one_agg(df, time_key, dim_cols, measures_dict, rollup_dict)
                validate_agg_qa(
                    agg_df,
                    time_key,
                    dim_cols,
                    additive,
                    output_name,
                    policy,
                    root,
                    allow_nan_additive=False,
                )

                write_parquet_atomic(agg_df, parquet_path)

                time_col = time_key if time_key in agg_df.columns and len(agg_df.columns) > 0 else None
                min_month = str(agg_df[time_col].min()) if time_col and not agg_df.empty else None
                max_month = str(agg_df[time_col].max()) if time_col and not agg_df.empty else None
                min_month_end = min_month
                max_month_end = max_month
                created_at = datetime.now(timezone.utc).isoformat()
                group_cols = [time_key] + dim_cols
                key_unique = not agg_df.duplicated(subset=group_cols, keep=False).any()
                meta = {
                    "dataset_version": dataset_version,
                    "policy_hash": current_policy_hash,
                    "rowcount": len(agg_df),
                    "schema_hash": agg_meta.hash_schema(agg_df),
                    "grain_dims": dim_keys,
                    "grain": dim_keys,
                    "min_month": min_month,
                    "max_month": max_month,
                    "min_month_end": min_month_end,
                    "max_month_end": max_month_end,
                    "created_at": created_at,
                    "key_unique": key_unique,
                    "columns": agg_df.columns.tolist(),
                }
                join_coverage_summary = {
                    k: join_coverage_stats.get(k)
                    for k in (
                        "total_rows",
                        "pct_missing_channel_l1",
                        "pct_missing_segment",
                        "n_missing_channel_l1",
                        "n_missing_segment",
                    )
                    if k in join_coverage_stats
                }
                if join_coverage_summary:
                    meta["join_coverage"] = join_coverage_summary
                agg_meta.write_meta(meta_path, meta)

                table_infos.append({
                    "name": output_name,
                    "path": rel_path,
                    "grain": grain_name,
                    "rowcount": len(agg_df),
                    "min_month_end": min_month_end,
                    "max_month_end": max_month_end,
                    "columns": agg_df.columns.tolist(),
                })
                month_range = f"{min_month_end or '?'} .. {max_month_end or '?'}"
                logger.info("%s | %s | rows=%s | %s | written", output_name, grain_name, len(agg_df), month_range)

    except (AggQAError, JoinCoverageError):
        raise
    except (ValueError, FileNotFoundError, DimJoinError) as e:
        # Preserve missing_columns from an inner write (e.g. per-table failure already wrote context)
        missing = _missing_from_error(e)
        fail_path = root / QA_DIR / FAIL_CONTEXT_FILENAME
        if fail_path.exists() and not missing:
            try:
                existing = json.loads(fail_path.read_text(encoding="utf-8"))
                missing = existing.get("missing_columns") or []
            except Exception:
                pass
        _write_fail_context_and_reraise(
            root, policy, df, table_name=current_table or "unknown", missing_columns=missing or None, err=e
        )

    join_coverage_summary = {
        k: join_coverage_stats.get(k)
        for k in (
            "total_rows",
            "pct_missing_channel_l1",
            "pct_missing_segment",
            "n_missing_channel_l1",
            "n_missing_segment",
        )
        if k in join_coverage_stats
    }
    manifest = {
        "dataset_version": dataset_version,
        "policy_hash": current_policy_hash,
        "tables": table_infos,
    }
    if join_coverage_summary:
        manifest["join_coverage"] = join_coverage_summary
    manifest_path = agg_dir / "manifest.json"
    write_meta_json(manifest, manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build agg tables from curated metrics using agg_policy.yml")
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY_PATH,
        help="Path to agg_policy.yml (default: configs/agg_policy.yml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: cwd)",
    )
    args = parser.parse_args()
    run(policy_path=args.policy, root=args.root)


if __name__ == "__main__":
    main()
