"""
Vectorized core metrics: deterministic math from metric_frame (no row loops).
Input: curated/intermediate/metric_frame.parquet; optional policy/dim_channel/meta.
Applies: ogr, market_pnl, market_impact_rate, fee_yield; guards + optional clamp + inf -> NaN for rate columns.
Output: curated/metrics_monthly.parquet + meta + QA gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.metrics_policy_contract import (
    load_metrics_policy,
    validate_metrics_policy,
    write_policy_snapshot_if_requested,
)
from legacy.legacy_pipelines.metrics.rate_policies import safe_divide, coerce_inf_to_nan, apply_clamp

logger = logging.getLogger(__name__)

METRIC_FRAME_PATH = "curated/intermediate/metric_frame.parquet"
METRIC_FRAME_META_PATH = "curated/intermediate/metric_frame.meta.json"
METRICS_POLICY_CONFIG = "configs/metrics_policy.yml"
DIM_CHANNEL_PATH = "curated/dim_channel.parquet"
CURATED_DIR = "curated"
METRICS_TABLE = "metrics_monthly"
METRICS_META_JSON = "metrics_monthly.meta.json"

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
    "market_pnl",
    "ogr",
    "market_impact_rate",
    "fee_yield",
    "guard_begin_aum",
    "guard_nnb",
    "ogr_clamped_flag",
    "market_impact_rate_clamped_flag",
    "fee_yield_clamped_flag",
    "channel_map_status",
    "channel_key",
    "dataset_version",
]
RATE_COLS = ["ogr", "market_impact_rate", "fee_yield"]


def _values_equal(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """NaN-safe scalar comparison with tolerance."""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def cache_key_for_metrics(meta: dict[str, Any]) -> str:
    """Build stable cache key from meta.json contents."""
    dv = meta.get("dataset_version", "")
    return f"metrics_monthly:{dv}"


def compute_metrics_vectorized(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    """
    Compute ogr, market_pnl, market_impact_rate, fee_yield with float64; then coerce inf -> NaN on rate cols.
    No guards; deterministic division (NaN propagates, inf replaced).
    """
    policies = policy.get("policies") or {}
    begin_guard_cfg = policies.get("begin_aum_guard") or {}
    fee_guard_cfg = policies.get("fee_yield_guard") or {}

    begin_mode = (begin_guard_cfg.get("mode") or "nan").strip().lower()
    if begin_mode not in ("nan", "zero"):
        begin_mode = "nan"
    begin_threshold = float(begin_guard_cfg.get("threshold", 0.0))

    fee_mode = (fee_guard_cfg.get("mode") or "nan").strip().lower()
    if fee_mode not in ("nan", "zero", "cap"):
        fee_mode = "nan"
    fee_threshold = float(fee_guard_cfg.get("threshold", 0.0))
    fee_cap_value = float(fee_guard_cfg.get("cap_value", 0.0))

    out = df.copy()
    b = out["begin_aum"].astype("float64")
    e = out["end_aum"].astype("float64")
    nnb = out["nnb"].astype("float64")
    nnf = out["nnf"].astype("float64")

    out["market_pnl"] = e - b - nnb

    ogr_raw = safe_divide(nnb, b)
    mir_raw = safe_divide(out["market_pnl"].astype("float64"), b)
    fee_raw = safe_divide(nnf, nnb)

    # 1) begin_aum guard
    guard_begin = (b <= begin_threshold) | b.isna()
    if begin_mode == "nan":
        ogr = ogr_raw.where(~guard_begin, float("nan"))
        mir = mir_raw.where(~guard_begin, float("nan"))
    else:  # zero
        ogr = ogr_raw.where(~guard_begin, 0.0)
        mir = mir_raw.where(~guard_begin, 0.0)

    # 2) nnb guard for fee_yield
    guard_nnb = (nnb <= fee_threshold) | nnb.isna()
    if fee_mode == "nan":
        fee = fee_raw.where(~guard_nnb, float("nan"))
    elif fee_mode == "zero":
        fee = fee_raw.where(~guard_nnb, 0.0)
    else:  # cap
        fee = fee_raw.where(~guard_nnb, fee_cap_value)

    out["ogr"] = ogr.astype("float64")
    out["market_impact_rate"] = mir.astype("float64")
    out["fee_yield"] = fee.astype("float64")

    # Optional clamp
    clamp_cfg = policies.get("clamp") or {}
    rate_out, ogr_flag = apply_clamp(out["ogr"], "ogr", clamp_cfg)
    out["ogr"] = rate_out
    out["ogr_clamped_flag"] = ogr_flag

    rate_out, mir_flag = apply_clamp(out["market_impact_rate"], "market_impact_rate", clamp_cfg)
    out["market_impact_rate"] = rate_out
    out["market_impact_rate_clamped_flag"] = mir_flag

    rate_out, fee_flag = apply_clamp(out["fee_yield"], "fee_yield", clamp_cfg)
    out["fee_yield"] = rate_out
    out["fee_yield_clamped_flag"] = fee_flag

    # Audit masks
    out["guard_begin_aum"] = guard_begin.astype(bool)
    out["guard_nnb"] = guard_nnb.astype(bool)

    # 3) Inf handling safety pass: +/-inf -> NaN for rate columns
    out = coerce_inf_to_nan(out, RATE_COLS)

    return out[[c for c in OUTPUT_COLUMNS_ORDER if c in out.columns]]


def run(root: Path) -> pd.DataFrame:
    """Load metric_frame (and optional meta), compute vectorized metrics, write parquet + meta."""
    frame_path = root / METRIC_FRAME_PATH.replace("\\", "/").lstrip("/")
    if not frame_path.exists():
        raise FileNotFoundError(
            f"Metric frame not found: {frame_path}. Run pipelines.metrics.build_canonical_metric_frame first."
        )
    df = pd.read_parquet(frame_path)

    required = ["path_id", "slice_id", "slice_key", "month_end", "begin_aum", "end_aum", "nnb", "nnf"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metric_frame missing required columns: {missing}")

    # Load and validate policy (guards + clamp config, even if we only use guards here)
    policy_path = root / METRICS_POLICY_CONFIG
    raw_policy = load_metrics_policy(policy_path)
    policy = validate_metrics_policy(raw_policy)
    write_policy_snapshot_if_requested(policy, root)

    out_df = compute_metrics_vectorized(df, policy)

    # QA gates BEFORE enrichment / write
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    # 1) Uniqueness gate
    grain = ["path_id", "slice_id", "month_end"]
    dup_mask = out_df.duplicated(subset=grain, keep=False)
    if dup_mask.any():
        dup_df = out_df.loc[dup_mask, grain]
        dup_summary = (
            dup_df.groupby(grain, dropna=False, sort=False)
            .size()
            .reset_index(name="row_count")
        )
        dup_summary = dup_summary.sort_values(
            ["row_count", "path_id", "slice_id", "month_end"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(200)
        dup_path = qa_dir / "dup_metrics_monthly_keys.csv"
        dup_summary.to_csv(dup_path, index=False, date_format="%Y-%m-%d")
        raise ValueError(
            f"metrics_monthly must be unique at (path_id, slice_id, month_end); "
            f"found {int(dup_mask.sum())} duplicate row(s). See {dup_path}"
        )

    # 2) No +/-inf gate
    inf_cols = ["market_pnl", "ogr", "market_impact_rate", "fee_yield"]
    inf_mask_any = pd.Series(False, index=out_df.index)
    for c in inf_cols:
        if c not in out_df.columns:
            continue
        s = out_df[c].astype("float64")
        inf_mask = (s == float("inf")) | (s == float("-inf"))
        inf_mask_any |= inf_mask
    if inf_mask_any.any():
        bad = out_df.loc[inf_mask_any, grain + inf_cols].head(200)
        inf_path = qa_dir / "inf_rows_metrics.csv"
        bad.to_csv(inf_path, index=False, date_format="%Y-%m-%d")
        raise ValueError(
            f"metrics_monthly has +/-inf in rate columns; {int(inf_mask_any.sum())} row(s) affected. See {inf_path}"
        )

    # 3) Guard compliance
    policies = policy.get("policies") or {}
    begin_guard_cfg = policies.get("begin_aum_guard") or {}
    fee_guard_cfg = policies.get("fee_yield_guard") or {}

    begin_mode = (begin_guard_cfg.get("mode") or "nan").strip().lower()
    if begin_mode not in ("nan", "zero"):
        begin_mode = "nan"
    fee_mode = (fee_guard_cfg.get("mode") or "nan").strip().lower()
    if fee_mode not in ("nan", "zero", "cap"):
        fee_mode = "nan"
    fee_cap_value = float(fee_guard_cfg.get("cap_value", 0.0))

    violations = []
    if "guard_begin_aum" in out_df.columns:
        gb = out_df["guard_begin_aum"].astype(bool)
        if begin_mode == "nan":
            bad = gb & ((~out_df["ogr"].isna()) | (~out_df["market_impact_rate"].isna()))
        else:  # zero
            bad = gb & (
                ((~out_df["ogr"].isna()) & (out_df["ogr"] != 0.0))
                | ((~out_df["market_impact_rate"].isna()) & (out_df["market_impact_rate"] != 0.0))
            )
        violations.append(bad)
    if "guard_nnb" in out_df.columns:
        gn = out_df["guard_nnb"].astype(bool)
        if fee_mode == "nan":
            bad_fee = gn & (~out_df["fee_yield"].isna())
        elif fee_mode == "zero":
            bad_fee = gn & ((~out_df["fee_yield"].isna()) & (out_df["fee_yield"] != 0.0))
        else:  # cap
            diff = out_df["fee_yield"].astype("float64") - fee_cap_value
            bad_fee = gn & ((~out_df["fee_yield"].isna()) & (diff.abs() > 1e-9))
        violations.append(bad_fee)
    if violations:
        bad_mask = violations[0].copy()
        for v in violations[1:]:
            bad_mask |= v
        if bad_mask.any():
            cols = grain + [
                "begin_aum",
                "end_aum",
                "nnb",
                "nnf",
                "guard_begin_aum",
                "guard_nnb",
                "ogr",
                "market_impact_rate",
                "fee_yield",
            ]
            cols = [c for c in cols if c in out_df.columns]
            bad_df = out_df.loc[bad_mask, cols].head(200)
            guard_path = qa_dir / "guard_policy_violations.csv"
            bad_df.to_csv(guard_path, index=False, date_format="%Y-%m-%d")
            raise ValueError(
                f"Guard policy violation in metrics_monthly; {int(bad_mask.sum())} row(s) non-compliant. See {guard_path}"
            )

    # 4) Spot-sampled recompute audit
    n_rows = len(out_df)
    if n_rows > 0:
        audit_cfg = policy.get("audit") or {}
        seed = int(audit_cfg.get("recompute_sample_seed", 42))
        sample_size = int(audit_cfg.get("recompute_sample_size", 200))
        size = min(sample_size, n_rows)
        rng = random.Random(seed)
        indices = rng.sample(range(n_rows), size)
        df_sample = df.iloc[indices].copy()
        recomputed = compute_metrics_vectorized(df_sample, policy)
        stored = out_df.iloc[indices]
        metrics_to_check = ["market_pnl", "ogr", "market_impact_rate", "fee_yield"]
        mismatch_mask = pd.Series(False, index=range(size))
        for m in metrics_to_check:
            if m not in stored.columns or m not in recomputed.columns:
                continue
            a = stored[m].reset_index(drop=True)
            b = recomputed[m].reset_index(drop=True)
            eq = a.combine(b, lambda x, y: _values_equal(x, y))
            mismatch_mask |= ~eq
        if mismatch_mask.any():
            bad_idx = mismatch_mask[mismatch_mask].index
            stored_bad = stored.reset_index(drop=True).loc[bad_idx, grain + metrics_to_check]
            recomputed_bad = (
                recomputed.reset_index(drop=True)
                .loc[bad_idx, metrics_to_check]
                .add_suffix("_recomputed")
            )
            mismatch_df = pd.concat([stored_bad.reset_index(drop=True), recomputed_bad.reset_index(drop=True)], axis=1)
            mismatch_path = qa_dir / "metric_recompute_mismatch.csv"
            mismatch_df.to_csv(mismatch_path, index=False, date_format="%Y-%m-%d")
            raise ValueError(
                f"Spot-sampled recompute mismatch in metrics_monthly; {int(mismatch_mask.sum())} row(s) differ. "
                f"See {mismatch_path}"
            )

    # 5) Identity QA: end_aum ≈ begin_aum + nnb + market_pnl
    if all(c in out_df.columns for c in ["begin_aum", "end_aum", "nnb", "market_pnl"]):
        begin_f = out_df["begin_aum"].astype("float64")
        end_f = out_df["end_aum"].astype("float64")
        nnb_f = out_df["nnb"].astype("float64")
        pnl_f = out_df["market_pnl"].astype("float64")

        resid = end_f - (begin_f + nnb_f + pnl_f)
        abs_resid = resid.abs()
        ref = end_f.abs().clip(lower=1.0)
        atol = 1e-6
        rtol = 1e-9
        tol = atol + rtol * ref
        viol_mask = abs_resid > tol

        violations_count = int(viol_mask.sum())
        total_rows = int(len(out_df))
        max_abs_resid = float(abs_resid.max()) if total_rows > 0 else 0.0
        max_rel_resid = float((abs_resid / ref).max()) if total_rows > 0 else 0.0

        summary = {
            "total_rows": total_rows,
            "violations_count": violations_count,
            "violations_rate": float(violations_count / total_rows) if total_rows > 0 else 0.0,
            "max_abs_resid": max_abs_resid,
            "max_rel_resid": max_rel_resid,
            "atol": atol,
            "rtol": rtol,
        }
        (qa_dir / "metrics_identity_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        if violations_count > 0:
            cols = [
                "path_id",
                "slice_id",
                "slice_key",
                "month_end",
                "begin_aum",
                "end_aum",
                "nnb",
                "market_pnl",
            ]
            cols = [c for c in cols if c in out_df.columns]
            viol_df = out_df.loc[viol_mask, cols].copy()
            viol_df["resid"] = resid[viol_mask]
            viol_df["abs_resid"] = abs_resid[viol_mask]
            viol_df["rel_resid"] = (abs_resid / ref)[viol_mask]
            viol_df = viol_df.sort_values(
                "abs_resid", ascending=False, kind="mergesort"
            ).head(2000)
            viol_path = qa_dir / "metrics_identity_violations.csv"
            viol_df.to_csv(viol_path, index=False, date_format="%Y-%m-%d")

            strict_env = os.getenv("METRICS_IDENTITY_STRICT", "true").strip().lower()
            strict = strict_env in ("1", "true", "yes")
            msg = (
                f"Identity check failed: end_aum ≈ begin_aum + nnb + market_pnl violated "
                f"for {violations_count} row(s). See {viol_path}"
            )
            if strict:
                raise ValueError(msg)
            logger.warning("METRICS_IDENTITY_STRICT=false: %s", msg)

    # Channel enrichment: channel_map_status (and optional channel_key)
    total_rows = len(out_df)
    if "preferred_label" in out_df.columns and total_rows > 0:
        dim_path = root / DIM_CHANNEL_PATH.replace("\\", "/").lstrip("/")
        if dim_path.exists():
            dim = pd.read_parquet(dim_path)
            cols = ["preferred_label"]
            if "channel_map_status" in dim.columns:
                cols.append("channel_map_status")
            if "channel_key" in dim.columns:
                cols.append("channel_key")
            dim_small = dim[cols].drop_duplicates(subset=["preferred_label"], keep="first")
            out_df = out_df.merge(dim_small, on="preferred_label", how="left", sort=False)
        if "channel_map_status" not in out_df.columns:
            out_df["channel_map_status"] = "UNKNOWN"
        else:
            mask_unknown = out_df["channel_map_status"].isna()
            if mask_unknown.any():
                out_df.loc[mask_unknown, "channel_map_status"] = "UNKNOWN"
        if "channel_key" not in out_df.columns:
            out_df["channel_key"] = pd.NA
    else:
        # No preferred_label: everything is UNKNOWN
        out_df["channel_map_status"] = "UNKNOWN"
        out_df["channel_key"] = pd.NA

    # Channel map coverage QA
    status = out_df["channel_map_status"].astype("object")
    unknown_mask = status.isna() | (status == "UNKNOWN")
    unknown_rows = int(unknown_mask.sum())
    unknown_rate = float(unknown_rows / total_rows) if total_rows > 0 else 0.0
    by_path: dict[str, Any] = {}
    if "path_id" in out_df.columns and total_rows > 0:
        grp = out_df.groupby("path_id", sort=False)
        for pid, g in grp:
            t = int(len(g))
            u = int((g["channel_map_status"].astype("object") == "UNKNOWN").sum())
            by_path[str(pid)] = {
                "total_rows": t,
                "unknown_rows": u,
                "unknown_rate": float(u / t) if t > 0 else 0.0,
            }
    channel_qa = {
        "total_rows": total_rows,
        "unknown_rows": unknown_rows,
        "unknown_rate": unknown_rate,
        "by_path_id": by_path,
    }
    (qa_dir / "channel_map_status_coverage.json").write_text(
        json.dumps(channel_qa, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Fill any missing columns from frame and enforce schema + types BEFORE dataset_version/meta/write
    for c in OUTPUT_COLUMNS_ORDER:
        if c not in out_df.columns:
            out_df[c] = df[c] if c in df.columns else pd.NA

    # Coerce dtypes consistently
    # month_end
    if "month_end" in out_df.columns:
        out_df["month_end"] = pd.to_datetime(out_df["month_end"], errors="coerce")
    # ids / strings
    id_cols = [
        "path_id",
        "slice_id",
        "slice_key",
        "preferred_label",
        "product_ticker",
        "src_country_canonical",
        "product_country_canonical",
        "channel_map_status",
        "channel_key",
    ]
    for c in id_cols:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype("string")
    # numeric float64
    num_cols = [
        "begin_aum",
        "end_aum",
        "nnb",
        "nnf",
        "market_pnl",
        "ogr",
        "market_impact_rate",
        "fee_yield",
    ]
    for c in num_cols:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype("float64")
    # guard / clamp flags boolean
    bool_cols = [
        "guard_begin_aum",
        "guard_nnb",
        "ogr_clamped_flag",
        "market_impact_rate_clamped_flag",
        "fee_yield_clamped_flag",
    ]
    for c in bool_cols:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype("bool")

    # Deterministic sort by (path_id, slice_id, month_end)
    sort_cols = ["path_id", "slice_id", "month_end"]
    sort_cols = [c for c in sort_cols if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    out_df = out_df[OUTPUT_COLUMNS_ORDER]

    # Dataset version and meta
    schema_parts = [f"{c}:{str(out_df[c].dtype)}" for c in out_df.columns]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()

    upstream_version = ""
    meta_path_in = root / METRIC_FRAME_META_PATH.replace("\\", "/").lstrip("/")
    if meta_path_in.exists():
        try:
            meta_in = json.loads(meta_path_in.read_text(encoding="utf-8"))
            upstream_version = meta_in.get("dataset_version") or meta_in.get("schema_hash", "") or ""
        except Exception:
            upstream_version = ""

    policy_hash = hashlib.sha1(json.dumps(raw_policy, sort_keys=True).encode("utf-8")).hexdigest()
    pipeline_version = os.getenv("PIPELINE_VERSION", "dev")
    payload = f"{upstream_version}|{policy_hash}|{pipeline_version}"
    dataset_version = hashlib.sha1(payload.encode("utf-8")).hexdigest()

    out_df["dataset_version"] = dataset_version

    # Final uniqueness enforcement on grain
    key_cols = ["path_id", "slice_id", "month_end"]
    key_unique_mask = out_df.duplicated(subset=key_cols, keep=False)
    if key_unique_mask.any():
        dup_df = out_df.loc[key_unique_mask, key_cols]
        dup_summary = (
            dup_df.groupby(key_cols, dropna=False, sort=False)
            .size()
            .reset_index(name="row_count")
        )
        dup_summary = dup_summary.sort_values(
            ["row_count", "path_id", "slice_id", "month_end"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(200)
        dup_path = qa_dir / "dup_metrics_monthly_keys_final.csv"
        dup_summary.to_csv(dup_path, index=False, date_format="%Y-%m-%d")
        raise ValueError(
            f"Final metrics_monthly must be unique at (path_id, slice_id, month_end); "
            f"found {int(key_unique_mask.sum())} duplicate row(s). See {dup_path}"
        )
    key_unique = not key_unique_mask.any()

    meta = {
        "row_count": int(len(out_df)),
        "schema_hash": schema_hash,
        "dataset_version": dataset_version,
        "policy_hash": policy_hash,
        "primary_key": key_cols,
        "key_unique": bool(key_unique),
    }

    # Reproducibility / defensibility report (write BEFORE parquet)
    # NaN counts per metric
    nan_counts = {}
    for m in ["ogr", "market_impact_rate", "fee_yield"]:
        if m in out_df.columns:
            nan_counts[m] = int(out_df[m].isna().sum())
    # Guard counts
    guard_counts = {
        "guard_begin_aum_true": int(out_df.get("guard_begin_aum", pd.Series(False)).astype(bool).sum())
        if "guard_begin_aum" in out_df.columns
        else 0,
        "guard_nnb_true": int(out_df.get("guard_nnb", pd.Series(False)).astype(bool).sum())
        if "guard_nnb" in out_df.columns
        else 0,
    }
    # Clamp flag counts
    clamp_counts = {}
    for m, col in [
        ("ogr", "ogr_clamped_flag"),
        ("market_impact_rate", "market_impact_rate_clamped_flag"),
        ("fee_yield", "fee_yield_clamped_flag"),
    ]:
        if col in out_df.columns:
            clamp_counts[m] = int(out_df[col].astype(bool).sum())
        else:
            clamp_counts[m] = 0
    # Unknown channel_map_status count
    unknown_channel = 0
    if "channel_map_status" in out_df.columns:
        s = out_df["channel_map_status"].astype("string")
        unknown_channel = int((s.isna()) | (s == "UNKNOWN")).sum()

    # Month range
    min_month = None
    max_month = None
    if "month_end" in out_df.columns and len(out_df) > 0:
        min_month = out_df["month_end"].min()
        max_month = out_df["month_end"].max()

    repro = {
        "dataset_version": dataset_version,
        "policy_hash": policy_hash,
        "schema_hash": schema_hash,
        "row_count": int(len(out_df)),
        "min_month_end": min_month.isoformat() if hasattr(min_month, "isoformat") else None,
        "max_month_end": max_month.isoformat() if hasattr(max_month, "isoformat") else None,
        "nan_counts": nan_counts,
        "guard_counts": guard_counts,
        "clamp_counts": clamp_counts,
        "unknown_channel_status_count": int(unknown_channel),
        "cache_key": cache_key_for_metrics(meta),
    }

    # Atomic write helpers: curated/.tmp and qa/.tmp
    curated_dir = root / CURATED_DIR.replace("\\", "/").lstrip("/")
    curated_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = curated_dir / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_path = curated_dir / f"{METRICS_TABLE}.parquet"
    tmp_path = tmp_dir / f"{METRICS_TABLE}.parquet"

    qa_tmp_dir = qa_dir / ".tmp"
    qa_tmp_dir.mkdir(parents=True, exist_ok=True)

    # Atomic write meta.json
    meta_path = final_path.with_suffix(".meta.json")
    meta_tmp = tmp_dir / METRICS_META_JSON
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(meta_tmp, meta_path)

    # Atomic write repro report
    repro_path = qa_dir / "metrics_repro_report.json"
    repro_tmp = qa_tmp_dir / "metrics_repro_report.json"
    repro_tmp.write_text(json.dumps(repro, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(repro_tmp, repro_path)

    # Atomic write parquet
    out_df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, final_path)

    # Log summary line
    if "month_end" in out_df.columns and len(out_df) > 0:
        min_month = out_df["month_end"].min()
        max_month = out_df["month_end"].max()
        logger.info(
            "metrics_monthly written: rows=%d, months=[%s..%s], dataset_version=%s",
            len(out_df),
            str(min_month.date()) if hasattr(min_month, "date") else str(min_month),
            str(max_month.date()) if hasattr(max_month, "date") else str(max_month),
            dataset_version,
        )
    else:
        logger.info(
            "metrics_monthly written: rows=%d, months=[N/A], dataset_version=%s",
            len(out_df),
            dataset_version,
        )

    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute metrics vectorized from metric_frame; write curated/metrics_monthly."
    )
    parser.add_argument("--run", action="store_true", help="Run compute and write outputs")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    root = args.root or Path(__file__).resolve().parents[2]
    if not args.run:
        logger.info("Use --run to compute and write %s/%s", CURATED_DIR, METRICS_TABLE + ".parquet")
        return 0
    try:
        run(root)
        return 0
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
