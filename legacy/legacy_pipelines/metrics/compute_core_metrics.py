"""
Core metrics: contract enforcement + deterministic rate computation from canonical metric_frame.
Input: curated/intermediate/metric_frame.parquet at (path_id, slice_id, month_end)
with begin_aum, end_aum, nnb, nnf already aligned to the slice grain.
Output: curated/metrics_monthly.parquet + meta + qa/metrics_policy_effects.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.metrics_policy_contract import (
    load_metrics_policy,
    validate_metrics_policy,
    write_policy_snapshot_if_requested,
)
from legacy.legacy_pipelines.metrics.metrics_policy_gate import run_gate as run_metrics_policy_gate
from legacy.legacy_pipelines.metrics.rate_policies import (
    safe_divide,
    apply_begin_aum_guard,
    apply_fee_yield_guard,
    apply_clamp,
    coerce_inf_to_nan,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS_POLICY_CONFIG = "configs/metrics_policy.yml"
METRIC_FRAME_PATH = "curated/intermediate/metric_frame.parquet"
CURATED_DIR = "curated"
QA_DIR = "qa"
METRICS_TABLE = "metrics_monthly"
METRICS_META_JSON = "metrics_monthly.meta.json"
QA_POLICY_EFFECTS_JSON = "metrics_policy_effects.json"

GRAIN_COLS = ["path_id", "slice_id", "month_end"]
COL_SLICE_KEY = "slice_key"
COL_BEGIN_AUM = "begin_aum"
COL_END_AUM = "end_aum"
COL_NNB = "nnb"
COL_NNF = "nnf"
COL_MARKET_PNL = "market_pnl"
COL_OGR = "ogr"
COL_MARKET_IMPACT_RATE = "market_impact_rate"
COL_TOTAL_GROWTH_RATE = "total_growth_rate"
COL_FEE_YIELD = "fee_yield"

RATE_COLS = [COL_OGR, COL_MARKET_IMPACT_RATE, COL_TOTAL_GROWTH_RATE, COL_FEE_YIELD]
CLAMP_FLAG_SUFFIX = "_clamped_flag"

REQUIRED_INPUT_COLS = GRAIN_COLS + [COL_SLICE_KEY, COL_BEGIN_AUM, COL_END_AUM, COL_NNB, COL_NNF]
QA_DUP_METRIC_FRAME_KEYS = "dup_metric_frame_keys.csv"


def _load_input_at_grain(root: Path) -> pd.DataFrame:
    """
    Load canonical metric_frame at (path_id, slice_id, month_end) with begin_aum, end_aum, nnb, nnf.
    Enforce required columns and uniqueness at the grain before policy application.
    """
    metric_path = root / METRIC_FRAME_PATH.replace("\\", "/").lstrip("/")
    if not metric_path.exists():
        raise FileNotFoundError(
            f"Metric frame not found: {metric_path}. Run pipelines.metrics.build_canonical_metric_frame first."
        )
    df = pd.read_parquet(metric_path)

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"metric_frame missing required column(s) for core metrics: {missing}. "
            f"Expected at least {REQUIRED_INPUT_COLS}."
        )

    dup_mask = df.duplicated(subset=GRAIN_COLS, keep=False)
    if dup_mask.any():
        dup_df = df.loc[dup_mask, GRAIN_COLS]
        dup_summary = (
            dup_df.groupby(GRAIN_COLS, dropna=False, sort=False)
            .size()
            .reset_index(name="row_count")
        )
        dup_summary = dup_summary.sort_values(
            ["row_count", "path_id", "slice_id", "month_end"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).head(200)
        qa_dir = root / QA_DIR
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / QA_DUP_METRIC_FRAME_KEYS
        dup_summary.to_csv(qa_path, index=False, date_format="%Y-%m-%d")
        total_dup = int(dup_mask.sum())
        raise ValueError(
            f"metric_frame must be unique at (path_id, slice_id, month_end); "
            f"found {total_dup} duplicate row(s). See {qa_path}"
        )

    return df


def _enforce_contract(df: pd.DataFrame, policy: dict[str, Any]) -> None:
    """Hard fail if required columns missing, grain not unique, or measure columns not numeric."""
    inputs_cfg = policy.get("inputs") or {}
    required = list(inputs_cfg.get("required_columns") or [])
    grain = list(inputs_cfg.get("grain_required") or [])
    for c in required + grain:
        if c not in df.columns:
            raise ValueError(f"Metrics input missing required column: {c}")
    if df.duplicated(subset=GRAIN_COLS, keep=False).any():
        n = int(df.duplicated(subset=GRAIN_COLS, keep=False).sum())
        raise ValueError(f"Metrics input grain (path_id, slice_id, month_end) must be unique; found {n} duplicate row(s)")
    for col in [COL_BEGIN_AUM, COL_END_AUM, COL_NNB, COL_NNF]:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Metrics input column {col} must be numeric; got {df[col].dtype}")


def _policy_hash(policy: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def compute_core_metrics(
    df: pd.DataFrame,
    policy: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute deterministic metrics with policy order: raw rates -> begin_aum_guard -> fee_yield_guard -> inf -> clamp.
    Returns (metrics_df, qa_effects) where qa_effects has guard NaN counts, inf->NaN counts, clamp flag counts.
    """
    policies = policy.get("policies") or {}
    begin_guard = policies.get("begin_aum_guard") or {}
    fee_guard = policies.get("fee_yield_guard") or {}
    clamp_policy = policies.get("clamp") or {}

    out = df[GRAIN_COLS + [COL_SLICE_KEY, COL_BEGIN_AUM, COL_END_AUM, COL_NNB, COL_NNF]].copy()

    begin_aum = out[COL_BEGIN_AUM]
    end_aum = out[COL_END_AUM]
    nnb = out[COL_NNB]
    nnf = out[COL_NNF]

    out[COL_MARKET_PNL] = (end_aum.astype(float) - begin_aum.astype(float) - nnb.astype(float))

    ogr_raw = safe_divide(nnb, begin_aum)
    market_impact_raw = safe_divide(out[COL_MARKET_PNL], begin_aum)
    total_growth_raw = safe_divide(end_aum - begin_aum, begin_aum)
    fee_yield_raw = safe_divide(nnf, nnb)

    ogr = apply_begin_aum_guard(ogr_raw, begin_aum, begin_guard)
    market_impact_rate = apply_begin_aum_guard(market_impact_raw, begin_aum, begin_guard)
    total_growth_rate = apply_begin_aum_guard(total_growth_raw, begin_aum, begin_guard)
    fee_yield = apply_fee_yield_guard(fee_yield_raw, nnb, fee_guard)

    out[COL_OGR] = ogr
    out[COL_MARKET_IMPACT_RATE] = market_impact_rate
    out[COL_TOTAL_GROWTH_RATE] = total_growth_rate
    out[COL_FEE_YIELD] = fee_yield

    qa_effects: dict[str, Any] = {
        "guard_nan_counts": {},
        "inf_to_nan_count": 0,
        "clamp_counts": {},
    }
    for name, before, after in [
        (COL_OGR, ogr_raw, ogr),
        (COL_MARKET_IMPACT_RATE, market_impact_raw, market_impact_rate),
        (COL_TOTAL_GROWTH_RATE, total_growth_raw, total_growth_rate),
    ]:
        qa_effects["guard_nan_counts"][name] = int((after.isna() & before.notna()).sum())
    qa_effects["guard_nan_counts"][COL_FEE_YIELD] = int((fee_yield.isna() & fee_yield_raw.notna()).sum())

    inf_count = 0
    for c in RATE_COLS:
        if c in out.columns:
            s = out[c]
            inf_count += int((s == math.inf).sum() + (s == -math.inf).sum())
    qa_effects["inf_to_nan_count"] = inf_count
    out = coerce_inf_to_nan(out, RATE_COLS)

    rate_out, ogr_flag = apply_clamp(out[COL_OGR], COL_OGR, clamp_policy)
    out[COL_OGR] = rate_out
    out[COL_OGR + CLAMP_FLAG_SUFFIX] = ogr_flag
    qa_effects["clamp_counts"][COL_OGR] = int(ogr_flag.sum())

    rate_out, mi_flag = apply_clamp(out[COL_MARKET_IMPACT_RATE], COL_MARKET_IMPACT_RATE, clamp_policy)
    out[COL_MARKET_IMPACT_RATE] = rate_out
    out[COL_MARKET_IMPACT_RATE + CLAMP_FLAG_SUFFIX] = mi_flag
    qa_effects["clamp_counts"][COL_MARKET_IMPACT_RATE] = int(mi_flag.sum())

    rate_out, tg_flag = apply_clamp(out[COL_TOTAL_GROWTH_RATE], COL_TOTAL_GROWTH_RATE, clamp_policy)
    out[COL_TOTAL_GROWTH_RATE] = rate_out
    out[COL_TOTAL_GROWTH_RATE + CLAMP_FLAG_SUFFIX] = tg_flag
    qa_effects["clamp_counts"][COL_TOTAL_GROWTH_RATE] = int(tg_flag.sum())

    rate_out, fy_flag = apply_clamp(out[COL_FEE_YIELD], COL_FEE_YIELD, clamp_policy)
    out[COL_FEE_YIELD] = rate_out
    out[COL_FEE_YIELD + CLAMP_FLAG_SUFFIX] = fy_flag
    qa_effects["clamp_counts"][COL_FEE_YIELD] = int(fy_flag.sum())

    output_cols = (
        GRAIN_COLS + [COL_SLICE_KEY, COL_BEGIN_AUM, COL_END_AUM, COL_NNB, COL_NNF, COL_MARKET_PNL]
        + RATE_COLS
        + [c + CLAMP_FLAG_SUFFIX for c in RATE_COLS]
    )
    out = out[[c for c in output_cols if c in out.columns]]
    return out, qa_effects


def run(root: Path, policy_config_path: Path) -> pd.DataFrame:
    """Load metric_frame, enforce contract, compute metrics, write parquet + meta + QA. Returns metrics DataFrame."""
    raw_policy = load_metrics_policy(policy_config_path)
    policy = validate_metrics_policy(raw_policy)
    write_policy_snapshot_if_requested(policy, root)

    df = _load_input_at_grain(root)
    _enforce_contract(df, policy)

    metrics_df, qa_effects = compute_core_metrics(df, policy)

    out_rel = f"{CURATED_DIR}/{METRICS_TABLE}.parquet"
    out_path = root / out_rel.replace("\\", "/").lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=out_path.parent, prefix="metrics_")
    try:
        os.close(fd)
        metrics_df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
    except Exception:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%d rows)", out_path, len(metrics_df))

    schema_parts = [f"{c}:{str(metrics_df[c].dtype)}" for c in metrics_df.columns]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    policy_hash_val = _policy_hash(policy)
    meta = {
        "row_count": len(metrics_df),
        "schema_hash": schema_hash,
        "dataset_version": f"core_metrics_{policy_hash_val}",
        "policy_hash": policy_hash_val,
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote %s", meta_path)

    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / QA_POLICY_EFFECTS_JSON
    qa_path.write_text(json.dumps(qa_effects, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote QA %s", qa_path)

    gate_report = run_metrics_policy_gate(root, metrics_path=out_path, policy_config_path=policy_config_path)
    gate_report_path = qa_dir / "metrics_policy_gate_report.json"
    gate_report_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True), encoding="utf-8")
    if not gate_report.get("passed", True):
        errors = []
        for k, v in gate_report.items():
            if isinstance(v, dict) and v.get("error"):
                errors.append(f"{k}: {v['error']}")
        if gate_report.get("recompute_check", {}).get("mismatch_count", 0) > 0:
            errors.append("recompute_check: mismatch(es) in spot-sample")
        raise ValueError("Metrics policy gate failed: " + "; ".join(errors))
    logger.info("Metrics policy gate passed. Wrote %s", gate_report_path)

    return metrics_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute core metrics with policy enforcement; write curated/metrics_monthly.")
    parser.add_argument("--run", action="store_true", help="Run compute and write outputs")
    parser.add_argument("--config", default=DEFAULT_METRICS_POLICY_CONFIG, help="Path to metrics_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    root = args.root or Path(__file__).resolve().parents[2]
    if not args.run:
        logger.info("Use --run to compute metrics and write outputs.")
        return 0
    try:
        run(root, root / args.config)
        return 0
    except (ValueError, FileNotFoundError) as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
