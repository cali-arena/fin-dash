"""
Compliance gate: prove metrics_monthly was built with policy applied consistently.
Validates invariants (no inf, guards, clamp) and spot-sample recompute; writes qa/metrics_policy_gate_report.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.metrics_policy_contract import load_metrics_policy, validate_metrics_policy
from legacy.legacy_pipelines.metrics.rate_policies import (
    safe_divide,
    apply_begin_aum_guard,
    apply_fee_yield_guard,
    apply_clamp,
    coerce_inf_to_nan,
)

logger = logging.getLogger(__name__)

DEFAULT_POLICY_CONFIG = "configs/metrics_policy.yml"
METRICS_PARQUET = "curated/metrics_monthly.parquet"
QA_DIR = "qa"
GATE_REPORT_JSON = "metrics_policy_gate_report.json"

GRAIN_COLS = ["path_id", "slice_id", "month_end"]
COL_BEGIN_AUM = "begin_aum"
COL_END_AUM = "end_aum"
COL_NNB = "nnb"
COL_NNF = "nnf"
COL_MARKET_PNL = "market_pnl"
COL_OGR = "ogr"
COL_MARKET_IMPACT_RATE = "market_impact_rate"
COL_TOTAL_GROWTH_RATE = "total_growth_rate"
COL_FEE_YIELD = "fee_yield"
CLAMP_FLAG_SUFFIX = "_clamped_flag"

METRIC_COLUMNS = [COL_MARKET_PNL, COL_OGR, COL_MARKET_IMPACT_RATE, COL_TOTAL_GROWTH_RATE, COL_FEE_YIELD]
RATE_COLS = [COL_OGR, COL_MARKET_IMPACT_RATE, COL_TOTAL_GROWTH_RATE, COL_FEE_YIELD]
BEGIN_AUM_GUARDED_RATES = [COL_OGR, COL_MARKET_IMPACT_RATE, COL_TOTAL_GROWTH_RATE]

RECOMPUTE_SAMPLE_SIZE = 200
RECOMPUTE_SEED = 42
TOLERANCE = 1e-9


def _values_equal(a: float | Any, b: float | Any, tol: float = TOLERANCE) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


def _check_no_inf(df: pd.DataFrame, report: dict[str, Any]) -> None:
    """A) No +/-inf in any metric column. Hard fail if any found."""
    inv = report.setdefault("invariant_inf", {})
    for c in METRIC_COLUMNS:
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        inf_count = int((s == math.inf).sum() + (s == -math.inf).sum())
        inv[c] = inf_count
        if inf_count > 0:
            inv["passed"] = False
            inv["error"] = f"column {c} contains {inf_count} +/-inf value(s); policy requires inf -> NaN."
            return
    inv["passed"] = True


def _check_begin_aum_guard(df: pd.DataFrame, policy: dict[str, Any], report: dict[str, Any]) -> None:
    """B) Where begin_aum <= threshold, rates must be NaN (mode nan) or 0.0 (mode zero)."""
    begin_guard = (policy.get("policies") or {}).get("begin_aum_guard") or {}
    mode = (begin_guard.get("mode") or "nan").strip().lower()
    if mode not in ("nan", "zero"):
        mode = "nan"
    threshold = float(begin_guard.get("threshold", 0.0))
    mask = (df[COL_BEGIN_AUM] <= threshold) & df[COL_BEGIN_AUM].notna()
    r = report.setdefault("invariant_begin_aum_guard", {})
    r["guarded_rows"] = int(mask.sum())
    if not mask.any():
        r["passed"] = True
        return
    fail_count = 0
    for col in BEGIN_AUM_GUARDED_RATES:
        if col not in df.columns:
            continue
        subset = df.loc[mask, col]
        if mode == "nan":
            bad = subset.notna()
        else:
            bad = ~((subset == 0.0) | subset.isna())
        fail_count += int(bad.sum())
    if fail_count > 0:
        r["passed"] = False
        r["error"] = f"begin_aum_guard mode={mode!r}, threshold={threshold}; {int(fail_count)} guarded row(s) non-compliant."
        return
    r["passed"] = True


def _check_fee_yield_guard(df: pd.DataFrame, policy: dict[str, Any], report: dict[str, Any]) -> None:
    """C) Where nnb <= threshold, fee_yield must match mode (NaN / 0 / cap_value)."""
    fee_guard = (policy.get("policies") or {}).get("fee_yield_guard") or {}
    mode = (fee_guard.get("mode") or "nan").strip().lower()
    if mode not in ("nan", "zero", "cap"):
        mode = "nan"
    threshold = float(fee_guard.get("threshold", 0.0))
    cap_value = float(fee_guard.get("cap_value", 0.0))
    mask = (df[COL_NNB] <= threshold) & df[COL_NNB].notna()
    r = report.setdefault("invariant_fee_yield_guard", {})
    r["guarded_rows"] = int(mask.sum())
    if not mask.any():
        r["passed"] = True
        return
    subset = df.loc[mask, COL_FEE_YIELD]
    if mode == "nan":
        bad = subset.notna()
    elif mode == "zero":
        bad = ~((subset == 0.0) | subset.isna())
    else:
        bad = ~((subset - cap_value).abs() <= TOLERANCE)
    fail_count = int(bad.sum())
    if fail_count > 0:
        r["passed"] = False
        r["error"] = f"fee_yield_guard mode={mode!r}, threshold={threshold}; {fail_count} guarded row(s) non-compliant."
        return
    r["passed"] = True


def _check_clamp(df: pd.DataFrame, policy: dict[str, Any], report: dict[str, Any]) -> None:
    """D) hard_clamp => all values within caps; warn_only => flags correct where out of bounds."""
    clamp_cfg = (policy.get("policies") or {}).get("clamp") or {}
    r = report.setdefault("invariant_clamp", {})
    if not clamp_cfg.get("enabled", True):
        r["passed"] = True
        r["mode"] = "disabled"
        return
    mode = (clamp_cfg.get("mode") or "warn_only").strip().lower()
    if mode not in ("warn_only", "hard_clamp"):
        mode = "warn_only"
    caps = clamp_cfg.get("caps") or {}
    r["mode"] = mode

    for metric in RATE_COLS:
        if metric not in df.columns or metric not in caps:
            continue
        cap = caps[metric]
        mn = float(cap.get("min", -math.inf))
        mx = float(cap.get("max", math.inf))
        flag_col = metric + CLAMP_FLAG_SUFFIX
        values = df[metric].astype(float)
        out_of_bounds = (values < mn) | (values > mx)
        out_of_bounds = out_of_bounds & values.notna()
        if mode == "hard_clamp":
            if out_of_bounds.any():
                n = int(out_of_bounds.sum())
                r["passed"] = False
                r["error"] = f"hard_clamp but {metric} has {n} value(s) outside [{mn}, {mx}]."
                return
        else:
            if flag_col not in df.columns:
                r["passed"] = False
                r["error"] = f"warn_only but missing column {flag_col!r}."
                return
            stored_flag = df[flag_col].fillna(False).astype(bool)
            flag_should_be = out_of_bounds
            mismatch = (stored_flag != flag_should_be).sum()
            if mismatch > 0:
                r["passed"] = False
                r["error"] = f"warn_only; {metric} has {int(mismatch)} row(s) with incorrect clamped_flag."
                return
    r["passed"] = True


def _recompute_sample(df: pd.DataFrame, policy: dict[str, Any], report: dict[str, Any]) -> None:
    """Spot-sample: recompute market_pnl, ogr, fee_yield; compare to stored within tolerance. Hard fail on mismatch."""
    n = len(df)
    if n == 0:
        report["recompute_check"] = {"passed": True, "sampled": 0, "mismatches": []}
        return
    size = min(RECOMPUTE_SAMPLE_SIZE, n)
    rng = random.Random(RECOMPUTE_SEED)
    indices = rng.sample(range(n), size)
    sample = df.iloc[indices].copy()

    policies_cfg = policy.get("policies") or {}
    begin_guard = policies_cfg.get("begin_aum_guard") or {}
    fee_guard = policies_cfg.get("fee_yield_guard") or {}
    clamp_policy = policies_cfg.get("clamp") or {}

    begin_aum = sample[COL_BEGIN_AUM].astype(float)
    end_aum = sample[COL_END_AUM].astype(float)
    nnb = sample[COL_NNB].astype(float)
    nnf = sample[COL_NNF].astype(float)

    recomputed_pnl = end_aum - begin_aum - nnb
    ogr_raw = safe_divide(nnb, begin_aum)
    ogr_guarded = apply_begin_aum_guard(ogr_raw, begin_aum, begin_guard)
    ogr_clean = coerce_inf_to_nan(
        pd.DataFrame({COL_OGR: ogr_guarded}), [COL_OGR]
    )[COL_OGR]
    ogr_out, _ = apply_clamp(ogr_clean, COL_OGR, clamp_policy)

    fee_raw = safe_divide(nnf, nnb)
    fee_guarded = apply_fee_yield_guard(fee_raw, nnb, fee_guard)
    fee_clean = coerce_inf_to_nan(
        pd.DataFrame({COL_FEE_YIELD: fee_guarded}), [COL_FEE_YIELD]
    )[COL_FEE_YIELD]
    fee_out, _ = apply_clamp(fee_clean, COL_FEE_YIELD, clamp_policy)

    mismatches = []
    for i, idx in enumerate(indices):
        row = sample.iloc[i]
        stored_pnl = df.at[idx, COL_MARKET_PNL]
        stored_ogr = df.at[idx, COL_OGR]
        stored_fee = df.at[idx, COL_FEE_YIELD]
        if not _values_equal(stored_pnl, recomputed_pnl.iloc[i]):
            mismatches.append({
                "index": int(idx),
                "metric": COL_MARKET_PNL,
                "stored": float(stored_pnl) if not pd.isna(stored_pnl) else None,
                "recomputed": float(recomputed_pnl.iloc[i]) if not pd.isna(recomputed_pnl.iloc[i]) else None,
            })
        if not _values_equal(stored_ogr, ogr_out.iloc[i]):
            mismatches.append({
                "index": int(idx),
                "metric": COL_OGR,
                "stored": float(stored_ogr) if not pd.isna(stored_ogr) else None,
                "recomputed": float(ogr_out.iloc[i]) if not pd.isna(ogr_out.iloc[i]) else None,
            })
        if not _values_equal(stored_fee, fee_out.iloc[i]):
            mismatches.append({
                "index": int(idx),
                "metric": COL_FEE_YIELD,
                "stored": float(stored_fee) if not pd.isna(stored_fee) else None,
                "recomputed": float(fee_out.iloc[i]) if not pd.isna(fee_out.iloc[i]) else None,
            })

    report["recompute_check"] = {
        "passed": len(mismatches) == 0,
        "sampled": size,
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:20],
    }


def run_gate(root: Path, metrics_path: Path | None = None, policy_config_path: Path | None = None) -> dict[str, Any]:
    """
    Load metrics and policy, run all invariant checks and spot-sample recompute.
    Returns report dict with passed=True only if all checks pass; otherwise passed=False and details.
    Caller should treat passed=False as hard fail (exit 1).
    """
    metrics_path = metrics_path or (root / METRICS_PARQUET.replace("\\", "/").lstrip("/"))
    policy_config_path = policy_config_path or (root / DEFAULT_POLICY_CONFIG)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics table not found: {metrics_path}")
    if not policy_config_path.exists():
        raise FileNotFoundError(f"Policy config not found: {policy_config_path}")

    df = pd.read_parquet(metrics_path)
    policy = validate_metrics_policy(load_metrics_policy(policy_config_path))

    report: dict[str, Any] = {"passed": True, "row_count": len(df)}

    _check_no_inf(df, report)
    if not report.get("invariant_inf", {}).get("passed", True):
        report["passed"] = False
        return report

    _check_begin_aum_guard(df, policy, report)
    if not report.get("invariant_begin_aum_guard", {}).get("passed", True):
        report["passed"] = False
        return report

    _check_fee_yield_guard(df, policy, report)
    if not report.get("invariant_fee_yield_guard", {}).get("passed", True):
        report["passed"] = False
        return report

    _check_clamp(df, policy, report)
    if not report.get("invariant_clamp", {}).get("passed", True):
        report["passed"] = False
        return report

    _recompute_sample(df, policy, report)
    if not report.get("recompute_check", {}).get("passed", True):
        report["passed"] = False

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run metrics policy compliance gate.")
    parser.add_argument("--run", action="store_true", help="Run gate and write report")
    parser.add_argument("--config", default=DEFAULT_POLICY_CONFIG, help="Path to metrics_policy.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)
    root = args.root or Path(__file__).resolve().parents[2]
    if not args.run:
        logger.info("Use --run to execute the gate and write qa/%s", GATE_REPORT_JSON)
        return 0
    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    report_path = qa_dir / GATE_REPORT_JSON
    try:
        report = run_gate(root, policy_config_path=root / args.config)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        if report.get("passed", True):
            logger.info("Gate passed. Wrote %s", report_path)
            return 0
        errors = []
        for k, v in report.items():
            if isinstance(v, dict) and v.get("error"):
                errors.append(f"{k}: {v['error']}")
        if report.get("recompute_check", {}).get("mismatch_count", 0) > 0:
            errors.append(f"recompute_check: {report['recompute_check']['mismatch_count']} mismatch(es)")
        logger.error("Gate failed: %s", "; ".join(errors) if errors else "see report")
        return 1
    except FileNotFoundError as e:
        logger.error("%s", e)
        report_path.write_text(json.dumps({"passed": False, "error": str(e)}, indent=2, sort_keys=True), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
