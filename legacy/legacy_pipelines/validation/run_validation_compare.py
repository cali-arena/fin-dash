"""
End-to-end runner for Step 3: contract → expected → firm → compare → artifacts.

CLI: python -m pipelines.validation.run_validation_compare

1) Load & validate policy (Step 1 contract)
2) Load expected rates from Excel (Step 1 reader)
3) Load or recompute firm_level_recomputed (Step 2 artifact; recompute if missing)
4) Build report (Step 3)
5) Write qa/validation_report.csv and qa/validation_summary.json
6) If fail_fast_triggered => exit code 2 (CI), else exit 0.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    ValidationPolicyError,
    load_and_validate_validation_policy,
    policy_hash,
)
from legacy.legacy_pipelines.validation.compare_to_data_summary import (
    RATES,
    build_validation_report,
    write_validation_report,
)
from legacy.legacy_pipelines.validation.read_expected_data_summary import load_expected_rates

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/validation_policy.yml"
FIRM_RECOMPUTED_PARQUET = "curated/qa/firm_level_recomputed.parquet"
VALIDATION_REPORT_CSV = "qa/validation_report.csv"
VALIDATION_SUMMARY_JSON = "qa/validation_summary.json"


def _ensure_firm_level_recomputed(root: Path) -> Path:
    """Load firm_level_recomputed from curated/qa; if missing, run Step 2 recompute. Returns parquet path."""
    parquet_path = root / FIRM_RECOMPUTED_PARQUET
    if parquet_path.exists():
        return parquet_path
    logger.info("firm_level_recomputed not found; running Step 2 recompute")
    subprocess.run(
        [sys.executable, "-m", "pipelines.validation.recompute_firm_level", "--root", str(root)],
        check=True,
        capture_output=False,
    )
    return parquet_path


def _month_range_str(df: pd.DataFrame, col: str = "month_end") -> str:
    """Return 'min -- max' or 'N/A' for empty."""
    if df.empty or col not in df.columns:
        return "N/A"
    mn = pd.Timestamp(df[col].min())
    mx = pd.Timestamp(df[col].max())
    return f"{mn.strftime('%Y-%m-%d')} -- {mx.strftime('%Y-%m-%d')}"


def _compute_fail_fast_decision(report: pd.DataFrame, policy) -> tuple[bool, dict]:
    """
    Use only highlighted (drives_fail_fast) rows. Return (fail_fast_triggered, reasons_dict).
    reasons_dict: mismatched_count, max_abs_err, missing_highlighted_months (bool).
    """
    ff = policy.fail_fast
    subset = report.loc[report["drives_fail_fast"]]
    mismatched_count = int(subset["any_fail"].sum())
    # Max absolute error over fail-fast subset (any rate)
    abs_cols = [f"abs_err_{r}" for r in RATES]
    if subset.empty:
        max_abs_err = 0.0
        missing_highlighted = False
    else:
        row_max = report.loc[report["drives_fail_fast"], abs_cols].max(axis=1)
        max_abs_err = float(row_max.max(skipna=True)) if not row_max.empty else 0.0
        if pd.isna(max_abs_err):
            max_abs_err = float("inf")
        missing_highlighted = (subset["reason"] == "missing_actual").any()

    triggered = (
        mismatched_count > ff.max_mismatched_months
        or max_abs_err > ff.max_deviation
        or (ff.fail_on_missing_months and missing_highlighted)
    )
    reasons = {
        "mismatched_count": mismatched_count,
        "max_mismatched_months": ff.max_mismatched_months,
        "max_abs_err": max_abs_err,
        "max_deviation": ff.max_deviation,
        "fail_on_missing_months": ff.fail_on_missing_months,
        "missing_highlighted_months": bool(missing_highlighted),
    }
    return triggered, reasons


def run(
    config_path: str | Path = DEFAULT_CONFIG,
    root: Path | None = None,
) -> tuple[pd.DataFrame, dict, bool]:
    """
    Run full Step 3 pipeline. Returns (report, summary_dict, fail_fast_triggered).
    summary_dict is what we write to validation_summary.json.
    """
    root = root or Path.cwd()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path

    # 1) Load policy
    policy = load_and_validate_validation_policy(config_path)

    # Resolve paths from root (reader uses cwd for policy.workbook.path)
    orig_cwd = os.getcwd()
    try:
        os.chdir(root)
        # 2) Load expected
        expected_df = load_expected_rates(policy)
    finally:
        os.chdir(orig_cwd)

    # 3) Load or recompute actual
    parquet_path = _ensure_firm_level_recomputed(root)
    actual_df = pd.read_parquet(parquet_path)
    for c in ["month_end"] + RATES:
        if c not in actual_df.columns:
            raise ValueError(f"firm_level_recomputed missing column: {c}")
    actual_df = actual_df[["month_end"] + RATES].copy()

    # 4) Build report
    report = build_validation_report(expected_df, actual_df, policy)

    # 5) Fail-fast decision and summary
    fail_fast_triggered, reasons = _compute_fail_fast_decision(report, policy)
    highlighted_count = int(report["drives_fail_fast"].sum())
    highlighted_months = report.loc[report["drives_fail_fast"], "month_end"].astype(str).tolist()

    summary = {
        "policy_hash": policy_hash(policy),
        "expected_rowcount": len(expected_df),
        "expected_month_range": _month_range_str(expected_df),
        "actual_rowcount": len(actual_df),
        "actual_month_range": _month_range_str(actual_df),
        "report_rowcount": len(report),
        "highlighted_count": highlighted_count,
        "highlighted_months": highlighted_months,
        "fail_fast_triggered": fail_fast_triggered,
        "reasons": reasons,
        "decision": "fail" if fail_fast_triggered else "pass",
    }

    return report, summary, fail_fast_triggered


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 3: load policy + expected, load/recompute firm-level, compare, write qa/validation_report.csv and qa/validation_summary.json. Exit 2 if fail_fast_triggered (CI)."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Validation policy YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving paths (default: cwd)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        report, summary, fail_fast_triggered = run(config_path=args.config, root=args.root)
    except (ValidationPolicyError, FileNotFoundError, ValueError, RuntimeError) as e:
        logger.exception("%s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    root = args.root
    # Write artifacts
    write_validation_report(report, root=root)
    summary_path = root / VALIDATION_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure JSON-serializable (replace inf/nan in reasons with None)
    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj or abs(obj) == float("inf")):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(x) for x in obj]
        return obj
    summary_path.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)

    # Logging: rowcounts, month ranges, highlighted, decision
    print("--- Validation compare ---")
    print(f"  expected:  rowcount={summary['expected_rowcount']}  month_range={summary['expected_month_range']}")
    print(f"  actual:    rowcount={summary['actual_rowcount']}  month_range={summary['actual_month_range']}")
    print(f"  highlighted: count={summary['highlighted_count']}  months={summary['highlighted_months']}")
    r = summary["reasons"]
    print(f"  decision: {summary['decision']}  (fail_fast_triggered={summary['fail_fast_triggered']})")
    print(f"  reasons: mismatched_count={r['mismatched_count']} (max_allowed={r['max_mismatched_months']}), max_abs_err={r['max_abs_err']:.6g} (max_allowed={r['max_deviation']:.6g}), missing_highlighted_months={r['missing_highlighted_months']}")
    print("---")

    return 2 if fail_fast_triggered else 0


if __name__ == "__main__":
    sys.exit(main())
