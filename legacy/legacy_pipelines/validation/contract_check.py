"""
Contract validation CLI and preflight for the validation pipeline.

- CLI: python -m pipelines.validation.contract_check
  Loads configs/validation_policy.yml, validates, runs preflight, prints summary; exits non-zero on failure.
- preflight_validation_inputs(policy): check workbook and curated/metrics_monthly.parquet exist; optional stats.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    ValidationPolicy,
    ValidationPolicyError,
    load_and_validate_validation_policy,
    required_expected_excel_columns,
)

METRICS_MONTHLY_REL = "curated/metrics_monthly.parquet"


def print_validation_summary(policy: ValidationPolicy) -> None:
    """Print a concise summary: workbook, sheet, month column, mapped columns, tolerances, fail-fast, normalization."""
    wb = policy.workbook
    exp = policy.expected_columns
    norm = policy.normalization
    tol = policy.tolerance
    ff = policy.fail_fast
    print("--- Validation policy summary ---")
    print(f"  workbook.path:       {wb.path}")
    print(f"  workbook.sheet:      {wb.sheet}")
    print(f"  workbook.month_col:  {wb.month_column}")
    print(f"  workbook.month_fmt:  {wb.month_format!r}")
    print("  expected_columns (source -> canonical):")
    print(f"    {exp.asset_growth_rate!r} -> asset_growth_rate")
    print(f"    {exp.organic_growth_rate!r} -> organic_growth_rate")
    print(f"    {exp.external_market_growth_rate!r} -> external_market_growth_rate")
    print("  tolerance:")
    print(f"    abs_tol: {tol.abs_tol}, rel_tol: {tol.rel_tol}")
    print("  fail_fast:")
    print(f"    max_mismatched_months: {ff.max_mismatched_months}, max_deviation: {ff.max_deviation}")
    print(f"    fail_on_missing_months: {ff.fail_on_missing_months}")
    print("  normalization:")
    print(f"    percent_to_decimal: {norm.percent_to_decimal}, percent_scale: {norm.percent_scale}")
    print(f"    month_align: {norm.month_align}, timezone_naive: {norm.timezone_naive}")
    print("---")


def preflight_validation_inputs(
    policy: ValidationPolicy,
    root: Path | None = None,
    *,
    verbose: bool = True,
) -> None:
    """
    Check that inputs required for validation exist; optionally print last-modified and sizes.
    Raises with an actionable message if workbook or curated/metrics_monthly.parquet is missing.
    If the workbook exists, also checks that the sheet and required columns exist (actionable error).
    """
    root = root or Path.cwd()
    workbook_path = root / policy.workbook.path
    parquet_path = root / "curated" / "metrics_monthly.parquet"

    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Validation workbook not found: {workbook_path}\n"
            f"  Resolved from root={root} and policy.workbook.path={policy.workbook.path!r}."
        )

    # Sheet and columns check (actionable: required vs found)
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("pandas (and openpyxl) required for preflight. Install: pip install pandas openpyxl") from e
    try:
        xl = pd.ExcelFile(workbook_path)
    except Exception as e:
        raise RuntimeError(f"Cannot open workbook {workbook_path}: {e}") from e
    sheet = policy.workbook.sheet
    if sheet not in xl.sheet_names:
        raise ValueError(
            f"Sheet {sheet!r} not found in workbook {workbook_path}.\n"
            f"  Available sheets: {sorted(xl.sheet_names)}."
        )
    try:
        head = pd.read_excel(workbook_path, sheet_name=sheet, nrows=0)
    except Exception as e:
        raise RuntimeError(f"Cannot read sheet {sheet!r} from {workbook_path}: {e}") from e
    found_columns = set(head.columns)
    required = required_expected_excel_columns(policy)
    missing = required - found_columns
    if missing:
        raise ValueError(
            f"Required columns missing in sheet {sheet!r} of {workbook_path}.\n"
            f"  Required: {sorted(required)}.\n"
            f"  Found:    {sorted(found_columns)}.\n"
            f"  Missing:  {sorted(missing)}."
        )

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Metrics artifact not found: {parquet_path}\n"
            f"  Resolved from root={root}. "
            "Run the metrics pipeline to produce curated/metrics_monthly.parquet."
        )

    if verbose:
        def _stat(p: Path) -> str:
            if not p.exists():
                return "N/A"
            st = p.stat()
            try:
                mtime = st.st_mtime
                from datetime import datetime
                mt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                mt = str(mtime)
            return f"{st.st_size} bytes, modified {mt}"
        print(f"  Workbook: {_stat(workbook_path)}")
        print(f"  Parquet:  {_stat(parquet_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and validate configs/validation_policy.yml; run preflight; print summary. Exit non-zero on failure (CI)."
    )
    parser.add_argument(
        "--config",
        default="configs/validation_policy.yml",
        help="Path to validation policy YAML (default: configs/validation_policy.yml)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving workbook and parquet paths (default: cwd)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip printing file stats in preflight",
    )
    args = parser.parse_args()

    try:
        policy = load_and_validate_validation_policy(args.config)
    except ValidationPolicyError as e:
        print(f"Validation policy error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error loading policy: {e}", file=sys.stderr)
        return 1

    try:
        preflight_validation_inputs(policy, root=args.root, verbose=not args.quiet)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Preflight failed: {e}", file=sys.stderr)
        return 1

    print_validation_summary(policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
