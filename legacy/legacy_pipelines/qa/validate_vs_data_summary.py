"""
Step 4: Fail-fast gate evaluator driven by validation_policy.yml.

Uses only highlighted months (drives_fail_fast == True). Triggers on:
- fail_months > max_fail_months
- worst_abs_err_overall > max_abs_err
- worst_rel_err_overall > max_rel_err
- fail_on_missing_months and any missing actual in highlighted

CLI: python -m pipelines.qa.validate_vs_data_summary --run
  Exit 2 on fail (when --strict-exit), 0 on pass; deterministic paths, no prompts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    ValidationPolicy,
    ValidationPolicyError,
    load_and_validate_validation_policy,
)
from legacy.legacy_pipelines.validation.compare_to_data_summary import RATES

DEFAULT_REPORT_PATHS = {
    "validation_report": "qa/validation_report.csv",
    "validation_summary": "qa/validation_summary.json",
    "validation_fail_examples": "qa/validation_fail_examples.csv",
}
MAX_FAILING_MONTHS_DISPLAY = 10


def format_failure_message(result: dict, report_paths: dict | None = None) -> str:
    """
    Return a concise, actionable failure message for CI.

    result: dict from evaluate_fail_fast (triggered, reasons, failing_months,
            failing_metrics, worst).
    report_paths: optional dict with keys validation_report, validation_summary,
                  validation_fail_examples -> path strings. Defaults to DEFAULT_REPORT_PATHS.
    """
    paths = {**DEFAULT_REPORT_PATHS, **(report_paths or {})}
    lines = [
        "VALIDATION FAILED",
        "",
        "Reasons:",
    ]
    for r in result.get("reasons", []):
        lines.append(f"  - {r}")
    lines.append("")
    failing_metrics = result.get("failing_metrics", [])
    lines.append(f"Failing metrics: {', '.join(failing_metrics) if failing_metrics else '—'}")
    failing_months = result.get("failing_months", [])
    if len(failing_months) > MAX_FAILING_MONTHS_DISPLAY:
        shown = failing_months[:MAX_FAILING_MONTHS_DISPLAY]
        extra = len(failing_months) - MAX_FAILING_MONTHS_DISPLAY
        lines.append(f"Failing month_end (first {MAX_FAILING_MONTHS_DISPLAY}): {', '.join(shown)} +{extra} more")
    else:
        lines.append(f"Failing month_end: {', '.join(failing_months) if failing_months else '—'}")
    lines.append("")
    lines.append("Worst deltas per metric:")
    for metric, w in result.get("worst", {}).items():
        me = w.get("month_end") or "—"
        ae = w.get("abs_err") if w.get("abs_err") is not None else "—"
        re = w.get("rel_err") if w.get("rel_err") is not None else "—"
        exp = w.get("expected") if w.get("expected") is not None else "—"
        act = w.get("actual") if w.get("actual") is not None else "—"
        lines.append(f"  {metric}: month_end={me}, abs_err={ae}, rel_err={re}, expected={exp}, actual={act}")
    lines.append("")
    lines.append("QA artifacts:")
    lines.append(f"  {paths.get('validation_report', DEFAULT_REPORT_PATHS['validation_report'])}")
    lines.append(f"  {paths.get('validation_summary', DEFAULT_REPORT_PATHS['validation_summary'])}")
    lines.append(f"  {paths.get('validation_fail_examples', DEFAULT_REPORT_PATHS['validation_fail_examples'])}")
    return "\n".join(lines)


def format_pass_message(result: dict, report_paths: dict | None = None) -> str:
    """
    Return a concise pass message for CI.

    result: dict from evaluate_fail_fast (highlighted_count, fail_months_count,
            worst_abs_err_overall, worst_rel_err_overall).
    report_paths: optional; unused for pass message but kept for API consistency.
    """
    highlighted = result.get("highlighted_count", 0)
    fail_months = result.get("fail_months_count", 0)
    worst_abs = result.get("worst_abs_err_overall")
    worst_rel = result.get("worst_rel_err_overall")
    worst_abs_str = f"{worst_abs:.6g}" if worst_abs is not None else "—"
    worst_rel_str = f"{worst_rel:.6g}" if worst_rel is not None else "—"
    lines = [
        "VALIDATION PASSED",
        f"  Highlighted months: {highlighted}",
        f"  Fail months count: {fail_months}",
        f"  Worst abs_err (highlighted): {worst_abs_str}",
        f"  Worst rel_err (highlighted): {worst_rel_str}",
    ]
    return "\n".join(lines)


def evaluate_fail_fast(report_df: pd.DataFrame, policy: ValidationPolicy) -> dict:
    """
    Evaluate fail-fast gates using only highlighted months (drives_fail_fast == True).

    Returns dict with:
      triggered: bool
      reasons: list[str]
      failing_months: list[str] (month_end as string, highlighted rows with any_fail)
      failing_metrics: list[str] (metric names that had at least one fail in highlighted)
      worst: dict[metric, {month_end, abs_err, rel_err, expected, actual}]
    """
    required = ["month_end", "drives_fail_fast", "any_fail"] + [
        c for m in RATES for c in (f"abs_err_{m}", f"rel_err_{m}", f"expected_{m}", f"actual_{m}")
    ]
    missing = [c for c in required if c not in report_df.columns]
    if missing:
        raise ValueError(f"report_df missing columns: {missing}")

    ff = policy.fail_fast
    # Use new fields when set (from YAML); fallback to legacy for backward compat
    max_fail_months = ff.max_fail_months if ff.max_fail_months > 0 else ff.max_mismatched_months
    max_abs_err = ff.max_abs_err if ff.max_abs_err > 0.0 else ff.max_deviation
    max_rel_err = ff.max_rel_err

    report_hi = report_df.loc[report_df["drives_fail_fast"] == True].copy()  # noqa: E712

    reasons: list[str] = []
    triggered = False

    # fail_months = count of highlighted rows where any_fail == True
    fail_months = int(report_hi["any_fail"].sum())
    if fail_months > max_fail_months:
        triggered = True
        reasons.append(f"fail_months ({fail_months}) > max_fail_months ({max_fail_months})")

    # worst abs_err / rel_err over highlighted only (skipna; if all NaN do not trigger on that rule)
    abs_cols = [f"abs_err_{m}" for m in RATES]
    rel_cols = [f"rel_err_{m}" for m in RATES]
    if report_hi.empty:
        worst_abs_err_overall = np.nan
        worst_rel_err_overall = np.nan
    else:
        worst_abs_err_overall = report_hi[abs_cols].max().max(skipna=True)
        worst_rel_err_overall = report_hi[rel_cols].max().max(skipna=True)
        if pd.isna(worst_abs_err_overall):
            worst_abs_err_overall = np.nan
        if pd.isna(worst_rel_err_overall):
            worst_rel_err_overall = np.nan

    if not pd.isna(worst_abs_err_overall) and float(worst_abs_err_overall) > max_abs_err:
        triggered = True
        reasons.append(f"worst_abs_err_overall ({float(worst_abs_err_overall):.6g}) > max_abs_err ({max_abs_err})")
    if not pd.isna(worst_rel_err_overall) and float(worst_rel_err_overall) > max_rel_err:
        triggered = True
        reasons.append(f"worst_rel_err_overall ({float(worst_rel_err_overall):.6g}) > max_rel_err ({max_rel_err})")

    # missing actual in highlighted
    missing_actual_highlighted = False
    if not report_hi.empty:
        for m in RATES:
            if report_hi[f"actual_{m}"].isna().any():
                missing_actual_highlighted = True
                break
    if ff.fail_on_missing_months and missing_actual_highlighted:
        triggered = True
        reasons.append("fail_on_missing_months and at least one highlighted row has missing actual")

    # failing_months: highlighted month_end where any_fail
    failing_months = report_hi.loc[report_hi["any_fail"], "month_end"].astype(str).tolist()

    # failing_metrics: metrics where at least one highlighted row has pass_<m> == False
    failing_metrics = [m for m in RATES if f"pass_{m}" in report_df.columns and (report_hi[f"pass_{m}"] == False).any()]  # noqa: E712

    # worst: per metric, the highlighted row with max abs_err for that metric (nanmax; ignore NaN)
    worst: dict[str, dict] = {}
    for m in RATES:
        abs_col = f"abs_err_{m}"
        rel_col = f"rel_err_{m}"
        exp_col = f"expected_{m}"
        act_col = f"actual_{m}"
        if report_hi.empty:
            worst[m] = {"month_end": None, "abs_err": None, "rel_err": None, "expected": None, "actual": None}
            continue
        max_abs = report_hi[abs_col].max(skipna=True)
        if pd.isna(max_abs):
            worst[m] = {"month_end": None, "abs_err": None, "rel_err": None, "expected": None, "actual": None}
            continue
        idx = report_hi[abs_col].idxmax()
        row = report_hi.loc[idx]
        worst[m] = {
            "month_end": str(row["month_end"]),
            "abs_err": float(row[abs_col]) if not pd.isna(row[abs_col]) else None,
            "rel_err": float(row[rel_col]) if not pd.isna(row[rel_col]) else None,
            "expected": float(row[exp_col]) if not pd.isna(row[exp_col]) else None,
            "actual": float(row[act_col]) if not pd.isna(row[act_col]) else None,
        }

    highlighted_count = len(report_hi)
    worst_abs = float(worst_abs_err_overall) if not pd.isna(worst_abs_err_overall) else None
    worst_rel = float(worst_rel_err_overall) if not pd.isna(worst_rel_err_overall) else None

    return {
        "triggered": triggered,
        "reasons": reasons,
        "failing_months": failing_months,
        "failing_metrics": failing_metrics,
        "worst": worst,
        "highlighted_count": highlighted_count,
        "fail_months_count": fail_months,
        "worst_abs_err_overall": worst_abs,
        "worst_rel_err_overall": worst_rel,
    }


def write_fail_examples(
    report_df: pd.DataFrame,
    out_path: str | Path = "qa/validation_fail_examples.csv",
    top_n: int = 50,
) -> None:
    """
    Write top N worst months per metric (highlighted only) to a single CSV.

    Only rows with drives_fail_fast==True are considered. For each metric, rows are
    sorted by abs_err descending (NaNs last), top_n taken, then output columns:
    metric, month_end, expected, actual, diff, abs_err, rel_err, pass, drives_fail_fast.
    month_end is written in ISO format (YYYY-MM-DD). Creates parent dirs of out_path.
    """
    report_hi = report_df.loc[report_df["drives_fail_fast"] == True].copy()  # noqa: E712
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[pd.DataFrame] = []
    for m in RATES:
        abs_col = f"abs_err_{m}"
        if abs_col not in report_hi.columns:
            continue
        # Sort descending by abs_err; NaNs last
        sorted_df = report_hi.sort_values(abs_col, ascending=False, na_position="last")
        top = sorted_df.head(top_n)
        out = pd.DataFrame({
            "metric": m,
            "month_end": pd.to_datetime(top["month_end"]).dt.strftime("%Y-%m-%d"),
            "expected": top[f"expected_{m}"].values,
            "actual": top[f"actual_{m}"].values,
            "diff": top[f"diff_{m}"].values,
            "abs_err": top[f"abs_err_{m}"].values,
            "rel_err": top[f"rel_err_{m}"].values,
            "pass": top[f"pass_{m}"].values if f"pass_{m}" in top.columns else False,
            "drives_fail_fast": top["drives_fail_fast"].values,
        })
        blocks.append(out)

    if not blocks:
        # No metrics / empty highlighted: write header only
        out_empty = pd.DataFrame(columns=[
            "metric", "month_end", "expected", "actual", "diff", "abs_err", "rel_err", "pass", "drives_fail_fast"
        ])
        out_empty.to_csv(path, index=False, date_format="%Y-%m-%d")
        return
    combined = pd.concat(blocks, ignore_index=True)
    combined.to_csv(path, index=False, date_format="%Y-%m-%d")


def main() -> int:
    """CLI: load policy, load or generate report, evaluate_fail_fast; exit 2 on triggered, 0 on pass."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Step 4: Evaluate fail-fast gates on validation report. Exit 2 on fail (CI)."
    )
    parser.add_argument(
        "--no-run",
        dest="run",
        action="store_false",
        default=True,
        help="Do not run Step 3 if report missing; fail with error",
    )
    parser.add_argument(
        "--run",
        dest="run",
        action="store_true",
        help="Run full pipeline (Step 3) if report missing (default)",
    )
    parser.add_argument(
        "--policy",
        default="configs/validation_policy.yml",
        help="Validation policy YAML path (default: configs/validation_policy.yml)",
    )
    parser.add_argument(
        "--report",
        default="qa/validation_report.csv",
        help="Path to validation report CSV; if missing and --run, generate via Step 3 (default: qa/validation_report.csv)",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        default=True,
        help="Exit 2 when gates triggered (default: True)",
    )
    parser.add_argument(
        "--no-strict-exit",
        dest="strict_exit",
        action="store_false",
        help="Exit 0 even when gates triggered",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root for resolving policy and report paths (default: cwd)",
    )
    args = parser.parse_args()

    root = args.root
    policy_path = root / args.policy if not Path(args.policy).is_absolute() else Path(args.policy)
    report_path = root / args.report if not Path(args.report).is_absolute() else Path(args.report)

    try:
        policy = load_and_validate_validation_policy(policy_path)
    except ValidationPolicyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if report_path.exists():
        report_df = pd.read_csv(report_path)
        report_df["month_end"] = pd.to_datetime(report_df["month_end"])
    else:
        if not args.run:
            print(f"Error: Report not found: {report_path} (use --run to generate)", file=sys.stderr)
            return 1
        try:
            from legacy.legacy_pipelines.validation.run_validation_compare import run as run_step3
            report_df, _summary, _ = run_step3(config_path=args.policy, root=root)
        except (ValidationPolicyError, FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    result = evaluate_fail_fast(report_df, policy)
    report_paths = {
        "validation_report": str(report_path),
        "validation_summary": str(root / DEFAULT_REPORT_PATHS["validation_summary"]),
        "validation_fail_examples": str(root / DEFAULT_REPORT_PATHS["validation_fail_examples"]),
    }

    if result["triggered"]:
        write_fail_examples(
            report_df,
            out_path=report_paths["validation_fail_examples"],
            top_n=50,
        )
        print(format_failure_message(result, report_paths), file=sys.stderr)
        return 2 if args.strict_exit else 0
    print(format_pass_message(result, report_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
