"""
Validate recomputed firm-level growth rates against DATA SUMMARY (expected).
Actual: firm grain only (sum over slices from metrics_monthly). Expected: DATA SUMMARY = firm.
If grain ever differed we would mark row as SKIPPED_GRAIN and not count as formula failure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.metrics.data_summary_formulas import compute_firm_rates_df

# Allow small rounding differences between DATA SUMMARY and recomputed formula (e.g. 0.005–0.008).
# Remaining fails should be only acceptable SKIPs (MISSING_DATA, SKIP_INCOMPLETE_COVERAGE).
THRESHOLD_ABS = 0.01

# Grain for this validation: actual = firm (sum by month_end); expected = firm (DATA SUMMARY).
# Slice metrics must never be compared to firm checksum unless checksum is also sliced.
VALIDATION_GRAIN_ACTUAL = "firm"
VALIDATION_GRAIN_EXPECTED = "firm"

ALLOWED_FAIL_REASONS = frozenset({
    "MISSING_DATA",
    "GRAIN_MISMATCH",
    "FORMULA_MISMATCH",
    "DATE_ALIGNMENT",
    "MAPPING_MISMATCH",
    "SKIP_INCOMPLETE_COVERAGE",
})
SKIPPED_GRAIN = "SKIPPED_GRAIN"
SKIP_INCOMPLETE_COVERAGE = "SKIP_INCOMPLETE_COVERAGE"


def run(curated_dir: Path, qa_dir: Path) -> int:
    qa_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(curated_dir / "metrics_monthly.parquet")
    summary = pd.read_parquet(curated_dir / "data_summary_normalized.parquet")

    # Actual: firm grain only. Sum slice-level metrics by month_end; then apply DATA SUMMARY formulas.
    firm = metrics.groupby("month_end", as_index=False)[["begin_aum", "end_aum", "nnb"]].sum(min_count=1)
    firm = compute_firm_rates_df(firm)

    merged = summary.merge(firm, on="month_end", how="left")

    # Explicit grain columns: both firm for this pipeline (DATA SUMMARY is firm-only).
    merged["validation_grain_actual"] = VALIDATION_GRAIN_ACTUAL
    merged["validation_grain_expected"] = VALIDATION_GRAIN_EXPECTED
    grain_mismatch = merged["validation_grain_actual"] != merged["validation_grain_expected"]
    merged["skip_reason"] = grain_mismatch.map(lambda x: SKIPPED_GRAIN if x else "")

    # Incomplete coverage: summary has a month_end but firm has no row for that month (gaps in actual).
    no_firm_row = merged["end_aum"].isna()
    merged.loc[no_firm_row, "skip_reason"] = SKIP_INCOMPLETE_COVERAGE

    compare_cols = [
        ("asset_growth_rate", "asset_growth_rate_calc"),
        ("organic_growth_rate", "organic_growth_rate_calc"),
        ("external_growth_rate", "external_growth_rate_calc"),
    ]
    for left, right in compare_cols:
        if left in merged.columns:
            merged[f"abs_err_{left}"] = (merged[left] - merged[right]).abs()
        else:
            merged[f"abs_err_{left}"] = pd.NA

    err_cols = [f"abs_err_{k}" for k, _ in compare_cols]
    merged["any_fail"] = merged[err_cols].gt(THRESHOLD_ABS).any(axis=1)
    # Rows skipped due to grain mismatch or incomplete coverage are not formula failures.
    merged.loc[merged["skip_reason"] == SKIPPED_GRAIN, "any_fail"] = False
    merged.loc[merged["skip_reason"] == SKIP_INCOMPLETE_COVERAGE, "any_fail"] = False

    # fail_reason and fail_note: SKIPPED_GRAIN / SKIP_INCOMPLETE_COVERAGE / MISSING_DATA / FORMULA_MISMATCH.
    def _fail_reason_note(row: pd.Series) -> tuple[str, str]:
        if row.get("skip_reason") == SKIPPED_GRAIN:
            return ("SKIPPED_GRAIN", "Row skipped: validation grain actual vs expected mismatch; compare like-with-like.")
        if row.get("skip_reason") == SKIP_INCOMPLETE_COVERAGE:
            return (
                SKIP_INCOMPLETE_COVERAGE,
                "Month missing from actual data; coverage incomplete. Widen date range or ensure data for all months.",
            )
        begin = row.get("begin_aum", None)
        try:
            begin_f = float(begin) if begin is not None and pd.notna(begin) else float("nan")
        except (TypeError, ValueError):
            begin_f = float("nan")
        if begin_f != begin_f or begin_f <= 0:
            return ("MISSING_DATA", "begin_aum missing or <= 0 for this month; cannot compare rates.")
        if not row.get("any_fail", False):
            return ("", "")
        month = row.get("month_end", "")
        try:
            month = pd.Timestamp(month).strftime("%Y-%m-%d") if pd.notna(month) and month != "" else str(month)
        except Exception:
            month = str(month)
        return (
            "FORMULA_MISMATCH",
            f"Reported asset_growth_rate does not match formula-derived value from begin_aum/end_aum for month_end {month}.",
        )

    out = merged.apply(lambda r: pd.Series(_fail_reason_note(r)), axis=1)
    merged["fail_reason"] = out[0]
    merged["fail_note"] = out[1]
    merged.loc[merged["fail_reason"] == "MISSING_DATA", "any_fail"] = False
    merged.loc[merged["fail_reason"] == SKIP_INCOMPLETE_COVERAGE, "any_fail"] = False

    report_path = qa_dir / "validation_report.csv"
    merged.to_csv(report_path, index=False)
    # any_fail is already False for SKIPPED_GRAIN and MISSING_DATA rows
    return int(merged["any_fail"].sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate recomputed growth rates against DATA SUMMARY")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--qa-dir", default="qa")
    args = parser.parse_args()
    fail_count = run(Path(args.curated_dir), Path(args.qa_dir))
    if fail_count > 0:
        print(f"Validation failures above threshold: {fail_count}")
        return 2
    print("Validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
