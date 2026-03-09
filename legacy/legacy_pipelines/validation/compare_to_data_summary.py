"""
Step 3: Compare recomputed firm-level rates to Excel DATA SUMMARY (expected).
Produces qa/validation_report.csv with per-rate pass/fail, all_pass/any_fail,
highlighted, drives_fail_fast, fail_fast_any_fail.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.validation_policy_contract import ValidationPolicy

logger = logging.getLogger(__name__)

RATES = ["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]
REPORT_PATH = "qa/validation_report.csv"
REL_ERR_DENOM_MIN = 1e-12


def _coerce_highlighted_column(series: pd.Series) -> pd.Series:
    """Coerce to bool: Y/y/yes/1/True/true -> True; N/n/no/0/False/false -> False; else False."""
    out = pd.Series(False, index=series.index, dtype=bool)
    for i, v in series.items():
        if pd.isna(v):
            continue
        if isinstance(v, bool):
            out.iloc[out.index.get_loc(i)] = bool(v)
            continue
        if isinstance(v, (int, float)):
            out.iloc[out.index.get_loc(i)] = bool(v != 0)
            continue
        s = str(v).strip().upper()
        if s in ("Y", "YES", "1", "TRUE"):
            out.iloc[out.index.get_loc(i)] = True
        elif s in ("N", "NO", "0", "FALSE"):
            out.iloc[out.index.get_loc(i)] = False
        else:
            out.iloc[out.index.get_loc(i)] = False
    return out


def _parse_highlighted_months_to_month_end(months: list[str]) -> pd.DatetimeIndex:
    """Parse date strings to month-end; timezone-naive datetime64[ns]."""
    if not months:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    dt = pd.to_datetime(months, errors="coerce")
    month_end = dt.to_period("M").to_timestamp("M")
    if month_end.tz is not None:
        month_end = month_end.tz_localize(None)
    return month_end


def _ensure_month_end_dtype(series: pd.Series) -> pd.Series:
    """Ensure timezone-naive datetime64[ns]."""
    me = pd.to_datetime(series)
    if me.dt.tz is not None:
        me = me.dt.tz_localize(None)
    return me.astype("datetime64[ns]")


def build_validation_report(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    policy: ValidationPolicy,
) -> pd.DataFrame:
    """
    Left join expected to actual on month_end; add expected_<m>, actual_<m>, diff_<m>,
    abs_err_<m>, rel_err_<m>, pass_<m> for each rate; all_pass, any_fail, reason.
    Sorted by month_end asc; month_end timezone-naive datetime64[ns].
    """
    abs_tol = policy.tolerance.abs_tol
    rel_tol = policy.tolerance.rel_tol

    expected = expected_df.copy()
    actual = actual_df.copy()
    for col in ["month_end"] + RATES:
        if col not in expected.columns:
            raise ValueError(f"expected_df missing column: {col}")
        if col not in actual.columns:
            raise ValueError(f"actual_df missing column: {col}")

    expected["month_end"] = _ensure_month_end_dtype(expected["month_end"])
    actual["month_end"] = _ensure_month_end_dtype(actual["month_end"])

    # Keep optional columns from expected (e.g. highlighted)
    extra_expected = [c for c in expected.columns if c not in (["month_end"] + RATES)]
    merge_cols = ["month_end"]
    actual_for_merge = actual[merge_cols + RATES].copy()
    actual_for_merge = actual_for_merge.rename(columns={m: f"actual_{m}" for m in RATES})
    expected_renamed = expected[merge_cols + RATES].copy()
    expected_renamed = expected_renamed.rename(columns={m: f"expected_{m}" for m in RATES})
    report = expected_renamed.merge(
        actual_for_merge,
        on="month_end",
        how="left",
    )
    if extra_expected:
        report = report.merge(
            expected[["month_end"] + extra_expected],
            on="month_end",
            how="left",
        )

    report["month_end"] = _ensure_month_end_dtype(report["month_end"])

    # reason: "missing_actual" when any actual is NaN
    missing_actual = pd.Series(False, index=report.index)
    for m in RATES:
        ac = report[f"actual_{m}"]
        missing_actual = missing_actual | ac.isna()
    report["reason"] = missing_actual.map(lambda x: "missing_actual" if x else "")

    for m in RATES:
        exp_col = f"expected_{m}"
        act_col = f"actual_{m}"
        report[f"diff_{m}"] = report[act_col].astype("float64") - report[exp_col].astype("float64")
        report[f"abs_err_{m}"] = report[f"diff_{m}"].abs()
        denom = report[exp_col].abs().clip(lower=REL_ERR_DENOM_MIN)
        report[f"rel_err_{m}"] = report[f"abs_err_{m}"] / denom
        # pass = (abs_err <= abs_tol) OR (rel_err <= rel_tol); if actual NaN => False (reason already set)
        pass_ = (report[f"abs_err_{m}"] <= abs_tol) | (report[f"rel_err_{m}"] <= rel_tol)
        pass_ = pass_.fillna(False)
        report[f"pass_{m}"] = pass_
        report.loc[report[act_col].isna(), f"pass_{m}"] = False
        report[f"pass_{m}"] = report[f"pass_{m}"].astype(bool)

    report["all_pass"] = report["pass_asset_growth_rate"] & report["pass_organic_growth_rate"] & report["pass_external_market_growth_rate"]
    report["any_fail"] = ~report["all_pass"]

    # Highlighted months (fail-fast driving subset)
    hl = getattr(policy, "highlighted", None)
    if hl is None or (hl.mode == "none"):
        report["highlighted"] = False
    elif hl.mode == "column" and hl.column and hl.column in report.columns:
        report["highlighted"] = _coerce_highlighted_column(report[hl.column])
    elif hl.mode == "list" and hl.months:
        month_end_set = _parse_highlighted_months_to_month_end(hl.months)
        report["highlighted"] = report["month_end"].isin(month_end_set)
    else:
        report["highlighted"] = False
    report["highlighted"] = report["highlighted"].astype(bool)
    report["drives_fail_fast"] = report["highlighted"]
    report["fail_fast_any_fail"] = report["drives_fail_fast"] & report["any_fail"]

    report = report.sort_values("month_end", kind="mergesort").reset_index(drop=True)
    report["month_end"] = report["month_end"].astype("datetime64[ns]")
    return report


def write_validation_report(report: pd.DataFrame, root: Path | None = None) -> Path:
    """Write report to qa/validation_report.csv. Creates qa/ if needed. Returns path."""
    root = root or Path.cwd()
    qa_dir = root / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / "validation_report.csv"
    report.to_csv(path, index=False, date_format="%Y-%m-%d")
    logger.info("Wrote %s (%d rows)", path, len(report))
    return path
