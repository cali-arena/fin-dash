"""
Excel DATA SUMMARY reader + normalizer for expected rates.

Reads the workbook/sheet defined in ValidationPolicy, keeps month + expected rate columns,
renames to canonical names, normalizes (month-end, percent-to-decimal, float64), and
writes invalid rows to qa/expected_rates_rejects.csv.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from legacy.legacy_pipelines.contracts.validation_policy_contract import (
    ValidationPolicy,
    required_expected_excel_columns,
)

logger = logging.getLogger(__name__)

REJECTS_PATH = "qa/expected_rates_rejects.csv"

CANONICAL_COLUMNS = [
    "month_end",
    "asset_growth_rate",
    "organic_growth_rate",
    "external_market_growth_rate",
]


def _parse_month_to_end(series: pd.Series, month_format: str | None, timezone_naive: bool) -> pd.Series:
    """Parse month column to datetime and align to month-end. Return timezone-naive if requested."""
    if month_format:
        dt = pd.to_datetime(series, format=month_format, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    # Align to month-end (last day of month)
    month_end = dt.dt.to_period("M").dt.to_timestamp("M")
    if timezone_naive and month_end.dt.tz is not None:
        month_end = month_end.dt.tz_localize(None)
    return month_end


def _percent_to_decimal_series(series: pd.Series, percent_scale: float) -> pd.Series:
    """
    Convert a series to decimal. If string, strip '%' and divide by 100.
    If numeric, divide by percent_scale. Invalid -> NaN (caller handles rejects).
    """
    out = pd.Series(index=series.index, dtype=float)
    for i, v in series.items():
        if pd.isna(v):
            out.iloc[out.index.get_loc(i)] = float("nan")
            continue
        if isinstance(v, str):
            s = v.strip().rstrip("%").strip()
            try:
                out.iloc[out.index.get_loc(i)] = float(s) / 100.0
            except (TypeError, ValueError):
                out.iloc[out.index.get_loc(i)] = float("nan")
        else:
            try:
                out.iloc[out.index.get_loc(i)] = float(v) / percent_scale
            except (TypeError, ValueError):
                out.iloc[out.index.get_loc(i)] = float("nan")
    return out


def _coerce_float64_series(series: pd.Series) -> pd.Series:
    """Coerce to float64; invalid entries become NaN."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def normalize_expected_rates_frame(
    df: pd.DataFrame,
    month_col: str,
    rate_columns: list[str],
    month_format: str | None,
    timezone_naive: bool,
    percent_to_decimal: bool,
    percent_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize a dataframe with a month column and rate columns (already with canonical names).
    Returns (clean_df, rejects_df). clean_df has month_end + rate columns as float64, unique on month_end.
    rejects_df has index (original row index), reason, and optional row snapshot.
    """
    if df.empty:
        return df.copy(), pd.DataFrame(columns=["original_index", "reason"])

    out = pd.DataFrame(index=df.index)
    rejects: list[dict] = []

    # Month parsing
    month_end = _parse_month_to_end(df[month_col], month_format, timezone_naive)
    out["month_end"] = month_end
    for idx in df.index[month_end.isna()]:
        if df[month_col].loc[idx] is not None and not (isinstance(df[month_col].loc[idx], float) and pd.isna(df[month_col].loc[idx])):
            rejects.append({"original_index": idx, "reason": "invalid_month"})

    # Rate columns
    for col in rate_columns:
        if col not in df.columns:
            for idx in df.index:
                rejects.append({"original_index": idx, "reason": f"missing_column_{col}"})
            out[col] = float("nan")
            continue
        raw = df[col].copy()
        original_raw = raw
        if percent_to_decimal:
            raw = _percent_to_decimal_series(raw, percent_scale)
        vals = _coerce_float64_series(raw)
        out[col] = vals
        for idx in df.index:
            v_orig = original_raw.loc[idx]
            if pd.isna(vals.loc[idx]) and not (pd.isna(v_orig) or (isinstance(v_orig, (int, float)) and pd.isna(v_orig))):
                rejects.append({"original_index": idx, "reason": f"invalid_rate_{col}"})

    # Drop rows that had invalid month
    out = out.loc[~out["month_end"].isna()].copy()
    out = out.astype({"month_end": "datetime64[ns]"})
    for c in rate_columns:
        out[c] = out[c].astype("float64")

    # Dedupe on month_end: keep first occurrence
    valid_month_idx = month_end.notna()
    out = out.drop_duplicates(subset=["month_end"], keep="first")
    dropped_idx = df.index.difference(out.index)
    for idx in dropped_idx:
        if valid_month_idx.loc[idx]:
            rejects.append({"original_index": idx, "reason": "duplicate_month_end"})

    rejects_df = pd.DataFrame(rejects)
    if not rejects_df.empty and "original_index" not in rejects_df.columns:
        rejects_df = pd.DataFrame(rejects)
    return out, rejects_df


def load_expected_rates(policy: ValidationPolicy) -> pd.DataFrame:
    """
    Read the Excel workbook at policy.workbook.path, sheet policy.workbook.sheet;
    select month_column + mapped expected columns; rename to canonical names;
    normalize (month-end, timezone-naive, percent-to-decimal, float64); enforce
    month_end uniqueness; write rejects to qa/expected_rates_rejects.csv.
    Returns a clean DataFrame unique on month_end.
    """
    path = Path(policy.workbook.path)
    if not path.exists():
        raise FileNotFoundError(f"Validation workbook not found: {path}")

    required = required_expected_excel_columns(policy)
    try:
        raw = pd.read_excel(path, sheet_name=policy.workbook.sheet)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel {path} sheet {policy.workbook.sheet}: {e}") from e

    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Required columns missing in sheet '{policy.workbook.sheet}': {sorted(missing)}. "
            f"Found: {sorted(raw.columns)}"
        )

    month_col = policy.workbook.month_column
    rename_map = {
        policy.workbook.month_column: "month_end",
        policy.expected_columns.asset_growth_rate: "asset_growth_rate",
        policy.expected_columns.organic_growth_rate: "organic_growth_rate",
        policy.expected_columns.external_market_growth_rate: "external_market_growth_rate",
    }
    selected = raw[[month_col] + [policy.expected_columns.asset_growth_rate, policy.expected_columns.organic_growth_rate, policy.expected_columns.external_market_growth_rate]].copy()
    selected = selected.rename(columns=rename_map)

    clean, rejects = normalize_expected_rates_frame(
        selected,
        month_col="month_end",
        rate_columns=["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"],
        month_format=policy.workbook.month_format,
        timezone_naive=policy.normalization.timezone_naive,
        percent_to_decimal=policy.normalization.percent_to_decimal,
        percent_scale=policy.normalization.percent_scale,
    )

    rejects_path = Path(REJECTS_PATH)
    rejects_path.parent.mkdir(parents=True, exist_ok=True)
    if not rejects.empty:
        rejects.to_csv(rejects_path, index=False)
        logger.warning("Wrote %d rejected rows to %s", len(rejects), rejects_path)
    else:
        if rejects_path.exists():
            rejects_path.unlink()

    return clean.reset_index(drop=True)
