"""
Fail-fast QA checks for each aggregated table. On failure writes qa/agg_qa_fail_<name>.json and raises.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.agg_policy_contract import AggPolicy, summarize_agg_policy

logger = logging.getLogger(__name__)

QA_DIR = "qa"
FAIL_PREFIX = "agg_qa_fail_"
SAMPLE_DUPE_ROWS = 20


class AggQAError(Exception):
    """Raised when an agg table fails QA. Fail context is written to qa/ before raise."""


def _ensure_time_key_col(time_key: str, df: pd.DataFrame, output_name: str) -> None:
    """month_end / time_key: must exist."""
    if time_key not in df.columns:
        raise AggQAError(
            f"QA {output_name!r}: time_key {time_key!r} missing. Columns: {list(df.columns)}"
        )


def validate_month_end(
    df: pd.DataFrame,
    time_key: str,
    output_name: str,
) -> None:
    """time_key must exist, be timezone-naive datetime64[ns], and have no missing values."""
    _ensure_time_key_col(time_key, df, output_name)
    ser = df[time_key]
    if not pd.api.types.is_datetime64_any_dtype(ser):
        raise AggQAError(
            f"QA {output_name!r}: {time_key!r} must be datetime64[ns]; got {ser.dtype}"
        )
    # timezone-naive: dtype has no tz (e.g. datetime64[ns] not datetime64[ns, UTC])
    if hasattr(ser.dtype, "tz") and ser.dtype.tz is not None:
        raise AggQAError(
            f"QA {output_name!r}: {time_key!r} must be timezone-naive (got tz={ser.dtype.tz})"
        )
    nulls = ser.isna()
    if nulls.any():
        n = int(nulls.sum())
        raise AggQAError(
            f"QA {output_name!r}: {time_key!r} has {n} missing value(s)"
        )


def validate_grain_uniqueness(
    df: pd.DataFrame,
    time_key: str,
    grain_dims: list[str],
    output_name: str,
) -> None:
    """Key columns = [time_key] + grain_dims; assert no duplicates. Raises with sample duplicate keys."""
    key_cols = [time_key] + list(grain_dims)
    for c in key_cols:
        if c not in df.columns:
            raise AggQAError(f"QA {output_name!r}: key column {c!r} missing")
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        n_dup = int(dupes.sum())
        sample_df = df.loc[dupes][key_cols].drop_duplicates().head(SAMPLE_DUPE_ROWS)
        sample_duplicate_keys = sample_df.to_dict(orient="records")
        for row in sample_duplicate_keys:
            for k, v in list(row.items()):
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat() if v is not None else None
        raise AggQAError(
            f"QA {output_name!r}: duplicate keys on {key_cols}. Duplicate rows: {n_dup}. "
            f"Sample keys (up to {SAMPLE_DUPE_ROWS}): {sample_duplicate_keys}"
        )


def validate_rowcount(df: pd.DataFrame, output_name: str) -> None:
    """rowcount > 0."""
    if len(df) == 0:
        raise AggQAError(f"QA {output_name!r}: rowcount is 0")


def validate_additive_finite(
    df: pd.DataFrame,
    additive_cols: list[str],
    output_name: str,
    *,
    allow_nan: bool = False,
) -> None:
    """Additive measures must be finite (no inf). NaN disallowed unless allow_nan."""
    cols = [c for c in additive_cols if c in df.columns]
    if not cols:
        return
    for c in cols:
        inf_count = (df[c] == float("inf")).sum() + (df[c] == float("-inf")).sum()
        if inf_count > 0:
            raise AggQAError(
                f"QA {output_name!r}: additive column {c!r} has {int(inf_count)} inf value(s)"
            )
    if not allow_nan:
        for c in cols:
            nan_count = df[c].isna().sum()
            if nan_count > 0:
                raise AggQAError(
                    f"QA {output_name!r}: additive column {c!r} has {int(nan_count)} NaN(s); policy disallows NaN"
                )


def write_qa_fail_context(
    root: Path,
    output_name: str,
    reason: str,
    counts: dict[str, Any],
    sample_duplicate_keys: list[dict],
    dtypes: dict[str, str],
    policy_excerpt: dict[str, Any],
) -> Path:
    """Write qa/agg_qa_fail_<name>.json. Returns path written."""
    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / f"{FAIL_PREFIX}{output_name}.json"
    ctx = {
        "table_name": output_name,
        "reason": reason,
        "counts": counts,
        "sample_duplicate_keys": sample_duplicate_keys,
        "dtypes": dtypes,
        "policy_excerpt": policy_excerpt,
    }
    path.write_text(json.dumps(ctx, indent=2, default=str), encoding="utf-8")
    logger.warning("Wrote QA fail context to %s", path)
    return path


def validate_agg_qa(
    df_agg: pd.DataFrame,
    time_key: str,
    grain_dims: list[str],
    additive_cols: list[str],
    output_name: str,
    policy: AggPolicy,
    root: Path,
    *,
    allow_nan_additive: bool = False,
) -> None:
    """
    Run all QA rules (month_end, grain uniqueness, rowcount, additive finite). Fail-fast: on first
    failure write qa/agg_qa_fail_<name>.json and raise AggQAError.
    """
    counts: dict[str, Any] = {"rowcount": len(df_agg)}
    dtypes = {str(c): str(df_agg.dtypes[c]) for c in df_agg.columns}
    try:
        policy_excerpt = summarize_agg_policy(policy)
    except Exception:
        policy_excerpt = {}

    def fail(reason: str, sample_dupes: list[dict] | None = None) -> None:
        write_qa_fail_context(
            root,
            output_name,
            reason,
            counts,
            sample_dupes or [],
            dtypes,
            policy_excerpt,
        )
        raise AggQAError(reason)

    # 1) month_end
    try:
        validate_month_end(df_agg, time_key, output_name)
    except AggQAError as e:
        fail(str(e))

    # 2) grain uniqueness
    key_cols = [time_key] + list(grain_dims)
    try:
        validate_grain_uniqueness(df_agg, time_key, grain_dims, output_name)
    except AggQAError as e:
        dupes = df_agg.duplicated(subset=key_cols, keep=False)
        sample_df = df_agg.loc[dupes][key_cols].drop_duplicates().head(SAMPLE_DUPE_ROWS)
        sample_list = sample_df.to_dict(orient="records")
        for row in sample_list:
            for k, v in list(row.items()):
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat() if v is not None else None
        counts["n_duplicate_rows"] = int(dupes.sum())
        fail(str(e), sample_dupes=sample_list)

    # 3) rowcount
    try:
        validate_rowcount(df_agg, output_name)
    except AggQAError as e:
        fail(str(e))

    # 4) additive finite
    try:
        validate_additive_finite(
            df_agg, additive_cols, output_name, allow_nan=allow_nan_additive
        )
    except AggQAError as e:
        fail(str(e))
