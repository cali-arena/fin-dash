"""
Reconciliation QA: firm_monthly must reconcile to source global per month (within tolerance).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

QA_DIR = "qa"
RECONCILE_FAIL_CSV = "agg_firm_reconcile_fail.csv"
WORST_N_MONTHS = 50

DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-6
DEFAULT_MEASURE = "end_aum"


class ReconcileError(Exception):
    """Raised when firm_monthly does not reconcile to source within tolerance."""


def load_reconcile_config(root: Path, policy_path: Path | None = None) -> dict:
    """
    Load reconcile config: abs_tol, rel_tol, measure.
    Prefer configs/agg_qa_policy.yml reconcile section, else agg_policy.yml (configs/ or policy_path).
    """
    root = Path(root)
    candidates = [root / "configs" / "agg_qa_policy.yml"]
    if policy_path:
        candidates.append(Path(policy_path))
    candidates.append(root / "configs" / "agg_policy.yml")
    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            rec = (raw or {}).get("reconcile") or (raw or {}).get("agg", {}).get("reconcile")
            if isinstance(rec, dict):
                return {
                    "abs_tol": float(rec.get("abs_tol", DEFAULT_ABS_TOL)),
                    "rel_tol": float(rec.get("rel_tol", DEFAULT_REL_TOL)),
                    "measure": str(rec.get("measure", DEFAULT_MEASURE)).strip() or DEFAULT_MEASURE,
                }
        except Exception:
            pass
    return {
        "abs_tol": DEFAULT_ABS_TOL,
        "rel_tol": DEFAULT_REL_TOL,
        "measure": DEFAULT_MEASURE,
    }


def reconcile_firm_monthly(
    source_df: pd.DataFrame,
    agg_df: pd.DataFrame,
    time_key: str,
    measure: str,
    abs_tol: float,
    rel_tol: float,
    root: Path | None = None,
) -> None:
    """
    Validate firm_monthly aggregation vs source global per month.
    Source is the global slice (e.g. full metrics_monthly); agg is firm_monthly.parquet.
    For each month_end: pass if abs_err <= abs_tol OR rel_err <= rel_tol.
    On any failure: write qa/agg_firm_reconcile_fail.csv (worst 50 months) and raise ReconcileError.
    """
    root = Path(root or ".")
    if time_key not in source_df.columns or measure not in source_df.columns:
        raise ReconcileError(
            f"Source missing required columns: time_key={time_key!r}, measure={measure!r}. "
            f"Source columns: {list(source_df.columns)}"
        )
    if time_key not in agg_df.columns or measure not in agg_df.columns:
        raise ReconcileError(
            f"Agg missing required columns: time_key={time_key!r}, measure={measure!r}. "
            f"Agg columns: {list(agg_df.columns)}"
        )

    # Per-month: source sum (global = all rows), agg sum (firm_monthly usually one row per month)
    source_by_month = source_df.groupby(time_key, dropna=False)[measure].sum()
    agg_by_month = agg_df.groupby(time_key, dropna=False)[measure].sum()

    months = source_by_month.index.union(agg_by_month.index)
    rows = []
    failures = []
    for m in months:
        src_val = float(source_by_month.get(m, 0.0))
        agg_val = float(agg_by_month.get(m, 0.0))
        diff = agg_val - src_val
        abs_err = abs(diff)
        denom = max(1e-12, abs(src_val))
        rel_err = abs_err / denom
        passed = abs_err <= abs_tol or rel_err <= rel_tol
        rows.append({
            "month_end": m,
            "source": src_val,
            "agg": agg_val,
            "diff": diff,
            "abs_err": abs_err,
            "rel_err": rel_err,
        })
        if not passed:
            failures.append({
                "month_end": m,
                "source": src_val,
                "agg": agg_val,
                "diff": diff,
                "abs_err": abs_err,
                "rel_err": rel_err,
            })

    if not failures:
        return

    # Worst 50 by abs_err descending
    fail_df = pd.DataFrame(failures)
    fail_df = fail_df.sort_values("abs_err", ascending=False).head(WORST_N_MONTHS)
    out_path = root / QA_DIR / RECONCILE_FAIL_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail_df.to_csv(out_path, index=False)
    logger.warning("Wrote reconcile failures to %s (%d months)", out_path, len(fail_df))
    raise ReconcileError(
        f"firm_monthly reconciliation failed for {len(failures)} month(s). "
        f"Worst {min(len(failures), WORST_N_MONTHS)} written to {out_path}. "
        f"abs_tol={abs_tol}, rel_tol={rel_tol}, measure={measure!r}"
    )
