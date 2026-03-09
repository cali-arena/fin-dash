"""
QA checks for firm-level recomputed series: identity, guard alignment, inf/NaN sweep.

Runs after recomputation; reads curated/qa/firm_level_recomputed.parquet (or accepts DataFrame).
Writes: firm_identity_violations.csv, firm_guard_violations.csv, firm_nan_summary.json.

Thresholds from configs/firm_qa_policy.yml (or defaults).
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_QA_POLICY_PATH = "configs/firm_qa_policy.yml"
QA_DIR_NAME = "curated/qa"
FIRM_RECOMPUTED_NAME = "firm_level_recomputed.parquet"
IDENTITY_VIOLATIONS_CSV = "firm_identity_violations.csv"
GUARD_VIOLATIONS_CSV = "firm_guard_violations.csv"
NAN_SUMMARY_JSON = "firm_nan_summary.json"

RATE_COLUMNS = ["asset_growth_rate", "organic_growth_rate", "external_market_growth_rate"]

DEFAULT_QA_CONFIG = {
    "identity": {"tol_pct": 1e-6, "atol": 1e-6, "max_violations_to_report": 50},
    "guard": {"threshold": 0.0},
    "rate_columns": list(RATE_COLUMNS),
}


def load_qa_policy(path: str | Path | None = None) -> dict[str, Any]:
    """Load firm QA policy YAML. Returns default config if file missing or invalid."""
    path = path or Path(DEFAULT_QA_POLICY_PATH)
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_QA_CONFIG)
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not load firm_qa_policy %s: %s; using defaults.", path, e)
        return dict(DEFAULT_QA_CONFIG)
    if not isinstance(raw, dict):
        return dict(DEFAULT_QA_CONFIG)
    identity = (raw.get("identity") or {}).copy()
    identity.setdefault("tol_pct", DEFAULT_QA_CONFIG["identity"]["tol_pct"])
    identity.setdefault("atol", DEFAULT_QA_CONFIG["identity"]["atol"])
    identity.setdefault("max_violations_to_report", DEFAULT_QA_CONFIG["identity"]["max_violations_to_report"])
    guard = (raw.get("guard") or {}).copy()
    guard.setdefault("threshold", DEFAULT_QA_CONFIG["guard"]["threshold"])
    rate_cols = raw.get("rate_columns") or RATE_COLUMNS
    return {"identity": identity, "guard": guard, "rate_columns": list(rate_cols)}


def run_identity_check(
    df: pd.DataFrame,
    config: dict[str, Any],
    qa_dir: Path,
) -> int:
    """
    Identity: end_aum_firm ≈ begin_aum_firm + nnb_firm + market_pnl_firm.
    tolerance = max(atol, tol_pct * max(1, abs(end_aum_firm))).
    Returns violation count. Writes worst max_violations_to_report to firm_identity_violations.csv.
    """
    required = ["month_end", "begin_aum_firm", "end_aum_firm", "nnb_firm", "market_pnl_firm"]
    if not all(c in df.columns for c in required):
        logger.warning("Identity check skipped: missing columns %s", [c for c in required if c not in df.columns])
        return 0
    identity_cfg = config.get("identity") or {}
    atol = float(identity_cfg.get("atol", 1e-6))
    tol_pct = float(identity_cfg.get("tol_pct", 1e-6))
    max_report = int(identity_cfg.get("max_violations_to_report", 50))

    b = df["begin_aum_firm"].astype("float64")
    e = df["end_aum_firm"].astype("float64")
    nnb = df["nnb_firm"].astype("float64")
    pnl = df["market_pnl_firm"].astype("float64")
    resid = e - (b + nnb + pnl)
    ref = e.abs().clip(lower=1.0)
    tol = pd.Series(atol, index=df.index).combine(ref * tol_pct, max)
    viol_mask = resid.abs() > tol

    n = int(viol_mask.sum())
    if n == 0:
        return 0
    cols = [c for c in ["month_end", "begin_aum_firm", "end_aum_firm", "nnb_firm", "market_pnl_firm"] if c in df.columns]
    viol_df = df.loc[viol_mask, cols].copy()
    viol_df["resid"] = resid[viol_mask].values
    viol_df["abs_resid"] = resid[viol_mask].abs().values
    viol_df["tolerance"] = tol[viol_mask].values
    viol_df = viol_df.sort_values("abs_resid", ascending=False, kind="mergesort").head(max_report)
    out_path = qa_dir / IDENTITY_VIOLATIONS_CSV
    viol_df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    logger.warning("Identity check: %d violation(s); worst %d written to %s", n, len(viol_df), out_path)
    return n


def run_guard_check(
    df: pd.DataFrame,
    config: dict[str, Any],
    qa_dir: Path,
) -> int:
    """
    Guard alignment: if begin_aum_firm <= threshold then rate columns must be NaN (policy outcome).
    Count violations; write rows where guarded but rate is non-NaN to firm_guard_violations.csv.
    """
    rate_cols = [c for c in (config.get("rate_columns") or RATE_COLUMNS) if c in df.columns]
    if not rate_cols:
        return 0
    guard_cfg = config.get("guard") or {}
    threshold = float(guard_cfg.get("threshold", 0.0))
    b = df["begin_aum_firm"].astype("float64") if "begin_aum_firm" in df.columns else pd.Series(0.0, index=df.index)
    guarded = (b <= threshold) & b.notna()
    # Violation: guarded row but any rate is not NaN
    rate_not_nan = df[rate_cols[0]].notna()
    for c in rate_cols[1:]:
        rate_not_nan = rate_not_nan | df[c].notna()
    viol_mask = guarded & rate_not_nan
    n = int(viol_mask.sum())
    if n == 0:
        return 0
    cols = [c for c in ["month_end", "begin_aum_firm"] + rate_cols if c in df.columns]
    viol_df = df.loc[viol_mask, cols].copy()
    viol_df = viol_df.sort_values("month_end", kind="mergesort")
    out_path = qa_dir / GUARD_VIOLATIONS_CSV
    viol_df.to_csv(out_path, index=False, date_format="%Y-%m-%d")
    logger.warning("Guard alignment: %d violation(s) written to %s", n, out_path)
    return n


def run_inf_nan_sweep(
    df: pd.DataFrame,
    config: dict[str, Any],
    qa_dir: Path,
) -> dict[str, Any]:
    """
    Ensure no inf in rate columns; report NaN counts per rate. Writes firm_nan_summary.json.
    Returns summary dict (has_inf: bool, nan_counts: {...}).
    """
    rate_cols = [c for c in (config.get("rate_columns") or RATE_COLUMNS) if c in df.columns]
    nan_counts = {}
    has_inf = False
    for c in rate_cols:
        s = df[c].astype("float64")
        nan_counts[c] = int(s.isna().sum())
        if ((s == math.inf) | (s == -math.inf)).any():
            has_inf = True
    summary = {"has_inf": has_inf, "nan_counts": nan_counts, "rowcount": len(df)}
    out_path = qa_dir / NAN_SUMMARY_JSON
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if has_inf:
        logger.warning("Inf sweep: rate columns contain inf; see %s", out_path)
    return summary


def run_firm_qa(
    df: pd.DataFrame | None = None,
    root: Path | None = None,
    qa_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run all firm-level QA checks. If df is None, loads from curated/qa/firm_level_recomputed.parquet under root.
    Writes firm_identity_violations.csv, firm_guard_violations.csv, firm_nan_summary.json to curated/qa.
    Returns dict with identity_violations, guard_violations, nan_summary.
    """
    root = root or Path.cwd()
    qa_dir = root / "curated" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        parquet_path = qa_dir / FIRM_RECOMPUTED_NAME
        if not parquet_path.exists():
            raise FileNotFoundError(f"Firm-level parquet not found: {parquet_path}. Run recompute_firm_level first.")
        df = pd.read_parquet(parquet_path)
    if df.empty:
        return {"identity_violations": 0, "guard_violations": 0, "nan_summary": {"rowcount": 0, "nan_counts": {}, "has_inf": False}}

    if qa_policy_path is not None:
        config = load_qa_policy(Path(qa_policy_path))
    else:
        config = load_qa_policy(root / DEFAULT_QA_POLICY_PATH)

    n_id = run_identity_check(df, config, qa_dir)
    n_guard = run_guard_check(df, config, qa_dir)
    nan_summary = run_inf_nan_sweep(df, config, qa_dir)
    return {"identity_violations": n_id, "guard_violations": n_guard, "nan_summary": nan_summary}


def main() -> int:
    """CLI: run firm QA from curated/qa/firm_level_recomputed.parquet under --root."""
    import argparse
    parser = argparse.ArgumentParser(description="Run firm-level QA checks; write violations and nan summary to curated/qa/")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--policy", default=None, help="Path to firm_qa_policy.yml (default: configs/firm_qa_policy.yml)")
    args = parser.parse_args()
    try:
        result = run_firm_qa(df=None, root=args.root, qa_policy_path=args.policy)
        print("Firm QA done:", result)
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Firm QA failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
