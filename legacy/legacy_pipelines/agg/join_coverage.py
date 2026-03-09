"""
Join coverage diagnostics: pct missing channel_l1/segment after dim joins.
Warn above warn_threshold (write qa/agg_join_coverage_warnings.json); fail above fail_threshold.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

QA_DIR = "qa"
WARNINGS_FILENAME = "agg_join_coverage_warnings.json"
SAMPLE_MISS_ROWS = 50
DEFAULT_WARN_THRESHOLD = 0.001
DEFAULT_FAIL_THRESHOLD = 0.05


class JoinCoverageError(Exception):
    """Raised when join coverage missing pct exceeds fail_threshold."""


def _serialize_sample(rows: list[dict]) -> list[dict]:
    """Convert timestamps etc. to JSON-serializable."""
    out = []
    for row in rows:
        r = {}
        for k, v in row.items():
            if hasattr(v, "isoformat") and v is not None:
                r[k] = v.isoformat()
            elif pd.isna(v):
                r[k] = None
            else:
                r[k] = v
        out.append(r)
    return out


def load_join_coverage_config(root: Path, policy_path: Path | None = None) -> tuple[float, float]:
    """Load join_coverage.warn_threshold and fail_threshold. Prefer configs/agg_qa_policy.yml, then policy_path / agg_policy."""
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
            jc = raw.get("join_coverage") if isinstance(raw, dict) else None
            if not isinstance(jc, dict):
                continue
            warn = float(jc.get("warn_threshold", DEFAULT_WARN_THRESHOLD))
            fail = float(jc.get("fail_threshold", DEFAULT_FAIL_THRESHOLD))
            return warn, fail
        except Exception:
            continue
    return DEFAULT_WARN_THRESHOLD, DEFAULT_FAIL_THRESHOLD


def compute_join_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """
    After dim joins: pct_missing_channel_l1, pct_missing_segment, counts, sample rows (50 each).
    Only considers columns that exist. Uses mean(isnull) for pct.
    """
    total = len(df)
    if total == 0:
        return {
            "total_rows": 0,
            "pct_missing_channel_l1": 0.0,
            "pct_missing_segment": 0.0,
            "n_missing_channel_l1": 0,
            "n_missing_segment": 0,
            "sample_channel_l1_misses": [],
            "sample_segment_misses": [],
        }

    out: dict[str, Any] = {"total_rows": total}

    # Channel
    if "channel_l1" in df.columns:
        null_ch = df["channel_l1"].isna()
        n_miss_ch = int(null_ch.sum())
        pct_ch = n_miss_ch / total
        out["n_missing_channel_l1"] = n_miss_ch
        out["pct_missing_channel_l1"] = round(pct_ch, 8)
        sample_cols = ["preferred_label"] + [c for c in ("channel_raw", "channel_standard", "channel_best") if c in df.columns]
        sample_cols = [c for c in sample_cols if c in df.columns]
        if sample_cols and null_ch.any():
            sample_df = df.loc[null_ch, sample_cols].head(SAMPLE_MISS_ROWS)
            out["sample_channel_l1_misses"] = _serialize_sample(sample_df.to_dict(orient="records"))
        else:
            out["sample_channel_l1_misses"] = []
    else:
        out["n_missing_channel_l1"] = 0
        out["pct_missing_channel_l1"] = 0.0
        out["sample_channel_l1_misses"] = []

    # Segment
    if "segment" in df.columns:
        null_seg = df["segment"].isna()
        n_miss_seg = int(null_seg.sum())
        pct_seg = n_miss_seg / total
        out["n_missing_segment"] = n_miss_seg
        out["pct_missing_segment"] = round(pct_seg, 8)
        if "product_ticker" in df.columns and null_seg.any():
            sample_df = df.loc[null_seg, ["product_ticker"]].head(SAMPLE_MISS_ROWS)
            out["sample_segment_misses"] = _serialize_sample(sample_df.to_dict(orient="records"))
        else:
            out["sample_segment_misses"] = []
    else:
        out["n_missing_segment"] = 0
        out["pct_missing_segment"] = 0.0
        out["sample_segment_misses"] = []

    return out


def write_join_coverage_warnings(root: Path, coverage: dict[str, Any]) -> Path:
    """Write qa/agg_join_coverage_warnings.json with counts + sample keys."""
    qa_dir = root / QA_DIR
    qa_dir.mkdir(parents=True, exist_ok=True)
    path = qa_dir / WARNINGS_FILENAME
    path.write_text(json.dumps(coverage, indent=2, default=str), encoding="utf-8")
    logger.warning("Wrote join coverage warnings to %s", path)
    return path


def check_join_coverage(
    df: pd.DataFrame,
    policy_path: Path | None,
    root: Path,
) -> dict[str, Any]:
    """
    Compute coverage; if pct > fail_threshold write warnings then raise JoinCoverageError;
    if pct > warn_threshold write warnings. Returns coverage dict for manifest/meta enrichment.
    """
    warn_threshold, fail_threshold = load_join_coverage_config(root, policy_path)
    coverage = compute_join_coverage(df)

    pct_ch = coverage.get("pct_missing_channel_l1", 0.0)
    pct_seg = coverage.get("pct_missing_segment", 0.0)

    if pct_ch > fail_threshold or pct_seg > fail_threshold:
        write_join_coverage_warnings(root, coverage)
        raise JoinCoverageError(
            f"Join coverage exceeds fail_threshold={fail_threshold}. "
            f"pct_missing_channel_l1={pct_ch}, pct_missing_segment={pct_seg}. "
            f"Wrote {root / QA_DIR / WARNINGS_FILENAME}"
        )

    if pct_ch > warn_threshold or pct_seg > warn_threshold:
        write_join_coverage_warnings(root, coverage)

    return coverage
