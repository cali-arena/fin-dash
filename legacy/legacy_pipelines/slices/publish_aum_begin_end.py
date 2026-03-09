"""
Publish aum_begin_end_by_slice from aum_series_all with first_row_rule applied.
Output: curated/aum_begin_end_by_slice.parquet + meta.json. QA gates and QA outputs from qa_thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.contracts.aum_rules_contract import load_aum_rules, validate_aum_rules

logger = logging.getLogger(__name__)

DEFAULT_AUM_RULES_CONFIG = "configs/aum_rules.yml"
INTERMEDIATE_DIR = "curated/intermediate"
CURATED_DIR = "curated"
QA_DIR = "qa"
AUM_SERIES_ALL = "aum_series_all.parquet"
AUM_SERIES_ALL_META = "aum_series_all.meta.json"

OUTPUT_ARTIFACT = "aum_begin_end_by_slice.parquet"
OUTPUT_META = "aum_begin_end_by_slice.meta.json"
QA_MOM_RATIO_OUTLIERS = "aum_mom_ratio_outliers.csv"
QA_SPIKE_TOP200 = "aum_spike_top200.csv"
QA_LAG_MISMATCH_SAMPLES = "aum_lag_mismatch_samples.csv"
QA_QUALITY_JSON = "aum_begin_end_quality.json"
SPIKE_TOP_N = 200
LAG_TOLERANCE = 1e-6
FACT_MONTHLY_REL = "curated/fact_monthly.parquet"
FACT_MONTHLY_ENRICHED_REL = "curated/fact_monthly_enriched.parquet"

COL_MONTH_END = "month_end"
OUT_END_AUM = "end_aum"
OUT_BEGIN_AUM = "begin_aum"
IS_FIRST_MONTH_IN_SLICE = "is_first_month_in_slice"
MONTHS_IN_SLICE = "months_in_slice"

OUTPUT_COLUMNS = [
    "path_id",
    "slice_id",
    "slice_key",
    COL_MONTH_END,
    "preferred_label",
    "product_ticker",
    "src_country_canonical",
    "product_country_canonical",
    OUT_END_AUM,
    OUT_BEGIN_AUM,
    IS_FIRST_MONTH_IN_SLICE,
    MONTHS_IN_SLICE,
]
PRIMARY_KEY = ["path_id", "slice_id", COL_MONTH_END]


def _dataset_version_inputs(root: Path, aum_series_dir: Path, rules_path: Path) -> str:
    """Build string for dataset_version: aum_series meta version + rules content hash + PIPELINE_VERSION."""
    parts = []

    meta_path = aum_series_dir / AUM_SERIES_ALL_META
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            parts.append(meta.get("dataset_version", ""))
        except Exception:
            pass
    if not parts or not parts[0]:
        try:
            df = pd.read_parquet(aum_series_dir / AUM_SERIES_ALL)
            schema_parts = [f"{c}:{str(df[c].dtype)}" for c in sorted(df.columns)]
            parts.append(hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest())
        except Exception:
            parts.append("")

    rules_content = Path(rules_path).read_text(encoding="utf-8")
    parts.append(hashlib.sha1(rules_content.encode("utf-8")).hexdigest())
    parts.append(os.environ.get("PIPELINE_VERSION", "dev"))

    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


def _lag_alignment_audit(
    out: pd.DataFrame,
    qa_thresholds: dict[str, Any],
    qa_dir: Path,
) -> None:
    """
    Spot-sample N slices per path_id (seed from config). For each sampled slice, verify
    begin_aum == prior_row.end_aum within LAG_TOLERANCE for all non-first-month rows.
    On any mismatch: write qa/aum_lag_mismatch_samples.csv and raise.
    """
    sample_slices = int(qa_thresholds.get("sample_slices", 50))
    random_seed = qa_thresholds.get("random_seed", 42)
    rng = random.Random(random_seed)

    mismatches: list[dict[str, Any]] = []
    for path_id in out["path_id"].unique():
        sub = out[out["path_id"] == path_id]
        slice_ids = sub["slice_id"].unique().tolist()
        if not slice_ids:
            continue
        n = min(sample_slices, len(slice_ids))
        chosen = rng.sample(slice_ids, n)
        for sid in chosen:
            sl = sub[sub["slice_id"] == sid].sort_values(COL_MONTH_END, kind="mergesort")
            prev_end = None
            for _, row in sl.iterrows():
                if row[IS_FIRST_MONTH_IN_SLICE]:
                    prev_end = row[OUT_END_AUM]
                    continue
                expected = prev_end
                actual = row[OUT_BEGIN_AUM]
                if expected is None or (pd.isna(expected) and pd.isna(actual)):
                    prev_end = row[OUT_END_AUM]
                    continue
                if pd.isna(expected):
                    expected = float("nan")
                else:
                    expected = float(expected)
                if pd.isna(actual):
                    actual = float("nan")
                else:
                    actual = float(actual)
                if pd.isna(expected) and pd.isna(actual):
                    prev_end = row[OUT_END_AUM]
                    continue
                if pd.isna(expected) or pd.isna(actual) or abs(expected - actual) > LAG_TOLERANCE:
                    mismatches.append({
                        "path_id": path_id,
                        "slice_id": sid,
                        COL_MONTH_END: row[COL_MONTH_END],
                        "expected_begin": expected,
                        "actual_begin": actual,
                        "prior_end": prev_end,
                    })
                prev_end = row[OUT_END_AUM]

    if mismatches:
        qa_dir.mkdir(parents=True, exist_ok=True)
        mismatch_path = qa_dir / QA_LAG_MISMATCH_SAMPLES
        pd.DataFrame(mismatches).to_csv(mismatch_path, index=False, date_format="%Y-%m-%d")
        raise ValueError(
            f"Lag alignment audit: {len(mismatches)} mismatch(es). begin_aum must equal prior row end_aum within {LAG_TOLERANCE}. See {mismatch_path}"
        )


def _enrich_fact_monthly_if_requested(
    root: Path,
    out: pd.DataFrame,
    output_cfg: dict[str, Any],
) -> None:
    """
    If output.enrich_fact_monthly=true: verify fact grain has path_id, slice_id, month_end;
    left join begin_aum (and end_aum) onto fact, write curated/fact_monthly_enriched.parquet.
    If grain does not match, raise with clear message and point to aum_begin_end_by_slice.parquet.
    """
    if not output_cfg.get("enrich_fact_monthly", False):
        return

    fact_path = root / FACT_MONTHLY_REL
    if not fact_path.exists():
        raise FileNotFoundError(
            f"enrich_fact_monthly=true but fact not found: {fact_path}. "
            f"Use curated/aum_begin_end_by_slice.parquet as the official artifact."
        )
    fact = pd.read_parquet(fact_path)

    join_keys = ["path_id", "slice_id", COL_MONTH_END]
    missing = [k for k in join_keys if k not in fact.columns]
    if missing:
        raise ValueError(
            f"Enrichment refused: fact_monthly grain does not match slice grain. "
            f"fact must contain path_id, slice_id, month_end for safe left join. Missing: {missing}. "
            f"Enrichment is unsafe when fact has a different grain. Use curated/aum_begin_end_by_slice.parquet as the official artifact."
        )

    fact_work = fact.drop(columns=["begin_aum", "end_aum"], errors="ignore")
    enrich_cols = out[join_keys + [OUT_BEGIN_AUM, OUT_END_AUM]].copy()
    enrich_cols = enrich_cols.rename(columns={OUT_BEGIN_AUM: "begin_aum", OUT_END_AUM: "end_aum"})
    fact_enriched = fact_work.merge(enrich_cols, on=join_keys, how="left", sort=False)
    out_path = root / FACT_MONTHLY_ENRICHED_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fact_enriched.to_parquet(out_path, index=False)
    logger.info("Wrote %s (enriched with begin_aum/end_aum)", out_path)


def run(root: Path, rules_config_path: Path) -> pd.DataFrame:
    """
    Load aum_series_all, apply first_row_rule where is_first_month_in_slice, output parquet + meta.
    Returns the published DataFrame (keyed by path_id, slice_id, month_end; unique).
    """
    rules = load_aum_rules(rules_config_path)
    validate_aum_rules(rules)

    inputs = rules.get("inputs") or {}
    aum_series_rel = inputs.get("aum_series", f"{INTERMEDIATE_DIR}/{AUM_SERIES_ALL}")
    aum_series_path = root / aum_series_rel.replace("\\", "/").lstrip("/")
    if not aum_series_path.exists():
        raise FileNotFoundError(f"AUM series not found: {aum_series_path}")

    df = pd.read_parquet(aum_series_path)

    for c in [OUT_END_AUM, OUT_BEGIN_AUM]:
        if c not in df.columns:
            raise ValueError(f"aum_series must have column {c}")
    if IS_FIRST_MONTH_IN_SLICE not in df.columns:
        raise ValueError(f"aum_series must have column {IS_FIRST_MONTH_IN_SLICE!r}")

    df[OUT_END_AUM] = df[OUT_END_AUM].astype(float, errors="ignore")
    df[OUT_BEGIN_AUM] = df[OUT_BEGIN_AUM].astype(float, errors="ignore")

    first_row = rules.get("first_row_rule") or {}
    mode = (first_row.get("mode") or "nan").strip().lower()
    if mode not in ("nan", "zero", "carry_end"):
        mode = "nan"

    mask_first = df[IS_FIRST_MONTH_IN_SLICE] == True
    if mode == "zero":
        df.loc[mask_first, OUT_BEGIN_AUM] = 0.0
    elif mode == "carry_end":
        df.loc[mask_first, OUT_BEGIN_AUM] = df.loc[mask_first, OUT_END_AUM].values

    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[OUTPUT_COLUMNS].copy()

    qa_dir = root / QA_DIR
    qa_thresholds = rules.get("qa_thresholds") or {}
    forbid_negative = qa_thresholds.get("forbid_negative", True)
    max_abs_aum = float(qa_thresholds.get("max_abs_aum", 1e13))
    max_mom_ratio = float(qa_thresholds.get("max_month_over_month_ratio", 10.0))

    negative_count = 0
    over_max_abs_count = 0

    if forbid_negative:
        neg_end = (out[OUT_END_AUM].notna()) & (out[OUT_END_AUM] < 0)
        neg_begin = (out[OUT_BEGIN_AUM].notna()) & (out[OUT_BEGIN_AUM] < 0)
        if neg_end.any() or neg_begin.any():
            negative_count = (neg_end | neg_begin).sum()
            raise ValueError(
                f"QA forbid_negative: {int(negative_count)} row(s) with end_aum < 0 or begin_aum < 0"
            )

    if max_abs_aum > 0:
        over_end = (out[OUT_END_AUM].notna()) & (out[OUT_END_AUM].abs() > max_abs_aum)
        over_begin = (out[OUT_BEGIN_AUM].notna()) & (out[OUT_BEGIN_AUM].abs() > max_abs_aum)
        if over_end.any() or over_begin.any():
            over_max_abs_count = (over_end | over_begin).sum()
            raise ValueError(
                f"QA max_abs_aum: {int(over_max_abs_count)} row(s) with |end_aum| or |begin_aum| > {max_abs_aum}"
            )

    if out.duplicated(subset=PRIMARY_KEY, keep=False).any():
        raise ValueError("Output grain (path_id, slice_id, month_end) must be unique")
    key_unique = True

    _lag_alignment_audit(out, qa_thresholds, qa_dir)

    build_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    ratio_outliers_count = 0
    mom_mask = (~out[IS_FIRST_MONTH_IN_SLICE]) & (out[OUT_BEGIN_AUM].notna()) & (out[OUT_BEGIN_AUM] > 0)
    if mom_mask.any() and max_mom_ratio > 0:
        ratio = out.loc[mom_mask, OUT_END_AUM] / out.loc[mom_mask, OUT_BEGIN_AUM]
        outlier = (ratio > max_mom_ratio) | (ratio < (1.0 / max_mom_ratio))
        if outlier.any():
            ratio_outliers_count = int(outlier.sum())
            mom_df = out.loc[mom_mask].copy()
            mom_df["ratio"] = mom_df[OUT_END_AUM] / mom_df[OUT_BEGIN_AUM]
            mom_outliers = mom_df[(mom_df["ratio"] > max_mom_ratio) | (mom_df["ratio"] < (1.0 / max_mom_ratio))]
            mom_outliers = mom_outliers[["slice_id", COL_MONTH_END, OUT_BEGIN_AUM, OUT_END_AUM, "ratio"]].sort_values(["slice_id", COL_MONTH_END], kind="mergesort")
            qa_dir.mkdir(parents=True, exist_ok=True)
            mom_outliers.to_csv(qa_dir / QA_MOM_RATIO_OUTLIERS, index=False, date_format="%Y-%m-%d")
            logger.warning("QA: %d month-over-month ratio outlier(s) -> %s", ratio_outliers_count, QA_MOM_RATIO_OUTLIERS)

    denom = out[OUT_BEGIN_AUM].fillna(1).clip(lower=1)
    pct_change = (out[OUT_END_AUM] - out[OUT_BEGIN_AUM]) / denom
    spike = out[["slice_id", COL_MONTH_END, OUT_BEGIN_AUM, OUT_END_AUM]].copy()
    spike["pct_change"] = pct_change
    spike["_abs"] = spike["pct_change"].abs()
    spike = spike.sort_values(["_abs", "slice_id", COL_MONTH_END], ascending=[False, True, True], kind="mergesort").head(SPIKE_TOP_N)
    spike = spike.drop(columns=["_abs"])
    qa_dir.mkdir(parents=True, exist_ok=True)
    spike.to_csv(qa_dir / QA_SPIKE_TOP200, index=False, date_format="%Y-%m-%d")

    quality = {
        "total_rows": len(out),
        "negative_count": int(negative_count),
        "over_max_abs_count": int(over_max_abs_count),
        "ratio_outliers_count": ratio_outliers_count,
        "first_row_rule_mode": mode,
        "build_timestamp": build_timestamp,
    }
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / QA_QUALITY_JSON).write_text(json.dumps(quality, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote QA %s", QA_QUALITY_JSON)

    schema_parts = [f"{c}:{str(out[c].dtype)}" for c in OUTPUT_COLUMNS]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    dataset_version = _dataset_version_inputs(root, aum_series_path.parent, rules_config_path)

    output_cfg = rules.get("output") or {}
    artifact_rel = output_cfg.get("artifact_path", f"{CURATED_DIR}/{OUTPUT_ARTIFACT}")
    artifact_path = root / artifact_rel.replace("\\", "/").lstrip("/")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=artifact_path.parent, prefix="aum_begin_end_")
    try:
        os.close(fd)
        out.to_parquet(tmp, index=False)
        os.replace(tmp, artifact_path)
    except Exception:
        if Path(tmp).exists():
            Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("Wrote %s (%d rows)", artifact_path, len(out))

    if output_cfg.get("write_meta", True):
        meta = {
            "row_count": len(out),
            "primary_key": PRIMARY_KEY,
            "key_unique": key_unique,
            "schema_hash": schema_hash,
            "dataset_version": dataset_version,
        }
        meta_path = artifact_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Wrote %s", meta_path)

    _enrich_fact_monthly_if_requested(root, out, output_cfg)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish aum_begin_end_by_slice from aum_series_all with first_row_rule applied."
    )
    parser.add_argument("--run", action="store_true", help="Run publish and write curated artifact + meta")
    parser.add_argument("--config", default=DEFAULT_AUM_RULES_CONFIG, help="Path to aum_rules.yml")
    parser.add_argument("--root", type=Path, default=None, help="Project root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = Path(args.root) if args.root is not None else Path.cwd()
    if not args.run:
        logger.info("Use --run to publish aum_begin_end_by_slice.")
        return 0

    try:
        run(root, root / args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
