"""
Star join contract: fact is authoritative; dims are lookup/enrichment only.
No ad-hoc merges in UI; all joins via this module with explicit keys.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from legacy.legacy_pipelines.dimensions.build_dimensions import normalize_country

logger = logging.getLogger(__name__)

# Paths (relative to project root or cwd)
FACT_PATH = "curated/fact_monthly.parquet"
DIM_PATHS = {
    "dim_time": "curated/dim_time.parquet",
    "dim_channel": "curated/dim_channel.parquet",
    "dim_product": "curated/dim_product.parquet",
    "dim_geo": "curated/dim_geo.parquet",
}
FACT_ENRICHED_PATH = "curated/fact_enriched.parquet"
QA_JOIN_COVERAGE_PATH = "qa/join_coverage.json"

# Explicit join keys (fact column -> dim column or (fact_col, dim_col))
JOIN_TIME_FACT = "month_end"
JOIN_TIME_DIM = "month_end"

JOIN_PRODUCT_FACT = "product_ticker"
JOIN_PRODUCT_DIM = "product_ticker"

# Channel: prefer channel_key if present in both; else preferred_label
JOIN_CHANNEL_KEY = "channel_key"
JOIN_CHANNEL_LABEL = "preferred_label"

# Geo: fact uses canonical country -> dim_geo.country (two joins: src and product)
SRC_COUNTRY_CANONICAL = "src_country_canonical"
PRODUCT_COUNTRY_CANONICAL = "product_country_canonical"
GEO_DIM_COUNTRY = "country"


def _root_path(root: Path | None) -> Path:
    return Path(root) if root is not None else Path.cwd()


def _ensure_geo_canonical(fact: pd.DataFrame) -> pd.DataFrame:
    """Ensure fact has src_country_canonical and product_country_canonical; compute from raw if missing."""
    fact = fact.copy()
    if SRC_COUNTRY_CANONICAL not in fact.columns and "src_country" in fact.columns:
        fact[SRC_COUNTRY_CANONICAL] = fact["src_country"].astype(object).fillna("").astype(str).apply(normalize_country)
    if PRODUCT_COUNTRY_CANONICAL not in fact.columns and "product_country" in fact.columns:
        fact[PRODUCT_COUNTRY_CANONICAL] = fact["product_country"].astype(object).fillna("").astype(str).apply(normalize_country)
    return fact


def _channel_join_key(fact: pd.DataFrame, dim_channel: pd.DataFrame) -> str:
    """Use channel_key if present in both; else preferred_label."""
    has_key_f = JOIN_CHANNEL_KEY in fact.columns
    has_key_d = JOIN_CHANNEL_KEY in dim_channel.columns and len(dim_channel) > 0
    if has_key_f and has_key_d:
        return JOIN_CHANNEL_KEY
    return JOIN_CHANNEL_LABEL


def _validate_required_columns(fact: pd.DataFrame, dims: dict[str, pd.DataFrame], channel_join_col: str) -> None:
    """Raise if required fact columns for joins are missing (only for dims that are non-empty)."""
    if "dim_time" in dims and len(dims["dim_time"]) > 0:
        if JOIN_TIME_FACT not in fact.columns:
            raise ValueError(f"Fact missing required column for time join: {JOIN_TIME_FACT}")
    if "dim_product" in dims and len(dims["dim_product"]) > 0:
        if JOIN_PRODUCT_FACT not in fact.columns:
            raise ValueError(f"Fact missing required column for product join: {JOIN_PRODUCT_FACT}")
    if "dim_channel" in dims and len(dims["dim_channel"]) > 0:
        if channel_join_col not in fact.columns:
            raise ValueError(f"Fact missing required column for channel join: {channel_join_col}")
    if "dim_geo" in dims and len(dims["dim_geo"]) > 0:
        if SRC_COUNTRY_CANONICAL not in fact.columns:
            raise ValueError(f"Fact missing {SRC_COUNTRY_CANONICAL} (or src_country for derivation)")
        if PRODUCT_COUNTRY_CANONICAL not in fact.columns:
            raise ValueError(f"Fact missing {PRODUCT_COUNTRY_CANONICAL} (or product_country for derivation)")


def load_fact_enriched(
    root: Path | None = None,
    fact_path: str | Path = FACT_PATH,
    dim_paths: dict[str, str] | None = None,
    write_output: bool = False,
    output_path: str | Path = FACT_ENRICHED_PATH,
) -> pd.DataFrame:
    """
    Load fact + dims, validate required columns, join ONLY via explicit keys (left joins).
    Fact is authoritative: row count unchanged after joins.
    Adds columns: time_join_ok, channel_join_ok, product_join_ok, src_geo_join_ok, product_geo_join_ok.
    Optionally writes curated/fact_enriched.parquet.
    """
    root = _root_path(root)
    dim_paths = dim_paths or DIM_PATHS
    fact_file = root / fact_path
    if not fact_file.exists():
        raise FileNotFoundError(f"Fact not found: {fact_file}")
    fact = pd.read_parquet(fact_file)
    fact = _ensure_geo_canonical(fact)
    n_fact = len(fact)

    dims: dict[str, pd.DataFrame] = {}
    for name, rel_path in dim_paths.items():
        path = root / rel_path
        if path.exists():
            dims[name] = pd.read_parquet(path)
        else:
            dims[name] = pd.DataFrame()
            logger.warning("Dim not found (empty join): %s", path)

    channel_join_col = _channel_join_key(fact, dims.get("dim_channel", pd.DataFrame()))
    _validate_required_columns(fact, dims, channel_join_col)

    result = fact.copy()

    # Time (dedupe dim on join key so left join never expands rows)
    time_join_ok = pd.Series(False, index=result.index)
    if "dim_time" in dims and len(dims["dim_time"]) > 0:
        dim_time = dims["dim_time"].drop_duplicates(subset=[JOIN_TIME_DIM], keep="first")
        result = result.merge(
            dim_time,
            on=JOIN_TIME_FACT,
            how="left",
            suffixes=("", "_dim_time"),
        )
        # Match = dim-only column present (quarter is in dim_time)
        if "quarter" in result.columns:
            time_join_ok = result["quarter"].notna()
    result["time_join_ok"] = time_join_ok

    # Channel
    channel_join_ok = pd.Series(False, index=result.index)
    if "dim_channel" in dims and len(dims["dim_channel"]) > 0 and channel_join_col in result.columns:
        dim_ch = dims["dim_channel"].drop_duplicates(subset=[channel_join_col], keep="first")
        result = result.merge(
            dim_ch,
            left_on=channel_join_col,
            right_on=channel_join_col,
            how="left",
            suffixes=("", "_dim_channel"),
        )
        # Matched if any dim-only column is non-null (e.g. channel_l1)
        dim_only = [c for c in dim_ch.columns if c not in (channel_join_col,)]
        if dim_only:
            channel_join_ok = result[dim_only[0]].notna()
        result["channel_join_ok"] = channel_join_ok.values if hasattr(channel_join_ok, "values") else channel_join_ok
    else:
        result["channel_join_ok"] = False

    # Product
    product_join_ok = pd.Series(False, index=result.index)
    if "dim_product" in dims and len(dims["dim_product"]) > 0:
        dim_pr = dims["dim_product"].drop_duplicates(subset=[JOIN_PRODUCT_DIM], keep="first")
        result = result.merge(
            dim_pr,
            on=JOIN_PRODUCT_FACT,
            how="left",
            suffixes=("", "_dim_product"),
        )
        if "product_key" in result.columns:
            product_join_ok = result["product_key"].notna()
        else:
            dim_only = [c for c in dim_pr.columns if c != JOIN_PRODUCT_FACT]
            if dim_only:
                product_join_ok = result[dim_only[0]].notna()
    result["product_join_ok"] = product_join_ok

    # Geo: src
    src_geo_join_ok = pd.Series(False, index=result.index)
    if "dim_geo" in dims and len(dims["dim_geo"]) > 0 and SRC_COUNTRY_CANONICAL in result.columns:
        dim_geo = dims["dim_geo"]
        dim_geo_src = dim_geo.rename(columns={c: f"{c}_src" for c in dim_geo.columns})
        result = result.merge(
            dim_geo_src,
            left_on=SRC_COUNTRY_CANONICAL,
            right_on=f"{GEO_DIM_COUNTRY}_src",
            how="left",
        )
        if "region_src" in result.columns:
            src_geo_join_ok = result["region_src"].notna()
        result["src_geo_join_ok"] = src_geo_join_ok.values if hasattr(src_geo_join_ok, "values") else src_geo_join_ok
    else:
        result["src_geo_join_ok"] = False

    # Geo: product
    product_geo_join_ok = pd.Series(False, index=result.index)
    if "dim_geo" in dims and len(dims["dim_geo"]) > 0 and PRODUCT_COUNTRY_CANONICAL in result.columns:
        dim_geo = dims["dim_geo"].drop_duplicates(subset=[GEO_DIM_COUNTRY], keep="first")
        dim_geo_prd = dim_geo.rename(columns={c: f"{c}_product" for c in dim_geo.columns})
        result = result.merge(
            dim_geo_prd,
            left_on=PRODUCT_COUNTRY_CANONICAL,
            right_on=f"{GEO_DIM_COUNTRY}_product",
            how="left",
        )
        if "region_product" in result.columns:
            product_geo_join_ok = result["region_product"].notna()
        result["product_geo_join_ok"] = product_geo_join_ok.values if hasattr(product_geo_join_ok, "values") else product_geo_join_ok
    else:
        result["product_geo_join_ok"] = False

    if len(result) != n_fact:
        raise AssertionError(f"Join contract violated: fact row count {n_fact} != result {len(result)}")

    if write_output:
        out_path = root / output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d rows)", out_path, len(result))

    return result


def validate_join_coverage(
    fact: pd.DataFrame,
    dims: dict[str, pd.DataFrame],
    root: Path | None = None,
    output_path: str | Path = QA_JOIN_COVERAGE_PATH,
    top_missing_n: int = 20,
) -> None:
    """
    Compute join coverage rates and top missing keys; write qa/join_coverage.json.
    """
    root = _root_path(root)
    fact = _ensure_geo_canonical(fact)
    report: dict[str, Any] = {"joins": {}, "top_missing_keys": {}}

    n = len(fact)

    # Time
    if "dim_time" in dims and len(dims["dim_time"]) > 0 and JOIN_TIME_FACT in fact.columns:
        dim_time = dims["dim_time"]
        valid = fact[JOIN_TIME_FACT].notna()
        in_dim = fact[JOIN_TIME_FACT].isin(dim_time[JOIN_TIME_DIM])
        matched = valid & in_dim
        rate = matched.sum() / n if n else 0.0
        report["joins"]["time"] = {"key_fact": JOIN_TIME_FACT, "key_dim": JOIN_TIME_DIM, "coverage_rate": round(float(rate), 6), "matched": int(matched.sum()), "total": n}
        missing = fact.loc[valid & ~in_dim, JOIN_TIME_FACT].dropna().unique().tolist()
        report["top_missing_keys"]["time"] = missing[:top_missing_n]
    else:
        report["joins"]["time"] = {"coverage_rate": None, "note": "dim_time empty or missing"}

    # Channel
    channel_join_col = _channel_join_key(fact, dims.get("dim_channel", pd.DataFrame()))
    if "dim_channel" in dims and len(dims["dim_channel"]) > 0 and channel_join_col in fact.columns:
        dim_ch = dims["dim_channel"]
        valid = fact[channel_join_col].notna() & (fact[channel_join_col].astype(str).str.strip() != "")
        in_dim = fact[channel_join_col].isin(dim_ch[channel_join_col])
        matched = valid & in_dim
        rate = matched.sum() / n if n else 0.0
        report["joins"]["channel"] = {"key_fact": channel_join_col, "key_dim": channel_join_col, "coverage_rate": round(float(rate), 6), "matched": int(matched.sum()), "total": n}
        missing = fact.loc[valid & ~in_dim, channel_join_col].dropna().astype(str).unique().tolist()
        report["top_missing_keys"]["channel"] = missing[:top_missing_n]
    else:
        report["joins"]["channel"] = {"coverage_rate": None, "note": "dim_channel empty or missing"}
        report["top_missing_keys"]["channel"] = []

    # Product
    if "dim_product" in dims and len(dims["dim_product"]) > 0 and JOIN_PRODUCT_FACT in fact.columns:
        dim_pr = dims["dim_product"]
        valid = fact[JOIN_PRODUCT_FACT].notna() & (fact[JOIN_PRODUCT_FACT].astype(str).str.strip() != "")
        in_dim = fact[JOIN_PRODUCT_FACT].isin(dim_pr[JOIN_PRODUCT_DIM])
        matched = valid & in_dim
        rate = matched.sum() / n if n else 0.0
        report["joins"]["product"] = {"key_fact": JOIN_PRODUCT_FACT, "key_dim": JOIN_PRODUCT_DIM, "coverage_rate": round(float(rate), 6), "matched": int(matched.sum()), "total": n}
        missing = fact.loc[valid & ~in_dim, JOIN_PRODUCT_FACT].dropna().astype(str).unique().tolist()
        report["top_missing_keys"]["product"] = missing[:top_missing_n]
    else:
        report["joins"]["product"] = {"coverage_rate": None, "note": "dim_product empty or missing"}

    # Geo src
    if "dim_geo" in dims and len(dims["dim_geo"]) > 0 and SRC_COUNTRY_CANONICAL in fact.columns:
        dim_geo = dims["dim_geo"]
        valid = fact[SRC_COUNTRY_CANONICAL].notna() & (fact[SRC_COUNTRY_CANONICAL].astype(str).str.strip() != "")
        in_dim = fact[SRC_COUNTRY_CANONICAL].isin(dim_geo[GEO_DIM_COUNTRY])
        matched = valid & in_dim
        rate = matched.sum() / n if n else 0.0
        report["joins"]["src_geo"] = {"key_fact": SRC_COUNTRY_CANONICAL, "key_dim": GEO_DIM_COUNTRY, "coverage_rate": round(float(rate), 6), "matched": int(matched.sum()), "total": n}
        missing = fact.loc[valid & ~in_dim, SRC_COUNTRY_CANONICAL].dropna().astype(str).unique().tolist()
        report["top_missing_keys"]["src_geo"] = missing[:top_missing_n]
    else:
        report["joins"]["src_geo"] = {"coverage_rate": None, "note": "dim_geo empty or fact missing canonical"}

    # Geo product
    if "dim_geo" in dims and len(dims["dim_geo"]) > 0 and PRODUCT_COUNTRY_CANONICAL in fact.columns:
        dim_geo = dims["dim_geo"]
        valid = fact[PRODUCT_COUNTRY_CANONICAL].notna() & (fact[PRODUCT_COUNTRY_CANONICAL].astype(str).str.strip() != "")
        in_dim = fact[PRODUCT_COUNTRY_CANONICAL].isin(dim_geo[GEO_DIM_COUNTRY])
        matched = valid & in_dim
        rate = matched.sum() / n if n else 0.0
        report["joins"]["product_geo"] = {"key_fact": PRODUCT_COUNTRY_CANONICAL, "key_dim": GEO_DIM_COUNTRY, "coverage_rate": round(float(rate), 6), "matched": int(matched.sum()), "total": n}
        missing = fact.loc[valid & ~in_dim, PRODUCT_COUNTRY_CANONICAL].dropna().astype(str).unique().tolist()
        report["top_missing_keys"]["product_geo"] = missing[:top_missing_n]
    else:
        report["joins"]["product_geo"] = {"coverage_rate": None, "note": "dim_geo empty or fact missing canonical"}

    out_path = root / output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("Wrote %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Star contract: load fact enriched or validate join coverage.")
    parser.add_argument("--build-enriched", action="store_true", help="Load fact + dims, join via contract, write fact_enriched.parquet and join_coverage.json")
    parser.add_argument("--root", type=Path, default=None, help="Project root (default cwd)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

    root = _root_path(args.root)
    if not args.build_enriched:
        logger.info("Use --build-enriched to build fact_enriched.parquet and qa/join_coverage.json")
        return 0

    fact_path = root / FACT_PATH
    if not fact_path.exists():
        logger.error("Fact not found: %s", fact_path)
        return 1
    fact = pd.read_parquet(fact_path)
    fact = _ensure_geo_canonical(fact)
    dims = {}
    for name, rel in DIM_PATHS.items():
        p = root / rel
        dims[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    try:
        enriched = load_fact_enriched(root=root, write_output=True)
        validate_join_coverage(fact, dims, root=root)
    except Exception as e:
        logger.exception("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
