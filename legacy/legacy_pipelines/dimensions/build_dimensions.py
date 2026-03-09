"""
Build dimension tables deterministically from curated/fact_monthly.parquet.
Writes parquet to curated/ and CSV to curated/dims_csv/.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CHANNEL_MAP_STATUS_MAPPED = "MAPPED"
CHANNEL_MAP_STATUS_FALLBACK = "FALLBACK"
CHANNEL_MAP_STATUS_EMPTY = "EMPTY"

logger = logging.getLogger(__name__)

# Default paths (relative to cwd or root)
DEFAULT_FACT_PATH = "curated/fact_monthly.parquet"
DEFAULT_CURATED_DIR = "curated"
DEFAULT_DIMS_CSV_DIR = "curated/dims_csv"

# Column sets expected in fact
FACT_COLS_DIM_TIME = ["month_end"]
FACT_COLS_DIM_CHANNEL = [
    "channel_raw",
    "channel_standard",
    "channel_best",
    "preferred_label",
    "channel_l1",
    "channel_l2",
]
FACT_COLS_DIM_PRODUCT = ["product_ticker", "segment", "sub_segment"]
FACT_COLS_DIM_GEO = ["src_country", "product_country"]

DEFAULT_GEO_REGION_MAP_PATH = "schemas/geo_region_map.csv"
DEFAULT_QA_DIR = "qa"
DEFAULT_MANIFESTS_DIR = "curated/manifests"
UNMAPPED_REGION = "UNMAPPED"


def _dim_strict() -> bool:
    """True if DIM_STRICT env var is true (default False)."""
    return os.environ.get("DIM_STRICT", "false").strip().lower() == "true"


def _str_fill(s: pd.Series) -> pd.Series:
    """Fill null/NA with empty string; coerce to string."""
    return s.astype(object).fillna("").astype(str).str.strip()


# Geo: 2/3-letter ISO-like codes (uppercase); else full uppercase. No fuzzy matching.
_GEO_ISO2 = re.compile(r"^[A-Za-z]{2}$")
_GEO_ISO3 = re.compile(r"^[A-Za-z]{3}$")


def normalize_country(s: str) -> str:
    """
    Canonical country (deterministic): trim, collapse spaces, remove leading/trailing punctuation.
    Convert separators (/ - _) to spaces and collapse again. 2- or 3-letter alpha -> uppercase; else uppercase.
    Used for dim_geo and region join.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    x = str(s).strip()
    for sep in ["/", "-", "_"]:
        x = x.replace(sep, " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return ""
    if _GEO_ISO2.match(x) or _GEO_ISO3.match(x):
        return x.upper()
    return x.upper()


# Segment/sub_segment: code-like (^[A-Z0-9_ -]+$, len<=12) -> uppercase; else title-case; empty -> UNSPECIFIED
_SEGMENT_CODE_PATTERN = re.compile(r"^[A-Z0-9_ \-]+$")

def _normalize_segment_label(series: pd.Series) -> pd.Series:
    """Strip, collapse spaces; code-like (^[A-Z0-9_ -]+$, len<=12) -> uppercase, else title-case; empty -> UNSPECIFIED."""
    s = series.astype(object).fillna("").astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    def _one(val: str) -> str:
        if not val:
            return "UNSPECIFIED"
        if len(val) <= 12 and _SEGMENT_CODE_PATTERN.match(val):
            return val.upper()
        return val.title()
    return s.map(_one).astype("string")


DIM_TIME_COLUMNS = [
    "month_end",
    "year",
    "month",
    "month_name",
    "quarter",
    "year_month",
    "month_start",
    "is_latest_month",
    "is_ytd",
    "is_current_quarter",
]


def build_dim_time(
    fact_df: pd.DataFrame,
    max_month_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Output columns (deterministic order): month_end, year, month, month_name, quarter, year_month,
    month_start, is_latest_month, is_ytd, is_current_quarter.
    month_end/month_start at midnight. year_month = YYYY-MM. month_name = Jan..Dec.
    max_month_end: computed once from fact if not provided.
    """
    if "month_end" not in fact_df.columns:
        logger.warning("build_dim_time: required column 'month_end' missing in fact.")
        return pd.DataFrame(columns=DIM_TIME_COLUMNS)
    df = fact_df[["month_end"]].drop_duplicates()
    if df.empty:
        return pd.DataFrame(columns=DIM_TIME_COLUMNS)

    if max_month_end is None:
        max_month_end = pd.to_datetime(fact_df["month_end"]).max()
    max_month_end = pd.to_datetime(max_month_end).normalize()
    max_year = int(max_month_end.year)
    max_quarter = f"Q{(max_month_end.month - 1) // 3 + 1}"

    dt = pd.to_datetime(df["month_end"]).dt.normalize()
    month = dt.dt.month
    year = dt.dt.year.astype("int64")
    quarter = ("Q" + dt.dt.quarter.astype(str)).astype("string")
    year_month = dt.dt.strftime("%Y-%m").astype("string")
    month_start = dt.dt.to_period("M").dt.to_timestamp().dt.normalize()

    out = pd.DataFrame({
        "month_end": dt,
        "year": year,
        "month": month.astype("int64"),
        "month_name": dt.dt.strftime("%b").astype("string"),
        "quarter": quarter,
        "year_month": year_month,
        "month_start": month_start,
        "is_latest_month": (dt == max_month_end).values,
        "is_ytd": (year == max_year).values,
        "is_current_quarter": (quarter == max_quarter) & (year == max_year),
    })
    out = out.sort_values("month_end", kind="mergesort").reset_index(drop=True)
    return out


def build_dim_channel(fact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Unique rows of channel_raw, channel_standard, channel_best, preferred_label, channel_l1, channel_l2,
    plus channel_map_status (MAPPED/FALLBACK/EMPTY). preferred_label filled from fallback (best->standard->raw)
    when empty unless DIM_STRICT=true (then hard fail). channel_key = sha1 of 6-tuple. Deterministic sort.
    """
    missing = [c for c in FACT_COLS_DIM_CHANNEL if c not in fact_df.columns]
    if missing:
        logger.warning("build_dim_channel: required columns missing in fact: %s", missing)
        return _empty_dim_channel()
    df = fact_df[FACT_COLS_DIM_CHANNEL].copy()
    for c in FACT_COLS_DIM_CHANNEL:
        df[c] = _str_fill(df[c])

    raw = df["channel_raw"]
    std = df["channel_standard"]
    best = df["channel_best"]
    pl = df["preferred_label"].copy()
    fallback = best.where(best != "", std).where(std != "", raw)
    original_pl_empty = (pl == "")

    if _dim_strict() and original_pl_empty.any():
        missing_rate = original_pl_empty.sum() / len(df)
        raise ValueError(
            f"build_dim_channel: preferred_label missing (strict mode); missing_rate={missing_rate:.4f}"
        )

    pl = pl.where(~original_pl_empty, fallback)
    df["preferred_label"] = pl
    used_fallback = original_pl_empty | ((pl != "") & (pl == fallback))
    status = pd.Series(CHANNEL_MAP_STATUS_MAPPED, index=df.index)
    status = status.mask(pl == "", CHANNEL_MAP_STATUS_EMPTY)
    status = status.mask((pl != "") & used_fallback, CHANNEL_MAP_STATUS_FALLBACK)
    df["channel_map_status"] = status

    # Dedupe: same 6-tuple may appear with different status; prefer MAPPED > FALLBACK > EMPTY
    status_order = {CHANNEL_MAP_STATUS_MAPPED: 2, CHANNEL_MAP_STATUS_FALLBACK: 1, CHANNEL_MAP_STATUS_EMPTY: 0}
    df["_status_ord"] = df["channel_map_status"].map(status_order)
    df = df.sort_values(FACT_COLS_DIM_CHANNEL + ["_status_ord"], kind="mergesort")
    df = df.drop_duplicates(subset=FACT_COLS_DIM_CHANNEL, keep="last").drop(columns=["_status_ord"])
    if df.empty:
        return _empty_dim_channel()

    sort_cols = [
        "preferred_label", "channel_l1", "channel_l2",
        "channel_best", "channel_standard", "channel_raw",
    ]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    def row_hash(row: pd.Series) -> str:
        parts = [str(row[c]) for c in FACT_COLS_DIM_CHANNEL]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    channel_key = df.apply(row_hash, axis=1).astype("string")
    out = pd.concat([channel_key.rename("channel_key"), df], axis=1)
    out = out[["channel_key"] + FACT_COLS_DIM_CHANNEL + ["channel_map_status"]]
    return out


def _empty_dim_channel() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "channel_key",
        "channel_raw", "channel_standard", "channel_best",
        "preferred_label", "channel_l1", "channel_l2",
        "channel_map_status",
    ])


def build_dim_product(fact_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """
    Output: (dim DataFrame, conflicts DataFrame or None, coverage dict).
    Dim: product_key (= product_ticker), product_ticker, segment, sub_segment.
    QA gates (hard fail): product_ticker non-null and non-empty after strip; unique in output.
    Segment/sub_segment: normalized (code -> upper, else title-case), missing -> UNSPECIFIED.
    Conflict resolution: most frequent (segment, sub_segment) per ticker; tie-break segment then sub_segment lexicographically.
    """
    out_cols = ["product_key", "product_ticker", "segment", "sub_segment"]
    empty_dim = pd.DataFrame(columns=out_cols)
    missing = [c for c in FACT_COLS_DIM_PRODUCT if c not in fact_df.columns]
    if missing:
        logger.warning("build_dim_product: required columns missing in fact: %s", missing)
        return empty_dim, None, {}

    df = fact_df[FACT_COLS_DIM_PRODUCT].copy()
    # product_ticker: hard fail if null or empty after strip
    pt = df["product_ticker"].astype(object).fillna("").astype(str).str.strip()
    if pt.isna().any():
        raise ValueError("build_dim_product: product_ticker must be non-null")
    if (pt == "").any():
        n = (pt == "").sum()
        raise ValueError(f"build_dim_product: product_ticker must be non-empty after strip; {n} row(s) empty")
    df["product_ticker"] = pt.astype("string")

    # Normalize segment / sub_segment (strip, collapse spaces; code -> upper, else title-case; empty -> UNSPECIFIED)
    df["segment"] = _normalize_segment_label(df["segment"])
    df["sub_segment"] = _normalize_segment_label(df["sub_segment"])

    def pick_mode_with_conflict(g: pd.DataFrame) -> pd.Series:
        cnt = g.groupby(["segment", "sub_segment"], sort=True).size().reset_index(name="n")
        # Most frequent first; tie-break by segment then sub_segment lexicographically
        cnt = cnt.sort_values(["n", "segment", "sub_segment"], ascending=[False, True, True], kind="mergesort")
        best_row = cnt.iloc[0]
        best = (best_row["segment"], best_row["sub_segment"])
        n_combos = len(cnt)
        top3 = cnt.head(3)
        top3_str = "; ".join(f"{row['segment']}|{row['sub_segment']}: {int(row['n'])}" for _, row in top3.iterrows())
        return pd.Series({
            "segment": best[0],
            "sub_segment": best[1],
            "_combos_count": n_combos,
            "_top_3_combos_with_counts": top3_str,
        })

    agg = df.groupby("product_ticker", sort=False).apply(
        pick_mode_with_conflict, include_groups=False
    ).reset_index()
    agg["product_key"] = agg["product_ticker"].astype("string")
    out = agg[["product_key", "product_ticker", "segment", "sub_segment"]].copy()
    out = out.sort_values("product_ticker", kind="mergesort").reset_index(drop=True)

    # Conflicts: tickers with more than one (segment, sub_segment) combo
    conflicts_df = agg[agg["_combos_count"] > 1].copy()
    if len(conflicts_df) > 0:
        conflicts_df = conflicts_df.rename(columns={
            "_combos_count": "combos_count",
            "_top_3_combos_with_counts": "top_3_combos_with_counts",
        })[["product_ticker", "combos_count", "segment", "sub_segment", "top_3_combos_with_counts"]]
        conflicts_df = conflicts_df.rename(columns={"product_ticker": "ticker", "segment": "chosen_segment", "sub_segment": "chosen_sub_segment"})
    else:
        conflicts_df = None

    # Coverage stats
    total_tickers = len(out)
    unspecified_segment_rate = (out["segment"] == "UNSPECIFIED").mean() if total_tickers else 0.0
    unspecified_sub_segment_rate = (out["sub_segment"] == "UNSPECIFIED").mean() if total_tickers else 0.0
    conflicts_count = int((agg["_combos_count"] > 1).sum())
    coverage = {
        "total_tickers": total_tickers,
        "unspecified_segment_rate": round(float(unspecified_segment_rate), 6),
        "unspecified_sub_segment_rate": round(float(unspecified_sub_segment_rate), 6),
        "conflicts_count": conflicts_count,
    }

    return out, conflicts_df, coverage


def build_dim_geo(
    fact_df: pd.DataFrame,
    geo_region_map_path: Path | None = None,
) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """
    Output: (dim DataFrame, dropped_empty_count, raw_examples DataFrame for QA).
    Dim columns: country_key, country, region, country_type_flags.
    Canonicalization: trim, collapse spaces, strip leading/trailing punctuation; separators to spaces; 2/3-letter alpha or else uppercase.
    Empty after normalization -> dropped and counted. Dedupe by canonical country; country_type_flags sorted: "product", "product,src", "src".
    Region: UNMAPPED when no map or missing in map. Validation: country non-empty, country_key = sha1(country) unique.
    """
    missing = [c for c in FACT_COLS_DIM_GEO if c not in fact_df.columns]
    if missing:
        logger.warning("build_dim_geo: required columns missing in fact: %s", missing)
        return _empty_dim_geo(), 0, pd.DataFrame(columns=["raw_value", "canonical_value", "fact_count"])

    # Raw -> canonical and fact counts for QA (top 50 raw examples)
    src_raw = fact_df["src_country"].astype(object).fillna("").astype(str)
    prd_raw = fact_df["product_country"].astype(object).fillna("").astype(str)
    raw_canonical_counts: dict[tuple[str, str], int] = {}
    for raw in src_raw:
        c = normalize_country(raw)
        raw_canonical_counts[(raw, c)] = raw_canonical_counts.get((raw, c), 0) + 1
    for raw in prd_raw:
        c = normalize_country(raw)
        raw_canonical_counts[(raw, c)] = raw_canonical_counts.get((raw, c), 0) + 1
    raw_examples_list = [
        {"raw_value": r, "canonical_value": c, "fact_count": cnt}
        for (r, c), cnt in sorted(raw_canonical_counts.items(), key=lambda x: -x[1])[:50]
    ]
    raw_examples_df = pd.DataFrame(raw_examples_list) if raw_examples_list else pd.DataFrame(
        columns=["raw_value", "canonical_value", "fact_count"]
    )

    # Canonical sets from src and product (only non-empty)
    src_canonical = src_raw.apply(normalize_country)
    prd_canonical = prd_raw.apply(normalize_country)
    src_set = set(src_canonical[src_canonical != ""].unique())
    prd_set = set(prd_canonical[prd_canonical != ""].unique())
    all_countries = src_set | prd_set

    dropped_empty_count = 0
    # Count empty canonical from fact (rows where canonical is "" for either column)
    empty_src = (src_canonical == "").sum()
    empty_prd = (prd_canonical == "").sum()
    # We don't add empty to dim; "dropped" = fact row contributions that mapped to empty
    dropped_empty_count = int(empty_src) + int(empty_prd)
    if dropped_empty_count > 0:
        logger.warning("build_dim_geo: %d fact country values normalized to empty (dropped)", dropped_empty_count)

    if not all_countries:
        return _empty_dim_geo(), dropped_empty_count, raw_examples_df

    rows = []
    for c in sorted(all_countries):
        flags = []
        if c in src_set:
            flags.append("src")
        if c in prd_set:
            flags.append("product")
        flags = sorted(flags)  # deterministic: "product", "product,src", "src"
        country_key = hashlib.sha1(c.encode("utf-8")).hexdigest()
        rows.append({
            "country_key": country_key,
            "country": c,
            "country_type_flags": ",".join(flags),
        })
    out = pd.DataFrame(rows)

    # Optional region enrichment from geo_region_map.csv
    if geo_region_map_path is not None and Path(geo_region_map_path).exists():
        map_df = pd.read_csv(geo_region_map_path)
        if "country" in map_df.columns and "region" in map_df.columns:
            map_df = map_df[["country", "region"]].drop_duplicates()
            map_df["country_norm"] = map_df["country"].astype(str).apply(normalize_country)
            map_df = map_df[map_df["country_norm"] != ""]
            map_df = map_df.sort_values("country_norm", kind="mergesort")
            map_df = map_df.drop_duplicates(subset=["country_norm"], keep="first")
            region_lookup = map_df.set_index("country_norm")["region"].astype(str).str.strip()
            out["region"] = out["country"].map(region_lookup).fillna(UNMAPPED_REGION)
        else:
            out["region"] = UNMAPPED_REGION
    else:
        out["region"] = UNMAPPED_REGION

    out = out[["country_key", "country", "region", "country_type_flags"]]
    out["country"] = out["country"].astype("string")
    out["region"] = out["region"].astype("string")
    out["country_type_flags"] = out["country_type_flags"].astype("string")
    out["country_key"] = out["country_key"].astype("string")

    assert out["country_key"].is_unique, "dim_geo: duplicate country_key"
    assert out["region"].notna().all(), "dim_geo: region must be non-null"
    assert (out["country"].str.strip() != "").all(), "dim_geo: country must be non-empty"

    return out, dropped_empty_count, raw_examples_df


def _empty_dim_geo() -> pd.DataFrame:
    return pd.DataFrame(columns=["country_key", "country", "region", "country_type_flags"])


def write_manifest(
    output_dir: Path,
    dim_name: str,
    df: pd.DataFrame,
    primary_key_col: str,
    sort_cols: list[str],
) -> None:
    """
    Write JSON manifest to output_dir/manifests/{dim_name}.json with row_count, columns, primary_key,
    sort_cols, build_timestamp_utc, sample_keys (first 5), schema_hash (sha1 of column names + dtypes),
    data_hash (sha1 of first 1000 rows CSV). Deterministic hashes.
    """
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"{dim_name}.json"

    build_timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    columns = list(df.columns)
    row_count = len(df)

    schema_parts = [f"{c}:{str(df[c].dtype)}" for c in sorted(columns)]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()

    sample_df = df.head(1000)
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False, lineterminator="\n")
    data_hash = hashlib.sha1(buf.getvalue().encode("utf-8")).hexdigest()

    sample_keys = []
    if primary_key_col in df.columns and len(df) > 0:
        sample_keys = df[primary_key_col].head(5).astype(str).tolist()

    manifest = {
        "dim_name": dim_name,
        "row_count": row_count,
        "columns": columns,
        "primary_key": primary_key_col,
        "sort_cols": sort_cols,
        "build_timestamp_utc": build_timestamp_utc,
        "sample_keys": sample_keys,
        "schema_hash": schema_hash,
        "data_hash": data_hash,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    logger.info("Wrote manifest %s", path)


def _fact_hashes(fact_df: pd.DataFrame) -> tuple[str, str]:
    """Return (schema_hash, data_hash_sample) for fact_monthly (first 1000 rows canonical CSV)."""
    cols = sorted(fact_df.columns)
    schema_parts = [f"{c}:{str(fact_df[c].dtype)}" for c in cols]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    sample = fact_df.head(1000)
    buf = io.StringIO()
    sample.to_csv(buf, index=False, lineterminator="\n")
    data_hash_sample = hashlib.sha1(buf.getvalue().encode("utf-8")).hexdigest()
    return schema_hash, data_hash_sample


def _dataset_version(fact_schema_hash: str, fact_data_hash_sample: str, code_version_string: str) -> str:
    """dataset_version = sha1(fact_monthly.schema_hash + fact_monthly.data_hash_sample + code_version_string)."""
    payload = f"{fact_schema_hash}{fact_data_hash_sample}{code_version_string}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _gate_persist_dim(df: pd.DataFrame, dim_name: str, primary_key: str) -> None:
    """Hard fail before persist: row_count > 0, primary key exists, unique, no nulls."""
    if len(df) == 0:
        raise ValueError(f"persist {dim_name}: row_count must be > 0")
    if primary_key not in df.columns:
        raise ValueError(f"persist {dim_name}: primary_key '{primary_key}' not in columns")
    if not df[primary_key].is_unique:
        raise ValueError(f"persist {dim_name}: primary_key must be unique")
    if df[primary_key].isna().any():
        n = int(df[primary_key].isna().sum())
        raise ValueError(f"persist {dim_name}: primary_key nulls must be 0, got {n}")


def _persist_dim_atomic(
    curated_dir: Path,
    dim_name: str,
    df: pd.DataFrame,
    primary_key: str,
    fact_schema_hash: str,
    fact_data_hash_sample: str,
    code_version_string: str,
) -> None:
    """
    Validate dim, then write curated/{dim_name}.parquet and curated/{dim_name}.meta.json atomically (temp then rename).
    Log: path, row_count, schema_hash, dataset_version.
    """
    _gate_persist_dim(df, dim_name, primary_key)
    row_count = len(df)
    columns = list(df.columns)
    schema_parts = [f"{c}:{str(df[c].dtype)}" for c in sorted(columns)]
    schema_hash = hashlib.sha1("|".join(schema_parts).encode("utf-8")).hexdigest()
    sample_df = df.head(1000)
    buf = io.StringIO()
    sample_df.to_csv(buf, index=False, lineterminator="\n")
    data_hash_sample = hashlib.sha1(buf.getvalue().encode("utf-8")).hexdigest()
    dataset_version = _dataset_version(fact_schema_hash, fact_data_hash_sample, code_version_string)
    created_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    schema_list = [{"name": c, "dtype": str(df[c].dtype)} for c in columns]
    key_unique = bool(df[primary_key].is_unique)
    key_nulls = int(df[primary_key].isna().sum())

    meta = {
        "name": dim_name,
        "dataset_version": dataset_version,
        "row_count": row_count,
        "primary_key": primary_key,
        "key_unique": key_unique,
        "key_nulls": key_nulls,
        "schema": schema_list,
        "schema_hash": schema_hash,
        "data_hash_sample": data_hash_sample,
        "created_at_utc": created_at_utc,
    }

    curated_dir.mkdir(parents=True, exist_ok=True)
    pq_path = curated_dir / f"{dim_name}.parquet"
    meta_path = curated_dir / f"{dim_name}.meta.json"

    # Atomic parquet write
    fd_pq, tmp_pq = tempfile.mkstemp(suffix=".parquet", dir=curated_dir, prefix="dim_")
    try:
        os.close(fd_pq)
        df.to_parquet(tmp_pq, index=False)
        os.replace(tmp_pq, pq_path)
    except Exception:
        if Path(tmp_pq).exists():
            Path(tmp_pq).unlink(missing_ok=True)
        raise

    # Atomic meta write
    fd_meta, tmp_meta = tempfile.mkstemp(suffix=".meta.json", dir=curated_dir, prefix="dim_")
    try:
        os.close(fd_meta)
        Path(tmp_meta).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    except Exception:
        if Path(tmp_meta).exists():
            Path(tmp_meta).unlink(missing_ok=True)
        raise

    logger.info(
        "persist %s: %s row_count=%d schema_hash=%s dataset_version=%s",
        dim_name,
        pq_path,
        row_count,
        schema_hash,
        dataset_version,
    )


def gate_dim_time(df: pd.DataFrame) -> None:
    """Hard failure: month_end non-null and unique; year_month unique and matches ^\\d{4}-\\d{2}$."""
    if len(df) == 0:
        return
    if "month_end" not in df.columns:
        raise ValueError("dim_time: missing column month_end")
    if df["month_end"].isna().any():
        raise ValueError("dim_time: month_end must be non-null")
    if not df["month_end"].is_unique:
        raise ValueError("dim_time: month_end must be unique")
    if "year_month" not in df.columns:
        raise ValueError("dim_time: missing column year_month")
    if not df["year_month"].is_unique:
        raise ValueError("dim_time: year_month must be unique")
    yymm = df["year_month"].astype(str)
    if not yymm.str.match(r"^\d{4}-\d{2}$", na=False).all():
        bad = yymm[~yymm.str.match(r"^\d{4}-\d{2}$", na=False)]
        raise ValueError(f"dim_time: year_month must match ^\\d{{4}}-\\d{{2}}$; invalid: {bad.tolist()[:5]}")


def gate_dim_channel(df: pd.DataFrame) -> None:
    """Hard failure: channel_key unique and non-null; preferred_label non-empty if DIM_STRICT=true."""
    if len(df) == 0:
        return
    if "channel_key" not in df.columns:
        raise ValueError("dim_channel: missing column channel_key")
    if df["channel_key"].isna().any():
        raise ValueError("dim_channel: channel_key must be non-null")
    if not df["channel_key"].is_unique:
        raise ValueError("dim_channel: channel_key must be unique")
    if _dim_strict() and "preferred_label" in df.columns:
        empty_pl = df["preferred_label"].astype(str).str.strip() == ""
        if empty_pl.any():
            raise ValueError(
                f"dim_channel: preferred_label must be non-empty (strict mode); {empty_pl.sum()} row(s) empty"
            )


def gate_dim_product(df: pd.DataFrame) -> None:
    """Hard failure: product_ticker non-null and unique."""
    if len(df) == 0:
        return
    if "product_ticker" not in df.columns:
        raise ValueError("dim_product: missing column product_ticker")
    if df["product_ticker"].isna().any():
        raise ValueError("dim_product: product_ticker must be non-null")
    empty = df["product_ticker"].astype(str).str.strip() == ""
    if empty.any():
        raise ValueError("dim_product: product_ticker must be non-empty")
    if not df["product_ticker"].is_unique:
        raise ValueError("dim_product: product_ticker must be unique")


def gate_dim_geo(df: pd.DataFrame) -> None:
    """Hard failure: country_key unique and non-null; country non-empty."""
    if len(df) == 0:
        return
    if "country_key" not in df.columns:
        raise ValueError("dim_geo: missing column country_key")
    if df["country_key"].isna().any():
        raise ValueError("dim_geo: country_key must be non-null")
    if not df["country_key"].is_unique:
        raise ValueError("dim_geo: country_key must be unique")
    if "country" not in df.columns:
        raise ValueError("dim_geo: missing column country")
    empty = df["country"].astype(str).str.strip() == ""
    if empty.any():
        raise ValueError("dim_geo: country must be non-empty")


def _write_channel_map_coverage(dim_channel: pd.DataFrame, qa_dir: Path) -> None:
    """Write curated/qa/channel_map_coverage.parquet and .csv with total/rates and breakdown by channel_l1, channel_l2, channel_map_status."""
    if len(dim_channel) == 0 or "channel_map_status" not in dim_channel.columns:
        return
    qa_dir.mkdir(parents=True, exist_ok=True)
    total = len(dim_channel)
    mapped = int((dim_channel["channel_map_status"] == CHANNEL_MAP_STATUS_MAPPED).sum())
    fallback = int((dim_channel["channel_map_status"] == CHANNEL_MAP_STATUS_FALLBACK).sum())
    empty = int((dim_channel["channel_map_status"] == CHANNEL_MAP_STATUS_EMPTY).sum())
    mapped_rate = mapped / total if total else 0.0
    fallback_rate = fallback / total if total else 0.0

    breakdown = dim_channel.groupby(["channel_l1", "channel_l2", "channel_map_status"], dropna=False).size().reset_index(name="count")
    breakdown["rate"] = (breakdown["count"] / total).round(6)

    summary_row = pd.DataFrame([{
        "total_rows": total,
        "mapped_rows": mapped,
        "fallback_rows": fallback,
        "empty_rows": empty,
        "mapped_rate": round(mapped_rate, 6),
        "fallback_rate": round(fallback_rate, 6),
        "channel_l1": "",
        "channel_l2": "",
        "channel_map_status": "",
        "count": total,
        "rate": 1.0,
    }])
    breakdown["channel_l1"] = breakdown["channel_l1"].astype("string")
    breakdown["channel_l2"] = breakdown["channel_l2"].astype("string")
    breakdown["channel_map_status"] = breakdown["channel_map_status"].astype("string")
    for c in ["channel_l1", "channel_l2", "channel_map_status"]:
        summary_row[c] = summary_row[c].astype("string")
    combined = pd.concat([summary_row, breakdown], ignore_index=True)
    path_pq = qa_dir / "channel_map_coverage.parquet"
    path_csv = qa_dir / "channel_map_coverage.csv"
    combined.to_parquet(path_pq, index=False)
    combined.to_csv(path_csv, index=False, lineterminator="\n")
    logger.info("Wrote %s and %s", path_pq, path_csv)


def run(
    fact_path: str | Path = DEFAULT_FACT_PATH,
    curated_dir: str | Path = DEFAULT_CURATED_DIR,
    dims_csv_dir: str | Path = DEFAULT_DIMS_CSV_DIR,
    geo_region_map_path: str | Path | None = None,
    qa_dir: str | Path = DEFAULT_QA_DIR,
    root: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Read fact from parquet, build all dimensions, write parquet + CSV. Optionally enrich dim_geo with region;
    emit qa/unmapped_geo_regions.csv for countries with region==UNMAPPED (warn only, do not fail).
    Return dict of dim name -> DataFrame.
    """
    root = Path(root) if root is not None else Path.cwd()
    fact_path = root / fact_path
    curated_dir = root / curated_dir
    dims_csv_dir = root / dims_csv_dir
    qa_dir = root / qa_dir
    geo_path = root / geo_region_map_path if geo_region_map_path is not None else root / DEFAULT_GEO_REGION_MAP_PATH

    if not fact_path.exists():
        raise FileNotFoundError(f"Fact parquet not found: {fact_path}")
    fact_df = pd.read_parquet(fact_path)
    logger.info("Loaded fact_monthly: %d rows from %s", len(fact_df), fact_path)

    max_month_end = pd.to_datetime(fact_df["month_end"]).max() if len(fact_df) > 0 and "month_end" in fact_df.columns else None
    dim_time = build_dim_time(fact_df, max_month_end=max_month_end)
    dim_channel = build_dim_channel(fact_df)
    dim_product, product_conflicts_df, product_coverage = build_dim_product(fact_df)
    dim_geo, geo_dropped_empty_count, geo_raw_examples_df = build_dim_geo(fact_df, geo_region_map_path=geo_path)

    dims = {
        "dim_time": dim_time,
        "dim_channel": dim_channel,
        "dim_product": dim_product,
        "dim_geo": dim_geo,
    }
    for name, df in dims.items():
        logger.info("Dimension %s: %d rows", name, len(df))

    # Hard quality gates (raise ValueError)
    gate_dim_time(dim_time)
    gate_dim_channel(dim_channel)
    gate_dim_product(dim_product)
    gate_dim_geo(dim_geo)

    # Soft QA: dim_time summary
    qa_dir.mkdir(parents=True, exist_ok=True)
    if len(dim_time) > 0 and "month_end" in dim_time.columns:
        max_me = dim_time["month_end"].max()
        max_me_ts = pd.to_datetime(max_me)
        dim_time_summary = {
            "max_month_end": max_me_ts.strftime("%Y-%m-%d"),
            "max_year": int(max_me_ts.year),
            "current_quarter": f"Q{(max_me_ts.month - 1) // 3 + 1}",
            "row_count": len(dim_time),
        }
        qa_time_path = qa_dir / "dim_time_summary.json"
        with open(qa_time_path, "w", encoding="utf-8") as f:
            json.dump(dim_time_summary, f, indent=2, sort_keys=True)
        logger.info("Wrote %s", qa_time_path)

    # Channel map coverage reporting (parquet + csv)
    if len(dim_channel) > 0 and "channel_map_status" in dim_channel.columns:
        _write_channel_map_coverage(dim_channel, qa_dir)

    # Product: conflict summary and coverage stats
    if product_conflicts_df is not None and len(product_conflicts_df) > 0:
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / "product_segment_conflicts.csv"
        product_conflicts_df.to_csv(qa_path, index=False, lineterminator="\n")
        logger.info("Wrote %s", qa_path)
    if product_coverage:
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_path = qa_dir / "dim_product_coverage.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(product_coverage, f, indent=2, sort_keys=True)
        logger.info("Wrote %s", qa_path)

    # Geo QA: dropped empty count + raw examples (top 50 distinct raw -> canonical + fact counts)
    if FACT_COLS_DIM_GEO[0] in fact_df.columns and FACT_COLS_DIM_GEO[1] in fact_df.columns:
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.joinpath("geo_dropped_empty_count.json").write_text(
            json.dumps({"dropped_empty_count": geo_dropped_empty_count}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        geo_raw_examples_df.to_csv(qa_dir / "geo_raw_examples.csv", index=False, lineterminator="\n")
        logger.info("Wrote qa/geo_dropped_empty_count.json and qa/geo_raw_examples.csv")

    # If empty_rows > 0: write qa/empty_channels.csv (6-tuple + count from fact); do not crash unless strict
    empty_rows = int((dim_channel["channel_map_status"] == CHANNEL_MAP_STATUS_EMPTY).sum()) if len(dim_channel) > 0 and "channel_map_status" in dim_channel.columns else 0
    if empty_rows > 0 and all(c in fact_df.columns for c in FACT_COLS_DIM_CHANNEL):
        ch = fact_df[FACT_COLS_DIM_CHANNEL].copy()
        for c in FACT_COLS_DIM_CHANNEL:
            ch[c] = _str_fill(ch[c])
        all_empty = (ch["channel_raw"] == "") & (ch["channel_standard"] == "") & (ch["channel_best"] == "")
        empty_fact = ch[all_empty]
        if len(empty_fact) > 0:
            empty_agg = empty_fact.groupby(FACT_COLS_DIM_CHANNEL, dropna=False).size().reset_index(name="fact_count")
            qa_dir.mkdir(parents=True, exist_ok=True)
            qa_path = qa_dir / "empty_channels.csv"
            empty_agg.to_csv(qa_path, index=False, lineterminator="\n")
            logger.warning("dim_channel: %d empty channel row(s); QA written to %s", empty_rows, qa_path)

    curated_dir.mkdir(parents=True, exist_ok=True)
    dims_csv_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: validation gates then atomic persist (parquet + meta.json)
    manifest_config = {
        "dim_time": ("month_end", ["month_end"]),
        "dim_channel": ("channel_key", ["preferred_label", "channel_l1", "channel_l2", "channel_best", "channel_standard", "channel_raw"]),
        "dim_product": ("product_ticker", ["product_ticker"]),
        "dim_geo": ("country_key", ["country"]),
    }
    for name, df in dims.items():
        pk, _ = manifest_config[name]
        _gate_persist_dim(df, name, pk)
    fact_schema_hash, fact_data_hash_sample = _fact_hashes(fact_df)
    code_version_string = os.environ.get("PIPELINE_VERSION", "dev")
    for name, df in dims.items():
        pk, _ = manifest_config[name]
        _persist_dim_atomic(
            curated_dir,
            name,
            df,
            pk,
            fact_schema_hash,
            fact_data_hash_sample,
            code_version_string,
        )

    for name, df in dims.items():
        csv_path = dims_csv_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, lineterminator="\n")
        logger.info("Wrote %s", csv_path)

    # Manifests
    for name, df in dims.items():
        pk, sort_cols = manifest_config[name]
        write_manifest(curated_dir, name, df, pk, sort_cols)

    # QA: unmapped geo regions (warn only)
    if len(dim_geo) > 0 and "region" in dim_geo.columns:
        unmapped = dim_geo[dim_geo["region"] == UNMAPPED_REGION]
        if len(unmapped) > 0:
            qa_dir.mkdir(parents=True, exist_ok=True)
            qa_path = qa_dir / "unmapped_geo_regions.csv"
            unmapped[["country", "country_type_flags"]].to_csv(
                qa_path, index=False, lineterminator="\n"
            )
            logger.warning(
                "dim_geo: %d country/countries have no region mapping (region=%s); QA written to %s",
                len(unmapped),
                UNMAPPED_REGION,
                qa_path,
            )

    return dims


def main() -> int:
    """CLI entry: python -m pipelines.dimensions.build_dimensions."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        run()
        return 0
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.exception("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
