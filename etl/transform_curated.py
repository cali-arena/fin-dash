from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SLICE_KEYS = ["channel", "product_ticker", "src_country", "segment", "sub_segment"]


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": "", "None": ""})


def _build_mapping(raw_df: pd.DataFrame) -> pd.DataFrame:
    combos = (
        raw_df[["channel_raw", "channel_standard", "channel_best"]]
        .drop_duplicates()
        .copy()
    )
    combos["channel"] = combos["channel_best"].where(combos["channel_best"].ne(""), combos["channel_standard"])
    combos["channel"] = combos["channel"].where(combos["channel"].ne(""), combos["channel_raw"])
    combos["mapping_source"] = "best>standard>raw"
    return combos


def _apply_mapping(raw_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.merge(
        mapping_df[["channel_raw", "channel_standard", "channel_best", "channel"]],
        on=["channel_raw", "channel_standard", "channel_best"],
        how="left",
    )
    out["channel"] = out["channel"].where(out["channel"].notna() & out["channel"].ne(""), out["channel_best"])
    out["channel"] = out["channel"].where(out["channel"].notna() & out["channel"].ne(""), out["channel_standard"])
    out["channel"] = out["channel"].where(out["channel"].notna() & out["channel"].ne(""), out["channel_raw"])
    return out


def _compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["month_end"] + SLICE_KEYS
    measures = ["end_aum", "nnb", "nnf"]
    agg = df.groupby(group_cols, as_index=False)[measures].sum(min_count=1)
    agg = agg.sort_values(SLICE_KEYS + ["month_end"]).reset_index(drop=True)
    agg["begin_aum"] = agg.groupby(SLICE_KEYS)["end_aum"].shift(1)
    agg["ogr"] = agg["nnb"] / agg["begin_aum"]
    agg["market_impact"] = agg["end_aum"] - agg["begin_aum"] - agg["nnb"]
    agg["market_impact_rate"] = agg["market_impact"] / agg["begin_aum"]
    agg["fee_yield"] = agg["nnf"] / agg["nnb"]

    invalid_begin = agg["begin_aum"].isna() | (agg["begin_aum"] <= 0)
    agg.loc[invalid_begin, ["ogr", "market_impact_rate"]] = pd.NA
    agg.loc[agg["nnb"].isna() | (agg["nnb"] <= 0), "fee_yield"] = pd.NA
    return agg


def run(curated_dir: Path, qa_dir: Path) -> None:
    curated_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(curated_dir / "data_raw_normalized.parquet")
    raw["month_end"] = pd.to_datetime(raw["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")

    for col in ["channel_raw", "channel_standard", "channel_best", "src_country", "product_ticker", "segment", "sub_segment"]:
        raw[col] = _clean_text(raw[col])

    mapping = _build_mapping(raw)
    mapped = _apply_mapping(raw, mapping)

    unmapped_mask = mapped["channel"].isna() | mapped["channel"].eq("")
    unmapped = mapped.loc[unmapped_mask, ["channel_raw", "channel_standard", "channel_best"]].drop_duplicates()
    unmapped.to_csv(qa_dir / "unmapped_channels.csv", index=False)

    # Strict mapping audit: distinct raw channel/segment combos that fail to map, with counts + sample month_end + sample ticker
    unmapped_full = mapped.loc[unmapped_mask].copy()
    key_cols = [c for c in ["channel_raw", "channel_standard", "channel_best", "segment", "sub_segment"] if c in mapped.columns]
    if not unmapped_full.empty and key_cols:
        agg_spec = {"row_count": ("month_end", "count")}
        if "month_end" in unmapped_full.columns:
            agg_spec["sample_month_end"] = ("month_end", "min")
        if "product_ticker" in unmapped_full.columns:
            agg_spec["sample_ticker"] = ("product_ticker", "min")
        unmapped_keys_df = unmapped_full.groupby(key_cols, dropna=False).agg(**agg_spec).reset_index()
        unmapped_keys_df = unmapped_keys_df.sort_values(key_cols).reset_index(drop=True)
    else:
        unmapped_keys_df = pd.DataFrame(columns=key_cols + ["row_count", "sample_month_end", "sample_ticker"])
    unmapped_keys_df.to_csv(qa_dir / "unmapped_keys.csv", index=False)
    unmapped_rows = int(unmapped_keys_df["row_count"].sum()) if "row_count" in unmapped_keys_df.columns and len(unmapped_keys_df) > 0 else 0
    meta_qa = {
        "total_raw_rows": len(raw),
        "unmapped_rows": unmapped_rows,
        "unmapped_keys": len(unmapped_keys_df),
    }
    (qa_dir / "unmapped_keys.meta.json").write_text(json.dumps(meta_qa, indent=2), encoding="utf-8")

    metrics = _compute_core_metrics(mapped)
    metrics.to_parquet(curated_dir / "metrics_monthly.parquet", index=False)
    mapping.to_parquet(curated_dir / "channel_mapping.parquet", index=False)

    dataset_version = hashlib.sha1(pd.util.hash_pandas_object(metrics, index=True).values.tobytes()).hexdigest()[:12]
    meta = {
        "dataset_version": dataset_version,
        "row_count": int(len(metrics)),
        "slice_keys": SLICE_KEYS,
    }
    (curated_dir / "metrics_monthly.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transform normalized inputs into curated metrics")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--qa-dir", default="qa")
    args = parser.parse_args()
    run(Path(args.curated_dir), Path(args.qa_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
