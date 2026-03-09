from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MEASURES = ["begin_aum", "end_aum", "nnb", "nnf", "ogr", "market_impact", "market_impact_rate", "fee_yield"]
MONTH_END_ALIASES = ("date", "month", "month_end_date", "period")
CHANNEL_ALIASES = ("preferred_label", "channel_l1", "channel_l2", "channel_standard", "channel_best", "canonical_channel")
TICKER_ALIASES = ("ticker", "ticker_symbol", "symbol")
GEO_ALIASES = ("geo", "region", "country", "src_country_canonical", "product_country_canonical")
SUBSEG_ALIASES = ("subsegment", "sub_segment_name")


def validate_required_columns(df: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"SchemaError: {dataset_name} is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    if "month_end" not in out.columns:
        lower_to_actual = {str(c).strip().lower(): c for c in out.columns}
        for alias in MONTH_END_ALIASES:
            actual = lower_to_actual.get(alias.lower())
            if actual:
                out = out.rename(columns={actual: "month_end"})
                break
    validate_required_columns(out, ["month_end"], dataset_name)
    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce").dt.to_period("M").dt.to_timestamp("M")
    return out


def _normalize_dimension_aliases(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = df.copy()
    alias_sets = {
        "channel": CHANNEL_ALIASES,
        "product_ticker": TICKER_ALIASES,
        "src_country": GEO_ALIASES,
        "sub_segment": SUBSEG_ALIASES,
    }
    lower_to_actual = {str(c).strip().lower(): c for c in out.columns}
    for canonical, aliases in alias_sets.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            actual = lower_to_actual.get(alias.lower())
            if actual:
                out = out.rename(columns={actual: canonical})
                break
    return out


def _validate_table_contract(df: pd.DataFrame, table_name: str, required_dims: list[str]) -> None:
    required = ["month_end"] + required_dims + [m for m in MEASURES if m in df.columns]
    validate_required_columns(df, required_dims + ["month_end"], table_name)
    # Keep deterministic ordering expectations for downstream loaders.
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"SchemaError: {table_name} is missing required output columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _sum_table(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    cols = ["month_end"] + dims
    present_measures = [c for c in MEASURES if c in df.columns]
    out = df.groupby(cols, as_index=False)[present_measures].sum(min_count=1)
    return out.sort_values(cols).reset_index(drop=True)


def run(curated_dir: Path, agg_dir: Path) -> None:
    agg_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(curated_dir / "metrics_monthly.parquet")
    metrics = _normalize_month_end(metrics, "metrics_monthly")
    metrics = _normalize_dimension_aliases(metrics, "metrics_monthly")
    validate_required_columns(
        metrics,
        ["month_end", "channel", "product_ticker", "src_country", "segment", "sub_segment"],
        "metrics_monthly",
    )

    table_dims: dict[str, list[str]] = {
        "firm_monthly": [],
        "channel_monthly": ["channel"],
        "ticker_monthly": ["product_ticker"],
        "geo_monthly": ["src_country"],
        "segment_monthly": ["segment", "sub_segment"],
    }
    tables: dict[str, pd.DataFrame] = {
        "firm_monthly": _sum_table(metrics, []),
        "channel_monthly": _sum_table(metrics, ["channel"]),
        "ticker_monthly": _sum_table(metrics, ["product_ticker"]),
        "geo_monthly": _sum_table(metrics, ["src_country"]),
        "segment_monthly": _sum_table(metrics, ["segment", "sub_segment"]),
    }

    manifest_tables = []
    for name, table in tables.items():
        _validate_table_contract(table, name, table_dims.get(name, []))
        path = agg_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        manifest_tables.append({"name": name, "path": f"data/agg/{name}.parquet"})

    meta_path = curated_dir / "metrics_monthly.meta.json"
    dataset_version = "unknown"
    if meta_path.exists():
        dataset_version = json.loads(meta_path.read_text(encoding="utf-8")).get("dataset_version", "unknown")

    manifest = {"dataset_version": dataset_version, "tables": manifest_tables}
    (agg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build monthly aggregate parquet tables")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--agg-dir", default="data/agg")
    args = parser.parse_args()
    run(Path(args.curated_dir), Path(args.agg_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
