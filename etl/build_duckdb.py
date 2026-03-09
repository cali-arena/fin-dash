from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

TABLES = ["firm_monthly", "channel_monthly", "ticker_monthly", "geo_monthly", "segment_monthly"]
MONTH_END_ALIASES = ("date", "month", "month_end_date", "period")
REQUIRED_DIMS_BY_TABLE = {
    "firm_monthly": [],
    "channel_monthly": ["channel"],
    "ticker_monthly": ["product_ticker"],
    "geo_monthly": ["src_country"],
    "segment_monthly": ["segment", "sub_segment"],
}


def _validate_table_schema(parquet_path: Path, table_name: str) -> None:
    cols = pd.read_parquet(parquet_path, columns=None).columns.tolist()
    if "month_end" in cols:
        required_dims = REQUIRED_DIMS_BY_TABLE.get(table_name, [])
        missing_dims = [d for d in required_dims if d not in cols]
        if missing_dims:
            raise ValueError(
                f"SchemaError: {table_name} parquet is missing required dimension column(s): {missing_dims}. "
                f"Found columns: {cols}"
            )
        return
    lower_cols = {str(c).strip().lower() for c in cols}
    if any(alias in lower_cols for alias in MONTH_END_ALIASES):
        raise ValueError(
            f"SchemaError: {table_name} parquet uses a legacy month column alias. "
            f"Rebuild agg outputs with canonical 'month_end'. Found columns: {cols}"
        )
    raise ValueError(
        f"SchemaError: {table_name} parquet is missing required column 'month_end'. "
        f"Found columns: {cols}"
    )


def run(agg_dir: Path, duckdb_path: Path, root: Path) -> None:
    con = duckdb.connect(str(duckdb_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS data")
    con.execute("CREATE SCHEMA IF NOT EXISTS meta")

    for table in TABLES:
        parquet = agg_dir / f"{table}.parquet"
        if not parquet.exists():
            continue
        _validate_table_schema(parquet, table)
        con.execute(f"DROP TABLE IF EXISTS data.agg_{table}")
        con.execute(f"CREATE TABLE data.agg_{table} AS SELECT * FROM read_parquet(?)", [str(parquet)])
        con.execute(f"CREATE OR REPLACE VIEW data.v_{table} AS SELECT * FROM data.agg_{table}")

    meta_json = root / "data" / "curated" / "metrics_monthly.meta.json"
    dataset_version = "unknown"
    if meta_json.exists():
        dataset_version = json.loads(meta_json.read_text(encoding="utf-8")).get("dataset_version", "unknown")

    con.execute("CREATE OR REPLACE TABLE meta.dataset_version AS SELECT ? AS dataset_version", [dataset_version])
    build_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    con.execute("CREATE OR REPLACE TABLE meta.build_timestamp AS SELECT ? AS build_timestamp", [build_ts])
    con.close()

    manifest = {
        "db_path": str(duckdb_path.relative_to(root)).replace("\\", "/"),
        "schema": "data",
        "dataset_version": dataset_version,
        "created_at": build_ts,
        "reads_views_only": True,
    }
    manifest_path = root / "analytics" / "duckdb_views_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build analytics.duckdb tables and governed views")
    parser.add_argument("--agg-dir", default="data/agg")
    parser.add_argument("--duckdb-path", default="analytics.duckdb")
    args = parser.parse_args()
    root = Path.cwd()
    run(root / args.agg_dir, root / args.duckdb_path, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
