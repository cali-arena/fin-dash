"""Canonical ETL entrypoint: ingest -> transform_curated -> build_agg -> build_duckdb."""
from __future__ import annotations

import argparse
from pathlib import Path

from etl.build_agg import run as run_agg
from etl.build_duckdb import run as run_duckdb
from etl.ingest_excel import run as run_ingest
from etl.transform_curated import run as run_transform


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical ETL chain for the asset dashboard")
    parser.add_argument("--excel-path", default="data/input/source.xlsx")
    parser.add_argument("--curated-dir", default="data/curated")
    parser.add_argument("--agg-dir", default="data/agg")
    parser.add_argument("--duckdb-path", default="analytics.duckdb")
    parser.add_argument("--qa-dir", default="qa")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    curated_dir = root / args.curated_dir
    agg_dir = root / args.agg_dir
    qa_dir = root / args.qa_dir
    duckdb_path = root / args.duckdb_path
    excel_path = root / args.excel_path

    run_ingest(excel_path, curated_dir)
    run_transform(curated_dir, qa_dir)
    run_agg(curated_dir, agg_dir)
    run_duckdb(agg_dir, duckdb_path, root)
    print(f"ETL complete: {curated_dir} | {agg_dir} | {duckdb_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
