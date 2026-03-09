from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def run(root: Path) -> None:
    curated = root / "data" / "curated" / "metrics_monthly.parquet"
    agg = root / "data" / "agg" / "firm_monthly.parquet"
    db = root / "analytics.duckdb"
    if not curated.exists():
        raise FileNotFoundError(curated)
    if not agg.exists():
        raise FileNotFoundError(agg)
    if not db.exists():
        raise FileNotFoundError(db)

    df = pd.read_parquet(curated)
    required = {"month_end", "begin_aum", "end_aum", "nnb", "market_impact"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics_monthly missing columns: {sorted(missing)}")

    con = duckdb.connect(str(db), read_only=True)
    try:
        for view in ["v_firm_monthly", "v_channel_monthly", "v_ticker_monthly", "v_geo_monthly", "v_segment_monthly"]:
            con.execute(f"SELECT COUNT(*) FROM data.{view}").fetchone()
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for dashboard artifacts")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    run(Path(args.root).resolve())
    print("Smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
