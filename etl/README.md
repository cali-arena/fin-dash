# ETL: raw → curated → agg → DuckDB

Single entrypoint builds all data artifacts the app needs. Idempotent: same inputs produce same outputs.

## Build steps (order)

1. **Validate input** — Raw directory must contain source files, or curated directory must already have `metrics_monthly.parquet` (allows rebuild from curated only).
2. **Curated** — Clean and write parquet under `--curated-dir`; write `metrics_monthly.meta.json` with `dataset_version`.
3. **Agg** — Rollups (firm_monthly, channel_monthly, ticker_monthly, etc.) from curated; write `manifest.json` under `--agg-dir`.
4. **DuckDB** — Create/replace `--duckdb-path`: tables from agg parquet (schema `data`), views `v_*`, schema `meta` with `dataset_version` and `build_timestamp`; write `analytics/duckdb_views_manifest.json` so the app finds the DB. The app reads `dataset_version` from `meta.dataset_version` when the DB is present.

## CLI

```bash
# Defaults: data/raw, data/curated, data/agg, analytics.duckdb
python etl/build_data.py

# Custom paths and version tag
python etl/build_data.py --raw-dir data/raw --curated-dir data/curated --agg-dir data/agg --duckdb-path analytics.duckdb --dataset-version v1.0.0

# Rebuild from existing curated (skip raw)
python etl/build_data.py --curated-dir data/curated --agg-dir data/agg --duckdb-path analytics.duckdb
```

## Outputs

- `data/curated/*.parquet` and `data/curated/metrics_monthly.meta.json`
- `data/agg/*.parquet` and `data/agg/manifest.json`
- `analytics.duckdb` (or path given by `--duckdb-path`)
- `analytics/duckdb_views_manifest.json` (points app at the DuckDB file)

After a successful run, start the app with `streamlit run app/main.py`; the gateway will use the DuckDB and read `dataset_version` from the DB meta table when available.
