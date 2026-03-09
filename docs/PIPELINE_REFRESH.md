# Pipeline refresh (DuckDB analytics layer)

## Full pipeline order

1. **Agg build** (produces `agg/*.parquet` and `agg/manifest.json`):

   ```bash
   python -m pipelines.agg.build_aggs --policy configs/agg_policy.yml
   ```

2. **DuckDB analytics layer** (single convenience runner; validates policy → build DB → create views → smoke):

   ```bash
   python -m pipelines.duckdb.rebuild_analytics_layer --policy configs/duckdb_policy.yml
   ```

   Optional: `--root /path/to/project` and `--force` to skip manifest cache and rebuild.

## Step-by-step (optional)

If you need to run DB and views separately:

```bash
# After agg build succeeds:
python -m pipelines.duckdb.build_analytics_db --policy configs/duckdb_policy.yml
python -m pipelines.duckdb.create_views --policy configs/duckdb_policy.yml
```

## CI / scripts

One-liner from project root (agg must already be built):

```bash
python -m pipelines.duckdb.rebuild_analytics_layer --policy configs/duckdb_policy.yml
```

Exit code: `0` if OK, `2` on any failure.
