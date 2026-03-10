# Analytics (DuckDB backend)

For **Streamlit Cloud** (DuckDB single-file backend), the following must exist:

- **Repository root:** `analytics.duckdb` (≈ 4 MB). Build via ETL or copy your local file; `.gitignore` allows it with `!analytics.duckdb`.
- **This folder:** `duckdb_views_manifest.json` — points the app to `analytics.duckdb` and defines schema (e.g. `data`) and `reads_views_only: true`.

See **docs/DEPLOY.md** § 3.2 for the full list of required files and paths.
