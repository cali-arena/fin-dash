# Localhost vs Streamlit Cloud — Data Parity Checklist

Use this to verify that KPIs (NNB, OGR, Market Movement, End AUM) match between local and deployed app.

## Root causes addressed

1. **Backend split** — Local used DuckDB views; Cloud used parquet. Views and parquet could differ (columns, formulas).  
   **Fix:** Set `APP_DATA_BACKEND=parquet` on both so both use the same parquet pipeline.

2. **Path fallbacks** — Manifest tried `data/agg/manifest.json` then `agg/manifest.json`; dataset version tried `data/curated/...` then `curated/...`. Different envs could resolve to different files.  
   **Fix:** Single canonical paths only; no fallbacks. Required files: `data/agg/manifest.json`, `data/agg/*.parquet`, `data/curated/metrics_monthly.meta.json`.

3. **Missing root** — Some code paths used `Path.cwd()` when `root` was None, so local vs Cloud cwd could differ.  
   **Fix:** Main sets `st.session_state["app_root"]`; `run_query` uses it when `root` is None. All pages should pass `ROOT` to `DataGateway(ROOT)` / `run_query(..., root=ROOT)`.

4. **Market movement 0.00** — Parquet path derives `market_impact` and `market_impact_rate` in `_canonicalize_monthly_for_ui` via `_compute_derived_metrics`. If `begin_aum` was missing or wrong (e.g. single row), derived values could be NaN/0.  
   **Fix:** Same pipeline and same data on both (parquet + canonicalization) so formulas match.

## Verification steps

### 1. Env and backend

- [ ] **Local:** Set `APP_DATA_BACKEND=parquet` (and optionally `DEBUG_DATA_PARITY=1` or `SHOW_PARITY_DEBUG=1`).
- [ ] **Cloud:** In Streamlit Cloud app settings, add secret/env `APP_DATA_BACKEND=parquet` (and optionally `DEBUG_DATA_PARITY=1`).
- [ ] Restart both; confirm no DuckDB is used (e.g. in Debug expander, Backend shows `parquet`).

### 2. Repo data files

- [ ] `data/agg/manifest.json` exists and lists `firm_monthly`, `channel_monthly`, etc. with paths `data/agg/*.parquet`.
- [ ] `data/agg/firm_monthly.parquet` (and other agg tables) exist and are committed.
- [ ] `data/curated/metrics_monthly.meta.json` exists and contains `"dataset_version": "<version>"`.

### 3. Same filters

- [ ] Use the same date range (e.g. same `date_start` / `date_end`) on local and Cloud.
- [ ] Use same scope (e.g. Enterprise-wide; or same slice_dim/slice_value if applicable).

### 4. Parity debug block

- [ ] Enable `DEBUG_DATA_PARITY=1` or `SHOW_PARITY_DEBUG=1` on both.
- [ ] Open "Data parity (DEBUG_DATA_PARITY / SHOW_PARITY_DEBUG)" expander.
- [ ] Compare:
  - Resolved path (should be same logical path, e.g. `.../data/agg/firm_monthly.parquet`).
  - File exists: true on both.
  - Firm monthly rows (unfiltered): same count.
  - Month range: same min/max.
  - Sum NNB, Sum NNF, Sum End AUM: same values.
  - Sum market_impact / market_impact_rate: same (and not 0 if data has movement).
  - OGR (last row): same.

### 5. KPI cards

- [ ] Executive Dashboard (Tab 1): Same End AUM, NNB, OGR, Market Movement (or rate) for the same filters.
- [ ] If one side shows 0.00 for Market Movement, confirm the other side uses same backend and same data; re-check parity debug sums.

### 6. Cache

- [ ] After changing `APP_DATA_BACKEND` or data files, clear cache (e.g. "Clear data cache (dev)" in Debug or redeploy).
- [ ] Dataset fingerprint in session should match the loaded data; if it changes, cache is cleared automatically.

## If parity still fails

- Confirm both use **parquet** (no DuckDB on one side).
- Confirm **same** `data/agg/*.parquet` and `data/curated/metrics_monthly.meta.json` (same commit / same deploy).
- Compare **Data parity** expander numbers side by side; any difference points to different data or different code path.
- Check for env-specific overrides (e.g. `DUCKDB_PATH` only on local) and remove or align them.
