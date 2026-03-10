# Release push — Git + Streamlit Cloud

One-time checklist and exact commands for a clean push and deploy with localhost/cloud parity.

---

## 1. Summary of what was prepared

- **Data parity:** Canonical paths only (`data/agg/manifest.json`, `data/curated/metrics_monthly.meta.json`); no path fallbacks. Optional `APP_DATA_BACKEND=parquet` for same backend on both envs. `app_root` in session so `run_query` always uses repo root when root is None.
- **Cloud readiness:** All paths use `Path(__file__).resolve()` or passed `root`; no hardcoded Windows paths. Missing manifest/meta raises clear `FileNotFoundError`. DEPLOY.md updated with parquet-backend and required env vars.
- **Deployment consistency:** Single manifest path; single dataset-version meta path; parity debug block when `DEBUG_DATA_PARITY=1` or `SHOW_PARITY_DEBUG=1`.
- **Mock/demo:** `APP_MOCK_DATA=1` is opt-in only; not set by default. No change.

---

## 2. Exact Git commands

Run from repository root.

```bash
# 1) See current status
git status

# 2) Stage code and docs (parity fix + cloud readiness)
git add app/agg_store.py app/data_contract.py app/data_gateway.py app/main.py
git add docs/DATA_PARITY_CHECKLIST.md docs/DEPLOY.md docs/RELEASE_PUSH.md

# 3) Optional but recommended: commit production parquet so Cloud runs without ETL
#    (.gitignore excludes them; -f forces add)
git add -f data/agg/firm_monthly.parquet data/agg/channel_monthly.parquet data/agg/ticker_monthly.parquet data/agg/geo_monthly.parquet data/agg/segment_monthly.parquet

# 4) Commit
git commit -m "Data parity and cloud readiness: canonical paths, parquet backend option, parity debug" -m "Single manifest/meta paths; APP_DATA_BACKEND=parquet for parity; app_root fallback; explicit missing-file errors. Optional: commit data/agg/*.parquet for Cloud run-without-ETL. Docs: DATA_PARITY_CHECKLIST, DEPLOY 3.3, RELEASE_PUSH."

# 5) Push
git push origin main
```

If you **do not** want to commit parquet (e.g. you will run ETL in Cloud or use DuckDB), omit the `git add -f data/agg/...` step and ensure Streamlit Cloud has data via ETL or `analytics.duckdb` (see DEPLOY §3.2).

---

## 3. Files to confirm before pushing

| File / path | Purpose |
|-------------|--------|
| `app/main.py` | Entrypoint; sets `app_root`; parity debug. |
| `app/data_gateway.py` | Parquet-only option; single meta path; run_query root fallback. |
| `app/data_contract.py` | Parquet-first resolve when `APP_DATA_BACKEND=parquet`. |
| `app/agg_store.py` | Single manifest path; no fallback. |
| `data/agg/manifest.json` | Must be committed (already tracked). |
| `data/curated/metrics_monthly.meta.json` | Must be committed (already tracked). |
| `data/agg/*.parquet` | Required for parquet backend if Cloud does not run ETL. Either commit with `git add -f` or provide data another way. |
| `requirements.txt` | At repo root; no local paths. |
| `.streamlit/config.toml` | Present (server/theme). |

---

## 4. Streamlit Cloud steps after push

1. **App settings**
   - Main file path: `app/main.py`.
   - Python 3.10 or 3.11.

2. **Environment variables (recommended for parity)**
   - `APP_DATA_BACKEND` = `parquet` (so Cloud uses same backend as local when you use parquet).
   - Optional: `DEBUG_DATA_PARITY=1` or `SHOW_PARITY_DEBUG=1` to show parity expander.

3. **Secrets (optional)**
   - `ANTHROPIC_API_KEY` for Intelligence Desk narrative; app is safe if unset.

4. **Redeploy** after push (or trigger deploy); wait for build to finish.

---

## 5. Smoke-test checklist

After deploy, verify:

- [ ] **App boots** — No startup crash; Streamlit Cloud app loads.
- [ ] **Datasets load** — No "Expected dataset file is missing" or "Agg manifest not found"; Debug / Data Contract shows backend and path.
- [ ] **KPI values match localhost** — Same date range and scope on local and Cloud; compare End AUM, NNB, OGR, Market Movement (use parity expander if enabled).
- [ ] **No false 0.00 market movement** — If data has movement, Cloud shows non-zero when using same backend and same data.
- [ ] **No permission/path/cache issue** — No permission errors in logs; path in parity block is under repo; after a redeploy, cache is invalidated by fingerprint change if data changed.

If any check fails, compare "Data parity" expander (when enabled) between local and Cloud and confirm `APP_DATA_BACKEND=parquet` and same committed data.
