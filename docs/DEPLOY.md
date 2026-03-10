# Deploy: Streamlit Cloud

Checklist and structure for production deploy of **FINANCE_DASH** (Asset Management & Distribution Analytics Dashboard).

---

## 1. Entrypoint

- **Canonical entrypoint:** `app/main.py`
- **Streamlit Cloud:** Set *Main file path* to `app/main.py` (run from repository root).
- **Alternative:** `streamlit run streamlit_app.py` (streamlit_app.py only calls `app.main.main()`); for Cloud, prefer `app/main.py` for clarity.

All paths in the app are relative to the **repository root** (e.g. `data/curated`, `analytics.duckdb`). The app resolves root via `Path(__file__).resolve().parents[2]` from `app/main.py`, so no Windows-specific or absolute paths are required.

---

## 2. Recommended repository structure (production)

```
FINANCE_DASH/
├── app/
│   ├── main.py              # Streamlit entrypoint
│   ├── pages/               # Tab renderers (visualisations, dynamic_report, nlq_chat)
│   ├── ui/                  # Theme, filters, guardrails, exports, formatters, observability
│   ├── data/                # data_gateway re-export, duckdb_views
│   ├── reporting/           # report_engine, report_pack, html_export, reconciliation
│   ├── nlq/                 # NLQ parser, executor, governance
│   ├── config/              # contract, filters.yml, ui_contract.yml
│   ├── state.py
│   ├── data_gateway.py      # Canonical gateway (re-exported as app.data.data_gateway)
│   └── ...                  # analytics, cache, metrics, observability, startup, etc.
├── etl/                     # build_data, build_duckdb, build_agg, ingest_excel, transform_curated
├── models/                  # schema_contract, query_spec, registries
├── qa/                      # Validation scripts (check_no_legacy_imports, check_encoding, etc.)
├── .streamlit/
│   └── config.toml          # Theme and client options
├── data/
│   ├── input/               # Source CSVs/Excel (e.g. DATA_RAW.csv, Pretotype fallbacks)
│   ├── curated/             # ETL output (parquet, meta)
│   └── agg/                 # Aggregates
├── configs/                 # agg_policy, metrics_policy, duckdb_policy, etc.
├── schemas/                 # data_raw.schema.yml, canonical_columns, geo_region_map
├── docs/                    # data_contract, pipeline, DEPLOY.md
├── legacy/                  # Deprecated code (do not import from app/etl/models)
├── requirements.txt
├── pyproject.toml
├── README.md
└── .gitignore
```

- **Do not rely on:** `curated/` at repo root (legacy policy/manifests); use `data/curated/` and `data/agg/` for ETL outputs.
- **Root-level `analytics/`:** Used for `duckdb_views_manifest.json`; ETL writes `analytics.duckdb` at root unless overridden.

---

## 3. Streamlit Cloud checklist

| Item | Status |
|------|--------|
| **Main file path** | `app/main.py` |
| **requirements.txt** | At repo root; no local path dependencies |
| **.streamlit/config.toml** | Present (theme, client options) |
| **Paths** | All relative to repo root; no `C:\` or `/Users/` in code |
| **Python** | 3.10+ (pyproject.toml / runtime) |
| **Secrets** | Use Streamlit Cloud secrets if needed; do not commit `.env` (in .gitignore) |
| **Data at runtime** | Either run ETL in Cloud (e.g. in a setup script) or mount/provide `data/`, `analytics.duckdb` |

### 3.1 Streamlit Cloud setup (what to configure)

- **Main file path:** `app/main.py` (run from repository root).
- **Python version:** 3.10 or 3.11 (set in Streamlit Cloud app settings if needed).
- **Secrets / environment variables (optional):**
  - `ANTHROPIC_API_KEY` – required for NLQ/Intelligence Desk (Claude). Set in *Secrets* or *Environment variables*. If unset, the tab shows a safe message and does not crash.
  - `DUCKDB_PATH` – optional; override path to DuckDB file (e.g. `analytics.duckdb`).
  - `DUCKDB_SCHEMA` – optional; default `data`.
- **Special notes:** Ensure `data/` and/or a prebuilt `analytics.duckdb` are available (e.g. committed sample or ETL run). The app uses relative paths from repo root.

### 3.2 DuckDB backend (recommended for Streamlit Cloud)

To run the dashboard in the cloud using the **DuckDB single-file data backend**, the repository must contain these files (all paths relative to repo root):

```
repo_root/
├── analytics.duckdb                    # ≈ 4 MB — primary data source
├── analytics/
│   └── duckdb_views_manifest.json     # < 1 KB — schema, db_path, dataset_version
├── data/
│   └── curated/
│       └── metrics_monthly.meta.json   # < 1 KB — dataset_version fallback
├── app/
│   └── main.py
└── requirements.txt
```

- **analytics.duckdb** — Built by ETL (e.g. `etl/build_duckdb.py` or rebuild_analytics). Place at **repository root**. `.gitignore` allows this file via `!analytics.duckdb` so it can be committed (or use Git LFS for large files).
- **analytics/duckdb_views_manifest.json** — Must contain `db_path` (e.g. `"analytics.duckdb"`), `schema` (e.g. `"data"`), and optionally `dataset_version`, `reads_views_only: true`. The data gateway loads config from this first; the DB file must exist or the app raises a clear error.
- **data/curated/metrics_monthly.meta.json** — Used when dataset version is not read from the DuckDB `meta.dataset_version` table; must include `dataset_version`.

The app resolves `db_path` relative to repo root. No environment variables are required; optional overrides: `DUCKDB_PATH`, `DUCKDB_SCHEMA`.

### 3.3 Parquet backend (localhost/cloud parity)

For **identical KPI values** on localhost and Streamlit Cloud, use the parquet backend on both:

- **Environment variable (required for parity):** Set `APP_DATA_BACKEND=parquet` in Streamlit Cloud app settings (and locally if you want parity).
- **Required committed files (repo root–relative):**
  - `data/agg/manifest.json` (already committed)
  - `data/curated/metrics_monthly.meta.json` (already committed)
  - `data/agg/firm_monthly.parquet`, `channel_monthly.parquet`, `ticker_monthly.parquet`, `geo_monthly.parquet`, `segment_monthly.parquet` (production data; see §3.3.1 if not yet committed).
- **Optional debug:** `DEBUG_DATA_PARITY=1` or `SHOW_PARITY_DEBUG=1` to show the "Data parity" expander (path, row count, sums, OGR).

#### 3.3.1 Committing production parquet (if needed)

The repo uses `.gitignore` exceptions for the five production parquet files, so they are tracked by default. If they are not yet committed, add and commit once (they are small, &lt; ~500 KB total):

```bash
git add data/agg/firm_monthly.parquet data/agg/channel_monthly.parquet data/agg/ticker_monthly.parquet data/agg/geo_monthly.parquet data/agg/segment_monthly.parquet
```

(If your `.gitignore` does not have exceptions for these files, use `git add -f` before the paths.) Then commit and push. If you prefer not to commit parquet, run ETL in Cloud (e.g. setup script) or use the DuckDB backend (§3.2) and commit `analytics.duckdb` instead.

---

## 4. Files moved / organized (pre-deploy cleanup)

- **Root:** Removed 5× `Distribution Analytics - Pretotype(*).csv` from repo root → moved to `data/input/` (ETL fallback already looks in `excel_path.parent` = `data/input/`).
- **Legacy:** `legacy/` kept but not used by app/etl/models; CI enforces no legacy imports.
- **.gitignore:** Added at repo root (Python cache, venv, `.pytest_cache`, local `analytics.duckdb`, `data/curated/*.parquet`, `data/agg/*.parquet`, `.env`, logs, OS files).

---

## 5. What can break deploy (avoid)

- **Imports from `legacy`** in `app/`, `etl/`, `models/` → run `python qa/check_no_legacy_imports.py`.
- **Hardcoded Windows paths** → none found; all use `Path(__file__).resolve()` and relative paths.
- **Missing optional deps** → `requirements.txt` includes `tabulate>=0.9.0` for markdown export.
- **Large files in repo** → `.gitignore` excludes large parquet and DuckDB; use Streamlit Cloud’s resource limits and optional ETL run or external data.

---

## 6. Validation before deploy

```bash
python qa/check_no_legacy_imports.py
python qa/check_encoding.py
python -m py_compile app/main.py app/pages/visualisations.py app/pages/dynamic_report.py app/pages/nlq_chat.py
streamlit run app/main.py
```

Then open the app and confirm no startup errors and correct dataset version in the UI.

---

## 7. Deploy files (final)

### requirements.txt (production runtime)

```
streamlit>=1.28
pandas>=2.0
pydantic>=2
pyyaml
plotly
duckdb
tabulate>=0.9.0
reportlab
```

All are used at runtime: Streamlit, Pandas, Pydantic (config/contracts), PyYAML (configs), Plotly (charts), DuckDB (data gateway), tabulate (markdown export), reportlab (PDF export; optional fallback if missing).

### .streamlit/config.toml

- **\[server\]** `headless = true`, `enableCORS = false` for Cloud/Linux.
- **\[theme\]** and **\[client\]** as in repo (theme light, sidebar navigation off).

### packages.txt

**Not required.** The app does not use system libraries (e.g. no OpenGL, no system libs for DuckDB/Plotly). Do not create `packages.txt` unless you add a feature that needs APT packages.

### Entrypoint

- **Streamlit Cloud → Main file path:** `app/main.py`
- Run from repository root. The app adds the repo root to `sys.path` in `app/main.py`, so imports and paths work on Linux without change.

---

## 8. Points of attention (can still affect deploy)

| Risk | Mitigation |
|------|------------|
| **No data at first run** | App expects `data/curated`, `data/agg`, and/or `analytics.duckdb`. Either run ETL in Cloud (e.g. in a setup script or GitHub Action) or commit small sample data / prebuilt DuckDB so the app does not error on empty state. |
| **Build timeout** | If install is slow, pin versions in `requirements.txt` to avoid resolution delays. |
| **Memory** | DuckDB and Pandas load data in memory; large datasets may hit Cloud memory limits. |
| **Secrets** | Use Streamlit Cloud Secrets for API keys; never commit `.env`. |
| **Python version** | Streamlit Cloud uses a supported Python (e.g. 3.10+); match in `pyproject.toml` and runtime. |

---

## 9. Cloud/local parity verification

To confirm localhost and Streamlit Cloud produce identical results for the same dataset and defaults:

1. **Enable parity debug in Cloud:** Set environment variable `SHOW_PARITY_DEBUG=1` in Streamlit Cloud app settings (or use the in-app observability toggle if enabled). This reveals the "Debug / Data Contract" expander with environment, dataset path, version, fingerprint, row count, date range, totals, active scope, and active period mode.
2. **Use the same dataset:** Deploy with the same `data/` and/or `analytics.duckdb` (or same DATA_VERSION and data files) as local.
3. **First-load test:** In both environments, open the app in a fresh session (no prior session history), leave default filters and period (YTD, all dimensions "All"), and compare the values in **docs/PARITY_VALIDATION.md** checklist.
4. **Pass criteria:** Dataset fingerprint, row count, date range, total AUM/NNB/NNF, active scope ("Firm-wide"), active period mode ("YTD"), and the five top KPIs (End AUM, NNB, NNF, Organic Growth, Market Movement) must match. If any differ, treat as FAIL and investigate before considering parity resolved.
