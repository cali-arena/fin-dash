# Finance Dashboard — Runbook

Reproducible runbook for the Streamlit BI/NLQ app. Use this to run the demo end-to-end, troubleshoot common issues, and update data safely.

---

## 1) Overview

- **What it is:** A three-tab Streamlit dashboard for finance metrics (AUM, NNB, OGR, etc.).
  - **Tab 1 (Visualisations):** Charts, ranked tables (Top Channels / Top Tickers), drill-down Details panel. All data via `app.data_gateway` (DuckDB views or Parquet fallback).
  - **Tab 2 (Dynamic Report):** Deterministic report from `ReportPack` + report engine. Sections: Overview, Channels, Products/ETF, Geo, Anomalies, Recommendations. No LLM; bullets and tables from governed rules and thresholds.
  - **Tab 3 (NLQ Chat):** Natural-language questions → governed parser → `QuerySpec` → executor. **No free-form SQL:** only validated intents, parameterized WHERE clauses, row caps, and export-mode caps (5k default, 50k when Export mode is on).
- **Governance:** All data access goes through `app.data_gateway.run_query` / `run_chart`. Pages must not use DuckDB or Parquet directly. Guardrail script enforces this at dev/CI (and at Streamlit startup when `DEV_MODE=1`).

---

## 2) Prerequisites

- **Python:** 3.10+ (recommended; check with `python --version`).
- **Docker:** Not required to run the app; use only if your ETL or runbooks use containers.
- **Memory:** 8GB+ RAM recommended for full pipeline (DuckDB + Parquet + Streamlit). Narrow date ranges and filters reduce load.
- **Optional dependencies:**
  - **duckdb:** For querying the analytics layer (DuckDB views). If missing or not configured, gateway falls back to `agg/*.parquet`.
  - **reportlab:** For PDF summary export (footer with dataset_version). If missing, PDF download is skipped; use browser **Print → Save as PDF** as fallback.
  - **streamlit-plotly-events:** Optional for click-to-select on charts (if used in the UI contract).
  - **weasyprint:** Not used by the app; PDF is generated via `reportlab` in `app.export_utils`.

---

## 3) First run steps

1. **Clone / open project** and go to project root:
   ```bash
   cd /path/to/FINANCE_DASH
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   # source .venv/bin/activate   # Linux/macOS
   pip install -e .   # if using pyproject.toml
   # OR: pip install -r requirements.txt   # if present
   ```
   Ensure at least: `streamlit`, `pandas`, `plotly`. For DuckDB backend: `duckdb`. For PDF: `reportlab`.

3. **Configure environment variables (optional but useful):**
   - `DEV_MODE=1` — Enables dev sidebar (observability, cache debug, state debug), and runs the no-direct-data-access guardrail at startup (app stops if violations found).
   - `DATASET_VERSION` — Override only when gateway cannot read version (e.g. `curated/metrics_monthly.meta.json` missing); the app will show this as dataset version.
   - `PREWARM=1` — Run cache prewarm on first load (Tab 1 queries) to reduce first-click latency.
   - `STRICT_GATEWAY=1` — Fail fast if any page imports `duckdb` (use for CI).

   Example (Windows PowerShell):
   ```powershell
   $env:DEV_MODE = "1"
   $env:PREWARM = "1"
   ```
   Example (Linux/macOS):
   ```bash
   export DEV_MODE=1
   export PREWARM=1
   ```

4. **Ensure data is built** (see §5). Minimum for app to start:
   - `data/curated/metrics_monthly.meta.json` (or `curated/`) with `"dataset_version": "<version>"`.
   - Either DuckDB (e.g. `analytics.duckdb` or `analytics/duckdb_views_manifest.json` + DB) or `data/agg/manifest.json` + Parquet files.
   - From project root: **`make build-data`** produces `data/curated/`, `data/agg/`, and `analytics.duckdb`; or use **`make build-data-docker`** inside Docker.

5. **Start the app:**
   - Local: **`make run`** or `streamlit run app/main.py`
   - Docker: **`make up`** or `docker compose up --build`
   Default URL: `http://localhost:8501`. From the sidebar, set date range and filters; use **Export mode** for full CSV exports (up to 50k rows) where supported.

---

## 4) Common issues + fixes

| Issue | Cause | Fix |
|-------|--------|-----|
| **DuckDB file locked** | Another process (e.g. pipeline or second Streamlit) has the DB open, or rebuild ran while app was running. | Stop the app and any pipeline; run rebuild with no other process using the DB. Use `read_only=True` (gateway already does). |
| **Dataset not found / wrong version** | `curated/metrics_monthly.meta.json` missing or invalid; or `dataset_version` key missing/empty. | Create or fix `curated/metrics_monthly.meta.json` with `{"dataset_version": "<version>"}`. Regenerate via `python -m pipelines.duckdb.rebuild_analytics_layer` (or your ETL that writes this file). |
| **No data under filters** | Date range or dimension filters exclude all rows. | Widen date range or relax filters (e.g. clear channel/ticker/slice). Message shown: "No data under current filters" with hint "Widen date range or relax filters." |
| **Slow query / timeout** | Heavy query over large range or many dimensions; or DuckDB statement timeout / budget exceeded. | Narrow filters (shorter date range, fewer dimensions). Enable **Export mode** only when needed. Use `PREWARM=1` to warm caches. Observability panel (dev) shows `perf_query_log` with `elapsed_ms` and `over_budget` warnings. |
| **PDF export missing** | `reportlab` not installed; `make_pdf_with_footer` returns `None`. | Install: `pip install reportlab`. Or use browser **Print → Save as PDF** for the current view. |

---

## 5) How to update data (ETL)

- **Where raw/curated data lives:**
  - Aggregates: `agg/*.parquet` and `agg/manifest.json` (from `pipelines.agg.build_aggs`).
  - Dataset version: `curated/metrics_monthly.meta.json` (single source of truth for `dataset_version`).
  - DuckDB: built from Parquet; path and schema come from `analytics/duckdb_views_manifest.json` or `analytics/duckdb_manifest.json` or `configs/duckdb_policy.yml`.

- **Refresh / rebuild datamart:**
  - **Single entrypoint (recommended):** **`make build-data`** runs `python etl/build_data.py` and produces `data/curated/`, `data/agg/`, and `analytics.duckdb`. Idempotent; same inputs → same outputs.
  - **Docker:** **`make build-data-docker`** (or `docker compose --profile etl run --rm etl`) builds the same artifacts into the mounted `./data` and `./analytics.duckdb`.
  - **Legacy pipelines:** Alternatively: `python -m pipelines.agg.build_aggs`, then `python -m pipelines.duckdb.rebuild_analytics_layer --policy configs/duckdb_policy.yml`. Ensure `data/curated/metrics_monthly.meta.json` (or `curated/`) contains `"dataset_version": "<version>"`.

- **Validate row counts / schema drift:**
  - Use pipeline QA steps if available (e.g. `python -m pipelines.duckdb.qa_duckdb_layer`).
  - Check `agg/manifest.json` for table list and paths; ensure Parquet files exist and columns match what the gateway expects (see `QUERY_SPECS` in `app.data_gateway`).

- **Value catalogs (NLQ):** Built at runtime from the gateway: NLQ tab calls `run_query(Q_CHANNEL_MONTHLY, gateway_dict, root)` and derives distinct values for dimensions (`channel`, `product_ticker`, `src_country`, `segment`, `month_end`) to validate filter values. Cached by `dataset_version`. No separate catalog file required.

---

## 6) Dataset versioning

- **How `dataset_version` is detected/declared:**
  - Read from `curated/metrics_monthly.meta.json` → key `dataset_version` (see `load_dataset_version()` in `app.data_gateway`). If the file or key is missing, the gateway raises; in some entry points a fallback uses `os.environ.get("DATASET_VERSION", "placeholder")` so the app can still start (e.g. for demos).

- **Switching versions safely:**
  - Point the app at a different project root (different `curated/` and `agg/`), or replace `curated/metrics_monthly.meta.json` and rebuild so the new version is written there. Restart the app so it picks up the new file. Avoid changing version while the app is running (cache keys include `dataset_version`).

- **Cache invalidation:** Cache keys include `dataset_version` (e.g. `details::{dataset_version}::{filter_hash}::{drill_hash}`, gateway query caches). Changing `dataset_version` in the meta file and restarting the app effectively invalidates caches. With Streamlit, restart the process or use dev tools (e.g. "Clear caches" in sidebar when `DEV_MODE=1`) to clear in-memory caches.

---

## 7) Smoke test checklist

Use this to verify the app end-to-end after setup or after data/version changes.

1. **Open app:** Run `streamlit run app/main.py`, open `http://localhost:8501`. Sidebar shows Data version (dataset_version, policy_hash). No startup errors.

2. **Tab 1 — Visualisations:**
   - Set a **date range** and optional filters (e.g. last 12 months). Confirm charts and ranked tables (Top Channels, Top Tickers) render.
   - Export a ranked table: click **Download CSV (current view)** under Top Channels or Top Tickers; confirm a CSV downloads with expected columns and filename like `tab1_top_channels__rows-N__YYYY-MM-DDTHH-MM.csv`.

3. **Tab 2 — Dynamic Report:**
   - Click **Dynamic Report**. Wait for report to load (one `get_report_pack` call). Confirm sections (Overview, Channels, Products/ETF, Geo, Anomalies, Recommendations) show bullets and tables where data exists.
   - Export HTML: click **Download HTML**; confirm an HTML file downloads.
   - Export a section CSV: under any section that has a table, click **Download CSV (current view)**; confirm filename like `tab2_overview_movers__rows-N__....csv`.

4. **Tab 3 — NLQ Chat:**
   - Run **3 curated NLQ queries** (e.g. "NNB by channel last 12 months", "AUM by ticker", "Top channels by NNB"). Confirm table and optional chart render; no parse/validation errors.
   - Export CSV: click **Download CSV (current view)** (and if Export mode is on, **Export full CSV**); confirm CSV uses raw data and filename like `tab3_nlq_result__rows-N__....csv`.
   - Open **Debug (dev)** expander; confirm **QuerySpec** JSON is present and matches the question.

5. **Observability (dev):** With `DEV_MODE=1`, open the observability panel (sidebar or in-tab). Confirm **perf_query_log** shows entries with `hit`/`miss`, `elapsed_ms`, and row caps where applicable; no unexpected errors.

---

## 8) Make and QA

- **`make build-data`** — Build curated + agg parquet and `analytics.duckdb` locally.
- **`make up`** — Start the app with Docker Compose (`docker compose up --build`).
- **`make qa`** — Run unit tests (`pytest -q`).
- **Demo scenarios:** Reproducible demo steps and pass/fail checklist are in **`qa/demo_scenarios.md`**.

---

## 9) QA: Validation report expected outcome

- **What it is:** The validation pipeline compares recomputed firm-level growth rates (from `metrics_monthly` + DATA SUMMARY formulas) to the expected **DATA SUMMARY** (`data_summary_normalized.parquet`). Output: **`qa/validation_report.csv`**.
- **How to run:** From repo root:  
  `python qa/validate_vs_data_summary.py --curated-dir data/curated --qa-dir qa`  
  (Ensure `PYTHONPATH` includes the project root or run after `pip install -e .`.)
- **Expected outcome:** Exit **0** (validation passed). Any row with `any_fail` = True that is **not** an acceptable SKIP is a failure; the pipeline exits **2** in that case.
- **Acceptable “fails”:** The only acceptable non-pass categories are:
  - **MISSING_DATA** — First month or month with `begin_aum` missing/≤ 0; rates not computable; not counted as formula failure.
  - **SKIP_INCOMPLETE_COVERAGE** — Summary has a month that does not exist in actual (firm) data; coverage incomplete; not counted as formula failure.
- **Triage:** See **`qa/validation_triage.md`** for counts by `fail_reason` and how to interpret the report.

---

## 10) How to interpret fail_reason categories

| fail_reason | Meaning | Counts as failure? |
|-------------|--------|---------------------|
| *(empty)* | Row passed: reported and recomputed rates within threshold. | No |
| **MISSING_DATA** | `begin_aum` missing or ≤ 0 for this month (e.g. first available month). Rates not comparable. | No (acceptable SKIP) |
| **SKIP_INCOMPLETE_COVERAGE** | Summary has this month but actual (firm) data has no row for it; coverage incomplete. | No (acceptable SKIP) |
| **SKIPPED_GRAIN** | Grain mismatch (actual vs expected); row excluded from formula comparison. | No |
| **FORMULA_MISMATCH** | Reported rate (e.g. asset_growth_rate) differs from formula-derived value by more than threshold. | **Yes** — investigate data or formula. |

Pipeline exit: **0** if no rows count as failure; **2** if any **FORMULA_MISMATCH** (or other non-SKIP) exists.

---

*Runbook version: 1.0. No secrets or credentials; use project/config docs for env-specific values.*
