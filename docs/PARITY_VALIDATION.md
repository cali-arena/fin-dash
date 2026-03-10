# Cloud / Local Parity Validation Checklist

Use this checklist to verify that **localhost** and **Streamlit Cloud** produce identical results for the same dataset and default first-load state.

---

## Prerequisites

- **Same dataset:** Both environments use the same data (same `data/` and/or `analytics.duckdb`, or same DATA_VERSION and files).
- **First-load parity test:** In both environments, open the app in a **new session** (clear cookies / incognito or new browser session). Do **not** change any filters or period before comparing.
- **Enable parity debug (optional but recommended):**  
  - **Local:** Turn on "Observability" / dev toggle, or set `DEV_MODE=1` or `SHOW_PARITY_DEBUG=1`.  
  - **Cloud:** Set app environment variable `SHOW_PARITY_DEBUG=1` in Streamlit Cloud settings.  
  Then open the **"Debug / Data Contract"** expander (and, if needed, **"State / Cache (filter, scope, period)"**).

---

## Items to compare (exact match required)

| # | Item | Where to read | Local value | Cloud value | Match? |
|---|------|----------------|-------------|-------------|--------|
| 1 | Dataset version | Debug / Data Contract → "Dataset version (DATA_VERSION)" | | | |
| 2 | Dataset fingerprint / hash | Debug / Data Contract → "Dataset fingerprint" | | | |
| 3 | Resolved dataset path | Debug / Data Contract → "Dataset path" | | | |
| 4 | Row count | Debug / Data Contract → "Row count" | | | |
| 5 | Min date | Debug / Data Contract → "Date range" (min) | | | |
| 6 | Max date | Debug / Data Contract → "Date range" (max) | | | |
| 7 | Total AUM (Sum End AUM) | Debug / Data Contract → "Sum End AUM" | | | |
| 8 | Total NNB | Debug / Data Contract → "Sum NNB" | | | |
| 9 | Total NNF | Debug / Data Contract → "Sum NNF" | | | |
| 10 | Active scope | Debug / Data Contract → "Active scope" | | | |
| 11 | Active period mode | Debug / Data Contract → "Active period mode" | | | |

---

## Top KPI outputs (Executive Dashboard tab, same first-load state)

With **default filters** and **default period (YTD)** and **all dimensions "All"** (firm-wide), compare:

| # | KPI | Where to read | Local value | Cloud value | Match? |
|---|-----|----------------|-------------|-------------|--------|
| 12 | End AUM | Core metrics / narrative | | | |
| 13 | Net New Business (NNB) | Core metrics | | | |
| 14 | Net New Flow (NNF) | Core metrics | | | |
| 15 | Organic Growth (OGR) | Core metrics / waterfall | | | |
| 16 | Market Movement | Core metrics / waterfall | | | |

---

## First-load parity test (mandatory)

1. **No prior session:** Use a fresh browser session (or clear app state) in both local and cloud.
2. **Same default filters:** Do not change date range, period mode, or any dimension filter before comparing.
3. **Same period mode:** Default is **YTD**; ensure both environments show YTD.
4. **Same scope:** Default is firm-wide (**Active scope: Firm-wide**); all dimension filters should be "All".

Then fill the tables above. If **any** cell in "Match?" is **No**, parity is **not** resolved—investigate (dataset, cache, environment, or code path) before considering the release validated.

---

## Pass / Fail criteria

- **PASS:** All 16 items match between local and cloud for the same dataset and first-load state. No hidden alternate KPI or data path is active.
- **FAIL:** Any of (1) dataset fingerprint differs, (2) default scope or period mode differs, (3) any of the 11 contract items or 5 KPI outputs differ for the same first-load state. **Stop and investigate**; do not consider parity resolved until all match.

---

## Rollback

If after deploy you find parity or correctness issues, revert to the previous release tag or commit and redeploy. See the release note for the exact rollback commit range. Re-run this checklist after rollback to confirm restored state.
