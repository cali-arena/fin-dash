# Join contract (star)

## Principles

- **Fact is authoritative.** All row counts and grain come from the fact table. The UI and reporting layer must never change row count.
- **Dimensions are lookup/enrichment only.** They add attributes (e.g. region, quarter, segment) via explicit joins. No ad-hoc merges in the Streamlit/UI layer.

## Allowed joins and keys

All joins are defined in `pipelines/contracts/star_contract.py`. The UI must load **either** `curated/fact_enriched.parquet` **or** call `load_fact_enriched()` — no direct `pd.merge` or `.merge()` in app code.

| Join        | Fact column(s)              | Dim table   | Dim column(s)   |
|------------|-----------------------------|-------------|------------------|
| Time       | `month_end`                 | dim_time    | `month_end`      |
| Channel    | `channel_key` or `preferred_label` (prefer key if present in both) | dim_channel | same             |
| Product    | `product_ticker`            | dim_product | `product_ticker`  |
| Geo (src)  | `src_country_canonical`     | dim_geo     | `country`        |
| Geo (product) | `product_country_canonical` | dim_geo     | `country`        |

- Joins are **left** joins: every fact row is kept; dimension attributes are added where keys match.
- Geo canonical columns are derived from `src_country` / `product_country` via `normalize_country` when not already present in the fact.

## Row-count invariant

After all joins, **row count must equal the fact row count.** The contract asserts this in `load_fact_enriched()`. Dimension tables are de-duplicated on their join key before merge so the left join never expands rows.

## How to regenerate dims and enriched fact

1. **Build dimensions** (from curated fact):
   ```bash
   python -m pipelines.dimensions.build_dimensions
   ```
   Reads `curated/fact_monthly.parquet`, writes `curated/dim_*.parquet` and sidecar `.meta.json`. Requires all dimension columns present and (for Step 4 persist) non-empty dims.

2. **Build enriched fact and join coverage** (contract):
   ```bash
   python -m pipelines.contracts.star_contract --build-enriched
   ```
   Reads fact + dims, joins via the contract only, writes:
   - `curated/fact_enriched.parquet`
   - `qa/join_coverage.json` (coverage rates and top missing keys per join)

3. **UI** loads data via `app.data_loader.get_dataset(mode="enriched")`, which uses `fact_enriched.parquet` if present, otherwise calls `load_fact_enriched()` once.

## Join coverage QA

- `qa/join_coverage.json` contains per-join coverage rates and (e.g. top 20) missing keys when fact keys are not in the dimension.
- Use it to audit gaps (e.g. product_tickers in fact but not in dim_product) and fix upstream curation or dim build.

## UI guardrails

- **Anti-pattern check:** On startup, the app scans the `app/` directory for `.merge(` or `pd.merge(`. If any are found, it logs a warning and, if `STRICT_UI_JOINS=true`, raises so the run fails.
- All visuals and KPIs must use the single enriched DataFrame returned by `get_dataset()`; no merges in the Streamlit layer.
