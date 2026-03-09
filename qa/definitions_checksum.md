# Definitions checksum — DATA SUMMARY and coded equivalents

Canonical metric definitions used by (or validated against) DATA SUMMARY. **Single source of truth for firm-level rates:** `app/metrics/data_summary_formulas.py`. Sources: `data_contract.md`, `etl/transform_curated.py`, `etl/ingest_excel.py`, `qa/validate_vs_data_summary.py`, `app/metrics/metric_contract.py`, `configs/metric_contract.yml`, `app/config/report_contract.yml`, `models/registries/metric_registry.yml`, `app/nlq/metric_registry.yml`.

**Note:** The DATA SUMMARY sheet (Excel) holds only firm-level *rates* (asset_growth_rate, organic_growth_rate, external_growth_rate). It does not define End AUM, Begin AUM, NNB, or Fee Yield; those definitions live in the data contract and ETL/agg builders. Validation uses `data_summary_formulas.compute_firm_rates_df()` so recomputed rates match DATA SUMMARY exactly. Rows with begin_aum missing or ≤ 0 are recategorized as MISSING_DATA (not FORMULA_MISMATCH).

---

## End AUM

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | No formula. Source column: summed AUM at slice and month. |
| **Source** | DATA RAW column "Asset Under Management" → `end_aum` (`etl/ingest_excel.py` RAW_COLUMN_MAP). Curated: `etl/transform_curated.py` groups by `month_end` + slice keys and sums `end_aum`. |
| **Numerator** | — |
| **Denominator** | — |
| **Grain** | `month_end`, plus slice: `channel`, `product_ticker`, `src_country`, `segment`, `sub_segment` (`etl/transform_curated.py` SLICE_KEYS). Firm-level: one row per `month_end` (sum over slices in `qa/validate_vs_data_summary.py`). |
| **Edge-case handling** | None (level, not derived). |

---

## Begin AUM

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | Previous month’s End AUM at the same slice keys. |
| **Source** | `data_contract.md`: "previous month End AUM at explicit slice keys". `etl/transform_curated.py`: `agg["begin_aum"] = agg.groupby(SLICE_KEYS)["end_aum"].shift(1)`. |
| **Numerator** | — |
| **Denominator** | — |
| **Grain** | Same as End AUM: `month_end` + slice keys. |
| **Edge-case handling** | **First month per slice:** no prior month → `begin_aum` is NA (`shift(1)`). Downstream: when `begin_aum` is NA or ≤ 0, OGR and market_impact_rate are set to `pd.NA` (`etl/transform_curated.py` lines 52–53). `app/metrics/metric_contract.py`: `coerce_num` turns None/invalid into NaN; divisions use `safe_divide` (zero/NaN denominator → NaN). |

---

## NNB

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | No formula. Source column: summed net new business at slice and month. |
| **Source** | DATA RAW column "Net new business" → `nnb` (`etl/ingest_excel.py`). Curated: `etl/transform_curated.py` groups by `month_end` + slice keys and sums `nnb`. |
| **Numerator** | — |
| **Denominator** | — |
| **Grain** | Same as End AUM: `month_end` + slice keys. Firm-level: one row per `month_end` (sum over slices). |
| **Edge-case handling** | None for the level. When NNB ≤ 0 or missing, **Fee yield** (in ETL) is set to `pd.NA` — see Fee Yield. |

---

## OGR (Organic Growth Rate)

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | `nnb / begin_aum`. |
| **Source** | `data_contract.md`: "nnb / begin_aum, null when begin_aum <= 0 or missing". `etl/transform_curated.py`: `agg["ogr"] = agg["nnb"] / agg["begin_aum"]`; then `invalid_begin` → `ogr` set to `pd.NA`. `app/metrics/metric_contract.py`: `compute_ogr(nnb, begin_aum)` = `safe_divide(nnb, begin_aum)`. Firm-level validation: `qa/validate_vs_data_summary.py` `organic_growth_rate_calc` = `_rate(firm, "nnb", "begin_aum")` with `_rate`: `num/den` where `den > 0`, else null. |
| **Numerator** | `nnb`. |
| **Denominator** | `begin_aum`. |
| **Grain** | Slice: `month_end` + slice keys. Firm: `month_end` (summed nnb and begin_aum then ratio). |
| **Edge-case handling** | **begin_aum missing or ≤ 0:** OGR = null/NaN. **First month per slice:** begin_aum = NA → OGR = NA. `configs/metric_contract.yml` ogr guards: "if begin_aum==0 => NaN". |

---

## Market Impact (absolute)

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | `end_aum - begin_aum - nnb`. |
| **Source** | `data_contract.md`: "end_aum - begin_aum - nnb". `etl/transform_curated.py`: `agg["market_impact"] = agg["end_aum"] - agg["begin_aum"] - agg["nnb"]`. `app/metrics/metric_contract.py`: `compute_market_impact(begin_aum, end_aum, nnb)` = `end - begin - n`; any NaN input → NaN. `configs/metric_contract.yml`: market_impact canonical "end_aum - begin_aum - nnb". Firm-level validation: firm-level sums then `external_growth_rate_calc` = `_rate(firm, "market_impact", "begin_aum")`. |
| **Numerator** | — (additive expression). |
| **Denominator** | — |
| **Grain** | Same as End AUM: `month_end` + slice keys. Firm: `month_end` (sum of market_impact over slices). |
| **Edge-case handling** | **First month per slice:** begin_aum = NA → expression yields NA (pandas/coerce_num). Any NaN in end, begin, or nnb → result NaN in `metric_contract.py`. |

---

## Market Impact (rate)

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | `market_impact / begin_aum`. |
| **Source** | `data_contract.md`: "market_impact / begin_aum, null when begin_aum <= 0 or missing". `etl/transform_curated.py`: `agg["market_impact_rate"] = agg["market_impact"] / agg["begin_aum"]`; `invalid_begin` → `market_impact_rate` set to `pd.NA`. `app/metrics/metric_contract.py`: `compute_market_impact_rate(market_impact, begin_aum)` = `safe_divide(market_impact, begin_aum)`. |
| **Numerator** | `market_impact` (absolute). |
| **Denominator** | `begin_aum`. |
| **Grain** | Same as OGR: `month_end` + slice keys (or firm `month_end` if aggregated). |
| **Edge-case handling** | **begin_aum missing or ≤ 0:** rate = null/NaN. Same as OGR for first month and zero/missing begin_aum. `configs/metric_contract.yml` market_impact_rate guards: "if begin_aum==0 => NaN". |

---

## Fee Yield

| Item | Value |
|------|--------|
| **Formula (DATA SUMMARY / closest coded equivalent)** | **Two definitions in codebase.** |
| **Definition A (data contract + ETL)** | `nnf / nnb`. Null when `nnb` ≤ 0 or missing. |
| **Source A** | `data_contract.md`: "nnf / nnb, null when nnb <= 0 or missing". `etl/transform_curated.py`: `agg["fee_yield"] = agg["nnf"] / agg["nnb"]`; `agg.loc[agg["nnb"].isna() \| (agg["nnb"] <= 0), "fee_yield"] = pd.NA`. |
| **Definition B (app + config + NLQ registry)** | `nnf / average_aum` with `average_aum = (begin_aum + end_aum) / 2`. Optional annualization: base × annualize_factor (default 12) when mode is "annualized". |
| **Source B** | `configs/metric_contract.yml`: fee_yield canonical "nnf / average_aum", average_aum "(begin_aum + end_aum) / 2". `app/metrics/metric_contract.py`: `compute_fee_yield(nnf, begin_aum, end_aum)` = nnf / ((begin+end)/2), then optional annualize. `app/config/report_contract.yml`: firm_fee_yield formula "nnf / avg_aum". `models/registries/metric_registry.yml` and `app/nlq/metric_registry.yml`: fee_yield formula "expr:nnf / ((begin_aum + end_aum) / 2)". |
| **Numerator** | A: `nnf`. B: `nnf`. |
| **Denominator** | A: `nnb`. B: `(begin_aum + end_aum) / 2`. |
| **Grain** | Same as other metrics: `month_end` + slice keys (or firm). |
| **Edge-case handling** | **A:** NNB ≤ 0 or missing → fee_yield = pd.NA (`etl/transform_curated.py`). **B:** `safe_divide(nnf, avg)` → when avg = 0 or NaN, result NaN; no explicit NNB≤0 guard in `metric_contract.py`. When `(begin+end)/2` is 0, fee_yield = NaN. Annualization: read from `configs/metric_contract.yml` (mode, annualize_factor default 12). |

**Discrepancy:** DATA SUMMARY / data_contract and ETL use **nnf/nnb**. App layer and report/NLQ registries use **nnf/avg_aum**. UNRESOLVED which is authoritative for "DATA SUMMARY" reporting; ETL is the only place that writes curated fee_yield into the pipeline used for validation inputs.

---

## DATA SUMMARY sheet (firm-level rates only)

| Item | Value |
|------|--------|
| **What it contains** | Expected firm-level rates only: `asset_growth_rate`, `organic_growth_rate`, `external_growth_rate`, plus velocity, standard_deviation, var, inflation, interest_rates, currency_impact. No AUM/NNB/Fee Yield columns. Ingest: `etl/ingest_excel.py` sheet "DATA SUMMARY" → `data_summary_normalized.parquet` (columns snake_cased; numeric columns divided by 100). |
| **Validation** | `qa/validate_vs_data_summary.py`: reads `metrics_monthly.parquet` and `data_summary_normalized.parquet`; aggregates metrics to firm by `month_end` (sum begin_aum, end_aum, nnb, market_impact); computes asset_growth_rate_calc = (end_aum - begin_aum) / begin_aum, organic_growth_rate_calc = nnb / begin_aum, external_growth_rate_calc = market_impact / begin_aum (with _rate: denominator > 0 only); compares to DATA SUMMARY expected rates; writes `qa/validation_report.csv`. |
| **Firm-level formulas (validation)** | asset_growth_rate = (end_aum - begin_aum) / begin_aum; organic_growth_rate = nnb / begin_aum; external_growth_rate = market_impact / begin_aum. Grain: one row per `month_end` (firm). Edge: `_rate` returns null where denominator ≤ 0. |

---

## Summary table

| Metric | Formula (primary) | Numerator | Denominator | Grain | Edge cases |
|--------|-------------------|-----------|-------------|--------|------------|
| End AUM | source column, sum at slice+month | — | — | month_end + slice | — |
| Begin AUM | prior month end_aum at same slice | — | — | month_end + slice | First month: NA |
| NNB | source column, sum at slice+month | — | — | month_end + slice | — |
| OGR | nnb / begin_aum | nnb | begin_aum | month_end + slice | begin_aum NA or ≤0 → null |
| Market Impact (abs) | end_aum - begin_aum - nnb | — | — | month_end + slice | Any input NA → NA |
| Market Impact (rate) | market_impact / begin_aum | market_impact | begin_aum | month_end + slice | begin_aum NA or ≤0 → null |
| Fee Yield | A: nnf/nnb; B: nnf/((begin+end)/2) | nnf | A: nnb; B: avg_aum | month_end + slice | A: nnb≤0 → null; B: avg=0 → NaN |
