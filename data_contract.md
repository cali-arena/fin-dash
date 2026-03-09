# Data Contract

## Required Sheets
- `DATA RAW`
- `DATA SUMMARY`
- `DATA MAPPING`
- `ETF`
- `Executive Summary`
- Pivot sheets: `All AUM by...`, `All by Date`, `All by SubSegment`, `All by Date&Ticker`

## Canonical Columns

### DATA RAW (required)
- `month_end` (from `Date`, month-end normalized)
- `channel_raw` (from `Channel`)
- `channel_standard` (from `Standard Channel`)
- `channel_best` (from `best_of_source`)
- `src_country`
- `product_country`
- `product_ticker`
- `segment`
- `sub_segment`
- `display_firm`
- `master_custodian_firm`
- `end_aum` (from `Asset Under Management`)
- `nnb` (from `Net new business`)
- `nnf` (from `net new base fees`)

### DATA SUMMARY (required)
- `month_end`
- `asset_growth_rate`
- `organic_growth_rate`
- `external_growth_rate`
- `velocity`
- `standard_deviation`
- `var`
- `inflation`
- `interest_rates`
- `currency_impact`

### DATA MAPPING (required for governance)
- `source_table`
- `source_field`
- `target_field`

## Metric Definitions
- `Begin AUM`: previous month `End AUM` at explicit slice keys (`channel`, `product_ticker`, `src_country`, `segment`, `sub_segment`).
- `End AUM`: summed AUM at the same slice keys and month.
- `NNB`: summed net new business at the same slice keys and month.
- `Market Impact`: `end_aum - begin_aum - nnb`.
- `OGR`: `nnb / begin_aum`, null when `begin_aum <= 0` or missing.
- `Market Impact Rate`: `market_impact / begin_aum`, null when `begin_aum <= 0` or missing.
- `Fee Yield`: `nnf / nnb`, null when `nnb <= 0` or missing.

## Aggregate Contract (data/agg + DuckDB v_* views)
- `firm_monthly`: `month_end` + metrics (`begin_aum`, `end_aum`, `nnb`, `nnf`, `ogr`, `market_impact`, `market_impact_rate`, `fee_yield`)
- `channel_monthly`: same metrics + `channel`
- `ticker_monthly`: same metrics + `product_ticker`
- `geo_monthly`: same metrics + `src_country`
- `segment_monthly`: same metrics + `segment`, `sub_segment`

All aggregate/view tables must expose canonical `month_end` (no `date/month/period` aliases).
