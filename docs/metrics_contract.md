# Metrics contract

Governed definition of grains, sign conventions, formulas, and missing-data behavior. Config: `configs/metric_contract.yml`. This doc explains dataset sources, grain rules, formulas in plain English, sign conventions, and fee yield.

---

## Dataset sources

Metrics are built from these logical datasets:

| Source | Grain | Use |
|--------|--------|-----|
| **firm_monthly** | One row per `month_end` (global slice) | Executive summary, firm snapshot, firm-level KPIs and time series. |
| **channel_monthly** | One row per `(month_end, channel)` | Channel view: AUM, flows, and rates by channel. |
| **ticker_monthly** | One row per `(month_end, ticker)` | Ticker view: AUM, flows, and rates by ticker. |

- **firm_snapshot** uses the latest `month_end` row from firm_monthly (global slice).
- **time_series** uses the monthly series at `month_end` (e.g. last 12 or 24 months).
- **channel_view** and **ticker_view** are filtered by a `month_end` range and grouped by `channel` or `ticker` respectively.

---

## Grain rules

- **firm_snapshot**: Key = `[month_end]`. One row per month; “latest” means the most recent `month_end` in the selected range. Definition: latest month_end row from firm_monthly global slice.
- **time_series**: Key = `[month_end]`. One row per month; used for sparklines and trend series.
- **channel_view**: Key = `[month_end, channel]`. Rows are filtered by month_end range then grouped by channel.
- **ticker_view**: Key = `[month_end, ticker]`. Rows are filtered by month_end range then grouped by ticker.

All rate and flow metrics must be defined with respect to one of these grains; aggregations (e.g. rollups) must preserve or explicitly document grain changes.

---

## Formulas (plain English)

### Market impact

- **Canonical**: `market_impact = end_aum - begin_aum - nnb`
- **Meaning**: The change in AUM that is not explained by net new business (NNB). It is the “market” component: price and FX moves, revaluations, etc.
- **Reconciliation**: The waterfall must satisfy: **Begin AUM + NNB + Market impact = End AUM**. Tolerances (absolute and relative) in the contract define when a mismatch is acceptable.

### OGR (organic growth rate)

- **Canonical**: `ogr = nnb / begin_aum`
- **Meaning**: Net new business as a fraction of beginning AUM. Interpreted as a decimal (e.g. 0.0123 = 1.23%).
- **Guard**: If `begin_aum == 0`, the result is **NaN** (no rate is defined).

### Market impact rate

- **Canonical**: `market_impact_rate = market_impact / begin_aum`
- **Meaning**: Market impact as a fraction of beginning AUM. Decimal (e.g. 0.004 = 0.40%).
- **Guard**: If `begin_aum == 0`, the result is **NaN**.

### Fee yield

- **Canonical**: `fee_yield = nnf / average_aum`, where `average_aum = (begin_aum + end_aum) / 2`.
- **Meaning**: Net fees (NNF) over average AUM over the period. Represents fee revenue relative to average balance.
- **Annualization**: The contract supports:
  - **monthly**: raw monthly ratio (nnf / average_aum for that month).
  - **annualized**: multiply by `annualize_factor` (e.g. 12) to express as an annualized rate.
- The chosen mode (monthly vs annualized) must be explicit in config so reporting and QA are consistent.

---

## Sign conventions

### AUM

- **begin_aum**, **end_aum**: Non-negative (≥ 0). Represent balances.

### Flows

- **NNB (net new business)**: **Signed**.  
  - **Positive** = net inflows (client money in).  
  - **Negative** = net outflows (client money out).  
  This is the standard convention for “flow” in AUM waterfall: inflows add to AUM, outflows subtract.

- **NNF (net new fees)**: **Signed**.  
  - Typically **positive** = fee revenue (income).  
  - **Negative** = if the business defines fees as expense or rebates; the contract leaves “expense negative” as an optional definition to be set per deployment.

### Market

- **market_pnl** (or market impact in currency): **Signed**.  
  - **Positive** = market gain (prices/FX moved in our favor).  
  - **Negative** = market loss.

### Rates

- **ogr**, **market_impact_rate**, **fee_yield**: Stored as **decimals** (e.g. 0.0123 = 1.23%, 0.004 = 0.40%). Display layer multiplies by 100 and adds “%” when needed.

---

## Fee yield definition and annualization

- **Definition**: Fee yield is **NNF / average AUM**, with `average_aum = (begin_aum + end_aum) / 2` for the same period (and grain).
- **Monthly vs annualized**:
  - **monthly**: One period (one month); ratio is “fees in that month over average AUM in that month.” No scaling.
  - **annualized**: Same ratio multiplied by an **annualize_factor** (e.g. 12) so the number is comparable to an annual rate. Config must set `mode` and `annualize_factor` explicitly.
- Keeping this explicit in the contract avoids mixing monthly and annualized fee yields in the same view without documentation.

---

## Tolerances and missing-data policy

- **Waterfall**: Reconciliation (Begin + NNB + Market = End) is checked with:
  - **waterfall_abs**: Max allowed absolute difference (e.g. 1e-6).
  - **waterfall_rel**: Max allowed relative difference (e.g. 1e-9).
- **Division by zero**: Yields **nan** (per `missing_data_policy.division_by_zero`).
- **Inf**: Converted to **nan** when `inf_to_nan` is true.
- **Empty selection**: No rows for the selected filters → **no_data_panel** (or equivalent) in the UI; no implicit zero or inf.

Reference: `configs/metric_contract.yml` for the authoritative YAML.
