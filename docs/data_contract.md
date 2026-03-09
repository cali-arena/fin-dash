## A) Overview

- **Source-of-truth**: `data/input/*.csv`
- **Contract version**: 2026-03-03

- **Scope**: This contract defines the **required presence, filenames, schemas, and grains** for the CSV inputs consumed by the finance dashboard.
- **Breaking change (definition)**: Any change that can break existing consumers, including but not limited to:
  - Removing a required file.
  - Renaming a required file without versioning.
  - Removing a required column.
  - Renaming a required column.
  - Changing the semantic grain of a file (what a row represents).
  - Changing a column’s data type in a non-backward-compatible way.
  - Changing nullability of a previously required column.

All breaking changes **must** be implemented via a **new contract version** (e.g. new subdirectory `data/input_vYYYYMMDD` or similar versioning scheme agreed by the team) and documented in this file.

---

## B) Required inputs

The following CSV files **must** exist under `data/input/` with the exact filenames below.

| Filename (exact)        | Description                                         | Expected grain (row meaning)                                                                 |
|-------------------------|-----------------------------------------------------|----------------------------------------------------------------------------------------------|
| `DATA_RAW.csv`          | Raw transactional channel-level data                | One row per \[date × channel × country × product ticker × segment\] observation             |
| `DATA_SUMMARY.csv`      | Time-series summary metrics                         | One row per \[date\] of firm-level aggregated summary metrics                               |
| `DATA_MAPPING.csv`      | Column/field mapping metadata                       | One row per mapping of a source field to a standardized/display field                       |
| `ETF.csv`               | ETF master + performance and characteristics table  | One row per ETF fund (ticker) with associated static and performance attributes             |
| `EXECUTIVE_SUMMARY.csv` | Narrative executive summary content for the dashboard | One row per narrative item (e.g. section/key driver text block)                           |

> Note: Filenames above are **part of the contract**. Any renaming requires a new versioned contract.

---

## C) Per-file schema contract

### C.1 `DATA_RAW.csv`

**Grain**

- **Row grain**: one row per combination of:
  - `Date`
  - `Channel`
  - `src_country`
  - `product_ticker`
  - `Segment`
  - `sub_segment`
  - `display_firm`
  - `product_country`

**Required columns**

| Column name (exact)        | Type    | Required? | Brief meaning                                                             |
|----------------------------|---------|----------|---------------------------------------------------------------------------|
| `Date`                     | date    | required | Calendar date of the observation                                         |
| `Channel`                  | text    | required | Distribution/sales channel name                                          |
| `src_country`              | text    | required | Country code or name of the source                                      |
| ` Asset Under Management ` | number  | required | Asset under management value for the row’s grain                         |
| `Net new business`         | number  | required | Net new business amount for the row’s grain                              |
| ` net new base fees `      | number  | required | Net new base fee amount for the row’s grain                              |
| `display_firm`             | text    | required | Firm identifier for display                                              |
| `product_country`          | text    | required | Product country code or name                                             |
| `Standard Channel`         | text    | required | Standardized channel classification                                      |
| ` best_of_source `         | number  | required | Best-of-source indicator/flag (numeric)                                  |
| `product_ticker`           | text    | required | Product/ETF ticker symbol                                                |
| `Segment`                  | text    | required | Segment classification                                                   |
| `sub_segment`              | text    | required | Sub-segment classification                                               |
| `uswa_sales_focus_2020`    | text    | required | Sales focus classification (label)                                      |
| `master_custodian_firm`    | text    | required | Master custodian firm identifier                                         |

**Optional columns**

- None defined at this time.  
  - Any new columns added here must be documented in this section as **optional** and must not change the row grain.

**Example row**

| Date       | Channel       | src_country   |  Asset Under Management  | Net new business |  net new base fees  | display_firm | product_country | Standard Channel |  best_of_source  | product_ticker | Segment      | sub_segment | uswa_sales_focus_2020 | master_custodian_firm |
|-----------|---------------|--------------|--------------------------|------------------|---------------------|--------------|-----------------|------------------|------------------|----------------|-------------|------------|------------------------|------------------------|
| 2019-12-31 | Broker Dealer | UNITED STATES | 16512.81                 | 30.89           | 0.02                | company_123  | US              | Broker Dealer    | 1                | AGG            | Fixed Income | Multi Sector | 2a Fixed Income: Core | company_999            |

> Example values above are illustrative only; they preserve the grain and type expectations.

---

### C.2 `DATA_SUMMARY.csv`

**Grain**

- **Row grain**: one row per `Date` (time-series summary of firm-level metrics).

**Required columns**

From the header row:

| Column name (exact)       | Type   | Required? | Brief meaning                                      |
|---------------------------|--------|----------|----------------------------------------------------|
| *(first/leading column)*  | text   | required | Unnamed/blank header column (used for date labels) |
| `Asset growth Rate`       | text   | required | Asset growth rate, typically a percentage string   |
| `Organic growth rate`     | text   | required | Organic growth rate (percentage string)            |
| `External growth rate`    | text   | required | External growth rate (percentage string)           |
| `Velocity`                | text   | required | Velocity metric (percentage or numeric string)     |
| `Standard Deviation`      | text   | nullable | Standard deviation metric                          |
| `VAR`                     | text   | nullable | Value-at-Risk or similar risk metric               |
| `Inflation`               | text   | nullable | Inflation impact metric                             |
| `Interest rates`          | text   | nullable | Interest rate impact metric                         |
| `Currency impact`         | text   | nullable | Currency impact metric                              |

**Optional columns**

- None beyond the header fields listed above at this time.

**TBD: confirm columns**

- Confirm the intended name/usage of the first/leading column whose header is currently empty.
- Confirm whether growth-rate and risk fields should be stored as:
  - numeric percentages (e.g. `-2.55`) or
  - formatted strings with `%` symbols (e.g. `-2.55%`).

**Example row**

| *(first/leading column)* | Asset growth Rate | Organic growth rate | External growth rate | Velocity | Standard Deviation | VAR | Inflation | Interest rates | Currency impact |
|--------------------------|-------------------|---------------------|----------------------|----------|--------------------|-----|-----------|----------------|-----------------|
| 2021-01-31              | -2.55%            | -1.91%              | -0.64%               | -0.10%   |                    |     |           |                |                 |

---

### C.3 `DATA_MAPPING.csv`

**Grain**

- **Row grain**: one row per mapping rule from a source field to a standardized/display field.

**Observed structure**

The file currently has:

- A first header row with empty column names: `,,, ,`
- Subsequent rows such as:
  - `channels_final_2022,asof_dt,Date,,`
  - `channels_final_2022,ibp_channel,Channel,,`
  - `channels_final_2022,src_country,src_country,,`
  - `channels_final_2022,aum,Asset under management,,`

Because the header row is empty, exact column names cannot be reliably inferred from the data alone.

**TBD: confirm columns**

The following contract elements **must be confirmed and then fixed here**:

- Name and order of each column (e.g. expected something like):
  - `source_table`
  - `source_column`
  - `standard_name`
  - `description`
  - `notes`
- Which columns are required vs nullable.
- Expected types (likely all text).

Until confirmed, treat **all existing columns** in `DATA_MAPPING.csv` as:

- Type: `text`
- Required: `required` (rows may still be skipped/ignored if incomplete, per downstream logic).

**Example row** (illustrative, based on observed data)

| (col1)             | (col2)       | (col3)                 | (col4) | (col5) |
|--------------------|--------------|------------------------|--------|--------|
| channels_final_2022 | asof_dt      | Date                   |        |        |

> Column labels `(col1)`…`(col5)` are placeholders until true header names are confirmed.

---

### C.4 `ETF.csv`

**Grain**

- **Row grain**: one row per ETF fund (ticker).

**Required columns**

The file contains a wide set of columns. At minimum, the following columns (from the first header row) are treated as **required** and must exist with exact names:

| Column name (exact)                         | Type   | Required? | Brief meaning                                      |
|---------------------------------------------|--------|----------|----------------------------------------------------|
| `Ticker`                                    | text   | required | ETF ticker symbol                                  |
| `Name`                                      | text   | required | ETF name                                          |
| `SEDOL`                                     | text   | nullable | SEDOL identifier                                   |
| `ISIN`                                      | text   | nullable | ISIN identifier                                    |
| `CUSIP`                                     | text   | nullable | CUSIP identifier                                   |
| `Incept. Date`                              | text   | required | Inception date (string; may be parsed as date)     |
| `Gross Expense Ratio (%)`                   | number | nullable | Gross expense ratio percentage                     |
| `Net Expense Ratio (%)`                     | number | nullable | Net expense ratio percentage                       |
| `Net Assets (USD)`                          | number | required | Net assets in USD                                  |
| `Net Assets as of`                          | text   | required | As-of date for net assets                          |
| `Asset Class`                               | text   | required | Asset class (e.g., Equity, Fixed Income)          |
| `Sub Asset Class`                           | text   | required | Sub-asset class                                    |
| `Region`                                    | text   | required | Region classification                              |
| `Market`                                    | text   | required | Market classification                              |
| `Location`                                  | text   | required | Location                                          |
| `Investment Style`                          | text   | required | Investment style (e.g., Index)                    |
| `Key Facts`                                 | text   | nullable | Key facts field                                   |
| *(all remaining return/yield/risk columns)* | text   | nullable | Time-series performance, yield, and ESG metrics    |

> Exact names of all remaining columns (performance, yield, fixed income characteristics, ESG, etc.) are part of the contract and must not be changed without versioning. They are omitted here for brevity but should be treated as **existing required columns** unless explicitly documented as optional later.

**Optional columns**

- None currently documented as optional.  
  - New columns may be added as **optional** provided:
    - They do not change the row grain.
    - They are appended and documented here as optional.

**TBD: confirm columns**

- Provide a complete, explicit list of all header names and mark which ones are:
  - required (used by the dashboard),
  - optional (not required for core visuals).
- Confirm exact types (number vs text) for:
  - all return/yield fields,
  - all risk/ESG fields.

**Example row**

| Ticker | Name                      | SEDOL | ISIN         | CUSIP    | Incept. Date | Gross Expense Ratio (%) | Net Expense Ratio (%) | Net Assets (USD) | Net Assets as of | Asset Class | Sub Asset Class | Region        | Market    | Location     | Investment Style | Key Facts | … (other columns unchanged) |
|--------|---------------------------|-------|--------------|----------|--------------|--------------------------|-----------------------|------------------|------------------|------------|-----------------|--------------|----------|-------------|------------------|-----------|-----------------------------|
| AGG    | Example Core US Bond ETF | -     | US0000000000 | 000000000 | May 15, 2000 | 0.04                     | 0.03                  | 82,344,435,311.26 | Aug 30, 2022     | Fixed Income | Multi Sectors   | North America | Developed | United States | Index            | 2.02      | …                           |

---

### C.5 `EXECUTIVE_SUMMARY.csv`

**Grain**

- **Row grain**: one row per narrative text block or key message for the executive summary dashboard.

**Observed structure**

Sample rows look like:

- `Firm Level Snapshot,"""Assets under management grew by (+ 0.89%) during the month of June. ...`,,
- `Firm Level Trend,"Year to date assets have grown (+ 3.69%) with an impressive organic growth rate of (+ 13.88) ...`,,
- `Key Drivers,"""Institutional drives 45% of Net New Business but only 28% of fees.""",,`

This suggests a structure like:

- Column 1: section or category (e.g. `Firm Level Snapshot`, `Firm Level Trend`, `Key Drivers`)
- Column 2: narrative text
- Columns 3–4: currently empty/unused

However, the header row is not clearly defined from the sample.

**TBD: confirm columns**

- Confirm exact header row and column names (e.g. expected something like):
  - `section`
  - `text`
  - `tag`
  - `notes`
- Confirm which columns are required and nullable.
- Confirm the maximum expected length and encoding for narrative text.

**Interim contract (until confirmed)**

- Treat all existing columns as:
  - Type: `text`
  - Required: `required` for column 1 and 2; nullable for any additional columns.

**Example row**

| (col1 - section)      | (col2 - text)                                                                                                                         | (col3) | (col4) |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------|--------|
| Firm Level Snapshot   | Assets under management grew by (+ 0.89%) during the month of June. Organic growth accounted for (+ 3.05%) but was offset by market… |        |        |

---

## D) Breaking changes

The following changes are **breaking** and must not be applied to an existing version of the contract:

- **Renamed file**
  - Changing any required filename (e.g. `DATA_RAW.csv` → `DATA_RAW_v2.csv`) without introducing a new versioned contract.
- **Renamed column**
  - Changing the spelling, spacing, or casing of any existing column name (including leading/trailing spaces).
- **Type changes**
  - Changing a column from:
    - text → number/date, or
    - number/date → text, or
    - any other incompatible format change that breaks existing consumers.
- **Grain changes**
  - Modifying what a row represents (e.g. changing `DATA_RAW.csv` from daily per-ticker to monthly per-ticker, or aggregating/de-duplicating rows).

Any of the above require:

- A new contract version identifier.
- Updated documentation in this file.

---

## E) Compatibility policy

- **Additive columns**
  - Adding **new columns** to existing files is allowed if:
    - The row grain remains unchanged.
    - New columns are appended (do not reorder existing columns if avoidable).
    - New columns are documented here (per file) as **optional** or **required**.

- **Deprecation approach**
  - To deprecate a column:
    1. Mark the column as **deprecated** in this document for **one full contract version** while still providing it in the CSV.
    2. Ensure downstream consumers are updated to stop using the deprecated column.
    3. In the **next** contract version:
       - Remove the deprecated column from the CSV.
       - Update this document to reflect its removal.

- **Backwards compatibility**
  - Within a single contract version:
    - Do **not** remove or rename existing required columns.
    - Do **not** change data types in a non-backward-compatible way.
  - New versions must clearly specify:
    - What changed.
    - Migration expectations for downstream consumers.

