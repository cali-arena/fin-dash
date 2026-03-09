# Demo scenarios — deterministic replay

Use this document to run the same demo every time. Numbers may change if the dataset changes; focus on **expected structure**, **expected top entities** where possible, and **export actions**.

---

## 1) Demo preconditions

- **dataset_version:** Use the version shown in the sidebar (from `curated/metrics_monthly.meta.json`). For a stable demo, pin to a known version (e.g. from a tagged release or a fixed ETL run).
- **Recommended date range:** Last 12 months (or a fixed window, e.g. `2024-01-01` → `2024-12-31`) so that charts and ranked tables are populated and comparable across runs.
- **Segment / filters:** Use default segment or a single slice if needed; avoid “All” across many dimensions if it leads to empty or very large result sets.
- **Note:** Absolute numbers may change when the dataset is refreshed. Define **expected structure** (sections, columns, chart types) and **expected top entities** (e.g. “top channel by NNB is one of [A, B, C]”) where possible.

---

## 2) Tab 1 — Visualisations (2–3 scenarios)

### Scenario 1: Firm snapshot view

**Steps:**
1. Open **Tab 1 (Visualisations)**.
2. Set **date range** (e.g. last 12 months).
3. Set **metric toggles** (e.g. NNB vs AUM for ranked tables) and path selector if present.
4. Leave drill at firm level (no channel/ticker selected).

**Expected:**
- **Waterfall** chart renders (AUM/flow breakdown).
- **Treemap** shows top channels (by selected metric).
- **Ranked tables:** “Top Channels” and “Top Tickers” tabs are populated; columns include rank, channel/ticker, NNB, AUM (or as per UI).
- **Details panel** shows firm-level KPIs (AUM, NNB, OGR, etc.) and a breakdown table (e.g. Top Channels or Top Tickers).

**Exports:**
- Click **Download CSV (current view)** under **Top Channels**. File name pattern: `tab1_top_channels__rows-N__YYYY-MM-DDTHH-MM.csv`. CSV has expected columns (e.g. rank, channel, NNB, AUM).

---

### Scenario 2: Drill to a specific channel

**Steps:**
1. Stay on **Tab 1** with same date range as Scenario 1.
2. Set **drill mode** to **Channel** (if applicable).
3. From the **Top Channels** ranked table, note the **top channel** by NNB (or AUM). Select it (e.g. via click-to-select on chart or selectbox “Selected: Channel = …”).
4. Confirm selection is reflected in the drill state (caption or selectbox shows the channel name).

**Expected:**
- **Details panel** header updates to “Details — Channel: &lt;name&gt;”.
- **KPIs** (AUM, NNB, OGR, Market Impact Rate, Fee Yield) update to the selected channel.
- **Breakdown** title becomes “Top Tickers inside Channel”; table shows top tickers within that channel.
- Mini trend (OGR & Market Impact Rate) and monthly totals reflect the drilled slice.

**Exports:**
- Optional: **Download CSV (current view)** for the breakdown table (`details_breakdown__rows-N__….csv`).

---

### Scenario 3 (optional): Drill to ticker

**Steps:**
1. From Scenario 2, in the **Top Tickers inside Channel** table, select the **first ticker** (or use Top Tickers tab and select a ticker).
2. Confirm “Details — Ticker: &lt;name&gt;” and optional “Geo split” / “Channel split” if available.

**Expected:**
- Details KPIs and breakdown reflect the selected ticker (geo or channel split as configured).

---

## 3) Tab 2 — Dynamic Report (1–2 scenarios)

### Scenario: Generate deterministic report

**Steps:**
1. Open **Tab 2 (Dynamic Report)**.
2. Use the same **date range** (and segment) as in Tab 1 so the report pack is consistent.
3. Wait for the report to load (single `get_report_pack` call).

**Expected:**
- **Sections appear in fixed order:** Overview → Channels → Products/ETF → Geo → Anomalies → Recommendations.
- Each section has **2–5 bullets** (or “No movers…” / “No anomalies…” when empty) and, where data exists, a **table** (movers / top-bottom).
- **Report metadata** shows dataset version and filter hash; reconciliation table may appear at the bottom.

**Exports:**
- **Download HTML:** click **Download HTML**; file name pattern `report_&lt;dataset_version&gt;_&lt;filter_hash&gt;.html`.
- **Export “Channel movers” CSV:** In the **Channels** section, click **Download CSV (current view)**. File name pattern: `tab2_channel_movers__rows-N__YYYY-MM-DDTHH-MM.csv`. Confirm CSV has the expected mover columns.

---

## 4) Tab 3 — NLQ Chat (3–4 questions)

Each question: enter in “Ask a question”, click **Run**. Check **Result** (or **Answer**), **Table**, and **Debug (dev)** expander for QuerySpec JSON and validation logs.

---

### Q1: “NNB by channel last 12 months”

- **Expected QuerySpec shape:** `metric_id` = nnb (or equivalent); `dimensions` = [channel]; time range = last 12 months; limit ≤ 20 (or governed default).
- **Expected chart type:** Bar (x = channel, y = metric) or table.
- **Expected structure:** Table columns include channel and metric; rows ≤ limit.

---

### Q2: “Trend of OGR over time YTD”

- **Expected QuerySpec shape:** `metric_id` = ogr; trend / time series (month_end); chart type = line; x = month_end, y = metric.
- **Expected chart type:** Line (x = month_end, y = metric).
- **Expected structure:** Table has month_end and metric; multiple rows for YTD months.

---

### Q3: “Top tickers by NNB current month”

- **Expected QuerySpec shape:** `metric_id` = nnb; `dimensions` = [product_ticker] (or ticker); snapshot intent (current month); limit ≤ 20.
- **Expected chart type:** Bar or table.
- **Expected structure:** Table columns include ticker and metric; rows ≤ limit.

---

### Q4: “Market impact rate by country since 2025-01”

- **Expected QuerySpec shape:** `metric_id` = market_impact_rate; `dimensions` = [src_country] (or country); time_range start = 2025-01; chart bar.
- **Expected chart type:** Bar (x = country dimension, y = metric).
- **Expected structure:** Table has country and metric; time filter applied.

---

## 5) Pass/Fail checklist

Use this after running all scenarios to decide pass/fail.

| Check | Pass criteria |
|-------|----------------|
| **No timeouts** | No “query timeout” or “over_budget” message for normal demo filters; NLQ and report complete without timeout state. |
| **No >5k rows rendered** | Tables and exports stay within governed caps (current view ≤ 5k; full export only when Export mode on, ≤ 50k). No unbounded table render. |
| **Exports work** | Tab 1: Top Channels CSV downloads. Tab 2: HTML and Channel movers CSV download. Tab 3: NLQ result CSV (current view) downloads. Filenames follow `{base_filename}__rows-N__{timestamp}.csv` or report HTML pattern. |
| **Debug expander** | Tab 3: “Debug (dev)” shows **QuerySpec** JSON and **validation logs**; QuerySpec matches intent (metric_id, dimensions, time_range, limit). |
| **Observability** | With DEV_MODE=1: Observability panel shows **perf_query_log**; after rerun or same filters, **cache hits** appear where applicable; row caps and elapsed_ms present. |

**Overall:** Pass = all checks met for the chosen scenarios; Fail = any check fails or any scenario errors (parse error, validation error, missing section, or missing export).

---

*Document version: 1.0. Adjust date ranges and “expected top entities” to match your dataset.*
