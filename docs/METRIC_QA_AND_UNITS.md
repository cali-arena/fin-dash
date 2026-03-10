# Metric QA and unit expectations

**Project:** AI infin8 | Institutional Asset Management Intelligence

This document explains how NNB and NNF are calculated and formatted, the root cause of apparent scale differences, and the guardrails in place to avoid misleading figures.

---

## 1. Root cause of NNF vs NNB scale difference

**Observed pattern:** NNB can display in **$M** (e.g. $455.95M) while NNF displays in **$K** (e.g. $481.65K). That is a ~1000× magnitude difference.

**Trace (deterministic):**

- **Source:** Both NNB and NNF come from the same pipeline:
  - **Data:** `v_firm_monthly` (or `data/agg/firm_monthly.parquet`) with columns `nnb`, `nnf` (or aliases: NNB from `net_new_business`, `net_flow`, etc.; NNF from `nnf`, `net_net_flow`, `net_fee_flow`, `fee_flow`, `fees`).
  - **Normalization:** `app.data_gateway` maps aliases only; **no scaling** (no ×1000 or ÷1e6) is applied to either column.
  - **Aggregation:** For the selected period and scope, both are **summed** by `month_end` (and by dimension when applicable). Same grain and same aggregation.
  - **Formatting:** Both are passed through **the same formatter** `app.ui.formatters.fmt_currency(..., unit="auto")`, which chooses $K / $M / $B from the **numeric value** only.

So the dashboard does **not** introduce a unit or aggregation mismatch. The magnitude difference comes from one of:

1. **Source data:** NNB and NNF are stored in different units (e.g. NNB in dollars, NNF in thousands), or one is pre-scaled in the view/ETL.
2. **Semantics:** NNF is used as **fee revenue** (net new fees), which is typically much smaller than **flow** (NNB). In that case, NNF in $K and NNB in $M can be correct; the product then shows a compact **metric note** so users know to interpret NNF as fee revenue.

**Conclusion:** There is no in-app scaling bug. If the source uses the same unit for both, the ETL or view should be corrected so both are in the same currency unit. If NNF is intentionally fee revenue, the displayed scale is expected and the note clarifies it.

---

## 2. Calculation paths (end-to-end)

| Metric | Formula / path | Where |
|--------|----------------|--------|
| **NNB** | From row/sum of `nnb` (alias from `net_flow`, `net_new_business`, or derived from subscriptions−redemptions / end−begin−market_impact when missing). | `data_gateway` → `groupby().sum()` → `compute_kpi` latest row. |
| **NNF** | From row/sum of `nnf` (alias from `nnf`, `net_net_flow`, `net_fee_flow`, `fee_flow`, `fees`). No scaling. | Same path as NNB. |
| **Fee yield** | `compute_fee_yield(nnf, begin_aum, end_aum, nnb=nnb)` = NNF / avg_aum, with avg_aum = (begin_aum + end_aum)/2; optional annualization from contract. When NNB ≤ 0, returns NaN. | `app.metrics.metric_contract`; used in snapshot, report, Tab 1. |
| **OGR** | `compute_ogr(nnb, begin_aum)` = NNB / begin_aum. NaN when begin_aum missing or ≤ 0. | Same. |
| **Market impact** | `compute_market_impact(begin_aum, end_aum, nnb)` = end_aum − begin_aum − NNB. | Same. |

All monetary values are treated as being in the **same currency unit** (e.g. dollars). Fee yield and OGR are dimensionless ratios.

---

## 3. Expected units (data contract)

- **begin_aum, end_aum, nnb, nnf, market_impact_abs:** Same currency unit (e.g. USD). No in-app conversion or scaling.
- **Rates (ogr, market_impact_rate, fee_yield):** Dimensionless (decimal); display as % when needed.
- If the **source** stores NNF in thousands (or NNB in millions), the **ETL or view** should normalize to one unit so the dashboard receives a consistent scale. The dashboard does not re-scale.

---

## 4. QA guardrails (automatic checks)

Implemented in **`app.metrics.qa_guardrails`** and used in **`app.kpi.service`** and (in dev) in the KPI parity expander:

1. **AUM reconciliation:** `begin_aum + NNB + market_movement = end_aum` within a small tolerance (default 1.0 currency unit). Variance is stored and, when large, a warning is added and the UI can show “AUM consistency variance”.
2. **Fee yield consistency:** Implied fee yield from NNF/avg_aum is compared to any precomputed `fee_yield` on the row; if they differ beyond tolerance, a warning is added.
3. **NNB/NNF magnitude ratio:** If |NNB|/|NNF| or |NNF|/|NNB| exceeds 100, `unit_consistency` is set to `"possible_mismatch"` and a **metric note** is shown in the UI (see below).
4. **Single formatter:** All monetary KPIs and charts use `app.ui.formatters.fmt_currency(..., unit="auto")` ($K/$M/$B from value). No raw long integers; same formatter for cards, tables, and charts.

---

## 5. UI behaviour

- **When reconciliation is within tolerance:** Caption: “AUM consistency: Begin AUM + NNB + Market = End AUM (selected slice; reconciled).”
- **When reconciliation variance &gt; 1.0:** Caption shows the variance and “Verify source aggregation.”
- **When NNF/NNB ratio &gt; 100×:** A compact **metric note** (caption) explains that NNF is in lower magnitude than NNB and that either source scaling should be verified or NNF is correctly fee revenue.
- **Dev mode:** “KPI parity check” expander shows validation and the result of **`run_metric_qa(kpi_result)`** (reconciliation, fee yield, NNB/NNF ratio) for debugging. No noisy internals in the main UI.

---

## 6. Files touched for this work

| File | Role |
|------|--------|
| **app/metrics/qa_guardrails.py** | New: `check_aum_reconciliation`, `check_fee_yield_consistency`, `check_nnb_nnf_magnitude_ratio`, `run_metric_qa`. |
| **app/metrics/__init__.py** | Export QA guardrail functions. |
| **app/kpi/service.py** | Fee yield implied from NNF/avg_aum; fee_yield_consistency vs row; reconciliation_variance; warn when |recon| &gt; 1.0. |
| **app/pages/visualisations.py** | Professional metric note (caption) for possible_mismatch; reconciliation caption when variance &gt; 1; dev expander calls `run_metric_qa`. |
| **docs/METRIC_QA_AND_UNITS.md** | This document. |

---

## 7. Before / after summary

| Aspect | Before | After |
|--------|--------|--------|
| NNF vs NNB scale | Suspected bug; no clear explanation. | Trace documented; cause is source/ETL or semantics (fee revenue); no in-app scaling. |
| AUM reconciliation | Check at 1e-6; warning on any drift. | Tolerance 1.0; variance stored; caption only when &gt; 1.0. |
| Fee yield | Not cross-checked. | Implied fee yield computed and compared to row; warning on mismatch. |
| Unit mismatch | Single long warning. | Short caption (metric note) and dev-only QA block. |
| Formatting | Same formatter. | Unchanged; confirmed single `fmt_currency` for all monetary KPIs/charts. |
| Debugging | Validation dict only. | Dev expander runs `run_metric_qa` and shows checks/messages. |

Validation remains deterministic; KPI cards and downstream charts are unchanged except for the added caption and dev-only QA section.
