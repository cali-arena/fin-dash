# Tab 2: Deterministic Python NLG Report — Implementation Summary

**Project:** AI infin8 | Institutional Asset Management Intelligence  
**Tab:** Investment Commentary (Dynamic Report)  
**Constraint:** No LLM/Claude in Tab 2; all narrative from Python templates only.

---

## 1. Files changed

| File | Change |
|------|--------|
| **app/data_gateway.py** | Added `fee_yield_prior` to firm snapshot (prior month fee yield). Added firm-level **fee yield deterioration** anomaly when current fee yield drops vs prior month (≥5 bps). |
| **app/reporting/nlg_templates.py** | Executive headline includes NNF when available. Recommendations capped at 5 bullets. New public API: `generate_executive_overview`, `generate_channel_analysis`, `generate_product_analysis`, `generate_geographic_analysis`, `generate_anomalies`, `generate_recommendations`. |
| **app/pages/dynamic_report.py** | Executive chart note text updated to client wording; AUM trend chart title/axis use "selected slice". |
| **docs/TAB2_NLG_SPEC.md** | This summary and section mapping. |

**Unchanged (already compliant):**  
- **app/reporting/report_engine.py** — Section renderers (`render_overview`, `render_channel_commentary`, `render_product_commentary`, `render_geo_commentary`, `render_anomalies`, `render_recommendations`) already use only `nlg_templates.select_*` and rules; no LLM.  
- **app/reporting/report_pack.py**, **app/reporting/rules.py** — No changes.  
- **app/ui/formatters.py** — Used for $K/$M/$B, %, bps; no changes.

---

## 2. New Python report-generation functions

### In **app/reporting/nlg_templates.py** (template layer)

- **`generate_executive_overview(metrics)`** — Bullets from snapshot dict (AUM, MoM, YTD, NNB, NNF when present, OGR; growth/quality/source/changed templates). Uses `app.ui.formatters` for currency/percent.
- **`generate_channel_analysis(channel_df, comparisons)`** — NLG layer for channel: concentration, share gain/loss from rank table and snapshot. (Full section = report_engine `render_channel_commentary(pack)` which adds top/bottom + mix-shift rules then calls this.)
- **`generate_product_analysis(product_df, flags)`** — NLG layer for product: concentration, mix shift. `flags` reserved for future pricing/underutilized flags.
- **`generate_geographic_analysis(geo_df)`** — NLG layer for geography: share gain/loss, concentration.
- **`generate_anomalies(anomaly_df)`** — Bullets from anomaly table (count, high/medium severity, reversals).
- **`generate_recommendations(context)`** — 3–5 recommendation bullets from pack (or (pack, snap)); context = `ReportPack` or `(pack, snap)`.

### In **app/reporting/report_engine.py** (section output)

- **`render_overview(pack)`** → `SectionOutput` with bullets from `select_executive_overview(snap, …)` and optional Top Movers table.
- **`render_channel_commentary(pack)`** → Top/bottom channel by NNB, mix shift, then `select_channel_commentary(bullets, channel_rank, snap, …)`.
- **`render_product_commentary(pack)`** → Top/bottom product, mix shift, optional ETF mover, then `select_product_commentary(bullets, ticker_rank, …)`.
- **`render_geo_commentary(pack)`** → Top/bottom geography, mix shift, then `select_geo_commentary(bullets, geo_rank, …)`.
- **`render_anomalies(pack)`** → `select_anomaly_bullets(anomalies, …)` and table.
- **`render_recommendations(pack)`** → `select_recommendations(pack, snap, …)` (3–5 bullets).

---

## 3. Streamlit/UI integration (Tab 2)

- **Entry:** `app/pages/dynamic_report.py` → `render(state, contract)`.
- **Data:** `_get_gateway().get_report_pack(filters)` returns `ReportPack` (firm_snapshot, time_series, channel_rank, ticker_rank, geo_rank, etf_rank, anomalies).
- **Sections:** Each section calls the corresponding `report_engine.render_*` and displays bullets (and optional charts/tables). No LLM or API calls.
- **Export:** Markdown, HTML, PDF (if available) built from the same `SectionOutput` bullets and meta.
- **Hierarchy:** Title/subtitle → reporting window/filters → KPI strip (Executive) → section headers + subtitles → bullets → charts/tables → export expander.

---

## 4. Helper formatting and anomaly utilities

- **Formatting:** All narrative uses `app.ui.formatters`: `fmt_currency` ($K/$M/$B), `fmt_percent` (%, optional signed), `fmt_number` (e.g. z-scores). Report engine uses internal `_fmt_money`, `_fmt_pct`, `_fmt_num` that delegate to these.
- **Anomalies (data_gateway):**
  - Existing: firm-level z-score (NNB, AUM_CHANGE, MARKET_IMPACT) over 12m rolling; dimension-level cross-sectional z and NNB **sign reversals** (channel, ticker, geo).
  - **New:** Firm-level **fee yield deterioration** when current month fee yield drops by ≥5 bps vs prior month (only if `time_series` has `fee_yield`).
- **NLG anomaly wording:** `nlg_templates.ANOMALY_TEMPLATES` and `select_anomaly_bullets` handle none/count/high/medium/reversal; no change to column contract (level, entity, metric, value_current, baseline, zscore, rule_id, reason, severity, month_end).

---

## 5. Section-to-specification mapping

| Client requirement | Section | Implementation |
|--------------------|---------|----------------|
| **1. Are we growing?** | Executive Overview | End AUM, NNB, NNF (when available), OGR, MoM/YTD; growth templates (strong/weak flows, tailwind/headwind). |
| **2. Is growth good (profitable, sustainable)?** | Executive Overview | Fee yield level and fee_yield_prior comparison; quality templates (fee improving/deteriorating/stable, high NNB + fee yield). |
| **3. Where is it coming from?** | Executive + Channel + Product + Geo | Source templates (NNB vs market dominant/mixed); top/bottom channels/products/geographies by NNB; mix shift and concentration. |
| **4. What has changed vs last period?** | Executive Overview | What-changed templates (strong flow/mkt down, weak flow/mkt up, both positive/negative); MoM/YTD; fee_yield_prior. |
| **5. What should we do next?** | Recommendations | 3–5 actions from anomalies, mix shift, OGR, market negative, fee deterioration, concentration. |
| Top/bottom channels by NNB; fee yield by channel; concentration | Channel Analysis | `render_channel_commentary`: rules top/bottom + mix shift; NLG concentration/share gain/loss. Rank table has aum_end, nnb, fee_yield, aum_share_delta, nnb_share_delta. |
| Top/bottom products by NNB/NNF; concentration; ETF | Product & ETF Analysis | Same pattern; optional ETF mover from etf_rank. |
| NNB by country/region; concentration; underperforming geographies | Geographic Analysis | Top/bottom geo by NNB; share gain/loss; concentration. |
| Rolling z-score; fee yield deterioration; inflection; sign reversals; market divergence | Anomalies | Firm z-score (NNB, AUM_CHANGE, MARKET_IMPACT); fee_yield_deterioration rule; dimension reversals; severity. |
| 3–5 recommendations from flags/anomalies/concentrations | Recommendations | `select_recommendations`: strong flows + negative market, weak flows, fee deteriorating, mix-shift allocation, investigate high/reversal, concentration diversify; cap 5. |

---

## Validation checklist

- [x] Tab 2 renders with no LLM/API dependency (no Claude/Anthropic/OpenAI in dynamic_report, report_engine, nlg_templates).
- [x] Changing filters updates the report deterministically (get_report_pack is filter-driven; all sections use pack only).
- [x] All six sections appear (Executive, Channel, Product & ETF, Geographic, Anomalies, Recommendations).
- [x] Sections handle missing data (empty rank/snapshot/anomalies yield fallback bullets and no crash).
- [x] Formatting is consistent ($K/$M/$B, %, bps via formatters).
- [x] No placeholder or lorem ipsum (all text from templates and metrics).
- [x] No developer wording in user-facing copy (financial/executive language in nlg_templates and dynamic_report notes).
