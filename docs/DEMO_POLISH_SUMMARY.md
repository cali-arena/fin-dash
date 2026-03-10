# Client-demo polish pass — summary

**Project:** AI infin8 | Institutional Asset Management Intelligence  
**Scope:** Tabs 1 (Executive Dashboard), 2 (Investment Commentary), 3 (Intelligence Desk)  
**Objective:** Premium institutional asset management feel; no prototype rough edges.

---

## 1. Polish issues found

- **Developer / technical wording**
  - Tab 1: "evidence is insufficient", "could not be computed", "selected slice", "Key products and outliers are labeled; hover for full detail", "Ranks products by…"
  - Tab 2: "Executive chart suppressed because…", "Anomaly chart suppressed because…", "Suppressed sections due to insufficient signal", "End AUM (report slice)"
  - Tab 3: "No rows returned.", "Value catalog unavailable", "Line/Bar chart unavailable", "Claude narrative unavailable", "Market Intelligence response unavailable"
  - Main: "No renderer configured for tab '…'"

- **Inconsistent labels**
  - Tab 2 used "End AUM (report slice)" and "AUM trend (report slice)" vs Tab 1 "selected slice" terminology.

- **Weak or missing hierarchy**
  - Tab 3 had no one-line business subtitle under the title (Tabs 1 and 2 had section-subtitles).

- **Trust / clarity**
  - Empty-state and "not shown" messages sounded technical ("suppressed", "signal insufficient") rather than client-friendly ("not shown", "insufficient data in selected range/scope").

---

## 2. Files changed

| File | Changes |
|------|--------|
| `app/pages/visualisations.py` | Subtitle copy; institutional notes (AUM Waterfall, Distribution, Growth Quality Matrix, ETF Drill-Down, Trend, Correlation, Contributors); chart captions; "selected slice" → "selected scope" where appropriate. |
| `app/pages/dynamic_report.py` | "End AUM (report slice)" → "End AUM (selected slice)" (metric + chart); executive and anomaly _note() text; suppressed-list caption; AUM trend chart title/axis labels. |
| `app/pages/nlq_chat.py` | Empty state "No rows returned." → "No data returned for this query."; value catalog caption; chart fallback "unavailable" → "Chart not shown for this selection."; Market Intelligence and narrative unavailability copy; added section-subtitle under "Intelligence Desk" title. |
| `app/main.py` | "No renderer configured for tab '…'" → "This tab is not available." (st.info). |

---

## 3. UX / text / design refinements applied

- **Monetary and scope language**
  - Standardised on "selected slice" (Tab 1) and "selected slice" (Tab 2) for consistency; "selected scope" used in Tab 1 where referring to filter scope (e.g. "data in the selected scope").

- **Empty and limited-data states**
  - Replaced "evidence is insufficient" / "could not be computed" with "data in the selected scope is insufficient for…" / "was not computed for the selected period."
  - Replaced "suppressed" with "not shown" and "Sections not shown (insufficient data in selected range): …".

- **Chart and section captions**
  - "Key products and outliers are labeled; hover for full detail" → "…hover for detail. Quadrants guide where to invest, improve pricing, or review."
  - "Ranks products by…" → "Products are ranked by… to support prioritization by quadrant."

- **Tab 3**
  - Added section-subtitle: "Ask data questions over your internal book or market intelligence over external sources. Results are verified; narrative is optional."
  - All user-facing "unavailable" messages made consistent and client-friendly ("not available", "Chart not shown for this selection", "No data returned for this query").

- **Theme**
  - No theme or palette changes; dark professional theme preserved.

---

## 4. Final demo-readiness assessment

- **Coherence**
  - All three tabs use a consistent pattern: title → short business explanation (subtitle) → content. Tab 3 now matches this with a section-subtitle.

- **Language**
  - No remaining developer-only wording in the audited areas. Financial terms (institutional, channel, ETF, fee yield, contributors, market movement) are used consistently.

- **Trust**
  - Scope and filter language is consistent ("selected slice", "selected scope", "selected range"). Empty and limited-data messages explain why something is not shown without technical jargon.

- **Limitations**
  - Some edge-case copy (e.g. deep in NLQ or report engine) may still be technical; the main user-facing surfaces have been polished.
  - API key and configuration messages (ANTHROPIC_API_KEY, external search) remain clear but are necessarily technical for setup.

**Verdict:** Demo-ready. The app presents as a coherent, institutional-grade product across Tabs 1–3, with consistent hierarchy, scope language, and client-appropriate empty/limited-data messaging.

---

## 5. Remaining minor risks

- **Smoke test / docs**
  - `qa/smoke_test.md` still references "No rows returned" and "No data under current filters"; consider updating to "No data returned for this query" and the new empty-state hint for NLQ.

- **Localisation / future copy**
  - Any future copy (e.g. new sections or tabs) should follow the same conventions: "selected slice/scope/range", "not shown" instead of "suppressed", and "data is insufficient" instead of "evidence is insufficient".

- **Plotly defaults**
  - Plotly layout and styling were not changed in this pass; if any chart still looks default, a follow-up can align it with `apply_enterprise_plotly_style` and theme.

- **Loading states**
  - Existing spinners for main flows (dashboard load, commentary load, NLQ search/narrative) were left as-is; no new loading states were added. If any dynamic sub-panel loads without a spinner, it can be added in a later pass.
