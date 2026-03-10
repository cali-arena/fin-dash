# NLQ Architecture — Intelligence Desk

**Project:** AI infin8 | Institutional Asset Management Intelligence

This document describes the architecture layer for Natural Language Queries so the Intelligence Desk can interpret user questions and route them correctly. **The LLM never directly queries raw data.** Python classifies intent, extracts parameters, runs deterministic queries, and passes only verified results to the LLM.

---

## 1. Query types

| Type | Description | Who runs the query |
|------|-------------|--------------------|
| **Data Questions** | Internal portfolio data (AUM, NNB, fee yield, channel, segment, etc.) | Python only (deterministic query on processed dataset) |
| **Market Intelligence** | Macro, rates, sentiment, competitors, external conditions | External search + LLM; Python only passes question text |

---

## 2. Module overview

| Module | Role |
|--------|------|
| **intent_classifier** | Detect `data_question` vs `market_intelligence` (and `ambiguous`) using keyword signals. No LLM, no data access. |
| **parameter_extractor** | Extract ticker, channel, segment, country, date range, thresholds (e.g. above $100k, below 0.5%). Rule-based. |
| **query_executor** | Orchestrator: if data_question → run deterministic query and return aggregated table, metrics, chart-ready payload; if market_intelligence → return query payload for LLM (no raw data). |
| **response_formatter** | Build response object: `response_text`, `optional_table`, `optional_chart_data`. |

---

## 3. Example flow: handling a user question

### Step 1 — Classify intent

```python
from app.nlq.intent_classifier import classify_intent, is_data_question

query = "Which ETFs had high NNB but low fee yield in Q3?"
intent_result = classify_intent(query)
# intent_result.intent -> "data_question"
# intent_result.reason -> "Question references governed internal metrics or dimensions."

query2 = "What are current Fed rate expectations and how could they affect flows?"
intent_result2 = classify_intent(query2)
# intent_result2.intent -> "market_intelligence"
```

### Step 2 — Extract parameters (for data questions)

```python
from app.nlq.parameter_extractor import extract_parameters, extract_thresholds, extract_date_range

params = extract_parameters(
    "Which channels had NNB above $100k in Q3?",
    value_catalog={"channel": {"Wealth", "Institutional"}, "product_ticker": {"AGG", "HYG"}},
    today=date.today(),
)
# params.tickers, params.channel, params.segment, params.country
# params.date_start, params.date_end (Q3 range)
# params.thresholds -> [ThresholdSpec(op="gt", value=100_000, unit="currency")]
# params.to_filter_dict() -> for QuerySpec.filters
```

### Step 3 — Execute by intent

```python
from app.nlq.query_executor import run_intent, ExecutorResult

# Caller provides: metric_reg, dim_reg, value_catalog, df (processed dataset), allowlist
result = run_intent(
    "Which ETFs had high NNB but low fee yield in Q3?",
    prefer_data_mode=True,
    metric_reg=metric_reg,
    dim_reg=dim_reg,
    value_catalog=value_catalog,
    df=monthly_df,
    allowlist=allowlist,
    today=date.today(),
)

if result.error:
    # Show result.error to user
    pass
elif result.data_result:
    # Deterministic result: result.data_result.data (DataFrame), .metrics, .chart_spec
    # Pass only these to LLM for narrative (no raw dataset)
    pass
elif result.market_payload:
    # result.market_payload.query_text -> send to external search + LLM
    # No internal data in market_payload
    pass
```

### Step 4 — Format response

```python
from app.nlq.response_formatter import format_executor_result, FormattedNLQResponse

# After LLM returns narrative (for data path) or full answer (for market path):
formatted = format_executor_result(
    result,
    narrative_text=llm_narrative_from_verified_facts,  # data path
    market_response_text=llm_market_answer,             # market path
)
# formatted.response_text
# formatted.optional_table  -> DataFrame or None
# formatted.optional_chart_data -> dict with type, x, y, preview_rows, etc.
```

---

## 4. Data flow (no raw data to LLM)

```
User question
     |
     v
intent_classifier.classify_intent()  -->  data_question | market_intelligence | ambiguous
     |
     +-- market_intelligence --> query_executor returns MarketQueryPayload(query_text)
     |                              |
     |                              v
     |                         LLM layer: external search + answer from external sources only
     |
     +-- data_question --> parameter_extractor.extract_parameters()
     |                              |
     |                              v
     |                         parser.parse_nlq() -> QuerySpec (validated)
     |                              |
     |                              v
     |                         executor.execute_queryspec(QuerySpec, df, ...) -> QueryResult
     |                              |
     |                              v
     |                         DataQueryResult(data, metrics, chart_spec)  [verified only]
     |                              |
     |                              v
     |                         LLM receives: narrative from these facts only (no raw rows)
     |
     v
response_formatter.format_executor_result() -> FormattedNLQResponse(response_text, optional_table, optional_chart_data)
```

---

## 5. File locations

| File | Path |
|------|------|
| Intent classifier | `app/nlq/intent_classifier.py` |
| Parameter extractor | `app/nlq/parameter_extractor.py` |
| Query executor | `app/nlq/query_executor.py` |
| Response formatter | `app/nlq/response_formatter.py` |
| Existing parser | `app/nlq/parser.py` (QuerySpec, ParseError) |
| Existing executor | `app/nlq/executor.py` (execute_queryspec, QueryResult) |

---

## 6. Constraints enforced

- **No direct LLM access to raw dataset:** Only QuerySpec (structured) drives execution; results are aggregated table + metrics + chart spec. LLM receives only verified numbers/summaries for narrative.
- **All calculations Python-controlled:** execute_queryspec uses metric_reg, dim_reg, and allowlist; no user text concatenated into SQL.
- **Market intelligence:** Only the question text and optional context hint are passed to the LLM layer; no internal portfolio data.
