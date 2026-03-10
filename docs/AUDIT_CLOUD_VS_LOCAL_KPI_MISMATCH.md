# Audit: Streamlit Cloud vs Local KPI Mismatch

**Objective:** Identify why Cloud deployment shows different KPI values (NNB, Organic Growth, Market Movement) from localhost while End AUM matches.

**Conclusion:** The most likely root cause is **default filter state (date range) differing** between environments, leading to different “latest” month or different underlying rows. A secondary cause is **same “latest” row but different derived metrics** if `begin_aum` or aggregation order differs. This document traces the full execution path and lists every file and risky logic.

---

## 1. Root-cause summary

| Hypothesis | Evidence | Type |
|------------|----------|------|
| **Default date range differs (Cloud vs local)** | `resolve_best_default_filters()` is data-dependent (`list_month_ends`); fallback uses `date.today()` (server time). Cloud can get different `(date_start, date_end)` → different filter hash → different query result → different “latest” row or different NNB/OGR/Market. | **State mismatch** |
| **Same End AUM, different NNB/OGR/Market from same row** | End AUM = `latest["end_aum"]`; NNB = `latest["nnb"]`; Market = `compute_market_impact(begin_aum, end_aum, nnb)`; OGR = `compute_ogr(nnb, begin_aum)`. If `begin_aum` is wrong (e.g. from `shift(1)` with different row order or missing prior month), End AUM can match while NNB/Market/OGR differ. | **Formula / data mismatch** |
| **Different dataset or dataset_version** | `dataset_version` from filesystem (DuckDB meta or `data/curated/metrics_monthly.meta.json`). Different version → different cache keys; Cloud could serve different or stale data. | **Data mismatch** |
| **Stale or cross-session cache** | `st.cache_data` key = `(dataset_version, filter_hash, query_name)`. If Cloud reuses cache from a previous deploy or session with different filters, one query could return stale data. | **Cache mismatch** |

**Why End AUM can match but NNB/OGR/Market differ**

- **Scenario A (different “latest” month):** If by coincidence the **last** month in the Cloud result set has the same `end_aum` as the last month on local (e.g. same Dec value, but Cloud is showing Nov), then displayed End AUM can match while NNB/Market/OGR are from a different month → **different values**. (Uncommon unless data is very similar across months.)
- **Scenario B (same row, wrong begin_aum):** Same `latest` row is used in both environments. `end_aum` and `nnb` come directly from that row; `market_impact` and `ogr` are computed from `begin_aum` (and `nnb`/`end_aum`). If `begin_aum` is derived via `monthly["end_aum"].shift(1)` and the **order or presence of months** differs (e.g. missing prior month, or groupby/sort non-determinism), then `begin_aum` can differ while `end_aum` and `nnb` in that row are correct. Result: **End AUM and NNB match; Market and OGR differ** (formula mismatch driven by wrong `begin_aum`).
- **Scenario C (different scope/slice):** If Cloud and local use different `source_for_monthly` (e.g. one uses `firm_period`, the other uses a scoped frame due to `tab1_filter_*`), then the “latest” row can be from different aggregations. Unlikely to preserve identical End AUM unless slice and firm totals align by chance.

---

## 2. Execution path (concise)

### 2.1 Dataset loading (where data comes from)

| Step | File | What happens |
|------|------|----------------|
| 1 | `app/main.py` | `render_global_filters()` → `get_filter_state()`; then tab `render(state)`. |
| 2 | `app/ui/filters.py` | First run: `resolve_best_default_filters(root)` → `set_filter_state(best_state)`. Later: date picker updates FilterState. |
| 3 | `app/state.py` | `resolve_best_default_filters()` uses `DataGateway(root).list_month_ends(base, view_name="v_firm_monthly")` to get available month_ends; tries candidate windows; falls back to `FilterState.from_dict({})` (which uses `_default_date_start()` / `_default_date_end()` from **date.today()**) if no candidates have rows, or to `(month_ends[0], month_ends[-1])`. |
| 4 | `app/state.py` | `_default_date_end()` / `_default_date_start()` use `date.today()` → **environment-dependent** (server timezone on Cloud). |
| 5 | `app/pages/visualisations.py` | `gateway.run_query("firm_monthly", state)` (and channel_monthly, ticker_monthly, geo_monthly, segment_monthly). `state` = global FilterState (date_start, date_end, …). |
| 6 | `app/data_gateway.py` | `run_query("firm_monthly", state)` → not in `RUN_QUERY_ALLOWED` → `filter_state = filter_state_to_gateway_dict(state)` (only `month_end_range`), then `cache_pyramid.get_filtered(dv, query_name, h, filter_state_json, root_str)`. |
| 7 | `app/state.py` | `filter_state_to_gateway_dict(state)` returns `{"month_end_range": (pd.Timestamp(state.date_start), pd.Timestamp(state.date_end))}`. |
| 8 | `app/data_gateway.py` | `hash_filters(filter_state)` hashes that dict → cache key includes **date range only** for this path. |
| 9 | `app/cache/pyramid.py` | `get_filtered` → `_level_a_impl` → `_run_query_uncached(query_name, filter_state, root)`. |
| 10 | `app/data_gateway.py` | `_run_query_uncached` uses `build_where(filter_state)` (month_end >= start, month_end <= end), then DuckDB view or Parquet. |

**Files that load or decide the data range**

- `app/ui/filters.py` (lines 21–27) – init and default FilterState.
- `app/state.py` (lines 302–316, 398–431, 447–469, 587–596, 699–731) – default dates, FilterState.from_dict, resolve_best_default_filters, filter_state_to_gateway_dict.
- `app/pages/visualisations.py` (lines 1651–1657) – calls `run_query(..., state)`.
- `app/data_gateway.py` (lines 622–701, 1654–1677, 1565–1613) – get_config (dataset_version), run_query, hash_filters, _run_query_uncached, build_where.

### 2.2 Cache usage (every st.cache_data / st.cache_resource)

| File | Usage | Key / effect |
|------|--------|----------------|
| `app/data_gateway.py` | `st.cache_data(ttl=3600)(_run_query_templated_impl)` (line 1383) | Governed path: dataset_version + filter_hash + query_name. |
| `app/data_gateway.py` | `@st.cache_data(show_spinner=False)` (lines 1630, 2748, 2809, 2822, 2835, 2848, 2860, 2871), `cached_run_query` (1630) | Pyramid path: dataset_version + filter_state_hash + query_name (filter_state = month_end_range only). |
| `app/data_gateway.py` | `st.cache_resource` (lines 900, 897) | DuckDB connection. |
| `app/cache/pyramid.py` | `@st.cache_data(..., ttl=...)` get_filtered_fast/medium/heavy (76, 86, 96) | Level A: dataset_version, filter_state_hash, query_name, filter_state_json, root_str. |
| `app/cache/cache_gateway.py` | `@st.cache_data`, `@st.cache_resource` (48, 73) | Policy and cached_query. |
| `app/queries/firm_snapshot.py` | `@st.cache_data(show_spinner=False)` (623) | dataset_version, filter_hash, etc. |
| `app/agg_store.py` | `@st.cache_data(ttl=3600)` (41, 132) | dataset_version, table_name, columns. |
| `app/duckdb_store.py` | `@st.cache_data(ttl=3600)` (90) | dataset_version, sql, params. |

**Risky:** If Cloud and local differ on `dataset_version` or `filter_state_hash`, they never share cache. If Cloud reuses an old session’s cache (same key) after a deploy or data change, it can show stale NNB/OGR/Market while End AUM might still match by chance.

### 2.3 Session state and filter defaults

| Location | Key / default | Risk |
|----------|----------------|------|
| `app/state.py` | `FilterState.from_dict({})`: `date_start` = `_default_date_start()`, `date_end` = `_default_date_end()`, `period_mode` = `"1M"` | Date defaults use **date.today()** → server time on Cloud. |
| `app/state.py` | `resolve_best_default_filters()`: uses `list_month_ends()`; if no candidates have rows, returns `base` = `FilterState.from_dict({})` with date.today() defaults | When Cloud has no data or different month_ends, fallback is server-date-based. |
| `app/ui/filters.py` | `SMART_INIT_DONE_KEY`: first run only calls `resolve_best_default_filters` and sets FilterState | First run wins; later runs keep that state until user changes dates. |
| `app/pages/visualisations.py` | `tab1_period` = `st.session_state.get("tab1_period", "YTD")` (1661) | Default "YTD" is consistent; no env dependency. |
| `app/pages/visualisations.py` | `tab1_filter_channel` etc. = `st.session_state.get(key, "All")` (378–382, 396) | Default "All" is consistent. |
| `app/data_gateway.py` | `_get_dataset_version()`: when in Streamlit, returns `st.session_state.get("dataset_version", "dev")` (310) | dataset_version is **not** set from load_dataset_version in session; gateway uses config from filesystem. So session "dataset_version" can be stale/wrong for display; cache uses config’s dataset_version in run_governed_query path; pyramid path uses `load_dataset_version(root)` (1672). |

**Risky:** Initial FilterState is **data-dependent** (list_month_ends) or **time-dependent** (date.today()). That directly changes the query date range and thus which row is “latest” and what NNB/OGR/Market are.

### 2.4 Scope logic (firm-wide vs selected slice)

| File | Logic |
|------|--------|
| `app/pages/visualisations.py` (1720–1747) | `source_for_monthly` = `firm_period` by default; overridden to ticker/segment/geo/channel-scoped frame when the corresponding `tab1_filter_*` != "All". So with all filters "All", KPIs are from **firm_monthly** (firm-wide). |
| `app/pages/visualisations.py` (418–451) | `_render_core_metrics(monthly, firm_monthly)`: "End AUM (selected slice)" and NNB/OGR/Market all come from **monthly** (selected slice). Firm-wide is only shown in caption as comparison. |

So if both environments have "All" filters, both use firm-wide data for the KPI row. Difference must come from **which month** is “latest” or **which data** is in that month (date range / dataset / cache).

### 2.5 Period mode and date defaults

| File | Detail |
|------|--------|
| `app/state.py` (352, 411–431) | `period_mode` default `"1M"`; `_default_date_end()` = last day of current month; `_default_date_start()` = month-end 12 months before that; both use **date.today()**. |
| `app/pages/visualisations.py` (318–335) | `_apply_period(df, period)`: "1M" = last month only; "YTD" = from year start to last_dt; then `latest = monthly.sort_values("month_end").iloc[-1]`. So "latest" is always the last row of the **filtered** monthly dataframe. |
| `app/data_gateway.py` (1086–1142) | `build_time_frames(state)` uses `state.period_mode` and `state.date_start` / `state.date_end` for governed queries. Tab 1 base queries use `filter_state_to_gateway_dict` (month_end_range only); period mode for display is `tab1_period` (YTD/1M/etc.) applied in the page. |

So the **backend** query range is set by FilterState (date_start/date_end). The **front-end** period (YTD/1M) only filters the already-fetched dataframe; it does not change the query. Different date_start/date_end → different rows returned → different “latest” and different KPIs.

### 2.6 KPI calculations (End AUM, NNB, OGR, Market Movement)

| Metric | Source | File:lines |
|--------|--------|------------|
| End AUM | `latest.get("end_aum")` from `monthly.sort_values("month_end").iloc[-1]` | `visualisations.py`:424–425, 433 |
| Net New Business | `latest.get("nnb")` same row | 425, 434 |
| Net New Flow | `latest.get("nnf")` same row | 426, 435 |
| Organic Growth | `latest.get("ogr")`; `ogr` computed in `_build_monthly_metrics` as `compute_ogr(nnb, begin_aum)` | 436; 351–352 |
| Market Movement | `latest.get("market_impact")`; `market_impact` = `compute_market_impact(begin_aum, end_aum, nnb)` | 426, 437; 349–350 |

`_build_monthly_metrics` (339–359): groupby `month_end`, sum begin_aum, end_aum, nnb, nnf; if all `begin_aum` are NA, set `monthly["begin_aum"] = monthly["end_aum"].shift(1)`. Then compute market_impact, ogr, market_impact_rate, fee_yield per row.

**Risky:** If the **order** of rows after groupby/sort differs (e.g. timezone or duplicate month_end), or if the **first** month has no prior row, `shift(1)` gives NA for that row → `begin_aum` wrong for that month → market_impact and ogr wrong even if end_aum and nnb are correct. So **same End AUM and NNB, different Market and OGR** is possible with wrong `begin_aum`.

---

## 3. Exact files and risky logic

### 3.1 Default date range (highest impact)

- **`app/state.py`**
  - **411–431** `_default_date_end()` / `_default_date_start()`: use `date.today()`. On Cloud this is server date/timezone → different default range than local.
  - **447–469** `FilterState.from_dict()`: when `date_start`/`date_end` missing, uses above defaults.
  - **699–731** `resolve_best_default_filters()`: calls `list_month_ends(base, view_name="v_firm_monthly")`; if no candidates have rows, returns `base` (from_dict with no dates → date.today() defaults); else returns first candidate with rows or full range. So Cloud with empty/different list_month_ends gets **server-date-based** defaults.

- **`app/ui/filters.py`**
  - **21–27** First run only: `resolve_best_default_filters(root)` then `set_filter_state(best_state)`. So the very first load sets the date range for the rest of the session.

### 3.2 Dataset version and cache keys

- **`app/data_gateway.py`**
  - **302–311** `_get_dataset_version()`: in Streamlit returns `st.session_state.get("dataset_version", "dev")`; otherwise `load_dataset_version(root)`. For `run_query` (pyramid path), **1672** uses `load_dataset_version(root)` (not session) → dataset_version from filesystem. So cache key is stable per deploy, but if Cloud has different or missing meta file, dataset_version can be "unknown" or "placeholder" (2708–2710) → different cache namespace.
  - **701–731** `load_dataset_version(root)`: reads from DuckDB meta table or `data/curated/metrics_monthly.meta.json` or fallback. If Cloud has no DB or different file, value differs.

### 3.3 begin_aum and derived metrics

- **`app/pages/visualisations.py`**
  - **342–348** `_build_monthly_metrics`: `monthly["begin_aum"] = monthly["end_aum"].shift(1)` when all begin_aum are NA. If there is only one month in the dataframe (e.g. Cloud default range is one month), that row gets NA for begin_aum → market_impact and ogr for that row are wrong (or NaN), while end_aum and nnb can still be correct.

### 3.4 Scope and “latest” row

- **`app/pages/visualisations.py`**
  - **1720–1733** `source_for_monthly`: depends on `tab1_filter_*` session_state. If Cloud had a leftover filter from a previous session (e.g. a specific channel), scope would differ. Default "All" is consistent; only risky if session state is reused incorrectly.

---

## 4. Prioritized fix plan

1. **Pin default date range to data, not server date (P0)**  
   - In `resolve_best_default_filters`, when `list_month_ends` is empty or fails, **do not** fall back to `FilterState.from_dict({})` (date.today()). Prefer:  
     - Fallback to a fixed range (e.g. last 12 months from a fixed “reference” month), or  
     - Require explicit config (e.g. env `DEFAULT_DATE_END`) so Cloud and local can be aligned.  
   - **Files:** `app/state.py` (e.g. 699–731), optionally `app/ui/filters.py` if you want a single “default range” source.

2. **Make default date range deterministic on Cloud (P0)**  
   - If you keep using `date.today()` anywhere for defaults, use a **fixed timezone** (e.g. UTC) and document it, so Cloud and local can match when data is the same.  
   - **Files:** `app/state.py` (`_default_date_end`, `_default_date_start`).

3. **Ensure begin_aum is never wrong for “latest” row (P1)**  
   - In `_build_monthly_metrics`, when filling `begin_aum` from `end_aum.shift(1)`, ensure the **last** row has a valid begin_aum (e.g. explicitly set last row’s begin_aum from the previous row’s end_aum, or from first available prior month).  
   - **Files:** `app/pages/visualisations.py` (347–348, 349–356).

4. **Cache and dataset_version (P1)**  
   - Ensure Cloud has the same (or intended) `dataset_version` as local (same meta file or same DuckDB).  
   - Optionally: add a small **cache-bust** or version in cache key when you know data or code changed (e.g. deploy timestamp or app version).

5. **Observability (P2)**  
   - Log or display on the dashboard: `state.date_start`, `state.date_end`, `dataset_version`, and the `month_end` of the row used as “latest”. That will immediately show whether Cloud and local are using the same range and same “latest” month.

6. **Session state (P2)**  
   - Ensure `tab1_filter_*` and `tab1_period` are never restored from a previous session in a way that differs between Cloud and local (e.g. don’t persist these to a store that differs by environment). Current defaults ("All", "YTD") are fine; only relevant if you add persistence.

---

## 5. Validation checklist

- [ ] Cloud and local use the **same** `date_start` / `date_end` for the same user intent (e.g. “last 12 months” or “latest month”).  
- [ ] When `list_month_ends` is empty or fails, default range does **not** depend on `date.today()` alone (or is explicitly aligned via config).  
- [ ] The row used as “latest” has a valid `begin_aum` (no NA from shift(1) for that row).  
- [ ] `dataset_version` on Cloud matches the one used for the data you expect (same meta file or DB).  
- [ ] No stale cache: after a data or code change, cache keys change or TTL is short enough / cache is cleared.

---

## 6. Why End AUM can match but NNB/OGR/Market differ (explicit)

- **End AUM** is read directly from the same “latest” row as **NNB** (`latest["end_aum"]`, `latest["nnb"]`). So if that row is the same, End AUM and NNB would both match. If they differ, the row is different (e.g. different month or different scope).
- **Market Movement** and **OGR** are **derived**:  
  - `market_impact = end_aum - begin_aum - nnb`,  
  - `ogr = nnb / begin_aum`.  
  So if **begin_aum** for that row is wrong (e.g. NA from shift(1), or from a different month due to sort/groupby), then:
  - End AUM and NNB (from the row) can still match,
  - Market and OGR (computed from begin_aum) will differ.  
  That gives **same End AUM, same NNB, different Market Movement and different Organic Growth** with no formula bug, only wrong or missing `begin_aum` for the “latest” row.
