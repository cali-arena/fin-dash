# Mapping audit and MAPPING_MISMATCH

## Rows labeled MAPPING_MISMATCH

Validation report rows with `fail_reason == "MAPPING_MISMATCH"` indicate that the **expected** value (e.g. from DATA SUMMARY or a checksum) was produced using different mapping or canonical keys than the **actual** value. Currently no rows in `validation_report.csv` are labeled MAPPING_MISMATCH; validation is firm vs firm and fail reasons are FORMULA_MISMATCH or SKIPPED_GRAIN.

If you see MAPPING_MISMATCH, trace:

- **Actual:** Built from `metrics_monthly.parquet`, which uses canonical keys produced by ETL mapping (see below).
- **Expected:** DATA SUMMARY is firm-level only (no slice keys). Any slice-level or differently mapped checksum would be a different source (e.g. mappings.yml, another sheet).

## Mapping tables / logic involved

| Source | Role | Where used |
|--------|------|------------|
| **DATA MAPPING sheet** | Excel sheet "DATA MAPPING" (source_table, source_field, target_field). Ingested to `data_mapping_normalized.parquet` by `etl/ingest_excel.py`. | Not used by `etl/transform_curated.py`; transform derives channel mapping from raw. |
| **Derived channel mapping** | Built in `etl/transform_curated.py`: from DATA RAW, distinct combos of `channel_raw`, `channel_standard`, `channel_best` → canonical `channel` via rule `best > standard > raw` (first non-empty). | Applied in `_apply_mapping()`; output `data/curated/channel_mapping.parquet`. |
| **Parquet mapping tables** | `data/curated/channel_mapping.parquet` (channel only). No separate segment/ticker mapping in this ETL. | Consumed implicitly via `metrics_monthly.parquet` (canonical `channel`). |
| **mappings.yml** | If present elsewhere in repo (e.g. pipelines or app config). | Check codebase for references; this ETL path does not use mappings.yml. |

So for this pipeline the only mapping that affects canonical keys is the **derived channel mapping** (best>standard>raw). Segment and sub_segment are passed through from raw; product_ticker is not remapped here.

## Strict mapping audit (qa/unmapped_keys.csv)

- **Produced by:** `etl/transform_curated.py` after applying the channel mapping.
- **Content:** Distinct raw key combos that **fail to map** to a non-empty canonical `channel`:
  - Key columns: `channel_raw`, `channel_standard`, `channel_best`, `segment`, `sub_segment`
  - `row_count`: number of raw rows with that combo
  - `sample_month_end`: min(month_end) for that combo (deterministic)
  - `sample_ticker`: min(product_ticker) for that combo (deterministic)
- **Also written:** `qa/unmapped_keys.meta.json` with `total_raw_rows`, `unmapped_rows`, `unmapped_keys` for the quality gate.
- **Related:** `qa/unmapped_channels.csv` lists distinct unmapped channel triples (no segment/count/sample).

## Adjusting mapping

- **To match DATA SUMMARY expectation:**  
  - Either **update mapping rules** in `etl/transform_curated.py` (e.g. change fallback order, or load DATA MAPPING and join instead of deriving).  
  - Or **add a translation layer** in ETL: after building canonical `channel`, if DATA SUMMARY uses a different canonical key, add a small table or if/else that maps our canonical → DATA SUMMARY canonical and apply it before writing `metrics_monthly.parquet` (or only for validation inputs).
- **If DATA SUMMARY uses a different canonical key:** Add the translation in ETL so the keys used for validation match exactly; do not change the definition of FORMULA_MISMATCH for real formula errors.

## Quality gate (unmapped ratio > 1%)

- **Script:** `qa/mapping_gate.py`
- **Logic:** Reads `qa/unmapped_keys.meta.json`. `ratio = unmapped_rows / max(total_raw_rows, 1)`. If `ratio > 0.01` (1%): **exit 2** (QA fail), print message. Otherwise exit 0.
- **Deterministic:** Uses only the meta file written by `etl/transform_curated.py`. Run after ETL transform.

```bash
python qa/mapping_gate.py --qa-dir qa
```
