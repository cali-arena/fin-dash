# Validation grain guardrails

## Computation path

| Role | Source | Grain |
|------|--------|--------|
| **Actual** | `qa/validate_vs_data_summary.run()`: reads `metrics_monthly.parquet` (slice × month_end), aggregates with `groupby("month_end").sum()` over begin_aum, end_aum, nnb, market_impact; then computes asset_growth_rate_calc, organic_growth_rate_calc, external_growth_rate_calc on that firm-level table. | **firm** (one row per month_end) |
| **Expected** | DATA SUMMARY: `data_summary_normalized.parquet` from Excel sheet "DATA SUMMARY" (asset_growth_rate, organic_growth_rate, external_growth_rate). | **firm** (one row per month_end) |

Tab 1 / gateway / report pack: firm snapshot metrics are computed from the same firm-level view (e.g. `v_firm_monthly` or aggregated metrics). They must use **firm grain only** (no slice leakage). Slice-level metrics (channel, ticker, geo) are never compared to DATA SUMMARY unless the checksum is also sliced (not currently the case).

## GRAIN_MISMATCH and SKIPPED_GRAIN

- **GRAIN_MISMATCH**: Reserved for rows where the failure was classified as due to comparing different grains (e.g. slice actual vs firm expected). None in current report; validation is firm vs firm only.
- **SKIPPED_GRAIN**: If validation were ever run with mismatched grain (e.g. actual at slice, expected at firm), those rows are marked `skip_reason = "SKIPPED_GRAIN"` and `fail_reason = "SKIPPED_GRAIN"`; `any_fail` is set to False so they do not count as formula failures. Real math issues are not hidden—only grain alignment is corrected.

## Report columns (from pipeline)

- `validation_grain_actual`, `validation_grain_expected`: always "firm" for this pipeline.
- `skip_reason`: "SKIPPED_GRAIN" when grain actual ≠ grain expected; else "".
- `fail_reason`: One of allowed (FORMULA_MISMATCH, GRAIN_MISMATCH, …) or "SKIPPED_GRAIN" when skipped.
- `fail_note`: One-sentence explanation.

## Regenerating the report

Run validation after ETL so curated parquet exists:

```bash
# From repo root, after ETL has produced data/curated/
python qa/validate_vs_data_summary.py --curated-dir data/curated --qa-dir qa
```

Output: `qa/validation_report.csv` with grain columns and fail_reason/fail_note populated.
