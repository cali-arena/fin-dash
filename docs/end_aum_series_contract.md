# End AUM Series Contract

## Why the intermediate exists

Downstream metrics (**curated/metrics_monthly.parquet**) must **never** recompute `end_aum` from raw fact ad-hoc. A single, deterministic rollup of AUM at (month_end + slice keys) is defined once in **curated/intermediate/end_aum_series__{path_id}.parquet**. The metrics pipeline consumes these files only; it does not apply any `pd.groupby` on fact for `end_aum`.

- **One place for rollup rules**: Duplicate resolution (sum / max / last_non_null) is configured in **configs/rollup_rules.yml** and applied in the end_aum_series builder. Changing rollup or snapshot logic happens there, not in the metrics step.
- **Auditability**: QA duplicate reports (see below) and reconciliation checks are attached to the intermediate step, so discrepancies are caught before metrics.
- **Consistency**: All consumers of AUM (metrics, UI, reports) see the same end_aum values for a given (path_id, slice, month_end).

## Rollup rules config

**configs/rollup_rules.yml** defines:

- **grain_keys**: Must include `month_end`; slice keys come from drill_paths per path.
- **measures.end_aum**:
  - **rollup**: One of `sum`, `max`, `last_non_null`, `weighted_avg`. Determines how multiple fact rows at the same (month_end + keys) are collapsed to one `end_aum`.
  - **snapshot_tiebreak** (for `last_non_null`): `order_by` columns and `direction` (e.g. desc) to pick the last non-null value deterministically.
- **defaults**: e.g. `null_end_aum: "drop"` to drop rows with null AUM before aggregation; `strict: true` for strict validation.

The end_aum_series builder reads this config and applies the rollup per enabled drill path. The metrics pipeline does **not** read rollup_rules; it only reads the produced parquet files.

## QA duplicate audit

Before aggregation, the end_aum_series step computes:

- **dup_groups**: Count of (month_end + keys) groups with more than one row.
- **dup_rows_total**: Total rows in those groups.
- **dup_rate**: dup_rows_total / total input rows.

It writes:

- **qa/end_aum_duplicates__{path_id}.csv**: One row per duplicate grain (up to 500), with `month_end`, key columns, `rows_in_group`, `end_aum_min`, `end_aum_max`, `end_aum_sum`, and `sample_row_ids` (if a row-id column exists in fact).
- **qa/end_aum_series_stats__{path_id}.json**: `input_rows`, `output_rows`, `dup_groups`, `dup_rows_total`, `dup_rate`, `rollup_used`, `build_timestamp`.

After aggregation, for **sum** rollup it runs a reconciliation: sum of `end_aum` by month in the aggregated output must match sum of raw AUM by month within a small tolerance, or the step fails.

## How to change snapshot logic safely

1. **Change rollup type or tiebreak**: Edit **configs/rollup_rules.yml** (e.g. `measures.end_aum.rollup` or `snapshot_tiebreak.order_by` / `direction`). No code change if the builder already supports the option.
2. **Rebuild intermediates**: Run `python -m pipelines.slices.end_aum_series --build`. This regenerates all **curated/intermediate/end_aum_series__{path_id}.parquet** and QA artifacts.
3. **Rebuild metrics**: Run `python -m pipelines.metrics.compute_metrics --build`. Metrics will consume the new end_aum values; no change to metrics code required.
4. **Validate**: Check qa/end_aum_duplicates__*.csv and qa/end_aum_series_stats__*.json for the new duplicate stats and any new reconciliation failures.

Do **not** change how end_aum is derived inside the metrics pipeline; that would break the contract. All end_aum (and thus begin_aum) must come from the intermediate parquet files.
