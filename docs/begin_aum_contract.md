# Begin AUM Contract

## Overview

**begin_aum** is the previous month’s **end_aum** within the same slice. It is produced once in the slices pipeline and consumed by the metrics layer. The metrics pipeline must **never** recompute begin_aum (no groupby/shift on AUM).

## Strict ordering requirement

- **month_end** must be non-null and normalized (timezone-naive datetime).
- Within each slice (partition by slice keys), rows must be sorted by **month_end** ascending.
- **month_end** must be **strictly increasing** within each slice (no duplicate months). This is enforced by grain uniqueness of (month_end + keys) in the upstream end_aum_series and preserved in aum_series.

So begin_aum is defined as: for each slice, the value of end_aum from the row with the **previous** month_end. The first month in each slice has no previous month, so begin_aum is null and **is_first_month_in_slice** is true.

## Grain uniqueness prerequisite

- Upstream **end_aum_series** and **aum_series** are built with grain **(month_end + keys)** unique per path.
- The combined **aum_series_all.parquet** has grain **(path_id, slice_id, month_end)** unique (validated at build time).
- The metrics pipeline assumes this uniqueness and does not re-validate grain; it only validates that metrics rows reference slices present in slices_index.

## Authoritative sources

- **AUM-related metrics** (end_aum, begin_aum): **curated/intermediate/aum_series_all.parquet** (or per-path **aum_series__{path_id}.parquet**). The metrics pipeline loads this and never recomputes begin_aum.
- **nnb/nnf**: Optional merge from fact at the same grain (month_end + path keys); not part of the begin_aum contract.

## Diagnostics produced

- **months_in_slice**: Count of rows in that slice (repeated per row).
- **slice_min_month_end**, **slice_max_month_end**: Min and max month_end in that slice (repeated per row).
- **is_first_month_in_slice**: True when begin_aum is null (first month in the slice).

These are produced in **pipelines/slices/begin_aum_series.py** and written into the per-path **aum_series__{path_id}.parquet** and the combined **aum_series_all.parquet** (combined omits slice_min/max to stay lean; it does include months_in_slice and is_first_month_in_slice).

## Reconciliation QA (metrics pipeline)

When building metrics_monthly, the pipeline runs two checks on the loaded aum_series:

1. **First-month count vs unique slices**: For each path_id, sum(is_first_month_in_slice) must equal the number of unique slices. If not, the run fails and **qa/begin_aum_first_month_anomaly__{path_id}.csv** is written (per-slice n_first_month and expected 1).
2. **Lag spot-check**: A seeded random sample of 50 slices is checked: for each non-first month, begin_aum must equal the previous row’s end_aum. If any mismatch, the run fails and **qa/begin_aum_mismatch_samples.csv** is written (slice_id, month_end, expected, got).

These ensure that begin_aum is consistent with the strict ordering and grain uniqueness requirements.
