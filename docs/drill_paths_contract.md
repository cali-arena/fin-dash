# Drill Paths Contract

## Overview

Supported drill paths are defined in **configs/drill_paths.yml**. The pipeline computes metrics strictly per this contract, and the UI only selects **path** and **slice** (slice_id); no ad-hoc groupbys or merges for core metrics.

## Config: configs/drill_paths.yml

- **drill_paths**: List of paths. Each has `id`, `label`, `keys` (list of fact column names), and `enabled` (true/false).
- Only **enabled** paths are used by the pipeline and exposed in the UI.
- Changing the config (enabling/disabling paths, or changing keys) changes available slices **deterministically after rebuild**:
  1. Run `python -m pipelines.slices.slice_keys --build` to regenerate **curated/slices_index.parquet**.
  2. Run `python -m pipelines.metrics.compute_metrics --build` to regenerate **curated/metrics_monthly.parquet**.

## Pipeline

- **Slice index** (`pipelines/slices/slice_keys.py`): Builds one row per (path_id, key combination) with `slice_id` (stable hash) and human `slice_key`. Output: `curated/slices_index.parquet`.
- **Metrics** (`pipelines/metrics/compute_metrics.py`): For each enabled path, groups fact by `[month_end] + path.keys`, computes `end_aum`, `nnb`, `nnf`, `begin_aum`, `aum_growth_rate`; attaches `path_id`, `slice_id`, `slice_key`. Output: **curated/metrics_monthly.parquet**. All (path_id, slice_id) in metrics are validated against slices_index (contract).

## UI

- **app/drill_paths.py**: Loads config and slices_index; exposes `get_enabled_paths()` and `get_slices_for_path(path_id)`.
- **Data source**: The UI loads data **only** from **curated/metrics_monthly.parquet**, filtered by the user-selected **(path_id, slice_id)**.
- **Dropdown 1**: Drill path (path_id / label from enabled paths).
- **Dropdown 2**: Slice (slice_key displayed, slice_id used for filtering).
- No `pd.groupby` in the UI layer for these KPIs: only **filtering + plotting**. A startup guardrail (`STRICT_UI_SLICES=true`) fails if `.groupby(` is found in app code (e.g. app/*.py, app/pages/*.py).

## Acceptance

- Changing **configs/drill_paths.yml** changes available slices deterministically after running slice_keys and compute_metrics.
- The UI works with no ad-hoc merges/groupbys for core metrics; all metric data comes from metrics_monthly filtered by path_id and slice_id.
