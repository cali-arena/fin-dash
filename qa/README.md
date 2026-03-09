# QA: demo scenarios, unmapped reports, tests

## Demo scenarios

Deterministic demo replay is described in **`qa/demo_scenarios.md`**.

- **How to run:** Use the scenarios as a manual checklist. From project root:
  1. Ensure data is built (`make build-data` or `make build-data-docker`).
  2. Start the app (`make run` or `make up`).
  3. Follow each scenario (Tab 1 firm snapshot, drill to channel; Tab 2 report + exports; Tab 3 NLQ questions).
  4. Use the **Pass/Fail checklist** at the end of `qa/demo_scenarios.md` to verify exports, no timeouts, and observability.

- **Reproducibility:** Use the recommended date range and filters in the doc so outputs are stable. Exports (CSV, HTML) should work from every table listed.

## Unmapped reports

QA artifacts and unmapped/coverage reports live in **`qa/`**:

- `metrics_policy_gate_report.json`
- `join_coverage.json`, `dim_product_coverage.json`, `dim_time_summary.json`
- `geo_raw_examples.csv`, `geo_dropped_empty_count.json`

These are used for data-quality and coverage checks; paths and naming may be referenced by pipelines or internal tooling.

## Unit tests

From project root:

```bash
make qa
```

or:

```bash
pytest -q
```

To run with verbose output or a specific path:

```bash
pytest tests/ -v
pytest tests/test_query_spec.py tests/test_executor.py -v
```

Tests require the same dependencies as the app (see `requirements.txt` / `pyproject.toml`). No data or DuckDB is required for unit tests that mock the gateway.
