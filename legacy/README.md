# Legacy modules (deprecated)

**DO NOT IMPORT** from this package in `app/`, `etl/`, or `models/`.

This folder contains deprecated/legacy code preserved for history and for tests that still exercise the old pipelines and `src` layout:

- **legacy_src/** — former `src/` (curate, ingest, mapping, persist, quality, schemas, star, transform, validators, versioning).
- **legacy_pipelines/** — former `pipelines/` (agg, contracts, dimensions, duckdb, metrics, qa, slices, validation).

Canonical code lives under:

- **app/** — dashboard, gateway, contracts (cache policy, star join), metrics, UI.
- **etl/** — canonical ETL entrypoints (ingest, transform, build_agg, build_duckdb).
- **models/** — schema and query specs.

CI fails if any file under `app/`, `etl/`, or `models/` imports from `legacy.` or references the old top-level names `src` or `pipelines`.
