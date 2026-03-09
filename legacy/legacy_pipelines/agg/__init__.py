"""Pre-aggregation pipeline: materialize agg tables from metrics_monthly with caching."""

from legacy.legacy_pipelines.agg.manifest import get_table, load_manifest
from legacy.legacy_pipelines.agg.materialize_aggs import materialize_aggs

__all__ = ["materialize_aggs", "load_manifest", "get_table"]
