"""Contract and join definitions for fact + dimensions (star)."""
from legacy.legacy_pipelines.contracts.star_contract import (
    FACT_PATH,
    DIM_PATHS,
    load_fact_enriched,
    validate_join_coverage,
)

__all__ = [
    "FACT_PATH",
    "DIM_PATHS",
    "load_fact_enriched",
    "validate_join_coverage",
]
