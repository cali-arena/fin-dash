"""
Curated fact tables at defined grains.
"""
from legacy.legacy_src.curate.fact_monthly import (
    GRAIN,
    aum_snapshot_rule,
    build_fact_monthly,
    derive_channels,
)
from legacy.legacy_src.curate.mapping_loader import extract_channel_mapping, load_data_mapping

__all__ = [
    "GRAIN",
    "aum_snapshot_rule",
    "build_fact_monthly",
    "derive_channels",
    "extract_channel_mapping",
    "load_data_mapping",
]
