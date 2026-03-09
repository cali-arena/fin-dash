"""
Data access layer: load only from curated outputs. Downstream KPIs/visuals use curated/fact_monthly only.
"""
from legacy.legacy_src.data_access.fact_store import load_fact_monthly

__all__ = ["load_fact_monthly"]
