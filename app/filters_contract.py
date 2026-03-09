"""
Filter contract loader and mode → column resolver. Used by gateway/query builders, not UI.
Loads app/config/filters.yml; resolves UI modes to canonical column names for SQL.
No DuckDB execution.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent


@lru_cache(maxsize=2)
def load_filters_contract(path: str = "app/config/filters.yml") -> dict[str, Any]:
    """Load filters contract YAML; return parsed dict. Cached."""
    full = Path(path) if Path(path).is_absolute() else _ROOT / path
    if not full.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(full.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve_channel_column(channel_view: str, contract: dict[str, Any]) -> str:
    """Resolve channel_view mode to canonical column name for SQL."""
    mapping = (contract.get("channel_view") or {}).get("column_mapping") or {}
    return mapping.get(channel_view, "preferred_label")


def resolve_geo_column(geo_dim: str, contract: dict[str, Any]) -> str:
    """Resolve geo_dim mode to canonical column name for SQL."""
    geo = contract.get("geo") or {}
    mapping = (geo.get("geo_dim") or {}).get("column_mapping") or {}
    return mapping.get(geo_dim, "src_country_canonical")


def resolve_product_column(product_dim: str, contract: dict[str, Any]) -> str:
    """Resolve product_dim mode to canonical column name for SQL."""
    product = contract.get("product") or {}
    mapping = (product.get("product_dim") or {}).get("column_mapping") or {}
    return mapping.get(product_dim, "product_ticker")


def is_optional_filter_enabled(
    filter_name: str,
    available_columns: set[str],
    contract: dict[str, Any],
) -> bool:
    """
    True if the optional filter is enabled: contract defines enabled_if_column_exists
    and that column is in available_columns. Used e.g. for custodian_firm.
    """
    section = contract.get(filter_name)
    if not isinstance(section, dict):
        return False
    col = section.get("enabled_if_column_exists")
    if not col or not isinstance(col, str):
        return False
    return col.strip() in available_columns
