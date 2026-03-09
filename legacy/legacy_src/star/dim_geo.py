"""
Build dim_geo from fact_monthly (src_country + product_country). SCD Type 1; deterministic.
Role-playing: fact joins to dim_geo twice via src_country and product_country -> country_key.
"""
from __future__ import annotations

import pandas as pd

COLS = ["src_country", "product_country"]
KEY = "country_key"


def _norm(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return str(s).strip()


def build_dim_geo(fact_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per distinct country value from src_country and product_country.
    country_key = normalized value (join key); country_raw = original; country_norm = normalized display.
    iso2, iso3, region nullable (empty for now). Sorted by country_key.
    """
    missing = [c for c in COLS if c not in fact_monthly.columns]
    if missing:
        raise ValueError(f"fact_monthly missing columns: {missing}.")
    raw = pd.concat([
        fact_monthly["src_country"].dropna().astype(str).str.strip(),
        fact_monthly["product_country"].dropna().astype(str).str.strip(),
    ], ignore_index=True)
    raw = raw[raw != ""].drop_duplicates().sort_values(kind="mergesort").reset_index(drop=True)
    if raw.empty:
        return _empty_dim_geo()
    raw_str = raw.astype("string")
    out = pd.DataFrame({
        "country_key": raw_str,
        "country_raw": raw_str,
        "country_norm": raw_str,
        "iso2": pd.Series([pd.NA] * len(raw_str), dtype="string"),
        "iso3": pd.Series([pd.NA] * len(raw_str), dtype="string"),
        "region": pd.Series([pd.NA] * len(raw_str), dtype="string"),
    })
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _empty_dim_geo() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "country_key", "country_raw", "country_norm", "iso2", "iso3", "region"
    ])
