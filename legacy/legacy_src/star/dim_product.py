"""
Build dim_product from fact_monthly. SCD Type 1; deterministic.
"""
from __future__ import annotations

import hashlib
import pandas as pd

KEY = "product_ticker"
COLS = ["product_ticker", "segment", "sub_segment"]
SURROGATE = "product_id"


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_dim_product(fact_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per distinct product_ticker; segment, sub_segment from first occurrence.
    Natural key: product_ticker. Optional surrogate: product_id (stable hash). Sorted by product_ticker.
    """
    missing = [c for c in COLS if c not in fact_monthly.columns]
    if missing:
        raise ValueError(f"fact_monthly missing columns: {missing}.")
    agg = fact_monthly.groupby(KEY, sort=False, dropna=False).agg(
        segment=("segment", "first"),
        sub_segment=("sub_segment", "first"),
    ).reset_index()
    if agg.empty:
        return _empty_dim_product()
    agg[SURROGATE] = agg[KEY].astype(str).map(_stable_hash)
    out = agg[["product_ticker", "segment", "sub_segment", SURROGATE]]
    for c in ["product_ticker", "segment", "sub_segment"]:
        out[c] = out[c].astype("string")
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _empty_dim_product() -> pd.DataFrame:
    return pd.DataFrame(columns=["product_ticker", "segment", "sub_segment", SURROGATE])
