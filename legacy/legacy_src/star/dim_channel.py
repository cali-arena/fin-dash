"""
Build dim_channel from fact_monthly. SCD Type 1; deterministic.
"""
from __future__ import annotations

import hashlib
import pandas as pd

KEY = "preferred_label"
COLS = ["preferred_label", "channel_l1", "channel_l2"]
SURROGATE = "channel_id"


def _stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_dim_channel(fact_monthly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per distinct preferred_label; channel_l1, channel_l2 from first occurrence.
    Natural key: preferred_label. Optional surrogate: channel_id (stable hash). Sorted by preferred_label.
    """
    missing = [c for c in COLS if c not in fact_monthly.columns]
    if missing:
        raise ValueError(f"fact_monthly missing columns: {missing}.")
    agg = fact_monthly.groupby(KEY, sort=False, dropna=False).agg(
        channel_l1=("channel_l1", "first"),
        channel_l2=("channel_l2", "first"),
    ).reset_index()
    if agg.empty:
        return _empty_dim_channel()
    agg[SURROGATE] = agg[KEY].astype(str).map(_stable_hash)
    out = agg[["preferred_label", "channel_l1", "channel_l2", SURROGATE]]
    out["channel_l1"] = out["channel_l1"].astype("string")
    out["channel_l2"] = out["channel_l2"].astype("string")
    out["preferred_label"] = out["preferred_label"].astype("string")
    return out.sort_values(KEY, kind="mergesort").reset_index(drop=True)


def _empty_dim_channel() -> pd.DataFrame:
    return pd.DataFrame(columns=["preferred_label", "channel_l1", "channel_l2", SURROGATE])
