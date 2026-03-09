"""
Single loader for curated fact_monthly. Downstream KPIs and Streamlit visuals must use this only.
No reads of raw/data_raw.parquet or data/input/source.xlsx for visuals.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FACT_PATH = "curated/fact_monthly.parquet"
DEFAULT_META_PATH = "curated/fact_monthly.meta.json"


def load_fact_monthly(
    path: str | Path = DEFAULT_FACT_PATH,
    *,
    root: Path | None = None,
    check_meta: bool = True,
) -> pd.DataFrame:
    """
    Load curated fact_monthly from parquet (or CSV fallback at same stem).
    If check_meta is True and curated/fact_monthly.meta.json exists, log dataset_version.
    """
    root = Path(root) if root is not None else Path.cwd()
    full_path = root / path if not Path(path).is_absolute() else Path(path)
    meta_path = full_path.parent / "fact_monthly.meta.json"

    if check_meta and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dataset_version = meta.get("dataset_version")
            if dataset_version is not None:
                logger.info("fact_monthly meta: dataset_version=%s", dataset_version)
        except Exception:
            pass

    if full_path.is_file():
        if full_path.suffix.lower() == ".parquet":
            return pd.read_parquet(full_path)
        if full_path.suffix.lower() == ".csv":
            return pd.read_csv(full_path)
    csv_path = full_path.parent / (full_path.stem + ".csv")
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.DataFrame()
