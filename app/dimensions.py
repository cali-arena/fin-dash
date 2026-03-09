"""
Canonical dimension helpers used by star join contract (fact enrichment).
Single source for normalize_country used in app.contracts.star_contract.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

_GEO_ISO2 = re.compile(r"^[A-Za-z]{2}$")
_GEO_ISO3 = re.compile(r"^[A-Za-z]{3}$")


def normalize_country(s: str | Any) -> str:
    """
    Canonical country (deterministic): trim, collapse spaces, remove leading/trailing punctuation.
    Convert separators (/ - _) to spaces and collapse again. 2- or 3-letter alpha -> uppercase; else uppercase.
    Used for dim_geo and region join.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    x = str(s).strip()
    for sep in ["/", "-", "_"]:
        x = x.replace(sep, " ")
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return ""
    if _GEO_ISO2.match(x) or _GEO_ISO3.match(x):
        return x.upper()
    return x.upper()
