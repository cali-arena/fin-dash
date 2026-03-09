"""
Optional cache pre-warming on app start: global filter + top channels.
Populates Level A/B/C caches so first load feels instant.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def prewarm_enabled() -> bool:
    """True when PREWARM=1 (DEV_MODE not required)."""
    return os.environ.get("PREWARM") == "1"


def _build_global_filter_state() -> dict[str, Any]:
    """Build filter_state: latest 24 months, no dimension filters."""
    end = pd.Timestamp.now().normalize()
    start = end - pd.DateOffset(months=24)
    return {"month_end_range": (start, end)}


def prewarm_common(root: Path | None = None) -> dict[str, Any]:
    """
    Pre-warm Level A/B/C caches with global filter (latest 24 months).
    Returns {"ok": bool, "elapsed_ms": float, "calls": [...]}.
    On failure: logs warning, returns ok=False, does not raise.
    """
    root = root or ROOT
    filter_state = _build_global_filter_state()
    timings: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    try:
        from app.data_gateway import (
            Q_CHANNEL_MONTHLY,
            Q_FIRM_MONTHLY,
            run_aggregate,
            run_chart,
            run_query,
        )
        from app.cache.specs import AGG_CHANNEL_MIX, AGG_KPI_CARDS, CHART_WATERFALL
    except ImportError as e:
        logger.warning("prewarm: import failed: %s", e)
        return {"ok": False, "elapsed_ms": 0, "calls": [], "error": str(e)}

    try:
        run_query(Q_FIRM_MONTHLY, filter_state, root=root)
    except Exception as e:
        logger.warning("prewarm run_query failed: %s", e)
        timings.append({"name": "run_query", "elapsed_ms": 0, "error": str(e)})
    else:
        timings.append({"name": "run_query", "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2)})

    t1 = time.perf_counter()
    try:
        run_aggregate(AGG_CHANNEL_MIX, Q_CHANNEL_MONTHLY, filter_state, root=root, top_n=10)
    except Exception as e:
        logger.warning("prewarm run_aggregate(channel_mix) failed: %s", e)
        timings.append({"name": "run_aggregate", "elapsed_ms": round((time.perf_counter() - t1) * 1000, 2), "error": str(e)})
    else:
        timings.append({"name": "run_aggregate", "elapsed_ms": round((time.perf_counter() - t1) * 1000, 2)})

    t2 = time.perf_counter()
    try:
        run_chart(CHART_WATERFALL, AGG_KPI_CARDS, Q_FIRM_MONTHLY, filter_state, root=root)
    except Exception as e:
        logger.warning("prewarm run_chart failed: %s", e)
        timings.append({"name": "run_chart", "elapsed_ms": round((time.perf_counter() - t2) * 1000, 2), "error": str(e)})
    else:
        timings.append({"name": "run_chart", "elapsed_ms": round((time.perf_counter() - t2) * 1000, 2)})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if st is not None:
        if "prewarm_timings" not in st.session_state:
            st.session_state["prewarm_timings"] = []
        st.session_state["prewarm_timings"] = [{
            "total_ms": round(elapsed_ms, 2),
            "calls": timings,
        }]

    return {"ok": True, "elapsed_ms": elapsed_ms, "calls": timings}
