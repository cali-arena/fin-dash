"""
Performance budget enforcement: configurable limits, warn/fail in dev.
Level A (queries): firm_monthly 200ms, channel/ticker/geo/segment 500ms.
Level C (charts): heavy_chart 1000ms.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)
DEFAULT_PERF_BUDGET_PATH = "configs/perf_budget.yml"

# Default budgets when config missing
DEFAULT_FIRM_MS = 200
DEFAULT_CHANNEL_MS = 500
DEFAULT_TICKER_MS = 500
DEFAULT_GEO_MS = 500
DEFAULT_SEGMENT_MS = 500
DEFAULT_HEAVY_CHART_MS = 1000
DEFAULT_WARN_MULTIPLIER = 1.25


@dataclass
class PerfBudgetConfig:
    firm_monthly_ms: int
    channel_monthly_ms: int
    ticker_monthly_ms: int
    geo_monthly_ms: int
    segment_monthly_ms: int
    heavy_chart_ms: int
    warn_multiplier: float
    fail_in_dev: bool


def _default_config() -> PerfBudgetConfig:
    return PerfBudgetConfig(
        firm_monthly_ms=DEFAULT_FIRM_MS,
        channel_monthly_ms=DEFAULT_CHANNEL_MS,
        ticker_monthly_ms=DEFAULT_TICKER_MS,
        geo_monthly_ms=DEFAULT_GEO_MS,
        segment_monthly_ms=DEFAULT_SEGMENT_MS,
        heavy_chart_ms=DEFAULT_HEAVY_CHART_MS,
        warn_multiplier=DEFAULT_WARN_MULTIPLIER,
        fail_in_dev=True,
    )


def _load_from_path(path: Path) -> PerfBudgetConfig:
    """Load perf_budget from YAML; return defaults on missing/invalid."""
    if not path.exists():
        return _default_config()
    try:
        import yaml
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("perf_budget: failed to load %s: %s; using defaults", path, e)
        return _default_config()
    pb = raw.get("perf_budget") or raw
    if not isinstance(pb, dict):
        return _default_config()
    return PerfBudgetConfig(
        firm_monthly_ms=int(pb.get("firm_monthly_ms", DEFAULT_FIRM_MS)),
        channel_monthly_ms=int(pb.get("channel_monthly_ms", DEFAULT_CHANNEL_MS)),
        ticker_monthly_ms=int(pb.get("ticker_monthly_ms", DEFAULT_TICKER_MS)),
        geo_monthly_ms=int(pb.get("geo_monthly_ms", DEFAULT_GEO_MS)),
        segment_monthly_ms=int(pb.get("segment_monthly_ms", DEFAULT_SEGMENT_MS)),
        heavy_chart_ms=int(pb.get("heavy_chart_ms", DEFAULT_HEAVY_CHART_MS)),
        warn_multiplier=float(pb.get("warn_multiplier", DEFAULT_WARN_MULTIPLIER)),
        fail_in_dev=bool(pb.get("fail_in_dev", True)),
    )


def _resolve_path(path: str | Path | None, root: Path | None) -> Path:
    p = Path(path) if path else Path(DEFAULT_PERF_BUDGET_PATH)
    if not p.is_absolute() and root is not None:
        p = root / p
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


if st is not None:
    @st.cache_resource
    def _load_perf_budget_cached(_path_key: str) -> PerfBudgetConfig:
        return _load_from_path(Path(_path_key))

    def load_perf_budget(path: str | Path | None = None, root: Path | None = None) -> PerfBudgetConfig:
        """Load perf budget config (cached when in Streamlit)."""
        p = _resolve_path(path, root)
        return _load_perf_budget_cached(str(p.resolve()))
else:
    def load_perf_budget(path: str | Path | None = None, root: Path | None = None) -> PerfBudgetConfig:
        """Load perf budget config."""
        p = _resolve_path(path, root)
        return _load_from_path(p)


def _is_dev_mode() -> bool:
    if os.environ.get("DEV_MODE") == "1":
        return True
    if st is not None:
        try:
            return bool(st.secrets.get("DEV_MODE"))
        except Exception:
            pass
    return False


QUERY_BUDGET_MAP: dict[str, str] = {
    "firm_monthly": "firm_monthly_ms",
    "channel_monthly": "channel_monthly_ms",
    "ticker_monthly": "ticker_monthly_ms",
    "geo_monthly": "geo_monthly_ms",
    "segment_monthly": "segment_monthly_ms",
}


def get_budget_ms(level: str, name: str, config: PerfBudgetConfig) -> float:
    """
    Return budget in ms for level+name. Pure function for testing.
    Level A: query_name -> firm/channel/ticker/geo/segment budget.
    Level C: chart_name -> heavy_chart_ms.
    Level B: no explicit budget; return heavy_chart as fallback (typically not checked).
    """
    if level == "A":
        key = QUERY_BUDGET_MAP.get(name)
        if key == "firm_monthly_ms":
            return float(config.firm_monthly_ms)
        if key == "channel_monthly_ms":
            return float(config.channel_monthly_ms)
        if key == "ticker_monthly_ms":
            return float(config.ticker_monthly_ms)
        if key == "geo_monthly_ms":
            return float(config.geo_monthly_ms)
        if key == "segment_monthly_ms":
            return float(config.segment_monthly_ms)
        return float(config.channel_monthly_ms)  # default for unknown query
    if level == "C":
        return float(config.heavy_chart_ms)
    return float(config.heavy_chart_ms)  # Level B or unknown


def check_perf_budget(
    level: str,
    name: str,
    elapsed_ms: float,
    config: PerfBudgetConfig | None = None,
    root: Path | None = None,
) -> None:
    """
    Enforce perf budget: if exceeded and (dev + fail_in_dev) -> raise RuntimeError.
    Else log warning and append to slow_queries.
    """
    if config is None:
        config = load_perf_budget(root=root)
    budget = get_budget_ms(level, name, config)
    if budget <= 0:
        return
    warn_threshold = budget * config.warn_multiplier
    exceeded = elapsed_ms > budget
    warn_exceeded = elapsed_ms > warn_threshold
    if not exceeded:
        return
    msg = f"Perf budget exceeded: {level}/{name} took {elapsed_ms:.0f}ms (budget {budget:.0f}ms)"
    if _is_dev_mode() and config.fail_in_dev:
        raise RuntimeError(
            f"{msg}. Fix query/index or increase budget in configs/perf_budget.yml"
        )
    logger.warning("%s", msg)
    if st is not None:
        if "slow_queries" not in st.session_state:
            st.session_state["slow_queries"] = []
        st.session_state["slow_queries"] = (
            st.session_state["slow_queries"] + [{
                "level": level,
                "name": name,
                "elapsed_ms": round(elapsed_ms, 2),
                "budget_ms": budget,
                "type": "perf_budget",
            }]
        )[-50:]
