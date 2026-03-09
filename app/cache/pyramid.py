"""
3-layer caching pyramid: Level A (filtered base), Level B (aggregates), Level C (chart payloads).
Heavy intermediates never rerun unnecessarily; all layers keyed by dataset_version + filter_state_hash + names.
TTL per layer via configs/cache_policy.yml; Level A uses query_classes, B=medium, C=heavy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None

# TTL seconds per class (must be literal for Streamlit decorators)
_TTL_FAST = 60
_TTL_MEDIUM = 300
_TTL_HEAVY = 1800


def get_level_a_ttl_class(query_name: str, policy: Any) -> str:
    """
    Return TTL class for Level A (filtered) by query_name: 'fast' | 'medium' | 'heavy'.
    Used by the router; testable without Streamlit.
    """
    if policy is None:
        return "medium"
    return policy.query_classes.get(query_name, "medium")


def _get_ttl_seconds(ttl_class: str, policy: Any) -> int:
    """Return TTL seconds for class from policy; fallback to defaults."""
    if policy is not None and hasattr(policy, "ttl_seconds"):
        return policy.ttl_seconds.get(ttl_class, _TTL_MEDIUM)
    return {  # type: ignore[return-value]
        "fast": _TTL_FAST,
        "medium": _TTL_MEDIUM,
        "heavy": _TTL_HEAVY,
    }.get(ttl_class, _TTL_MEDIUM)


def _filter_state_from_json(filter_state_json: str) -> dict[str, Any]:
    if not filter_state_json:
        return {}
    return json.loads(filter_state_json)


def _root_from_str(root_str: str | None) -> Path | None:
    if root_str is None or root_str == "":
        return None
    return Path(root_str)


def _level_a_impl(
    dataset_version: str,
    query_name: str,
    filter_state_hash: str,
    filter_state_json: str,
    root_str: str | None,
) -> pd.DataFrame:
    """Level A inner: run query uncached."""
    from app.data_gateway import _run_query_uncached
    filter_state = _filter_state_from_json(filter_state_json)
    root = _root_from_str(root_str)
    return _run_query_uncached(query_name, filter_state, root=root)


# --- Level A: TTL-routed filtered base ---

if st is not None:
    @st.cache_data(show_spinner=False, ttl=_TTL_FAST)
    def get_filtered_fast(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _level_a_impl(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)

    @st.cache_data(show_spinner=False, ttl=_TTL_MEDIUM)
    def get_filtered_medium(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _level_a_impl(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)

    @st.cache_data(show_spinner=False, ttl=_TTL_HEAVY)
    def get_filtered_heavy(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _level_a_impl(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)

    def get_filtered(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        """Level A: route to fast/medium/heavy by policy query_class; record cache debug and obs."""
        from app.cache.cache_debug import record_cache_call
        from app.observability.debug_panel import register_cache_call as obs_register
        root = _root_from_str(root_str)
        policy = None
        try:
            from app.cache.cache_gateway import load_cache_policy
            policy = load_cache_policy(root=root)
        except Exception:
            pass
        query_class = get_level_a_ttl_class(query_name, policy)
        ttl = _get_ttl_seconds(query_class, policy)
        t0 = time.perf_counter()
        if query_class == "fast":
            result = get_filtered_fast(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)
        elif query_class == "heavy":
            result = get_filtered_heavy(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)
        else:
            result = get_filtered_medium(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        record_cache_call("A", query_name, dataset_version, filter_state_hash, elapsed_ms, len(result), ttl_seconds=ttl)
        backend = st.session_state.get("_last_backend", "") if st is not None else ""
        obs_register("A", query_name, dataset_version, filter_state_hash, elapsed_ms, len(result), extra_key="", backend=backend)
        try:
            from app.perf_budget import check_perf_budget, load_perf_budget
            policy = load_perf_budget(root=root)
            check_perf_budget("A", query_name, elapsed_ms, config=policy, root=root)
        except RuntimeError:
            raise
        except Exception:
            pass
        return result
else:
    def get_filtered(
        dataset_version: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
    ) -> pd.DataFrame:
        return _level_a_impl(dataset_version, query_name, filter_state_hash, filter_state_json, root_str)


# --- Level B: Derived aggregates ---

def _compute_aggregate(
    agg_name: str,
    df: pd.DataFrame,
    agg_params: dict[str, Any] | None = None,
) -> dict[str, Any] | pd.DataFrame:
    """Compute aggregate by agg_name from filtered DataFrame. Uses app.cache.specs when registered."""
    from app.cache import specs as cache_specs
    if df.empty:
        if agg_name == "raw":
            return df
        if agg_name in cache_specs.AGG_SPECS:
            spec = cache_specs.AGG_SPECS[agg_name]
            fn = spec["fn"]
            if spec.get("params"):
                return fn(df, agg_params or {})
            return fn(df)
        return {}
    if agg_name in cache_specs.AGG_SPECS:
        spec = cache_specs.AGG_SPECS[agg_name]
        fn = spec["fn"]
        if spec.get("params"):
            return fn(df, agg_params or {})
        return fn(df)
    # Legacy aggregates (backward compat)
    if agg_name == "kpi_totals":
        out: dict[str, Any] = {"row_count": len(df)}
        for col in ("begin_aum", "end_aum", "nnb", "nnf", "market_pnl"):
            if col in df.columns:
                out[col] = float(df[col].sum())
        return out
    if agg_name == "by_month":
        if "month_end" not in df.columns:
            return {}
        return df.groupby("month_end", as_index=False).sum().to_dict(orient="list") if not df.empty else {}
    if agg_name == "raw":
        return df
    cache_specs.validate_agg_name(agg_name)
    return {"agg_name": agg_name, "row_count": len(df)}


def _aggregate_impl(
    dataset_version: str,
    agg_name: str,
    query_name: str,
    filter_state_hash: str,
    filter_state_json: str,
    root_str: str | None,
    agg_params_json: str,
) -> dict[str, Any] | pd.DataFrame:
    """Inner implementation: hashable args for cache."""
    df = get_filtered(
        dataset_version, query_name, filter_state_hash, filter_state_json, root_str
    )
    agg_params: dict[str, Any] = {}
    if agg_params_json:
        try:
            import json as _json
            agg_params = _json.loads(agg_params_json)
        except Exception:
            pass
    return _compute_aggregate(agg_name, df, agg_params or None)


if st is not None:
    @st.cache_data(show_spinner=False, ttl=_TTL_MEDIUM)
    def get_aggregate_medium(
        dataset_version: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any] | pd.DataFrame:
        """Level B: cached aggregates (medium TTL)."""
        return _aggregate_impl(
            dataset_version, agg_name, query_name,
            filter_state_hash, filter_state_json, root_str, agg_params_json
        )

    def get_aggregate(
        dataset_version: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any] | pd.DataFrame:
        """Level B: route through medium TTL; record cache debug and obs."""
        from app.cache.cache_debug import record_cache_call
        from app.observability.debug_panel import register_cache_call as obs_register
        root = _root_from_str(root_str)
        policy = None
        try:
            from app.cache.cache_gateway import load_cache_policy
            policy = load_cache_policy(root=root)
        except Exception:
            pass
        ttl = _get_ttl_seconds("medium", policy)
        t0 = time.perf_counter()
        result = get_aggregate_medium(
            dataset_version, agg_name, query_name,
            filter_state_hash, filter_state_json, root_str, agg_params_json
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        size = len(result) if isinstance(result, pd.DataFrame) else len(str(result))
        record_cache_call("B", agg_name, dataset_version, filter_state_hash, elapsed_ms, size, ttl_seconds=ttl)
        obs_register(
            "B", agg_name, dataset_version, filter_state_hash, elapsed_ms,
            size if isinstance(result, pd.DataFrame) else None,
            extra_key=agg_params_json or "", backend="",
        )
        return result
else:
    def get_aggregate(
        dataset_version: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any] | pd.DataFrame:
        return _aggregate_impl(
            dataset_version, agg_name, query_name,
            filter_state_hash, filter_state_json, root_str, agg_params_json
        )


# --- Level C: Heavy chart payloads ---

def _build_chart_payload(chart_name: str, agg_result: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Produce chart-specific payload from aggregate result. Uses app.cache.specs when registered."""
    from app.cache import specs as cache_specs
    if chart_name in cache_specs.CHART_SPECS:
        return cache_specs.CHART_SPECS[chart_name]["fn"](agg_result)
    if chart_name == "waterfall_inputs":
        if isinstance(agg_result, dict):
            return {"type": "waterfall_inputs", "data": agg_result}
        return {"type": "waterfall_inputs", "data": agg_result.to_dict(orient="list") if not agg_result.empty else {}}
    if chart_name == "corr_matrix":
        if isinstance(agg_result, pd.DataFrame):
            numeric = agg_result.select_dtypes(include=["number"])
            corr = numeric.corr() if len(numeric.columns) > 0 else {}
            return {"type": "corr_matrix", "data": corr.to_dict() if hasattr(corr, "to_dict") else {}}
        return {"type": "corr_matrix", "data": {}}
    if chart_name == "aum_line":
        if isinstance(agg_result, pd.DataFrame) and "month_end" in agg_result.columns and "end_aum" in agg_result.columns:
            return {"type": "aum_line", "data": agg_result[["month_end", "end_aum"]].to_dict(orient="list")}
        return {"type": "aum_line", "data": {}}
    cache_specs.validate_chart_name(chart_name)
    return {"type": chart_name, "data": {}}


if st is not None:
    @st.cache_data(show_spinner=False, ttl=_TTL_HEAVY)
    def get_chart_payload_heavy(
        dataset_version: str,
        chart_name: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any]:
        """Level C: cached chart payloads (heavy TTL)."""
        agg_result = get_aggregate(
            dataset_version, agg_name, query_name, filter_state_hash, filter_state_json, root_str, agg_params_json
        )
        return _build_chart_payload(chart_name, agg_result)

    def get_chart_payload(
        dataset_version: str,
        chart_name: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any]:
        """Level C: route through heavy TTL; record cache debug and obs."""
        from app.cache.cache_debug import record_cache_call
        from app.observability.debug_panel import register_cache_call as obs_register
        root = _root_from_str(root_str)
        policy = None
        try:
            from app.cache.cache_gateway import load_cache_policy
            policy = load_cache_policy(root=root)
        except Exception:
            pass
        ttl = _get_ttl_seconds("heavy", policy)
        t0 = time.perf_counter()
        result = get_chart_payload_heavy(
            dataset_version, chart_name, agg_name, query_name,
            filter_state_hash, filter_state_json, root_str, agg_params_json
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        size = len(str(result))
        record_cache_call("C", chart_name, dataset_version, filter_state_hash, elapsed_ms, size, ttl_seconds=ttl)
        obs_register("C", chart_name, dataset_version, filter_state_hash, elapsed_ms, None, extra_key=f"{agg_name}:{query_name}", backend="")
        try:
            from app.perf_budget import check_perf_budget, load_perf_budget
            policy = load_perf_budget(root=root)
            check_perf_budget("C", chart_name, elapsed_ms, config=policy, root=root)
        except RuntimeError:
            raise
        except Exception:
            pass
        return result
else:
    def get_chart_payload(
        dataset_version: str,
        chart_name: str,
        agg_name: str,
        query_name: str,
        filter_state_hash: str,
        filter_state_json: str,
        root_str: str | None = None,
        agg_params_json: str = "",
    ) -> dict[str, Any]:
        agg_result = get_aggregate(
            dataset_version, agg_name, query_name, filter_state_hash, filter_state_json, root_str, agg_params_json
        )
        return _build_chart_payload(chart_name, agg_result)
