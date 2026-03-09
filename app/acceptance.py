"""
Acceptance checks: warm-cache performance budget (per-tab < 1000 ms).
Runs in dev when APP_DEV_ACCEPTANCE=1. One representative set of gateway calls per tab.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

DEV_ACCEPTANCE = os.environ.get("APP_DEV_ACCEPTANCE", "1") == "1"
BUDGET_MS_PER_TAB = 1000


def _run_tab_calls(
    calls: list[tuple[str, Callable[[], Any]]],
) -> tuple[float, list[dict[str, Any]]]:
    """Execute each (name, thunk), return (total_ms, list of {name, ms})."""
    total_ms = 0.0
    results: list[dict[str, Any]] = []
    for name, thunk in calls:
        t0 = time.perf_counter()
        try:
            thunk()
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_ms += elapsed_ms
        results.append({"name": name, "ms": round(elapsed_ms, 2)})
    return total_ms, results


def run_acceptance_checks(gw: Any, default_state: Any) -> dict[str, Any]:
    """
    Execute representative gateway calls per tab: cold run then warm run.
    Assert per-tab warm total < BUDGET_MS_PER_TAB (1000 ms).
    Returns { "passed": bool, "tabs": { tab_id: { "total_ms", "calls": [{ name, ms }] } }, "failures": [str] }.
    """
    state = default_state
    tabs_config: dict[str, list[tuple[str, Callable[[], Any]]]] = {
        "visualisations": [
            ("kpi_firm_global", lambda: gw.kpi_firm_global(state)),
            ("chart_aum_trend", lambda: gw.chart_aum_trend(state)),
            ("chart_nnb_trend", lambda: gw.chart_nnb_trend(state)),
            ("growth_decomposition_inputs", lambda: gw.growth_decomposition_inputs(state)),
        ],
        "dynamic_report": [
            ("top_channels", lambda: gw.top_channels(state, n=10)),
            ("top_movers", lambda: gw.top_movers(state, n=10)),
            ("notable_months", lambda: gw.notable_months(state)),
            ("coverage_stats", lambda: gw.coverage_stats(state)),
        ],
        "nlq_chat": [
            ("chart_aum_trend", lambda: gw.chart_aum_trend(state)),
            ("top_movers", lambda: gw.top_movers(state, n=10)),
            ("top_channels", lambda: gw.top_channels(state, n=10)),
        ],
    }

    # Cold run (fill cache)
    for tab_id, calls in tabs_config.items():
        for _name, thunk in calls:
            try:
                thunk()
            except Exception:
                pass

    # Warm run: measure per tab
    outcome: dict[str, Any] = {
        "passed": True,
        "tabs": {},
        "failures": [],
    }
    for tab_id, calls in tabs_config.items():
        total_ms, call_results = _run_tab_calls(calls)
        outcome["tabs"][tab_id] = {"total_ms": round(total_ms, 2), "calls": call_results}
        if total_ms >= BUDGET_MS_PER_TAB:
            outcome["passed"] = False
            outcome["failures"].append(
                f"{tab_id}: total {outcome['tabs'][tab_id]['total_ms']} ms > {BUDGET_MS_PER_TAB} ms budget"
            )

    return outcome


def render_acceptance_failure_panel(result: dict[str, Any]) -> None:
    """Show Streamlit warning panel with which call/tab exceeded budget and timings."""
    try:
        import streamlit as st
    except ImportError:
        return
    if result.get("passed", True):
        return
    with st.expander("Acceptance checks: FAIL — performance budget exceeded", expanded=True):
        st.warning("Per-tab warm run total must be < 1000 ms. The following exceeded budget:")
        for f in result.get("failures", []):
            st.text(f"  • {f}")
        st.markdown("**Timings (warm run)**")
        for tab_id, data in result.get("tabs", {}).items():
            total = data.get("total_ms", 0)
            over = " ⚠ over budget" if total >= BUDGET_MS_PER_TAB else ""
            st.text(f"  {tab_id}: {total} ms total{over}")
            for c in data.get("calls", []):
                st.text(f"    — {c.get('name', '')}: {c.get('ms', 0)} ms")
        st.caption("Fix: reduce query cost or increase cache hit rate so warm run stays under 1000 ms per tab.")
