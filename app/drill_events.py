"""
Standardized event parsing for chart clicks → drill selection.
Pure functions only (no Streamlit). Caller applies returned state and runs validate_drill_selection.
"""
from __future__ import annotations

from typing import Any

from app.state import DrillState


def _clean_label(value: Any) -> str | None:
    """Return stripped non-empty string or None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def parse_treemap_click(events: list[dict]) -> str | None:
    """
    From first event dict, read label in priority order: label, pointLabel, hovertext, text.
    Return cleaned string or None.
    """
    if not events or not isinstance(events[0], dict):
        return None
    d = events[0]
    for key in ("label", "pointLabel", "hovertext", "text"):
        if key in d:
            out = _clean_label(d[key])
            if out is not None:
                return out
    return None


def parse_scatter_click(events: list[dict]) -> dict[str, str | None]:
    """
    Return {"ticker": str|None, "channel": str|None}.
    Prefer customdata: list [ticker, channel] or dict {"ticker":..,"channel":..}.
    Fallback only when explicit (no guess from x/y/text).
    """
    out: dict[str, str | None] = {"ticker": None, "channel": None}
    if not events or not isinstance(events[0], dict):
        return out
    d = events[0]
    custom = d.get("customdata")
    if custom is not None:
        if isinstance(custom, (list, tuple)):
            if len(custom) >= 1 and custom[0] is not None:
                out["ticker"] = _clean_label(custom[0])
            if len(custom) >= 2 and custom[1] is not None:
                out["channel"] = _clean_label(custom[1])
        elif isinstance(custom, dict):
            if "ticker" in custom:
                out["ticker"] = _clean_label(custom["ticker"])
            if "channel" in custom:
                out["channel"] = _clean_label(custom["channel"])
    return out


def apply_chart_selection(
    drill_state: DrillState,
    payload: dict[str, str | None],
    source: str = "chart",
) -> DrillState:
    """
    Compute next DrillState from chart click payload. Additive; does not mutate.
    - If payload has non-empty ticker -> mode=ticker, selected_ticker=value, clear channel.
    - Else if payload has non-empty channel -> mode=channel, selected_channel=value, clear ticker.
    - Else return drill_state unchanged.
    Caller must persist and run validate_drill_selection so invalid selections are cleared.
    """
    ticker = payload.get("ticker") if isinstance(payload.get("ticker"), str) else None
    ticker = _clean_label(ticker) if ticker is not None else None
    channel = payload.get("channel") if isinstance(payload.get("channel"), str) else None
    channel = _clean_label(channel) if channel is not None else None

    if ticker:
        return DrillState(
            drill_mode="ticker",
            selected_channel=None,
            selected_ticker=ticker,
            selection_source=source,
            details_level="selected",
        )
    if channel:
        return DrillState(
            drill_mode="channel",
            selected_channel=channel,
            selected_ticker=None,
            selection_source=source,
            details_level="selected",
        )
    return drill_state
