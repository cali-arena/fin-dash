"""
Filter validation engine: auto-heal broken states so invalid queries are prevented.
Deterministic; no UI rendering. All healed state remains hash-stable (sorted lists, All -> None).
"""
from __future__ import annotations

from typing import Any

from app.filters_contract import is_optional_filter_enabled, load_filters_contract
from app.state import FilterState

try:
    import streamlit as st
except ImportError:
    st = None

SESSION_LAST_CHANNEL_VIEW = "_filters_validation_last_channel_view"

# Slice dims that are channel-specific (when channel_view changes, clear slice)
CHANNEL_SLICE_DIMS = frozenset({"channel", "channel_l1", "channel_l2", "preferred_label", "channel_raw", "channel_standard", "channel_best"})


def _normalize_all_to_none(val: Any) -> Any:
    """Normalize 'All' or empty list to None for deterministic state."""
    if val is None:
        return None
    if isinstance(val, list):
        if not val:
            return None
        if len(val) == 1 and str(val[0]).strip().lower() == "all":
            return None
        return sorted([str(x).strip() for x in val if str(x).strip()])
    if isinstance(val, str) and val.strip().lower() == "all":
        return None
    return val


def _snap_to_nearest(available: list[str], requested: str, prefer_ge: bool) -> str:
    """Snap requested (YYYY-MM-DD) to nearest available; prefer_ge=True -> nearest >= requested else nearest <= requested."""
    if not available:
        return requested
    req = requested.strip()[:10]
    if req in available:
        return req
    if prefer_ge:
        for m in available:
            if m >= req:
                return m
        return available[-1] if available else requested
    else:
        for m in reversed(available):
            if m <= req:
                return m
        return available[0] if available else requested


def validate_and_heal_filters(
    state: FilterState,
    gw: Any,
    contract_filters: dict[str, Any],
) -> tuple[FilterState, list[str], list[str]]:
    """
    Validate and auto-heal filter state. Returns (healed_state, warnings, infos).
    Deterministic; no UI. Uses gateway for list_geo_values, list_product_values,
    list_custodian_firms, list_month_ends, available_columns.
    """
    warnings: list[str] = []
    infos: list[str] = []
    d = state.to_dict()

    # ---- A) Validate selected values exist ----
    geo_values = getattr(state, "geo_values", None)
    if geo_values is not None:
        allowed_geo = gw.list_geo_values(state, limit=500) if hasattr(gw, "list_geo_values") else []
        allowed_set = set(str(x).strip() for x in allowed_geo)
        selected = [str(x).strip() for x in geo_values if str(x).strip()]
        kept = [x for x in selected if x in allowed_set]
        removed = set(selected) - set(kept)
        if removed:
            warnings.append(f"Geo selection contained values not in data; removed: {sorted(removed)}")
        if not kept:
            geo_values = None
            warnings.append("Geo selection was empty or invalid; reset to All.")
        else:
            geo_values = sorted(kept)

    product_values = getattr(state, "product_values", None)
    if product_values is not None:
        allowed_product = gw.list_product_values(state, limit=500) if hasattr(gw, "list_product_values") else []
        allowed_set = set(str(x).strip() for x in allowed_product)
        selected = [str(x).strip() for x in product_values if str(x).strip()]
        kept = [x for x in selected if x in allowed_set]
        removed = set(selected) - set(kept)
        if removed:
            warnings.append(f"Product selection contained values not in data; removed: {sorted(removed)}")
        if not kept:
            product_values = None
            warnings.append("Product selection was empty or invalid; reset to All.")
        else:
            product_values = sorted(kept)

    custodian_firm = d.get("custodian_firm")
    if custodian_firm and str(custodian_firm).strip():
        firms = gw.list_custodian_firms(state, limit=500) if hasattr(gw, "list_custodian_firms") else []
        allowed_set = set(str(x).strip() for x in firms)
        if str(custodian_firm).strip() not in allowed_set:
            d["custodian_firm"] = None
            warnings.append("Selected custodian firm not in data; cleared.")

    # ---- B) Date range alignment to available month_end ----
    month_ends = gw.list_month_ends(state, limit=None) if hasattr(gw, "list_month_ends") else []
    if month_ends:
        start_req = (d.get("date_start") or "").strip()[:10]
        end_req = (d.get("date_end") or "").strip()[:10]
        start_healed = _snap_to_nearest(month_ends, start_req, prefer_ge=True)
        end_healed = _snap_to_nearest(month_ends, end_req, prefer_ge=False)
        if start_healed != start_req or end_healed != end_req:
            d["date_start"] = start_healed
            d["date_end"] = end_healed
        if start_healed > end_healed:
            # Reset to last 12 months available
            if len(month_ends) >= 12:
                d["date_start"] = month_ends[-12]
                d["date_end"] = month_ends[-1]
            else:
                d["date_start"] = month_ends[0]
                d["date_end"] = month_ends[-1]
            warnings.append("Date range was invalid after snapping; reset to last 12 months available.")
    # If no month_ends, leave dates as-is (e.g. mock or empty DB).

    # ---- C) Channel view change: clear channel-specific slice ----
    last_channel_view: str | None = None
    if st is not None:
        last_channel_view = st.session_state.get(SESSION_LAST_CHANNEL_VIEW)
    current_channel_view = d.get("channel_view") or state.channel_view
    slice_dim = (d.get("slice_dim") or "").strip().lower() or None
    if last_channel_view is not None and last_channel_view != current_channel_view:
        if slice_dim and slice_dim in CHANNEL_SLICE_DIMS:
            d["slice_dim"] = None
            d["slice_value"] = None
            warnings.append("Channel view changed; cleared channel slice selection.")
    if st is not None:
        st.session_state[SESSION_LAST_CHANNEL_VIEW] = current_channel_view

    # ---- D) Custodian optional: remove if contract defines but dataset lacks column ----
    view_columns = gw.available_columns("v_firm_monthly") if hasattr(gw, "available_columns") else set()
    contract = contract_filters or load_filters_contract()
    custodian_defined = isinstance(contract.get("custodian_firm"), dict)
    custodian_enabled = is_optional_filter_enabled("custodian_firm", view_columns, contract) if contract else False
    if custodian_defined and not custodian_enabled and d.get("custodian_firm"):
        d["custodian_firm"] = None
        infos.append("Custodian filter removed; dataset has no custodian column.")

    # Build healed state
    healed = FilterState.from_dict(d)
    if geo_values is not None:
        setattr(healed, "geo_values", _normalize_all_to_none(geo_values))
    else:
        setattr(healed, "geo_values", None)
    if product_values is not None:
        setattr(healed, "product_values", _normalize_all_to_none(product_values))
    else:
        setattr(healed, "product_values", None)

    return healed, warnings, infos
