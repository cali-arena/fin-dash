"""
Single state model for all tabs. FilterState + DrillState + stable hashing.
All pages must read/write filters only via get_filter_state / set_filter_state / update_filter_state.
Drill state: read via get_drill_state; write only via update_drill_state (or set_drill_state).
No other module should mutate st.session_state[DRILL_STATE_KEY] directly—this is the single writer path.
"""
from __future__ import annotations

import calendar
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:
    import streamlit as st
except ImportError:
    st = None

SESSION_KEY = "filter_state"

# Enum allowed values (from filters.yml)
PERIOD_MODES = ("1M", "QoQ", "YTD", "YoY")
CHANNEL_VIEWS = ("raw", "standard", "best", "canonical")
GEO_DIMS = ("src_country", "product_country")
PRODUCT_DIMS = ("ticker", "segment", "sub_segment")

DEFAULT_CURRENCY = "native"
DEFAULT_UNIT = "units"
DEFAULT_DRILL_PATH = ["channel", "geo", "product"]

# Drill state: single source of truth for drill-down selection (ranked table → selectbox → details)
DrillMode = Literal["channel", "ticker"]
SelectionSource = Literal["table", "widget", "url", "chart"]
DetailsLevel = Literal["firm", "selected"]

DRILL_STATE_KEY = "drill_state"
FILTER_HASH_KEY = "filter_state_hash"
DRILL_RESET_FLAG_KEY = "drill_selection_reset"  # boolean or message when selection was reset


@dataclass
class DrillState:
    """Deterministic drill state: drill_mode, selected channel/ticker, source, details_level. JSON-serializable via to_dict()."""
    drill_mode: DrillMode = "channel"
    selected_channel: Optional[str] = None
    selected_ticker: Optional[str] = None
    selection_source: Optional[SelectionSource] = None
    details_level: DetailsLevel = "selected"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (for session storage and serialization)."""
        return asdict(self)

    def to_json(self) -> str:
        """Canonical JSON with sorted keys."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def drill_state_hash(self) -> str:
        """SHA1 of canonical JSON for deterministic keys."""
        return hashlib.sha1(self.to_json().encode("utf-8")).hexdigest()

    def hash(self) -> str:
        """Alias for drill_state_hash (sha1 of canonical json)."""
        return self.drill_state_hash()

    @classmethod
    def from_dict(cls, d: Dict[str, Any] | None) -> DrillState:
        """Build DrillState from dict (e.g. from session); missing keys use defaults. Stores as dict => JSON-serializable."""
        if not d:
            return cls()
        src = d.get("selection_source")
        if src == "chart":
            src = "widget"  # backward compat
        if src is not None and src not in ("table", "widget", "url", "chart"):
            src = None
        return cls(
            drill_mode=d.get("drill_mode", "channel"),
            selected_channel=d.get("selected_channel"),
            selected_ticker=d.get("selected_ticker"),
            selection_source=src,
            details_level=d.get("details_level", "selected"),
        )


def get_drill_state() -> DrillState:
    """Return current drill state; init default and store if not in session. Stored value is always a dict (JSON-serializable)."""
    if st is None:
        return DrillState()
    if DRILL_STATE_KEY not in st.session_state:
        default = DrillState()
        st.session_state[DRILL_STATE_KEY] = default.to_dict()
        return default
    val = st.session_state[DRILL_STATE_KEY]
    if isinstance(val, dict):
        return DrillState.from_dict(val)
    if isinstance(val, DrillState):
        return val
    return DrillState()


def debug_drill_state() -> list[str]:
    """
    Lightweight checks on current drill state (no pytest). Returns list of issue messages; empty if OK.
    Checks: drill_mode in ("channel","ticker"); channel mode => selected_ticker is None;
    ticker mode => selected_channel is None; details_level in ("firm","selected");
    selection_source in ("table","widget","url") or None.
    """
    issues: list[str] = []
    d = get_drill_state()
    if d.drill_mode not in ("channel", "ticker"):
        issues.append(f"invalid drill_mode: {d.drill_mode!r}")
    if d.drill_mode == "channel" and d.selected_ticker is not None:
        issues.append("drill_mode=channel but selected_ticker is set")
    if d.drill_mode == "ticker" and d.selected_channel is not None:
        issues.append("drill_mode=ticker but selected_channel is set")
    if d.details_level not in ("firm", "selected"):
        issues.append(f"invalid details_level: {d.details_level!r}")
    if d.selection_source is not None and d.selection_source not in ("table", "widget", "url", "chart"):
        issues.append(f"invalid selection_source: {d.selection_source!r}")
    raw = st.session_state.get(DRILL_STATE_KEY) if st else None
    if raw is not None and not isinstance(raw, dict):
        issues.append(f"session_state[{DRILL_STATE_KEY!r}] is not a dict (not JSON-serializable)")
    return issues


def set_drill_state(state: DrillState) -> None:
    """Store drill state in session as dict. Prefer update_drill_state for writes."""
    if st is not None:
        st.session_state[DRILL_STATE_KEY] = state.to_dict()


def normalize_drill_state(state: DrillState) -> DrillState:
    """Apply invariants deterministically. Returns a new DrillState."""
    # details_level=="firm" => clear selections and source
    if state.details_level == "firm":
        state = DrillState(
            drill_mode=state.drill_mode,
            selected_channel=None,
            selected_ticker=None,
            selection_source=None,
            details_level="firm",
        )
    # drill_mode=="channel" => no ticker
    if state.drill_mode == "channel":
        state = DrillState(
            drill_mode=state.drill_mode,
            selected_channel=state.selected_channel,
            selected_ticker=None,
            selection_source=state.selection_source,
            details_level=state.details_level,
        )
    # drill_mode=="ticker" => no channel
    if state.drill_mode == "ticker":
        state = DrillState(
            drill_mode=state.drill_mode,
            selected_channel=None,
            selected_ticker=state.selected_ticker,
            selection_source=state.selection_source,
            details_level=state.details_level,
        )
    # If a selection exists, set details_level="selected" (unless we're explicitly firm)
    if state.details_level != "firm" and (
        state.selected_channel is not None or state.selected_ticker is not None
    ):
        state = DrillState(
            drill_mode=state.drill_mode,
            selected_channel=state.selected_channel,
            selected_ticker=state.selected_ticker,
            selection_source=state.selection_source,
            details_level="selected",
        )
    return state


def update_drill_state(
    *,
    drill_mode: Optional[str] = None,
    selected_channel: Optional[str] = None,
    selected_ticker: Optional[str] = None,
    selection_source: Optional[str] = None,
    details_level: Optional[str] = None,
) -> DrillState:
    """Single writer for drill state: load, apply only provided fields, normalize, persist, return."""
    current = get_drill_state()
    d = current.to_dict()
    if drill_mode is not None:
        d["drill_mode"] = drill_mode
    if selected_channel is not None:
        d["selected_channel"] = selected_channel
    if selected_ticker is not None:
        d["selected_ticker"] = selected_ticker
    if selection_source is not None:
        d["selection_source"] = selection_source
    if details_level is not None:
        d["details_level"] = details_level
    next_state = DrillState.from_dict(d)
    next_state = normalize_drill_state(next_state)
    set_drill_state(next_state)
    return next_state


def set_drill_mode(mode: Literal["channel", "ticker"]) -> None:
    """Set drill mode and enforce: channel mode clears ticker; ticker mode clears channel."""
    current = get_drill_state()
    if mode == "channel":
        next_state = normalize_drill_state(
            DrillState(
                drill_mode="channel",
                selected_channel=current.selected_channel,
                selected_ticker=None,
                selection_source=current.selection_source,
                details_level=current.details_level,
            )
        )
    else:
        next_state = normalize_drill_state(
            DrillState(
                drill_mode="ticker",
                selected_channel=None,
                selected_ticker=current.selected_ticker,
                selection_source=current.selection_source,
                details_level=current.details_level,
            )
        )
    set_drill_state(next_state)


def set_selected_channel(value: str | None, source: SelectionSource = "widget") -> None:
    """Set selected channel; clears ticker when in channel mode. Stores as dict in session."""
    current = get_drill_state()
    next_state = normalize_drill_state(
        DrillState(
            drill_mode="channel",
            selected_channel=value,
            selected_ticker=None,
            selection_source=source,
            details_level="selected" if value else current.details_level,
        )
    )
    set_drill_state(next_state)


def set_selected_ticker(value: str | None, source: SelectionSource = "widget") -> None:
    """Set selected ticker; clears channel when in ticker mode. Stores as dict in session."""
    current = get_drill_state()
    next_state = normalize_drill_state(
        DrillState(
            drill_mode="ticker",
            selected_channel=None,
            selected_ticker=value,
            selection_source=source,
            details_level="selected" if value else current.details_level,
        )
    )
    set_drill_state(next_state)


def validate_drill_selection(
    valid_channels: set[str],
    valid_tickers: set[str],
) -> tuple[bool, Optional[str]]:
    """
    Validate current drill selection against valid sets. Auto-clears invalid selection and persists.
    Returns (True, None) if valid; (False, "selection reset") if selection was cleared.
    Rules: channel mode => selected_ticker None; ticker mode => selected_channel None;
    selected value must be in valid set.
    """
    current = get_drill_state()
    reset = False
    if current.drill_mode == "channel":
        if current.selected_ticker is not None:
            next_state = normalize_drill_state(
                DrillState(
                    drill_mode="channel",
                    selected_channel=current.selected_channel,
                    selected_ticker=None,
                    selection_source=None,
                    details_level="firm",
                )
            )
            set_drill_state(next_state)
            reset = True
        elif (
            current.selected_channel is not None
            and valid_channels
            and current.selected_channel not in valid_channels
        ):
            next_state = normalize_drill_state(
                DrillState(
                    drill_mode="channel",
                    selected_channel=None,
                    selected_ticker=None,
                    selection_source=None,
                    details_level="firm",
                )
            )
            set_drill_state(next_state)
            reset = True
    elif current.drill_mode == "ticker":
        if current.selected_channel is not None:
            next_state = normalize_drill_state(
                DrillState(
                    drill_mode="ticker",
                    selected_channel=None,
                    selected_ticker=current.selected_ticker,
                    selection_source=None,
                    details_level="firm",
                )
            )
            set_drill_state(next_state)
            reset = True
        elif (
            current.selected_ticker is not None
            and valid_tickers
            and current.selected_ticker not in valid_tickers
        ):
            next_state = normalize_drill_state(
                DrillState(
                    drill_mode="ticker",
                    selected_channel=None,
                    selected_ticker=None,
                    selection_source=None,
                    details_level="firm",
                )
            )
            set_drill_state(next_state)
            reset = True
    if reset:
        return (False, "selection reset")
    return (True, None)


def _validate_drill_against_available_sets(
    drill: DrillState,
    *,
    available_channels: Optional[set[str]] = None,
    available_tickers: Optional[set[str]] = None,
) -> tuple[DrillState, Optional[str]]:
    """Re-validate drill selection against available channel/ticker sets. Returns (state, reset_message or None)."""
    reset_message: Optional[str] = None
    state = drill
    if (
        state.drill_mode == "channel"
        and available_channels is not None
        and state.selected_channel is not None
        and state.selected_channel not in available_channels
    ):
        state = normalize_drill_state(
            DrillState(
                drill_mode=state.drill_mode,
                selected_channel=None,
                selected_ticker=None,
                selection_source=None,
                details_level="firm",
            )
        )
        reset_message = "Selection reset due to filter change (no longer available in current slice)."
    elif (
        state.drill_mode == "ticker"
        and available_tickers is not None
        and state.selected_ticker is not None
        and state.selected_ticker not in available_tickers
    ):
        state = normalize_drill_state(
            DrillState(
                drill_mode=state.drill_mode,
                selected_channel=None,
                selected_ticker=None,
                selection_source=None,
                details_level="firm",
            )
        )
        reset_message = "Selection reset due to filter change (no longer available in current slice)."
    return (state, reset_message)


def revalidate_drill_on_filter_change(
    new_filter_hash: str,
    *,
    available_channels: Optional[set[str]] = None,
    available_tickers: Optional[set[str]] = None,
) -> DrillState:
    """When filter hash changes, re-validate drill selection; set DRILL_RESET_FLAG_KEY if reset. No UI calls."""
    old_hash = st.session_state.get(FILTER_HASH_KEY) if st else None
    if old_hash != new_filter_hash:
        drill = get_drill_state()
        validated, reset_message = _validate_drill_against_available_sets(
            drill,
            available_channels=available_channels,
            available_tickers=available_tickers,
        )
        if st is not None:
            if reset_message is not None:
                st.session_state[DRILL_RESET_FLAG_KEY] = reset_message
            else:
                st.session_state[DRILL_RESET_FLAG_KEY] = None
            st.session_state[FILTER_HASH_KEY] = new_filter_hash
        set_drill_state(validated)
        return validated
    return get_drill_state()


def _default_date_end() -> str:
    """Last day of current month (month_end boundary)."""
    today = date.today()
    _, last_day = calendar.monthrange(today.year, today.month)
    end_d = date(today.year, today.month, last_day)
    return end_d.isoformat()


def _default_date_start() -> str:
    """Month-end 12 months before default date_end (last 12 months)."""
    today = date.today()
    _, last_day = calendar.monthrange(today.year, today.month)
    end_d = date(today.year, today.month, last_day)
    # 12 months back
    year_s, month_s = end_d.year, end_d.month - 12
    if month_s <= 0:
        month_s += 12
        year_s -= 1
    _, last_s = calendar.monthrange(year_s, month_s)
    start_d = date(year_s, month_s, last_s)
    return start_d.isoformat()


def _norm_enum(value: Any, allowed: tuple[str, ...], default: str) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    s = str(value).strip()
    return s if s in allowed else default


@dataclass
class FilterState:
    date_start: str
    date_end: str
    period_mode: str
    channel_view: str
    geo_dim: str
    product_dim: str
    custodian_firm: str | None
    drill_path: list[str]
    slice_dim: str | None
    slice_value: str | None
    currency: str | None
    unit: str | None

    def to_dict(self) -> dict[str, Any]:
        """Stable, JSON-safe dict with sorted keys for deterministic hashing."""
        return {
            "channel_view": self.channel_view,
            "currency": self.currency,
            "custodian_firm": self.custodian_firm,
            "date_end": self.date_end,
            "date_start": self.date_start,
            "drill_path": list(self.drill_path),
            "geo_dim": self.geo_dim,
            "period_mode": self.period_mode,
            "product_dim": self.product_dim,
            "slice_dim": self.slice_dim,
            "slice_value": self.slice_value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> FilterState:
        """From dict with defaults + validation."""
        if not d:
            d = {}
        date_start = str(d.get("date_start") or _default_date_start())
        date_end = str(d.get("date_end") or _default_date_end())
        period_mode = _norm_enum(d.get("period_mode"), PERIOD_MODES, "1M")
        channel_view = _norm_enum(d.get("channel_view"), CHANNEL_VIEWS, "canonical")
        geo_dim = _norm_enum(d.get("geo_dim"), GEO_DIMS, "src_country")
        product_dim = _norm_enum(d.get("product_dim"), PRODUCT_DIMS, "ticker")
        custodian_firm = d.get("custodian_firm")
        custodian_firm = str(custodian_firm).strip() if custodian_firm else None
        drill_path = d.get("drill_path")
        if drill_path is None:
            drill_path = list(DEFAULT_DRILL_PATH)
        else:
            drill_path = [str(x) for x in drill_path]
        slice_dim = d.get("slice_dim")
        slice_dim = str(slice_dim).strip() if slice_dim else None
        slice_value = d.get("slice_value") or d.get("slice")
        slice_value = str(slice_value).strip() if slice_value else None
        currency = d.get("currency", DEFAULT_CURRENCY)
        currency = str(currency) if currency else None
        unit = d.get("unit", DEFAULT_UNIT)
        unit = str(unit) if unit else None

        state = cls(
            date_start=date_start,
            date_end=date_end,
            period_mode=period_mode,
            channel_view=channel_view,
            geo_dim=geo_dim,
            product_dim=product_dim,
            custodian_firm=custodian_firm,
            drill_path=drill_path,
            slice_dim=slice_dim,
            slice_value=slice_value,
            currency=currency,
            unit=unit,
        )
        state._validate()
        return state

    def _validate(self) -> None:
        if self.date_start > self.date_end:
            raise ValueError(
                f"date_start ({self.date_start}) must be <= date_end ({self.date_end})"
            )
        if not self.drill_path:
            raise ValueError("drill_path must be non-empty")
        if self.period_mode not in PERIOD_MODES:
            raise ValueError(f"period_mode must be one of {PERIOD_MODES}")
        if self.channel_view not in CHANNEL_VIEWS:
            raise ValueError(f"channel_view must be one of {CHANNEL_VIEWS}")
        if self.geo_dim not in GEO_DIMS:
            raise ValueError(f"geo_dim must be one of {GEO_DIMS}")
        if self.product_dim not in PRODUCT_DIMS:
            raise ValueError(f"product_dim must be one of {PRODUCT_DIMS}")

    def canonical_json(self) -> str:
        """Sorted keys, no whitespace differences."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def filter_state_hash(self) -> str:
        """SHA1 of canonical_json for repeatable cache keys."""
        return hashlib.sha1(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def slice(self) -> str | None:
        """Backward compat: alias for slice_value."""
        return self.slice_value


def get_filter_state() -> FilterState:
    """Get FilterState from session; create default if absent."""
    if st is None:
        return FilterState.from_dict({})
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = FilterState.from_dict({})
    val = st.session_state[SESSION_KEY]
    if isinstance(val, dict):
        st.session_state[SESSION_KEY] = FilterState.from_dict(val)
        val = st.session_state[SESSION_KEY]
    return val


def set_filter_state(state: FilterState) -> None:
    """Save FilterState to session."""
    if st is not None:
        st.session_state[SESSION_KEY] = state


def update_filter_state(**kwargs: Any) -> FilterState:
    """Merge kwargs into current state, validate enums and date_start <= date_end, save. Returns new FilterState."""
    current = get_filter_state()
    d = current.to_dict()
    for k, v in kwargs.items():
        if k in d:
            d[k] = v
        if k == "slice":
            d["slice_value"] = str(v).strip() if v else None
    new_state = FilterState.from_dict(d)
    set_filter_state(new_state)
    return new_state


def filter_state_to_gateway_dict(state: FilterState) -> dict[str, Any]:
    """
    Convert FilterState to the dict shape expected by run_query (month_end_range, etc.).
    Uses date_start/date_end for month_end_range; pass through for caching keys.
    """
    import pandas as pd
    return {
        "month_end_range": (
            pd.Timestamp(state.date_start),
            pd.Timestamp(state.date_end),
        ),
    }


def _parse_month_end_values(values: list[str]) -> list[str]:
    """Normalize month_end strings to sorted ISO YYYY-MM-DD."""
    import pandas as pd

    if not values:
        return []
    dt = pd.to_datetime(pd.Series(values, dtype="string"), errors="coerce")
    dt = dt.dropna().sort_values().drop_duplicates()
    return [pd.Timestamp(v).strftime("%Y-%m-%d") for v in dt.tolist()]


def _candidate_date_windows(month_ends: list[str]) -> list[tuple[str, str]]:
    """
    Candidate windows ordered by preference:
    1) latest 12 months (if possible)
    2) latest available full range
    3) latest single month -> earliest single month (nearest valid fallback)
    """
    if not month_ends:
        return []
    out: list[tuple[str, str]] = []
    end = month_ends[-1]
    if len(month_ends) >= 12:
        out.append((month_ends[-12], end))
    out.append((month_ends[0], end))
    # Scan recent months first to find nearest valid startup state without probing an excessive history.
    for month in reversed(month_ends[-36:]):
        out.append((month, month))
    # De-duplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for w in out:
        if w not in seen:
            seen.add(w)
            ordered.append(w)
    return ordered


def _state_has_rows(state: FilterState, root: Path) -> bool:
    """True when the state produces rows in at least one core startup dataset."""
    try:
        from app.data_gateway import (
            get_channel_breakdown,
            get_firm_snapshot,
            get_growth_quality,
            get_trend_series,
        )

        snapshot = get_firm_snapshot(state, root=root)
        if snapshot is not None and not snapshot.empty:
            return True
        trend = get_trend_series(state, root=root)
        if trend is not None and not trend.empty:
            return True
        channel = get_channel_breakdown(state, metric="end_aum", root=root)
        if channel is not None and not channel.empty:
            return True
        ticker = get_growth_quality(state, view="ticker", root=root)
        if ticker is not None and not ticker.empty:
            return True
    except Exception:
        return False
    return False


def get_dataset_date_bounds(root: str | Path | None = None) -> tuple[date | None, date | None]:
    """
    Return (min_date, max_date) from the dataset month_end range, or (None, None) if unavailable.
    Use for date picker min_value/max_value so the UI only shows valid dates.
    """
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        from app.data_gateway import DataGateway

        gw = DataGateway(root_path)
        month_ends = _parse_month_end_values(gw.list_month_ends(None, view_name="v_firm_monthly", limit=None))
    except Exception:
        return (None, None)
    if not month_ends:
        return (None, None)
    try:
        min_d = date.fromisoformat(month_ends[0][:10])
        max_d = date.fromisoformat(month_ends[-1][:10])
        return (min_d, max_d)
    except (ValueError, TypeError):
        return (None, None)


def resolve_best_default_filters(root: str | Path | None = None) -> FilterState:
    """
    Resolve curated startup defaults from available data.
    Preference:
    1) latest available 12 months
    2) latest available full range
    3) nearest valid single-month fallback
    """
    base = FilterState.from_dict({})
    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    try:
        from app.data_gateway import DataGateway

        gw = DataGateway(root_path)
        month_ends = _parse_month_end_values(gw.list_month_ends(base, view_name="v_firm_monthly", limit=None))
    except Exception:
        return base

    if not month_ends:
        return base

    base_dict = base.to_dict()
    # Startup should not apply hidden slices/custodian filters.
    base_dict["slice_dim"] = None
    base_dict["slice_value"] = None
    base_dict["custodian_firm"] = None

    for start_iso, end_iso in _candidate_date_windows(month_ends):
        candidate = FilterState.from_dict(
            {
                **base_dict,
                "date_start": start_iso,
                "date_end": end_iso,
            }
        )
        if _state_has_rows(candidate, root_path):
            return candidate

    # Deterministic fallback: full available range.
    return FilterState.from_dict(
        {
            **base_dict,
            "date_start": month_ends[0],
            "date_end": month_ends[-1],
        }
    )
