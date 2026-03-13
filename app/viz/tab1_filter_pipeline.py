from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

LABEL_ALL = "All"
VALUE_UNASSIGNED = "Unassigned"
# Values excluded from dropdown options (case-insensitive)
FILTER_OPTION_EXCLUDE = frozenset({"", "unassigned", "—", "nan", "none", "null"})


@dataclass(frozen=True)
class Tab1FilterSpec:
    label: str
    session_key: str
    source_column: str
    aliases: tuple[str, ...]


TAB1_FILTER_SPECS: tuple[Tab1FilterSpec, ...] = (
    Tab1FilterSpec(
        label="Channel (grouped)",
        session_key="tab1_filter_channel",
        source_column="channel_group",
        aliases=("channel_group", "channel", "channel_final", "channel_standard"),
    ),
    Tab1FilterSpec(
        label="Sub-Channel (standard)",
        session_key="tab1_filter_sub_channel",
        source_column="sub_channel",
        aliases=("channel_final", "channel", "sub_channel"),
    ),
    Tab1FilterSpec(
        label="Country",
        session_key="tab1_filter_country",
        source_column="country",
        aliases=("country", "src_country", "geo"),
    ),
    Tab1FilterSpec(
        label="Sub-Segment",
        session_key="tab1_filter_sub_segment",
        source_column="sub_segment",
        aliases=("sub_segment", "segment"),
    ),
    Tab1FilterSpec(
        label="Sales Focus",
        session_key="tab1_filter_sales_focus",
        source_column="sales_focus",
        aliases=("sales_focus", "uswa_sales_focus_2020"),
    ),
    Tab1FilterSpec(
        label="Product Ticker",
        session_key="tab1_filter_ticker",
        source_column="product_ticker",
        aliases=("product_ticker", "ticker"),
    ),
)


def _normalize_dim_value(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if pd.isna(v):
        return None
    s = str(v).strip()
    if not s:
        return None
    low = s.lower()
    if low in FILTER_OPTION_EXCLUDE or low == "nat":
        return None
    if s in {"—", "â€”", "-", "–"}:
        return None
    return s


def _has_real_values(series: pd.Series) -> bool:
    """True if series has at least one value that is not a placeholder after normalization."""
    for v in series.dropna().astype(str).str.strip():
        if v and v.lower() not in FILTER_OPTION_EXCLUDE:
            return True
    return False


def _extract_first_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    for alias in aliases:
        if alias in df.columns:
            return df[alias].map(_normalize_dim_value)
    return pd.Series([None] * len(df), index=df.index, dtype="object")


def _extract_best_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> pd.Series:
    """Use first alias that yields at least one real (non-placeholder) value; else merge all aliases."""
    best: pd.Series | None = None
    for alias in aliases:
        if alias not in df.columns:
            continue
        candidate = df[alias].map(_normalize_dim_value)
        filled = candidate.dropna().astype(str).str.strip()
        if filled.empty:
            if best is None:
                best = candidate
            else:
                best = best.fillna(candidate)
            continue
        if _has_real_values(filled):
            return candidate
        if best is None:
            best = candidate
        else:
            best = best.fillna(candidate)
    return best if best is not None else pd.Series([None] * len(df), index=df.index, dtype="object")


def build_canonical_filter_frame(
    frames: dict[str, pd.DataFrame],
    specs: tuple[Tab1FilterSpec, ...] = TAB1_FILTER_SPECS,
) -> pd.DataFrame:
    canonical_cols = [spec.source_column for spec in specs]
    parts: list[pd.DataFrame] = []
    for frame in frames.values():
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        out = pd.DataFrame(index=frame.index)
        for spec in specs:
            out[spec.source_column] = _extract_best_alias(frame, spec.aliases)
        parts.append(out[canonical_cols].drop_duplicates())
    if not parts:
        return pd.DataFrame(columns=canonical_cols)
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def canonicalize_lookup_frame(
    lookup_df: pd.DataFrame,
    specs: tuple[Tab1FilterSpec, ...] = TAB1_FILTER_SPECS,
) -> pd.DataFrame:
    canonical_cols = [spec.source_column for spec in specs]
    if lookup_df is None or lookup_df.empty:
        return pd.DataFrame(columns=canonical_cols)
    out = pd.DataFrame(index=lookup_df.index)
    for spec in specs:
        aliases = (spec.source_column,) + tuple(a for a in spec.aliases if a != spec.source_column)
        out[spec.source_column] = _extract_first_alias(lookup_df, aliases)
    return out[canonical_cols].drop_duplicates().reset_index(drop=True)


def _exclude_self_mask(
    df: pd.DataFrame,
    selections_by_col: dict[str, str],
    target_col: str,
) -> pd.Series:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, selected in selections_by_col.items():
        if col == target_col or selected in (None, "", LABEL_ALL):
            continue
        if col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        mask = mask & (df[col] == selected)
    return mask


def _distinct_options(df: pd.DataFrame, col: str, mask: pd.Series | None = None) -> list[str]:
    if df.empty or col not in df.columns:
        return []
    src = df.loc[mask, col] if mask is not None else df[col]
    vals: list[str] = []
    for v in src.tolist():
        norm = _normalize_dim_value(v)
        if norm is None:
            continue
        vals.append(norm)
    return sorted(dict.fromkeys(vals))


def build_cascaded_options(
    runtime_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    selections: dict[str, str],
    specs: tuple[Tab1FilterSpec, ...] = TAB1_FILTER_SPECS,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    options_by_session: dict[str, list[str]] = {}
    source_by_session: dict[str, str] = {}
    selections_by_col = {
        spec.source_column: selections.get(spec.session_key, LABEL_ALL)
        for spec in specs
    }
    for spec in specs:
        runtime_mask = _exclude_self_mask(runtime_df, selections_by_col, spec.source_column)
        runtime_opts = _distinct_options(runtime_df, spec.source_column, runtime_mask)
        if runtime_opts:
            options_by_session[spec.session_key] = runtime_opts
            source_by_session[spec.session_key] = "runtime"
            continue
        lookup_mask = _exclude_self_mask(lookup_df, selections_by_col, spec.source_column)
        lookup_opts = _distinct_options(lookup_df, spec.source_column, lookup_mask)
        options_by_session[spec.session_key] = lookup_opts
        source_by_session[spec.session_key] = "lookup" if lookup_opts else "none"
    return options_by_session, source_by_session


def validate_selections(
    selections: dict[str, str],
    options_by_session: dict[str, list[str]],
) -> dict[str, str]:
    healed: dict[str, str] = {}
    for key, selected in selections.items():
        valid = set(options_by_session.get(key, []))
        healed[key] = selected if selected in valid or selected == LABEL_ALL else LABEL_ALL
    return healed
