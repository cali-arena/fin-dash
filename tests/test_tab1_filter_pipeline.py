from __future__ import annotations

import pandas as pd

from app.viz.tab1_filter_pipeline import (
    LABEL_ALL,
    TAB1_FILTER_SPECS,
    VALUE_UNASSIGNED,
    build_canonical_filter_frame,
    build_cascaded_options,
    canonicalize_lookup_frame,
    validate_selections,
)


def test_runtime_builder_keeps_real_values_and_ignores_blank_nan() -> None:
    frames = {
        "ticker_monthly": pd.DataFrame(
            {
                "product_ticker": ["AGG", " ", None, "nan", "TLT"],
                "sales_focus": ["ETF", "ETF", None, "", "Rates"],
                "country": ["US", "US", "US", "US", "US"],
                "channel_group": ["Institutional", "Institutional", "Institutional", "Institutional", "Institutional"],
                "sub_channel": ["Bank", "Bank", "Bank", "Bank", "Bank"],
                "sub_segment": ["FI Core", "FI Core", "FI Core", "FI Core", "FI Core"],
            }
        )
    }
    runtime_df = build_canonical_filter_frame(frames, TAB1_FILTER_SPECS)
    tickers = sorted(runtime_df["product_ticker"].dropna().unique().tolist())
    assert tickers == ["AGG", "TLT"]


def test_unassigned_appears_only_when_explicit_in_runtime() -> None:
    frames = {
        "ticker_monthly": pd.DataFrame(
            {
                "product_ticker": ["AGG", VALUE_UNASSIGNED],
                "sales_focus": ["ETF", "ETF"],
                "country": ["US", "US"],
                "channel_group": ["Institutional", "Institutional"],
                "sub_channel": ["Bank", "Bank"],
                "sub_segment": ["FI Core", "FI Core"],
            }
        )
    }
    runtime_df = build_canonical_filter_frame(frames, TAB1_FILTER_SPECS)
    lookup_df = canonicalize_lookup_frame(pd.DataFrame(), TAB1_FILTER_SPECS)
    selections = {spec.session_key: LABEL_ALL for spec in TAB1_FILTER_SPECS}
    options, _ = build_cascaded_options(runtime_df, lookup_df, selections, TAB1_FILTER_SPECS)
    assert VALUE_UNASSIGNED in options["tab1_filter_ticker"]


def test_exclude_self_cascade_uses_other_active_filters() -> None:
    runtime_df = pd.DataFrame(
        {
            "channel_group": ["Institutional", "Institutional", "Broker Dealer"],
            "sub_channel": ["Bank", "Wirehouse", "Wirehouse"],
            "country": ["US", "US", "US"],
            "sub_segment": ["FI Core", "FI Core", "FI Core"],
            "sales_focus": ["ETF", "ETF", "ETF"],
            "product_ticker": ["AGG", "TLT", "MUB"],
        }
    )
    lookup_df = pd.DataFrame(columns=runtime_df.columns)
    selections = {spec.session_key: LABEL_ALL for spec in TAB1_FILTER_SPECS}
    selections["tab1_filter_sub_channel"] = "Wirehouse"
    options, _ = build_cascaded_options(runtime_df, lookup_df, selections, TAB1_FILTER_SPECS)
    # Channel options must be narrowed by sub-channel filter (exclude-self).
    assert options["tab1_filter_channel"] == ["Broker Dealer", "Institutional"]
    # Ticker options must be narrowed by sub-channel filter too.
    assert options["tab1_filter_ticker"] == ["MUB", "TLT"]


def test_validator_resets_stale_selection_to_all() -> None:
    selections = {
        "tab1_filter_channel": "Stale value",
        "tab1_filter_sub_channel": LABEL_ALL,
    }
    options = {
        "tab1_filter_channel": ["Institutional"],
        "tab1_filter_sub_channel": ["Bank"],
    }
    healed = validate_selections(selections, options)
    assert healed["tab1_filter_channel"] == LABEL_ALL
    assert healed["tab1_filter_sub_channel"] == LABEL_ALL
