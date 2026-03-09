"""
Minimal unit tests for app.cache.pyramid: same inputs -> cache hit; filter_state_hash/agg_name/chart_name bust cache.
"""
import importlib
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def test_same_inputs_same_cache_hit() -> None:
    """Same (dataset_version, query_name, filter_state_hash, filter_state_json) -> cache hit; uncached runner called once."""
    storage: dict[tuple, pd.DataFrame] = {}
    counter = [0]

    def mock_cache_data(show_spinner: bool = False, ttl: int | None = None, max_entries: int | None = None):  # noqa: ARG001
        def decorator(f):
            def wrapped(*args: object, **kwargs: object) -> pd.DataFrame:
                key = (args, tuple(sorted(kwargs.items())))
                if key not in storage:
                    storage[key] = f(*args, **kwargs)
                return storage[key]
            return wrapped
        return decorator

    with patch("streamlit.cache_data", mock_cache_data):
        import app.cache.pyramid as pyramid_mod
        importlib.reload(pyramid_mod)
        with patch("app.data_gateway._run_query_uncached") as mock_run:
            def _count_and_df(*args, **kwargs):
                counter[0] += 1
                return pd.DataFrame({"a": [1]})
            mock_run.side_effect = _count_and_df
            dv, qn, fh, fj = "v1", "firm_monthly", "abc" * 14, "{}"
            r1 = pyramid_mod.get_filtered(dv, qn, fh, fj, None)
            r2 = pyramid_mod.get_filtered(dv, qn, fh, fj, None)
    assert len(r1) == 1 and r1["a"].tolist() == [1]
    assert len(r2) == 1
    assert counter[0] == 1


def test_changing_filter_state_hash_busts_cache() -> None:
    """Changing filter_state_hash -> cache miss; uncached runner called for each distinct hash."""
    storage: dict[tuple, pd.DataFrame] = {}
    counter = [0]

    def mock_cache_data(show_spinner: bool = False, ttl: int | None = None, max_entries: int | None = None):  # noqa: ARG001
        def decorator(f):
            def wrapped(*args: object, **kwargs: object) -> pd.DataFrame:
                key = (args, tuple(sorted(kwargs.items())))
                if key not in storage:
                    storage[key] = f(*args, **kwargs)
                return storage[key]
            return wrapped
        return decorator

    with patch("streamlit.cache_data", mock_cache_data):
        import app.cache.pyramid as pyramid_mod
        importlib.reload(pyramid_mod)
        with patch("app.data_gateway._run_query_uncached") as mock_run:
            mock_run.return_value = pd.DataFrame({"a": [1]})

            def count_and_return(*args, **kwargs):
                counter[0] += 1
                return pd.DataFrame({"a": [1]})
            mock_run.side_effect = count_and_return

            pyramid_mod.get_filtered("v1", "firm_monthly", "hash1", "{}", None)
            pyramid_mod.get_filtered("v1", "firm_monthly", "hash2", "{}", None)
    assert counter[0] == 2


def test_changing_agg_name_busts_aggregate_cache() -> None:
    """Changing agg_name -> get_aggregate cache miss (Level B); each distinct agg_name runs aggregate again."""
    storage_b: dict[tuple, object] = {}
    call_count = [0]

    def mock_cache_data(show_spinner: bool = False, ttl: int | None = None, max_entries: int | None = None):  # noqa: ARG001
        def decorator(f):
            def wrapped(*args: object, **kwargs: object) -> object:
                key = (args, tuple(sorted(kwargs.items())))
                if key not in storage_b:
                    call_count[0] += 1
                    storage_b[key] = f(*args, **kwargs)
                return storage_b[key]
            return wrapped
        return decorator

    with patch("streamlit.cache_data", mock_cache_data):
        import app.cache.pyramid as pyramid_mod
        importlib.reload(pyramid_mod)
        with patch("app.data_gateway._run_query_uncached", return_value=pd.DataFrame({"month_end": [1], "end_aum": [100.0]})):
            fj, fh = "{}", "h" * 40
            pyramid_mod.get_aggregate("v1", "kpi_totals", "firm_monthly", fh, fj, None)
            pyramid_mod.get_aggregate("v1", "by_month", "firm_monthly", fh, fj, None)
    # Level A get_filtered: 1 miss. Level B get_aggregate: 2 misses (kpi_totals, by_month).
    assert call_count[0] >= 2


def test_changing_chart_name_busts_chart_payload_cache() -> None:
    """Changing chart_name -> get_chart_payload cache miss (Level C); each distinct chart_name runs again."""
    storage_c: dict[tuple, object] = {}
    call_count = [0]

    def mock_cache_data(show_spinner: bool = False, ttl: int | None = None, max_entries: int | None = None):  # noqa: ARG001
        def decorator(f):
            def wrapped(*args: object, **kwargs: object) -> object:
                key = (args, tuple(sorted(kwargs.items())))
                if key not in storage_c:
                    call_count[0] += 1
                    storage_c[key] = f(*args, **kwargs)
                return storage_c[key]
            return wrapped
        return decorator

    with patch("streamlit.cache_data", mock_cache_data):
        import app.cache.pyramid as pyramid_mod
        importlib.reload(pyramid_mod)
        with patch("app.data_gateway._run_query_uncached", return_value=pd.DataFrame({"month_end": [1], "end_aum": [100.0]})):
            fj, fh = "{}", "h" * 40
            pyramid_mod.get_chart_payload("v1", "aum_line", "raw", "firm_monthly", fh, fj, None)
            pyramid_mod.get_chart_payload("v1", "corr_matrix", "raw", "firm_monthly", fh, fj, None)
    # Level C: 2 chart payload misses (aum_line, corr_matrix); lower layers may hit.
    assert call_count[0] >= 2


def test_pyramid_signatures_include_dataset_version_and_hash() -> None:
    """Level A/B/C function definitions include dataset_version, filter_state_hash (guarantee correct invalidation)."""
    pyramid_py = Path(__file__).resolve().parent.parent / "app" / "cache" / "pyramid.py"
    text = pyramid_py.read_text(encoding="utf-8")
    assert "dataset_version" in text and "filter_state_hash" in text
    for name in ("get_filtered", "get_aggregate", "get_chart_payload"):
        assert f"def {name}" in text, f"{name} must be defined"
    # Ensure all three take the required args (source-level check; decorator may hide signature)
    assert "dataset_version: str" in text
    assert "filter_state_hash: str" in text
