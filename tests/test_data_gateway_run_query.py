"""
Minimal unit tests for data_gateway run_query API: hash_filters stability, query_name validation, cache wrapper signature.
Canonical filter hashing: same dict different ordering => same hash; list order => same hash; date objects normalize.
"""
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data_gateway import (
    ALLOWED_QUERIES,
    Q_FIRM_MONTHLY,
    QUERY_SPECS,
    canonicalize,
    to_json_safe,
    cached_run_query,
    hash_filters,
    load_dataset_version,
    run_query,
    _run_query_uncached,
    _use_duckdb_views,
)
from app.state import DrillState, FilterState


def test_hash_filters_stability() -> None:
    """Same logical filter state with different key order and list order yields same hash."""
    a = {"z": 1, "a": 2, "channel": ["B", "A", "C"]}
    b = {"a": 2, "channel": ["C", "A", "B"], "z": 1}
    assert hash_filters(a) == hash_filters(b)
    assert len(hash_filters(a)) == 40  # sha1 hex


def test_same_dict_different_ordering_same_hash() -> None:
    """Same dict with different key ordering => same hash."""
    a = {"z": 3, "a": 1, "m": 2}
    b = {"a": 1, "m": 2, "z": 3}
    assert hash_filters(a) == hash_filters(b)


def test_list_order_differences_same_hash() -> None:
    """Selector list order differences => same hash (lists sorted in canonical form)."""
    a = {"channel": ["B", "A", "C"]}
    b = {"channel": ["C", "A", "B"]}
    assert hash_filters(a) == hash_filters(b)
    assert canonicalize(a)["channel"] == ["A", "B", "C"]


def test_date_objects_normalize() -> None:
    """date/datetime objects normalize to ISO YYYY-MM-DD; same logical date => same hash."""
    d = date(2024, 3, 1)
    dt = datetime(2024, 3, 1, 12, 30, 0)
    ts = pd.Timestamp("2024-03-01")
    assert canonicalize({"d": d})["d"] == "2024-03-01"
    assert canonicalize({"dt": dt})["dt"] == "2024-03-01"
    assert canonicalize({"ts": ts})["ts"] == "2024-03-01"
    assert hash_filters({"x": d}) == hash_filters({"x": "2024-03-01"})
    assert hash_filters({"x": dt}) == hash_filters({"x": "2024-03-01"})


def test_hash_filters_accepts_filter_state_object() -> None:
    """hash_filters must accept FilterState directly without JSON serialization errors."""
    fs = FilterState.from_dict(
        {
            "date_start": "2024-01-31",
            "date_end": "2024-12-31",
            "period_mode": "1M",
            "channel_view": "canonical",
            "geo_dim": "src_country",
            "product_dim": "ticker",
        }
    )
    h1 = hash_filters(fs)
    h2 = hash_filters(fs.to_dict())
    assert isinstance(h1, str) and len(h1) == 40
    assert isinstance(h2, str) and len(h2) == 40


def test_hash_filters_accepts_drill_state_dataclass_in_dict() -> None:
    """Nested dataclass-like state objects should serialize deterministically."""
    payload = {"drill": DrillState(drill_mode="channel", selected_channel="Institutional")}
    h1 = hash_filters(payload)
    h2 = hash_filters(payload)
    assert h1 == h2


def test_hash_filters_unsupported_type_has_clear_error() -> None:
    """Unsupported objects should raise actionable error message."""
    class _Unsupported:
        pass

    with pytest.raises(TypeError) as exc_info:
        hash_filters({"bad": _Unsupported()})
    assert "Unsupported state type for hashing" in str(exc_info.value) or "State serialization failed" in str(exc_info.value)


def test_to_json_safe_normalizes_datetime_path_and_nested() -> None:
    """to_json_safe recursively normalizes dates, paths and nested containers."""
    payload = {
        "when": datetime(2024, 3, 1, 12, 30),
        "path": Path("abc/def"),
        "items": [date(2024, 3, 2), {"x": pd.Timestamp("2024-03-03")}],
    }
    out = to_json_safe(payload)
    assert out["when"] == "2024-03-01"
    assert out["path"] == str(Path("abc/def"))
    assert out["items"][0] == "2024-03-02"
    assert out["items"][1]["x"] == "2024-03-03"


def test_run_query_governed_normalizes_state_like_object() -> None:
    """Governed query path should normalize any state-like object, not rely on strict isinstance(FilterState)."""
    class LegacyFilterState:
        def to_dict(self):
            return {
                "date_start": "2024-01-31",
                "date_end": "2024-02-29",
                "period_mode": "1M",
                "channel_view": "canonical",
                "geo_dim": "src_country",
                "product_dim": "ticker",
            }

    with patch("app.data_gateway.run_governed_query") as rgq:
        rgq.return_value = {"ok": True}
        out = run_query("kpi_firm_global", LegacyFilterState(), root=None)
        assert out == {"ok": True}
        args, _kwargs = rgq.call_args
        assert args[0] == "kpi_firm_global"
        assert isinstance(args[1], FilterState)


def test_query_name_validation() -> None:
    """Invalid query_name raises ValueError with allowed set."""
    with pytest.raises(ValueError) as exc_info:
        _run_query_uncached("invalid_query", {}, root=None)
    assert "invalid_query" in str(exc_info.value).lower()
    assert "allowed" in str(exc_info.value).lower() or "allowed" in str(exc_info.value)


def test_query_name_allowed() -> None:
    """ALLOWED_QUERIES contains expected constants."""
    assert Q_FIRM_MONTHLY in ALLOWED_QUERIES
    assert "firm_monthly" in ALLOWED_QUERIES
    assert "invalid" not in ALLOWED_QUERIES


def test_cached_run_query_signature_includes_dataset_version_query_name_filter_hash() -> None:
    """cached_run_query is keyed by dataset_version, query_name, filter_state_hash (and hashable filter_state_json)."""
    import inspect
    sig = inspect.signature(cached_run_query)
    params = list(sig.parameters)
    assert "dataset_version" in params
    assert "query_name" in params
    assert "filter_state_hash" in params
    assert "filter_state_json" in params


def test_cached_run_query_returns_dataframe(tmp_path: Path) -> None:
    """cached_run_query returns DataFrame when _run_query_uncached is mocked."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"dataset_version": "v1"}', encoding="utf-8"
    )
    with patch("app.data_gateway._run_query_uncached", return_value=pd.DataFrame({"a": [1]})):
        with patch("app.data_gateway._cached_run_query_impl") as mock_impl:
            mock_impl.return_value = pd.DataFrame({"a": [1]})
            result = cached_run_query(
                "v1", Q_FIRM_MONTHLY, "abc" * 14, "{}", str(tmp_path)
            )
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1]


def test_load_dataset_version_missing_raises(tmp_path: Path) -> None:
    """load_dataset_version raises FileNotFoundError when meta file is missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset_version(root=tmp_path)
    assert "metrics_monthly.meta.json" in str(exc_info.value).lower() or "curated" in str(exc_info.value).lower()


def test_load_dataset_version_missing_key_raises(tmp_path: Path) -> None:
    """load_dataset_version raises ValueError when dataset_version key is missing."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"other_key": "x"}', encoding="utf-8"
    )
    with pytest.raises(ValueError) as exc_info:
        load_dataset_version(root=tmp_path)
    assert "dataset_version" in str(exc_info.value).lower()


def test_use_duckdb_views_true_when_manifest_present_and_reads_views_only(tmp_path: Path) -> None:
    """When analytics/duckdb_views_manifest.json exists and reads_views_only true, _use_duckdb_views is True."""
    (tmp_path / "analytics").mkdir(parents=True)
    (tmp_path / "analytics" / "duckdb_views_manifest.json").write_text(
        '{"db_path": "analytics/db.duckdb", "schema": "analytics", "reads_views_only": true}',
        encoding="utf-8",
    )
    assert _use_duckdb_views(tmp_path) is True


def test_use_duckdb_views_false_when_manifest_absent(tmp_path: Path) -> None:
    """When analytics/duckdb_views_manifest.json does not exist, _use_duckdb_views is False."""
    assert _use_duckdb_views(tmp_path) is False


def test_use_duckdb_views_false_when_manifest_exists_but_reads_views_only_false(tmp_path: Path) -> None:
    """When views manifest exists but reads_views_only is false, _use_duckdb_views is False."""
    (tmp_path / "analytics").mkdir(parents=True)
    (tmp_path / "analytics" / "duckdb_views_manifest.json").write_text(
        '{"db_path": "analytics/db.duckdb", "schema": "analytics", "reads_views_only": false}',
        encoding="utf-8",
    )
    assert _use_duckdb_views(tmp_path) is False


def test_query_specs_mapping() -> None:
    """QUERY_SPECS has correct view and parquet_table for each monthly query."""
    for name in (Q_FIRM_MONTHLY, "channel_monthly", "ticker_monthly", "geo_monthly", "segment_monthly"):
        spec = QUERY_SPECS.get(name)
        assert spec is not None, name
        assert "view" in spec and "parquet_table" in spec
        assert spec["view"] == f"v_{name}" or spec["view"] is None
        assert spec.get("allowed_filters") is not None


def test_main_uses_run_query_only_no_load_table() -> None:
    """Streamlit pages must use run_query only; main should not reference load_table in code."""
    main_py = Path(__file__).resolve().parent.parent / "app" / "main.py"
    text = main_py.read_text(encoding="utf-8")
    lines_with_load_table = [ln for ln in text.splitlines() if "load_table" in ln]
    for ln in lines_with_load_table:
        assert ln.strip().startswith("#"), "main must not use load_table; use run_query only"
    assert "run_query" in text
    assert "from app.data_gateway import" in text


def test_run_query_parquet_fallback_when_no_views_manifest(tmp_path: Path) -> None:
    """When DuckDB views manifest is absent, run_query uses Parquet fallback and returns DataFrame."""
    (tmp_path / "curated").mkdir(parents=True)
    (tmp_path / "curated" / "metrics_monthly.meta.json").write_text(
        '{"dataset_version": "v1"}', encoding="utf-8"
    )
    (tmp_path / "agg").mkdir(parents=True)
    (tmp_path / "agg" / "manifest.json").write_text(
        '{"dataset_version": "v1", "tables": [{"name": "firm_monthly", "path": "agg/firm_monthly.parquet"}]}',
        encoding="utf-8",
    )
    df_agg = pd.DataFrame({
        "month_end": [pd.Timestamp("2024-01-01")],
        "begin_aum": [90.0], "end_aum": [100.0], "nnb": [10.0], "nnf": [0.5], "market_pnl": [5.0],
    })
    df_agg.to_parquet(tmp_path / "agg" / "firm_monthly.parquet", index=False)
    result = run_query(Q_FIRM_MONTHLY, {}, root=tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "month_end" in result.columns and "end_aum" in result.columns
