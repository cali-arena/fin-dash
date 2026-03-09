"""
Unit tests for app.guardrails.no_adhoc_agg: groupby/merge raise when guardrail is active.
"""
import os

import pandas as pd
import pytest

from app.guardrails.no_adhoc_agg import (
    ADHOC_AGG_MESSAGE,
    ban_adhoc_agg,
    is_strict_agg_only,
)


def test_ban_adhoc_agg_groupby_raises_when_active() -> None:
    """When ban_adhoc_agg() is active, DataFrame.groupby raises RuntimeError with the required message."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    with ban_adhoc_agg():
        with pytest.raises(RuntimeError) as exc_info:
            df.groupby("a").sum()
    assert ADHOC_AGG_MESSAGE in str(exc_info.value)
    assert "agg tables" in str(exc_info.value).lower() and "pipelines/agg" in str(exc_info.value)


def test_ban_adhoc_agg_merge_raises_when_active() -> None:
    """When ban_adhoc_agg() is active, DataFrame.merge raises RuntimeError."""
    df1 = pd.DataFrame({"k": [1, 2], "v": [10, 20]})
    df2 = pd.DataFrame({"k": [1, 2], "w": [100, 200]})
    with ban_adhoc_agg():
        with pytest.raises(RuntimeError) as exc_info:
            df1.merge(df2, on="k")
    assert ADHOC_AGG_MESSAGE in str(exc_info.value)


def test_ban_adhoc_agg_restores_after_exit() -> None:
    """After exiting the context, groupby works again."""
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    with ban_adhoc_agg():
        with pytest.raises(RuntimeError):
            df.groupby("a").sum()
    # Must not raise
    out = df.groupby("a", as_index=False).sum()
    assert len(out) == 2
    assert out["b"].tolist() == [30, 30]


def test_is_strict_agg_only_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """STRICT_AGG_ONLY=1 in env -> is_strict_agg_only() is True."""
    monkeypatch.delenv("STRICT_AGG_ONLY", raising=False)
    assert is_strict_agg_only() is False
    monkeypatch.setenv("STRICT_AGG_ONLY", "1")
    assert is_strict_agg_only() is True
    monkeypatch.setenv("STRICT_AGG_ONLY", "0")
    assert is_strict_agg_only() is False
