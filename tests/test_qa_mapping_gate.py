"""
Unit tests for qa/mapping_gate: deterministic quality gate for unmapped ratio > 1%.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import after path fix if needed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.mapping_gate import run, UNMAPPED_RATIO_THRESHOLD


def test_gate_skip_no_meta(tmp_path: Path) -> None:
    ok, msg = run(tmp_path)
    assert ok is True
    assert "unmapped_keys.meta.json" in msg or "skipped" in msg.lower()


def test_gate_pass_under_threshold(tmp_path: Path) -> None:
    meta = {"total_raw_rows": 10000, "unmapped_rows": 50, "unmapped_keys": 5}
    (tmp_path / "unmapped_keys.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ok, msg = run(tmp_path)
    assert ok is True
    assert "0.50%" in msg or "50" in msg


def test_gate_fail_over_threshold(tmp_path: Path) -> None:
    # 2% unmapped > 1%
    meta = {"total_raw_rows": 1000, "unmapped_rows": 20, "unmapped_keys": 3}
    (tmp_path / "unmapped_keys.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ok, msg = run(tmp_path)
    assert ok is False
    assert "QA FAIL" in msg or "2.00%" in msg
    assert str(UNMAPPED_RATIO_THRESHOLD) in msg or "1%" in msg


def test_gate_pass_exactly_at_threshold(tmp_path: Path) -> None:
    meta = {"total_raw_rows": 1000, "unmapped_rows": 10, "unmapped_keys": 1}
    (tmp_path / "unmapped_keys.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    ok, msg = run(tmp_path)
    assert ok is True
