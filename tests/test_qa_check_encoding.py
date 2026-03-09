"""
Tests for qa/check_encoding: UTF-8 validation and problematic character detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qa.check_encoding import (
    check_python_weird_chars,
    check_utf8,
    main,
    ROOT as ENCODING_ROOT,
)


def test_check_utf8_valid(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    assert check_utf8(tmp_path / "ok.py") == []


def test_check_utf8_invalid(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_bytes(b"x = \xff\xfe\n")
    errs = check_utf8(tmp_path / "bad.py")
    assert len(errs) == 1
    assert "invalid UTF-8" in errs[0][1] or "utf-8" in errs[0][1].lower()


def test_check_python_weird_chars_clean() -> None:
    assert check_python_weird_chars(Path("x.py"), "a = 1\n") == []


def test_check_python_weird_chars_replacement_char() -> None:
    errs = check_python_weird_chars(Path("x.py"), "a = 1\ufffd\n")
    assert len(errs) >= 1
    assert "problematic" in errs[0][1] or "replacement" in errs[0][1].lower()


def test_main_returns_zero_on_clean_repo() -> None:
    # Assumes repo has no encoding violations (run after fixes)
    assert ENCODING_ROOT.is_dir()
    # main(argv=[]) so pytest's sys.argv doesn't get parsed
    result = main(argv=[])
    assert result == 0, "Repo should have no encoding violations; run python qa/check_encoding.py"
