"""
Pytest coverage for XLSX fingerprint generator (src.versioning.fingerprint).
Uses tmp_path; fast and deterministic.
"""
import hashlib
import subprocess
from pathlib import Path

import pytest

from legacy.legacy_src.versioning.fingerprint import (
    compute_dataset_version,
    get_dataset_version,
    resolve_pipeline_version,
)


@pytest.fixture
def xlsx_path(tmp_path: Path) -> Path:
    """data/input/source.xlsx with fixed bytes."""
    inp = tmp_path / "data" / "input"
    inp.mkdir(parents=True)
    path = inp / "source.xlsx"
    path.write_bytes(b"PK\x03\x04dummy_xlsx_content")
    return path


# --- A) Stable: same inputs -> same dataset_version ---


def test_stable_same_inputs_same_dataset_version(xlsx_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_VERSION", "v1")
    d1 = get_dataset_version(xlsx_path)
    d2 = get_dataset_version(xlsx_path)
    assert d1["dataset_version"] == d2["dataset_version"]
    assert d1["pipeline_version"] == "v1"
    assert len(d1["dataset_version"]) == 64


# --- B) Input change: 1 byte -> dataset_version changes ---


def test_one_byte_change_changes_dataset_version(xlsx_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_VERSION", "v1")
    d_before = get_dataset_version(xlsx_path)
    xlsx_path.write_bytes(b"PK\x03\x04dummy_xlsx_contentX")  # one byte added
    d_after = get_dataset_version(xlsx_path)
    assert d_after["dataset_version"] != d_before["dataset_version"]


# --- C) Pipeline change: same file, different PIPELINE_VERSION -> dataset_version changes ---


def test_pipeline_version_change_changes_dataset_version(
    xlsx_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIPELINE_VERSION", "v1")
    d1 = get_dataset_version(xlsx_path)
    monkeypatch.setenv("PIPELINE_VERSION", "v2")
    d2 = get_dataset_version(xlsx_path)
    assert d1["dataset_version"] != d2["dataset_version"]
    assert d1["pipeline_version"] == "v1"
    assert d2["pipeline_version"] == "v2"


# --- D) Delimiter enforced: exactly b"\\n" between bytes and pipeline_version ---


def test_compute_dataset_version_uses_newline_delimiter() -> None:
    excel_bytes = b"abc"
    pipeline_version = "v1"
    expected_payload = excel_bytes + b"\n" + pipeline_version.encode("utf-8")
    expected = hashlib.sha256(expected_payload).hexdigest()
    assert compute_dataset_version(excel_bytes, pipeline_version) == expected


def test_compute_dataset_version_delimiter_regression() -> None:
    """No other delimiter (e.g. space or double newline) is used."""
    excel_bytes = b"x"
    pv = "p"
    # Correct: b"x" + b"\n" + b"p" -> sha256(b"x\np")
    correct = hashlib.sha256(b"x\np").hexdigest()
    assert compute_dataset_version(excel_bytes, pv) == correct
    # Wrong alternatives must not match
    assert compute_dataset_version(excel_bytes, pv) != hashlib.sha256(b"x p").hexdigest()
    assert compute_dataset_version(excel_bytes, pv) != hashlib.sha256(b"x\n\np").hexdigest()


# --- E) Git fallback does not crash ---


def test_resolve_pipeline_version_git_fallback_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When PIPELINE_VERSION unset and get_git_sha returns None -> "unknown"."""
    monkeypatch.delenv("PIPELINE_VERSION", raising=False)
    monkeypatch.setattr("src.versioning.git_info.get_git_sha", lambda: None)
    result = resolve_pipeline_version()
    assert result == "unknown"


def test_resolve_pipeline_version_git_fallback_called_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When get_git_sha internally catches CalledProcessError, still returns "unknown"."""
    monkeypatch.delenv("PIPELINE_VERSION", raising=False)
    monkeypatch.setattr(
        "src.versioning.git_info.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["git", "rev-parse", "HEAD"])
        ),
    )
    result = resolve_pipeline_version()
    assert result == "unknown"