"""
Pytest: version file persistence and schema correctness (ensure_version_file).
Uses tmp_path; deterministic; no real git repo required.
"""
import hashlib
from pathlib import Path

import pytest

from legacy.legacy_src.versioning.ensure_version import ensure_version_file

REQUIRED_KEYS = (
    "created_at",
    "dataset_version",
    "git_dirty",
    "pipeline_version",
    "source_file",
    "source_sha256",
    "source_size_bytes",
)


@pytest.fixture
def version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_VERSION", "v1")


@pytest.fixture
def source_and_version_path(tmp_path: Path) -> tuple[Path, Path]:
    """data/input/source.xlsx with b'abc' and data/.version.json path."""
    inp = tmp_path / "data" / "input"
    inp.mkdir(parents=True)
    xlsx = inp / "source.xlsx"
    xlsx.write_bytes(b"abc")
    version_path = tmp_path / "data" / ".version.json"
    return xlsx, version_path


# --- 1) Schema keys exist ---


def test_schema_keys_exist(
    source_and_version_path: tuple[Path, Path],
    version_env: None,
) -> None:
    source_path, version_path = source_and_version_path
    manifest = ensure_version_file(str(source_path), str(version_path))

    for key in REQUIRED_KEYS:
        assert key in manifest, f"missing key: {key}"

    assert manifest["source_size_bytes"] == 3
    assert manifest["source_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert manifest["created_at"].endswith("Z")


# --- 2) Deterministic rewrite rule ---


def test_deterministic_rewrite_no_change(
    source_and_version_path: tuple[Path, Path],
    version_env: None,
) -> None:
    source_path, version_path = source_and_version_path
    ensure_version_file(str(source_path), str(version_path))
    content1 = version_path.read_text()

    ensure_version_file(str(source_path), str(version_path))
    content2 = version_path.read_text()

    assert content1 == content2


# --- 3) Change detection ---


def test_change_detection_rewrites(
    source_and_version_path: tuple[Path, Path],
    version_env: None,
) -> None:
    source_path, version_path = source_and_version_path
    m1 = ensure_version_file(str(source_path), str(version_path))
    dataset_v1 = m1["dataset_version"]

    source_path.write_bytes(b"abd")  # 1 byte changed
    m2 = ensure_version_file(str(source_path), str(version_path))
    dataset_v2 = m2["dataset_version"]

    assert dataset_v2 != dataset_v1


# --- 4) Atomic write: no temp files left ---


def test_atomic_write_leaves_no_temp_files(
    source_and_version_path: tuple[Path, Path],
    version_env: None,
) -> None:
    source_path, version_path = source_and_version_path
    ensure_version_file(str(source_path), str(version_path))

    data_dir = version_path.parent
    temp_files = list(data_dir.glob("*.tmp.*"))
    assert temp_files == [], f"temp files left: {temp_files}"


# --- 5) git_dirty optional ---


def test_git_dirty_optional_none(
    source_and_version_path: tuple[Path, Path],
    version_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.versioning.ensure_version.is_git_dirty", lambda: None)
    source_path, version_path = source_and_version_path
    manifest = ensure_version_file(str(source_path), str(version_path))
    assert manifest["git_dirty"] is None
