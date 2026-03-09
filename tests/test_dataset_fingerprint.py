"""
Pytest: dataset_version is stable for same inputs+pipeline, and changes when inputs or pipeline change.
"""
from pathlib import Path

import pytest

from app.versioning.dataset_fingerprint import (
    REQUIRED_INPUT_FILES,
    compute_dataset_version,
    read_input_bytes,
)


@pytest.fixture
def data_input_dir(tmp_path: Path) -> Path:
    """data/input with all 5 required CSV files (minimal content)."""
    inp = tmp_path / "data" / "input"
    inp.mkdir(parents=True)
    for name in REQUIRED_INPUT_FILES:
        (inp / name).write_bytes(b"a,b\n1,2")
    return inp


def test_same_inputs_same_pipeline_same_dataset_version(data_input_dir: Path) -> None:
    """Same inputs + same pipeline_version -> same dataset_version."""
    raw = read_input_bytes(str(data_input_dir))
    pv = "pipeline-v1"
    v1 = compute_dataset_version(raw, pv)
    v2 = compute_dataset_version(raw, pv)
    assert v1 == v2
    assert len(v1) == 64
    assert all(c in "0123456789abcdef" for c in v1)


def test_one_byte_change_changes_dataset_version(data_input_dir: Path) -> None:
    """Change 1 byte in DATA_RAW.csv -> dataset_version changes."""
    raw_before = read_input_bytes(str(data_input_dir))
    pv = "pipeline-v1"
    v_before = compute_dataset_version(raw_before, pv)

    raw_path = data_input_dir / "DATA_RAW.csv"
    raw_path.write_bytes(b"a,b\n9,2")  # one byte changed

    raw_after = read_input_bytes(str(data_input_dir))
    v_after = compute_dataset_version(raw_after, pv)
    assert v_after != v_before


def test_different_pipeline_version_changes_dataset_version(data_input_dir: Path) -> None:
    """Same inputs + different pipeline_version -> dataset_version changes."""
    raw = read_input_bytes(str(data_input_dir))
    v1 = compute_dataset_version(raw, "pipeline-v1")
    v2 = compute_dataset_version(raw, "pipeline-v2")
    assert v1 != v2


def test_missing_file_raises_clear_message(tmp_path: Path) -> None:
    """Missing required file -> raises FileNotFoundError with clear message."""
    inp = tmp_path / "data" / "input"
    inp.mkdir(parents=True)
    # Only create first two files
    for name in REQUIRED_INPUT_FILES[:2]:
        (inp / name).write_bytes(b"x,y\n1,2")

    with pytest.raises(FileNotFoundError) as exc_info:
        read_input_bytes(str(inp))

    err = str(exc_info.value)
    assert "Missing" in err or "missing" in err
    assert "DATA_MAPPING" in err or "ETF" in err or "EXECUTIVE_SUMMARY" in err
    assert "DATA_RAW" in err or "DATA_SUMMARY" in err or "Required" in err
