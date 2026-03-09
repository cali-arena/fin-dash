"""
Load and validate YAML schema shape. No pandas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_schema(path: str | Path) -> dict[str, Any]:
    """Load schema from YAML file. Raises on missing file or invalid YAML."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Schema file not found: {p.resolve()}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for load_yaml_schema. Install with: pip install pyyaml") from e
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Schema root must be a YAML mapping, got {type(data)}")
    return data


def validate_schema_shape(schema: dict[str, Any]) -> None:
    """
    Ensure required keys exist: required_columns, parsing_rules, validation_rules.
    Raises ValueError with explicit message if any missing.
    """
    required_keys = ("required_columns", "parsing_rules", "validation_rules")
    missing = [k for k in required_keys if k not in schema]
    if missing:
        raise ValueError(f"Schema missing required keys: {missing}. Required: {list(required_keys)}")
    if not isinstance(schema["required_columns"], list):
        raise ValueError("Schema 'required_columns' must be a list")
    if not isinstance(schema["parsing_rules"], dict):
        raise ValueError("Schema 'parsing_rules' must be a dict")
    if not isinstance(schema["validation_rules"], dict):
        raise ValueError("Schema 'validation_rules' must be a dict")


def format_schema_load_error(path: str | Path, err: Exception) -> str:
    """Human-readable message for schema load/shape failures (include path and message)."""
    p = Path(path).resolve()
    return f"Schema error at {p}: {err}"
