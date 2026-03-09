"""
AUM rules contract: config for first-row behavior and QA thresholds.
Loader + validator only; no computation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AUM_RULES_CONFIG = "configs/aum_rules.yml"

ALLOWED_FIRST_ROW_MODES = frozenset({"nan", "zero", "carry_end"})
CURATED_PREFIX = "curated/"


def load_aum_rules(path: str | Path) -> dict[str, Any]:
    """Load and return the aum_rules YAML config as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AUM rules config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for aum_rules config. Install with: pip install pyyaml") from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"AUM rules config must be a YAML object; got {type(raw).__name__}")
    return raw


def validate_aum_rules(rules: dict[str, Any]) -> None:
    """
    Validate aum_rules config. Raises ValueError with clear messages if invalid.
    - first_row_rule.mode in allowed list (nan | zero | carry_end)
    - qa_thresholds: max_abs_aum > 0, max_month_over_month_ratio > 0, sample_slices > 0, random_seed present
    - output.artifact_path under curated/
    """
    first_row = rules.get("first_row_rule")
    if not isinstance(first_row, dict):
        raise ValueError("aum_rules: 'first_row_rule' must be an object")
    mode = first_row.get("mode")
    if mode not in ALLOWED_FIRST_ROW_MODES:
        raise ValueError(
            f"aum_rules: first_row_rule.mode must be one of {sorted(ALLOWED_FIRST_ROW_MODES)}; got {mode!r}"
        )

    qa = rules.get("qa_thresholds")
    if not isinstance(qa, dict):
        raise ValueError("aum_rules: 'qa_thresholds' must be an object")
    max_abs = qa.get("max_abs_aum")
    if max_abs is not None:
        try:
            if float(max_abs) <= 0:
                raise ValueError("aum_rules: qa_thresholds.max_abs_aum must be positive")
        except (TypeError, ValueError) as e:
            if "positive" in str(e):
                raise
            raise ValueError("aum_rules: qa_thresholds.max_abs_aum must be a positive number") from e
    max_ratio = qa.get("max_month_over_month_ratio")
    if max_ratio is not None:
        try:
            if float(max_ratio) <= 0:
                raise ValueError("aum_rules: qa_thresholds.max_month_over_month_ratio must be positive")
        except (TypeError, ValueError) as e:
            if "positive" in str(e):
                raise
            raise ValueError("aum_rules: qa_thresholds.max_month_over_month_ratio must be a positive number") from e
    sample_slices = qa.get("sample_slices")
    if sample_slices is not None:
        try:
            n = int(sample_slices)
            if n <= 0:
                raise ValueError("aum_rules: qa_thresholds.sample_slices must be positive")
        except (TypeError, ValueError) as e:
            if "positive" in str(e):
                raise
            raise ValueError("aum_rules: qa_thresholds.sample_slices must be a positive integer") from e
    if "random_seed" not in qa:
        raise ValueError("aum_rules: qa_thresholds.random_seed is required")

    output = rules.get("output")
    if not isinstance(output, dict):
        raise ValueError("aum_rules: 'output' must be an object")
    artifact_path = output.get("artifact_path")
    if artifact_path is None or not isinstance(artifact_path, str):
        raise ValueError("aum_rules: output.artifact_path is required and must be a string")
    path_str = artifact_path.replace("\\", "/")
    if not path_str.startswith(CURATED_PREFIX):
        raise ValueError(
            f"aum_rules: output.artifact_path must be under {CURATED_PREFIX!r}; got {artifact_path!r}"
        )
