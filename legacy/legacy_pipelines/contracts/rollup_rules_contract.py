"""
Rollup rules contract: config for collapsing fact_monthly to one end_aum per (month_end + slice keys).
Validates config and optional fact columns; no aggregation logic here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ROLLUP_RULES_CONFIG = "configs/rollup_rules.yml"

ALLOWED_ROLLUP = frozenset({"sum", "max", "last_non_null", "weighted_avg"})
REQUIRED_GRAIN = "month_end"


def load_rollup_rules(path: str | Path) -> dict[str, Any]:
    """Load and return the rollup rules YAML config as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rollup rules config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for rollup_rules config. Install with: pip install pyyaml") from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Rollup rules config must be a YAML object; got {type(raw).__name__}")
    return raw


def validate_rollup_rules(rules: dict[str, Any], fact_columns: list[str] | None = None) -> None:
    """
    Validate rollup rules config. Raises ValueError with clear messages if invalid.
    - grain_keys includes month_end
    - measures.end_aum exists and has rollup in allowed set (sum | max | last_non_null | weighted_avg)
    - when fact_columns provided: end_aum column exists in fact
    """
    grain_keys = rules.get("grain_keys")
    if not isinstance(grain_keys, list):
        raise ValueError("rollup_rules: 'grain_keys' must be a list")
    if REQUIRED_GRAIN not in grain_keys:
        raise ValueError(
            f"rollup_rules: 'grain_keys' must include {REQUIRED_GRAIN!r}; got {grain_keys}"
        )

    measures = rules.get("measures")
    if not isinstance(measures, dict):
        raise ValueError("rollup_rules: 'measures' must be an object")
    if "end_aum" not in measures:
        raise ValueError("rollup_rules: 'measures' must define 'end_aum'")
    end_aum_cfg = measures["end_aum"]
    if not isinstance(end_aum_cfg, dict):
        raise ValueError("rollup_rules: 'measures.end_aum' must be an object")
    rollup = end_aum_cfg.get("rollup")
    if rollup not in ALLOWED_ROLLUP:
        raise ValueError(
            f"rollup_rules: 'measures.end_aum.rollup' must be one of {sorted(ALLOWED_ROLLUP)}; got {rollup!r}"
        )

    if fact_columns is not None:
        fact_set = set(fact_columns)
        if "end_aum" not in fact_set:
            # fact schema uses asset_under_management; rollup config may refer to output name
            if "asset_under_management" not in fact_set:
                raise ValueError(
                    "rollup_rules: fact must contain 'end_aum' or 'asset_under_management' for measure end_aum; "
                    f"fact columns: {sorted(fact_set)}"
                )
    return None
