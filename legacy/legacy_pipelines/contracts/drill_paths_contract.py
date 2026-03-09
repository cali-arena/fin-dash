"""
Drill paths contract: strict config for slice keys (no ad-hoc drill dimensions).
Validates config and optional fact columns; logs summary table.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DRILL_PATHS_CONFIG = "configs/drill_paths.yml"


def load_drill_paths_config(path: str | Path) -> dict[str, Any]:
    """Load and return the drill paths YAML config as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Drill paths config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for drill_paths config. Install with: pip install pyyaml") from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Drill paths config must be a YAML object; got {type(raw).__name__}")
    return raw


def validate_drill_paths(config: dict[str, Any], fact_columns: list[str] | None = None) -> None:
    """
    Validate drill_paths config. Raises ValueError with clear messages if invalid.
    - duplicate drill_path ids
    - keys not in allowed set (from config.keys values)
    - keys repeated inside a path
    - > max_keys_per_path
    - enabled path has missing key columns in fact (when fact_columns provided and rules.require_keys_exist_in_fact)
    """
    allowed_keys = set()
    if "keys" in config and isinstance(config["keys"], dict):
        allowed_keys = set(config["keys"].values())

    rules = config.get("rules") or {}
    allow_duplicates = bool(rules.get("allow_duplicates", False))
    max_keys_per_path = int(rules.get("max_keys_per_path", 3))
    require_keys_exist = bool(rules.get("require_keys_exist_in_fact", True))
    fact_set = set(fact_columns) if fact_columns else set()

    paths = config.get("drill_paths")
    if not isinstance(paths, list):
        raise ValueError("config must have 'drill_paths' as a list")

    seen_ids: set[str] = set()
    for i, item in enumerate(paths):
        if not isinstance(item, dict):
            raise ValueError(f"drill_paths[{i}] must be an object; got {type(item).__name__}")

        pid = item.get("id")
        if pid is None or not isinstance(pid, str):
            raise ValueError(f"drill_paths[{i}]: missing or invalid 'id'")
        if not allow_duplicates and pid in seen_ids:
            raise ValueError(f"drill_paths: duplicate id '{pid}'")
        seen_ids.add(pid)

        keys = item.get("keys")
        if keys is None:
            keys = []
        if not isinstance(keys, list):
            raise ValueError(f"drill_paths[{i}] (id={pid}): 'keys' must be a list")
        keys = [k for k in keys if isinstance(k, str)]

        if allowed_keys and keys:
            invalid = [k for k in keys if k not in allowed_keys]
            if invalid:
                raise ValueError(
                    f"drill_paths id='{pid}': keys not in allowed set {sorted(allowed_keys)}: {invalid}"
                )

        if len(keys) != len(set(keys)):
            raise ValueError(f"drill_paths id='{pid}': duplicate keys within path")

        if len(keys) > max_keys_per_path:
            raise ValueError(
                f"drill_paths id='{pid}': {len(keys)} keys exceeds max_keys_per_path={max_keys_per_path}"
            )

        if require_keys_exist and fact_columns is not None and item.get("enabled", True):
            missing = [k for k in keys if k not in fact_set]
            if missing:
                raise ValueError(
                    f"drill_paths id='{pid}' (enabled): fact missing key columns: {missing}"
                )

    _log_drill_paths_summary(paths)


def _log_drill_paths_summary(paths: list[dict]) -> None:
    """Print a summary table: id | enabled | keys | label."""
    if not paths:
        return
    rows = []
    for p in paths:
        pid = p.get("id", "")
        enabled = p.get("enabled", True)
        keys = p.get("keys") or []
        keys_str = ",".join(str(k) for k in keys) if keys else ""
        label = p.get("label", "")
        rows.append((str(pid), str(enabled), keys_str, str(label)))
    col_id = "id"
    col_en = "enabled"
    col_keys = "keys"
    col_label = "label"
    w_id = max(len(col_id), max(len(r[0]) for r in rows), 4)
    w_en = max(len(col_en), max(len(r[1]) for r in rows), 4)
    w_keys = max(len(col_keys), max(len(r[2]) for r in rows), 4)
    w_label = max(len(col_label), max(len(r[3]) for r in rows), 4)
    h = f"{{:<{w_id}}} | {{:<{w_en}}} | {{:<{w_keys}}} | {{:<{w_label}}}"
    logger.info("Drill paths summary:")
    logger.info(h.format(col_id, col_en, col_keys, col_label))
    for r in rows:
        logger.info(h.format(r[0], r[1], r[2], r[3]))
