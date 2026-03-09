"""
Metrics policy contract: deterministic policies for rates (guards, inf handling, clamp).
Loader, validator, and policy snapshot writer only; no metric computation.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_METRICS_POLICY_CONFIG = "configs/metrics_policy.yml"

ALLOWED_BEGIN_AUM_GUARD_MODES = frozenset({"nan", "zero"})
ALLOWED_FEE_YIELD_GUARD_MODES = frozenset({"nan", "zero", "cap"})
ALLOWED_INF_HANDLING_MODES = frozenset({"nan"})
ALLOWED_CLAMP_MODES = frozenset({"warn_only", "hard_clamp"})

CLAMP_RATE_KEYS = ("ogr", "market_impact_rate", "total_growth_rate", "fee_yield")


def load_metrics_policy(path: str | Path) -> dict[str, Any]:
    """Load metrics policy YAML from path and return raw dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics policy config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required for metrics_policy. Install with: pip install pyyaml") from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Metrics policy config must be a YAML object; got {type(raw).__name__}")
    return raw


def _finite(x: Any) -> bool:
    try:
        f = float(x)
        return abs(f) != float("inf")
    except (TypeError, ValueError):
        return False


def validate_metrics_policy(policy_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Hard validation of metrics policy. Returns a normalized policy dict.
    - inputs.required_columns list exists
    - inputs.grain_required list exists
    - policies.begin_aum_guard.mode in {nan, zero}
    - policies.fee_yield_guard.mode in {nan, zero, cap}; if cap, cap_value present and finite
    - policies.inf_handling.mode is nan (fixed)
    - policies.clamp: mode in {warn_only, hard_clamp}; each cap has min < max and finite
    """
    if not isinstance(policy_dict, dict):
        raise ValueError("Metrics policy must be a dict")

    normalized = deepcopy(policy_dict)

    # --- inputs ---
    inputs = normalized.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("metrics_policy: 'inputs' must be an object")
    required_cols = inputs.get("required_columns")
    if not isinstance(required_cols, list):
        raise ValueError("metrics_policy: inputs.required_columns must be a list")
    grain = inputs.get("grain_required")
    if not isinstance(grain, list):
        raise ValueError("metrics_policy: inputs.grain_required must be a list")

    # --- policies.begin_aum_guard ---
    policies = normalized.get("policies")
    if not isinstance(policies, dict):
        raise ValueError("metrics_policy: 'policies' must be an object")

    begin_guard = policies.get("begin_aum_guard")
    if not isinstance(begin_guard, dict):
        raise ValueError("metrics_policy: policies.begin_aum_guard must be an object")
    mode_b = begin_guard.get("mode")
    if mode_b not in ALLOWED_BEGIN_AUM_GUARD_MODES:
        raise ValueError(
            f"metrics_policy: policies.begin_aum_guard.mode must be one of {sorted(ALLOWED_BEGIN_AUM_GUARD_MODES)}; got {mode_b!r}"
        )
    applies = begin_guard.get("applies_to_rates")
    if applies is not None and not isinstance(applies, list):
        raise ValueError("metrics_policy: policies.begin_aum_guard.applies_to_rates must be a list or omitted")

    # --- policies.fee_yield_guard ---
    fee_guard = policies.get("fee_yield_guard")
    if not isinstance(fee_guard, dict):
        raise ValueError("metrics_policy: policies.fee_yield_guard must be an object")
    mode_f = fee_guard.get("mode")
    if mode_f not in ALLOWED_FEE_YIELD_GUARD_MODES:
        raise ValueError(
            f"metrics_policy: policies.fee_yield_guard.mode must be one of {sorted(ALLOWED_FEE_YIELD_GUARD_MODES)}; got {mode_f!r}"
        )
    if mode_f == "cap":
        if "cap_value" not in fee_guard:
            raise ValueError("metrics_policy: policies.fee_yield_guard.cap_value is required when mode is 'cap'")
        cv = fee_guard["cap_value"]
        if not _finite(cv):
            raise ValueError("metrics_policy: policies.fee_yield_guard.cap_value must be finite when mode is 'cap'")

    # --- policies.inf_handling ---
    inf_handling = policies.get("inf_handling")
    if not isinstance(inf_handling, dict):
        raise ValueError("metrics_policy: policies.inf_handling must be an object")
    mode_inf = inf_handling.get("mode")
    if mode_inf not in ALLOWED_INF_HANDLING_MODES:
        raise ValueError(
            f"metrics_policy: policies.inf_handling.mode must be one of {sorted(ALLOWED_INF_HANDLING_MODES)}; got {mode_inf!r}"
        )

    # --- policies.clamp ---
    clamp = policies.get("clamp")
    if not isinstance(clamp, dict):
        raise ValueError("metrics_policy: policies.clamp must be an object")
    mode_c = clamp.get("mode")
    if mode_c not in ALLOWED_CLAMP_MODES:
        raise ValueError(
            f"metrics_policy: policies.clamp.mode must be one of {sorted(ALLOWED_CLAMP_MODES)}; got {mode_c!r}"
        )
    caps = clamp.get("caps")
    if not isinstance(caps, dict):
        raise ValueError("metrics_policy: policies.clamp.caps must be an object")
    for key, cap_spec in caps.items():
        if not isinstance(cap_spec, dict):
            raise ValueError(f"metrics_policy: policies.clamp.caps.{key} must be an object with min/max")
        mn = cap_spec.get("min")
        mx = cap_spec.get("max")
        if mn is None or mx is None:
            raise ValueError(f"metrics_policy: policies.clamp.caps.{key} must have 'min' and 'max'")
        if not _finite(mn) or not _finite(mx):
            raise ValueError(f"metrics_policy: policies.clamp.caps.{key}.min and .max must be finite")
        if float(mn) >= float(mx):
            raise ValueError(f"metrics_policy: policies.clamp.caps.{key} must have min < max")

    # --- audit (optional but structure if present) ---
    audit = normalized.get("audit")
    if audit is not None and not isinstance(audit, dict):
        raise ValueError("metrics_policy: 'audit' must be an object or omitted")

    return normalized


def write_policy_snapshot_if_requested(policy_dict: dict[str, Any], root: Path) -> None:
    """
    If audit.write_policy_snapshot is true, write normalized policy to audit.policy_snapshot_path under root.
    """
    audit = policy_dict.get("audit") if isinstance(policy_dict, dict) else None
    if not isinstance(audit, dict) or not audit.get("write_policy_snapshot", False):
        return
    rel = audit.get("policy_snapshot_path")
    if not rel or not isinstance(rel, str):
        logger.warning("metrics_policy: audit.write_policy_snapshot is true but policy_snapshot_path missing; skipping snapshot")
        return
    out_path = root / rel.replace("\\", "/").lstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy_dict, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote metrics policy snapshot: %s", out_path)
