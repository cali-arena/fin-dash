"""
Aggregation policy contract: strict config + schema validation for pre-aggregation (Step 1).

Validates configs/agg_policy.yml and exposes:
- load_and_validate_agg_policy(path) -> AggPolicy
- summarize_agg_policy(policy) -> dict (for logging / hashing)
- policy_hash(policy) -> str (stable SHA-256 of canonical JSON; cache invalidation)

Validations:
- source_table and time_key non-empty
- measures.additive non-empty list
- grains includes exactly: firm_monthly, channel_monthly, ticker_monthly, geo_monthly, segment_monthly
- each grain entry is a list of dim keys that exist in dims (firm_monthly must be [])
- null_handling.strategy in {"DROP", "UNKNOWN"}
- rollup.rates_method in {"recompute", "weighted_avg"}
- if weighted_avg, weights provided for each rate in measures.rates
- no duplicate grain definitions (same dim list twice)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_AGG_POLICY_PATH = "configs/agg_policy.yml"

REQUIRED_GRAINS = frozenset({
    "firm_monthly",
    "channel_monthly",
    "ticker_monthly",
    "geo_monthly",
    "segment_monthly",
})

NULL_STRATEGIES = frozenset({"DROP", "UNKNOWN"})
RATES_METHODS = frozenset({"recompute", "weighted_avg", "store_numerators"})


class AggPolicyError(Exception):
    """Raised when agg_policy.yml is missing or invalid."""


@dataclass
class MeasuresConfig:
    additive: list[str]
    rates: list[str]


@dataclass
class NullHandlingConfig:
    strategy: str
    unknown_label: str


@dataclass
class RollupConfig:
    additive_method: str
    rates_method: str
    weights: dict[str, str]


@dataclass
class AggPolicy:
    source_table: str
    time_key: str
    measures: MeasuresConfig
    dims: dict[str, str]
    grains: dict[str, list[list[str]]]
    null_handling: NullHandlingConfig
    rollup: RollupConfig


def _require_mapping(obj: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise AggPolicyError(f"{ctx} must be a mapping/object; got {type(obj).__name__}")
    return obj


def _require_str(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AggPolicyError(f"{ctx} must be a non-empty string")
    return value.strip()


def _require_list(value: Any, ctx: str, of_str: bool = False) -> list:
    if not isinstance(value, list):
        raise AggPolicyError(f"{ctx} must be a list; got {type(value).__name__}")
    if of_str:
        for i, v in enumerate(value):
            if not isinstance(v, str) or not v.strip():
                raise AggPolicyError(f"{ctx}[{i}] must be a non-empty string")
    return value


def _normalize_grain_value(raw: Any, grain_name: str, ctx: str) -> list[list[str]]:
    """Normalize grain value to list[list[str]]. [] -> [[]]; ['a'] -> [['a']]; [['a'],['a','b']] -> same."""
    if raw is None:
        raise AggPolicyError(f"{ctx} is required")
    if isinstance(raw, list) and len(raw) == 0:
        return [[]]
    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, str):
            return [list(raw)]
        if isinstance(first, list):
            out: list[list[str]] = []
            for i, group in enumerate(raw):
                if not isinstance(group, list):
                    raise AggPolicyError(f"{ctx}[{i}] must be a list of dim keys")
                out.append([str(x).strip() for x in group if str(x).strip()])
            return out if out else [[]]
    raise AggPolicyError(f"{ctx} must be a list or list of lists of dim keys; got {type(raw).__name__}")


def _parse_measures(cfg: Mapping[str, Any]) -> MeasuresConfig:
    m = _require_mapping(cfg.get("measures"), "agg.measures")
    additive = _require_list(m.get("additive"), "agg.measures.additive", of_str=True)
    if not additive:
        raise AggPolicyError("agg.measures.additive must be a non-empty list")
    rates_raw = m.get("rates")
    rates = _require_list(rates_raw, "agg.measures.rates", of_str=True) if rates_raw is not None else []
    return MeasuresConfig(additive=additive, rates=rates)


def _parse_dims(cfg: Mapping[str, Any]) -> dict[str, str]:
    d = cfg.get("dims")
    if d is None or not isinstance(d, Mapping):
        raise AggPolicyError("agg.dims must be a mapping")
    return {_require_str(k, f"agg.dims key {k!r}"): _require_str(v, f"agg.dims[{k!r}]") for k, v in d.items()}


def _parse_grains(cfg: Mapping[str, Any], dim_keys: set[str]) -> dict[str, list[list[str]]]:
    g = _require_mapping(cfg.get("grains"), "agg.grains")
    missing = REQUIRED_GRAINS - set(g.keys())
    if missing:
        raise AggPolicyError(f"agg.grains missing required keys: {sorted(missing)}. Required: {sorted(REQUIRED_GRAINS)}")
    extra = set(g.keys()) - REQUIRED_GRAINS
    if extra:
        raise AggPolicyError(f"agg.grains has unknown keys: {sorted(extra)}. Allowed: {sorted(REQUIRED_GRAINS)}")

    grains_out: dict[str, list[list[str]]] = {}
    seen_dims: set[tuple[str, ...]] = set()

    for name in sorted(REQUIRED_GRAINS):
        raw = g[name]
        normalized = _normalize_grain_value(raw, name, f"agg.grains.{name}")
        if name == "firm_monthly":
            if normalized != [[]]:
                raise AggPolicyError("agg.grains.firm_monthly must be an empty list []")
        for dim_list in normalized:
            for d in dim_list:
                if d not in dim_keys:
                    raise AggPolicyError(f"agg.grains.{name} references dim {d!r} which is not in agg.dims. dims: {sorted(dim_keys)}")
            key = tuple(sorted(dim_list))
            if key in seen_dims:
                raise AggPolicyError(f"agg.grains: duplicate grain definition (same dim list {list(key)!r})")
            seen_dims.add(key)
        grains_out[name] = normalized
    return grains_out


def _parse_null_handling(cfg: Mapping[str, Any]) -> NullHandlingConfig:
    n = _require_mapping(cfg.get("null_handling"), "agg.null_handling")
    strategy = _require_str(n.get("strategy"), "agg.null_handling.strategy").upper()
    if strategy not in NULL_STRATEGIES:
        raise AggPolicyError(f"agg.null_handling.strategy must be one of {sorted(NULL_STRATEGIES)}; got {strategy!r}")
    unknown_label = _require_str(n.get("unknown_label"), "agg.null_handling.unknown_label")
    return NullHandlingConfig(strategy=strategy, unknown_label=unknown_label)


def _parse_rollup(cfg: Mapping[str, Any], rates: list[str]) -> RollupConfig:
    r = _require_mapping(cfg.get("rollup"), "agg.rollup")
    additive_method = _require_str(r.get("additive_method"), "agg.rollup.additive_method")
    rates_method = _require_str(r.get("rates_method"), "agg.rollup.rates_method").lower()
    if rates_method not in RATES_METHODS:
        raise AggPolicyError(f"agg.rollup.rates_method must be one of {sorted(RATES_METHODS)}; got {rates_method!r}")
    weights_raw = r.get("weights")
    weights = dict(weights_raw) if isinstance(weights_raw, Mapping) else {}
    if rates_method == "weighted_avg":
        for rate in rates:
            if rate not in weights or not str(weights.get(rate, "")).strip():
                raise AggPolicyError(
                    f"agg.rollup.rates_method is weighted_avg but agg.rollup.weights missing or empty for rate {rate!r}. "
                    f"Required weights for: {rates!r}"
                )
    return RollupConfig(additive_method=additive_method, rates_method=rates_method, weights=weights)


def load_and_validate_agg_policy(path: str | Path = DEFAULT_AGG_POLICY_PATH) -> AggPolicy:
    """
    Load configs/agg_policy.yml and return a validated AggPolicy.
    Raises AggPolicyError on any schema or validation issue.
    """
    path = Path(path)
    if not path.exists():
        raise AggPolicyError(f"Aggregation policy config not found: {path}")
    try:
        import yaml
    except ImportError as e:
        raise AggPolicyError(
            "PyYAML is required to load agg policy. Install with: pip install pyyaml"
        ) from e
    raw_text = path.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(raw_text)
    except Exception as e:
        raise AggPolicyError(f"Failed to parse YAML {path}: {e}") from e

    cfg = _require_mapping(raw.get("agg") or raw, "agg")

    source_table = _require_str(cfg.get("source_table"), "agg.source_table")
    time_key = _require_str(cfg.get("time_key"), "agg.time_key")
    measures = _parse_measures(cfg)
    dims = _parse_dims(cfg)
    dim_keys = set(dims.keys())
    grains = _parse_grains(cfg, dim_keys)
    null_handling = _parse_null_handling(cfg)
    rollup = _parse_rollup(cfg, measures.rates)

    return AggPolicy(
        source_table=source_table,
        time_key=time_key,
        measures=measures,
        dims=dims,
        grains=grains,
        null_handling=null_handling,
        rollup=rollup,
    )


def summarize_agg_policy(policy: AggPolicy) -> dict[str, Any]:
    """Return a JSON-serializable dict of the policy (for logging and policy_hash)."""
    return {
        "source_table": policy.source_table,
        "time_key": policy.time_key,
        "measures": {
            "additive": policy.measures.additive,
            "rates": policy.measures.rates,
        },
        "dims": dict(policy.dims),
        "grains": {k: list(v) for k, v in policy.grains.items()},
        "null_handling": {
            "strategy": policy.null_handling.strategy,
            "unknown_label": policy.null_handling.unknown_label,
        },
        "rollup": {
            "additive_method": policy.rollup.additive_method,
            "rates_method": policy.rollup.rates_method,
            "weights": dict(policy.rollup.weights),
        },
    }


def _canonical_json_dumps(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace. For hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def policy_hash(policy: AggPolicy) -> str:
    """Return SHA-256 hex digest of canonical JSON of summarize_agg_policy(policy). Stable for cache keys."""
    payload = _canonical_json_dumps(summarize_agg_policy(policy))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
