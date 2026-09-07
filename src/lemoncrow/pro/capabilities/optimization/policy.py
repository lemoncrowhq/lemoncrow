"""Policy presets and YAML persistence for the Optimization Advisor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml

ModelTier = Literal["cheap", "medium", "expensive"]
PresetName = Literal["conservative", "balanced", "economy", "custom", "recommended", "maximum_saving"]
ConfidenceLevel = Literal["low", "medium", "high"]

DEFAULT_ESCALATE_ON: tuple[str, ...] = (
    "low_confidence",
    "failed_tests",
    "repeated_tool_error",
    "high_diff_risk",
    "user_marks_wrong",
)
DEFAULT_PRESERVE: tuple[str, ...] = (
    "user_requirements",
    "repo_facts",
    "active_plan",
    "open_files",
    "failing_tests",
    "tool_results",
)


@dataclass(frozen=True)
class CompactionPolicy:
    prompt_cache_reorder: bool
    dedup: bool
    retrieval_filter: bool
    lossy_summary: bool
    trigger_at_context_fraction: float
    preserve: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_cache_reorder": self.prompt_cache_reorder,
            "dedup": self.dedup,
            "retrieval_filter": self.retrieval_filter,
            "lossy_summary": self.lossy_summary,
            "trigger_at_context_fraction": self.trigger_at_context_fraction,
            "preserve": list(self.preserve),
        }


def should_compact(fill_fraction: float, policy: CompactionPolicy) -> bool:
    """Return True when live context fill has reached the compaction trigger.

    Pure decision function: history compaction should fire once the measured
    fill fraction meets or exceeds the policy's ``trigger_at_context_fraction``.
    """
    return fill_fraction >= policy.trigger_at_context_fraction


@dataclass(frozen=True)
class RoutingPolicy:
    policy: str
    simple: ModelTier
    medium: ModelTier
    hard: ModelTier
    escalate_on: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "simple": self.simple,
            "medium": self.medium,
            "hard": self.hard,
            "escalate_on": list(self.escalate_on),
        }


@dataclass(frozen=True)
class Policy:
    name: str
    preset: PresetName
    quality_floor: float
    confidence_required: ConfidenceLevel
    routing: RoutingPolicy
    compaction: CompactionPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "preset": self.preset,
            "quality_floor": self.quality_floor,
            "confidence_required": self.confidence_required,
            "routing": self.routing.to_dict(),
            "compaction": self.compaction.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkEvidence:
    runs_path: str | None = None
    baseline_cost_usd: float | None = None
    candidate_cost_usd: float | None = None
    margin: float = 0.05
    confidence: float = 0.95

    def configured(self) -> bool:
        return bool(self.runs_path) and self.baseline_cost_usd is not None and self.candidate_cost_usd is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_path": self.runs_path,
            "baseline_cost_usd": self.baseline_cost_usd,
            "candidate_cost_usd": self.candidate_cost_usd,
            "margin": self.margin,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class AutomationConfig:
    enabled: bool = False
    minimum_projected_tokens_saved: int = 1000
    benchmark_evidence: BenchmarkEvidence = BenchmarkEvidence()
    last_proposal_fingerprint: str | None = None
    last_proposal_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "minimum_projected_tokens_saved": self.minimum_projected_tokens_saved,
            "benchmark_evidence": self.benchmark_evidence.to_dict(),
            "last_proposal_fingerprint": self.last_proposal_fingerprint,
            "last_proposal_at": self.last_proposal_at,
        }


def _base_compaction(
    *,
    prompt_cache_reorder: bool = True,
    dedup: bool = True,
    retrieval_filter: bool = True,
    lossy_summary: bool = False,
    trigger_at_context_fraction: float = 0.72,
) -> CompactionPolicy:
    return CompactionPolicy(
        prompt_cache_reorder=prompt_cache_reorder,
        dedup=dedup,
        retrieval_filter=retrieval_filter,
        lossy_summary=lossy_summary,
        trigger_at_context_fraction=trigger_at_context_fraction,
        preserve=list(DEFAULT_PRESERVE),
    )


def _routing(policy: str, simple: ModelTier, medium: ModelTier, hard: ModelTier) -> RoutingPolicy:
    return RoutingPolicy(
        policy=policy,
        simple=simple,
        medium=medium,
        hard=hard,
        escalate_on=list(DEFAULT_ESCALATE_ON),
    )


def preset_policy(preset: str) -> Policy:
    normalized = preset.strip().lower().replace("-", "_")
    if normalized == "conservative":
        return Policy(
            name="Conservative",
            preset="conservative",
            quality_floor=0.98,
            confidence_required="medium",
            routing=_routing("prefer_strongest", "medium", "medium", "expensive"),
            compaction=_base_compaction(retrieval_filter=False),
        )
    if normalized == "balanced":
        return Policy(
            name="Balanced",
            preset="balanced",
            quality_floor=0.96,
            confidence_required="medium",
            routing=_routing("complexity_escalate", "cheap", "medium", "expensive"),
            compaction=_base_compaction(),
        )
    if normalized == "economy":
        return Policy(
            name="Economy",
            preset="economy",
            quality_floor=0.93,
            confidence_required="low",
            routing=_routing("cheap_first", "cheap", "cheap", "medium"),
            compaction=_base_compaction(lossy_summary=True, trigger_at_context_fraction=0.65),
        )
    if normalized == "maximum_saving":
        return Policy(
            name="Maximum saving",
            preset="maximum_saving",
            quality_floor=0.88,
            confidence_required="low",
            routing=_routing("cheap_first", "cheap", "cheap", "cheap"),
            compaction=_base_compaction(lossy_summary=True, trigger_at_context_fraction=0.45),
        )
    if normalized == "recommended":
        return Policy(
            name="Recommended",
            preset="recommended",
            quality_floor=0.96,
            confidence_required="medium",
            routing=_routing("complexity_escalate", "cheap", "medium", "expensive"),
            compaction=_base_compaction(),
        )
    raise ValueError(f"unknown optimization preset: {preset}")


def identify_policy(policy: Policy, *, name: str, preset: PresetName) -> Policy:
    return Policy(
        name=name,
        preset=preset,
        quality_floor=policy.quality_floor,
        confidence_required=policy.confidence_required,
        routing=policy.routing,
        compaction=policy.compaction,
    )


def optimization_config_path(root: Path) -> Path:
    return Path(root) / "optimization.yaml"


def load_optimization_config(root: Path) -> dict[str, Any]:
    path = optimization_config_path(root)
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid optimization config at {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"optimization config at {path} must be a mapping")
    return cast(dict[str, Any], loaded)


def _write_optimization_config(root: Path, config: dict[str, Any]) -> Path:
    path = optimization_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _tier(value: object, fallback: ModelTier) -> ModelTier:
    if value in {"cheap", "medium", "expensive"}:
        return value
    return fallback


def _confidence(value: object, fallback: ConfidenceLevel) -> ConfidenceLevel:
    if value in {"low", "medium", "high"}:
        return value
    return fallback


def _bool(value: object, fallback: bool) -> bool:
    return value if isinstance(value, bool) else fallback


def _float(value: object, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return fallback
    return fallback


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _string_list(value: object, fallback: tuple[str, ...] | list[str]) -> list[str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return list(fallback)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def policy_from_config(data: dict[str, Any]) -> Policy:
    optimization_map = _mapping(data.get("optimization"))
    preset_value = str(optimization_map.get("preset", data.get("preset", "balanced")))
    try:
        base = preset_policy(preset_value)
    except ValueError:
        base = identify_policy(preset_policy("balanced"), name="Custom", preset="custom")

    routing_map = _mapping(data.get("routing"))
    compaction_map = _mapping(data.get("compaction"))

    routing = RoutingPolicy(
        policy=str(routing_map.get("policy", base.routing.policy)),
        simple=_tier(routing_map.get("simple"), base.routing.simple),
        medium=_tier(routing_map.get("medium"), base.routing.medium),
        hard=_tier(routing_map.get("hard"), base.routing.hard),
        escalate_on=_string_list(routing_map.get("escalate_on"), base.routing.escalate_on),
    )
    compaction = CompactionPolicy(
        prompt_cache_reorder=_bool(compaction_map.get("prompt_cache_reorder"), base.compaction.prompt_cache_reorder),
        dedup=_bool(compaction_map.get("dedup"), base.compaction.dedup),
        retrieval_filter=_bool(compaction_map.get("retrieval_filter"), base.compaction.retrieval_filter),
        lossy_summary=_bool(compaction_map.get("lossy_summary"), base.compaction.lossy_summary),
        trigger_at_context_fraction=_float(
            compaction_map.get("trigger_at_context_fraction"),
            base.compaction.trigger_at_context_fraction,
        ),
        preserve=_string_list(compaction_map.get("preserve"), base.compaction.preserve),
    )
    name = str(optimization_map.get("name", base.name))
    return Policy(
        name=name,
        preset=base.preset,
        quality_floor=_float(optimization_map.get("quality_floor"), base.quality_floor),
        confidence_required=_confidence(
            optimization_map.get("confidence_required"),
            base.confidence_required,
        ),
        routing=routing,
        compaction=compaction,
    )


def _free_baseline_policy() -> Policy:
    """Unoptimized baseline returned for Free installs (the savings engine is Pro).

    Compaction never triggers and routing is a strongest-model passthrough, so
    the runtime behaves as if no optimization policy were active.
    """
    return Policy(
        name="Free (unoptimized)",
        preset="custom",
        quality_floor=1.0,
        confidence_required="high",
        routing=_routing("prefer_strongest", "expensive", "expensive", "expensive"),
        compaction=CompactionPolicy(
            prompt_cache_reorder=False,
            dedup=False,
            retrieval_filter=False,
            lossy_summary=False,
            trigger_at_context_fraction=1.0,
            preserve=[],
        ),
    )


def load_current_policy(root: Path) -> Policy:
    # The savings engine (compaction, routing, prefix-cache, budget) is Pro.
    # Free installs run unoptimized regardless of any on-disk policy -- which can
    # only be written by a Pro `optimize apply` anyway.
    from lemoncrow.core.capabilities import licensing

    if not licensing.has_feature("optimizer"):
        return _free_baseline_policy()
    config = load_optimization_config(root)
    if not config:
        return preset_policy("balanced")
    return policy_from_config(config)


def save_policy(root: Path, policy: Policy) -> Path:
    config = load_optimization_config(root)
    optimization_map = _mapping(config.get("optimization"))
    optimization_map.update(
        {
            "name": policy.name,
            "preset": policy.preset,
            "quality_floor": policy.quality_floor,
            "confidence_required": policy.confidence_required,
        }
    )
    config["optimization"] = optimization_map
    config["routing"] = policy.routing.to_dict()
    config["compaction"] = policy.compaction.to_dict()
    return _write_optimization_config(root, config)


def automation_from_config(data: dict[str, Any]) -> AutomationConfig:
    optimization_map = _mapping(data.get("optimization"))
    automation_map = _mapping(optimization_map.get("automation"))
    evidence_map = _mapping(automation_map.get("benchmark_evidence"))
    threshold = int(
        _float(
            automation_map.get("minimum_projected_tokens_saved"),
            1000.0,
        )
    )
    return AutomationConfig(
        enabled=_bool(automation_map.get("enabled"), False),
        minimum_projected_tokens_saved=max(0, threshold),
        benchmark_evidence=BenchmarkEvidence(
            runs_path=str(evidence_map.get("runs_path")).strip() or None,
            baseline_cost_usd=_optional_float(evidence_map.get("baseline_cost_usd")),
            candidate_cost_usd=_optional_float(evidence_map.get("candidate_cost_usd")),
            margin=max(0.0, _float(evidence_map.get("margin"), 0.05)),
            confidence=_float(evidence_map.get("confidence"), 0.95),
        ),
        last_proposal_fingerprint=(str(automation_map.get("last_proposal_fingerprint")).strip() or None),
        last_proposal_at=str(automation_map.get("last_proposal_at")).strip() or None,
    )


def load_automation_config(root: Path) -> AutomationConfig:
    config = load_optimization_config(root)
    if not config:
        return AutomationConfig()
    return automation_from_config(config)


def save_automation_config(root: Path, automation: AutomationConfig) -> Path:
    config = load_optimization_config(root)
    optimization_map = _mapping(config.get("optimization"))
    automation_map = _mapping(optimization_map.get("automation"))
    automation_map.update(automation.to_dict())
    optimization_map["automation"] = automation_map
    config["optimization"] = optimization_map
    return _write_optimization_config(root, config)


def shadow_consent_at(root: Path) -> str | None:
    optimization = load_optimization_config(root).get("optimization")
    if not isinstance(optimization, dict):
        return None
    value = optimization.get("shadow_consent_at")
    return value if isinstance(value, str) and value else None


def record_shadow_consent(root: Path, when: datetime | None = None) -> str:
    config = load_optimization_config(root)
    optimization_map = _mapping(config.get("optimization"))
    accepted_at = (when or datetime.now(UTC)).isoformat()
    optimization_map["shadow_consent_at"] = accepted_at
    config["optimization"] = optimization_map
    _write_optimization_config(root, config)
    return accepted_at


def forget_shadow_consent(root: Path) -> bool:
    config = load_optimization_config(root)
    optimization_map = _mapping(config.get("optimization"))
    if "shadow_consent_at" not in optimization_map:
        return False
    del optimization_map["shadow_consent_at"]
    config["optimization"] = optimization_map
    _write_optimization_config(root, config)
    return True
