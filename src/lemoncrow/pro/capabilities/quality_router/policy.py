"""Pure quality-aware routing policy functions."""

from __future__ import annotations

from collections.abc import Mapping
from fnmatch import fnmatch
from pathlib import PurePosixPath

from lemoncrow.core.foundation.routing_models import (
    AgentRequest,
    ContextBudgetPolicy,
    ExecutionTier,
    RouteDecision,
    StepType,
)
from lemoncrow.pro.capabilities.quality_router.config import RoutingPolicyConfig

EvidenceSummary = Mapping[str, object]


def _normalize_path(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).as_posix().lstrip("./")


def _matches_any(value: str, patterns: list[str]) -> bool:
    normalized = _normalize_path(value)
    return any(fnmatch(normalized, pattern) for pattern in patterns)


def has_protected_file_match(
    changed_files: list[str],
    config: RoutingPolicyConfig,
) -> bool:
    """Return whether any changed file matches configured protected patterns."""

    return any(_matches_any(path, config.protected_file_patterns) for path in changed_files)


def is_high_risk_domain(domain: str | None, config: RoutingPolicyConfig) -> bool:
    """Return whether a domain string matches configured high-risk patterns."""

    if not domain:
        return False
    return _matches_any(domain, config.high_risk_domain_patterns)


def evidence_confidence(evidence_summary: EvidenceSummary | None) -> float:
    """Extract bounded evidence confidence from a provider-neutral summary."""

    if not evidence_summary:
        return 1.0
    raw = evidence_summary.get("confidence", 1.0)
    if not isinstance(raw, str | int | float):
        return 1.0
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return min(max(confidence, 0.0), 1.0)


def evidence_refs(evidence_summary: EvidenceSummary | None) -> list[str]:
    """Extract observable evidence references from a provider-neutral summary."""

    if not evidence_summary:
        return []
    raw = evidence_summary.get("refs", [])
    if not isinstance(raw, list):
        return []
    return [str(ref) for ref in raw]


def _context_ratio(budget: ContextBudgetPolicy, evidence_summary: EvidenceSummary | None) -> float:
    if not evidence_summary or budget.max_input_tokens <= 0:
        return 0.0
    raw = evidence_summary.get("estimated_input_tokens", 0)
    if not isinstance(raw, str | int | float):
        return 0.0
    try:
        estimated_tokens = int(raw)
    except (TypeError, ValueError):
        return 0.0
    ratio = estimated_tokens / budget.max_input_tokens
    return min(max(ratio, 0.0), 1.0)


def required_verifiers(
    *,
    config: RoutingPolicyConfig,
    protected_file_match: bool,
    high_risk: bool,
    low_confidence: bool,
) -> list[str]:
    """Return verifier requirements without duplicates, preserving config order."""

    verifiers = list(config.verifiers.default)
    if protected_file_match:
        verifiers.extend(config.verifiers.protected_file)
    if high_risk:
        verifiers.extend(config.verifiers.high_risk)
    if low_confidence:
        verifiers.extend(config.verifiers.low_confidence)

    deduped: list[str] = []
    seen: set[str] = set()
    for verifier in verifiers:
        if verifier not in seen:
            deduped.append(verifier)
            seen.add(verifier)
    return deduped


def select_tier(
    *,
    request: AgentRequest,
    budget: ContextBudgetPolicy,
    config: RoutingPolicyConfig,
    domain: str | None = None,
    evidence_summary: EvidenceSummary | None = None,
) -> tuple[ExecutionTier, str | None]:
    """Select an execution tier and optional escalation trigger."""

    protected_file_match = has_protected_file_match(request.changed_files, config)
    high_risk = request.risk_level == "high" or is_high_risk_domain(domain, config)
    confidence = evidence_confidence(evidence_summary)
    context_ratio = _context_ratio(budget, evidence_summary)

    if protected_file_match:
        return "premium", "protected_file"
    if high_risk:
        return "premium", "high_risk"
    if confidence <= config.thresholds.premium_evidence_confidence_max:
        return "premium", "low_evidence_confidence"
    if context_ratio >= config.thresholds.premium_context_ratio_min:
        return "premium", "context_pressure"
    if (
        request.risk_level == "low"
        and request.task_type in {"docs", "explain", "test"}
        and context_ratio <= config.thresholds.cheap_context_ratio_max
        and confidence >= config.thresholds.cheap_evidence_confidence_min
    ):
        return "cheap", None
    return "mid", None


def selected_model_for_tier(tier: ExecutionTier, budget: ContextBudgetPolicy) -> str:
    """Return the configured model string for a selected execution tier."""

    if tier == "cheap":
        return budget.cheap_model
    if tier == "mid":
        return budget.mid_model
    if tier == "premium":
        return budget.premium_model
    return ""


def draft_route_decision(
    *,
    request: AgentRequest,
    budget: ContextBudgetPolicy,
    config: RoutingPolicyConfig | None = None,
    step_index: int = 0,
    step_type: StepType = "plan",
    domain: str | None = None,
    evidence_summary: EvidenceSummary | None = None,
) -> RouteDecision:
    """Map request, budget, and evidence into a draft route decision."""

    active_config = config or RoutingPolicyConfig()
    protected_file_match = has_protected_file_match(request.changed_files, active_config)
    high_risk = request.risk_level == "high" or is_high_risk_domain(domain, active_config)
    confidence = evidence_confidence(evidence_summary)
    tier, escalation_trigger = select_tier(
        request=request,
        budget=budget,
        config=active_config,
        domain=domain,
        evidence_summary=evidence_summary,
    )
    low_confidence = confidence <= active_config.thresholds.premium_evidence_confidence_max
    verifiers = required_verifiers(
        config=active_config,
        protected_file_match=protected_file_match,
        high_risk=high_risk,
        low_confidence=low_confidence,
    )

    reason_parts = [f"risk={request.risk_level}", f"task={request.task_type}", f"tier={tier}"]
    if domain:
        reason_parts.append(f"domain={domain}")
    if protected_file_match:
        reason_parts.append("protected_file_match=true")
    if escalation_trigger:
        reason_parts.append(f"escalation={escalation_trigger}")

    return RouteDecision(
        session_id=request.session_id or request.id,
        request_id=request.id,
        step_index=step_index,
        step_type=step_type,
        risk_level=request.risk_level,
        tier=tier,
        selected_model=selected_model_for_tier(tier, budget),
        confidence=confidence,
        reason=", ".join(reason_parts),
        protected_file_match=protected_file_match,
        verifier_required=verifiers,
        escalation_trigger=escalation_trigger,
        evidence_refs=evidence_refs(evidence_summary),
    )


__all__ = [
    "EvidenceSummary",
    "draft_route_decision",
    "evidence_confidence",
    "evidence_refs",
    "has_protected_file_match",
    "is_high_risk_domain",
    "required_verifiers",
    "select_tier",
    "selected_model_for_tier",
]
