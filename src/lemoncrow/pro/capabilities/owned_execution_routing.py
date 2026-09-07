"""Owned-execution routing orchestrator.

This is the top-level entry point for routing agent sessions that LemonCrow
controls directly (as opposed to advisory routing for host CLIs).

Routing pipeline — four layers:

    Layer 1 — model_routing.ModelRouter
        Scores the current tool call + task text into a cheap/medium/expensive
        tier *within* the current vendor.  Advisory to the host CLI.

    Layer 2 — cross_vendor_routing.CrossVendorRouter  (wraps Layer 1)
        Picks the best vendor + model pair across all configured providers.
        Applies lesson-based preferences and per-session cost caps.

    Layer 3 — quality_router  (parallel to Layers 1-2)
        Issues execution contracts and quality-gate checks for a given route.
        Used here via ``route_execution_contract``; also used by the engine's
        ``QualityRouterCapability``.

    Layer 4 — this module (OwnedExecutionRouter)
        Orchestrates Layers 2-3 plus:
        * counterfactual simulation (what would each candidate cost / deliver?)
        * provider catalog construction (runner profiles, transports)
        * cache-affinity injection
        * runner + transport resolution for actual execution
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from lemoncrow.pro.capabilities.counterfactual.capabilities import (
    TurnRequirements,
    infer_turn_requirements,
    supports_turn,
)
from lemoncrow.pro.capabilities.counterfactual.pricing import (
    CandidateModel,
    PricingTable,
    load_pricing_table,
)
from lemoncrow.pro.capabilities.cross_vendor_routing.configuration import (
    RouteConfig,
    detect_configured_vendors,
    load_route_config_or_default,
)
from lemoncrow.pro.capabilities.cross_vendor_routing.router import (
    CrossVendorRouter,
    NoFeasibleRouteError,
)
from lemoncrow.pro.capabilities.owned_execution_cache_affinity import cache_affinity_for_route
from lemoncrow.pro.capabilities.quality_router.execution_contract import route_execution_contract

OwnedRouteMode = Literal["auto", "explicit"]
OwnedRouteBudget = Literal["cheap", "balanced", "best"]
OwnedCachePolicy = Literal["inherit", "fresh"]

_UNHEALTHY_PROVIDER_STATES = frozenset({"down", "offline", "unhealthy"})
_RUNNER_COMMANDS: dict[str, tuple[str, ...]] = {
    "claude": ("claude",),
    "copilot": ("copilot",),
    "codex": ("codex",),
}
_PROVIDER_RUNNERS: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude",),
    "openai": ("copilot", "codex"),
    "google": (),
}
_PROVIDER_TRANSPORTS: dict[str, str] = {
    "anthropic": "litellm",
    "openai": "openai",
    "google": "litellm",
    "bedrock": "litellm",
    "vertex": "litellm",
    "azure": "litellm",
    "openrouter": "litellm",
    "groq": "litellm",
    "mistral": "litellm",
    "ollama": "litellm",
    "together": "litellm",
    "fireworks": "litellm",
    "zen": "litellm",
}
_TIER_TO_ROUTE_TIER = {"cheap": "cheap_llm", "high": "frontier_llm"}


@dataclass(frozen=True)
class ProviderCapability:
    supports_tool_use: bool
    max_context_window: int
    runner_profiles: tuple[str, ...]
    transport: str
    execution_mode: str
    can_block_start: bool
    can_force_model: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_tool_use": self.supports_tool_use,
            "max_context_window": self.max_context_window,
            "runner_profiles": list(self.runner_profiles),
            "transport": self.transport,
            "execution_mode": self.execution_mode,
            "can_block_start": self.can_block_start,
            "can_force_model": self.can_force_model,
        }


@dataclass(frozen=True)
class ProviderCatalogEntry:
    provider: str
    default_runner: str
    runner_profiles: tuple[str, ...]
    transport: str
    models: tuple[str, ...]
    cheap_model: str
    high_model: str
    capabilities: ProviderCapability
    candidates: tuple[CandidateModel, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "default_runner": self.default_runner,
            "runner_profiles": list(self.runner_profiles),
            "transport": self.transport,
            "models": list(self.models),
            "cheap_model": self.cheap_model,
            "high_model": self.high_model,
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass(frozen=True)
class OwnedRouteRequest:
    tool_name: str
    task_text: str
    mode: OwnedRouteMode = "auto"
    budget: OwnedRouteBudget = "balanced"
    provider: str = ""
    model: str = ""
    runner: str = ""
    actual_provider: str = ""
    host_agent: str = ""
    cache_policy: OwnedCachePolicy = "inherit"
    session_state: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OwnedRouteAlternative:
    provider: str
    model: str
    runner: str
    transport: str
    tier: str
    estimated_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "runner": self.runner,
            "transport": self.transport,
            "tier": self.tier,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(frozen=True)
class OwnedRouteDecision:
    mode: OwnedRouteMode
    provider: str
    model: str
    runner: str
    transport: str
    tier: str
    route_tier: str
    reason: str
    reasons: tuple[str, ...]
    execution_mode: str
    can_block_start: bool
    can_force_model: bool
    enabled_providers: tuple[str, ...]
    available_providers: tuple[ProviderCatalogEntry, ...]
    alternatives: tuple[OwnedRouteAlternative, ...] = ()
    applied_lessons: tuple[str, ...] = ()
    cost_cap_triggered: bool = False
    cost_cap_limit_usd_per_session: float | None = None
    projected_session_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "runner": self.runner,
            "transport": self.transport,
            "tier": self.tier,
            "route_tier": self.route_tier,
            "reason": self.reason,
            "rationale": self.reason,
            "reasons": list(self.reasons),
            "execution_mode": self.execution_mode,
            "can_block_start": self.can_block_start,
            "can_force_model": self.can_force_model,
            "enabled_providers": list(self.enabled_providers),
            "available_providers": [item.to_dict() for item in self.available_providers],
            "alternatives": [item.to_dict() for item in self.alternatives],
            "applied_lessons": list(self.applied_lessons),
            "cost_cap_triggered": self.cost_cap_triggered,
            "cost_cap_limit_usd_per_session": self.cost_cap_limit_usd_per_session,
            "projected_session_cost_usd": self.projected_session_cost_usd,
        }


class OwnedExecutionRouteSelector:
    def __init__(
        self,
        root: Path | str,
        *,
        env: Mapping[str, str] | None = None,
        pricing_table: PricingTable | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._env = env
        self._pricing_table = pricing_table or load_pricing_table()

    def catalog(self, request: OwnedRouteRequest) -> tuple[ProviderCatalogEntry, ...]:
        config = load_route_config_or_default(self._root, env=self._env)
        requirements = infer_turn_requirements(request.tool_name)
        configured_vendors = detect_configured_vendors(self._env)
        unhealthy = _unhealthy_providers(request.session_state)
        entries: list[ProviderCatalogEntry] = []

        for provider in config.enabled_vendors:
            if provider not in configured_vendors or provider in unhealthy:
                continue
            runner_profiles = _runner_profiles_for_provider(provider, host_agent=request.host_agent)
            if not runner_profiles:
                continue
            candidates = tuple(
                candidate
                for candidate in self._pricing_table.candidates_for_vendor(provider)
                if supports_turn(candidate, requirements)
            )
            if not candidates:
                continue
            entries.append(_catalog_entry(provider, runner_profiles, candidates))

        if not entries:
            fallback = self._zen_fallback_entry(request, requirements, unhealthy)
            if fallback is None:
                raise NoFeasibleRouteError("no configured provider has an executable owned runner for this turn")
            entries.append(fallback)
        return tuple(entries)

    def _zen_fallback_entry(
        self,
        request: OwnedRouteRequest,
        requirements: TurnRequirements,
        unhealthy: set[str],
    ) -> ProviderCatalogEntry | None:
        """Keyless OpenCode Zen entry, used only when nothing else can run a turn.

        Zen serves its zero-cost models against the literal ``public`` bearer
        token, so a machine with no credentials at all still has an executable
        route instead of failing into the OpenAI default model.
        """
        from lemoncrow.core.capabilities.providers.zen import public_tier_enabled

        if "zen" in unhealthy or not public_tier_enabled():
            return None
        runner_profiles = _runner_profiles_for_provider("zen", host_agent=request.host_agent)
        if not runner_profiles:
            return None
        candidates = tuple(
            candidate
            for candidate in self._pricing_table.candidates_for_vendor("zen")
            if supports_turn(candidate, requirements)
        )
        if not candidates:
            return None
        return _catalog_entry("zen", runner_profiles, candidates)

    def select(self, request: OwnedRouteRequest) -> OwnedRouteDecision:
        available = self.catalog(request)
        enabled = tuple(entry.provider for entry in available)
        if request.mode == "explicit":
            return _explicit_route_decision(request=request, available=available)
        return self._auto_route_decision(request=request, available=available, enabled=enabled)

    def _auto_route_decision(
        self,
        *,
        request: OwnedRouteRequest,
        available: tuple[ProviderCatalogEntry, ...],
        enabled: tuple[str, ...],
    ) -> OwnedRouteDecision:
        if request.budget == "cheap":
            return _budget_route_decision(
                request=request,
                available=available,
                enabled=enabled,
                choose_highest=False,
            )
        if request.budget == "best":
            return _budget_route_decision(
                request=request,
                available=available,
                enabled=enabled,
                choose_highest=True,
            )

        config = load_route_config_or_default(self._root, env=self._env)
        filtered_config = RouteConfig(
            enabled_vendors=list(enabled),
            risk_class=config.risk_class,
            read_mode=config.read_mode,
            edit_mode=config.edit_mode,
            agent_mode=config.agent_mode,
        )
        recommendation = CrossVendorRouter(
            filtered_config,
            pricing_table=self._pricing_table,
            env=self._env,
        ).recommend(
            tool_name=request.tool_name,
            task_text=request.task_text,
            session_state=request.session_state,
            actual_vendor=request.actual_provider or None,
        )
        if recommendation is None:
            raise NoFeasibleRouteError("owned routing unavailable while bench mode is off")

        selected_entry = next(entry for entry in available if entry.provider == recommendation.vendor)
        selected_candidate = next(
            candidate for candidate in selected_entry.candidates if candidate.model_id == recommendation.model
        )
        sticky = (
            _sticky_affinity_candidate(available, request.session_state) if request.cache_policy == "inherit" else None
        )
        if sticky is not None:
            sticky_entry, sticky_candidate = sticky
            eviction_cost = _cache_eviction_cost(request.session_state)
            recommended_cost = _estimate_cost(selected_candidate, request.session_state)
            sticky_cost = _estimate_cost(sticky_candidate, request.session_state)
            if (sticky_entry.provider, sticky_candidate.model_id) != (
                selected_entry.provider,
                selected_candidate.model_id,
            ) and sticky_cost <= recommended_cost + eviction_cost:
                selected_entry = sticky_entry
                selected_candidate = sticky_candidate
        selected_runner = request.runner or selected_entry.default_runner
        contract = route_execution_contract(_contract_host_for_runner(selected_runner))
        alternatives = tuple(
            OwnedRouteAlternative(
                provider=alternative.vendor,
                model=alternative.model,
                runner=next(entry.default_runner for entry in available if entry.provider == alternative.vendor),
                transport=next(entry.transport for entry in available if entry.provider == alternative.vendor),
                tier=alternative.tier,
                estimated_cost_usd=alternative.estimated_cost_usd,
            )
            for alternative in recommendation.alternatives
            if any(entry.provider == alternative.vendor for entry in available)
        )
        reasons = tuple(
            [
                *recommendation.reasons,
                f"runner={selected_runner}; transport={selected_entry.transport}: executable owned path",
                *(
                    ["cache_affinity: retained warm sticky route"]
                    if (selected_entry.provider, selected_candidate.model_id)
                    != (recommendation.vendor, recommendation.model)
                    else []
                ),
            ]
        )
        return OwnedRouteDecision(
            mode="auto",
            provider=selected_entry.provider,
            model=selected_candidate.model_id,
            runner=selected_runner,
            transport=selected_entry.transport,
            tier=selected_candidate.tier,
            route_tier=_TIER_TO_ROUTE_TIER.get(selected_candidate.tier, "frontier_llm"),
            reason="; ".join(reasons),
            reasons=reasons,
            execution_mode=contract.mode,
            can_block_start=contract.can_block_start,
            can_force_model=contract.can_force_model,
            enabled_providers=enabled,
            available_providers=available,
            alternatives=alternatives,
            applied_lessons=recommendation.applied_lessons,
            cost_cap_triggered=recommendation.cost_cap_triggered,
            cost_cap_limit_usd_per_session=recommendation.cost_cap_limit_usd_per_session,
            projected_session_cost_usd=recommendation.projected_session_cost_usd,
        )


def select_owned_route(
    root: Path | str,
    request: OwnedRouteRequest,
    *,
    env: Mapping[str, str] | None = None,
    pricing_table: PricingTable | None = None,
) -> OwnedRouteDecision:
    return OwnedExecutionRouteSelector(root, env=env, pricing_table=pricing_table).select(request)


def _catalog_entry(
    provider: str,
    runner_profiles: tuple[str, ...],
    candidates: tuple[CandidateModel, ...],
) -> ProviderCatalogEntry:
    cheap_model = _pick_model_for_tier(candidates, "cheap")
    high_model = _pick_model_for_tier(candidates, "high")
    contract = route_execution_contract(_contract_host_for_runner(runner_profiles[0]))
    transport = _transport_for_provider(provider)
    return ProviderCatalogEntry(
        provider=provider,
        default_runner=runner_profiles[0],
        runner_profiles=runner_profiles,
        transport=transport,
        models=tuple(candidate.model_id for candidate in candidates),
        cheap_model=cheap_model.model_id,
        high_model=high_model.model_id,
        capabilities=ProviderCapability(
            supports_tool_use=all(candidate.supports_tool_use for candidate in candidates),
            max_context_window=max(candidate.context_window for candidate in candidates),
            runner_profiles=runner_profiles,
            transport=transport,
            execution_mode=contract.mode,
            can_block_start=contract.can_block_start,
            can_force_model=contract.can_force_model,
        ),
        candidates=candidates,
    )


def _explicit_route_decision(
    *,
    request: OwnedRouteRequest,
    available: tuple[ProviderCatalogEntry, ...],
) -> OwnedRouteDecision:
    provider = request.provider.strip().lower()
    if not provider:
        raise NoFeasibleRouteError("explicit owned routing requires provider")
    model = request.model.strip()
    if not model:
        raise NoFeasibleRouteError("explicit owned routing requires model")
    selected_entry = next((entry for entry in available if entry.provider == provider), None)
    if selected_entry is None:
        raise NoFeasibleRouteError(f"provider {provider!r} is not executable for owned routing")
    if model not in selected_entry.models:
        raise NoFeasibleRouteError(f"model {model!r} is not available for provider {provider!r}")
    runner = request.runner.strip().lower() or selected_entry.default_runner
    if runner not in selected_entry.runner_profiles:
        raise NoFeasibleRouteError(f"runner {runner!r} cannot execute provider {provider!r}")
    contract = route_execution_contract(_contract_host_for_runner(runner))
    candidate = next(candidate for candidate in selected_entry.candidates if candidate.model_id == model)
    reasons = (
        f"explicit provider={provider}",
        f"explicit model={model}",
        f"runner={runner}; transport={selected_entry.transport}: executable owned path",
    )
    return OwnedRouteDecision(
        mode="explicit",
        provider=provider,
        model=model,
        runner=runner,
        transport=selected_entry.transport,
        tier=candidate.tier,
        route_tier=_TIER_TO_ROUTE_TIER.get(candidate.tier, "frontier_llm"),
        reason="; ".join(reasons),
        reasons=reasons,
        execution_mode=contract.mode,
        can_block_start=contract.can_block_start,
        can_force_model=contract.can_force_model,
        enabled_providers=tuple(entry.provider for entry in available),
        available_providers=available,
        alternatives=(
            OwnedRouteAlternative(
                provider=provider,
                model=model,
                runner=runner,
                transport=selected_entry.transport,
                tier=candidate.tier,
                estimated_cost_usd=_estimate_cost(candidate, request.session_state),
            ),
        ),
    )


def _budget_route_decision(
    *,
    request: OwnedRouteRequest,
    available: tuple[ProviderCatalogEntry, ...],
    enabled: tuple[str, ...],
    choose_highest: bool,
) -> OwnedRouteDecision:
    candidates: list[tuple[ProviderCatalogEntry, CandidateModel]] = []
    for entry in available:
        target_model = entry.high_model if choose_highest else entry.cheap_model
        candidate = next(item for item in entry.candidates if item.model_id == target_model)
        candidates.append((entry, candidate))
    ranked = sorted(
        candidates,
        key=lambda item: _estimate_cost(item[1], request.session_state),
        reverse=choose_highest,
    )
    selected_entry, selected_candidate = ranked[0]
    selected_runner = request.runner or selected_entry.default_runner
    contract = route_execution_contract(_contract_host_for_runner(selected_runner))
    alternatives = tuple(
        OwnedRouteAlternative(
            provider=entry.provider,
            model=candidate.model_id,
            runner=entry.default_runner,
            transport=entry.transport,
            tier=candidate.tier,
            estimated_cost_usd=_estimate_cost(candidate, request.session_state),
        )
        for entry, candidate in ranked
    )
    reasons = (
        f"budget={request.budget}: deterministic owned route override",
        f"runner={selected_runner}; transport={selected_entry.transport}: executable owned path",
    )
    return OwnedRouteDecision(
        mode="auto",
        provider=selected_entry.provider,
        model=selected_candidate.model_id,
        runner=selected_runner,
        transport=selected_entry.transport,
        tier=selected_candidate.tier,
        route_tier=_TIER_TO_ROUTE_TIER.get(selected_candidate.tier, "frontier_llm"),
        reason="; ".join(reasons),
        reasons=reasons,
        execution_mode=contract.mode,
        can_block_start=contract.can_block_start,
        can_force_model=contract.can_force_model,
        enabled_providers=enabled,
        available_providers=available,
        alternatives=alternatives,
    )


def _pick_model_for_tier(candidates: tuple[CandidateModel, ...], tier: str) -> CandidateModel:
    matching = [candidate for candidate in candidates if candidate.tier == tier]
    if not matching:
        matching = list(candidates)
    matching.sort(key=lambda candidate: _estimate_cost(candidate, {}), reverse=(tier == "high"))
    return matching[0]


def _estimate_cost(candidate: CandidateModel, session_state: Mapping[str, Any]) -> float:
    expected_input = _token_budget(session_state, key="expected_input_tokens", default=1000)
    expected_output = _token_budget(
        session_state, key="expected_output_tokens", default=max(1, int(expected_input * 0.2))
    )
    adjusted_output = int(expected_output * candidate.output_multiplier)
    return round(candidate.pricing.cost_usd(input_tokens=expected_input, output_tokens=adjusted_output), 6)


def _token_budget(session_state: Mapping[str, Any], *, key: str, default: int) -> int:
    value = session_state.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return default
        return parsed if parsed > 0 else default
    return default


def _unhealthy_providers(session_state: Mapping[str, Any]) -> set[str]:
    unhealthy: set[str] = set()
    health = session_state.get("provider_health")
    if isinstance(health, Mapping):
        for provider, status in health.items():
            if str(status).strip().lower() in _UNHEALTHY_PROVIDER_STATES:
                unhealthy.add(str(provider).strip().lower())
    failures = session_state.get("provider_failures")
    if isinstance(failures, Mapping):
        for provider, count in failures.items():
            if isinstance(count, bool):
                continue
            try:
                numeric = int(count)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                unhealthy.add(str(provider).strip().lower())
    return unhealthy


def _runner_profiles_for_provider(provider: str, *, host_agent: str) -> tuple[str, ...]:
    preferred = host_agent.strip().lower()
    candidates = list(_PROVIDER_RUNNERS.get(provider, ()))
    available = [runner for runner in candidates if _runner_is_available(runner)]
    transport = _transport_for_provider(provider)
    ordered: list[str] = [transport] if transport else []
    if preferred and preferred in available:
        available = [preferred, *[runner for runner in available if runner != preferred]]
    for runner in available:
        if runner not in ordered:
            ordered.append(runner)
    return tuple(ordered)


def _runner_is_available(runner: str) -> bool:
    commands = _RUNNER_COMMANDS.get(runner, ())
    return any(shutil.which(command) is not None for command in commands)


def _contract_host_for_runner(runner: str) -> str:
    if runner in {"claude", "copilot", "codex", "opencode", "lemoncode", "antigravity"}:
        return runner
    return "claude"


def _transport_for_provider(provider: str) -> str:
    return _PROVIDER_TRANSPORTS.get(provider, "")


def _sticky_affinity_candidate(
    available: tuple[ProviderCatalogEntry, ...], session_state: Mapping[str, Any]
) -> tuple[ProviderCatalogEntry, CandidateModel] | None:
    affinity = cache_affinity_for_route(session_state)
    if _cache_affinity_stickiness(affinity) <= 0:
        return None
    provider = str(affinity.get("provider") or "")
    model = str(affinity.get("model") or "")
    transport = str(affinity.get("transport") or "")
    if not provider or not model or not transport or not affinity.get("stable_prefix_hash"):
        return None
    entry = next(
        (
            candidate_entry
            for candidate_entry in available
            if candidate_entry.provider == provider and candidate_entry.transport == transport
        ),
        None,
    )
    if entry is None:
        return None
    candidate = next((item for item in entry.candidates if item.model_id == model), None)
    if candidate is None:
        return None
    return entry, candidate


def _cache_eviction_cost(session_state: Mapping[str, Any]) -> float:
    affinity = cache_affinity_for_route(session_state)
    value = affinity.get("eviction_cost_usd")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, float(value))
    return 0.0


def _cache_affinity_stickiness(affinity: Mapping[str, Any]) -> int:
    value = affinity.get("stickiness_remaining")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return int(max(0.0, value))
    return 0


__all__ = [
    "NoFeasibleRouteError",
    "OwnedCachePolicy",
    "OwnedExecutionRouteSelector",
    "OwnedRouteAlternative",
    "OwnedRouteBudget",
    "OwnedRouteDecision",
    "OwnedRouteMode",
    "OwnedRouteRequest",
    "ProviderCapability",
    "ProviderCatalogEntry",
    "select_owned_route",
]
