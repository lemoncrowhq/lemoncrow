"""Prospective per-turn model routing recommendations.

The router is advisory: host CLIs keep ownership of actual model selection.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any, Literal

from lemoncrow.core.capabilities.pricing import ModelPricing
from lemoncrow.pro.capabilities.model_routing.cache_cost import cache_eviction_cost_usd
from lemoncrow.pro.capabilities.model_routing.complexity import (
    signals_from_state,
    tier_for_complexity,
    tier_routing_enabled,
)
from lemoncrow.pro.capabilities.model_routing.stickiness import decrement_stickiness
from lemoncrow.pro.capabilities.model_routing.stickiness import (
    stickiness_remaining as _stickiness_remaining,
)
from lemoncrow.pro.capabilities.model_routing.success_predictor import (
    SuccessTable,
    features_from_state,
    learned_routing_enabled,
    learned_routing_min_samples,
    learned_routing_threshold,
    p_weak_succeeds,
)
from lemoncrow.pro.capabilities.prefix_cache.planner import PrefixCachePlan

ModelTier = Literal["cheap", "medium", "expensive"]

# Semantic 5-tier route type aligned with enterprise routing language:
#   deterministic  - pure function / regex / rule, no LLM call needed
#   local_slm      - small local model via Ollama (free, private)
#   cheap_llm      - small hosted model (Haiku, Flash, gpt-4.1-mini)
#   frontier_llm   - frontier hosted model (Opus, GPT-5, Sonnet)
#   human_review   - escalate to human; agent cannot proceed alone
RouteTier = Literal["deterministic", "local_slm", "cheap_llm", "frontier_llm", "human_review"]

# Map ModelTier → RouteTier for backward compat
_MODEL_TIER_TO_ROUTE_TIER: dict[ModelTier, RouteTier] = {
    "cheap": "cheap_llm",
    "medium": "cheap_llm",
    "expensive": "frontier_llm",
}


_CHEAP_TOOLS = {
    "bash",
    "shell",
    "read",
    "smart_read",
    "search",
    "smart_search",
    "grep",
    "glob",
    "ls",
    "list",
    "context",
    "memory",
}
_MEDIUM_TOOLS = {"edit", "smart_edit", "write", "apply_patch", "compact", "verify", "route"}
_EXPENSIVE_TOOLS = {"agent", "task", "spawn", "delegate", "architect"}

# Session-phase classification — used by _score_session_phase.
# Exploration tools: reading, searching, inspecting the codebase.
_EXPLORATION_TOOLS = frozenset(
    {
        "grep",
        "glob",
        "search",
        "smart_search",
        "context",
        "memory",
        "webfetch",
        "websearch",
        "toolsearch",
        "ls",
        "list",
        "read",
        "smart_read",
    }
)
# Execution tools: mutating the workspace or delegating sub-work.
_EXECUTION_TOOLS = frozenset(
    {
        "edit",
        "smart_edit",
        "write",
        "multiedit",
        "notebookedit",
        "bash",
        "shell",
        "agent",
        "task",
        "todowrite",
        "apply_patch",
    }
)

_CHEAP_VERBS_RE = re.compile(r"\b(explain|show|list|summari[sz]e|read|find|search|inspect)\b", re.IGNORECASE)
_MEDIUM_VERBS_RE = re.compile(r"\b(implement|fix|add|update|change|refactor|test|verify)\b", re.IGNORECASE)
_EXPENSIVE_VERBS_RE = re.compile(
    r"\b(design|architect|plan|strategy|migrate|rewrite|end[- ]to[- ]end)\b", re.IGNORECASE
)
_SMALL_OUTPUT_RE = re.compile(r"\b(<\s*500|under\s+500|brief|short|concise|one[- ]line)\b", re.IGNORECASE)
_OPEN_OUTPUT_RE = re.compile(r"\b(open[- ]ended|comprehensive|full|deep|thorough|all files|entire)\b", re.IGNORECASE)
_EXPLICIT_SESSION_PHASE_SCORE: dict[str, tuple[int, str]] = {
    "explore": (0, "explore"),
    "exploration": (0, "exploration"),
    "transition": (1, "transition"),
    "planning": (1, "planning"),
    "plan": (1, "plan"),
    "execute": (2, "execute"),
    "execution": (2, "execution"),
}


@dataclass(frozen=True)
class ModelRecommendation:
    tier: ModelTier
    model: str
    route_tier: RouteTier = "cheap_llm"
    reasons: list[str] = field(default_factory=list)
    score: int = 0
    cache_affinity_model: str | None = None
    cache_cost_usd: float = 0.0
    quality_gain_usd_estimated: float = 0.0
    decision: str = "baseline"
    baseline_tier: ModelTier | None = None
    sticky_until_tool_calls: int = 0
    p_weak_succeeds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "route_tier": self.route_tier,
            "model": self.model,
            "reasons": list(self.reasons),
            "score": self.score,
            "cache_affinity_model": self.cache_affinity_model,
            "cache_cost_usd": self.cache_cost_usd,
            "quality_gain_usd_estimated": self.quality_gain_usd_estimated,
            "decision": self.decision,
            "baseline_tier": self.baseline_tier,
            "sticky_until_tool_calls": self.sticky_until_tool_calls,
            "p_weak_succeeds": self.p_weak_succeeds,
        }


def _detect_local_slm() -> str | None:
    """Return the Ollama base URL if a local SLM is available, else None."""
    import os

    return os.environ.get("OLLAMA_BASE_URL") or os.environ.get("LEMONCROW_LOCAL_SLM_URL") or None


class ModelRouter:
    """Score a prospective turn into cheap, medium, or expensive model tiers."""

    def __init__(
        self,
        *,
        cheap_model: str = "claude-haiku-4-5",
        medium_model: str = "claude-sonnet-4.6",
        expensive_model: str = "claude-opus-4-7",
    ) -> None:
        self._models: dict[ModelTier, str] = {
            "cheap": cheap_model,
            "medium": medium_model,
            "expensive": expensive_model,
        }

    def score(
        self,
        tool_name: str,
        task_text: str,
        session_state: Mapping[str, Any] | None = None,
    ) -> ModelRecommendation | None:
        from lemoncrow.bench.mode import is_off as _bench_is_off

        if _bench_is_off():
            return None
        state = session_state or {}
        reasons: list[str] = []
        score = 0

        tool_score, tool_reason = self._score_tool(tool_name)
        score += tool_score
        reasons.append(tool_reason)

        verb_score, verb_reason = self._score_task_text(task_text)
        score += verb_score
        reasons.append(verb_reason)

        output_score, output_reason = self._score_output_target(task_text, state)
        score += output_score
        reasons.append(output_reason)

        error_score, error_reason = self._score_prior_errors(state)
        score += error_score
        reasons.append(error_reason)

        phase_score, phase_reason = self._score_session_phase(state)
        score += phase_score
        reasons.append(phase_reason)

        tier = self._tier_for_score(score)

        # T9 (opt-in): learned routing-confidence over the heuristic step-down.
        # Default-off; only consulted when LEMONCROW_LEARNED_ROUTING is on.  None
        # means "uncertain" -> the heuristic step-down rule is used unchanged.
        p_weak = self._predict_weak_success(tool_name, task_text, state)

        # N1 (opt-in): complexity-scored tier routing.  Default-off so the
        # baseline decision above is unchanged for existing callers/tests.
        tier = self._apply_complexity_tier(tier, task_text, state, reasons, p_weak=p_weak)

        model = self._models[tier]

        cache_affinity_model = _clean_string(state.get("cache_affinity_model"))
        if cache_affinity_model:
            affinity_tier = self._tier_for_model(cache_affinity_model, fallback=tier)
            _tier_rank: dict[ModelTier, int] = {"cheap": 0, "medium": 1, "expensive": 2}
            # Only follow cache affinity when it is at the same tier or at most one
            # step below the scored tier.  Never downgrade an expensive task to cheap
            # just because a haiku wrote the last cache block.
            if _tier_rank[affinity_tier] >= _tier_rank[tier] - 1:
                model = cache_affinity_model
                tier = affinity_tier
                reasons.append("cache_affinity: follow previous cache-writing model")
            else:
                reasons.append(f"cache_affinity ignored: scored {tier}, affinity suggests {affinity_tier}")

        # Compute semantic RouteTier: prefer local_slm when Ollama is available
        # and the task scored into the cheap tier.
        route_tier = self._compute_route_tier(tier, score, state)

        return ModelRecommendation(
            tier=tier,
            route_tier=route_tier,
            model=model,
            reasons=reasons,
            score=score,
            cache_affinity_model=cache_affinity_model,
            p_weak_succeeds=p_weak,
        )

    def recommend(
        self,
        tool_name: str,
        task_text: str,
        session_state: Mapping[str, Any] | None = None,
        *,
        prior_plan: PrefixCachePlan | None = None,
        current_plan: PrefixCachePlan | None = None,
        prior_route: ModelRecommendation | None = None,
        stickiness_remaining: int = 0,
        pricing: ModelPricing | None = None,
        route_decision_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> ModelRecommendation | None:
        """Return a cache-aware recommendation while preserving score() behavior."""
        baseline = self.score(tool_name, task_text, session_state)
        if baseline is None:
            return None

        state = session_state or {}
        remaining = _stickiness_remaining(stickiness_remaining)
        cache_cost_usd = 0.0
        quality_gain_usd_estimated = 0.0
        chosen = replace(baseline, baseline_tier=baseline.tier)

        if prior_route is not None and remaining > 0:
            next_remaining = decrement_stickiness(remaining).remaining_tool_calls
            chosen = replace(
                prior_route,
                reasons=[*prior_route.reasons, "stickiness: preserved prior route"],
                cache_cost_usd=0.0,
                quality_gain_usd_estimated=0.0,
                decision="sticky",
                baseline_tier=baseline.tier,
                sticky_until_tool_calls=next_remaining,
            )
            self._emit_route_decision(route_decision_sink, chosen, baseline)
            return chosen

        if prior_route is not None and prior_plan is not None and current_plan is not None and pricing is not None:
            cache_cost_usd = cache_eviction_cost_usd(prior_plan, current_plan, pricing)
            quality_gain_usd_estimated = _estimated_quality_gain_usd(prior_route, baseline, state)
            if cache_cost_usd > quality_gain_usd_estimated:
                chosen = replace(
                    prior_route,
                    reasons=[
                        *prior_route.reasons,
                        f"cache_preserve: eviction ${cache_cost_usd:.8f} exceeds quality gain "
                        f"${quality_gain_usd_estimated:.8f}",
                    ],
                    cache_cost_usd=cache_cost_usd,
                    quality_gain_usd_estimated=quality_gain_usd_estimated,
                    decision="cache_preserve",
                    baseline_tier=baseline.tier,
                    sticky_until_tool_calls=0,
                )
            elif cache_cost_usd > 0.0 or quality_gain_usd_estimated > 0.0:
                chosen = replace(
                    baseline,
                    reasons=[
                        *baseline.reasons,
                        f"quality_gain: estimated ${quality_gain_usd_estimated:.8f} covers cache eviction "
                        f"${cache_cost_usd:.8f}",
                    ],
                    cache_cost_usd=cache_cost_usd,
                    quality_gain_usd_estimated=quality_gain_usd_estimated,
                    decision="quality_gain",
                    baseline_tier=baseline.tier,
                    sticky_until_tool_calls=0,
                )

        self._emit_route_decision(route_decision_sink, chosen, baseline)
        return chosen

    def _emit_route_decision(
        self,
        sink: Callable[[dict[str, Any]], None] | None,
        recommendation: ModelRecommendation,
        baseline: ModelRecommendation,
    ) -> None:
        if sink is None:
            return
        payload = {
            "kind": "route_decision",
            "decision": recommendation.decision,
            "chosen_tier": recommendation.tier,
            "chosen_model": recommendation.model,
            "baseline_tier": baseline.tier,
            "baseline_model": baseline.model,
            "cache_cost_usd": recommendation.cache_cost_usd,
            "quality_gain_usd_estimated": recommendation.quality_gain_usd_estimated,
            "sticky_until_tool_calls": recommendation.sticky_until_tool_calls,
            "p_weak_succeeds": recommendation.p_weak_succeeds,
        }
        try:
            sink(payload)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return

    def _predict_weak_success(
        self,
        tool_name: str,
        task_text: str,
        state: Mapping[str, Any],
    ) -> float | None:
        """Learned P(weak model succeeds) for this turn, or ``None`` (T9).

        Default-off: returns ``None`` unless the learned layer is opted in via
        ``session_state["learned_routing"]`` or ``LEMONCROW_LEARNED_ROUTING``.
        The outcome-frequency table is supplied read-only by the caller as
        ``session_state["learned_routing_table"]`` (a ``SuccessTable`` built from
        persisted route outcomes); without it the prediction is ``None`` and the
        router stays purely heuristic.
        """
        if not learned_routing_enabled(state):
            return None
        table = state.get("learned_routing_table")
        if not isinstance(table, SuccessTable):
            return None
        features = features_from_state(tool_name, task_text, state)
        return p_weak_succeeds(
            features,
            table=table,
            min_samples=learned_routing_min_samples(state),
        )

    def _apply_complexity_tier(
        self,
        baseline_tier: ModelTier,
        task_text: str,
        state: Mapping[str, Any],
        reasons: list[str],
        *,
        p_weak: float | None = None,
    ) -> ModelTier:
        """Apply complexity-scored tier routing when opted in (N1).

        Returns ``baseline_tier`` unchanged unless tier routing is enabled via
        ``session_state["tier_routing"]`` or ``LEMONCROW_TIER_ROUTING``.  When
        enabled it may step **up** (escalation preserved -- hard work never
        downgraded) or step **down** by one level for clearly simple work.

        T9 (opt-in): when the learned routing-confidence layer is enabled, the
        heuristic step-down is *gated* by ``p_weak`` -- the downgrade is only
        taken when a weak model is predicted to succeed with confidence >= the
        learned threshold.  ``p_weak`` of ``None`` (uncertain / no data) or below
        threshold falls back to the heuristic step-down.  The learned layer only
        ever gates the existing one-level clamp; it never widens the downgrade
        and never relaxes the safety floor.
        """
        if not tier_routing_enabled(state):
            return baseline_tier

        signals = signals_from_state(task_text, state)
        result = tier_for_complexity(signals)
        complexity_tier = result.model_tier  # "cheap" | "medium" | "expensive"
        reasons.extend(result.reasons)

        rank: dict[str, int] = {"cheap": 0, "medium": 1, "expensive": 2}
        ranks: list[ModelTier] = ["cheap", "medium", "expensive"]
        base_rank = rank[baseline_tier]
        complexity_rank = rank[complexity_tier]

        # Step up: complexity (incl. escalation/cross-project/error signals)
        # outranks the baseline -> take the stronger tier.  Never downgrades.
        if complexity_rank > base_rank:
            reasons.append(f"tier_routing: step up {baseline_tier} -> {complexity_tier}")
            return complexity_tier  # type: ignore[return-value]

        # Step down (silent): only for clearly simple work with no risk signals.
        # Drop exactly one level toward the complexity tier and never below it.
        # The clamp below is the SAFETY FLOOR and is identical whether or not the
        # learned layer is engaged -- learned routing only decides *whether* to
        # take this same step, never how far it goes.
        if complexity_rank < base_rank and not result.stepped_up and not signals.escalate and not signals.cross_project:
            stepped = ranks[max(complexity_rank, base_rank - 1)]
            # T9 gate: when learned routing is on, only downgrade if a weak model
            # is confidently predicted to succeed; uncertain (None) or below the
            # threshold defers to holding the baseline (no heuristic downgrade).
            if learned_routing_enabled(state):
                threshold = learned_routing_threshold(state)
                if p_weak is None:
                    reasons.append(
                        f"learned_routing: no calibrated estimate -> hold {baseline_tier} (heuristic fallback)"
                    )
                    return baseline_tier
                if p_weak < threshold:
                    reasons.append(
                        f"learned_routing: p_weak {p_weak:.3f} < threshold {threshold:.3f} -> hold {baseline_tier}"
                    )
                    return baseline_tier
                reasons.append(f"learned_routing: p_weak {p_weak:.3f} >= threshold {threshold:.3f} -> allow step down")
            reasons.append(f"tier_routing: step down {baseline_tier} -> {stepped}")
            return stepped

        reasons.append(f"tier_routing: hold {baseline_tier}")
        return baseline_tier

    def _compute_route_tier(self, tier: ModelTier, score: int, state: Mapping[str, Any]) -> RouteTier:
        """Map scored ModelTier to semantic RouteTier.

        Escalation to human_review is triggered when the caller sets
        state["escalate"] = True (e.g. from quality_router after repeated
        failures or protected-file writes).
        """
        if state.get("escalate"):
            return "human_review"
        # Pure-function tasks (no LLM needed): regex/grep/deterministic ops
        if score == 0 and tier == "cheap" and state.get("deterministic"):
            return "deterministic"
        # Local SLM available and task is cheap — prefer free/private
        if tier == "cheap" and _detect_local_slm():
            return "local_slm"
        if tier == "expensive":
            return "frontier_llm"
        # cheap / medium both map to cheap_llm when no local SLM
        return "cheap_llm"

    def _score_tool(self, tool_name: str) -> tuple[int, str]:
        normalized = tool_name.strip().lower().replace("-", "_")
        if normalized in _EXPENSIVE_TOOLS or "agent" in normalized:
            return 2, f"tool={tool_name}: agent/delegation work"
        if normalized in _MEDIUM_TOOLS or "edit" in normalized or "write" in normalized:
            return 1, f"tool={tool_name}: edit/write work"
        if normalized in _CHEAP_TOOLS or normalized.startswith(("read", "search")):
            return 0, f"tool={tool_name}: read/search work"
        return 1, f"tool={tool_name}: unknown tool defaults to medium"

    def _score_task_text(self, task_text: str) -> tuple[int, str]:
        if _EXPENSIVE_VERBS_RE.search(task_text):
            return 2, "task_verb: design/architecture/planning signal"
        if _MEDIUM_VERBS_RE.search(task_text):
            return 1, "task_verb: implementation/fix signal"
        if _CHEAP_VERBS_RE.search(task_text):
            return 0, "task_verb: explain/show/list signal"
        return 1, "task_verb: no clear cheap signal"

    def _score_output_target(self, task_text: str, state: Mapping[str, Any]) -> tuple[int, str]:
        target = " ".join([task_text, str(state.get("output_target", ""))])
        if _OPEN_OUTPUT_RE.search(target):
            return 2, "output_target: open-ended or comprehensive"
        if _SMALL_OUTPUT_RE.search(target):
            return 0, "output_target: bounded under 500 tokens"
        max_tokens = _safe_int(state.get("max_output_tokens"))
        if max_tokens and max_tokens < 500:
            return 0, "output_target: max_output_tokens < 500"
        if max_tokens and max_tokens < 3_000:
            return 1, "output_target: max_output_tokens < 3000"
        return 1, "output_target: medium default"

    def _score_session_phase(self, state: Mapping[str, Any]) -> tuple[int, str]:
        """Score based on where we are in the session lifecycle.

        Exploration (score 0): early turns or read-dominant recent history.
          Haiku can discover the repo just fine.
        Transition (score 1): mix of reads and edits, not clearly either phase.
        Execution (score 2): edit-dominant recent history or deep into the session.
          Sonnet/Opus needed — model must track accumulated context about specific
          files and paths it already read, not re-discover them.
        """
        explicit = _clean_string(state.get("session_phase")) or _clean_string(state.get("workflow_step"))
        if explicit:
            normalized = explicit.lower().replace("-", "_")
            mapped = _EXPLICIT_SESSION_PHASE_SCORE.get(normalized)
            if mapped is not None:
                return mapped[0], f"session_phase: explicit {mapped[1]}"

        turn_number = _safe_int(state.get("turn_number"))
        recent: list[str] = []
        raw = state.get("recent_tool_calls")
        if isinstance(raw, list):
            recent = [str(t).strip().lower().replace("-", "_") for t in raw[-10:]]

        if not recent and turn_number == 0:
            return 0, "session_phase: no history, treat as exploration"

        explore_count = sum(1 for t in recent if t in _EXPLORATION_TOOLS or t.startswith(("read", "search", "grep")))
        exec_count = sum(1 for t in recent if t in _EXECUTION_TOOLS or "edit" in t or "write" in t)
        total = len(recent) or 1
        explore_ratio = explore_count / total
        exec_ratio = exec_count / total

        # Deep into session with execution-heavy recent history → execution phase
        if turn_number > 20 and exec_ratio >= 0.30:
            return 2, f"session_phase: execution (turn={turn_number}, exec_ratio={exec_ratio:.0%})"
        if exec_ratio >= 0.40:
            return 2, f"session_phase: execution (exec_ratio={exec_ratio:.0%})"

        # Clearly exploration-dominant → exploration phase
        if explore_ratio >= 0.60 and turn_number <= 15:
            return (
                0,
                f"session_phase: exploration (explore_ratio={explore_ratio:.0%}, turn={turn_number})",
            )

        return (
            1,
            f"session_phase: transition (turn={turn_number}, exec={exec_ratio:.0%}, explore={explore_ratio:.0%})",
        )

    def _score_prior_errors(self, state: Mapping[str, Any]) -> tuple[int, str]:
        errors = _safe_int(state.get("prior_errors"))
        if errors >= 3:
            return 2, "prior_errors: 3+ errors need deeper reasoning"
        if errors >= 1:
            return 1, "prior_errors: 1-2 errors"
        return 0, "prior_errors: none"

    def _tier_for_score(self, score: int) -> ModelTier:
        if score <= 2:
            return "cheap"
        if score <= 4:
            return "medium"
        return "expensive"

    def _tier_for_model(self, model: str, *, fallback: ModelTier) -> ModelTier:
        normalized = model.lower()
        if "haiku" in normalized or "mini" in normalized or "flash" in normalized:
            return "cheap"
        if "opus" in normalized or "pro" in normalized or "gpt-5.5" in normalized:
            return "expensive"
        if "sonnet" in normalized or "gpt-5.4" in normalized:
            return "medium"
        return fallback


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _clean_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _estimated_quality_gain_usd(
    prior_route: ModelRecommendation,
    baseline: ModelRecommendation,
    state: Mapping[str, Any],
) -> float:
    tier_rank: dict[ModelTier, int] = {"cheap": 0, "medium": 1, "expensive": 2}
    rank_delta = max(0, tier_rank[baseline.tier] - tier_rank[prior_route.tier])
    explicit = _safe_float(state.get("quality_gain_usd_estimated"))
    if explicit is not None:
        return explicit

    calibrated_delta = _safe_float(state.get("route_outcome_score_delta"))
    if calibrated_delta is not None and calibrated_delta > 0.0 and rank_delta > 0:
        expected_input_tokens = max(1_000, _safe_int(state.get("expected_input_tokens")) or 0)
        return round(rank_delta * calibrated_delta * (expected_input_tokens / 1_000_000), 8)

    return round(rank_delta * 0.001, 8)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        cleaned = float(value)
    elif isinstance(value, str):
        try:
            cleaned = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if cleaned < 0.0 or not isfinite(cleaned):
        return None
    return cleaned
