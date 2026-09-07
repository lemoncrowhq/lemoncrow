"""Historical Optimization Advisor algorithm."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.models import Trace
from lemoncrow.pro.capabilities.optimization.compaction_types import ALL_COMPACTION_TYPES
from lemoncrow.pro.capabilities.optimization.complexity import (
    ComplexityLabel,
    score_trace_complexity,
)
from lemoncrow.pro.capabilities.optimization.golden_runner import GoldenSuiteResult, run_golden_suite
from lemoncrow.pro.capabilities.optimization.policy import (
    ModelTier,
    Policy,
    identify_policy,
    preset_policy,
)
from lemoncrow.pro.capabilities.session_optimizer import trace_cost_usd

INSUFFICIENT_HISTORY_MESSAGE = "Need more session history before recommending — try again after 50+ sessions."

_TIER_COST_FACTOR: dict[ModelTier, float] = {
    "cheap": 0.35,
    "medium": 0.72,
    "expensive": 1.0,
}
_COMPACTION_SAVINGS = {
    "prompt_cache_reorder": 0.08,
    "dedup": 0.10,
    "retrieval_filter": 0.06,
    "lossy_summary": 0.16,
}
_TIER_QUALITY_PENALTY: dict[tuple[ModelTier, ComplexityLabel], float] = {
    ("cheap", "simple"): 0.004,
    ("cheap", "medium"): 0.030,
    ("cheap", "hard"): 0.110,
    ("medium", "simple"): 0.001,
    ("medium", "medium"): 0.008,
    ("medium", "hard"): 0.025,
    ("expensive", "simple"): 0.0,
    ("expensive", "medium"): 0.0,
    ("expensive", "hard"): 0.0,
}


@dataclass(frozen=True)
class Candidate:
    id: str
    policy: Policy
    weekly_cost_usd: float
    estimated_quality: float
    latency_mult: float
    escalation_rate: float
    compaction_breakdown: dict[str, float]
    routing_breakdown: dict[str, float]
    routing_saved_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "policy": self.policy.to_dict(),
            "weekly_cost_usd": self.weekly_cost_usd,
            "estimated_quality": self.estimated_quality,
            "latency_mult": self.latency_mult,
            "escalation_rate": self.escalation_rate,
            "compaction_breakdown": dict(self.compaction_breakdown),
            "routing_breakdown": dict(self.routing_breakdown),
            "routing_saved_usd": self.routing_saved_usd,
        }


@dataclass(frozen=True)
class OptimizationResult:
    current_policy: Policy
    recommended_policy: Policy
    candidates: list[Candidate]
    current_candidate_id: str
    recommended_candidate_id: str | None
    confidence: str
    confidence_reason: str
    sessions_analysed: int
    replayable_tasks: int
    weekly_savings_usd: float
    quality_delta: float
    baseline_weekly_cost_usd: float
    has_recommendation: bool
    message: str
    bucket_counts: dict[str, int]
    golden: GoldenSuiteResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_policy": self.current_policy.to_dict(),
            "recommended_policy": self.recommended_policy.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "current_candidate_id": self.current_candidate_id,
            "recommended_candidate_id": self.recommended_candidate_id,
            "confidence": self.confidence,
            "confidence_reason": self.confidence_reason,
            "sessions_analysed": self.sessions_analysed,
            "replayable_tasks": self.replayable_tasks,
            "weekly_savings_usd": self.weekly_savings_usd,
            "quality_delta": self.quality_delta,
            "baseline_weekly_cost_usd": self.baseline_weekly_cost_usd,
            "has_recommendation": self.has_recommendation,
            "message": self.message,
            "bucket_counts": dict(self.bucket_counts),
            "golden": self.golden.to_dict(),
            "estimation": _estimation_metadata(),
        }


def potential_savings_breakdown(
    candidate: Candidate,
    baseline_weekly_cost_usd: float,
    weekly_savings_usd: float,
) -> dict[str, float]:
    """Map a Candidate's compaction/routing figures onto the shared
    Read / Carry / Output / Routing / Total savings-component model, mirroring
    the realized-savings breakdown in savings_summary.py.

    - Read savings: prompt-cache reordering + retrieval filtering (both cut
      read-side cost).
    - Carry credit: dedup + lossy summary (both cut what's carried forward
      across turns).
    - Output savings: always 0.0 -- the advisor has no output-verbosity policy
      lever today. This is a known, deliberate gap, not a bug.
    - Routing savings: the candidate's precomputed `routing_saved_usd`.
    - Total: the caller-supplied weekly savings figure (e.g. `OptimizationResult
      .weekly_savings_usd` for the recommended candidate, or
      `baseline_weekly_cost_usd - candidate.weekly_cost_usd` for any other
      candidate). `baseline_weekly_cost_usd` is accepted for parity with that
      total and possible future use; the four components below are derived
      solely from the candidate's own breakdowns.
    """
    _ = baseline_weekly_cost_usd
    read_saved_usd = candidate.compaction_breakdown.get(
        "prompt_cache_reorder", 0.0
    ) + candidate.compaction_breakdown.get("retrieval_filter", 0.0)
    carry_saved_usd = candidate.compaction_breakdown.get("dedup", 0.0) + candidate.compaction_breakdown.get(
        "lossy_summary", 0.0
    )
    return {
        "read_saved_usd": round(read_saved_usd, 6),
        "carry_saved_usd": round(carry_saved_usd, 6),
        "output_saved_usd": 0.0,
        "routing_saved_usd": round(candidate.routing_saved_usd, 6),
        "total_saved_usd": round(weekly_savings_usd, 6),
    }


def _trace_created_after(trace: Trace, cutoff: datetime) -> bool:
    created = trace.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created >= cutoff


def _trace_cost(trace: Trace) -> float:
    usage_cost = sum(float(entry.cost_usd or 0.0) for entry in trace.usage_entries)
    if usage_cost > 0:
        return usage_cost
    return trace_cost_usd(trace)


def _status_quality(trace: Trace) -> float:
    if trace.status == "success":
        return 1.0
    if trace.status == "partial":
        return 0.75
    return 0.0


def _route_tier(policy: Policy, label: ComplexityLabel) -> ModelTier:
    if label == "simple":
        return policy.routing.simple
    if label == "medium":
        return policy.routing.medium
    return policy.routing.hard


def _compaction_enabled(policy: Policy) -> dict[str, bool]:
    return {
        "prompt_cache_reorder": policy.compaction.prompt_cache_reorder,
        "dedup": policy.compaction.dedup,
        "retrieval_filter": policy.compaction.retrieval_filter,
        "lossy_summary": policy.compaction.lossy_summary,
    }


def _candidate_for_policy(
    *,
    candidate_id: str,
    policy: Policy,
    traces: list[Trace],
    labels: list[ComplexityLabel],
    weekly_scale: float,
    baseline_quality: float,
    current_pre_compaction_weekly_cost_usd: float,
    current_compaction_dollars: dict[str, float],
) -> Candidate:
    total_cost = 0.0
    total_quality_penalty = 0.0
    routing_counts: Counter[str] = Counter()
    for trace, label in zip(traces, labels, strict=True):
        tier = _route_tier(policy, label)
        routing_counts[tier] += 1
        total_cost += _trace_cost(trace) * _TIER_COST_FACTOR[tier]
        total_quality_penalty += _TIER_QUALITY_PENALTY[(tier, label)]

    # Isolate the $/week attributable to routing tier choice alone, before the
    # compaction discount is applied below -- diffed against the same
    # pre-compaction quantity for the current policy's routing. For the
    # "current" candidate itself this is identical to
    # current_pre_compaction_weekly_cost_usd, so routing_saved_usd is exactly
    # 0.0 -- current vs. current has no routing delta.
    pre_compaction_weekly_cost_usd = total_cost * weekly_scale
    routing_saved_usd = round(current_pre_compaction_weekly_cost_usd - pre_compaction_weekly_cost_usd, 6)

    enabled = _compaction_enabled(policy)
    compaction_fraction = sum(_COMPACTION_SAVINGS[key] for key, value in enabled.items() if value)
    total_cost *= max(0.35, 1.0 - compaction_fraction)
    compaction_penalty = 0.0
    if enabled["retrieval_filter"]:
        compaction_penalty += 0.012
    if enabled["lossy_summary"]:
        compaction_penalty += 0.045

    task_count = max(1, len(traces))
    estimated_quality = max(0.0, baseline_quality - (total_quality_penalty / task_count) - compaction_penalty)
    routing_breakdown = {tier: round(count / task_count, 4) for tier, count in sorted(routing_counts.items())}
    # Each lever's $/week credit is this candidate's own pre-compaction cost
    # times its savings factor, minus what the current policy already banks
    # from that same lever -- an incremental delta vs. current, not a gross
    # figure. This keeps every candidate's breakdown (read + carry + output +
    # routing) summing exactly to its own weekly_savings_usd, including the
    # "current" candidate itself (whose pre-compaction cost and enabled
    # levers are, by construction, identical to the current-policy
    # reference), which nets to 0.0 for every lever with no special-casing.
    compaction_breakdown = {
        item.id: round(
            (pre_compaction_weekly_cost_usd * _COMPACTION_SAVINGS[item.id] if enabled.get(item.id, False) else 0.0)
            - current_compaction_dollars.get(item.id, 0.0),
            6,
        )
        for item in ALL_COMPACTION_TYPES
    }
    cheap_share = routing_breakdown.get("cheap", 0.0)
    medium_share = routing_breakdown.get("medium", 0.0)
    expensive_share = routing_breakdown.get("expensive", 0.0)
    latency = max(0.55, (0.65 * cheap_share) + (0.92 * medium_share) + (1.15 * expensive_share))
    escalation = min(0.35, (cheap_share * 0.18) + (medium_share * 0.06))
    return Candidate(
        id=candidate_id,
        policy=policy,
        weekly_cost_usd=round(total_cost * weekly_scale, 6),
        estimated_quality=round(estimated_quality, 4),
        latency_mult=round(latency, 4),
        escalation_rate=round(escalation, 4),
        compaction_breakdown=compaction_breakdown,
        routing_breakdown=routing_breakdown,
        routing_saved_usd=routing_saved_usd,
    )


def _strong_only_policy() -> Policy:
    base = preset_policy("conservative")
    return Policy(
        name="Strong-only",
        preset="custom",
        quality_floor=0.99,
        confidence_required="medium",
        routing=base.routing.__class__(
            policy="prefer_strongest",
            simple="expensive",
            medium="expensive",
            hard="expensive",
            escalate_on=list(base.routing.escalate_on),
        ),
        compaction=base.compaction,
    )


def _candidate_policies() -> list[tuple[str, Policy]]:
    return [
        ("strong_only", _strong_only_policy()),
        ("conservative", preset_policy("conservative")),
        ("balanced", preset_policy("balanced")),
        ("economy", preset_policy("economy")),
        ("maximum_saving", preset_policy("maximum_saving")),
    ]


def _confidence(replayable_tasks: int, bucket_counts: dict[str, int]) -> tuple[str, str]:
    hard_count = bucket_counts.get("hard", 0)
    if replayable_tasks < 15:
        return (
            "low",
            f"Only {replayable_tasks} replayable tasks in window; recommendations are directional.",
        )
    if replayable_tasks < 50 or hard_count < 15:
        reason = f"{replayable_tasks} replayable tasks classified."
        if hard_count < 15:
            reason += f" Only {hard_count} high-complexity coding tasks in window — quality estimate is noisy here."
        return ("medium", reason)
    return (
        "high",
        f"{replayable_tasks} replayable tasks classified across simple, medium, and hard buckets.",
    )


def _estimation_metadata() -> dict[str, Any]:
    return {
        "source": "stored_lemoncrow_traces",
        "replay": "not_replayed",
        "cost_basis": "recorded_usage_costs_or_model_pricing_fallback",
        "savings_basis": "simulated_policy_costs_using_fixed_tier_and_compaction_factors",
        "quality_basis": "trace_status_and_complexity_heuristics_with_golden_corpus_check",
        "savings_are_estimates": True,
        "quality_is_estimated": True,
        "limitations": [
            "does not prove that a cheaper model would solve the same task",
            "does not measure per-session compaction savings by replaying sessions",
            "unknown model pricing can understate cost until pricing is mapped",
        ],
    }


def optimize_from_traces(
    traces: Iterable[Trace],
    *,
    current_policy: Policy,
    days: int = 7,
    host: str | None = None,
) -> OptimizationResult:
    cutoff = datetime.now(UTC) - timedelta(days=max(1, days))
    normalized_host = (host or "").strip().lower()
    filtered = [
        trace
        for trace in traces
        if _trace_created_after(trace, cutoff)
        and (not normalized_host or (trace.host or trace.agent or "").lower() == normalized_host)
    ]
    replayable = [trace for trace in filtered if trace.status in {"success", "partial", "failed"}]
    complexities = [score_trace_complexity(trace).label for trace in replayable]
    bucket_counts: dict[str, int] = {str(key): value for key, value in Counter(complexities).items()}
    weekly_scale = 7.0 / max(1, days)
    baseline_quality = sum(_status_quality(trace) for trace in replayable) / len(replayable) if replayable else 0.0

    # The current policy is the pricing reference every candidate (including
    # itself) is diffed against for both routing-tier cost and compaction $
    # credit, so it must be priced through the exact same _candidate_for_policy
    # formula as every other candidate -- never as raw observed spend, which
    # would mix an actual-cost number with every other candidate's
    # stylized-model number and make weekly_savings_usd not apples-to-apples.
    current_enabled = _compaction_enabled(current_policy)
    current_pre_compaction_weekly_cost_usd = round(
        sum(
            _trace_cost(trace) * _TIER_COST_FACTOR[_route_tier(current_policy, label)]
            for trace, label in zip(replayable, complexities, strict=True)
        )
        * weekly_scale,
        6,
    )
    current_compaction_dollars = {
        key: (current_pre_compaction_weekly_cost_usd * _COMPACTION_SAVINGS[key] if value else 0.0)
        for key, value in current_enabled.items()
    }
    current = _candidate_for_policy(
        candidate_id="current",
        policy=identify_policy(current_policy, name=f"{current_policy.name} (current)", preset=current_policy.preset),
        traces=replayable,
        labels=complexities,
        weekly_scale=weekly_scale,
        baseline_quality=baseline_quality,
        current_pre_compaction_weekly_cost_usd=current_pre_compaction_weekly_cost_usd,
        current_compaction_dollars=current_compaction_dollars,
    )
    golden = run_golden_suite(current_policy)
    candidates = [current]
    if replayable:
        for candidate_id, policy in _candidate_policies():
            candidates.append(
                _candidate_for_policy(
                    candidate_id=candidate_id,
                    policy=policy,
                    traces=replayable,
                    labels=complexities,
                    weekly_scale=weekly_scale,
                    baseline_quality=baseline_quality,
                    current_pre_compaction_weekly_cost_usd=current_pre_compaction_weekly_cost_usd,
                    current_compaction_dollars=current_compaction_dollars,
                )
            )

    if len(replayable) < 10:
        confidence, confidence_reason = _confidence(len(replayable), bucket_counts)
        return OptimizationResult(
            current_policy=current_policy,
            recommended_policy=current_policy,
            candidates=sorted(candidates, key=lambda item: (-item.estimated_quality, item.weekly_cost_usd)),
            current_candidate_id="current",
            recommended_candidate_id=None,
            confidence=confidence,
            confidence_reason=confidence_reason,
            sessions_analysed=len(filtered),
            replayable_tasks=len(replayable),
            weekly_savings_usd=0.0,
            quality_delta=0.0,
            baseline_weekly_cost_usd=current.weekly_cost_usd,
            has_recommendation=False,
            message=INSUFFICIENT_HISTORY_MESSAGE,
            bucket_counts=bucket_counts,
            golden=golden,
        )

    survivors = [
        candidate
        for candidate in candidates
        if candidate.id != "current"
        and candidate.estimated_quality >= current_policy.quality_floor
        and candidate.escalation_rate <= 0.25
    ]
    recommended_candidate = min(survivors or [current], key=lambda item: item.weekly_cost_usd)
    recommended_policy = identify_policy(
        recommended_candidate.policy,
        name="Recommended",
        preset="recommended",
    )
    confidence, confidence_reason = _confidence(len(replayable), bucket_counts)
    # Both sides of this diff are now priced through the identical stylized
    # formula (_candidate_for_policy), so weekly_savings_usd is apples-to-
    # apples -- see the invariant enforced in potential_savings_breakdown().
    weekly_savings = max(0.0, current.weekly_cost_usd - recommended_candidate.weekly_cost_usd)
    quality_delta = recommended_candidate.estimated_quality - current.estimated_quality
    return OptimizationResult(
        current_policy=current_policy,
        recommended_policy=recommended_policy,
        candidates=sorted(candidates, key=lambda item: (-item.estimated_quality, item.weekly_cost_usd)),
        current_candidate_id="current",
        recommended_candidate_id=recommended_candidate.id if recommended_candidate.id != "current" else None,
        confidence=confidence,
        confidence_reason=confidence_reason,
        sessions_analysed=len(filtered),
        replayable_tasks=len(replayable),
        weekly_savings_usd=round(weekly_savings, 6),
        quality_delta=round(quality_delta, 4),
        baseline_weekly_cost_usd=current.weekly_cost_usd,
        has_recommendation=recommended_candidate.id != "current",
        message="Recommendation is advisory and must be explicitly applied.",
        bucket_counts=bucket_counts,
        golden=golden,
    )


def optimization_history_path(root: Path) -> Path:
    return Path(root) / "optimization_history.json"


def append_history(root: Path, result: OptimizationResult) -> Path:
    path = optimization_history_path(root)
    existing = load_history(root, limit=1000)
    payload = result.to_dict()
    payload["recorded_at"] = datetime.now(UTC).isoformat()
    existing.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing[-100:], indent=2), encoding="utf-8")
    return path


def load_history(root: Path, *, limit: int = 10) -> list[dict[str, Any]]:
    path = optimization_history_path(root)
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"optimization history at {path} must be a list")
    items = [item for item in loaded if isinstance(item, dict)]
    return items[-limit:]
