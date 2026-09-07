"""Complexity-scored model-tier routing (Workstream 6 / N1).

The legacy :class:`~lemoncrow.pro.capabilities.model_routing.router.ModelRouter`
optimizes tokens-per-call but never routes the call itself to a cheaper model.
This module adds a *pure, unit-testable* complexity score (weighted 0-100)
derived from signals the router already has access to via ``session_state``
(retrieval-set size, symbol / cross-file counts, task size, prior errors) and
maps that score onto a coarse model tier (cheap / standard / strong).

Design invariants:

* **Pure.** :func:`complexity_score` and :func:`tier_for_complexity` are pure
  functions of their inputs -- no environment, no I/O, no clock.
* **Default-safe / opt-in.** Nothing here runs unless a caller explicitly opts
  in (param or ``LEMONCROW_TIER_ROUTING`` env). The default routing decision is
  left untouched so existing tests pass unchanged.
* **Escalation preserved.** Genuinely hard work (high complexity, escalation
  flags, cross-project work, repeated errors) is *never* downgraded. The tier
  mapping only ever steps **down** for clearly simple work; it steps **up** the
  moment any risk signal fires.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from lemoncrow.core.environment import bool_env

# Coarse tier exposed by this module.  Maps 1:1 onto the legacy ModelTier
# (cheap -> cheap, standard -> medium, strong -> expensive) but uses
# enterprise-routing language for the complexity surface.
ComplexityTier = Literal["cheap", "standard", "strong"]

TIER_ROUTING_ENV_VAR = "LEMONCROW_TIER_ROUTING"

# Weight budget sums to 100.  Each signal contributes a normalized 0..weight
# fraction; the total is clamped to [0, 100].
_WEIGHT_RETRIEVAL = 25
_WEIGHT_SYMBOLS = 20
_WEIGHT_CROSS_FILE = 25
_WEIGHT_TASK_SIZE = 15
_WEIGHT_PRIOR_ERRORS = 15

# Saturation points: the signal value at which a signal contributes its full
# weight.  Chosen so that ordinary single-file edits land in the cheap band and
# large cross-file / multi-symbol work climbs into the strong band.
_RETRIEVAL_SATURATION = 12  # retrieved context blocks / refs
_SYMBOLS_SATURATION = 10  # distinct symbols touched
_CROSS_FILE_SATURATION = 6  # distinct files in scope
_TASK_SIZE_SATURATION = 1_200  # task-description characters
_PRIOR_ERRORS_SATURATION = 3  # consecutive prior errors

# Band thresholds for the base (risk-free) tier mapping.
_CHEAP_CEILING = 30  # score < 30 -> cheap (silent step-down)
_STRONG_FLOOR = 65  # score >= 65 -> strong

_COMPLEXITY_TO_MODEL_TIER: dict[ComplexityTier, str] = {
    "cheap": "cheap",
    "standard": "medium",
    "strong": "expensive",
}


@dataclass(frozen=True)
class ComplexitySignals:
    """Inputs to the complexity score.  All fields default to the cheapest case.

    These are the signals the router already computes or receives via
    ``session_state`` -- this dataclass just gives them a typed, pure home.
    """

    retrieval_set_size: int = 0
    symbol_count: int = 0
    cross_file_count: int = 0
    task_size_chars: int = 0
    prior_errors: int = 0
    # Hard escalation signals: when any is true the work is treated as
    # genuinely hard and must not be routed below the strong tier.
    escalate: bool = False
    cross_project: bool = False


@dataclass(frozen=True)
class ComplexityRouteResult:
    """Result of a complexity-scored tier decision."""

    score: int
    tier: ComplexityTier
    model_tier: str
    base_tier: ComplexityTier
    stepped_up: bool
    reasons: list[str]


def _normalized(value: int, saturation: int, weight: int) -> float:
    """Return ``value`` scaled into ``0..weight`` and saturating at ``saturation``."""
    if value <= 0 or saturation <= 0:
        return 0.0
    return min(1.0, value / saturation) * weight


def complexity_score(signals: ComplexitySignals) -> int:
    """Pure weighted complexity score in ``[0, 100]``.

    Higher means harder work that warrants a stronger (more expensive) model.
    The function is deterministic and side-effect free.
    """
    total = (
        _normalized(signals.retrieval_set_size, _RETRIEVAL_SATURATION, _WEIGHT_RETRIEVAL)
        + _normalized(signals.symbol_count, _SYMBOLS_SATURATION, _WEIGHT_SYMBOLS)
        + _normalized(signals.cross_file_count, _CROSS_FILE_SATURATION, _WEIGHT_CROSS_FILE)
        + _normalized(signals.task_size_chars, _TASK_SIZE_SATURATION, _WEIGHT_TASK_SIZE)
        + _normalized(signals.prior_errors, _PRIOR_ERRORS_SATURATION, _WEIGHT_PRIOR_ERRORS)
    )
    return max(0, min(100, round(total)))


def _base_tier(score: int) -> ComplexityTier:
    if score < _CHEAP_CEILING:
        return "cheap"
    if score >= _STRONG_FLOOR:
        return "strong"
    return "standard"


def tier_for_complexity(signals: ComplexitySignals) -> ComplexityRouteResult:
    """Map complexity signals to a tier with step-up / silent step-down.

    * **Silent step-down**: clearly simple work (low score, no risk signals)
      drops to the cheap tier without ceremony.
    * **Step-up confirmation**: when a cheap/standard pick collides with a hard
      signal (escalation flag, cross-project work, repeated errors) the tier is
      escalated and the reason is recorded so the caller can confirm the bump.
    * **Escalation preserved**: hard signals can only *raise* the tier; nothing
      here ever downgrades genuinely hard work.
    """
    score = complexity_score(signals)
    base = _base_tier(score)
    reasons = [f"complexity_score={score} -> base_tier={base}"]

    tier: ComplexityTier = base
    stepped_up = False

    # Hard escalation signals force the strong tier regardless of score.
    if signals.escalate:
        tier = "strong"
        reasons.append("step_up: escalate flag set")
    if signals.cross_project:
        tier = "strong"
        reasons.append("step_up: cross-project work")
    # Repeated prior errors are a strong risk signal: never stay cheap.
    if signals.prior_errors >= _PRIOR_ERRORS_SATURATION and tier == "cheap":
        tier = "standard"
        reasons.append("step_up: repeated prior errors lift cheap -> standard")

    stepped_up = _tier_rank(tier) > _tier_rank(base)
    if not stepped_up and tier != base:  # pragma: no cover - defensive
        # Mapping must never silently downgrade below the base tier.
        tier = base

    if tier == base and base == "cheap":
        reasons.append("step_down: simple work routed to cheap tier")

    return ComplexityRouteResult(
        score=score,
        tier=tier,
        model_tier=_COMPLEXITY_TO_MODEL_TIER[tier],
        base_tier=base,
        stepped_up=stepped_up,
        reasons=reasons,
    )


def _tier_rank(tier: ComplexityTier) -> int:
    return {"cheap": 0, "standard": 1, "strong": 2}[tier]


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


def _count(value: Any) -> int:
    """Best-effort cardinality for list/collection-or-int session-state fields."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        return _safe_int(value)
    if isinstance(value, Mapping):
        return len(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def signals_from_state(task_text: str, state: Mapping[str, Any]) -> ComplexitySignals:
    """Extract :class:`ComplexitySignals` from a router ``session_state`` mapping.

    Pure with respect to ``state`` -- it only reads.  Accepts both count fields
    (``symbol_count``) and collection fields (``refs``, ``changed_files``).
    """
    retrieval = _count(state.get("retrieval_set_size"))
    if retrieval == 0:
        retrieval = _count(state.get("refs")) or _count(state.get("playbook_count"))

    symbols = _count(state.get("symbol_count")) or _count(state.get("symbols"))

    cross_file = _count(state.get("cross_file_count"))
    if cross_file == 0:
        cross_file = _count(state.get("changed_files")) or _count(state.get("files"))

    task_size = _count(state.get("task_size_chars")) or len(task_text or "")

    return ComplexitySignals(
        retrieval_set_size=retrieval,
        symbol_count=symbols,
        cross_file_count=cross_file,
        task_size_chars=task_size,
        prior_errors=_safe_int(state.get("prior_errors")),
        escalate=bool(state.get("escalate")),
        cross_project=bool(state.get("cross_project")),
    )


def tier_routing_enabled(state: Mapping[str, Any] | None = None) -> bool:
    """Whether complexity-scored tier routing is opted in for this decision.

    Opt-in via an explicit ``tier_routing`` flag in ``session_state`` *or* the
    ``LEMONCROW_TIER_ROUTING`` environment variable.  Default is off, so the
    baseline routing decision is unchanged.
    """
    if state is not None and state.get("tier_routing"):
        return True
    return bool_env(TIER_ROUTING_ENV_VAR, default=False)


__all__ = [
    "TIER_ROUTING_ENV_VAR",
    "ComplexityRouteResult",
    "ComplexitySignals",
    "ComplexityTier",
    "complexity_score",
    "signals_from_state",
    "tier_for_complexity",
    "tier_routing_enabled",
]
