"""The ``best`` reducer: rank candidates by a fitness, accept the top one(s).

Two fitness sources:
- **heuristic** -- the run-quality scorer (``_score_child`` / ``rank_children``)
  used today both as each child's score and as the deterministic overlap-aware
  fallback selection. Moved here verbatim from ``capability.py``.
- **measured** -- a project-supplied ``FitnessSpec`` command run per worktree
  (added in Phase 2).

The heuristic scorer lives here so there is a single source of truth shared by
the per-child score (``run_child_once``) and the deterministic fallback used by
the ``merge`` reducer when no semantic backend is available.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lemoncrow.pro.capabilities.swarm.models import (
    SwarmChildState,
    SwarmConvergenceVerdict,
    SwarmRunState,
    SwarmValidationCheck,
    SwarmWaveDecision,
    SwarmWaveEvaluation,
)

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.swarm.reducers.base import WaveContext


def _is_structural_validation(check: SwarmValidationCheck) -> bool:
    return check.name.startswith("structural-")


def _has_non_structural_passing_validation(child: SwarmChildState) -> bool:
    return any(item.passed and not _is_structural_validation(item) for item in child.validation_results)


def _score_child(child: SwarmChildState) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if child.status == "success":
        score += 100.0
        reasons.append("+100 successful child run")
    elif child.status == "stopped":
        score -= 30.0
        reasons.append("-30 stopped before completion")
    else:
        score -= 60.0
        reasons.append("-60 failed child run")
    validation_passes = sum(1 for item in child.validation_results if item.passed)
    validation_failures = sum(1 for item in child.validation_results if not item.passed)
    if not child.validation_results:
        score -= 12.0
        reasons.append("-12 no validation evidence")
    only_structural_validation = bool(child.validation_results) and all(
        _is_structural_validation(item) for item in child.validation_results
    )
    if validation_passes:
        delta = validation_passes * (3.0 if only_structural_validation else 15.0)
        score += delta
        if only_structural_validation:
            reasons.append(f"+{delta:.1f} structural validation checks passed")
        else:
            reasons.append(f"+{delta:.1f} validation checks passed")
    if validation_failures:
        delta = validation_failures * 25.0
        score -= delta
        reasons.append(f"-{delta:.1f} validation checks failed")
    if child.files_changed:
        score += 5.0
        reasons.append("+5 produced a git diff")
    else:
        score -= 10.0
        reasons.append("-10 no files changed")
    file_penalty = min(len(child.files_changed), 50) * 0.2
    if file_penalty:
        score -= file_penalty
        reasons.append(f"-{file_penalty:.1f} changed-file penalty")
    if child.cost_usd > 0:
        cost_penalty = child.cost_usd * 10.0
        score -= cost_penalty
        reasons.append(f"-{cost_penalty:.2f} cost penalty")
    if child.duration_seconds > 0:
        duration_penalty = min(child.duration_seconds / 120.0, 10.0)
        score -= duration_penalty
        reasons.append(f"-{duration_penalty:.2f} duration penalty")
    return round(score, 3), reasons


def rank_children(children: list[SwarmChildState]) -> list[SwarmChildState]:
    for child in children:
        score, breakdown = _score_child(child)
        child.score = score
        child.score_breakdown = breakdown
    return sorted(
        children,
        key=lambda item: (
            item.score if item.score is not None else float("-inf"),
            sum(1 for check in item.validation_results if check.passed),
            -(len(item.files_changed)),
        ),
        reverse=True,
    )


def _measured_wave_evaluation(
    state: SwarmRunState,
    candidates: list[SwarmChildState],
) -> SwarmWaveEvaluation:
    """Score candidates by a measured ``FitnessSpec`` and accept the best.

    Per candidate: run the correctness gate (reject on fail), then the metric
    command, parse it, and rank by improvement over the baseline (honoring
    ``direction``). Accept the single best candidate that beats the baseline by
    ``improve_margin``. ``metric``/``gate_passed`` are recorded on each child.
    """
    from lemoncrow.pro.capabilities.swarm.fitness import (
        evaluate_candidate,
        improvement,
        rank_key,
        resolve_baseline,
    )

    spec = state.fitness_spec
    assert spec is not None  # reducer only routes here when a fitness is set
    baseline = resolve_baseline(spec)
    decisions: list[SwarmWaveDecision] = []
    pre_rejected: list[str] = []
    eligible: list[tuple[float, SwarmChildState, float]] = []  # (sort_value, child, metric)

    for child in candidates:
        if child.status != "success":
            child.metric = None
            child.gate_passed = None
            decisions.append(
                SwarmWaveDecision(child_id=child.child_id, verdict="reject", rationale="Child run did not succeed.")
            )
            pre_rejected.append(child.child_id)
            continue
        result = evaluate_candidate(spec, Path(child.worktree_path))
        child.gate_passed = result.gate_passed
        child.metric = result.metric
        if not result.gate_passed:
            decisions.append(
                SwarmWaveDecision(
                    child_id=child.child_id,
                    verdict="reject",
                    rationale=f"Correctness gate failed: {result.gate_detail}",
                )
            )
            pre_rejected.append(child.child_id)
            continue
        if result.metric is None:
            decisions.append(
                SwarmWaveDecision(
                    child_id=child.child_id,
                    verdict="reject",
                    rationale=f"Metric did not parse: {result.parse_error}",
                )
            )
            pre_rejected.append(child.child_id)
            continue
        sort_value = (
            improvement(spec, result.metric, baseline) if baseline is not None else rank_key(spec, result.metric)
        )
        eligible.append((sort_value, child, result.metric))

    eligible.sort(key=lambda row: row[0], reverse=True)
    accepted: list[str] = []
    margin_note = ""
    if eligible:
        _best_sort, best_child, best_metric = eligible[0]
        accept_ok = True
        if baseline is not None:
            gain = improvement(spec, best_metric, baseline)
            accept_ok = gain >= spec.improve_margin
            margin_note = f" (metric {best_metric:g} vs baseline {baseline:g}, improvement {gain:+g})"
        if accept_ok:
            accepted.append(best_child.child_id)
            decisions.append(
                SwarmWaveDecision(
                    child_id=best_child.child_id,
                    verdict="accept",
                    rationale=f"Best measured candidate for the objective{margin_note}.",
                )
            )
            for _sv, other, _metric in eligible[1:]:
                decisions.append(
                    SwarmWaveDecision(
                        child_id=other.child_id,
                        verdict="reject",
                        rationale="Did not beat the wave's best measured candidate.",
                    )
                )
        else:
            for _sv, other, _metric in eligible:
                decisions.append(
                    SwarmWaveDecision(
                        child_id=other.child_id,
                        verdict="reject",
                        rationale=f"Did not beat the baseline by the required margin {spec.improve_margin:g}.",
                    )
                )

    if accepted:
        verdict: SwarmConvergenceVerdict = "continue"
        summary = f"Accepted measured winner {accepted[0]}{margin_note}."
        directives = [f"Push further toward the objective: {spec.objective}"] if spec.objective else []
    elif eligible:
        verdict = "converged"
        summary = "No candidate beat the baseline within the improvement margin; measured search converged."
        directives = []
    else:
        verdict = "stagnating"
        summary = "No candidate produced a parseable metric with a passing gate."
        directives = []

    candidate_order = [child.child_id for _sv, child, _metric in eligible] + pre_rejected
    return SwarmWaveEvaluation(
        status="completed",
        evaluator_backend=state.evaluator_backend,
        evaluator_model=state.evaluator_model,
        summary=summary,
        verdict=verdict,
        candidate_order=candidate_order,
        accepted_child_ids=accepted,
        rejected_child_ids=[item.child_id for item in decisions if item.verdict == "reject"],
        deferred_child_ids=[],
        decisions=decisions,
        next_wave_directives=directives,
        finished_at=datetime.now(UTC),
    )


class BestReducer:
    """Best-of-N selection by a fitness.

    With a ``FitnessSpec`` on the run (``state.fitness_spec``) this measures each
    candidate (gate + metric) and accepts the single best that beats the
    baseline -- the ``optimize`` / ``tune`` capability. Without one it falls back
    to the heuristic deterministic selection already used by the ``merge``
    reducer, so behavior is unchanged for non-measured jobs.
    """

    name = "best"

    def reduce(
        self,
        candidates: list[SwarmChildState],
        ctx: WaveContext,
    ) -> SwarmWaveEvaluation:
        if ctx.state.fitness_spec is not None:
            return _measured_wave_evaluation(ctx.state, candidates)
        from lemoncrow.pro.capabilities.swarm.capability import _fallback_wave_evaluation

        return _fallback_wave_evaluation(ctx.state, candidates)


__all__ = [
    "BestReducer",
    "_has_non_structural_passing_validation",
    "_is_structural_validation",
    "_measured_wave_evaluation",
    "_score_child",
    "rank_children",
]
