"""The ``vote`` reducer: consensus by quorum over candidate answers.

For verification / consensus / repro (readonly exec mode): N candidates each
return an ``answer`` (a verdict like "survives"/"refuted", or a proposed answer).
Answers are normalized and grouped; if the largest group reaches the quorum
(``state.quorum`` if > 0, else a simple majority of the voters) it wins and its
candidates are accepted. ``merged_output`` carries the consensus answer and the
full tally. Supports the "N skeptics try to refute; keep if a majority fail to
refute" pattern.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from lemoncrow.pro.capabilities.swarm.models import (
    SwarmChildState,
    SwarmConvergenceVerdict,
    SwarmWaveDecision,
    SwarmWaveEvaluation,
)

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.swarm.reducers.base import WaveContext


def _normalize(answer: str) -> str:
    return " ".join(answer.split()).strip().lower()


class VoteReducer:
    """Quorum consensus over candidate answers."""

    name = "vote"

    def reduce(
        self,
        candidates: list[SwarmChildState],
        ctx: WaveContext,
    ) -> SwarmWaveEvaluation:
        voters = [child for child in candidates if child.status == "success" and child.answer.strip()]
        groups: dict[str, list[str]] = defaultdict(list)
        representative: dict[str, str] = {}
        for child in voters:
            key = _normalize(child.answer)
            groups[key].append(child.child_id)
            representative.setdefault(key, child.answer.strip())
        quorum = ctx.state.quorum if ctx.state.quorum > 0 else (len(voters) // 2 + 1)
        winner_key = max(groups, key=lambda key: len(groups[key])) if groups else ""
        winner_votes = len(groups[winner_key]) if winner_key else 0
        reached = bool(winner_key) and winner_votes >= quorum
        accepted = list(groups[winner_key]) if reached else []
        accepted_set = set(accepted)

        decisions: list[SwarmWaveDecision] = []
        for child in candidates:
            if child.child_id in accepted_set:
                decisions.append(
                    SwarmWaveDecision(
                        child_id=child.child_id,
                        verdict="accept",
                        rationale=f"In the consensus group ({winner_votes}/{len(voters)} >= quorum {quorum}).",
                    )
                )
            elif child.status != "success" or not child.answer.strip():
                decisions.append(
                    SwarmWaveDecision(
                        child_id=child.child_id,
                        verdict="reject",
                        rationale="No usable answer to vote with.",
                    )
                )
            else:
                decisions.append(
                    SwarmWaveDecision(
                        child_id=child.child_id,
                        verdict="reject",
                        rationale="Outside the consensus group.",
                    )
                )

        tally = {representative[key]: len(ids) for key, ids in groups.items()}
        if reached:
            verdict: SwarmConvergenceVerdict = "converged"
            summary = f"Consensus reached: {winner_votes}/{len(voters)} agreed (quorum {quorum})."
        else:
            verdict = "stagnating"
            summary = f"No consensus: top answer had {winner_votes}/{len(voters)} votes (quorum {quorum})."
        merged = {
            "consensus": representative.get(winner_key, "") if reached else None,
            "votes": winner_votes,
            "voters": len(voters),
            "quorum": quorum,
            "tally": tally,
        }
        return SwarmWaveEvaluation(
            status="completed",
            evaluator_backend=ctx.state.evaluator_backend,
            evaluator_model=ctx.state.evaluator_model,
            summary=summary,
            verdict=verdict,
            candidate_order=[child.child_id for child in candidates],
            accepted_child_ids=accepted,
            rejected_child_ids=[item.child_id for item in decisions if item.verdict == "reject"],
            deferred_child_ids=[],
            decisions=decisions,
            next_wave_directives=[],
            merged_output=merged,
            finished_at=datetime.now(UTC),
        )


__all__ = ["VoteReducer"]
