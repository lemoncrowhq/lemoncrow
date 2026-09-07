"""The ``union`` reducer: collect every candidate's findings, dedup by signature.

For search / discovery / audit jobs (readonly exec mode): N agents each search a
different way and the de-duplicated union of their findings is the result. Every
successful candidate is accepted; ``merged_output`` is the de-duplicated finding
list (best-effort signature derived from kind|file|line|title when a candidate
leaves ``signature`` empty).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from lemoncrow.pro.capabilities.swarm.models import (
    Finding,
    SwarmChildState,
    SwarmWaveDecision,
    SwarmWaveEvaluation,
)

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.swarm.reducers.base import WaveContext


def _signature(finding: Finding) -> str:
    if finding.signature.strip():
        return finding.signature.strip().lower()
    line = "" if finding.line is None else str(finding.line)
    return "|".join([finding.kind, finding.file, line, finding.title]).strip().lower()


class UnionReducer:
    """Dedup-by-signature union of all candidates' findings."""

    name = "union"

    def reduce(
        self,
        candidates: list[SwarmChildState],
        ctx: WaveContext,
    ) -> SwarmWaveEvaluation:
        deduped: dict[str, Finding] = {}
        duplicate_count = 0
        contributions: dict[str, int] = {}
        for child in candidates:
            if child.status != "success":
                continue
            for finding in child.findings:
                sig = _signature(finding)
                if not sig:
                    continue
                if sig in deduped:
                    duplicate_count += 1
                    continue
                deduped[sig] = finding
                contributions[child.child_id] = contributions.get(child.child_id, 0) + 1

        accepted: list[str] = []
        rejected: list[str] = []
        decisions: list[SwarmWaveDecision] = []
        for child in candidates:
            if child.status == "success":
                accepted.append(child.child_id)
                decisions.append(
                    SwarmWaveDecision(
                        child_id=child.child_id,
                        verdict="accept",
                        rationale=f"Contributed {contributions.get(child.child_id, 0)} unique finding(s).",
                    )
                )
            else:
                rejected.append(child.child_id)
                decisions.append(
                    SwarmWaveDecision(
                        child_id=child.child_id,
                        verdict="reject",
                        rationale="Child run did not succeed.",
                    )
                )
        merged = [finding.model_dump(mode="json") for finding in deduped.values()]
        summary = (
            f"Union of {len(merged)} unique finding(s) from {len(accepted)} candidate(s); "
            f"deduplicated {duplicate_count}."
        )
        return SwarmWaveEvaluation(
            status="completed",
            evaluator_backend=ctx.state.evaluator_backend,
            evaluator_model=ctx.state.evaluator_model,
            summary=summary,
            verdict="converged",
            candidate_order=[child.child_id for child in candidates],
            accepted_child_ids=accepted,
            rejected_child_ids=rejected,
            deferred_child_ids=[],
            decisions=decisions,
            next_wave_directives=[],
            merged_output=merged,
            finished_at=datetime.now(UTC),
        )


__all__ = ["UnionReducer"]
