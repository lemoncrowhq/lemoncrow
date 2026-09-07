"""The ``merge`` reducer: the LLM wave-evaluator (today's default for solve-task).

It accepts multiple compatible candidates, rejects weaker duplicates and
conflicting alternatives, emits next-wave directives, and judges convergence --
the semantic half of today's selection. It wraps the existing engine evaluator
(``capability._evaluate_wave``) so behavior is byte-identical; the heavy evidence
builders and the deterministic fallback remain in ``capability.py`` and are
invoked unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.swarm.models import SwarmChildState, SwarmWaveEvaluation
    from lemoncrow.pro.capabilities.swarm.reducers.base import WaveContext


class MergeReducer:
    """Semantic merge-compatible selection + iteration control (the default)."""

    name = "merge"

    def reduce(
        self,
        candidates: list[SwarmChildState],
        ctx: WaveContext,
    ) -> SwarmWaveEvaluation:
        from lemoncrow.pro.capabilities.swarm.capability import _evaluate_wave

        return _evaluate_wave(ctx.state, ctx.wave, candidates)


__all__ = ["MergeReducer"]
