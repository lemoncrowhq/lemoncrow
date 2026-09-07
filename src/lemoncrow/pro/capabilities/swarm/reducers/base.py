"""Reducer interface + registry for the swarm engine.

A *reducer* decides how the parallel candidates produced in a wave are combined:
which to accept / reject / defer, the best-first ranking, whether the search has
converged, and what the next wave should explore. It is the one pluggable knob
that turns the swarm from a single solve-task tool into a general
fan-out -> reduce -> iterate primitive.

The canonical reduce outcome is the existing ``SwarmWaveEvaluation`` model. Its
fields map onto the spec's ``ReduceOutcome``:

    accepted_child_ids / rejected_child_ids / deferred_child_ids
        == accepted / rejected / deferred
    candidate_order        == ranking (best-first)
    verdict                == converged signal (continue|converged|...)
    next_wave_directives   == next_wave_directives
    summary                == summary
    merged_output          == merged_output (union list / synthesized answer)

Today's behavior == the ``merge`` reducer (the LLM wave-evaluator). It stays the
default so existing runs are byte-identical; every other reducer is additive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

# Runtime (not TYPE_CHECKING) import: mypyc-compiled dataclasses/protocols resolve
# field/annotation types at runtime, so these must be importable when the compiled
# module loads. Imported from the open contract directly to avoid the shim.
from lemoncrow.core.capabilities.swarm_contract import (
    SwarmChildState,
    SwarmRunState,
    SwarmWaveEvaluation,
    SwarmWaveState,
)


@dataclass(slots=True)
class WaveContext:
    """Everything a reducer needs about the wave being reduced."""

    state: SwarmRunState
    wave: SwarmWaveState


class Reducer(Protocol):
    """Combine a wave's candidates into an integration decision."""

    name: str

    def reduce(
        self,
        candidates: list[SwarmChildState],
        ctx: WaveContext,
    ) -> SwarmWaveEvaluation: ...


REDUCERS: dict[str, Reducer] = {}


def register_reducer(reducer: Reducer) -> Reducer:
    """Register ``reducer`` under its ``name`` (last registration wins)."""

    REDUCERS[reducer.name] = reducer
    return reducer


def get_reducer(name: str) -> Reducer:
    """Look up a registered reducer by name."""

    try:
        return REDUCERS[name]
    except KeyError:
        raise KeyError(f"unknown swarm reducer {name!r}; registered: {sorted(REDUCERS)}") from None


__all__ = [
    "REDUCERS",
    "Reducer",
    "WaveContext",
    "get_reducer",
    "register_reducer",
]
