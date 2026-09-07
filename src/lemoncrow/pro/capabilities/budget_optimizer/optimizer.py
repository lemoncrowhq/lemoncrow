"""Prompt budget optimizer — context knapsack packing.

Given candidate content blocks from multiple capabilities (context
reuse chains, memory fragments, loop-rescue suggestions, tool summaries)
each annotated with an estimated token cost and expected utility, this
module selects the optimal subset that maximises total utility within a
fixed token budget.

Solver priority
---------------
1. **OR-Tools CP-SAT** — exact 0/1 knapsack, scales to ~500 items,
   2-second time limit for safety.
2. **Greedy** — sort by utility-per-token (with a source-diversity
   bonus), then take greedily.  Always available; used as fallback when
   OR-Tools is absent or the problem is too large.

Usage::

    from lemoncrow.pro.capabilities.budget_optimizer import (
        ContextBlock, PromptBudgetOptimizer,
    )

    opt = PromptBudgetOptimizer()
    blocks = [
        ContextBlock("r1", "prior chain ...", token_cost=120, utility=0.9, source="context_reuse"),
        ContextBlock("m1", "memory frag ...", token_cost=80,  utility=0.7, source="semantic_memory"),
        ContextBlock("l1", "diff context ...", token_cost=40,  utility=0.6, source="tool_supervision"),
    ]
    plan = opt.solve(blocks, token_budget=200)
    print(plan.to_dict())
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

# External utility source: maps a ContextBlock to a replacement utility in
# [0, 1], or a plain mapping of block id -> utility. Used to drive selection
# from perplexity/entropy signals (T10) instead of the static block utility.
UtilitySource = Callable[["ContextBlock"], float] | Mapping[str, float]

_PERPLEXITY_FLAG = "LEMONCROW_PERPLEXITY_COMPRESSION"


def _perplexity_compression_enabled() -> bool:
    """True when the default-off ``LEMONCROW_PERPLEXITY_COMPRESSION`` flag is set."""
    raw = os.environ.get(_PERPLEXITY_FLAG)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_utility(block: ContextBlock, source: UtilitySource) -> float:
    """Look up *block*'s external utility, clamped to ``[0, 1]``.

    Falls back to the block's own static utility when the source has no value
    for it (callable returning ``None`` or a mapping missing the id).
    """
    value: float | None
    if callable(source):
        value = source(block)
    else:
        value = source.get(block.id)
    if value is None:
        return block.utility
    return max(0.0, min(1.0, float(value)))


# ---------------------------------------------------------------------------
# Optional deps (import-guarded)

try:
    from ortools.sat.python import cp_model as _cp_model

    _HAS_ORTOOLS = True
except ImportError:  # pragma: no cover
    _cp_model = None  # type: ignore[assignment]
    _HAS_ORTOOLS = False

# ---------------------------------------------------------------------------
# Data models


@dataclass
class ContextBlock:
    """A single content block that is a candidate for prompt inclusion.

    Attributes:
        id:          Unique identifier for this block.
        content:     The actual text content.
        token_cost:  Estimated token count (used as knapsack weight).
        utility:     Expected prompt utility in ``[0, 1]``.
        source:      Originating capability (e.g. ``'context_reuse'``).
        metadata:    Free-form extra data from the emitting capability.
    """

    id: str
    content: str
    token_cost: int
    utility: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def utility_per_token(self) -> float:
        """Efficiency metric used by the greedy solver."""
        if self.token_cost <= 0:
            return 0.0
        return self.utility / self.token_cost


@dataclass
class BudgetPlan:
    """Result of a prompt budget optimisation run.

    Attributes:
        selected:       Blocks chosen for inclusion.
        dropped:        Blocks left out (over budget or low utility).
        total_tokens:   Sum of ``token_cost`` for selected blocks.
        total_utility:  Sum of ``utility`` for selected blocks.
        solver_used:    Either ``'ortools'`` or ``'greedy'``.
    """

    selected: list[ContextBlock]
    dropped: list[ContextBlock]
    total_tokens: int
    total_utility: float
    solver_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_ids": [b.id for b in self.selected],
            "dropped_ids": [b.id for b in self.dropped],
            "total_tokens": self.total_tokens,
            "total_utility": round(self.total_utility, 4),
            "solver_used": self.solver_used,
            "selected_count": len(self.selected),
        }


# ---------------------------------------------------------------------------
# Internal solver helpers


def _greedy_knapsack(
    blocks: list[ContextBlock],
    budget: int,
    diversity_bonus: float,
) -> tuple[list[ContextBlock], str]:
    """Sort by utility/token (+ source-diversity bonus) and take greedily."""
    source_seen: set[str] = set()
    scored: list[tuple[float, ContextBlock]] = []
    for b in blocks:
        bonus = diversity_bonus if b.source not in source_seen else 0.0
        scored.append((b.utility_per_token() + bonus, b))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[ContextBlock] = []
    remaining = budget
    source_seen.clear()
    for _, block in scored:
        if block.token_cost <= remaining:
            selected.append(block)
            remaining -= block.token_cost
            source_seen.add(block.source)
    return selected, "greedy"


def _ortools_knapsack(
    blocks: list[ContextBlock],
    budget: int,
    diversity_bonus: float,
) -> tuple[list[ContextBlock], str]:
    """Exact 0/1 knapsack via OR-Tools CP-SAT.

    Falls back to greedy if CP-SAT cannot produce a feasible solution
    within the 2-second time limit.
    """
    cp = _cp_model.CpModel()
    n = len(blocks)

    # Decision variables — one per block
    x = [cp.new_bool_var(f"x_{i}") for i in range(n)]

    # Objective: maximise total utility (scaled to integer)
    _SCALE = 1_000
    cp.maximize(sum(round(b.utility * _SCALE) * x[i] for i, b in enumerate(blocks)))

    # Budget constraint
    cp.add(sum(b.token_cost * x[i] for i, b in enumerate(blocks)) <= budget)

    solver = _cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0
    status = solver.solve(cp)

    feasible_statuses: set[Any] = {_cp_model.OPTIMAL, _cp_model.FEASIBLE}
    if status in feasible_statuses:
        selected = [blocks[i] for i in range(n) if solver.value(x[i])]
        return selected, "ortools"

    # No feasible solution found — fall back to greedy
    return _greedy_knapsack(blocks, budget, diversity_bonus)


# ---------------------------------------------------------------------------
# Public optimizer


class PromptBudgetOptimizer:
    """Solve the context knapsack for optimal prompt composition.

    The optimizer selects the subset of ``blocks`` that maximises the
    total utility while keeping the total token cost within
    ``token_budget``.  A ``diversity_bonus`` rewards blocks from
    capability sources not yet represented in the selection.

    Args:
        diversity_bonus: Extra utility/token score added when a block's
            source has not yet been seen in the selection pass.
            Defaults to ``0.1``.
    """

    def __init__(self, diversity_bonus: float = 0.1) -> None:
        self._diversity_bonus = diversity_bonus

    def solve(
        self,
        blocks: list[ContextBlock],
        token_budget: int,
        *,
        diversity_bonus: float | None = None,
        utility_source: UtilitySource | None = None,
    ) -> BudgetPlan:
        """Return a :class:`BudgetPlan` maximising utility within *token_budget*.

        Args:
            blocks:          Candidate context blocks.
            token_budget:    Maximum total token count to include.
            diversity_bonus: Override instance default.
            utility_source:  Optional external per-block utility source
                (callable ``block -> float`` or ``{id: utility}`` mapping),
                e.g. perplexity/entropy scores from T10. Applied **only** when
                the default-off ``LEMONCROW_PERPLEXITY_COMPRESSION`` flag is set;
                when the flag is off the static ``block.utility`` is used and
                behavior is identical to before.

        Returns:
            A :class:`BudgetPlan` with the selected / dropped blocks.
        """
        bonus = diversity_bonus if diversity_bonus is not None else self._diversity_bonus

        if not blocks:
            return BudgetPlan(
                selected=[],
                dropped=[],
                total_tokens=0,
                total_utility=0.0,
                solver_used="greedy",
            )

        # External utility override (T10) is gated behind the default-off flag
        # so existing callers keep their exact static-utility behavior.
        if utility_source is not None and _perplexity_compression_enabled():
            blocks = [replace(b, utility=_resolve_utility(b, utility_source)) for b in blocks]

        feasible = [b for b in blocks if b.token_cost <= token_budget]
        infeasible = [b for b in blocks if b.token_cost > token_budget]

        if not feasible:
            return BudgetPlan(
                selected=[],
                dropped=list(blocks),
                total_tokens=0,
                total_utility=0.0,
                solver_used="greedy",
            )

        # Choose solver
        if _HAS_ORTOOLS and len(feasible) <= 500:
            selected, solver_name = _ortools_knapsack(feasible, token_budget, bonus)
        else:
            selected, solver_name = _greedy_knapsack(feasible, token_budget, bonus)  # pragma: no cover

        selected_ids = {b.id for b in selected}
        dropped = [b for b in feasible if b.id not in selected_ids] + infeasible

        return BudgetPlan(
            selected=selected,
            dropped=dropped,
            total_tokens=sum(b.token_cost for b in selected),
            total_utility=sum(b.utility for b in selected),
            solver_used=solver_name,
        )
