"""Prefix cache planner.

Wraps compile_prompt to produce a PrefixCachePlan with explicit
static_prefix / dynamic_state split and per-turn cache diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from lemoncrow.pro.capabilities.prompt_compilation.compiler import compile_prompt
from lemoncrow.pro.capabilities.prompt_compilation.models import (
    BlockKind,
    PromptBlock,
    Stability,
)


@dataclass(frozen=True)
class PrefixCachePlan:
    """Result of planning a single agent turn for prefix-cache safety.

    Attributes:
        static_prefix: Blocks with STATIC/SESSION stability — these are the
            provider KV-cache anchor. Mutation here invalidates the cache.
        dynamic_state: BRANCH/TURN/VOLATILE blocks — changes every turn.
        prefix_hash: SHA-256 of static_prefix content. Compare across turns
            to detect cache invalidations.
        prefix_tokens: Estimated token count of static_prefix.
        dynamic_tokens: Estimated token count of dynamic_state.
        total_tokens: Sum of prefix and dynamic tokens.
        invalidated_reason: Non-empty if this plan would invalidate a prior
            prefix (populated by PrefixCachePlanner.plan_with_history).
    """

    static_prefix: tuple[PromptBlock, ...]
    dynamic_state: tuple[PromptBlock, ...]
    prefix_hash: str
    prefix_tokens: int
    dynamic_tokens: int
    total_tokens: int
    invalidated_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_hash": self.prefix_hash,
            "prefix_tokens": self.prefix_tokens,
            "dynamic_tokens": self.dynamic_tokens,
            "total_tokens": self.total_tokens,
            "static_prefix_blocks": len(self.static_prefix),
            "dynamic_state_blocks": len(self.dynamic_state),
            "invalidated_reason": self.invalidated_reason,
        }


_STATIC_STABILITIES = {Stability.STATIC, Stability.SESSION}
_BRANCH_STABILITIES = {Stability.BRANCH}


class PrefixCachePlanner:
    """Plans prompt assembly for maximum provider-side prefix cache reuse.

    Usage::

        planner = PrefixCachePlanner()
        plan = planner.plan(blocks)
        # plan.static_prefix → stable anchor for KV cache
        # plan.dynamic_state → per-turn tail
        # plan.prefix_hash   → compare across turns to detect invalidations
    """

    def plan(
        self,
        blocks: Iterable[PromptBlock],
        *,
        tail_budget_tokens: int | None = None,
    ) -> PrefixCachePlan:
        """Compile blocks and split into static prefix + dynamic state."""
        compiled = compile_prompt(blocks, tail_budget_tokens=tail_budget_tokens)
        all_blocks = compiled.blocks
        prefix_end = compiled.prefix_end_index

        static_prefix = tuple(b for b in all_blocks[: prefix_end + 1] if b.stability in _STATIC_STABILITIES)
        dynamic_state = tuple(b for b in all_blocks if b.stability not in _STATIC_STABILITIES)

        return PrefixCachePlan(
            static_prefix=static_prefix,
            dynamic_state=dynamic_state,
            prefix_hash=compiled.stable_prefix_hash,
            prefix_tokens=compiled.stable_prefix_tokens,
            dynamic_tokens=compiled.dynamic_tail_tokens,
            total_tokens=compiled.stable_prefix_tokens + compiled.dynamic_tail_tokens,
        )

    def plan_with_history(
        self,
        blocks: Iterable[PromptBlock],
        prior_prefix_hash: str | None,
        *,
        tail_budget_tokens: int | None = None,
    ) -> PrefixCachePlan:
        """Plan and populate invalidated_reason if prefix changed vs prior turn."""
        plan = self.plan(blocks, tail_budget_tokens=tail_budget_tokens)
        if prior_prefix_hash and plan.prefix_hash != prior_prefix_hash:
            reason = _detect_invalidation_reason(plan)
            return PrefixCachePlan(
                static_prefix=plan.static_prefix,
                dynamic_state=plan.dynamic_state,
                prefix_hash=plan.prefix_hash,
                prefix_tokens=plan.prefix_tokens,
                dynamic_tokens=plan.dynamic_tokens,
                total_tokens=plan.total_tokens,
                invalidated_reason=reason,
            )
        return plan


def _detect_invalidation_reason(plan: PrefixCachePlan) -> str:
    """Heuristic: identify what kind of static block likely changed."""
    kinds = {b.kind for b in plan.static_prefix}
    if BlockKind.TOOL_SCHEMA in kinds:
        return "tool_schema_changed"
    if BlockKind.SYSTEM in kinds or BlockKind.CODING_POLICY in kinds:
        return "system_prompt_changed"
    if BlockKind.REPO_SUMMARY in kinds:
        return "repo_summary_changed"
    return "static_prefix_changed"
