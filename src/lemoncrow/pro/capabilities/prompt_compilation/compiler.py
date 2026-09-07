"""Deterministic prompt compiler for cache-safe prefix assembly."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256

from lemoncrow.pro.capabilities.budget_optimizer import ContextBlock, PromptBudgetOptimizer

from .models import BlockKind, PromptBlock, Stability

STABILITY_ORDER: dict[Stability, int] = {
    Stability.STATIC: 0,
    Stability.SESSION: 1,
    Stability.BRANCH: 2,
    Stability.TURN: 3,
    Stability.VOLATILE: 4,
}

KIND_ORDER: dict[BlockKind, int] = {
    BlockKind.TOOL_SCHEMA: 0,
    BlockKind.SYSTEM: 1,
    BlockKind.CODING_POLICY: 2,
    BlockKind.REPO_SUMMARY: 10,
    BlockKind.PLAYBOOK: 20,
    BlockKind.FILE_SUMMARY: 21,
    BlockKind.USER_TASK: 30,
    BlockKind.GIT_DIFF: 31,
    BlockKind.TOOL_RESULT: 32,
    BlockKind.SCRATCHPAD: 40,
}

_STABLE_STABILITIES = {Stability.STATIC, Stability.SESSION, Stability.BRANCH}


from lemoncrow.core.capabilities.prompt_compilation_contract import (  # noqa: E402
    BudgetTooSmall as BudgetTooSmall,
)


@dataclass(frozen=True)
class CompiledPrompt:
    """Compiled prompt with deterministic ordering and stable-prefix metadata."""

    blocks: tuple[PromptBlock, ...]
    prefix_end_index: int
    stable_prefix_hash: str
    stable_prefix_tokens: int
    dynamic_tail_tokens: int


def _sort_key(block: PromptBlock) -> tuple[int, int, str]:
    return (STABILITY_ORDER[block.stability], KIND_ORDER[block.kind], block.id)


def _prefix_end_index(blocks: tuple[PromptBlock, ...]) -> int:
    end = -1
    for index, block in enumerate(blocks):
        if block.stability in _STABLE_STABILITIES:
            end = index
            continue
        break
    return end


def _prefix_hash(blocks: tuple[PromptBlock, ...], prefix_end_index: int) -> str:
    stable_prefix = blocks[: prefix_end_index + 1] if prefix_end_index >= 0 else ()
    payload = b"\n--BLOCK--\n".join(f"{block.kind}:{block.id}:{block.version_hash}".encode() for block in stable_prefix)
    return sha256(payload).hexdigest()


def _block_utility(block: PromptBlock) -> float:
    metadata_utility = block.metadata.get("utility")
    if isinstance(metadata_utility, int | float):
        return float(metadata_utility)

    if block.kind is BlockKind.USER_TASK:
        return 1.0
    if block.kind is BlockKind.GIT_DIFF:
        return 0.9
    if block.kind is BlockKind.TOOL_RESULT:
        is_error = bool(block.metadata.get("is_error"))
        if not is_error:
            lower = block.content.lower()
            is_error = "error" in lower or "traceback" in lower or "failed" in lower
        return 0.8 if is_error else 0.5
    if block.kind is BlockKind.SCRATCHPAD:
        return 0.3
    return 0.4


def _pack_tail(
    tail_blocks: tuple[PromptBlock, ...],
    *,
    token_budget: int,
) -> tuple[PromptBlock, ...]:
    if token_budget < 0:
        raise ValueError("tail_budget_tokens must be >= 0")

    total_tail_tokens = sum(block.token_estimate for block in tail_blocks)
    if total_tail_tokens <= token_budget:
        return tail_blocks

    user_task_blocks = [block for block in tail_blocks if block.kind is BlockKind.USER_TASK]
    if any(block.token_estimate > token_budget for block in user_task_blocks):
        raise BudgetTooSmall("tail_budget_tokens is smaller than at least one USER_TASK block token estimate")

    candidates = [
        ContextBlock(
            id=block.id,
            content=block.content,
            token_cost=block.token_estimate,
            utility=_block_utility(block),
            source="prompt_compilation",
            metadata={"kind": block.kind.value},
        )
        for block in tail_blocks
    ]
    plan = PromptBudgetOptimizer().solve(candidates, token_budget=token_budget)
    selected_ids = {block.id for block in plan.selected}
    selected_tail = tuple(block for block in tail_blocks if block.id in selected_ids)

    if user_task_blocks and not any(block.kind is BlockKind.USER_TASK for block in selected_tail):
        raise BudgetTooSmall("tail_budget_tokens dropped all USER_TASK blocks; increase budget")
    return selected_tail


def _validate_counterexample_blocks(blocks: tuple[PromptBlock, ...]) -> None:
    for block in blocks:
        if not block.is_counterexample:
            continue
        if block.kind is not BlockKind.TOOL_RESULT:
            raise ValueError(f"Counterexample block {block.id!r} must use kind=tool_result")
        if block.stability is not Stability.TURN:
            raise ValueError(f"Counterexample block {block.id!r} must use stability=turn")


def compile_prompt(
    blocks: Iterable[PromptBlock],
    *,
    tail_budget_tokens: int | None = None,
) -> CompiledPrompt:
    """Compile prompt blocks into deterministic cache-safe order."""
    input_blocks = tuple(blocks)
    _validate_counterexample_blocks(input_blocks)
    ordered_blocks = tuple(sorted(input_blocks, key=_sort_key))
    prefix_end = _prefix_end_index(ordered_blocks)

    stable_blocks = ordered_blocks[: prefix_end + 1] if prefix_end >= 0 else ()
    tail_blocks = ordered_blocks[prefix_end + 1 :]
    if tail_budget_tokens is not None:
        tail_blocks = _pack_tail(tail_blocks, token_budget=tail_budget_tokens)

    compiled_blocks = stable_blocks + tail_blocks
    stable_prefix_hash = _prefix_hash(compiled_blocks, prefix_end)
    stable_prefix_tokens = sum(block.token_estimate for block in stable_blocks)
    dynamic_tail_tokens = sum(block.token_estimate for block in tail_blocks)

    return CompiledPrompt(
        blocks=compiled_blocks,
        prefix_end_index=prefix_end,
        stable_prefix_hash=stable_prefix_hash,
        stable_prefix_tokens=stable_prefix_tokens,
        dynamic_tail_tokens=dynamic_tail_tokens,
    )
