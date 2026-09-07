"""Explicit Playbook lifecycle curation.

Drives block tier and removal off observed performance (success_rate + usage)
so the store self-prunes instead of accumulating stale or harmful procedures.

Levers (this repo hard-removes rather than deprecating — see project memory):
  promote  raise tier  e1 -> e2 -> e3   consistent winners with enough evidence
  demote   lower tier  e3 -> e2 -> e1   under-performers with enough evidence
  remove   hard delete                  persistent failures with enough evidence
  keep     no change                    insufficient evidence or healthy band

``curate`` is a pure function over a list of blocks; ``apply_curation`` is the
thin side-effecting wrapper that writes tier changes and hard-deletes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from lemoncrow.core.foundation.models import BlockTier, Playbook

CurationAction = Literal["promote", "demote", "remove", "keep"]

# Minimum (success + failure) attempts before any automated action.
MIN_EVIDENCE = 4
PROMOTE_SUCCESS_RATE = 0.80
PROMOTE_MIN_USAGE = 5
DEMOTE_SUCCESS_RATE = 0.50
REMOVE_SUCCESS_RATE = 0.25
REMOVE_MIN_FAILURES = 3

_TIER_ORDER: tuple[BlockTier, ...] = ("e1", "e2", "e3")


@dataclass(frozen=True)
class CurationDecision:
    block_id: str
    action: CurationAction
    reason: str
    tier_from: BlockTier
    tier_to: BlockTier
    block: Playbook


@dataclass
class CurationReport:
    decisions: list[CurationDecision] = field(default_factory=list)

    @property
    def changed(self) -> list[CurationDecision]:
        return [d for d in self.decisions if d.action != "keep"]


class _CuratorStore(Protocol):
    def upsert_block(self, block: Playbook, *, write_markdown: bool = True) -> None: ...
    def delete_block(self, block_id: str) -> bool: ...


def curate(blocks: Sequence[Playbook]) -> CurationReport:
    """Decide promote/demote/remove/keep for each block. Pure, no I/O."""
    return CurationReport(decisions=[_decide(block) for block in blocks])


def apply_curation(store: _CuratorStore, report: CurationReport) -> dict[str, int]:
    """Apply tier changes and hard-removals. Returns per-action counts."""
    counts = {"promote": 0, "demote": 0, "remove": 0}
    for decision in report.changed:
        if decision.action == "remove":
            store.delete_block(decision.block_id)
            counts["remove"] += 1
            continue
        updated = decision.block.model_copy(update={"tier": decision.tier_to, "updated_at": datetime.now(UTC)})
        store.upsert_block(updated)
        counts[decision.action] += 1
    return counts


def _decide(block: Playbook) -> CurationDecision:
    total = block.success_count + block.failure_count
    rate = block.success_rate()
    tier: BlockTier = block.tier if block.tier in _TIER_ORDER else "e2"

    if total < MIN_EVIDENCE:
        return _keep(block, tier, f"insufficient evidence ({total} < {MIN_EVIDENCE})")

    if rate >= PROMOTE_SUCCESS_RATE and block.usage_count >= PROMOTE_MIN_USAGE:
        higher = _tier_step(tier, +1)
        if higher != tier:
            return CurationDecision(
                block.id,
                "promote",
                f"success_rate {rate:.0%} over {total} attempts, {block.usage_count} uses",
                tier,
                higher,
                block,
            )
        return _keep(block, tier, "already at top tier")

    if rate <= REMOVE_SUCCESS_RATE and block.failure_count >= REMOVE_MIN_FAILURES:
        return CurationDecision(
            block.id,
            "remove",
            f"success_rate {rate:.0%} with {block.failure_count} failures",
            tier,
            tier,
            block,
        )

    if rate < DEMOTE_SUCCESS_RATE:
        lower = _tier_step(tier, -1)
        if lower != tier:
            return CurationDecision(
                block.id,
                "demote",
                f"success_rate {rate:.0%} over {total} attempts",
                tier,
                lower,
                block,
            )
        return _keep(block, tier, "lowest tier, retained")

    return _keep(block, tier, f"healthy (success_rate {rate:.0%})")


def _keep(block: Playbook, tier: BlockTier, reason: str) -> CurationDecision:
    return CurationDecision(block.id, "keep", reason, tier, tier, block)


def _tier_step(tier: BlockTier, delta: int) -> BlockTier:
    index = _TIER_ORDER.index(tier) if tier in _TIER_ORDER else 1
    index = max(0, min(len(_TIER_ORDER) - 1, index + delta))
    return _TIER_ORDER[index]


__all__ = [
    "CurationAction",
    "CurationDecision",
    "CurationReport",
    "apply_curation",
    "curate",
]
