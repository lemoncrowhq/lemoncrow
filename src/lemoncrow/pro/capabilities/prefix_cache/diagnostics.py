"""Prefix cache diagnostics — session-level cache hit tracking.

PrefixCacheDiagnostics accumulates per-turn PrefixCachePlan records and
computes cache_hit_ratio, invalidation frequency, and token split stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrefixTurnRecord:
    """One agent turn's prefix cache snapshot."""

    turn_index: int
    prefix_hash: str
    prefix_tokens: int
    dynamic_tokens: int
    total_tokens: int
    cache_hit: bool  # True when prefix_hash == prior turn's hash
    invalidated_reason: str = ""


class PrefixCacheDiagnostics:
    """Accumulates per-turn prefix records and surfaces hit-ratio metrics.

    Usage::

        diag = PrefixCacheDiagnostics()
        for turn_blocks in session_turns:
            plan = planner.plan(turn_blocks)
            diag.record(plan)
        print(diag.to_dict())
    """

    def __init__(self) -> None:
        self._turns: list[PrefixTurnRecord] = []
        self._last_hash: str | None = None

    def record_plan(self, plan: Any) -> PrefixTurnRecord:
        """Record a PrefixCachePlan (duck-typed to avoid circular imports)."""
        turn_index = len(self._turns)
        cache_hit = self._last_hash is not None and plan.prefix_hash == self._last_hash
        record = PrefixTurnRecord(
            turn_index=turn_index,
            prefix_hash=plan.prefix_hash,
            prefix_tokens=plan.prefix_tokens,
            dynamic_tokens=plan.dynamic_tokens,
            total_tokens=plan.total_tokens,
            cache_hit=cache_hit,
            invalidated_reason=getattr(plan, "invalidated_reason", "") or "",
        )
        self._turns.append(record)
        self._last_hash = plan.prefix_hash
        return record

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of turns (after the first) where the prefix was stable."""
        eligible = self._turns[1:]  # first turn has no prior → not scored
        if not eligible:
            return 0.0
        hits = sum(1 for t in eligible if t.cache_hit)
        return round(hits / len(eligible), 4)

    @property
    def cache_read_tokens_saved(self) -> int:
        """Estimated tokens saved by cache hits (sum of prefix_tokens on hit turns)."""
        return sum(t.prefix_tokens for t in self._turns[1:] if t.cache_hit)

    @property
    def invalidations(self) -> list[PrefixTurnRecord]:
        return [t for t in self._turns if not t.cache_hit and t.turn_index > 0]

    @property
    def last_prefix_hash(self) -> str | None:
        return self._last_hash

    @property
    def avg_prefix_tokens(self) -> int:
        if not self._turns:
            return 0
        return int(sum(t.prefix_tokens for t in self._turns) / len(self._turns))

    @property
    def avg_dynamic_tokens(self) -> int:
        if not self._turns:
            return 0
        return int(sum(t.dynamic_tokens for t in self._turns) / len(self._turns))

    def to_dict(self) -> dict[str, Any]:
        last = self._turns[-1] if self._turns else None
        invalidation_reasons = [t.invalidated_reason for t in self.invalidations if t.invalidated_reason]
        return {
            "turn_count": self.turn_count,
            "cache_hit_ratio": self.cache_hit_ratio,
            "cache_read_tokens_saved": self.cache_read_tokens_saved,
            "invalidation_count": len(self.invalidations),
            "prefix_invalidated_reason": invalidation_reasons[-1] if invalidation_reasons else "",
            "avg_prefix_tokens": self.avg_prefix_tokens,
            "avg_dynamic_tokens": self.avg_dynamic_tokens,
            "current_prefix_hash": last.prefix_hash if last else "",
            "current_prefix_tokens": last.prefix_tokens if last else 0,
            "current_dynamic_tokens": last.dynamic_tokens if last else 0,
        }
