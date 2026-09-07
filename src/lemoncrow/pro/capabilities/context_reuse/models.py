"""Data models for reasoning reuse capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReuseSavings:
    """Tracks reasoning reuse savings over a session."""

    procedures_retrieved: int = 0
    context_tokens_saved: int = 0
    reuse_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedures_retrieved": self.procedures_retrieved,
            "context_tokens_saved": self.context_tokens_saved,
            "reuse_events": self.reuse_events,
        }


@dataclass
class RankedProcedure:
    """A procedure block ranked for relevance to the current task."""

    block_id: str
    title: str
    domain: str
    score: float
    base_score: float
    recency_score: float
    success_rate: float
    reuse_count: int
    snippet: str
    is_dead_end: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "title": self.title,
            "domain": self.domain,
            "score": round(self.score, 4),
            "base_score": round(self.base_score, 4),
            "recency_score": round(self.recency_score, 4),
            "success_rate": round(self.success_rate, 4),
            "reuse_count": self.reuse_count,
            "snippet": self.snippet,
            "is_dead_end": self.is_dead_end,
        }


@dataclass
class ProcedureCluster:
    """A group of related procedure blocks."""

    cluster_id: str
    centroid_title: str
    member_ids: list[str] = field(default_factory=list)
    avg_score: float = 0.0
