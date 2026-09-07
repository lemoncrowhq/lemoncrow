"""Data models for scoped pull-context (M4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Subtask:
    """A focused unit of work to scope context for."""

    description: str
    affected_paths: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    excluded_paths: list[str] = field(default_factory=list)
    budget_tokens: int = 4000


@dataclass
class ContextChunk:
    """One ranked, packed unit of scoped context."""

    path: str
    symbol: str
    kind: str
    language: str = ""
    score: float = 0.0
    channel: str = ""
    signature: str = ""
    snippet: str = ""
    provenance: str = ""
    commit_sha: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextChunk:
        return cls(
            path=str(data.get("path", "")),
            symbol=str(data.get("symbol", "")),
            kind=str(data.get("kind", "")),
            language=str(data.get("language", "")),
            score=float(data.get("score", 0.0) or 0.0),
            channel=str(data.get("channel", "")),
            signature=str(data.get("signature", "")),
            snippet=str(data.get("snippet", "")),
            provenance=str(data.get("provenance", "")),
            commit_sha=str(data.get("commit_sha", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "language": self.language,
            "score": self.score,
            "channel": self.channel,
            "signature": self.signature,
            "snippet": self.snippet,
        }
        if self.provenance:
            payload["provenance"] = self.provenance
        if self.commit_sha:
            payload["commit_sha"] = self.commit_sha
        return payload


@dataclass
class ExclusionRecord:
    """A candidate dropped before packing, with the reason why."""

    path: str
    symbol: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "symbol": self.symbol, "reason": self.reason}


@dataclass
class ScopedContext:
    """The product of a scoped pull: packed chunks plus debuggable provenance."""

    chunks: list[ContextChunk]
    rationale: str
    excluded: list[ExclusionRecord]
    trace_id: str
    total_tokens: int = 0
    dropped_for_budget: int = 0
    provenance: str = "fresh"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "rationale": self.rationale,
            "excluded": [e.to_dict() for e in self.excluded],
            "trace_id": self.trace_id,
            "total_tokens": self.total_tokens,
            "dropped_for_budget": self.dropped_for_budget,
            "provenance": self.provenance,
        }
