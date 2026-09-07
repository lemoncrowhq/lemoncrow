"""Data models for context compression capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DroppedContext:
    kind: str
    summary: str
    original_chars: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "summary": self.summary, "original_chars": self.original_chars}


@dataclass
class EventScore:
    """Importance score assigned to a single ledger event."""

    event: dict[str, Any]
    score: float
    reason: str = ""
    keystone_protected: bool = False  # True if a keystone span prevented this event from being dropped


@dataclass
class CompressionResult:
    """Result of a context compression pass."""

    chars_before: int
    chars_after: int
    reduction_pct: float
    preserved_facts: list[str]
    dropped: list[DroppedContext]
    token_savings: int = 0  # chars_saved / 4 (rough token estimate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "reduction_pct": self.reduction_pct,
            "preserved_facts": self.preserved_facts,
            "dropped": [d.to_dict() for d in self.dropped],
            "token_savings": self.token_savings,
        }

    @classmethod
    def passthrough(cls) -> CompressionResult:
        """Zero-work result for bench-off mode (no events dropped)."""
        return cls(
            chars_before=0,
            chars_after=0,
            reduction_pct=0.0,
            preserved_facts=[],
            dropped=[],
            token_savings=0,
        )


@dataclass
class ProjectionDelta:
    """Compatibility telemetry model for read-side compact projections."""

    path: str
    lang: str
    original_tokens: int
    projected_tokens: int

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.projected_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "lang": self.lang,
            "original_tokens": self.original_tokens,
            "projected_tokens": self.projected_tokens,
            "saved_tokens": self.saved_tokens,
        }
