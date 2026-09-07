"""Base types for the cross-vendor memory adapter system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MemoryFact:
    """One atomic fact extracted from a native vendor memory file.

    ``fact_id`` is stable across reads: it's derived from a SHA-1 of the
    content, so identical text produces the same ID regardless of which file
    it came from.  Within a session the registry caches by ``fact_id``.
    """

    fact_id: str
    vendor: str
    source_path: Path
    source_kind: str
    content: str
    line_number: int | None
    captured_at: datetime
    raw_meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


class MemoryAdapter(Protocol):
    """Read-only adapter for one AI vendor's native memory files."""

    vendor: str

    def is_available(self) -> bool:
        """Return *True* if this vendor's memory files exist on this machine."""
        ...

    def list_facts(self) -> list[MemoryFact]:
        """Read and parse all facts. Pure read; no side effects."""
        ...

    def source_paths(self) -> list[Path]:
        """All file paths this adapter reads from (whether or not they exist)."""
        ...


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _fact_id(vendor: str, content: str) -> str:
    """Return a stable ``<vendor>-<sha1[:8]>`` fact ID from raw *content*."""
    import hashlib

    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{vendor}-{digest}"


__all__ = [
    "MemoryAdapter",
    "MemoryFact",
    "_fact_id",
    "_utcnow",
]
