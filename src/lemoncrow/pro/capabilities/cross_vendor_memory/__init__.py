"""Cross-vendor memory adapter package.

Provides **read-only** ingestion of native AI memory files from Claude
(Anthropic), Codex (OpenAI), and Gemini (Google) into a unified
``MemoryFact`` representation.

This package reads *other agents'* memory files from disk — it does NOT
interact with LemonCrow's own memory store (``MemoryStore`` / ``MemoryBlock``).
For LemonCrow's own fact store see ``memory.service.MemoryService``.

Quick start::

    from lemoncrow.pro.capabilities.cross_vendor_memory import MemoryRegistry

    registry = MemoryRegistry()
    for fact in registry.all_facts():
        print(fact.vendor, fact.fact_id, fact.content[:80])
"""

from __future__ import annotations

from lemoncrow.pro.capabilities.cross_vendor_memory.base import (
    MemoryAdapter,
    MemoryFact,
)
from lemoncrow.pro.capabilities.cross_vendor_memory.registry import MemoryRegistry

__all__ = [
    "MemoryAdapter",
    "MemoryFact",
    "MemoryRegistry",
]
