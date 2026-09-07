"""Host-neutral memory service contract.

Memory subsystem map
--------------------
LemonCrow has five "memory"-named packages, each serving a distinct purpose.
They are NOT competing systems — understand the scope before choosing one:

``memory/`` (this package)
    Canonical CRUD facade for **user-created named facts**.  Think: Copilot
    Memory / store_memory / vote_memory.  Stores facts as ``MemoryBlock``
    entries in the LemonCrow SQLite/Postgres store.  Uses ``archival_recall``
    for recall and ``memory_arbitration`` for write-time deduplication.

``archival_recall/``
    Archives session **transcript passages and code chunks** (chunked,
    embedded, ranked).  Powers the MCP ``memory`` recall tool and the
    ``ArchivalRecallCapability`` used by the engine.  Hybrid BM25 + cosine
    similarity recall over automatically-captured content.

``memory_arbitration/``
    **Write-time guard** for memory writes.  Before inserting a ``MemoryBlock``
    the arbiter asks a local LLM (Ollama) to decide ADD / UPDATE / DELETE /
    NOOP relative to similar existing blocks.  Fails open (defaults to ADD).
    Used by this service and directly by the MCP server.

``cross_vendor_memory/``
    **Read-only bridge to external AI tools' native memory files** (Claude's
    ``CLAUDE.md``, Codex memories, Gemini memories).  Has nothing to do with
    LemonCrow's own memory store — it imports facts FROM other agents.

``semantic_file_memory/``
    **AST-based code-structure indexing** — not session memory.  Produces
    outlines, symbol maps, and smart-read payloads for Python/TypeScript/etc.
    Used by the ``smart_read`` MCP tool and the code-intel engine.
"""

from lemoncrow.pro.capabilities.memory.service import (
    MemoryFactResult,
    MemoryRecallResult,
    MemoryService,
    MemoryVoteResult,
)

__all__ = [
    "MemoryFactResult",
    "MemoryRecallResult",
    "MemoryService",
    "MemoryVoteResult",
]
