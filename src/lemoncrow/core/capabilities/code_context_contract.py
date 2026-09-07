"""Public data + interface contract for the code-context engine.

Layering boundary: these are the SHARED types the public trust surface and the
``lemoncrow.pro`` engine exchange — data *schemas* and the provider Protocol,
not algorithms. Keeping them here lets the rest of the runtime depend on the
contract without depending on the engine implementation.

The Pro modules ``lemoncrow.pro.capabilities.code_context.{models, call_graph,
intel_store}`` re-export these names, so existing imports keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Symbol / context data models
# --------------------------------------------------------------------------- #


class CrossLangReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_id: str | None = None
    symbol_name: str
    qualified_name: str | None = None
    language: str
    file_path: str | None = None
    line: int | None = None
    direction: Literal["incoming", "outgoing"]
    provenance: str = "cross_lang"
    edge_kind: str
    confidence: float


class SymbolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol_id: str
    repo_id: str
    file_path: str
    language: str
    symbol_name: str
    qualified_name: str
    kind: str
    signature: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    parent_symbol: str | None = None
    doc_summary: str | None = None
    documentation: list[str] | None = None  # raw symbol documentation strings
    snippet: str | None = None
    content_hash: str
    score: float | None = None
    provenance: str = "local"
    origin: Literal["internal", "external"] = "internal"
    repo_name: str | None = None
    cross_lang_refs: list[CrossLangReference] | None = None
    commit_sha: str | None = None


class IndexStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str
    repo_root: str
    db_path: str
    files_indexed: int
    symbols_indexed: int
    imports_indexed: int
    index_version: int = 0
    # True when the Free-tier repo-size cap truncated indexing (see
    # code_context/engine.py's _FREE_TIER_MAX_FILES). Always False on Pro.
    capped: bool = False


class IndexedFileRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    language: str
    symbol_count: int = 0
    top_symbols: list[str] | None = None


class RouteRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    framework: str
    method: str
    route: str
    file_path: str
    line: int
    language: str
    handler: str | None = None
    router: str | None = None
    provenance: str = "local"


class TextMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int
    column: int
    text: str


class UsageReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    line: int
    column: int
    end_line: int
    end_column: int
    snippet: str | None = None
    caller: str | None = None
    provenance: str = "local"
    edge_kind: str | None = None
    confidence: float | None = None
    repo_name: str | None = None


class ContextPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    budget_tokens: int
    token_count: int
    tokens_saved_vs_full_files: int
    symbols: list[SymbolRecord]
    entry_points: list[dict[str, Any]] = Field(default_factory=list)
    related_symbols: list[dict[str, Any]] = Field(default_factory=list)
    code_blocks: list[dict[str, Any]] = Field(default_factory=list)
    repo_map: str
    import_neighbors: list[str]
    content: str
    telemetry: dict[str, Any]
    cache_hit: bool = False
    tokens_saved: int = 0
    provenance: str = "local"


# --------------------------------------------------------------------------- #
# Call-graph types
# --------------------------------------------------------------------------- #

CallGraphDirection = Literal["callers", "callees"]
CallGraphDataStatus = Literal["available", "empty", "unavailable"]


class CallGraphNode(BaseModel):
    """Compact symbol metadata for a related caller or callee."""

    model_config = ConfigDict(extra="forbid")

    symbol_id: str
    symbol_name: str
    qualified_name: str
    file_path: str
    kind: str
    start_line: int
    end_line: int
    provenance: str = "tree_sitter"


class CallGraphEdge(BaseModel):
    """A directed edge between caller and callee symbols."""

    model_config = ConfigDict(extra="forbid")

    caller_symbol_id: str
    callee_symbol_id: str
    depth: int


class CallGraphTraversalResult(BaseModel):
    """Traversal output plus cheap snapshot metadata."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[CallGraphNode]
    edges: list[CallGraphEdge]
    truncated: bool = False
    data_status: CallGraphDataStatus = "available"
    message: str | None = None
    snapshot: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderHealth:
    """Health status for a routed symbol-intelligence provider."""

    status: Literal["ok", "degraded", "unhealthy"] = "ok"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@runtime_checkable
class SymbolIntelProvider(Protocol):
    """Backend interface for routed symbol lookups."""

    name: str

    def refresh(self) -> bool: ...

    def health(self) -> ProviderHealth | object | None: ...

    def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        kind: str | None = None,
        language: str | None = None,
        scope: Literal["repo", "external"] = "repo",
    ) -> list[SymbolRecord]: ...

    def get_symbol(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> dict[str, Any] | None: ...

    def find_references(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[UsageReference] | None: ...

    def find_callers(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None: ...

    def find_callees(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None: ...


# --------------------------------------------------------------------------- #
# Engine exceptions
# --------------------------------------------------------------------------- #
# Held open (uncompiled): mypyc cannot compile a class that subclasses a builtin
# (RuntimeError/ValueError). Keeping these here lets the engine algorithm modules
# compile to .so; importing/raising/catching them is fine — only *subclassing* a
# builtin inside a compiled module is not.


class IndexLockTimeout(RuntimeError):
    """A required index-write lock could not be acquired before the timeout.

    Raised only when a caller passes ``require_lock=True`` (e.g. the CLI
    ``lc code index`` prewarm), so a contended/failed build fails loudly
    instead of silently returning a stale snapshot.
    """

    def __init__(self, db_path: Path) -> None:
        super().__init__(
            f"index-write lock not acquired for {db_path}: another LemonCrow process "
            "is indexing. Increase LEMONCROW_INDEX_LOCK_TIMEOUT_S or retry."
        )


class UnsupportedWorkspaceOperationError(ValueError):
    """Raised when a workspace-wide op is not supported by the router."""


__all__ = [
    "CallGraphDataStatus",
    "CallGraphDirection",
    "CallGraphEdge",
    "CallGraphNode",
    "CallGraphTraversalResult",
    "ContextPack",
    "CrossLangReference",
    "IndexLockTimeout",
    "IndexStats",
    "IndexedFileRecord",
    "ProviderHealth",
    "RouteRecord",
    "SymbolIntelProvider",
    "SymbolRecord",
    "TextMatch",
    "UnsupportedWorkspaceOperationError",
    "UsageReference",
]
