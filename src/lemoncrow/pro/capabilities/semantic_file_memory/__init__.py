"""Semantic file memory — AST-based code structure indexing.

Despite the "memory" in the name, this package is about **code structure**,
not session or fact memory.  It parses source files (Python, TypeScript, Rust,
SQL, …) into outlines and symbol maps, caches them on disk, and powers the
``smart_read`` MCP tool and the code-intel engine.

For session passage memory see ``archival_recall``.
For named user facts see ``memory.service.MemoryService``.
"""

from .capability import SemanticFileMemoryCapability
from .models import (
    ChangeImpact,
    FileOutline,
    ImportInfo,
    SemanticSummary,
    SymbolInfo,
    SymbolOutline,
)
from .python_ast import (
    _ast_truncated_source,
    _python_full_ast,
    analyze_python,
    stub_function_bodies,
)
from .python_ast import (
    outline as python_outline,
)
from .search import SymbolIndex
from .typescript_ast import analyze_typescript
from .typescript_ast import outline as typescript_outline

__all__ = [
    "ChangeImpact",
    "FileOutline",
    "ImportInfo",
    "SemanticFileMemoryCapability",
    "SemanticSummary",
    "SymbolIndex",
    "SymbolInfo",
    "SymbolOutline",
    "_ast_truncated_source",
    # Backward-compatible aliases (used by tests and engine.py)
    "_python_full_ast",
    "analyze_python",
    "analyze_typescript",
    "python_outline",
    "stub_function_bodies",
    "typescript_outline",
]
