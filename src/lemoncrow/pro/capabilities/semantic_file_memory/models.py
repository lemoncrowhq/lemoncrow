"""Data models for semantic file memory capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lemoncrow.core.capabilities.semantic_file_memory_contract import (
    FileOutline as FileOutline,
)
from lemoncrow.core.capabilities.semantic_file_memory_contract import (
    SymbolOutline as SymbolOutline,
)


@dataclass
class SymbolInfo:
    """Rich symbol extracted from a source file."""

    name: str
    kind: str  # 'function' | 'async_function' | 'class' | 'variable' | 'method'
    lineno: int
    signature: str  # full def line without body
    is_export: bool  # defined at module level

    # Extended fields (populated by full AST pass)
    end_lineno: int = 0
    docstring: str = ""
    decorators: list[str] = field(default_factory=list)
    is_private: bool = False
    type_hint: str = ""  # return type annotation
    complexity: int = 0  # branch/loop count (cyclomatic proxy)


@dataclass
class ImportInfo:
    """Single import statement extracted from source."""

    module: str
    names: list[str]
    lineno: int
    is_from: bool
    alias: str = ""  # 'as' alias if any


@dataclass
class ChangeImpact:
    """Impact analysis for a modified file."""

    modified_file: str
    direct_importers: list[str] = field(default_factory=list)
    transitive_importers: list[str] = field(default_factory=list)
    affected_tests: list[str] = field(default_factory=list)
    risk_level: str = "low"  # 'low' | 'medium' | 'high' | 'critical'
    exported_symbols_changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modified_file": self.modified_file,
            "direct_importers": self.direct_importers,
            "transitive_importers": self.transitive_importers,
            "affected_tests": self.affected_tests,
            "risk_level": self.risk_level,
            "exported_symbols_changed": self.exported_symbols_changed,
        }


@dataclass
class SemanticSummary:
    """Full cached semantic analysis for one file."""

    path: str
    language: str
    summary: str  # truncated source (AST-stubbed for Python)
    symbols: list[str]  # simple names list
    exports: list[str]
    lines_total: int
    ast_summary: str
    content_hash: str = ""  # SHA256 of file content (stable across moves)
    symbol_details: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    dependency_map: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    module_docstring: str = ""
    complexity_score: int = 0  # sum of all symbol complexities
    git_last_commit: str = ""
    git_last_author_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "symbols": self.symbols,
            "exports": self.exports,
            "lines_total": self.lines_total,
            "ast_summary": self.ast_summary,
            "content_hash": self.content_hash,
            "imports": self.imports,
            "dependency_map": self.dependency_map,
            "test_files": self.test_files,
            "module_docstring": self.module_docstring,
            "complexity_score": self.complexity_score,
            "git_last_commit": self.git_last_commit,
            "git_last_author_date": self.git_last_author_date,
        }
