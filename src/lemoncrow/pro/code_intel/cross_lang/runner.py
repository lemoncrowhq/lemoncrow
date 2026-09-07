"""Literal-only resolver orchestration for Phase 5 cross-language edges."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from .edges import CrossLangCandidate, CrossLangEdge, CrossLangEdgeStore
from .resolvers.ctypes_resolver import resolve_ctypes
from .resolvers.dynamic_import import resolve_dynamic_imports
from .resolvers.subprocess_resolver import resolve_subprocess


class CrossLangRunner:
    """Phase 5 cross-language resolver contract."""

    resolver_names = ("ctypes", "dynamic_import", "subprocess")
    scope_ceiling = "literal_only_static_edges"
    scope_exclusions = ("runtime_tracing", "phase6_external_scope", "phase6_multi_repo_routing", "workspace_routing")

    def __init__(
        self,
        *,
        repo_root: Path,
        repo_id: str,
        connection_factory: Callable[[], sqlite3.Connection],
    ) -> None:
        self.repo_root = Path(repo_root)
        self.repo_id = repo_id
        self.connection_factory = connection_factory
        self.edge_store = CrossLangEdgeStore(connection_factory)

    def resolve_all(self) -> list[CrossLangEdge]:
        candidates = [
            *resolve_ctypes(self.repo_root),
            *resolve_dynamic_imports(self.repo_root),
            *resolve_subprocess(self.repo_root),
        ]
        edges = [edge for candidate in candidates if (edge := self._resolve_candidate(candidate)) is not None]
        self.edge_store.replace_repo_edges(self.repo_id, edges)
        self._sync_signature(edges)
        return edges

    def _resolve_candidate(self, candidate: CrossLangCandidate) -> CrossLangEdge | None:
        source_row = self._find_enclosing_symbol(candidate.src_file_path, candidate.src_line)
        if source_row is None:
            return None
        target_row = self._find_target_symbol(
            symbol_name=candidate.tgt_symbol_name,
            file_path=candidate.tgt_file_path,
            language=candidate.tgt_language,
        )
        target_symbol_id = str(target_row["symbol_id"]) if target_row is not None else None
        target_file_path = str(target_row["file_path"]) if target_row is not None else candidate.tgt_file_path
        if target_row is None and candidate.tgt_language == "c":
            target_file_path = self._find_c_target_file(candidate.tgt_symbol_name)
            if target_file_path is not None:
                target_symbol_id = hashlib.sha256(
                    f"{self.repo_id}:{target_file_path}:{candidate.tgt_symbol_name}".encode()
                ).hexdigest()[:24]
        return CrossLangEdge(
            repo_id=self.repo_id,
            src_symbol_id=str(source_row["symbol_id"]),
            src_symbol_name=str(source_row["symbol_name"]),
            src_qualified_name=str(source_row["qualified_name"]),
            src_language=str(source_row["language"]),
            src_file_path=str(source_row["file_path"]),
            src_line=candidate.src_line,
            tgt_symbol_name=candidate.tgt_symbol_name,
            tgt_symbol_id=target_symbol_id,
            tgt_language=candidate.tgt_language,
            tgt_file_path=target_file_path,
            edge_kind=candidate.edge_kind,
            confidence=candidate.confidence,
        )

    def _find_enclosing_symbol(self, file_path: str, line: int) -> sqlite3.Row | None:
        with self.connection_factory() as conn:
            row = conn.execute(
                """
                SELECT symbol_id, symbol_name, qualified_name, language, file_path
                FROM symbols
                WHERE repo_id = ? AND file_path = ? AND start_line <= ? AND end_line >= ?
                ORDER BY (end_line - start_line) ASC, start_line ASC
                LIMIT 1
                """,
                (self.repo_id, file_path, line, line),
            ).fetchone()
        assert row is None or isinstance(row, sqlite3.Row)
        return row

    def _find_target_symbol(self, *, symbol_name: str, file_path: str | None, language: str) -> sqlite3.Row | None:
        clauses = ["repo_id = ?", "symbol_name = ?", "language = ?"]
        params: list[object] = [self.repo_id, symbol_name, language]
        if file_path:
            clauses.append("file_path = ?")
            params.append(file_path)
        with self.connection_factory() as conn:
            row = conn.execute(
                f"""
                SELECT symbol_id, file_path
                FROM symbols
                WHERE {" AND ".join(clauses)}
                ORDER BY file_path, start_line
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
        assert row is None or isinstance(row, sqlite3.Row)
        return row

    def _find_c_target_file(self, symbol_name: str) -> str | None:
        pattern = re.compile(rf"\b{re.escape(symbol_name)}\s*\(")
        for suffix in ("*.c", "*.h", "*.cc", "*.cpp"):
            for path in sorted(self.repo_root.rglob(suffix)):
                source = path.read_text(encoding="utf-8", errors="replace")
                if pattern.search(source):
                    return path.relative_to(self.repo_root).as_posix()
        return None

    def _sync_signature(self, edges: list[CrossLangEdge]) -> None:
        payload = [edge.model_dump(mode="json") for edge in edges]
        signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        with self.connection_factory() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """)
            row = conn.execute("SELECT value FROM engine_state WHERE key = 'cross_lang_signature'").fetchone()
            previous = str(row["value"]) if row is not None else None
            conn.execute(
                """
                INSERT INTO engine_state(key, value)
                VALUES ('cross_lang_signature', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (signature,),
            )
            if previous is not None and previous != signature:
                current = conn.execute("SELECT value FROM engine_state WHERE key = 'index_version'").fetchone()
                index_version = int(current["value"]) if current is not None else 0
                conn.execute(
                    """
                    INSERT INTO engine_state(key, value)
                    VALUES ('index_version', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (str(index_version + 1),),
                )


__all__ = ["CrossLangRunner"]
