"""Typed storage for literal-only static cross-language edges."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from lemoncrow.core.capabilities.code_intel_contract import CrossLangEdge, CrossLangEdgeKind


@dataclass(frozen=True)
class CrossLangCandidate:
    src_language: str
    src_file_path: str
    src_line: int
    tgt_symbol_name: str
    tgt_language: str
    tgt_file_path: str | None
    edge_kind: CrossLangEdgeKind
    confidence: float


def _row_to_edge(row: sqlite3.Row) -> CrossLangEdge:
    return CrossLangEdge.model_validate(dict(row))


class CrossLangEdgeStore:
    """SQLite-backed store for cross-language edges."""

    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connection_factory = connection_factory

    def upsert_edges(self, edges: Iterable[CrossLangEdge]) -> None:
        rows = list(edges)
        if not rows:
            return
        with self._connection_factory() as conn:
            self.init_schema(conn)
            conn.executemany(
                """
                INSERT INTO cross_lang_edges(
                    repo_id,
                    src_symbol_id,
                    src_symbol_name,
                    src_qualified_name,
                    src_language,
                    src_file_path,
                    src_line,
                    tgt_symbol_name,
                    tgt_symbol_id,
                    tgt_language,
                    tgt_file_path,
                    edge_kind,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, src_symbol_id, tgt_symbol_name, edge_kind) DO UPDATE SET
                    src_symbol_name = excluded.src_symbol_name,
                    src_qualified_name = excluded.src_qualified_name,
                    src_language = excluded.src_language,
                    src_file_path = excluded.src_file_path,
                    src_line = excluded.src_line,
                    tgt_symbol_id = excluded.tgt_symbol_id,
                    tgt_language = excluded.tgt_language,
                    tgt_file_path = excluded.tgt_file_path,
                    confidence = excluded.confidence
                """,
                [
                    (
                        edge.repo_id,
                        edge.src_symbol_id,
                        edge.src_symbol_name,
                        edge.src_qualified_name,
                        edge.src_language,
                        edge.src_file_path,
                        edge.src_line,
                        edge.tgt_symbol_name,
                        edge.tgt_symbol_id,
                        edge.tgt_language,
                        edge.tgt_file_path,
                        edge.edge_kind,
                        edge.confidence,
                    )
                    for edge in rows
                ],
            )

    def replace_repo_edges(self, repo_id: str, edges: Iterable[CrossLangEdge]) -> None:
        rows = list(edges)
        with self._connection_factory() as conn:
            self.init_schema(conn)
            conn.execute("DELETE FROM cross_lang_edges WHERE repo_id = ?", (repo_id,))
            if rows:
                conn.executemany(
                    """
                    INSERT INTO cross_lang_edges(
                        repo_id,
                        src_symbol_id,
                        src_symbol_name,
                        src_qualified_name,
                        src_language,
                        src_file_path,
                        src_line,
                        tgt_symbol_name,
                        tgt_symbol_id,
                        tgt_language,
                        tgt_file_path,
                        edge_kind,
                        confidence
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            edge.repo_id,
                            edge.src_symbol_id,
                            edge.src_symbol_name,
                            edge.src_qualified_name,
                            edge.src_language,
                            edge.src_file_path,
                            edge.src_line,
                            edge.tgt_symbol_name,
                            edge.tgt_symbol_id,
                            edge.tgt_language,
                            edge.tgt_file_path,
                            edge.edge_kind,
                            edge.confidence,
                        )
                        for edge in rows
                    ],
                )

    def query_by_source_symbol(self, src_symbol_id: str) -> list[CrossLangEdge]:
        with self._connection_factory() as conn:
            self.init_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    repo_id,
                    src_symbol_id,
                    src_symbol_name,
                    src_qualified_name,
                    src_language,
                    src_file_path,
                    src_line,
                    tgt_symbol_name,
                    tgt_symbol_id,
                    tgt_language,
                    tgt_file_path,
                    edge_kind,
                    confidence
                FROM cross_lang_edges
                WHERE src_symbol_id = ?
                ORDER BY src_file_path, src_line, tgt_symbol_name
                """,
                (src_symbol_id,),
            ).fetchall()
        return [_row_to_edge(row) for row in rows]

    def query_by_target_symbol(
        self, *, tgt_symbol_id: str | None = None, tgt_symbol_name: str | None = None
    ) -> list[CrossLangEdge]:
        if not tgt_symbol_id and not tgt_symbol_name:
            raise ValueError("tgt_symbol_id or tgt_symbol_name is required")
        clauses: list[str] = []
        params: list[str] = []
        if tgt_symbol_id:
            clauses.append("tgt_symbol_id = ?")
            params.append(tgt_symbol_id)
        if tgt_symbol_name:
            clauses.append("tgt_symbol_name = ?")
            params.append(tgt_symbol_name)
        with self._connection_factory() as conn:
            self.init_schema(conn)
            rows = conn.execute(
                f"""
                SELECT
                    repo_id,
                    src_symbol_id,
                    src_symbol_name,
                    src_qualified_name,
                    src_language,
                    src_file_path,
                    src_line,
                    tgt_symbol_name,
                    tgt_symbol_id,
                    tgt_language,
                    tgt_file_path,
                    edge_kind,
                    confidence
                FROM cross_lang_edges
                WHERE {" OR ".join(clauses)}
                ORDER BY src_file_path, src_line, tgt_symbol_name
                """,
                tuple(params),
            ).fetchall()
        return [_row_to_edge(row) for row in rows]

    @staticmethod
    def init_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_lang_edges (
                id INTEGER PRIMARY KEY,
                repo_id TEXT NOT NULL,
                src_symbol_id TEXT NOT NULL,
                src_symbol_name TEXT NOT NULL,
                src_qualified_name TEXT NOT NULL,
                src_language TEXT NOT NULL,
                src_file_path TEXT NOT NULL,
                src_line INTEGER NOT NULL,
                tgt_symbol_name TEXT NOT NULL,
                tgt_symbol_id TEXT,
                tgt_language TEXT NOT NULL,
                tgt_file_path TEXT,
                edge_kind TEXT NOT NULL,
                confidence REAL NOT NULL,
                UNIQUE(repo_id, src_symbol_id, tgt_symbol_name, edge_kind)
            )
            """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cross_lang_src ON cross_lang_edges(src_symbol_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cross_lang_tgt ON cross_lang_edges(tgt_symbol_id, tgt_symbol_name)"
        )


__all__ = ["CrossLangCandidate", "CrossLangEdge", "CrossLangEdgeKind", "CrossLangEdgeStore"]
