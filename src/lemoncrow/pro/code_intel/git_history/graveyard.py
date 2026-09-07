"""SQLite-backed deleted and renamed symbol storage."""

from __future__ import annotations

import sqlite3

from lemoncrow.pro.code_intel.git_history.models import GraveyardEntry


class SymbolGraveyard:
    """Persist and query deleted or renamed historical symbol entries."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self._schema_ready = False
        self._init_schema()

    def _init_schema(self) -> None:
        if self._schema_ready:
            return
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS symbol_graveyard (
                id INTEGER PRIMARY KEY,
                symbol_name TEXT NOT NULL,
                qualified_name TEXT,
                file_path TEXT NOT NULL,
                language TEXT,
                deleted_at_sha TEXT NOT NULL,
                deleted_at_ts INTEGER NOT NULL,
                last_author TEXT,
                last_commit_msg TEXT,
                rename_target TEXT,
                signature_hash TEXT,
                UNIQUE(symbol_name, file_path, deleted_at_sha)
            )
            """)
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_graveyard_query "
            "ON symbol_graveyard(symbol_name, qualified_name, file_path, deleted_at_ts)"
        )
        self._connection.commit()
        self._schema_ready = True

    def upsert(self, entry: GraveyardEntry) -> None:
        self._connection.execute(
            """
            INSERT INTO symbol_graveyard (
                symbol_name,
                qualified_name,
                file_path,
                language,
                deleted_at_sha,
                deleted_at_ts,
                last_author,
                last_commit_msg,
                rename_target,
                signature_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol_name, file_path, deleted_at_sha) DO UPDATE SET
                qualified_name = excluded.qualified_name,
                language = excluded.language,
                deleted_at_ts = excluded.deleted_at_ts,
                last_author = excluded.last_author,
                last_commit_msg = excluded.last_commit_msg,
                rename_target = excluded.rename_target,
                signature_hash = excluded.signature_hash
            """,
            (
                entry.symbol_name,
                entry.qualified_name,
                entry.file_path,
                entry.language,
                entry.deleted_at_sha,
                entry.deleted_at_ts,
                entry.last_author,
                entry.last_commit_msg,
                entry.rename_target,
                entry.signature_hash,
            ),
        )
        self._connection.commit()

    def find_deleted(
        self,
        query: str,
        since_ts: int | None,
        language: str | None,
    ) -> list[GraveyardEntry]:
        filters = [
            "(lower(symbol_name) LIKE ? OR lower(COALESCE(qualified_name, '')) LIKE ? OR lower(file_path) LIKE ?)"
        ]
        params: list[object] = [f"%{query.lower()}%"] * 3
        if since_ts is not None:
            filters.append("deleted_at_ts >= ?")
            params.append(since_ts)
        if language is not None:
            filters.append("language = ?")
            params.append(language)
        rows = self._connection.execute(
            """
            SELECT
                symbol_name,
                qualified_name,
                file_path,
                language,
                deleted_at_sha,
                deleted_at_ts,
                last_author,
                last_commit_msg,
                rename_target,
                signature_hash
            FROM symbol_graveyard
            WHERE
            """ + " AND ".join(filters) + " ORDER BY deleted_at_ts DESC, deleted_at_sha DESC, symbol_name ASC",
            params,
        ).fetchall()
        return [GraveyardEntry(*row) for row in rows]
