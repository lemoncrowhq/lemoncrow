"""Read-only SQL inspection capability for MCP and CLI surfaces.

This module executes bounded SQL queries against allowlisted aliases from
`.lemoncrow/sql_aliases.toml` using env-var-backed connection strings.
"""

from __future__ import annotations

import re
import sqlite3
import time
import tomllib
from pathlib import Path
from typing import Any

_WRITE_PREFIXES = {
    "insert",
    "update",
    "delete",
    "create",
    "alter",
    "drop",
    "truncate",
    "replace",
    "grant",
    "revoke",
    "comment",
    "vacuum",
    "attach",
    "detach",
    "merge",
}


class SqlInspectCapability:
    """Execute deterministic SQL introspection with safety guards."""

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            from lemoncrow.core.foundation.paths import default_store_root

            root = default_store_root()
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Repository root as a Path."""
        return self._root

    def inspect(
        self,
        *,
        connection_alias: str,
        sql: str,
        params: list[Any] | dict[str, Any] | None = None,
        row_limit: int = 200,
    ) -> dict[str, Any]:
        if not connection_alias.strip():
            raise ValueError("connection_alias is required")
        if not sql.strip():
            raise ValueError("sql is required")
        if row_limit <= 0:
            raise ValueError("row_limit must be > 0")

        aliases = self._load_aliases()
        entry = aliases.get(connection_alias)
        if entry is None:
            raise ValueError(f"unknown connection_alias: {connection_alias}")

        allow_writes = bool(entry.get("allow_writes", False))
        self._enforce_sql_policy(sql, allow_writes=allow_writes)

        backend = self._resolve_backend(entry)
        dsn = self._resolve_dsn(entry)

        started = time.perf_counter()
        if backend == "sqlite":
            columns, rows, row_count, truncated = self._query_sqlite(
                dsn=dsn,
                sql=sql,
                params=params,
                row_limit=row_limit,
            )
        elif backend == "postgres":
            columns, rows, row_count, truncated = self._query_postgres(
                dsn=dsn,
                sql=sql,
                params=params,
                row_limit=row_limit,
            )
        else:
            raise ValueError(f"unsupported backend for alias {connection_alias}: {backend}")

        took_ms = int((time.perf_counter() - started) * 1000)
        return {
            "columns": columns,
            "rows": rows,
            "row_count": row_count,
            "truncated": truncated,
            "took_ms": took_ms,
        }

    def _load_aliases(self) -> dict[str, dict[str, Any]]:
        path = self.root / "sql_aliases.toml"
        if not path.is_file():
            raise ValueError(f"sql alias config not found: {path}")

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        aliases_raw = payload.get("aliases")
        if not isinstance(aliases_raw, dict):
            raise ValueError("sql_aliases.toml must define an [aliases] table")

        aliases: dict[str, dict[str, Any]] = {}
        for alias, raw_entry in aliases_raw.items():
            if not isinstance(raw_entry, dict):
                continue
            aliases[str(alias)] = dict(raw_entry)
        return aliases

    def _resolve_backend(self, entry: dict[str, Any]) -> str:
        raw = str(entry.get("backend", "")).strip().lower()
        if raw in {"postgres", "postgresql", "psql"}:
            return "postgres"
        if raw in {"sqlite", "sqlite3"}:
            return "sqlite"

        env_name = str(entry.get("env", "")).strip()
        dsn = str(entry.get("dsn", "")).strip()
        candidate = dsn
        if env_name:
            import os

            candidate = os.environ.get(env_name, "")
        lowered = candidate.lower()
        if lowered.startswith("postgres://") or lowered.startswith("postgresql://"):
            return "postgres"
        if lowered.startswith("sqlite://") or lowered.startswith("file:"):
            return "sqlite"
        return "sqlite"

    def _resolve_dsn(self, entry: dict[str, Any]) -> str:
        env_name = str(entry.get("env", "")).strip()
        if env_name:
            import os

            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ValueError(f"env var {env_name} is not set for SQL alias")
            return value

        dsn = str(entry.get("dsn", "")).strip()
        if dsn:
            return dsn
        raise ValueError("SQL alias must provide env (preferred) or dsn")

    def _enforce_sql_policy(self, sql: str, *, allow_writes: bool) -> None:
        normalized = self._strip_comments(sql).strip()
        if not normalized:
            raise ValueError("sql is empty after stripping comments")

        statements = [part.strip() for part in normalized.split(";") if part.strip()]
        if len(statements) != 1:
            raise ValueError("exactly one SQL statement is allowed")

        if allow_writes:
            return

        first_word = re.split(r"\s+", statements[0], maxsplit=1)[0].lower()
        if first_word in _WRITE_PREFIXES:
            raise PermissionError("read-only alias rejected write SQL statement")

    def _strip_comments(self, sql: str) -> str:
        # Remove line comments first, then block comments.
        no_line = re.sub(r"--[^\n]*", "", sql)
        return re.sub(r"/\*.*?\*/", "", no_line, flags=re.S)

    def _query_sqlite(
        self,
        *,
        dsn: str,
        sql: str,
        params: list[Any] | dict[str, Any] | None,
        row_limit: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], int, bool]:
        db_path, uri = self._sqlite_connect_target(dsn)
        conn = sqlite3.connect(db_path, uri=uri)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            cur = conn.execute(sql, self._sqlite_params(params))
            fetched = cur.fetchmany(row_limit + 1)
            truncated = len(fetched) > row_limit
            rows = [dict(row) for row in fetched[:row_limit]]

            columns: list[dict[str, str]] = []
            if cur.description:
                for col in cur.description:
                    col_name = str(col[0])
                    col_type = self._infer_column_type(col_name, rows)
                    columns.append({"name": col_name, "type": col_type})
            return columns, rows, len(rows), truncated
        finally:
            conn.close()

    def _sqlite_connect_target(self, dsn: str) -> tuple[str, bool]:
        lowered = dsn.lower()
        if lowered.startswith("sqlite:///"):
            return dsn[len("sqlite:///") :], False
        if lowered.startswith("sqlite://"):
            return dsn[len("sqlite://") :], False
        if lowered.startswith("file:"):
            return dsn, True
        return dsn, False

    def _sqlite_params(self, params: list[Any] | dict[str, Any] | None) -> Any:
        if params is None:
            return ()
        if isinstance(params, (list, tuple, dict)):
            return params
        raise ValueError("params must be a list, tuple, or object")

    def _query_postgres(
        self,
        *,
        dsn: str,
        sql: str,
        params: list[Any] | dict[str, Any] | None,
        row_limit: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], int, bool]:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except Exception as exc:  # pragma: no cover - dependency optional in local setups
            raise RuntimeError("psycopg is required for postgres sql inspect (install lemoncrow[postgres])") from exc

        with psycopg.connect(dsn) as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SET LOCAL statement_timeout = '5s'")
            cur.execute(sql, self._postgres_params(params))
            fetched = cur.fetchmany(row_limit + 1)
            truncated = len(fetched) > row_limit
            rows = [dict(row) for row in fetched[:row_limit]]

            columns: list[dict[str, str]] = []
            if cur.description:
                for col in cur.description:
                    col_name = str(col.name)
                    col_type = self._infer_column_type(col_name, rows)
                    columns.append({"name": col_name, "type": col_type})
            return columns, rows, len(rows), truncated

    def _postgres_params(self, params: list[Any] | dict[str, Any] | None) -> Any:
        if params is None:
            return ()
        if isinstance(params, dict):
            return params
        if isinstance(params, (list, tuple)):
            return list(params)
        raise ValueError("params must be a list, tuple, or object")

    def _infer_column_type(self, name: str, rows: list[dict[str, Any]]) -> str:
        for row in rows:
            value = row.get(name)
            if value is None:
                continue
            if isinstance(value, bool):
                return "BOOLEAN"
            if isinstance(value, int):
                return "INTEGER"
            if isinstance(value, float):
                return "REAL"
            return "TEXT"
        return "TEXT"


__all__ = ["SqlInspectCapability"]
