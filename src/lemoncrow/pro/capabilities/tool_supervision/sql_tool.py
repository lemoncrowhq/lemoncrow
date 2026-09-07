"""LemonCrow-native SQL tool surface."""

from __future__ import annotations

import os
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities.plugin_runtime import postgres_try_auto_fix, sql_auto_limit

_CONNECTION_KEYS = ("DATABASE_URL", "POSTGRES_URL", "POSTGRESQL_URL", "MYSQL_URL", "SQLITE_URL")
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
    "vacuum",
    "attach",
    "detach",
}
# Verbs that are never allowed from the model-facing SQL surface, regardless of
# allow_writes or the server env flag: they touch other files (ATTACH/DETACH),
# change permissions (GRANT/REVOKE), or rewrite the whole DB file (VACUUM).
_ALWAYS_FORBIDDEN_VERBS = {"attach", "detach", "grant", "revoke", "vacuum"}


def mask_connection_string(dsn: str) -> str:
    return re.sub(r"(://[^:/@]+):([^@]+)@", r"\1:****@", dsn)


def detect_dialect(connection_string: str, dialect: str | None = None) -> str:
    if dialect:
        normalized = dialect.lower()
        if normalized in {"postgresql", "postgres", "psql"}:
            return "postgres"
        if normalized in {"mysql", "mariadb"}:
            return "mysql"
        return "sqlite"
    lowered = connection_string.lower()
    if lowered.startswith(("postgres://", "postgresql://")):
        return "postgres"
    if lowered.startswith(("mysql://", "mariadb://")):
        return "mysql"
    return "sqlite"


def _dotenv_values(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in (".env", ".env.local", ".env.development", ".env.production"):
        path = repo_root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values.setdefault(key.strip(), value.strip().strip("\"'"))
    return values


def discover_connection(repo_root: str | Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd())
    env_map = env or dict(os.environ)
    for key in _CONNECTION_KEYS:
        if env_map.get(key):
            return {
                "connection_string": env_map[key],
                "source": f"env:{key}",
                "dialect": detect_dialect(env_map[key]),
            }
    dotenv = _dotenv_values(root)
    for key in _CONNECTION_KEYS:
        if dotenv.get(key):
            return {
                "connection_string": dotenv[key],
                "source": f"dotenv:{key}",
                "dialect": detect_dialect(dotenv[key]),
            }
    return {"connection_string": None, "source": None, "dialect": None}


from lemoncrow.core.capabilities.tool_supervision_contract import SqlPathError as SqlPathError  # noqa: E402


def _filesystem_path(dsn: str) -> tuple[str, bool, str]:
    """Map a sqlite DSN to (raw_path_for_connect, uri_flag, filesystem_path).

    ``filesystem_path`` is the on-disk path to sandbox-check; it is empty for
    pure in-memory databases, which never touch disk.
    """
    if dsn.startswith("sqlite:///"):
        raw = dsn[len("sqlite:///") :]
        return raw, False, raw
    if dsn.startswith("sqlite://"):
        raw = dsn[len("sqlite://") :]
        return raw, False, raw
    if dsn.startswith("file:"):
        # file:path?mode=ro ... ; the on-disk path is everything between the
        # `file:` scheme and the first query separator. `file::memory:` and
        # any `mode=memory` URI stay in RAM and have no disk path to check.
        # sqlite3.connect(..., uri=True) percent-decodes the path per the
        # SQLite URI spec, so the sandbox check must run on the DECODED path
        # (otherwise `file:..%2F..%2Ftmp%2Fpwn.db` escapes the sandbox while
        # passing an encoded-path check). A NUL byte truncates the C string
        # SQLite opens, so reject it outright.
        body = dsn[len("file:") :]
        fs_part = urllib.parse.unquote(body.split("?", 1)[0])
        if "\x00" in fs_part:
            raise SqlPathError("sqlite file path contains a NUL byte")
        if fs_part.startswith(":memory:") or "mode=memory" in dsn:
            return dsn, True, ""
        return dsn, True, fs_part
    if dsn == ":memory:":
        return dsn, False, ""
    return dsn, False, dsn


def _sqlite_path(dsn: str, repo_root: Path) -> tuple[str, bool]:
    raw, uri, fs_path = _filesystem_path(dsn)
    if not fs_path:
        return raw, uri
    candidate = Path(fs_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = os.path.realpath(candidate)
    root_resolved = os.path.realpath(repo_root)
    if resolved != root_resolved and not resolved.startswith(root_resolved + os.sep):
        if not os.environ.get("LEMONCROW_SQL_ALLOW_EXTERNAL_DB"):
            raise SqlPathError(
                f"sqlite path resolves outside the repo sandbox ({resolved}); "
                "set LEMONCROW_SQL_ALLOW_EXTERNAL_DB=1 to allow external database files"
            )
    return raw, uri


def _strip_comments(sql: str) -> str:
    return re.sub(r"/\*.*?\*/", "", re.sub(r"--[^\n]*", "", sql), flags=re.S).strip()


def _is_multi_statement(sql: str) -> bool:
    """Quote-aware check for >1 statement; ignores `;` inside string literals."""
    body = sql.rstrip().rstrip(";").rstrip()
    in_single = in_double = False
    for ch in body:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return True
    return False


def _top_level_verb(sql: str) -> str:
    """Leading verb, skipping a leading WITH ... CTE list to the trailing verb."""
    first = re.split(r"\s+", sql, maxsplit=1)[0].lower()
    if first != "with":
        return first
    depth = 0
    in_single = in_double = False
    for match in re.finditer(r"[()'\"]|[A-Za-z_][A-Za-z_]*", sql):
        token = match.group(0)
        if in_single:
            if token == "'":
                in_single = False
            continue
        if in_double:
            if token == '"':
                in_double = False
            continue
        if token == "'":
            in_single = True
        elif token == '"':
            in_double = True
        elif token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        elif depth == 0 and token.lower() in _WRITE_PREFIXES | {"select"}:
            return token.lower()
    return "with"


def _has_data_modifying_cte(sql: str) -> bool:
    """True if a write verb opens a parenthesized sub-statement.

    A data-modifying CTE looks like ``WITH x AS (DELETE ... RETURNING ...)``.
    Normal subqueries always open with SELECT/VALUES, so a write verb as the
    first identifier after ``(`` is a reliable signal of a write that
    :func:`_top_level_verb` (which skips parenthesized bodies) would otherwise
    misclassify as a read.
    """
    in_single = in_double = False
    expect_verb = False
    for match in re.finditer(r"[()'\"]|[A-Za-z_][A-Za-z_]*", sql):
        token = match.group(0)
        if in_single:
            if token == "'":
                in_single = False
            continue
        if in_double:
            if token == '"':
                in_double = False
            continue
        if token == "'":
            in_single = True
        elif token == '"':
            in_double = True
        elif token == "(":
            expect_verb = True
        elif token == ")":
            expect_verb = False
        elif expect_verb:
            expect_verb = False
            if token.lower() in _WRITE_PREFIXES:
                return True
    return False


def _writes_enabled(allow_writes: bool) -> bool:
    """Effective write permission: the caller arg AND the server env flag.

    Writes proceed only when the caller opts in *and* the operator has set
    ``LEMONCROW_SQL_ALLOW_WRITES`` on the server. A model-settable arg alone is
    never sufficient to mutate the database.
    """
    return allow_writes and bool(os.environ.get("LEMONCROW_SQL_ALLOW_WRITES"))


def lint_sql(sql: str, *, allow_writes: bool = True) -> dict[str, Any]:
    normalized = _strip_comments(sql)
    if not normalized:
        return {"ok": False, "message": "sql is empty"}
    if _is_multi_statement(normalized):
        return {
            "ok": False,
            "message": "multiple statements are not allowed in one sql string; use queries[] for batching",
        }
    verb = _top_level_verb(normalized)
    if verb in _ALWAYS_FORBIDDEN_VERBS:
        return {"ok": False, "message": f"{verb.upper()} is not permitted from the SQL tool"}
    if not _writes_enabled(allow_writes) and (verb in _WRITE_PREFIXES or _has_data_modifying_cte(normalized)):
        return {"ok": False, "message": "write SQL rejected for read-only execution"}
    if not _writes_enabled(allow_writes):
        # The engine's query_only guard blocks most writes, but a PRAGMA
        # assignment (PRAGMA query_only=OFF / writable_schema=ON / user_version=N)
        # or REINDEX slips past both query_only and _WRITE_PREFIXES. Reject them so
        # read-only mode cannot be toggled off or the schema rewritten.
        if verb == "pragma" and "=" in normalized:
            return {"ok": False, "message": "PRAGMA assignments are rejected for read-only execution"}
        if verb == "reindex":
            return {"ok": False, "message": "REINDEX is rejected for read-only execution"}
    return {"ok": True, "message": "ok"}


def _sqlite_overview(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    table_info: dict[str, Any] = {}
    for table in tables[:20]:
        columns = [
            dict(cid=row[0], name=row[1], type=row[2], notnull=bool(row[3]), pk=bool(row[5]))
            for row in conn.execute(f"PRAGMA table_info({table!r})")
        ]
        fks = [
            dict(table=row[2], from_column=row[3], to_column=row[4])
            for row in conn.execute(f"PRAGMA foreign_key_list({table!r})")
        ]
        table_info[table] = {"columns": columns, "foreign_keys": fks}
    return {"tables": tables, "table_count": len(tables), "schema": table_info}


def _sqlite_all_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        dict(name=row[1], type=row[2], notnull=bool(row[3]), pk=bool(row[5]))
        for row in conn.execute(f"PRAGMA table_info({table!r})")
    ]


def _sqlite_table_fks(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [
        dict(from_column=row[3], table=row[2], to_column=row[4])
        for row in conn.execute(f"PRAGMA foreign_key_list({table!r})")
    ]


def _sqlite_relationships(conn: sqlite3.Connection) -> list[dict[str, str]]:
    rels: list[dict[str, str]] = []
    for table in _sqlite_all_tables(conn):
        for fk in _sqlite_table_fks(conn, table):
            rels.append({"from": f"{table}.{fk['from_column']}", "to": f"{fk['table']}.{fk['to_column']}"})
    return rels


def _sqlite_search(conn: sqlite3.Connection, terms: list[str], *, limit: int = 25) -> list[dict[str, Any]]:
    lowered = [t.lower() for t in terms if t]
    matches: list[dict[str, Any]] = []
    for table in _sqlite_all_tables(conn):
        columns = _sqlite_columns(conn, table)
        table_hit = any(t in table.lower() for t in lowered)
        col_hits = [c for c in columns if any(t in str(c["name"]).lower() for t in lowered)]
        if not table_hit and not col_hits:
            continue
        matches.append(
            {
                "table": table,
                "columns": columns if table_hit else col_hits,
                "foreign_keys": _sqlite_table_fks(conn, table),
            }
        )
        if len(matches) >= limit:
            break
    return matches


_MAX_SQL_CELL_BYTES = 4096


def _sql_spill_enabled() -> bool:
    """Mirrors the MCP dispatch layer's T7 kill switch (``LEMONCROW_TOOL_OUTPUT_SPILL``)."""
    return os.environ.get("LEMONCROW_TOOL_OUTPUT_SPILL", "1").strip().lower() in {"1", "true", "yes", "on"}


def _cell_spill_hint(full_text: str, *, kept_chars: int) -> str:
    """Canonical truncation footer for one oversized SQL cell.

    A large BLOB/TEXT cell clipped by ``_bound_cell`` used to have the rest
    silently discarded. Mirrors ``bash_exec._spill_hint`` /
    ``web_fetch._truncate_with_spill``: persists the untouched original to the
    shared T7 spill store and names the path in the canonical
    ``[lc: ...]`` footer when spill is enabled and the write succeeds;
    falls back to the spill-failed shape (no recovery path) otherwise -- always
    returns a non-empty, informative footer.
    """
    from lemoncrow.pro.capabilities.tool_supervision import tool_output_spill

    record = tool_output_spill.spill(full_text, tool_name="sql", kind="original") if _sql_spill_enabled() else None
    return " " + tool_output_spill.spill_notice(
        verb="truncated",
        original_chars=len(full_text),
        kept_chars=kept_chars,
        path=record.path if record is not None else None,
    )


def _bound_cell(value: Any) -> Any:
    """Cap one cell so a large BLOB/TEXT column can't return MBs in a single response."""
    if isinstance(value, str) and len(value) > _MAX_SQL_CELL_BYTES:
        return value[:_MAX_SQL_CELL_BYTES] + _cell_spill_hint(value, kept_chars=_MAX_SQL_CELL_BYTES)
    if isinstance(value, (bytes, bytearray)) and len(value) > _MAX_SQL_CELL_BYTES:
        hex_val = bytes(value).hex()
        hint = _cell_spill_hint(hex_val, kept_chars=_MAX_SQL_CELL_BYTES)
        return f"<{len(value)} byte blob (hex-encoded)>{hint}"
    return value


def _run_sqlite(conn: sqlite3.Connection, sql: str, max_rows: int) -> dict[str, Any]:
    max_rows = max(1, max_rows)
    cursor = conn.execute(sql)
    rows = cursor.fetchmany(max_rows + 1)
    columns = [col[0] for col in cursor.description or []]
    # Rows are positional arrays keyed by `columns` — repeating column names
    # per row wastes tokens on every multi-row result.
    return {
        "columns": columns,
        "rows": [[_bound_cell(v) for v in row] for row in rows[:max_rows]],
        "row_count": min(len(rows), max_rows),
        "truncated": len(rows) > max_rows,
    }


def sql_tool(
    *,
    action: str,
    name: str | list[str] | None = None,
    sql: str | None = None,
    queries: list[dict[str, str]] | None = None,
    connection_string: str | None = None,
    dialect: str | None = None,
    max_rows: int = 500,
    timeout_ms: int = 30_000,
    auto_limit: bool = True,
    repo_root: str | Path | None = None,
    allow_writes: bool = True,
) -> dict[str, Any]:
    """Run a structured SQL action with local-first behavior."""
    root = Path(repo_root or Path.cwd())
    discovered = discover_connection(root) if not connection_string else {}
    dsn = connection_string or discovered.get("connection_string")
    if action == "connect" and not dsn:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "No database connection configured. Pass connection_string or set DATABASE_URL in the environment or .env file.",
                }
            ],
        }
    if not dsn:
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "No database connection configured. First run sql(action='connect', connection_string='...') or set DATABASE_URL.",
                }
            ],
        }
    resolved_dialect = detect_dialect(str(dsn), dialect)
    if resolved_dialect != "sqlite":
        if action == "connect":
            return {
                "isError": False,
                "dialect": resolved_dialect,
                "connection": mask_connection_string(str(dsn)),
                "note": "Install the optional database driver to run live queries for this dialect.",
            }
        if resolved_dialect == "postgres" and sql:
            fixed = postgres_try_auto_fix(sql, "column does not exist")
            return {
                "isError": True,
                "dialect": resolved_dialect,
                "connection": mask_connection_string(str(dsn)),
                "driver_required": True,
                "auto_fix_preview": fixed,
            }
        return {
            "isError": True,
            "dialect": resolved_dialect,
            "connection": mask_connection_string(str(dsn)),
            "message": "Optional live driver not installed for this dialect.",
        }

    try:
        db_path, uri = _sqlite_path(str(dsn), root)
    except SqlPathError as exc:
        return {"isError": True, "message": str(exc)}
    started = time.perf_counter()
    conn = sqlite3.connect(db_path, uri=uri, timeout=max(1.0, timeout_ms / 1000.0))
    try:
        conn.row_factory = sqlite3.Row
        if not _writes_enabled(allow_writes):
            # Engine-level read-only enforcement (defense in depth beyond
            # lint_sql's verb list, which misses PRAGMA writable_schema /
            # ANALYZE / REINDEX): the connection itself refuses any statement
            # that writes to the database.
            conn.execute("PRAGMA query_only = ON")
        try:
            if action == "connect":
                overview = _sqlite_overview(conn)
                return {
                    "isError": False,
                    "dialect": "sqlite",
                    "connection": mask_connection_string(str(dsn)),
                    "overview": overview,
                    "source": discovered.get("source"),
                }
            if action == "tables":
                tables = _sqlite_all_tables(conn)
                return {"isError": False, "dialect": "sqlite", "tables": tables, "table_count": len(tables)}
            if action == "schema":
                return {"isError": False, "dialect": "sqlite", **_sqlite_overview(conn)}
            if action == "table":
                table_name = str(name or "")
                if not table_name:
                    return {"isError": True, "message": "action='table' requires name=<table>"}
                return {
                    "isError": False,
                    "table": table_name,
                    "columns": _sqlite_columns(conn, table_name),
                    "foreign_keys": _sqlite_table_fks(conn, table_name),
                }
            if action == "relationships":
                return {"isError": False, "dialect": "sqlite", "relationships": _sqlite_relationships(conn)}
            if action == "search":
                terms = name if isinstance(name, list) else [name] if name else []
                if not terms:
                    return {"isError": True, "message": "action='search' requires name=<keyword>"}
                return {"isError": False, "dialect": "sqlite", "matches": _sqlite_search(conn, terms)}
        except sqlite3.Error as exc:
            return {"isError": True, "message": str(exc)}
        if action == "lint":
            lint = lint_sql(sql or "", allow_writes=allow_writes)
            return {"isError": not lint["ok"], **lint}
        if action != "query":
            return {"isError": True, "message": f"unsupported action: {action}"}
        batch = queries or [{"name": "result", "sql": sql or ""}]
        outputs: list[dict[str, Any]] = []
        for item in batch:
            label = item.get("name") or "result"
            query_sql = item.get("sql") or ""
            lint = lint_sql(query_sql, allow_writes=allow_writes)
            if not lint["ok"]:
                outputs.append({"name": label, "isError": True, "message": lint["message"]})
                continue
            limited = sql_auto_limit(query_sql, max_rows=max_rows, auto_limit=auto_limit)
            try:
                if not _writes_enabled(allow_writes):
                    # Re-arm read-only on the shared connection before every batch
                    # item so an earlier item cannot leave query_only disabled.
                    conn.execute("PRAGMA query_only = ON")
                result = _run_sqlite(conn, limited["sql"], max_rows)
                if _top_level_verb(_strip_comments(query_sql)) in _WRITE_PREFIXES:
                    conn.commit()
                outputs.append({"name": label, **result, "auto_limit_changed": limited.get("changed", False)})
            except sqlite3.Error as exc:
                outputs.append({"name": label, "isError": True, "message": str(exc)})
        return {
            "isError": any(item.get("isError") for item in outputs),
            "dialect": "sqlite",
            "results": outputs,
            "took_ms": int((time.perf_counter() - started) * 1000),
        }
    finally:
        conn.close()


__all__ = [
    "detect_dialect",
    "discover_connection",
    "lint_sql",
    "mask_connection_string",
    "sql_tool",
]
