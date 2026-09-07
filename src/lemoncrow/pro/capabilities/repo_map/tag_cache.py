"""Persistent, mtime-keyed SQLite cache for per-file tree-sitter tags.

Building a repo map parses every source file with tree-sitter, which dominates
the cost of :func:`build_reference_graph`. The in-process dict cache in
``graph.py`` makes *repeated* calls within one process free, but every fresh
process re-parses the whole repo from scratch.

This module mirrors Aider's approach: cache each file's extracted tags in a
SQLite table keyed by ``(path, mtime, size)``. A lookup returns ``None`` (a
miss) whenever the on-disk ``(mtime, size)`` no longer matches the stored key,
so the cache is correctness-preserving by construction -- a changed file is
always re-parsed.

The cache is *default-on* and gated by the ``LEMONCROW_REPOMAP_TAG_CACHE``
environment variable (``0`` / ``false`` / ``off`` disable it). It is robust to a
missing, locked, or corrupt database: any failure falls back to an in-memory
dict so tag caching degrades gracefully and never raises into graph building.

The database lives inside the project's own runtime cache dir, matching the
code_context convention (``engine._default_db_path``)::

    <repo_root>/.lemoncrow/workspace/repo_map_tags.sqlite
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from lemoncrow.infra.tree_sitter.tags import Tag

_logger = logging.getLogger(__name__)

# Schema version: bump if the persisted Tag shape changes so stale rows are
# transparently ignored (mismatched table is dropped + recreated).
_SCHEMA_VERSION = 2  # v2: Tag.node_kind (definition node kind for symbol-kind mapping)
_DB_FILENAME = "repo_map_tags.sqlite"

_DISABLE_VALUES = {"0", "false", "off", "no"}


def tag_cache_enabled() -> bool:
    """Return whether the persistent tag cache is enabled.

    Default-on; the ``LEMONCROW_REPOMAP_TAG_CACHE`` kill switch disables it when
    set to ``0`` / ``false`` / ``off`` / ``no`` (case-insensitive).
    """
    raw = os.environ.get("LEMONCROW_REPOMAP_TAG_CACHE")
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLE_VALUES


def default_tag_cache_path(repo_root: str | Path) -> Path:
    """Return the per-project SQLite path for *repo_root*'s tag cache.

    Follows the code_context per-project convention: lives inside the
    project's own ``.lemoncrow/workspace/`` runtime cache dir.
    """
    from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

    return resolve_workspace_store_dir(workspace_root=Path(repo_root)) / _DB_FILENAME


def _tag_to_dict(tag: Tag) -> dict[str, Any]:
    return {
        "name": tag.name,
        "kind": tag.kind,
        "file": tag.file,
        "line": tag.line,
        "byte_range": list(tag.byte_range),
        "node_kind": tag.node_kind,
    }


def _dict_to_tag(data: dict[str, Any]) -> Tag:
    start, end = data["byte_range"]
    return Tag(
        name=str(data["name"]),
        kind=data["kind"],
        file=str(data["file"]),
        line=int(data["line"]),
        byte_range=(int(start), int(end)),
        node_kind=data.get("node_kind"),
    )


def _serialize(tags: list[Tag]) -> str:
    return json.dumps([_tag_to_dict(tag) for tag in tags], separators=(",", ":"))


def _deserialize(payload: str) -> list[Tag]:
    raw = json.loads(payload)
    return [_dict_to_tag(item) for item in raw]


# Upper bound on the in-memory mirror. The SQLite store is the durable cache;
# this dict is only a hot mirror, so a bounded FIFO keeps it from growing one
# entry per unique file path for the life of the process.
_MAX_MEMORY_ENTRIES = 8192


class TagCache:
    """SQLite-backed cache of per-file tags keyed by ``(path, mtime, size)``.

    All persistence is best-effort: a missing/locked/corrupt database degrades to
    an in-memory dict and never raises into graph building. Construct via
    :meth:`for_repo` to use the canonical per-project path.
    """

    def __init__(self, db_path: Path | None) -> None:
        # When db_path is None (disabled or no resolvable path) the cache is
        # purely in-memory for the process lifetime.
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        # In-memory fallback / mirror: key=str(path) -> (mtime_ns, size, tags).
        self._memory: dict[str, tuple[int, int, list[Tag]]] = {}
        if db_path is not None:
            self._conn = self._open(db_path)

    @classmethod
    def for_repo(cls, repo_root: str | Path) -> TagCache:
        """Build a cache for *repo_root*, honouring the kill switch.

        Returns an in-memory-only cache when disabled or when the per-project
        path cannot be resolved.
        """
        if not tag_cache_enabled():
            return cls(None)
        try:
            db_path = default_tag_cache_path(repo_root)
        except (OSError, ValueError):  # pragma: no cover - path resolution is defensive
            _logger.debug("tag cache: could not resolve db path", exc_info=True)
            return cls(None)
        return cls(db_path)

    # -- persistence helpers ------------------------------------------------

    def _open(self, db_path: Path) -> sqlite3.Connection | None:
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), timeout=1.0)
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._ensure_schema(conn)
            return conn
        except (sqlite3.Error, OSError):
            # Missing parent we cannot create, unwritable dir, locked/corrupt DB:
            # fall back to memory-only.
            _logger.debug("tag cache: falling back to in-memory store", exc_info=True)
            return None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tag_cache ("
            "  path TEXT PRIMARY KEY,"
            "  mtime_ns INTEGER NOT NULL,"
            "  size INTEGER NOT NULL,"
            "  schema_version INTEGER NOT NULL,"
            "  tags TEXT NOT NULL"
            ")"
        )
        conn.commit()

    def _drop_connection(self) -> None:
        """Disable persistence after a fatal DB error; keep the in-memory store."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
        self._conn = None

    # -- public API ---------------------------------------------------------

    def get(self, path: str | Path) -> list[Tag] | None:
        """Return cached tags for *path*, or ``None`` on miss or change.

        Returns ``None`` when the file is absent, the cache has no entry, or the
        on-disk ``(mtime, size)`` differs from the stored key.
        """
        p = Path(path)
        try:
            stat = p.stat()
        except OSError:
            return None
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
        key = str(p)

        cached = self._memory.get(key)
        if cached is not None:
            c_mtime, c_size, c_tags = cached
            if c_mtime == mtime_ns and c_size == size:
                return list(c_tags)

        if self._conn is None:
            return None
        try:
            row = self._conn.execute(
                "SELECT mtime_ns, size, schema_version, tags FROM tag_cache WHERE path = ?",
                (key,),
            ).fetchone()
        except sqlite3.Error:
            _logger.debug("tag cache: read failed; disabling persistence", exc_info=True)
            self._drop_connection()
            return None
        if row is None:
            return None
        r_mtime, r_size, r_version, r_tags = row
        if r_mtime != mtime_ns or r_size != size or r_version != _SCHEMA_VERSION:
            return None
        try:
            tags = _deserialize(r_tags)
        except (ValueError, TypeError, KeyError):
            # Corrupt/garbled payload: treat as a miss.
            return None
        # Warm the in-memory mirror so repeated lookups skip SQLite.
        self._memory.pop(key, None)
        self._memory[key] = (mtime_ns, size, list(tags))
        if len(self._memory) > _MAX_MEMORY_ENTRIES:
            self._memory.pop(next(iter(self._memory)))
        return list(tags)

    def put(self, path: str | Path, tags: list[Tag]) -> None:
        """Store *tags* for *path* keyed by its current ``(mtime, size)``.

        Best-effort: a stat or write failure leaves the persistent store
        untouched (and never raises). The in-memory mirror is always updated when
        the file is statable so warm reads work even without a usable DB.
        """
        p = Path(path)
        try:
            stat = p.stat()
        except OSError:
            return
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size
        key = str(p)
        self._memory.pop(key, None)
        self._memory[key] = (mtime_ns, size, list(tags))
        if len(self._memory) > _MAX_MEMORY_ENTRIES:
            self._memory.pop(next(iter(self._memory)))
        if self._conn is None:
            return
        try:
            payload = _serialize(tags)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        try:
            self._conn.execute(
                "INSERT INTO tag_cache (path, mtime_ns, size, schema_version, tags) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "  mtime_ns = excluded.mtime_ns,"
                "  size = excluded.size,"
                "  schema_version = excluded.schema_version,"
                "  tags = excluded.tags",
                (key, mtime_ns, size, _SCHEMA_VERSION, payload),
            )
            self._conn.commit()
        except sqlite3.Error:
            _logger.debug("tag cache: write failed; disabling persistence", exc_info=True)
            self._drop_connection()

    def close(self) -> None:
        self._drop_connection()

    def __enter__(self) -> TagCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "TagCache",
    "default_tag_cache_path",
    "tag_cache_enabled",
]
