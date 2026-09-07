"""SQLite-backed retrieval cache for code-context payloads."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import suppress
from pathlib import Path
from typing import Any

_CODE_FINGERPRINT: str | None = None


def _code_fingerprint() -> str:
    """Fingerprint of the retrieval code itself, folded into every cache key.

    ``index_version`` only tracks the *index*; a change to the ranking code (an
    upgrade, or a candidate build in a fitness sweep) must not serve payloads
    computed by older code. Hashing the code_context + code_intel + embeddings
    sources keys the cache by code version too, so stale-code hits are
    impossible. Computed once per process (~2 MB read); falls back to the
    package version when sources aren't readable (e.g. zipped install).
    """
    global _CODE_FINGERPRINT
    if _CODE_FINGERPRINT is not None:
        return _CODE_FINGERPRINT
    try:
        pkg_root = Path(__file__).resolve().parent  # .../capabilities/code_context
        infra_root = pkg_root.parents[2] / "infra"
        digest = hashlib.sha256()
        count = 0
        for root in (pkg_root, infra_root / "code_intel", infra_root / "embeddings"):
            if not root.is_dir():
                continue
            for source in sorted(root.rglob("*.py")):
                digest.update(source.read_bytes())
                count += 1
        if count == 0:
            raise OSError("no retrieval sources found")
        _CODE_FINGERPRINT = digest.hexdigest()[:16]
    except Exception:  # noqa: BLE001 - any failure falls back to package version
        try:
            from lemoncrow import __version__ as _pkg_version
        except Exception:  # noqa: BLE001
            _pkg_version = "unknown"
        _CODE_FINGERPRINT = f"v:{_pkg_version}"
    return _CODE_FINGERPRINT


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class RetrievalCache:
    """Content-addressed retrieval cache stored in the code-context SQLite DB."""

    def __init__(self, db_path: str | Path, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.db_path = Path(db_path)
        self.max_bytes = max_bytes
        self._schema_ready = False
        self._sets_since_evict_check = 0

    def get(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        index_version: int,
        repo_id: str,
        key_prefix: str = "",
    ) -> tuple[bool, dict[str, Any] | None]:
        query_hash = self.make_key(
            tool_name=tool_name, args=args, index_version=index_version, repo_id=repo_id, key_prefix=key_prefix
        )
        # Bounded busy wait: the cache lives inside code_context.sqlite, so a
        # concurrent writer (indexer, autosync) holding the write lock must
        # cost at most 500ms here, not the default 30s busy timeout (observed
        # as 30s/query stalls when a rival indexer wrote the same DB — the
        # hit-count UPDATE below is a write and queues behind the indexer).
        with self._connect(busy_timeout_ms=500) as conn:
            self._init_schema(conn)
            row = conn.execute(
                """
                SELECT payload_json
                FROM retrieval_cache
                WHERE query_hash = ? AND tool_name = ? AND index_version = ?
                """,
                (query_hash, tool_name, index_version),
            ).fetchone()
            if row is None:
                return False, None
            with suppress(sqlite3.OperationalError):
                conn.execute(
                    """
                    UPDATE retrieval_cache
                    SET hit_count = hit_count + 1,
                        last_hit_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                    WHERE query_hash = ?
                    """,
                    (query_hash,),
                )
        return True, json.loads(str(row["payload_json"]))

    def set(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        index_version: int,
        repo_id: str,
        payload: dict[str, Any],
        key_prefix: str = "",
    ) -> None:
        query_hash = self.make_key(
            tool_name=tool_name, args=args, index_version=index_version, repo_id=repo_id, key_prefix=key_prefix
        )
        payload_json = _canonical_json(payload)
        # Best-effort write on the query hot path: when another process holds
        # the write lock (indexer, autosync), skipping the cache write beats
        # stalling the query response for the full busy timeout (measured 8s
        # behind a warm-index subprocess). The payload is simply recomputed
        # and re-offered next time.
        with suppress(sqlite3.OperationalError), self._connect(busy_timeout_ms=500) as conn:
            self._init_schema(conn)
            conn.execute(
                """
                INSERT INTO retrieval_cache(query_hash, repo_id, tool_name, index_version, payload_json, hit_count, last_hit_at)
                VALUES (?, ?, ?, ?, ?, 0, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT(query_hash) DO UPDATE SET
                    repo_id = excluded.repo_id,
                    tool_name = excluded.tool_name,
                    index_version = excluded.index_version,
                    payload_json = excluded.payload_json,
                    last_hit_at = excluded.last_hit_at
                """,
                (query_hash, repo_id, tool_name, index_version, payload_json),
            )
            # Amortized eviction: the size check scans the whole table, so run
            # it on the first set() of the process (an inherited over-cap
            # cache shrinks immediately) and then every 32nd set. Worst-case
            # overshoot is 31 payloads (~KBs each) past max_bytes.
            if self._sets_since_evict_check == 0:
                self._evict_lru(conn)
            self._sets_since_evict_check = (self._sets_since_evict_check + 1) % 32

    def stats(
        self,
        *,
        repo_id: str,
        index_version: int,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            self._init_schema(conn)
            where = ["index_version = ?", "(repo_id = ? OR repo_id IS NULL)"]
            params: list[Any] = [index_version, repo_id]
            if tool_name:
                where.append("tool_name = ?")
                params.append(tool_name)
            where_sql = " AND ".join(where)
            totals = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS entry_count,
                    COALESCE(SUM(LENGTH(payload_json)), 0) AS total_bytes,
                    COALESCE(MAX(last_hit_at), '') AS last_hit_at
                FROM retrieval_cache
                WHERE {where_sql}
                """,
                params,
            ).fetchone()
            tool_rows = conn.execute(
                f"""
                SELECT tool_name, COUNT(*) AS entry_count
                FROM retrieval_cache
                WHERE {where_sql}
                GROUP BY tool_name
                ORDER BY tool_name ASC
                """,
                params,
            ).fetchall()
        return {
            "entry_count": int(totals["entry_count"]) if totals is not None else 0,
            "entries_by_tool": {str(row["tool_name"]): int(row["entry_count"]) for row in tool_rows},
            "total_bytes": int(totals["total_bytes"]) if totals is not None else 0,
            "last_hit_at": str(totals["last_hit_at"]) if totals is not None else "",
            "max_bytes": self.max_bytes,
        }

    def invalidate(
        self,
        *,
        repo_id: str,
        index_version: int,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            self._init_schema(conn)
            where = ["index_version = ?", "(repo_id = ? OR repo_id IS NULL)"]
            params: list[Any] = [index_version, repo_id]
            if tool_name:
                where.append("tool_name = ?")
                params.append(tool_name)
            where_sql = " AND ".join(where)
            rows = conn.execute(
                f"""
                SELECT tool_name, COUNT(*) AS entry_count
                FROM retrieval_cache
                WHERE {where_sql}
                GROUP BY tool_name
                ORDER BY tool_name ASC
                """,
                params,
            ).fetchall()
            conn.execute(f"DELETE FROM retrieval_cache WHERE {where_sql}", params)
        entries_by_tool = {str(row["tool_name"]): int(row["entry_count"]) for row in rows}
        return {
            "invalidated_entries": sum(entries_by_tool.values()),
            "entries_by_tool": entries_by_tool,
        }

    def make_key(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        index_version: int,
        repo_id: str,
        key_prefix: str = "",
    ) -> str:
        """Key shape: [key_prefix:]sha256(args + index version + repo + tool name + code fingerprint).

        The fingerprint invalidates cached payloads whenever the retrieval code
        changes -- index_version alone cannot see code upgrades. ``key_prefix``
        (e.g. from CodeContextEngine._retrieval_key_prefix) namespaces the hash
        by runtime retrieval config (zoekt mode, semantic on/off, embedder) so a
        query answered under one config never serves a stale payload computed
        under a different one against the same unchanged index -- kept as a
        literal, readable prefix on the stored key (not folded into the hashed
        payload) specifically so entries stay filterable/bulk-invalidatable by
        namespace via ``WHERE query_hash LIKE '<prefix>:%'``, and inspectable
        directly in the table without decoding a hash.
        """
        payload = f"{_canonical_json(args)}|{index_version}|{repo_id}|{tool_name}|{_code_fingerprint()}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{key_prefix}:{digest}" if key_prefix else digest

    def _connect(self, *, busy_timeout_ms: int = 30_000) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=busy_timeout_ms / 1000.0)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        # The cache is a recomputable optimization living in a WAL database:
        # synchronous=NORMAL drops the per-commit fsync (the dominant cost of the
        # hot-path set()); losing the most recent write on power failure only
        # means recomputing that payload once.
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                query_hash TEXT PRIMARY KEY,
                repo_id TEXT,
                tool_name TEXT NOT NULL,
                index_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                last_hit_at TEXT NOT NULL
            )
            """)
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(retrieval_cache)")}
        if "repo_id" not in columns:
            conn.execute("ALTER TABLE retrieval_cache ADD COLUMN repo_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS retrieval_cache_lru ON retrieval_cache(last_hit_at, hit_count)")
        self._schema_ready = True

    def _evict_lru(self, conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT COALESCE(SUM(LENGTH(payload_json)), 0) AS total_bytes, COUNT(*) AS entry_count FROM retrieval_cache"
        ).fetchone()
        total_bytes = int(row["total_bytes"]) if row is not None else 0
        entry_count = int(row["entry_count"]) if row is not None else 0
        if total_bytes <= self.max_bytes or entry_count == 0:
            return
        # Free down to 90% of the cap in ONE batched delete. The previous loop
        # re-ran the full-table SUM scan per evicted row (O(cache) per set(),
        # 11% of query wall time under py-spy). Estimating the batch from the
        # mean payload size can under-free when the oldest payloads are small;
        # the next check corrects that, so the cap stays soft by design.
        target_bytes = int(self.max_bytes * 0.9)
        mean_bytes = max(1, total_bytes // entry_count)
        batch = max(1, -(-(total_bytes - target_bytes) // mean_bytes))
        conn.execute(
            """
            DELETE FROM retrieval_cache WHERE query_hash IN (
                SELECT query_hash
                FROM retrieval_cache
                ORDER BY last_hit_at ASC, hit_count ASC, query_hash ASC
                LIMIT ?
            )
            """,
            (batch,),
        )


__all__ = ["RetrievalCache"]
