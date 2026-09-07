"""Deleted-history search adapter for the engine orchestration seam."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from lemoncrow.pro.code_intel.git_history import require_pygit2
from lemoncrow.pro.code_intel.git_history.graveyard import SymbolGraveyard
from lemoncrow.pro.code_intel.git_history.walker import resolve_history_bootstrap_commits, walk_history

logger = logging.getLogger(__name__)


def history_indexing_enabled() -> bool:
    """Whether deleted/renamed-symbol history indexing runs (default OFF; opt-in).

    Opt in with ``LEMONCROW_HISTORY_ENABLED=1``; then ``scope="deleted"`` works and
    the bootstrap graveyard walk runs. FTS/semantic search and the
    ``since``/``touched_by`` filters (a separate ``changed_files`` walk) are
    unaffected either way.

    TODO(perf, needs-improvement): this feature is DISABLED BY DEFAULT pending a
    rework. It is the dominant code-intel cold-start cost, NOT a correctness need
    for a normal explore/search. Findings from latency profiling (2026-06):

      * On a cold MCP process the graveyard warmup walks git history with a
        per-commit ``diff_to_tree`` -- ~7s of PURE-PYTHON, GIL-HELD work on a
        ~1k-commit repo, and far worse on large histories (django/sympy have
        tens of thousands of commits). Because it holds the GIL it STALLS the
        very next explore/search on that process (~9s observed; with this flag
        off explore drops from ~9.2s warm to <40ms -- ~250x). The worst single
        offender (an unbounded ``changed_files(since_ts=None)`` prewalk) is
        already removed in ``_background_warmup_worker``; the bounded bootstrap
        build still adds hundreds of ms.
      * ``changed_files`` caches results only IN MEMORY (``_changed_files_cache``),
        so EVERY fresh MCP process re-walks from scratch -- the on-disk graveyard
        does not help the recency/``since`` path. A benchmark that spawns one MCP
        server per task therefore re-pays this every single task.

    Before re-enabling by default, the next agent should: (1) make the warmup
    fully non-blocking w.r.t. the query path -- bound the walk or move the
    diff_to_tree work off the GIL (subprocess/process pool) so it can never
    starve a concurrent tool call; and (2) persist ``changed_files`` /the
    graveyard head-state cross-process so a new MCP server reuses prior work
    instead of re-walking. Then look elsewhere for remaining per-call cost.
    """
    return str(os.environ.get("LEMONCROW_HISTORY_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}


class DeletedHistorySearchAdapter:
    """Serve graveyard-backed deleted-history search results."""

    name = "graveyard"

    def __init__(self, *, repo_root: Path, repo_id: str, connection_factory: Any) -> None:
        self._repo_root = Path(repo_root)
        self._repo_id = repo_id
        self._connection_factory = connection_factory
        self._head_state_key = f"graveyard_head:{repo_id}"
        self._rename_target_cache: dict[tuple[str, str], str | None] = {}
        self._changed_files_cache: dict[tuple[int | None, str | None, str | None], frozenset[str]] = {}
        self._history_ready = False

    def search(
        self,
        query: str,
        *,
        since_ts: int | None,
        touched_by: str | None,
        language: str | None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._ensure_history_ready()
        lowered_query = query.strip().lower()
        if not lowered_query:
            return []
        with closing(self._connection_factory()) as conn:
            SymbolGraveyard(conn)
            direct_rows = self._search_rows(
                conn,
                query=lowered_query,
                since_ts=since_ts,
                touched_by=touched_by,
                language=language,
                limit=limit,
            )
            items = [
                self._row_to_item(
                    row,
                    matched_on=(
                        "rename_target"
                        if (
                            resolved_target := self._resolved_rename_target(
                                deleted_at_sha=str(row["deleted_at_sha"]),
                                old_path=str(row["file_path"]),
                                stored_target=cast(str | None, row["rename_target"]),
                            )
                        )
                        and lowered_query in resolved_target.lower()
                        else "graveyard"
                    ),
                )
                for row in direct_rows
            ]
            if len(items) >= limit:
                return items[:limit]
            seen = {str(item["symbol_id"]) for item in items}
            for row in self._rename_alias_rows(conn, since_ts=since_ts, touched_by=touched_by, language=language):
                if not self._rename_alias_matches(conn, row=row, query=lowered_query):
                    continue
                item = self._row_to_item(row, matched_on="rename_alias")
                if str(item["symbol_id"]) in seen:
                    continue
                seen.add(str(item["symbol_id"]))
                items.append(item)
                if len(items) >= limit:
                    break
            return items[:limit]

    def start_background_warmup(self) -> None:
        """Launch a daemon thread to warm the graveyard index in the background.

        Safe to call multiple times — only one thread is ever launched.
        """
        if not history_indexing_enabled():
            return
        if self._history_ready:
            return
        t = threading.Thread(
            target=self._background_warmup_worker,
            daemon=True,
            name=f"lemoncrow-graveyard-{self._repo_id[:8]}",
        )
        t.start()

    def _background_warmup_worker(self) -> None:
        try:
            self._ensure_history_ready()
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return
        # NB: do NOT pre-walk changed_files(since_ts=None) here. It walks the
        # ENTIRE git history (diff_to_tree per commit -- ~7s of pure-Python,
        # GIL-held, on a 1k-commit repo and far worse on large ones), which
        # starves the very next explore/search call on this process. The result
        # it caches -- every file ever changed -- is ~the whole repo and of no
        # practical use. changed_files is computed and cached lazily anyway, only
        # when a query actually passes a since/touched_by filter.

    def _ensure_history_ready(self, *, on_commit: Callable[[int, int], None] | None = None) -> dict[str, int]:
        import logging

        if not history_indexing_enabled():
            # Disabled via LEMONCROW_HISTORY_ENABLED=0: skip the historical diff +
            # tree-sitter parse walk. Ensure the (empty) graveyard table exists so
            # a scope="deleted" search degrades to no results instead of erroring.
            # FTS indexing and since/touched_by filters are unaffected.
            if not self._history_ready:
                with closing(self._connection_factory()) as conn:
                    SymbolGraveyard(conn)
                self._history_ready = True
            return {"commits_walked": 0, "symbols_found": 0, "renames_found": 0, "deletions_found": 0}

        current_head = self._current_head()
        if current_head is None:
            return {"commits_walked": 0, "symbols_found": 0, "renames_found": 0, "deletions_found": 0}

        with closing(self._connection_factory()) as conn:
            SymbolGraveyard(conn)
            row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (self._head_state_key,)).fetchone()
            previous_head = str(row["value"]) if row is not None else None
            count_row = conn.execute("SELECT COUNT(*) AS n FROM symbol_graveyard").fetchone()
            graveyard_count = int(count_row["n"]) if count_row is not None else 0

            # Check why we might need to re-walk
            if previous_head == current_head and graveyard_count > 0:
                logging.debug(f"Git history already indexed for HEAD {current_head[:8]}... ({graveyard_count} entries)")
                self._history_ready = True
                return {"commits_walked": 0, "symbols_found": 0, "renames_found": 0, "deletions_found": 0}

            # Log why we're walking
            since_sha = previous_head if previous_head != current_head else None
            if previous_head is None:
                logging.info("Git history not yet indexed, starting walk...")
            elif previous_head != current_head:
                logging.info(f"Git HEAD changed ({previous_head[:8]}... → {current_head[:8]}...), walking new commits")
            elif graveyard_count == 0:
                logging.info("Git history graveyard is empty, re-walking...")

            # Bootstrap: index only the most-recent N commits synchronously so the
            # eager index stays fast. Deeper history is left to the background
            # backfill (LEMONCROW_HISTORY_BOOTSTRAP_COMMITS / _MAX_COMMITS).
            summary = walk_history(
                self._repo_root,
                SymbolGraveyard(conn),
                since_sha=since_sha,
                limit=resolve_history_bootstrap_commits(),
                on_commit=on_commit,
            )

            conn.execute(
                """
                INSERT INTO engine_state(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (self._head_state_key, current_head),
            )
            conn.commit()
        self._history_ready = True
        return summary

    def _current_head(self) -> str | None:
        pygit2 = require_pygit2()
        try:
            repo = pygit2.Repository(str(self._repo_root))
            return str(repo.revparse_single("HEAD").id)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return None

    def changed_files(self, *, since_ts: int | None, touched_by: str | None) -> set[str]:
        current_head = self._current_head()
        cache_key = (since_ts, touched_by, current_head)
        cached = self._changed_files_cache.get(cache_key)
        if cached is not None:
            return set(cached)
        if current_head is None:
            return set()
        pygit2 = require_pygit2()
        try:
            repo = pygit2.Repository(str(self._repo_root))
            head = repo.revparse_single("HEAD")
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return set()
        changed: set[str] = set()
        touched_by_filter = touched_by.lower() if touched_by is not None else None
        for commit in repo.walk(head.id, pygit2.enums.SortMode.TIME):
            if since_ts is not None and commit.commit_time < since_ts:
                break
            author_haystacks = (
                str(getattr(commit.author, "email", "") or "").lower(),
                str(getattr(commit.author, "name", "") or "").lower(),
            )
            if touched_by_filter is not None and all(
                touched_by_filter not in haystack for haystack in author_haystacks
            ):
                continue
            changed.update(self._commit_changed_files(repo, commit))
        frozen = frozenset(path for path in changed if path)
        self._changed_files_cache[cache_key] = frozen
        return set(frozen)

    def _commit_changed_files(self, repo: Any, commit: Any) -> set[str]:
        if not commit.parents:
            return self._tree_paths(repo, commit.tree)
        changed: set[str] = set()
        diff = commit.parents[0].tree.diff_to_tree(commit.tree)
        for patch in diff:
            delta = patch.delta
            new_path = str(delta.new_file.path or "")
            old_path = str(delta.old_file.path or "")
            if new_path:
                changed.add(new_path)
            elif old_path:
                changed.add(old_path)
        return changed

    def _tree_paths(self, repo: Any, tree: Any) -> set[str]:
        paths: set[str] = set()
        for entry in tree:
            entry_obj = repo[entry.id]
            entry_path = str(entry.name)
            if hasattr(entry_obj, "__iter__") and type(entry_obj).__name__.lower().endswith("tree"):
                for child_path in self._tree_paths(repo, entry_obj):
                    paths.add(f"{entry_path}/{child_path}")
                continue
            paths.add(entry_path)
        return paths

    def _search_rows(
        self,
        conn: sqlite3.Connection,
        *,
        query: str,
        since_ts: int | None,
        touched_by: str | None,
        language: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        filters = [
            "("
            "lower(symbol_name) LIKE ? "
            "OR lower(COALESCE(qualified_name, '')) LIKE ? "
            "OR lower(file_path) LIKE ? "
            "OR lower(COALESCE(rename_target, '')) LIKE ?"
            ")"
        ]
        params: list[Any] = [f"%{query}%"] * 4
        if since_ts is not None:
            filters.append("deleted_at_ts >= ?")
            params.append(since_ts)
        if touched_by is not None:
            filters.append("lower(COALESCE(last_author, '')) LIKE ?")
            params.append(f"%{touched_by}%")
        if language is not None:
            filters.append("language = ?")
            params.append(language)
        params.append(limit)
        rows = conn.execute(
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
            """ + " AND ".join(filters) + " ORDER BY deleted_at_ts DESC, deleted_at_sha DESC, symbol_name ASC LIMIT ?",
            params,
        ).fetchall()
        return list(rows)

    def _rename_alias_rows(
        self,
        conn: sqlite3.Connection,
        *,
        since_ts: int | None,
        touched_by: str | None,
        language: str | None,
    ) -> list[sqlite3.Row]:
        filters: list[str] = []
        params: list[Any] = []
        if since_ts is not None:
            filters.append("deleted_at_ts >= ?")
            params.append(since_ts)
        if touched_by is not None:
            filters.append("lower(COALESCE(last_author, '')) LIKE ?")
            params.append(f"%{touched_by}%")
        if language is not None:
            filters.append("language = ?")
            params.append(language)
        where_clause = " WHERE " + " AND ".join(filters) if filters else ""
        rows = conn.execute(
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
            """ + where_clause + " ORDER BY deleted_at_ts DESC, deleted_at_sha DESC, symbol_name ASC",
            params,
        ).fetchall()
        return list(rows)

    def _rename_alias_matches(self, conn: sqlite3.Connection, *, row: sqlite3.Row, query: str) -> bool:
        rename_target = self._resolved_rename_target(
            deleted_at_sha=str(row["deleted_at_sha"]),
            old_path=str(row["file_path"]),
            stored_target=cast(str | None, row["rename_target"]),
        )
        if rename_target is None:
            return False
        rows = conn.execute(
            """
            SELECT symbol_name, qualified_name
            FROM symbols
            WHERE repo_id = ? AND file_path = ?
            """,
            (self._repo_id, rename_target),
        ).fetchall()
        for row in rows:
            if query in str(row["symbol_name"]).lower() or query in str(row["qualified_name"]).lower():
                return True
        return False

    def _resolved_rename_target(self, *, deleted_at_sha: str, old_path: str, stored_target: str | None) -> str | None:
        cache_key = (deleted_at_sha, old_path)
        if cache_key in self._rename_target_cache:
            return self._rename_target_cache[cache_key]
        if stored_target is not None:
            self._rename_target_cache[cache_key] = stored_target
            return stored_target
        pygit2 = require_pygit2()
        try:
            repo = pygit2.Repository(str(self._repo_root))
            commit = repo.revparse_single(deleted_at_sha)
            if not commit.parents:
                self._rename_target_cache[cache_key] = None
                return None
            diff = commit.parents[0].tree.diff_to_tree(commit.tree)
            diff.find_similar(
                flags=pygit2.enums.DiffFind.FIND_RENAMES,
                rename_threshold=0,
            )
            added_paths: list[str] = []
            for patch in diff:
                delta = patch.delta
                if delta.old_file.path == old_path and delta.status == pygit2.enums.DeltaStatus.RENAMED:
                    target = str(delta.new_file.path)
                    self._rename_target_cache[cache_key] = target
                    return target
                if delta.status == pygit2.enums.DeltaStatus.ADDED:
                    added_paths.append(str(delta.new_file.path))
            old_suffix = Path(old_path).suffix
            matching_suffix = [path for path in added_paths if Path(path).suffix == old_suffix]
            candidates = matching_suffix or added_paths
            if len(candidates) == 1:
                self._rename_target_cache[cache_key] = candidates[0]
                return candidates[0]
        except Exception:
            logging.exception("Recovered from broad exception handler")
            logger.debug("rename-target resolution failed", exc_info=True)
        self._rename_target_cache[cache_key] = None
        return None

    def _row_to_item(self, row: sqlite3.Row, *, matched_on: str) -> dict[str, Any]:
        raw_id = ":".join(
            [
                self._repo_id,
                str(row["file_path"]),
                str(row["qualified_name"] or row["symbol_name"]),
                str(row["deleted_at_sha"]),
            ]
        )
        deleted_at = datetime.fromtimestamp(int(row["deleted_at_ts"]), tz=UTC).isoformat().replace("+00:00", "Z")
        item = {
            "symbol_id": hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24],
            "repo_id": self._repo_id,
            "file_path": str(row["file_path"]),
            "language": str(row["language"] or "text"),
            "symbol_name": str(row["symbol_name"]),
            "qualified_name": str(row["qualified_name"] or row["symbol_name"]),
            "kind": "historical",
            "signature": str(row["qualified_name"] or row["symbol_name"]),
            "start_line": 1,
            "end_line": 1,
            "provenance": "graveyard",
            "deleted_at": deleted_at,
            "deleted_at_sha": str(row["deleted_at_sha"]),
            "last_author": row["last_author"],
            "last_commit_msg": row["last_commit_msg"],
            "matched_on": matched_on,
        }
        rename_target = self._resolved_rename_target(
            deleted_at_sha=str(row["deleted_at_sha"]),
            old_path=str(row["file_path"]),
            stored_target=cast(str | None, row["rename_target"]),
        )
        if rename_target is not None:
            item["rename_target"] = rename_target
            item["rename_note"] = "File moved; query matched the current public identity of this historical symbol"
        return item
