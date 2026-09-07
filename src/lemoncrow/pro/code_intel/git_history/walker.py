"""History walking helpers for deleted and renamed symbol ingestion."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from lemoncrow.infra.tree_sitter.tags import detect_language, extract_tags_from_text
from lemoncrow.pro.code_intel.git_history import require_pygit2
from lemoncrow.pro.code_intel.git_history.graveyard import SymbolGraveyard
from lemoncrow.pro.code_intel.git_history.models import CommitRecord, GraveyardEntry
from lemoncrow.pro.code_intel.git_history.renames import detect_renames


def iter_commit_records(
    repo_path: str | Path,
    *,
    limit: int = 5,
    since_sha: str | None = None,
) -> Iterator[CommitRecord]:
    """Yield up to `limit` CommitRecord objects in reverse-chronological order.

    Stops when `since_sha` is encountered (resume support).

    Skip rules (LINEAGE-01):
    - Initial commits (no parents).
    - Merge commits with zero file-level diff patches.
    - Commits with >50 files touched, unless message contains "[lineage:keep]".
    - Bot commits (dependabot/renovate[bot]), unless "[lineage:keep]" present.
    """
    pygit2 = require_pygit2()
    repo = pygit2.Repository(str(repo_path))
    try:
        head = repo.revparse_single("HEAD")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return
    count = 0
    for commit in repo.walk(head.id, pygit2.enums.SortMode.TIME):
        if count >= limit:
            break
        sha = str(commit.id)
        if since_sha is not None and sha == since_sha:
            break
        if not commit.parents:
            continue  # initial commit
        is_merge = len(commit.parents) > 1
        parent = commit.parents[0]
        diff = parent.tree.diff_to_tree(commit.tree)
        patches = list(diff)
        if is_merge and len(patches) == 0:
            continue  # pure merge commit
        files_touched = [p.delta.new_file.path for p in patches if p.delta.new_file.path]
        msg = commit.message.strip()
        keep_override = "[lineage:keep]" in msg
        if not keep_override and len(files_touched) > 50:
            continue
        author_email = (commit.author.email or "").lower()
        is_bot = "dependabot" in author_email or "renovate[bot]" in author_email
        if is_bot and not keep_override:
            continue
        yield CommitRecord(
            sha=sha,
            author_date=commit.commit_time,
            message=msg[:2000],
            files_touched=files_touched,
            is_merge=is_merge,
        )
        count += 1


def _load_blob_text(repo: Any, tree: Any, file_path: str) -> str | None:
    try:
        entry = tree[file_path]
    except KeyError:
        return None
    try:
        blob = repo[entry.id]
    except KeyError:
        return None
    try:
        raw_bytes = cast(bytes, blob.read_raw())
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _iter_definition_entries(
    *,
    source_text: str,
    file_path: str,
    deleted_at_sha: str,
    deleted_at_ts: int,
    last_author: str | None,
    last_commit_msg: str | None,
    rename_target: str | None,
) -> list[GraveyardEntry]:
    language = detect_language(Path(file_path))
    tags = extract_tags_from_text(source_text, file_path, language=language)
    return [
        GraveyardEntry(
            symbol_name=tag.name,
            qualified_name=tag.name,
            file_path=file_path,
            language=language,
            deleted_at_sha=deleted_at_sha,
            deleted_at_ts=deleted_at_ts,
            last_author=last_author,
            last_commit_msg=last_commit_msg,
            rename_target=rename_target,
        )
        for tag in tags
        if tag.kind == "definition"
    ]


def count_commits(repo_path: str | Path) -> int:
    """Count total commits reachable from HEAD."""
    pygit2 = require_pygit2()
    try:
        repo = pygit2.Repository(str(repo_path))
        head = repo.revparse_single("HEAD")
        return sum(1 for _ in repo.walk(head.id, pygit2.enums.SortMode.TOPOLOGICAL))
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return 0


_DEFAULT_HISTORY_MAX_COMMITS = 10000
_DEFAULT_HISTORY_BOOTSTRAP_COMMITS = 100


def resolve_history_bootstrap_commits() -> int:
    """Commits indexed synchronously by the first ``lc code index`` (bootstrap).

    Keeps the eager index fast: only the most-recent N commits are walked inline.
    Deeper history is left to the (separate) background backfill. Tunable via
    ``LEMONCROW_HISTORY_BOOTSTRAP_COMMITS`` (default 100).
    """
    raw = os.environ.get("LEMONCROW_HISTORY_BOOTSTRAP_COMMITS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_HISTORY_BOOTSTRAP_COMMITS
        if value > 0:
            return value
    return _DEFAULT_HISTORY_BOOTSTRAP_COMMITS


def _resolve_history_max_commits() -> int:
    """Commit cap for :func:`walk_history`.

    The deleted/renamed-symbol graveyard only needs recent history, but an
    unbounded ``repo.walk`` over a deep-history repo (~130k commits for VS Code)
    diffs every commit and dominates ``lc code index``. Cap the walk to the
    most recent N commits. ``LEMONCROW_HISTORY_MAX_COMMITS=0`` restores the
    unbounded walk for callers that want the complete graveyard.
    """
    raw = os.environ.get("LEMONCROW_HISTORY_MAX_COMMITS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return _DEFAULT_HISTORY_MAX_COMMITS
    return _DEFAULT_HISTORY_MAX_COMMITS


def walk_history(
    repo_path: str | Path,
    graveyard: SymbolGraveyard,
    *,
    since_sha: str | None = None,
    limit: int | None = None,
    on_commit: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Populate the graveyard from historical delete and rename commits.
    Args:
        repo_path: Path to git repository
        graveyard: SymbolGraveyard to upsert entries into
        since_sha: If provided, only walk commits newer than this SHA (incremental mode)
        limit: Max commits to walk (most recent first). Defaults to
            ``LEMONCROW_HISTORY_MAX_COMMITS`` (2000); pass 0 for unbounded.
        on_commit: Optional callback(current, total) called after each commit visited

    Returns:
        Summary dict with 'commits_walked', 'symbols_found', 'renames_found', 'deletions_found'
    """

    pygit2 = require_pygit2()
    repo = pygit2.Repository(str(repo_path))
    head = repo.revparse_single("HEAD")

    # Collect a bounded commit window lazily so deep-history repos don't
    # materialize (and then diff) every commit. In incremental mode we stop at
    # the previously-indexed SHA; in both modes we cap at ``max_commits`` (most
    # recent first) so the walk stays O(cap) rather than O(history).
    max_commits = _resolve_history_max_commits() if limit is None else max(0, limit)
    commits: list[Any] = []
    total_in_repo = 0  # count beyond the limit without materialising every commit
    _COUNT_CAP = 10_000  # safety cap for enormous repos
    for commit in repo.walk(head.id, pygit2.enums.SortMode.TOPOLOGICAL):
        total_in_repo += 1
        if max_commits == 0 or len(commits) < max_commits:
            commits.append(commit)
        at_since = since_sha is not None and str(commit.id) == since_sha
        if at_since or total_in_repo >= _COUNT_CAP:
            break

    total = len(commits)
    symbols_found = 0
    renames_found = 0
    deletions_found = 0

    for idx, commit in enumerate(commits, 1):
        if not commit.parents:
            continue
        try:
            parent = commit.parents[0]
            diff = parent.tree.diff_to_tree(commit.tree)
            renames = detect_renames(diff)
            for patch in diff:
                delta = patch.delta
                old_path = delta.old_file.path
                if delta.status == pygit2.enums.DeltaStatus.RENAMED:
                    source_text = _load_blob_text(repo, parent.tree, old_path)
                    if source_text is None:
                        continue
                    entries = list(
                        _iter_definition_entries(
                            source_text=source_text,
                            file_path=old_path,
                            deleted_at_sha=str(commit.id),
                            deleted_at_ts=commit.commit_time,
                            last_author=commit.author.email,
                            last_commit_msg=commit.message.strip()[:200],
                            rename_target=renames[old_path].new_path,
                        )
                    )
                    for entry in entries:
                        graveyard.upsert(entry)
                    symbols_found += len(entries)
                    renames_found += len(entries)
                elif delta.status == pygit2.enums.DeltaStatus.DELETED:
                    source_text = _load_blob_text(repo, parent.tree, old_path)
                    if source_text is None:
                        continue
                    entries = list(
                        _iter_definition_entries(
                            source_text=source_text,
                            file_path=old_path,
                            deleted_at_sha=str(commit.id),
                            deleted_at_ts=commit.commit_time,
                            last_author=commit.author.email,
                            last_commit_msg=commit.message.strip()[:200],
                            rename_target=None,
                        )
                    )
                    for entry in entries:
                        graveyard.upsert(entry)
                    symbols_found += len(entries)
                    deletions_found += len(entries)
        finally:
            if on_commit is not None:
                on_commit(idx, total)

    return {
        "commits_walked": total,
        "total_commits": total_in_repo,
        "symbols_found": symbols_found,
        "renames_found": renames_found,
        "deletions_found": deletions_found,
    }
