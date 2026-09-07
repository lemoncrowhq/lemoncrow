"""Blame and churn aggregation for history-aware line spans."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from lemoncrow.pro.code_intel.git_history import require_pygit2
from lemoncrow.pro.code_intel.git_history.models import (
    BlameHunk,
    BlameRangeAnnotation,
    BlameRequest,
    ChurnStats,
)

# Bound the per-annotator memo; unique (request, head_sha, local_edits) keys
# would otherwise accumulate one annotation each for the annotator's lifetime.
_MAX_BLAME_CACHE = 512


class BlameAnnotator:
    """Compose pygit2 blame data into cacheable typed annotations."""

    def __init__(self, repo_path: str | Path) -> None:
        pygit2 = require_pygit2()
        self._pygit2 = pygit2
        self._repo = pygit2.Repository(str(repo_path))
        self._cache: dict[tuple[Any, ...], BlameRangeAnnotation] = {}

    def annotate(self, request: BlameRequest) -> BlameRangeAnnotation:
        local_edits = self._has_local_edits(request.file_path)
        cache_key = (
            request,
            self._current_head_sha(),
            local_edits,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        hunks = self._collect_hunks(request)
        latest_commit = self._latest_commit(hunks)
        annotation = BlameRangeAnnotation(
            file_path=request.file_path,
            line_start=request.line_start,
            line_end=request.line_end,
            index_sha=request.index_sha,
            head_sha=request.head_sha,
            freshness="fresh" if request.index_sha == request.head_sha else "stale",
            last_author=latest_commit.author.email,
            last_commit_sha=str(latest_commit.id),
            last_commit_summary=latest_commit.message.splitlines()[0] if latest_commit.message else None,
            age_days=max(0, math.floor((time.time() - latest_commit.commit_time) / 86400)),
            local_edits=local_edits,
            hunks=hunks,
            churn=self._compute_churn(request) if request.include_churn else None,
        )
        self._cache[cache_key] = annotation
        if len(self._cache) > _MAX_BLAME_CACHE:
            self._cache.pop(next(iter(self._cache)))
        return annotation

    def _collect_hunks(self, request: BlameRequest) -> tuple[BlameHunk, ...]:
        blame = self._repo.blame(
            request.file_path,
            min_line=request.line_start,
            max_line=request.line_end,
        )
        hunks: list[BlameHunk] = []
        for hunk in blame:
            start_line = max(request.line_start, hunk.final_start_line_number)
            end_line = min(request.line_end, hunk.final_start_line_number + hunk.lines_in_hunk - 1)
            if start_line > end_line:
                continue
            commit = self._repo.get(hunk.final_commit_id)
            hunks.append(
                BlameHunk(
                    start_line=start_line,
                    end_line=end_line,
                    commit_sha=str(commit.id),
                    author_email=commit.author.email,
                    commit_time=commit.commit_time,
                )
            )
        if not hunks:
            raise ValueError(f"no blame hunks resolved for {request.file_path}:{request.line_start}-{request.line_end}")
        return tuple(hunks)

    def _latest_commit(self, hunks: tuple[BlameHunk, ...]) -> Any:
        latest_hunk = max(hunks, key=lambda item: item.commit_time)
        return self._repo.revparse_single(latest_hunk.commit_sha)

    def _compute_churn(self, request: BlameRequest) -> ChurnStats:
        cutoff = int(time.time()) - (request.churn_window_days * 86400)
        commit_count = 0
        head = self._repo.revparse_single("HEAD")
        for commit in self._repo.walk(head.id, self._pygit2.enums.SortMode.TIME):
            if commit.commit_time < cutoff:
                break
            if self._commit_touches_lines(
                commit,
                file_path=request.file_path,
                line_start=request.line_start,
                line_end=request.line_end,
            ):
                commit_count += 1
        return ChurnStats(
            commit_count=commit_count,
            score=min(1.0, commit_count / 20.0),
            window_days=request.churn_window_days,
        )

    def _commit_touches_lines(self, commit: Any, *, file_path: str, line_start: int, line_end: int) -> bool:
        if not commit.parents:
            return False
        parent = commit.parents[0]
        diff = parent.tree.diff_to_tree(commit.tree)
        for patch in diff:
            delta = patch.delta
            if delta.new_file.path != file_path and delta.old_file.path != file_path:
                continue
            for hunk in patch.hunks:
                hunk_start = hunk.new_start
                hunk_end = hunk.new_start + max(hunk.new_lines, 1) - 1
                if hunk_end < line_start or hunk_start > line_end:
                    continue
                return True
        return False

    def _current_head_sha(self) -> str:
        return str(self._repo.revparse_single("HEAD").id)

    def _has_local_edits(self, file_path: str) -> bool:
        status = self._repo.status_file(file_path)
        dirty_mask = (
            self._pygit2.enums.FileStatus.INDEX_DELETED
            | self._pygit2.enums.FileStatus.INDEX_MODIFIED
            | self._pygit2.enums.FileStatus.INDEX_RENAMED
            | self._pygit2.enums.FileStatus.INDEX_TYPECHANGE
            | self._pygit2.enums.FileStatus.WT_DELETED
            | self._pygit2.enums.FileStatus.WT_MODIFIED
            | self._pygit2.enums.FileStatus.WT_RENAMED
            | self._pygit2.enums.FileStatus.WT_TYPECHANGE
        )
        return bool(status & dirty_mask)


__all__ = ["BlameAnnotator"]
