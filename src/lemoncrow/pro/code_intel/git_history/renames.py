"""Rename helpers for git-history ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lemoncrow.pro.code_intel.git_history import require_pygit2


@dataclass(frozen=True)
class RenameRecord:
    old_path: str
    new_path: str


def detect_renames(diff: Any) -> dict[str, RenameRecord]:
    """Return rename records keyed by old path after mutating the diff in place."""

    pygit2 = require_pygit2()
    diff.find_similar(
        flags=pygit2.enums.DiffFind.FIND_RENAMES,
        rename_threshold=70,
    )
    renames: dict[str, RenameRecord] = {}
    for patch in diff:
        delta = patch.delta
        if delta.status == pygit2.enums.DeltaStatus.RENAMED:
            renames[delta.old_file.path] = RenameRecord(
                old_path=delta.old_file.path,
                new_path=delta.new_file.path,
            )
    return renames
