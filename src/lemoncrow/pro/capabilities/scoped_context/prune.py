"""Exclusion + dead-end filtering for scoped pull-context (M4)."""

from __future__ import annotations

import fnmatch
from typing import Any

from lemoncrow.pro.capabilities.context_reuse.dead_ends import DeadEndTracker

from .models import ExclusionRecord


def _path_matches(path: str, patterns: list[str]) -> str | None:
    """Return the first pattern that matches *path*, or None.

    A pattern matches if the path equals it, is under it (prefix), or matches
    it as a glob (``src/legacy/**``).
    """
    for pattern in patterns:
        if not pattern:
            continue
        if path == pattern or path.startswith(pattern.rstrip("/") + "/"):
            return pattern
        if fnmatch.fnmatch(path, pattern):
            return pattern
    return None


def apply_exclusions(
    candidates: list[dict[str, Any]],
    *,
    excluded_paths: list[str],
    dead_ends: DeadEndTracker,
) -> tuple[list[dict[str, Any]], list[ExclusionRecord]]:
    """Split candidates into (kept, excluded) honouring excluded paths + dead ends."""
    kept: list[dict[str, Any]] = []
    excluded: list[ExclusionRecord] = []
    for cand in candidates:
        path = str(cand.get("path", ""))
        symbol = str(cand.get("symbol", ""))
        matched = _path_matches(path, excluded_paths)
        if matched is not None:
            excluded.append(ExclusionRecord(path=path, symbol=symbol, reason=f"excluded_path:{matched}"))
            continue
        approach = f"{symbol} {path}".strip()
        if approach and dead_ends.is_dead_end(approach):
            excluded.append(ExclusionRecord(path=path, symbol=symbol, reason="dead_end"))
            continue
        kept.append(cand)
    return kept, excluded
