"""Scoped pull-context capability orchestrator (M4)."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import replace
from typing import Any

from lemoncrow.pro.capabilities.code_context.budget import BudgetPacker
from lemoncrow.pro.capabilities.context_reuse.dead_ends import DeadEndTracker

from .models import ScopedContext, Subtask
from .pull import pull as _pull


class ScopedContextCapability:
    """Pull minimal, scoped context for a subtask over the code-context engine.

    ``engine`` is any object exposing ``search_symbols(query, *, limit, mode,
    snippet, file_glob)`` — in production the ``CodeContextEngine``.
    """

    def __init__(self, engine: Any, *, dead_ends: DeadEndTracker | None = None) -> None:
        self._engine = engine
        self._dead_ends = dead_ends or DeadEndTracker()
        self._packer = BudgetPacker()
        self._cache: dict[str, ScopedContext] = {}

    def _index_version(self) -> int:
        current_index_version = getattr(self._engine, "_current_index_version", None)
        if callable(current_index_version):
            with contextlib.suppress(Exception):
                return int(current_index_version())

        tool_status = getattr(self._engine, "tool_status", None)
        if callable(tool_status):
            with contextlib.suppress(Exception):
                payload = tool_status(auto_index=False, budget_tokens=200)
                if isinstance(payload, dict):
                    return int(payload.get("index_version") or 0)
        return 0

    def _key(self, subtask: Subtask) -> str:
        parts = [
            subtask.description,
            "|".join(subtask.affected_paths),
            "|".join(subtask.keywords),
            "|".join(subtask.excluded_paths),
            str(subtask.budget_tokens),
            str(self._index_version()),
        ]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()

    def pull(self, subtask: Subtask) -> ScopedContext:
        key = self._key(subtask)
        cached = self._cache.get(key)
        if cached is not None:
            return replace(cached, provenance="cached")
        result = _pull(subtask, engine=self._engine, dead_ends=self._dead_ends, packer=self._packer)
        self._cache[key] = result
        return result

    def mark_dead_end(self, approach: str) -> None:
        self._dead_ends.mark_dead_end(approach)
