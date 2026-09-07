"""Aggregator registry across all vendor memory adapters."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from lemoncrow.pro.capabilities.cross_vendor_memory.base import MemoryAdapter, MemoryFact

if TYPE_CHECKING:
    pass


def _default_adapters() -> list[MemoryAdapter]:
    from lemoncrow.pro.capabilities.cross_vendor_memory.claude_adapter import ClaudeAdapter
    from lemoncrow.pro.capabilities.cross_vendor_memory.codex_adapter import CodexAdapter
    from lemoncrow.pro.capabilities.cross_vendor_memory.gemini_adapter import GeminiAdapter

    return [ClaudeAdapter(), CodexAdapter(), GeminiAdapter()]


class MemoryRegistry:
    """Aggregates facts from all available vendor memory adapters.

    Usage::

        registry = MemoryRegistry()
        for fact in registry.all_facts():
            print(fact.vendor, fact.content[:80])
    """

    def __init__(self, adapters: list[MemoryAdapter] | None = None) -> None:
        self._adapters: list[MemoryAdapter] = adapters if adapters is not None else _default_adapters()
        self._cache: list[MemoryFact] | None = None

    def _load(self) -> list[MemoryFact]:
        from lemoncrow.bench.mode import is_off as _bench_is_off

        if _bench_is_off():
            return []
        if self._cache is None:
            facts: list[MemoryFact] = []
            for adapter in self._adapters:
                if adapter.is_available():
                    facts.extend(adapter.list_facts())
            self._cache = facts
        return self._cache

    def invalidate(self) -> None:
        """Clear the in-memory cache so the next call re-reads from disk."""
        self._cache = None

    def all_facts(self) -> list[MemoryFact]:
        """Return all facts from all available adapters, sorted by source path."""
        return sorted(self._load(), key=lambda f: (str(f.source_path), f.line_number or 0))

    def by_vendor(self, vendor: str) -> list[MemoryFact]:
        """Return facts filtered to a specific *vendor* (case-insensitive)."""
        v = vendor.lower()
        return [f for f in self._load() if f.vendor.lower() == v]

    def show(self, fact_id: str) -> MemoryFact | None:
        """Return the fact with *fact_id*, or *None* if not found."""
        for fact in self._load():
            if fact.fact_id == fact_id:
                return fact
        return None

    def find(self, query: str, *, limit: int = 20) -> list[MemoryFact]:
        """Find facts using substring + fuzzy match on lowercased content.

        Results are ordered by relevance:
        1. Exact substring matches (highest priority)
        2. Fuzzy matches (``difflib.SequenceMatcher`` ratio ≥ 0.4)
        """
        q = query.lower()
        exact: list[MemoryFact] = []
        fuzzy: list[tuple[float, MemoryFact]] = []

        for fact in self._load():
            lc = fact.content.lower()
            if q in lc:
                exact.append(fact)
            else:
                ratio = difflib.SequenceMatcher(None, q, lc).ratio()
                if ratio >= 0.4:
                    fuzzy.append((ratio, fact))

        fuzzy.sort(key=lambda x: -x[0])
        combined: list[MemoryFact] = exact + [f for _, f in fuzzy]
        return combined[:limit]

    def source_paths_by_vendor(self) -> dict[str, list[str]]:
        """Return all source paths grouped by vendor (as strings)."""
        result: dict[str, list[str]] = {}
        for adapter in self._adapters:
            vendor = adapter.vendor
            paths = [str(p) for p in adapter.source_paths() if p.exists()]
            if paths:
                result[vendor] = paths
        return result


__all__ = ["MemoryRegistry"]
