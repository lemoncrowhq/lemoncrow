"""Routing contracts for code-intelligence symbol providers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from lemoncrow.core.capabilities.code_context_contract import ProviderHealth, SymbolIntelProvider
from lemoncrow.pro.capabilities.code_context.budget import BudgetPacker
from lemoncrow.pro.capabilities.code_context.cache import RetrievalCache
from lemoncrow.pro.capabilities.code_context.call_graph import CallGraphNode
from lemoncrow.pro.capabilities.code_context.models import SymbolRecord, UsageReference


class SymbolIntelStore:
    """Routes symbol lookups to healthy providers before local fallback."""

    def __init__(
        self,
        *,
        cache: RetrievalCache,
        packer: BudgetPacker,
        local_search: Callable[..., list[SymbolRecord]],
        local_get_symbol: Callable[..., dict[str, Any]],
        local_find_references: Callable[..., list[UsageReference]],
        local_find_callers: Callable[..., list[CallGraphNode] | None],
        local_find_callees: Callable[..., list[CallGraphNode] | None],
    ) -> None:
        self._cache = cache
        self._packer = packer
        self._local_search = local_search
        self._local_get_symbol = local_get_symbol
        self._local_find_references = local_find_references
        self._local_find_callers = local_find_callers
        self._local_find_callees = local_find_callees
        self._providers: list[SymbolIntelProvider] = []

    @property
    def providers(self) -> tuple[SymbolIntelProvider, ...]:
        return tuple(self._providers)

    def register(self, provider: SymbolIntelProvider) -> None:
        self._providers.append(provider)

    def refresh(self) -> bool:
        changed = False
        for provider in self._providers:
            try:
                changed = provider.refresh() or changed
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue
        return changed

    def search_symbols(
        self,
        query: str,
        *,
        limit: int = 20,
        kind: str | None = None,
        language: str | None = None,
        scope: Literal["repo", "external"] = "repo",
    ) -> list[SymbolRecord]:
        if scope == "external":
            return []
        return self._local_search(query, limit=limit, kind=kind, language=language)

    def get_symbol(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> dict[str, Any]:
        return self._local_get_symbol(
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            file_path=file_path,
            symbol_name=symbol_name,
        )

    def find_references(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[UsageReference]:
        return self._local_find_references(
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            file_path=file_path,
            symbol_name=symbol_name,
        )

    def find_callers(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None:
        return self._local_find_callers(
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            file_path=file_path,
            symbol_name=symbol_name,
        )

    def find_callees(
        self,
        *,
        symbol_id: str | None = None,
        qualified_name: str | None = None,
        file_path: str | None = None,
        symbol_name: str | None = None,
    ) -> list[CallGraphNode] | None:
        return self._local_find_callees(
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            file_path=file_path,
            symbol_name=symbol_name,
        )


__all__ = ["ProviderHealth", "SymbolIntelProvider", "SymbolIntelStore"]
