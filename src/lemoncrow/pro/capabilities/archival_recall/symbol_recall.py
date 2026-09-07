"""Symbol-linked recall bundle assembly for ``memory op="recall_symbol"``."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.memory_models import ArchivalPassage, MemoryBlock
from lemoncrow.core.foundation.models import FileEditRecord, Trace
from lemoncrow.infra.storage.memory_store import MemoryStore
from lemoncrow.pro.capabilities.code_context import CodeContextEngine
from lemoncrow.pro.capabilities.code_context.models import SymbolRecord
from lemoncrow.pro.capabilities.repo_map.budget import count_tokens

_DEFAULT_INCLUDES = ("definition", "memory")
_OPTIONAL_INCLUDES = ("traces", "decisions", "tests")
_ALL_INCLUDES = {*_DEFAULT_INCLUDES, *_OPTIONAL_INCLUDES}


class SymbolRecallCapability:
    """Resolve a symbol and fuse related low-token recall evidence."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        engine: CodeContextEngine,
        memory_store: MemoryStore,
        trace_store: Any,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._engine = engine
        self._memory_store = memory_store
        self._trace_store = trace_store

    def recall_symbol(
        self,
        *,
        query: str,
        agent_id: str | None = None,
        include: list[str] | None = None,
        horizon_days: int = 180,
        budget_tokens: int = 3000,
        top_k: int = 5,
    ) -> dict[str, Any]:
        include_set = self._resolve_includes(include)
        symbol_payload = self._resolve_symbol(query)
        if symbol_payload.get("error"):
            return self._finalize(symbol_payload, full_total_tokens=0)

        symbol = dict(symbol_payload)
        since = datetime.now(UTC) - timedelta(days=max(1, horizon_days))
        definition = self._definition_payload(symbol)
        payload: dict[str, Any] = {
            "query": query,
            "included": [name for name in (*_DEFAULT_INCLUDES, *_OPTIONAL_INCLUDES) if name in include_set],
            "definition": definition,
            "memory": self._memory_payload(symbol, agent_id=agent_id, query=query, since=since, top_k=top_k),
            "cache_hit": False,
            "provenance": "local",
        }
        if "traces" in include_set:
            payload["traces"] = self._trace_payload(symbol, since=since, limit=top_k)
        if "decisions" in include_set:
            payload["decisions"] = self._decision_payload(symbol, limit=top_k)
        if "tests" in include_set:
            payload["tests"] = self._related_tests_payload(symbol, limit=top_k)

        full_total_tokens = self._compute_total_tokens({**payload, "tokens_saved": 0, "truncated_sections": []})
        trimmed_payload = self._trim_to_budget(payload, budget_tokens=budget_tokens)
        if trimmed_payload["total_tokens"] > budget_tokens:
            return self._finalize(
                {
                    "error": "budget_too_small",
                    "message": "budget_tokens cannot fit the symbol definition anchor",
                    "budget_tokens": budget_tokens,
                    "minimum_required_tokens": int(trimmed_payload["total_tokens"]),
                    "definition": trimmed_payload["definition"],
                    "included": ["definition"],
                    "cache_hit": False,
                    "provenance": "local",
                },
                full_total_tokens=full_total_tokens,
            )
        return self._finalize(trimmed_payload, full_total_tokens=full_total_tokens)

    def _resolve_includes(self, include: list[str] | None) -> set[str]:
        requested = set(_DEFAULT_INCLUDES)
        for item in include or []:
            normalized = str(item).strip().lower()
            if normalized not in _ALL_INCLUDES:
                choices = ", ".join(sorted(_ALL_INCLUDES))
                raise ValueError(f"include entries must be one of: {choices}")
            requested.add(normalized)
        return requested

    def _resolve_symbol(self, query: str) -> dict[str, Any]:
        exact_query = query.strip()
        if not exact_query:
            raise ValueError("query is required for recall_symbol")
        hits = self._engine.search_symbols(exact_query, limit=20, mode="lexical", snippet="none")
        exact_matches = [
            hit
            for hit in hits
            if hit.symbol_id == exact_query or hit.qualified_name == exact_query or hit.symbol_name == exact_query
        ]
        if len(exact_matches) == 1:
            return self._engine.get_symbol(symbol_id=exact_matches[0].symbol_id)
        if len(exact_matches) > 1:
            return self._ambiguity_payload(exact_query, exact_matches)
        if len(hits) == 1:
            return self._engine.get_symbol(symbol_id=hits[0].symbol_id)
        if not hits:
            return {
                "error": "symbol_not_found",
                "message": "no matching symbol was found",
                "query": exact_query,
                "cache_hit": False,
                "provenance": "local",
            }
        return self._ambiguity_payload(exact_query, hits)

    def _ambiguity_payload(self, query: str, hits: list[SymbolRecord]) -> dict[str, Any]:
        return {
            "error": "disambiguation_required",
            "message": "multiple symbols matched the recall query",
            "query": query,
            "matches": [
                {
                    "symbol_id": hit.symbol_id,
                    "qualified_name": hit.qualified_name,
                    "symbol_name": hit.symbol_name,
                    "file_path": hit.file_path,
                    "start_line": hit.start_line,
                }
                for hit in hits[:10]
            ],
            "cache_hit": False,
            "provenance": "local",
        }

    def _definition_payload(self, symbol: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol_id": str(symbol["symbol_id"]),
            "symbol_name": str(symbol["symbol_name"]),
            "qualified_name": str(symbol["qualified_name"]),
            "file_path": str(symbol["file_path"]),
            "kind": str(symbol["kind"]),
            "signature": str(symbol["signature"]),
            "start_line": int(symbol["start_line"]),
            "end_line": int(symbol["end_line"]),
            "source": str(symbol["source"]),
            "provenance": str(symbol.get("provenance") or "local"),
        }

    def _memory_payload(
        self,
        symbol: dict[str, Any],
        *,
        agent_id: str | None,
        query: str,
        since: datetime,
        top_k: int,
    ) -> list[dict[str, Any]]:
        items = [
            *self._memory_block_items(symbol, agent_id=agent_id),
            *self._memory_passage_items(symbol, agent_id=agent_id, query=query, since=since, top_k=top_k),
        ]
        return items[: max(1, top_k * 2)]

    def _memory_block_items(self, symbol: dict[str, Any], *, agent_id: str | None) -> list[dict[str, Any]]:
        symbol_id = str(symbol["symbol_id"])
        blocks: list[MemoryBlock] = []
        for current_agent_id in self._agent_ids(agent_id):
            blocks.extend(self._memory_store.list_blocks(current_agent_id, limit=500))
        deduped = {block.id: block for block in blocks}.values()
        matched = [block for block in deduped if self._block_matches_symbol(block, symbol_id=symbol_id)]
        return [
            {
                "item_type": "block",
                "id": block.id,
                "label": block.label,
                "value": block.value,
                "metadata": block.metadata,
                "updated_at": block.updated_at.isoformat(),
            }
            for block in sorted(matched, key=lambda item: item.updated_at, reverse=True)
        ]

    def _memory_passage_items(
        self,
        symbol: dict[str, Any],
        *,
        agent_id: str | None,
        query: str,
        since: datetime,
        top_k: int,
    ) -> list[dict[str, Any]]:
        symbol_id = str(symbol["symbol_id"])
        patterns = self._symbol_patterns(symbol)
        passages: list[ArchivalPassage] = []
        for current_agent_id in self._agent_ids(agent_id):
            passages.extend(
                self._memory_store.search_passages(current_agent_id, query, top_k=max(top_k * 5, 10), since=since)
            )
        deduped = {passage.id: passage for passage in passages}.values()
        tagged = [passage for passage in deduped if f"symbol:{symbol_id}" in passage.tags]
        matched = tagged or [
            passage for passage in deduped if any(pattern.search(passage.text) for pattern in patterns)
        ]
        return [
            {
                "item_type": "passage",
                "id": passage.id,
                "text": passage.text,
                "source_ref": passage.source_ref,
                "tags": passage.tags,
                "created_at": passage.created_at.isoformat(),
            }
            for passage in sorted(matched, key=lambda item: item.created_at, reverse=True)[:top_k]
        ]

    def _trace_payload(self, symbol: dict[str, Any], *, since: datetime, limit: int) -> list[dict[str, Any]]:
        patterns = self._symbol_patterns(symbol)
        file_path = str(symbol["file_path"])
        traces = self._trace_store.list_traces(since=since, limit=50)
        matched = [
            trace for trace in traces if self._trace_matches_symbol(trace, file_path=file_path, patterns=patterns)
        ]
        return [
            {
                "id": trace.id,
                "task": trace.task,
                "status": trace.status,
                "created_at": trace.created_at.isoformat(),
                "summary": self._clip_text(trace.output_summary or trace.diff_summary, limit=180),
            }
            for trace in matched[:limit]
        ]

    def _decision_payload(self, symbol: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        decisions_root = self._repo_root / "docs" / "decisions"
        if not decisions_root.exists():
            return []
        patterns = self._symbol_patterns(symbol)
        items: list[dict[str, Any]] = []
        for path in sorted(decisions_root.rglob("*.md"))[:50]:
            text = path.read_text(encoding="utf-8")
            match = next((pattern.search(text) for pattern in patterns if pattern.search(text)), None)
            if match is None:
                continue
            items.append(
                {
                    "path": str(path.relative_to(self._repo_root).as_posix()),
                    "excerpt": self._excerpt(text, match.start(), match.end()),
                }
            )
            if len(items) >= limit:
                break
        return items

    def _related_tests_payload(self, symbol: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
        test_root = self._repo_root / "tests"
        if not test_root.exists():
            return []
        patterns = self._symbol_patterns(symbol)
        matches = self._engine.search_text(str(symbol["symbol_name"]), path="tests", limit=50)
        files: dict[str, list[int]] = {}
        for match in matches:
            text = (self._repo_root / match.file_path).read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in patterns):
                files.setdefault(match.file_path, []).append(match.line)
        items: list[dict[str, Any]] = []
        for file_path, lines in sorted(files.items()):
            outline = self._engine.file_outline(file_path=file_path, limit=50)
            for candidate in outline["files"].get(file_path, []):
                if not file_path.startswith("tests/"):
                    continue
                if any(int(candidate["line_start"]) <= line <= int(candidate["line_end"]) for line in lines):
                    items.append(
                        {
                            "file_path": file_path,
                            "name": str(candidate["name"]),
                            "qualified_name": str(candidate.get("qualified_name", candidate["name"])),
                            "kind": str(candidate["kind"]),
                            "signature": str(candidate["signature"]),
                            "start_line": int(candidate["line_start"]),
                            "end_line": int(candidate["line_end"]),
                        }
                    )
            if len(items) >= limit:
                break
        return items[:limit]

    def _trim_to_budget(self, payload: dict[str, Any], *, budget_tokens: int) -> dict[str, Any]:
        cloned = json.loads(json.dumps(payload, default=str))
        if not isinstance(cloned, dict):
            raise RuntimeError("symbol recall payload clone must stay a mapping")
        working: dict[str, Any] = cloned
        working["truncated_sections"] = []
        trim_order = ["tests", "decisions", "traces", "memory"]
        while self._compute_total_tokens({**working, "tokens_saved": 0}) > budget_tokens:
            trimmed = False
            for section in trim_order:
                items = working.get(section)
                if isinstance(items, list) and items:
                    items.pop()
                    if section not in working["truncated_sections"]:
                        working["truncated_sections"].append(section)
                    trimmed = True
                    break
            if not trimmed:
                break
        working["total_tokens"] = self._compute_total_tokens({**working, "tokens_saved": 0})
        return working

    def _finalize(self, payload: dict[str, Any], *, full_total_tokens: int) -> dict[str, Any]:
        finalized = dict(payload)
        finalized.setdefault("cache_hit", False)
        finalized.setdefault("provenance", "local")
        finalized.setdefault("truncated_sections", [])
        tokens_saved = 0
        while True:
            finalized["tokens_saved"] = tokens_saved
            total_tokens = self._compute_total_tokens(finalized)
            updated = max(0, full_total_tokens - total_tokens)
            if updated == tokens_saved:
                finalized["total_tokens"] = total_tokens
                return finalized
            tokens_saved = updated

    def _compute_total_tokens(self, payload: dict[str, Any]) -> int:
        total_tokens = 0
        while True:
            candidate = dict(payload)
            candidate["total_tokens"] = total_tokens
            measured = count_tokens(json.dumps(candidate, sort_keys=True, ensure_ascii=False, default=str))
            if measured == total_tokens:
                return measured
            total_tokens = measured

    def _agent_ids(self, agent_id: str | None) -> list[str | None]:
        if agent_id is None:
            return [None]
        if agent_id == "shared":
            return ["shared"]
        return [agent_id, "shared"]

    def _block_matches_symbol(self, block: MemoryBlock, *, symbol_id: str) -> bool:
        metadata = block.metadata
        if metadata.get("symbol_id") == symbol_id:
            return True
        symbol_ids = metadata.get("symbol_ids")
        return isinstance(symbol_ids, list) and symbol_id in symbol_ids

    def _trace_matches_symbol(self, trace: Trace, *, file_path: str, patterns: list[re.Pattern[str]]) -> bool:
        touched = [self._trace_file_path(item) for item in trace.files_touched]
        if file_path in touched:
            return True
        haystacks = [trace.task, trace.diff_summary, trace.output_summary, *trace.errors_seen]
        return any(pattern.search(text) for pattern in patterns for text in haystacks if isinstance(text, str))

    def _trace_file_path(self, item: str | FileEditRecord) -> str:
        if isinstance(item, FileEditRecord):
            return item.path
        return str(item)

    def _symbol_patterns(self, symbol: dict[str, Any]) -> list[re.Pattern[str]]:
        names = [str(symbol["qualified_name"]), str(symbol["symbol_name"])]
        patterns: list[re.Pattern[str]] = []
        for name in names:
            if "." in name:
                patterns.append(re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"))
            else:
                patterns.append(re.compile(rf"\b{re.escape(name)}\b"))
        return patterns

    def _excerpt(self, text: str, start: int, end: int, *, radius: int = 90) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return self._clip_text(text[left:right].strip(), limit=220)

    def _clip_text(self, text: str, *, limit: int) -> str:
        return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


__all__ = ["SymbolRecallCapability"]
