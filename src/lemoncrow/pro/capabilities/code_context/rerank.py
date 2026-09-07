"""Optional second-stage reranking for code search candidates."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lemoncrow.core.service.telemetry import emit_product_local
from lemoncrow.infra.embeddings.ollama_embedder import _resolve_host
from lemoncrow.infra.internal_llm.exceptions import OllamaUnavailable
from lemoncrow.pro.capabilities.code_context.embedding import render_embedding_text
from lemoncrow.pro.capabilities.code_context.models import SymbolRecord

DEFAULT_CODE_RERANKER_MODEL = "qllama/bge-reranker-v2-m3"
_DEFAULT_MIN_CANDIDATES = 8
_DEFAULT_TOP_N = 15
_DEFAULT_MAX_LATENCY_MS = 250
_DEFAULT_TIMEOUT_FLOOR_MS = 25
_MAX_DOCUMENT_CHARS = 1_200
# Raw-source prefix collapsed in _candidate_text before the _MAX_DOCUMENT_CHARS
# cut; large enough that the collapsed prefix always covers the cut for real code.
_SOURCE_PREFIX_CHARS = 16_384
_CACHE_FINGERPRINT_VERSION = 1

_RerankScorer = Callable[[str, list[str], float], list[float]]


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _int_env(name: str, *, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


class SearchReranker:
    """Best-effort local reranker for semantic and hybrid code search."""

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        min_candidates: int | None = None,
        top_n: int | None = None,
        max_latency_ms: int | None = None,
        scorer: _RerankScorer | None = None,
    ) -> None:
        self.model = (model or os.getenv("LEMONCROW_CODE_RERANKER_MODEL") or DEFAULT_CODE_RERANKER_MODEL).strip()
        self.host = _resolve_host(host or os.getenv("LEMONCROW_CODE_RERANKER_HOST"))
        self.endpoint = f"{self.host}/api/rerank"
        self.min_candidates = max(
            2,
            (
                min_candidates
                if min_candidates is not None
                else _int_env(
                    "LEMONCROW_CODE_RERANKER_MIN_CANDIDATES",
                    default=_DEFAULT_MIN_CANDIDATES,
                    minimum=2,
                )
            ),
        )
        self.top_n = max(
            2,
            (
                top_n
                if top_n is not None
                else _int_env("LEMONCROW_CODE_RERANKER_TOP_N", default=_DEFAULT_TOP_N, minimum=2)
            ),
        )
        self.max_latency_ms = max(
            _DEFAULT_TIMEOUT_FLOOR_MS,
            (
                max_latency_ms
                if max_latency_ms is not None
                else _int_env(
                    "LEMONCROW_CODE_RERANKER_MAX_LATENCY_MS",
                    default=_DEFAULT_MAX_LATENCY_MS,
                    minimum=_DEFAULT_TIMEOUT_FLOOR_MS,
                )
            ),
        )
        self.enabled = bool(self.model) and _bool_env("LEMONCROW_CODE_RERANKER_ENABLED", default=True)
        self._scorer = scorer or self._score_via_ollama

    def pre_rerank_limit(self, limit: int, *, mode: str, scope: str) -> int:
        if not self._is_enabled_for(mode=mode, scope=scope):
            return limit
        return max(limit, self.top_n)

    def cache_fingerprint(self, *, mode: str, scope: str) -> dict[str, object]:
        enabled = self._is_enabled_for(mode=mode, scope=scope)
        return {
            "version": _CACHE_FINGERPRINT_VERSION,
            "enabled": enabled,
            "model": self.model if enabled else None,
            "min_candidates": self.min_candidates if enabled else None,
            "top_n": self.top_n if enabled else None,
            "max_latency_ms": self.max_latency_ms if enabled else None,
        }

    def rerank(
        self,
        query: str,
        hits: Sequence[SymbolRecord],
        *,
        mode: str,
        scope: str,
        source_loader: Callable[[SymbolRecord], str],
    ) -> list[SymbolRecord]:
        ordered_hits = list(hits)
        if not self._is_enabled_for(mode=mode, scope=scope):
            return ordered_hits
        if len(ordered_hits) < self.min_candidates:
            self._emit(status="skipped", reason="candidate-threshold", candidates=len(ordered_hits))
            return ordered_hits

        started_at = time.perf_counter()
        rerank_window = ordered_hits[: self.top_n]
        documents = [self._candidate_text(symbol, source_loader=source_loader) for symbol in rerank_window]
        remaining_ms = self.max_latency_ms - ((time.perf_counter() - started_at) * 1000.0)
        if remaining_ms <= _DEFAULT_TIMEOUT_FLOOR_MS:
            self._emit(
                status="skipped",
                reason="latency-budget",
                candidates=len(ordered_hits),
                reranked=len(rerank_window),
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            )
            return ordered_hits

        try:
            scores = self._scorer(query, documents, remaining_ms / 1000.0)
        except OllamaUnavailable:
            self._emit(
                status="failed",
                reason="unavailable",
                candidates=len(ordered_hits),
                reranked=len(rerank_window),
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            )
            return ordered_hits

        if len(scores) != len(rerank_window):
            self._emit(
                status="failed",
                reason="shape-mismatch",
                candidates=len(ordered_hits),
                reranked=len(rerank_window),
                elapsed_ms=(time.perf_counter() - started_at) * 1000.0,
            )
            return ordered_hits

        ranked = sorted(
            enumerate(rerank_window),
            key=lambda item: (
                -scores[item[0]],
                -(item[1].score or 0.0),
                item[1].file_path,
                item[1].start_line,
                item[1].symbol_id,
            ),
        )
        reranked_hits = [symbol.model_copy(update={"score": float(scores[index])}) for index, symbol in ranked]
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self._emit(
            status="used",
            candidates=len(ordered_hits),
            reranked=len(rerank_window),
            elapsed_ms=elapsed_ms,
        )
        return reranked_hits + ordered_hits[self.top_n :]

    def _is_enabled_for(self, *, mode: str, scope: str) -> bool:
        return self.enabled and scope == "repo" and mode != "lexical"

    def _candidate_text(
        self,
        symbol: SymbolRecord,
        *,
        source_loader: Callable[[SymbolRecord], str],
    ) -> str:
        if symbol.provenance == "commit" or symbol.kind == "commit":
            text = "\n".join(part for part in (symbol.qualified_name, symbol.signature, symbol.doc_summary) if part)
        else:
            source = source_loader(symbol)
            if len(source) > _SOURCE_PREFIX_CHARS:
                # The document is cut to _MAX_DOCUMENT_CHARS after whitespace
                # collapsing, so for huge symbols (whole classes) collapsing the
                # full body is wasted work. Collapse a bounded prefix instead:
                # every character before the start of the prefix's final
                # (possibly cut-off) token is identical to what full collapsing
                # would produce, so when that boundary lies beyond the cut the
                # truncated document is byte-identical. Otherwise (pathological
                # whitespace-heavy source) fall back to the full text.
                bounded = render_embedding_text(symbol, source_text=source[:_SOURCE_PREFIX_CHARS])
                compact_prefix = " ".join(bounded.split())
                if compact_prefix.rfind(" ") >= _MAX_DOCUMENT_CHARS:
                    return compact_prefix[:_MAX_DOCUMENT_CHARS]
            text = render_embedding_text(symbol, source_text=source)
        compact = " ".join(text.split()).strip()
        if not compact:
            compact = " ".join(
                part
                for part in (
                    symbol.symbol_name,
                    symbol.qualified_name,
                    symbol.signature,
                    symbol.doc_summary,
                )
                if part
            ).strip()
        if len(compact) > _MAX_DOCUMENT_CHARS:
            compact = compact[:_MAX_DOCUMENT_CHARS]
        return compact

    def _score_via_ollama(self, query: str, documents: list[str], timeout_seconds: float) -> list[float]:
        request = Request(
            self.endpoint,
            data=json.dumps(
                {
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = self._request_json(request, timeout_seconds=max(timeout_seconds, _DEFAULT_TIMEOUT_FLOOR_MS / 1000.0))
        return self._parse_scores(payload, expected=len(documents))

    def _request_json(self, request: Request, *, timeout_seconds: float) -> dict[str, object]:
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            message = f"Ollama rerank request failed ({exc.code})"
            if detail:
                message = f"{message}: {detail}"
            raise OllamaUnavailable(message) from exc
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable("Ollama reranker is unavailable") from exc
        if not isinstance(payload, dict):
            raise OllamaUnavailable("Unexpected Ollama rerank payload")
        return payload

    def _parse_scores(self, payload: dict[str, object], *, expected: int) -> list[float]:
        raw_results = payload.get("results")
        if raw_results is None:
            raw_results = payload.get("data")
        if not isinstance(raw_results, list):
            raise OllamaUnavailable("Unexpected Ollama rerank response")

        indexed_scores: list[float | None] = [None] * expected
        ordered_scores: list[float] = []
        for idx, entry in enumerate(raw_results):
            if not isinstance(entry, dict):
                raise OllamaUnavailable("Unexpected Ollama rerank result entry")
            score_value = entry.get("relevance_score", entry.get("score"))
            if score_value is None:
                raise OllamaUnavailable("Missing Ollama rerank score")
            try:
                score = float(score_value)
            except (TypeError, ValueError) as exc:
                raise OllamaUnavailable("Invalid Ollama rerank score") from exc
            raw_index = entry.get("index")
            if isinstance(raw_index, int) and 0 <= raw_index < expected:
                indexed_scores[raw_index] = score
            elif idx < expected:
                ordered_scores.append(score)

        if any(score is not None for score in indexed_scores):
            if any(score is None for score in indexed_scores):
                raise OllamaUnavailable("Incomplete Ollama rerank result set")
            return [float(score) for score in indexed_scores if score is not None]
        if len(ordered_scores) != expected:
            raise OllamaUnavailable("Unexpected Ollama rerank result count")
        return ordered_scores

    def _emit(
        self,
        *,
        status: str,
        candidates: int,
        reason: str | None = None,
        reranked: int = 0,
        elapsed_ms: float | None = None,
    ) -> None:
        emit_product_local(
            "code_context_reranked",
            model=self.model,
            status=status,
            reason=reason,
            candidate_count=candidates,
            reranked_count=reranked,
            elapsed_ms=round(elapsed_ms, 2) if elapsed_ms is not None else None,
        )


__all__ = ["DEFAULT_CODE_RERANKER_MODEL", "SearchReranker"]
