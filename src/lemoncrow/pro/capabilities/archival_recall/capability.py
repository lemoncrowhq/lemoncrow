"""Archival memory archive and recall capability."""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import tiktoken
from blake3 import blake3

from lemoncrow.core.foundation.memory_models import ArchivalPassage, ArchivalSource, MemoryRecall
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.storage.memory_store import MemoryStore
from lemoncrow.pro.capabilities.archival_recall.ann import ArchivalAnnIndex
from lemoncrow.pro.capabilities.archival_recall.ranking import rank_archival_passages

_log = logging.getLogger(__name__)

# Candidate window pulled from the store before in-Python ranking. Session-recall
# can write thousands of passages per run, so a small window silently excludes
# strong older matches; raise via env for very large stores.
_RECALL_CANDIDATE_LIMIT = int(os.environ.get("LEMONCROW_RECALL_CANDIDATE_LIMIT", "2000"))
_window_saturation_warned = False

# Embedding is a blocking network round-trip for remote backends (openai/ollama/
# letta) with no provider-side timeout, so an unreachable provider would otherwise
# stall every memory(op=recall|store_fact|archive) call. Bound it. The default
# NullEmbedder returns empty in-process and well under this ceiling, so the guard is a
# no-op for it beyond the timeout itself.
_EMBED_TIMEOUT_S = float(os.environ.get("LEMONCROW_EMBED_TIMEOUT_S", "10"))

# Cap concurrent in-flight embeds. Each embed still runs on its own daemon thread
# (so a hung provider call dies with the process and never blocks shutdown), but a
# bounded semaphore caps how many such threads can exist at once. Under a sustained
# outage the first _EMBED_MAX_INFLIGHT calls leak their hung daemon threads and
# every later call fails the acquire and falls straight back to lexical/no-vector
# without spawning a thread, so leaked threads can never grow past the cap.
_EMBED_MAX_INFLIGHT = int(os.environ.get("LEMONCROW_EMBED_MAX_INFLIGHT", "3"))
_embed_inflight = threading.Semaphore(_EMBED_MAX_INFLIGHT)


def _embed_with_timeout(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    """Embed under a hard timeout; return [] on timeout/error or when saturated.

    The embed runs on a daemon thread joined with a timeout: a hung provider call
    is abandoned (the daemon thread dies with the process) instead of stalling
    the caller, which falls back to lexical/recency ranking (recall) or persists
    the passage without a vector (archive). A module-level semaphore caps how many
    embed threads are in flight at once; when the cap is already held by abandoned
    hung calls the acquire fails immediately and we fall back without spawning, so
    a sustained provider outage can never leak unbounded threads."""
    if not _embed_inflight.acquire(blocking=False):
        _log.warning(
            "embedder %s at in-flight cap (%d); falling back to lexical ranking", embedder.name, _EMBED_MAX_INFLIGHT
        )
        return []
    result: list[list[float]] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            result.extend(embedder.embed(texts))
        except BaseException as exc:  # noqa: BLE001 - surfaced via `error`, never raised here
            error.append(exc)
        finally:
            _embed_inflight.release()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=_EMBED_TIMEOUT_S)
    if worker.is_alive():
        _log.warning(
            "embedder %s timed out after %.1fs; falling back to lexical ranking", embedder.name, _EMBED_TIMEOUT_S
        )
        return []
    if error:
        _log.warning("embedder %s failed; falling back to lexical ranking", embedder.name, exc_info=error[0])
        return []
    return result


def _warn_window_saturated() -> None:
    """Warn at most once per process that the recall candidate window saturated.
    recall() is the shared path for memory.db, recall.db and get_context, so an
    unthrottled warning would repeat on every call against a saturated store."""
    global _window_saturation_warned
    if _window_saturation_warned:
        return
    _window_saturation_warned = True
    _log.warning(
        "recall candidate window saturated at %d passages; older matches are excluded from "
        "ranking (raise LEMONCROW_RECALL_CANDIDATE_LIMIT)",
        _RECALL_CANDIDATE_LIMIT,
    )


def _ann_persist_path(store: MemoryStore) -> Path | None:
    """On-disk location for this store's persistent ANN graph (next to its db).

    None for stores without a file path (non-SQLite backends) -- the ANN index
    then stays in-process only.
    """
    db_path = getattr(store, "db_path", None)
    if isinstance(db_path, Path):
        return db_path.with_name(f"{db_path.stem}.ann.pkl")
    return None


class ArchivalRecallCapability:
    def __init__(self, store: MemoryStore, embedder: Embedder, *, redactor: Callable[[str], str]):
        self._store = store
        self._embedder = embedder
        self._redactor = redactor
        self._ann_index = ArchivalAnnIndex(persist_path=_ann_persist_path(store))

    def archive(
        self,
        *,
        text: str,
        source: ArchivalSource,
        agent_id: str | None = None,
        source_ref: str = "",
        tags: list[str] | None = None,
    ) -> ArchivalPassage:
        clean = self._redactor(text)
        chunks = _chunk_text(clean)
        passages = [
            ArchivalPassage(
                agent_id=agent_id or "shared",
                text=chunk,
                tags=tags or [],
                source=source,
                source_ref=source_ref,
                dedup_hash=blake3(chunk.encode("utf-8")).hexdigest(),
            )
            for chunk in chunks
        ]
        if not passages:  # pragma: no cover - _chunk_text always returns one item
            raise ValueError("archive text produced no passages")

        # insert_passage does the dedup_hash lookup + insert and returns the
        # authoritative passage (existing id + dedup_hit=True on a duplicate),
        # which the caller reads as passage.id / passage.dedup_hit. That decision
        # must be synchronous, so the persist stays inline. Only the embed is a
        # slow remote round-trip: _embed_with_timeout bounds it so an unreachable
        # provider can't stall the call -- on timeout/failure it returns [] and
        # the passage persists without a vector (still lexically recallable).
        embeddings: list[list[float]] = []
        if self._embedder.dim > 0:
            embeddings = _embed_with_timeout(self._embedder, chunks)
        stored: list[ArchivalPassage] = []
        for idx, passage in enumerate(passages):
            embedding = embeddings[idx] if idx < len(embeddings) and embeddings[idx] else None
            to_store = passage.model_copy(
                update={
                    "embedding": embedding,
                    "embedding_model": self._embedder.name if embedding is not None else "",
                    "embedding_provenance": self._embedder.__class__.__name__,
                }
            )
            stored.append(self._store.insert_passage(to_store))
        return stored[0]

    def recall(
        self,
        *,
        agent_id: str | None,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
        since: datetime | None = None,
    ) -> tuple[list[ArchivalPassage], MemoryRecall]:
        clean_query = self._redactor(query)
        query_embedding: list[float] | None = None
        if self._embedder.dim > 0:
            # Timeout-guarded: recall needs the vector to rank, so it can't be
            # fire-and-forget, but a slow/unreachable provider must not stall the
            # call. On timeout/failure _embed_with_timeout returns [] and we fall
            # back to lexical/recency ranking below (query_embedding stays None).
            vectors = _embed_with_timeout(self._embedder, [clean_query])
            if vectors and vectors[0]:
                query_embedding = vectors[0]

        # G5: pass the live embedder model-id so the opt-in ANN can N5-gate
        # passages to the current vector space (no-op when the flag is off).
        embedding_model = self._embedder.name if query_embedding else None
        passages = self._store.list_passages(agent_id, tags=tags, since=since, limit=_RECALL_CANDIDATE_LIMIT)
        if len(passages) >= _RECALL_CANDIDATE_LIMIT:
            _warn_window_saturated()
        ranked = rank_archival_passages(
            query=clean_query,
            passages=passages,
            query_embedding=query_embedding,
            tags=tags,
            since=since,
            top_k=top_k,
            embedding_model=embedding_model,
            ann_index=self._ann_index,
        )
        recall_query = clean_query
        if not ranked:
            widened_query = _widen_query(clean_query)
            if widened_query and widened_query != clean_query:
                ranked = rank_archival_passages(
                    query=widened_query,
                    passages=passages,
                    query_embedding=query_embedding,
                    tags=tags,
                    since=since,
                    top_k=top_k,
                    embedding_model=embedding_model,
                    ann_index=self._ann_index,
                )
                recall_query = widened_query
        selected = [item.passage for item in ranked]
        recall = MemoryRecall(
            agent_id=agent_id or "shared",
            query=recall_query,
            top_passages=[passage.id for passage in selected],
            selected_passage_id=selected[0].id if selected else None,
        )
        self._store.record_recall(recall)
        return selected, recall


def _chunk_text(text: str, *, max_tokens: int = 800, window_tokens: int = 400, overlap: int = 80) -> list[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks: list[str] = []
    step = max(1, window_tokens - overlap)
    for start in range(0, len(tokens), step):
        piece = tokens[start : start + window_tokens]
        if not piece:
            break
        chunks.append(encoding.decode(piece))
        if start + window_tokens >= len(tokens):
            break
    return chunks


def _widen_query(query: str) -> str:
    without_quotes = re.sub(r"(['\"]).*?\1", " ", query.lower())
    without_bool = re.sub(r"\bAND\b", " OR ", without_quotes, flags=re.IGNORECASE)
    terms = re.findall(r"[a-z0-9_]+", without_bool)
    stop = {"and", "or", "the", "a", "an", "to", "of", "in", "for", "with", "on"}
    useful = [term for term in terms if term not in stop]
    return " OR ".join(useful[:3])


__all__ = ["ArchivalRecallCapability"]
