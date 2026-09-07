"""Lesson promoter capability.

Failed trace -> embedding -> nearest-neighbor cluster -> inbox candidate.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from lemoncrow.core.foundation.lesson_models import LessonCandidate, LessonPromotion
from lemoncrow.core.foundation.models import Rubric, Trace
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.embeddings.factory import get_embedder
from lemoncrow.infra.embeddings.null_embedder import NullEmbedder
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.infra.storage.vector import cosine_similarity
from lemoncrow.pro.capabilities.lesson_promotion.draft import draft_lesson_candidate
from lemoncrow.pro.capabilities.lesson_promotion.models import TypedLesson
from lemoncrow.pro.capabilities.lesson_promotion.reflection import draft_lesson_body
from lemoncrow.pro.capabilities.lesson_promotion.store import TypedLessonStore
from lemoncrow.pro.foundation.extractor import extract_candidate

_log = logging.getLogger(__name__)


def ingest_failed_trace(store: StoreBundle, trace: Trace) -> None:
    """Best-effort: feed a failed trace into the lesson inbox from any trace-record
    path (SDK, MCP tool, runtime session). Never raises -- lesson extraction must
    not break trace recording. Call after the trace is persisted so the
    store-backed cluster lookup can see it."""
    if trace.status != "failed" or not trace.errors_seen:
        return
    try:
        LessonPromoterCapability(store).ingest_trace(trace)
    except Exception:  # noqa: BLE001 - lesson ingest is best-effort
        _log.debug("lesson ingest skipped for trace %s", trace.id, exc_info=True)


class LessonPromoterCapability:
    """Create and review lesson candidates from failed traces."""

    def __init__(
        self,
        store: StoreBundle,
        *,
        now: Callable[[], datetime] | None = None,
        embedder: Embedder | None = None,
        cluster_threshold: float | None = None,
    ) -> None:
        self.store = store
        self._now = now or (lambda: datetime.now(UTC))
        self._embedder = embedder or get_embedder()
        self._cluster_threshold = cluster_threshold or float(
            os.environ.get("LEMONCROW_LESSON_CLUSTER_THRESHOLD", "0.85")
        )
        self._trace_embedding_cache: dict[str, list[float]] = {}

    def _trace_text(self, trace: Trace) -> str:
        commands: list[str] = []
        for item in trace.commands_run:
            if isinstance(item, str):
                commands.append(item)
            else:
                commands.append(str(item.command))
        errors = "\n".join(trace.errors_seen)
        return "\n".join([*commands, errors, trace.diff_summary, trace.output_summary])

    def _embed_trace(self, trace: Trace) -> list[float]:
        cached = self._trace_embedding_cache.get(trace.id)
        if cached is not None:
            return cached
        text = self._trace_text(trace)
        try:
            vectors = self._embedder.embed([text])
        except Exception as exc:  # noqa: BLE001 - an embedder failure must not abort ingest
            # Fall back to an empty vector for THIS trace only. Reassigning
            # self._embedder would permanently disable embeddings for the rest of
            # the instance's life on a single transient failure.
            _log.warning("lesson embedding unavailable for trace %s, using empty vector: %s", trace.id, exc)
            vectors = NullEmbedder().embed([text])
        embedding = vectors[0] if vectors and vectors[0] else []
        self._trace_embedding_cache[trace.id] = embedding
        return embedding

    def _cluster_key(self, trace: Trace, embedding: list[float]) -> str:
        if embedding:
            raw = ",".join(f"{value:.4f}" for value in embedding[:16])
        else:
            raw = self._trace_text(trace)
        return "semantic:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _recent_inbox(self, domain: str, days: int = 30) -> list[LessonCandidate]:
        cutoff = self._now() - timedelta(days=days)
        out: list[LessonCandidate] = []
        for item in self.store.lessons.list_lesson_candidates(domain=domain, status="inbox", limit=500):
            if item.created_at >= cutoff:
                out.append(item)
        return out

    def _nearest_cluster(
        self,
        *,
        domain: str,
        embedding: list[float],
        top_k: int = 8,
        cluster_fingerprint: str | None = None,
    ) -> list[LessonCandidate]:
        scored: list[tuple[float, LessonCandidate]] = []
        for candidate in self._recent_inbox(domain):
            if not embedding or not candidate.embedding:
                # No usable vector on one side (e.g. NullEmbedder in offline/test
                # runs, where every embedding is []) -- cosine similarity is
                # meaningless, so fall back to the exact-text fingerprint that
                # _cluster_key() already computes for this case. Without this,
                # every inbox candidate looks embedding-less forever and each
                # recurrence spawns a near-duplicate instead of refreshing it.
                if cluster_fingerprint and candidate.cluster_fingerprint == cluster_fingerprint:
                    scored.append((1.0, candidate))
                continue
            try:
                sim = cosine_similarity(embedding, candidate.embedding)
            except ValueError:
                sim = 0.0
            if sim < self._cluster_threshold:
                continue
            scored.append((sim, candidate))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry[1] for entry in scored[:top_k]]

    def _recent_trace_cluster(
        self,
        *,
        domain: str,
        current_trace_id: str,
        embedding: list[float],
        limit: int = 8,
    ) -> list[Trace]:
        scored: list[tuple[float, Trace]] = []
        for trace in self.store.history.list_traces(domain=domain, status="failed", limit=500):
            if trace.id == current_trace_id:
                continue
            if not trace.errors_seen:
                continue
            try:
                sim = cosine_similarity(embedding, self._embed_trace(trace))
            except ValueError:
                sim = 0.0
            if sim < self._cluster_threshold:
                continue
            scored.append((sim, trace))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def ingest_trace(self, trace: Trace) -> LessonCandidate | None:
        """Ingest one failed trace and create an inbox candidate when cluster size >= 3.

        Neighbor lookup is store-backed (``_recent_trace_cluster`` +
        ``_nearest_cluster``), not in-process state, so clustering works whether
        the caller holds a long-lived capability or builds a fresh one per
        recorded trace (the SDK ``record_trace`` seam). Persist the trace before
        calling this so the store-backed lookup can see it among its peers.
        """
        if trace.status != "failed":
            return None
        if not trace.errors_seen:
            return None

        embedding = self._embed_trace(trace)
        cluster_fingerprint = self._cluster_key(trace, embedding)
        # Cheap: scores stored candidate vectors only (no re-embedding).
        neighbors = self._nearest_cluster(
            domain=trace.domain, embedding=embedding, cluster_fingerprint=cluster_fingerprint
        )
        if neighbors:
            # An inbox candidate already represents this cluster -> refresh it in
            # place rather than inserting a near-duplicate per recurrence, and
            # skip the expensive recent-trace scan (it re-embeds up to 500 traces
            # and would only re-derive an already-formed cluster).
            existing_id: str | None = neighbors[0].id
            trace_neighbors: list[Trace] = []
        else:
            existing_id = None
            trace_neighbors = self._recent_trace_cluster(
                domain=trace.domain,
                current_trace_id=trace.id,
                embedding=embedding,
            )
            if len(trace_neighbors) + 1 < 3:
                return None

        traces = [trace, *trace_neighbors]
        seen_trace_ids = {item.id for item in traces}
        for neighbor in neighbors[:6]:
            for trace_id in neighbor.evidence_trace_ids[:1]:
                if trace_id in seen_trace_ids:
                    continue
                found = self.store.history.get_trace(trace_id)
                if found is not None:
                    traces.append(found)
                    seen_trace_ids.add(trace_id)

        traces = traces[:8]

        existing_blocks = self.store.knowledge.list_blocks(domain=trace.domain, include_deprecated=False)
        candidate = draft_lesson_candidate(
            traces=traces,
            domain=trace.domain,
            cluster_fingerprint=cluster_fingerprint,
            embedding=embedding,
            existing_blocks=existing_blocks,
        )
        candidate.body = draft_lesson_body(traces)
        candidate.evidence = {
            "trace_ids": [item.id for item in traces],
            "embedding_provenance": self._embedder.__class__.__name__,
            "cluster_threshold": self._cluster_threshold,
        }
        candidate.embedding_provenance = self._embedder.__class__.__name__
        if existing_id is not None:
            # Refresh the existing cluster candidate in place (ON CONFLICT(id) DO
            # UPDATE) instead of inserting a near-duplicate for every recurrence.
            candidate.id = existing_id
        self.store.lessons.upsert_lesson_candidate(candidate)
        return candidate

    def inbox(self, *, domain: str | None = None, limit: int = 25) -> list[LessonCandidate]:
        return self.store.lessons.list_lesson_candidates(domain=domain, status="inbox", limit=limit)

    def decide(
        self,
        *,
        lesson_id: str,
        decision: str,
        reviewer: str,
        reason: str,
    ) -> dict[str, Any]:
        candidate = self.store.lessons.get_lesson_candidate(lesson_id)
        if candidate is None:
            raise ValueError(f"lesson not found: {lesson_id}")

        now = self._now()
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")

        candidate.reviewer = reviewer
        candidate.decision_reason = reason
        candidate.decision_at = now

        if decision == "reject":
            candidate.status = "rejected"
            self.store.lessons.upsert_lesson_candidate(candidate)
            return {"lesson": candidate.model_dump(mode="json"), "promotion": None}

        # Build everything that can fail (promotion, typed lesson) BEFORE
        # persisting approval, so a raise can't leave a candidate marked
        # approved with no promotion or typed lesson.
        promotion = self._promote(candidate)
        typed_lesson: TypedLesson | None = None
        if candidate.kind in {"route-preference", "cost-cap"}:
            typed_lesson = self._typed_lesson_from_candidate(candidate)

        candidate.status = "approved"
        self.store.lessons.upsert_lesson_candidate(candidate)
        self.store.lessons.upsert_lesson_promotion(promotion)
        if typed_lesson is not None:
            TypedLessonStore(self.store.lessons.root).upsert_lesson(typed_lesson)
        return {
            "lesson": candidate.model_dump(mode="json"),
            "promotion": promotion.model_dump(mode="json"),
            "typed_lesson": typed_lesson.model_dump(mode="json") if typed_lesson is not None else None,
        }

    def _promote(self, candidate: LessonCandidate) -> LessonPromotion:
        kind = str(getattr(candidate, "kind", ""))

        if kind in {"new_block", "playbook"}:
            block = candidate.proposed_block
            if block is None:
                # Fallback to the existing extractor path from evidence traces.
                trace_id = candidate.evidence_trace_ids[0]
                trace = self.store.history.get_trace(trace_id)
                if trace is None:
                    raise ValueError("missing evidence trace for new_block promotion")
                block = extract_candidate(trace).block
            self.store.knowledge.upsert_block(block, write_markdown=False)
            return LessonPromotion(lesson_id=candidate.id, published_block_id=block.id)

        if kind == "edit_block":
            if not candidate.target_id:
                raise ValueError("edit_block promotion requires target_id")
            block = self.store.knowledge.get_block(candidate.target_id)
            if block is None:
                raise ValueError(f"target block not found: {candidate.target_id}")
            dead_end = candidate.cluster_fingerprint
            if dead_end and dead_end not in block.dead_ends:
                block.dead_ends.append(dead_end)
                block.updated_at = self._now()
                self.store.knowledge.upsert_block(block, write_markdown=False)
            return LessonPromotion(lesson_id=candidate.id, edited_block_id=block.id)

        if kind in {"new_rubric_check", "rubric_check"}:
            check = candidate.proposed_rubric_check
            if not check:
                raise ValueError("new_rubric_check promotion requires proposed_rubric_check")
            rubrics = self.store.knowledge.list_rubrics(domain=candidate.domain)
            if rubrics:
                rubric = rubrics[0]
            else:
                rubric = Rubric(
                    id=f"rubric_{candidate.domain.replace('.', '_')}",
                    domain=candidate.domain,
                    required_checks=[],
                    block_if_missing=[],
                )
            if check not in rubric.required_checks:
                rubric.required_checks.append(check)
            if check not in rubric.block_if_missing:
                rubric.block_if_missing.append(check)
            self.store.knowledge.upsert_rubric(rubric, write_yaml=False)
            return LessonPromotion(lesson_id=candidate.id)

        if kind in {"route-preference", "cost-cap"}:
            return LessonPromotion(lesson_id=candidate.id)

        raise ValueError(f"unsupported lesson candidate kind: {candidate.kind}")

    def _typed_lesson_from_candidate(self, candidate: LessonCandidate) -> TypedLesson:
        lesson_payload = dict(candidate.evidence.get("typed_lesson") or {})
        if not lesson_payload:
            raise ValueError(f"typed lesson candidate {candidate.id} is missing typed_lesson evidence")
        lesson_payload.setdefault("id", candidate.id)
        lesson_payload.setdefault("kind", candidate.kind)
        lesson_payload.setdefault(
            "source_session_id",
            lesson_payload.get("source_session_id") or candidate.evidence.get("source_session_id"),
        )
        lesson_payload.setdefault("confidence", candidate.confidence)
        lesson_payload.setdefault("captured_at", candidate.created_at)
        return TypedLesson.model_validate(lesson_payload)
