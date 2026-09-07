"""Hybrid ranking for archival memory recall."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from lemoncrow.core.foundation.memory_models import ArchivalPassage
from lemoncrow.infra.storage.vector import cosine_similarity
from lemoncrow.pro.capabilities.archival_recall.ann import ArchivalAnnIndex, ann_retrieval_enabled

# Process-wide ANN cache for archival recall. Reused across calls so the HNSW
# graph is built once per passage-set signature (N16) instead of per query.
_ARCHIVAL_ANN_INDEX = ArchivalAnnIndex()


@dataclass(frozen=True)
class RankedPassage:
    passage: ArchivalPassage
    score: float
    bm25_norm: float
    cosine: float


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_scores(query: str, passages: list[ArchivalPassage]) -> dict[str, float]:
    query_tokens = _tokens(query)
    if not query_tokens or not passages:
        return {p.id: 0.0 for p in passages}

    docs = [_tokens(p.text + " " + " ".join(p.tags)) for p in passages]
    avg_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))

    scores: dict[str, float] = {}
    total_docs = len(docs)
    for passage, doc in zip(passages, docs, strict=True):
        tf = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = math.log((total_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf[term] + 1.5 * (1 - 0.75 + 0.75 * doc_len / max(avg_len, 1.0))
            score += idf * ((tf[term] * 2.5) / denom)
        scores[passage.id] = score
    return scores


def rank_archival_passages(
    *,
    query: str,
    passages: list[ArchivalPassage],
    query_embedding: list[float] | None = None,
    tags: list[str] | None = None,
    since: datetime | None = None,
    top_k: int = 5,
    embedding_model: str | None = None,
    valid_as_of: datetime | None = None,
    ann_index: ArchivalAnnIndex | None = None,
) -> list[RankedPassage]:
    """Rank archival passages with hybrid BM25 and cosine scoring.

    G5: when ``LEMONCROW_ANN_RETRIEVAL`` is enabled and a ``query_embedding`` is
    supplied, an opt-in ANN (datasketch HNSW) accelerates the *vector* side --
    cosine is computed only for ANN neighbours plus the most-recent-N passages.
    Lexical (BM25) recall is unchanged (every filtered passage still scores), so
    no lexically-relevant passage is dropped. With the flag off, or whenever the
    set is small / the lib is missing, behaviour is byte-identical to the
    brute-force path. N5 (model-id/dim drift) is enforced inside the ANN.
    """
    filtered = passages
    if tags:
        required = set(tags)
        filtered = [p for p in filtered if required.issubset(set(p.tags))]
    if since is not None:
        filtered = [p for p in filtered if p.created_at >= since]
    # N13: opt-in bi-temporal recall filter. Default ``valid_as_of=None`` skips
    # this entirely, so existing recall behaviour is byte-identical; when a
    # moment is supplied, passages whose validity window does not cover it
    # (e.g. invalidated by a calibrated code change) are excluded from recall.
    if valid_as_of is not None:
        filtered = [p for p in filtered if p.is_valid_at(valid_as_of)]
    if not filtered:
        return []

    bm25 = _bm25_scores(query, filtered)
    max_bm25 = max(bm25.values(), default=0.0)
    bm25_norm = {pid: (score / max_bm25 if max_bm25 > 0 else 0.0) for pid, score in bm25.items()}

    vector_enabled = bool(query_embedding)
    # G5: ANN-narrowed set of passages eligible for cosine scoring. ``None`` means
    # "score every passage" (brute-force fallback / flag off / small set), which
    # reproduces today's results exactly.
    cosine_candidate_ids: set[str] | None = None
    if vector_enabled and embedding_model and ann_retrieval_enabled():
        cosine_candidate_ids = (ann_index if ann_index is not None else _ARCHIVAL_ANN_INDEX).candidate_ids(
            query_embedding or [],
            filtered,
            model_id=embedding_model,
            dim=len(query_embedding or []),
            top_k=top_k,
        )

    ranked: list[RankedPassage] = []
    for passage in filtered:
        cosine = 0.0
        score_cosine = cosine_candidate_ids is None or passage.id in cosine_candidate_ids
        if vector_enabled and score_cosine and passage.embedding and passage.embedding_provenance != "legacy_stub":
            try:
                cosine = max(0.0, min(1.0, cosine_similarity(query_embedding or [], passage.embedding)))
            except ValueError:
                cosine = 0.0
        score = (0.6 * cosine) + (0.4 * bm25_norm.get(passage.id, 0.0))
        ranked.append(
            RankedPassage(
                passage=passage,
                score=score,
                bm25_norm=bm25_norm.get(passage.id, 0.0),
                cosine=cosine,
            )
        )

    ranked.sort(key=lambda item: (item.score, item.bm25_norm, item.passage.created_at), reverse=True)
    return ranked[:top_k]


__all__ = ["RankedPassage", "rank_archival_passages"]
