"""Grow-and-refine: fold a new candidate into the nearest existing block.

ACE-style incremental update. Before inserting a freshly extracted candidate,
check whether a sufficiently similar block already exists in the same domain.
If so, merge the candidate's new knowledge (dead-ends, procedure steps,
verification, failure signals, triggers) into that block instead of inserting
a near-duplicate — avoiding 'context collapse' in the block store.

Similarity uses the local embedder when one is supplied (cosine over the block
'dedup text'); otherwise it falls back to token Jaccard, reusing the same
helpers the retriever uses for de-duplication. No vector DB required.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from lemoncrow.core.foundation.models import Playbook
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.pro.foundation.retriever import _dedup_text, _dedup_tokens, _jaccard_tokens

_EMBED_SIMILARITY_THRESHOLD = 0.86
_TOKEN_SIMILARITY_THRESHOLD = 0.60
_LIST_CAP = 16


@dataclass(frozen=True)
class GrowResult:
    """Outcome of a grow-or-create decision.

    ``merged`` is True when ``block`` is an existing block refined in place
    (``target_id`` set); False when ``block`` is the unchanged new candidate.
    """

    block: Playbook
    merged: bool
    target_id: str | None
    score: float


def grow_or_create(
    incoming: Playbook,
    existing: Sequence[Playbook],
    *,
    embedder: Embedder | None = None,
) -> GrowResult:
    """Merge *incoming* into the nearest same-domain block, or keep it new."""
    target, score = _best_match(incoming, existing, embedder)
    if target is None:
        return GrowResult(block=incoming, merged=False, target_id=None, score=score)
    return GrowResult(block=merge_blocks(target, incoming), merged=True, target_id=target.id, score=score)


def merge_blocks(target: Playbook, incoming: Playbook) -> Playbook:
    """Fold *incoming*'s knowledge into *target*, preserving target identity."""
    return target.model_copy(
        update={
            "triggers": _union(target.triggers, incoming.triggers),
            "file_patterns": _union(target.file_patterns, incoming.file_patterns),
            "tool_patterns": _union(target.tool_patterns, incoming.tool_patterns),
            "task_types": _union(target.task_types, incoming.task_types),
            "dead_ends": _union(target.dead_ends, incoming.dead_ends),
            "procedure": _union(target.procedure, incoming.procedure),
            "verification": _union(target.verification, incoming.verification),
            "failure_signals": _union(target.failure_signals, incoming.failure_signals),
            "updated_at": datetime.now(UTC),
        }
    )


def _best_match(
    incoming: Playbook,
    existing: Sequence[Playbook],
    embedder: Embedder | None,
) -> tuple[Playbook | None, float]:
    pool = [b for b in existing if b.domain == incoming.domain and b.id != incoming.id]
    if not pool:
        return None, 0.0
    scores = _embed_scores(incoming, pool, embedder) if embedder is not None else None
    if scores is not None:
        threshold = _EMBED_SIMILARITY_THRESHOLD
    else:
        incoming_tokens = _dedup_tokens(incoming)
        scores = [_jaccard_tokens(incoming_tokens, _dedup_tokens(b)) for b in pool]
        threshold = _TOKEN_SIMILARITY_THRESHOLD
    best_index = max(range(len(pool)), key=lambda i: scores[i])
    best_score = scores[best_index]
    if best_score >= threshold:
        return pool[best_index], best_score
    return None, best_score


def _embed_scores(
    incoming: Playbook,
    pool: Sequence[Playbook],
    embedder: Embedder,
) -> list[float] | None:
    texts = [_dedup_text(incoming), *[_dedup_text(b) for b in pool]]
    try:
        vectors = embedder.embed(texts)
    except Exception:  # noqa: BLE001 - embedding is optional; degrade to tokens
        return None
    if not vectors or len(vectors) != len(pool) + 1:
        return None
    base = vectors[0]
    return [_cosine(base, vec) for vec in vectors[1:]]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _union(existing: list[str], incoming: list[str], cap: int = _LIST_CAP) -> list[str]:
    out: list[str] = []
    for item in [*existing, *incoming]:
        if item and item not in out:
            out.append(item)
    return out[:cap]


__all__ = ["GrowResult", "grow_or_create", "merge_blocks"]
