"""Deterministic, query-gated relevance ranking for oversized tool output.

Only engages when a caller passes an explicit query -- with no query, a tool's
existing blind truncation is untouched; this module changes nothing about the
default path. Two tiers, tried in order:

  1. Semantic: cosine similarity via the repo's existing embedder factory
     (``lemoncrow.infra.embeddings.factory``), used only when a real (non-null)
     embedder is configured. Deterministic for a fixed model + fixed input --
     doesn't break the cache-stability the rest of the T7 spill/compaction
     pipeline depends on (see web_fetch._truncate_with_spill).
  2. Lexical (always available, zero deps): query-term coverage + frequency.
     Used automatically when no embedder is configured -- the default in a
     fresh checkout (LEMONCROW_CODE_EMBEDDER / LEMONCROW_EMBEDDER / OPENAI_API_KEY
     all unset -> NullEmbedder).

Either tier scores whole chunks verbatim -- no summarization, no generative
LLM call, fully reproducible for the same (query, content) pair.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")


def _terms(query: str) -> list[str]:
    return _WORD_RE.findall(query.lower())


def score_lexical(query: str, texts: Sequence[str]) -> list[float]:
    """Coverage-weighted term frequency.

    Distinct query terms present matter more than raw repetition, so a chunk
    mentioning 3 of 4 query words outranks one that repeats a single word 10
    times.
    """
    terms = _terms(query)
    if not terms:
        return [0.0] * len(texts)
    scores: list[float] = []
    for text in texts:
        low = text.lower()
        covered = 0
        occurrences = 0
        for term in set(terms):
            count = low.count(term)
            if count:
                covered += 1
                occurrences += count
        scores.append(covered * 10.0 + occurrences)
    return scores


def try_score_semantic(query: str, texts: Sequence[str]) -> list[float] | None:
    """Cosine similarity via the configured embedder, or ``None`` if unavailable.

    Never raises: returns ``None`` when no real embedder is configured (the
    default) or embedding fails for any reason -- callers fall back to
    :func:`score_lexical`.
    """
    try:
        from lemoncrow.infra.embeddings.factory import embed_documents, embed_queries, get_code_embedder

        embedder = get_code_embedder()
        if type(embedder).__name__ == "NullEmbedder":
            return None
        # Asymmetric embedding: the query gets the model's query-side
        # instruction prefix (when the embedder is task-aware), chunks get the
        # document-side one -- these dispatch helpers already handle both
        # TaskAwareEmbedder and plain Embedder (embed_queries/embed_documents
        # fall back to .embed() for the latter).
        query_vec = embed_queries(embedder, [query])[0]
        chunk_vecs = embed_documents(embedder, list(texts))
    except Exception:  # noqa: BLE001 -- ranking must never break the caller's fetch
        return None

    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    return [_cosine(query_vec, vec) for vec in chunk_vecs]


def rank_and_select(
    chunks: Sequence[tuple[str, str | None]],
    *,
    query: str,
    char_budget: int,
) -> tuple[str, dict[str, object]]:
    """Score *chunks* against *query*; keep the highest-scoring ones within
    ``char_budget``; reassemble the kept ones in ORIGINAL document order
    (readability) with an elision marker between non-adjacent selections.

    ``chunks`` is a list of ``(text, pin)`` pairs -- ``pin`` (e.g. a table
    header+separator) is charged once and shown once per contiguous run, not
    once per row, so pulling in 3 scattered rows from the same table doesn't
    triple-print the header.
    """
    if not chunks:
        return "", {"tier": "lexical", "chunks_total": 0, "chunks_kept": 0}
    texts = [c[0] for c in chunks]
    tier = "semantic"
    scores = try_score_semantic(query, texts)
    if scores is None:
        tier = "lexical"
        scores = score_lexical(query, texts)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    kept_idx: list[int] = []
    seen_pins: set[str] = set()
    used = 0
    for i in order:
        text, pin = chunks[i]
        pin_cost = len(pin) + 1 if pin and pin not in seen_pins else 0
        cost = len(text) + 1 + pin_cost
        if used + cost > char_budget:
            continue  # doesn't fit -- keep trying smaller, lower-ranked chunks
        used += cost
        kept_idx.append(i)
        if pin:
            seen_pins.add(pin)

    if not kept_idx:
        # Nothing fit at all (budget smaller than the single best chunk) --
        # hard-truncate the top match rather than return nothing.
        best = order[0]
        text, pin = chunks[best]
        head = f"{pin}\n{text}" if pin else text
        return head[:char_budget], {"tier": tier, "chunks_total": len(chunks), "chunks_kept": 1}

    kept_sorted = sorted(kept_idx)
    parts: list[str] = []
    last_pin_shown: str | None = None
    prev_i: int | None = None
    for i in kept_sorted:
        text, pin = chunks[i]
        if prev_i is not None and i != prev_i + 1:
            parts.append("...")
        if pin and pin != last_pin_shown:
            parts.append(pin)
            last_pin_shown = pin
        parts.append(text)
        prev_i = i
    assembled = "\n".join(parts)
    meta = {"tier": tier, "chunks_total": len(chunks), "chunks_kept": len(kept_idx)}
    return assembled, meta
