"""Hybrid ranking engine: BM25 + recency + success-rate + dead-end avoidance."""

from __future__ import annotations

import math
import re
from typing import Any

from lemoncrow.core.foundation._minhash import MinHash
from lemoncrow.pro.capabilities.prompt_compilation.tokens import count_tokens

from .bm25 import bm25_score, build_idf, tokenise
from .dead_ends import DeadEndTracker
from .models import RankedProcedure

# Ranking weights (sum should be ~1.0 for interpretability)
_W_BM25 = 0.45
_W_RECENCY = 0.25
_W_SUCCESS = 0.20
_W_BASE = 0.10
_DEFAULT_TOKEN_BUDGET = 2000
_DEDUP_THRESHOLD = 0.75
_MINHASH_PERMUTATIONS = 128
_MIN_DEDUP_TOKENS = 5


def _recency_score(last_used_ts: float, now_ts: float, *, half_life_days: float = 7.0) -> float:
    """Exponential decay: score=1.0 if used now, decays to ~0.5 after half_life_days."""
    if last_used_ts <= 0:
        return 0.1
    elapsed_days = (now_ts - last_used_ts) / 86_400
    return math.exp(-elapsed_days * math.log(2) / half_life_days)


def _success_score(success_rate: float, reuse_count: int) -> float:
    """Bayesian smoothed success rate (adds prior of 0.5 with weight 2)."""
    numerator = success_rate * reuse_count + 0.5 * 2
    denominator = reuse_count + 2
    return numerator / denominator


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def _signature_tokens(block: dict[str, Any]) -> set[str]:
    parts = [*_as_list(block.get("dead_ends")), *_as_list(block.get("procedure"))]
    return set(re.findall(r"[a-z0-9]+", " ".join(parts).lower()))


def _minhash(tokens: set[str]) -> MinHash:
    signature = MinHash(num_perm=_MINHASH_PERMUTATIONS)
    for token in sorted(tokens):
        signature.update(token.encode("utf-8"))
    return signature


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _dedupe_ranked(
    ranked: list[RankedProcedure],
    blocks_by_id: dict[str, dict[str, Any]],
    *,
    threshold: float = _DEDUP_THRESHOLD,
) -> list[RankedProcedure]:
    if len(ranked) < 2:
        return ranked
    kept: list[RankedProcedure] = []
    kept_sigs: list[MinHash] = []
    for item in ranked:
        block = blocks_by_id.get(item.block_id, {})
        tokens = _signature_tokens(block)
        if len(tokens) < _MIN_DEDUP_TOKENS:
            kept.append(item)
            continue
        sig = _minhash(tokens)
        if any(sig.jaccard(prior) >= threshold for prior in kept_sigs):
            continue
        kept_sigs.append(sig)
        kept.append(item)
    return kept


def _pack_ranked(
    ranked: list[RankedProcedure],
    *,
    limit: int,
    token_budget: int | None,
) -> list[RankedProcedure]:
    packed: list[RankedProcedure] = []
    tokens_used = 0
    for item in ranked:
        if len(packed) >= limit:
            break
        item_tokens = count_tokens(f"{item.title}\n{item.snippet}")
        if token_budget is not None and token_budget >= 0:
            if tokens_used + item_tokens > token_budget and packed:
                continue
            if token_budget == 0 and not packed:
                break
        packed.append(item)
        tokens_used += item_tokens
    return packed


def rank_blocks(
    query: str,
    blocks: list[dict[str, Any]],
    *,
    dead_end_tracker: DeadEndTracker | None = None,
    now_ts: float = 0.0,
    domain_filter: str = "",
    limit: int = 5,
    dedup: bool = True,
    token_budget: int | None = _DEFAULT_TOKEN_BUDGET,
) -> list[RankedProcedure]:
    """
    Rank procedure blocks against a query using a hybrid scoring model.

    Args:
        query:            The current task description or goal.
        blocks:           List of block dicts from the reasoning store.
        dead_end_tracker: Optional tracker for penalising dead-end approaches.
        now_ts:           Current Unix timestamp (0 = disabled recency scoring).
        domain_filter:    If set, prefer blocks from this domain.
        limit:            Maximum results to return.
        dedup:            Drop near-duplicate procedure blocks by MinHash LSH.
        token_budget:     Greedy token budget for ranked snippets; None disables it.

    Returns:
        List of :class:`RankedProcedure` sorted by score descending.
    """
    if not blocks:
        return []

    import time as _time

    if now_ts <= 0:
        now_ts = _time.time()

    # Build BM25 corpus
    query_tokens = tokenise(query)
    doc_texts: list[str] = []
    for block in blocks:
        parts = [
            str(block.get("title", "")),
            str(block.get("description", "")),
            str(block.get("tags", "")),
            str(block.get("domain", "")),
        ]
        doc_texts.append(" ".join(parts))

    corpus = [tokenise(t) for t in doc_texts]
    idf = build_idf(corpus)
    avg_dl = sum(len(d) for d in corpus) / max(len(corpus), 1)

    ranked: list[RankedProcedure] = []
    for block, doc_tokens in zip(blocks, corpus, strict=False):
        # BM25 score (normalised to [0, 1] via sigmoid)
        raw_bm25 = bm25_score(query_tokens, doc_tokens, idf, avg_dl)
        bm25_norm = raw_bm25 / (raw_bm25 + 3.0)  # soft normalisation

        # Recency score
        last_used = float(block.get("last_used_ts", 0) or block.get("last_used", 0) or 0)
        recency = _recency_score(last_used, now_ts) if last_used > 0 else 0.1

        # Success rate
        success_rate = float(block.get("success_rate", 0.5))
        reuse_count = int(block.get("reuse_count", 0))
        success = _success_score(success_rate, reuse_count)

        # Base score (normalised)
        base = float(block.get("base_score", 0.5))

        composite = _W_BM25 * bm25_norm + _W_RECENCY * recency + _W_SUCCESS * success + _W_BASE * base

        # Domain alignment bonus
        if domain_filter and str(block.get("domain", "")) == domain_filter:
            composite *= 1.15

        # Dead-end penalty
        is_dead_end = False
        if dead_end_tracker:
            title = str(block.get("title", ""))
            desc = str(block.get("description", ""))
            is_dead_end = dead_end_tracker.is_dead_end(title) or dead_end_tracker.is_dead_end(desc)
            if is_dead_end:
                composite *= 0.2

        snippet_text = str(block.get("description", ""))[:300]

        ranked.append(
            RankedProcedure(
                block_id=str(block.get("id", block.get("block_id", ""))),
                title=str(block.get("title", "")),
                domain=str(block.get("domain", "")),
                score=composite,
                base_score=base,
                recency_score=recency,
                success_rate=success_rate,
                reuse_count=reuse_count,
                snippet=snippet_text,
                is_dead_end=is_dead_end,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    blocks_by_id = {str(block.get("id", block.get("block_id", ""))): block for block in blocks}
    if dedup:
        ranked = _dedupe_ranked(ranked, blocks_by_id)
    return _pack_ranked(ranked, limit=limit, token_budget=token_budget)
