"""ContextReuseCapability — RRF + MMR + Bayesian scoring retrieval engine."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation._graph import Graph as _Graph
from lemoncrow.core.foundation._graph import pagerank as _pagerank
from lemoncrow.core.foundation.models import Playbook
from lemoncrow.core.foundation.renderer import render_block_for_agent
from lemoncrow.infra.embeddings.factory import get_embedder
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.infra.storage.vector import (
    cosine_similarity,
    get_cached_embedding,
    put_cached_embedding,
    vector_cache_key,
)
from lemoncrow.pro.foundation.retriever import (
    ScoredBlock,
    TaskContext,
    deduplicate_by_playbook,
    pack_by_playbook_token_budget,
    retrieve,
    score_block,
)

from .dead_ends import DeadEndTracker

nx: Any = None  # kept as sentinel; callers guard on `nx is None`


class _EWMean:
    """Exponentially weighted mean — replaces river.stats.EWMean."""

    def __init__(self, fading_factor: float = 0.2) -> None:
        self._mean: float | None = None
        self._alpha = fading_factor

    def update(self, x: float) -> None:
        if self._mean is None:
            self._mean = x
        else:
            self._mean = self._mean * (1.0 - self._alpha) + x * self._alpha

    def get(self) -> float:
        return self._mean if self._mean is not None else 0.5


# HNSW removed (datasketch dropped); brute-force cosine is the permanent fallback.
HNSW: Any = None

# ---------------------------------------------------------------------------
# Token budget for compact injected procedure blocks.
# ---------------------------------------------------------------------------
_DEFAULT_TOKEN_BUDGET = 2000  # max tokens of injected procedures

# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion constant (k=60 per Cormack et al.)
# RRF is scale-invariant — no manual weight tuning required
# ---------------------------------------------------------------------------
_RRF_K = 60

# ---------------------------------------------------------------------------
# Maximal Marginal Relevance diversity weight
# λ=0.7 → 70% relevance, 30% novelty
# ---------------------------------------------------------------------------
_MMR_LAMBDA = 0.7

# ---------------------------------------------------------------------------
# Bayesian prior for success rate (Beta(a, b) with a=b=1 = uniform prior)
# Smoothed rate = (successes + a) / (total + a + b)
# ---------------------------------------------------------------------------
_BAYES_ALPHA = 1.0
_BAYES_BETA = 1.0

# ---------------------------------------------------------------------------
# Rescue boost: blocks matching error signals get a relevance bonus
# ---------------------------------------------------------------------------
_RESCUE_BOOST = 0.25
_GRAPH_BOOST = 0.10
_ANN_BOOST = 0.15
_ADAPTIVE_MIN_MULTIPLIER = 0.90
_ADAPTIVE_MAX_MULTIPLIER = 1.10
_VECTOR_DIM = 128
_TRACE_ENV = "LEMONCROW_RETRIEVAL_TRACE"
_BASE_RETRIEVER_MIN_SCORE = 0.0
# Domain-only matches score ~0.35; a genuine domain+trigger/error match scores
# ~0.43. The threshold sits between so weakly-related blocks are excluded while
# real matches are injected. (0.80 was unreachable and starved all retrieval.)
_MIN_CONTEXT_MATCH_SCORE = 0.40
_MAX_CONTEXT_BLOCKS = 2
_RETRIEVER_VERSION = 2


def _tokenise(text: str) -> list[str]:
    """Tokenise with camelCase splitting for higher recall."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _retrieval_query_text(
    *,
    task: str,
    domain: str | None,
    files: list[str] | None,
    tools: list[str] | None,
    errors: list[str] | None,
) -> str:
    file_names = [Path(path_text).name for path_text in (files or []) if path_text]
    parts = [
        task,
        domain or "",
        " ".join(file_names),
        " ".join(tools or []),
        " ".join(errors or []),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def _bm25_document_text(block: Playbook) -> str:
    return " ".join(
        [
            block.title,
            block.title,
            block.domain,
            " ".join(block.triggers),
            block.situation,
            " ".join(block.failure_signals),
            " ".join(block.dead_ends),
            " ".join(block.procedure),
        ]
    )


def _fts_query_text(
    *,
    task: str,
    errors: list[str] | None,
    domain: str | None = None,
    files: list[str] | None = None,
    tools: list[str] | None = None,
) -> str:
    text = _retrieval_query_text(task=task, domain=domain, files=files, tools=tools, errors=errors)
    parts = _tokenise(text)
    if not parts:
        return ""
    return " ".join(dict.fromkeys(parts))


def _build_idf(docs: list[list[str]]) -> dict[str, float]:
    N = len(docs)
    if N == 0:
        return {}
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    return {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}


def _bm25(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    avg_len: float = 50.0,
) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        idf_val = idf.get(qt, math.log(2.0))
        tf_norm = (tf[qt] * (k1 + 1)) / (tf[qt] + k1 * (1 - b + b * dl / avg_len))
        score += idf_val * tf_norm
    return score


def _recency_score(block: Playbook) -> float:
    try:
        updated = block.updated_at
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        age_days = (datetime.now(UTC) - updated).days
    except (AttributeError, ValueError, TypeError):
        return 0.5
    return max(0.1, math.exp(-age_days / 86.6))


def _bayesian_success(block: Playbook) -> float:
    """Laplace-smoothed success rate: (S + alpha) / (S + F + alpha + beta).

    Avoids 0.0 extremes for untested blocks and 1.0 for single-success blocks.
    """
    return (block.success_count + _BAYES_ALPHA) / (
        block.success_count + block.failure_count + _BAYES_ALPHA + _BAYES_BETA
    )


def _jaccard(a: list[str], b: list[str]) -> float:
    """Token-set Jaccard similarity — O(n) with sets, good enough for MMR."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _hash_vector(tokens: list[str], *, dim: int = _VECTOR_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in tokens:
        # Stable cross-process hash — Python's built-in hash() is salted per
        # process, but these vectors are persisted to disk and compared across
        # later processes, so bucketing must be deterministic.
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    return max(0.0, 1.0 - cosine_similarity(a, b))


_MAX_TRACKED_DOMAINS = 4096


class _AdaptivePriorTracker:
    """Online prior tracker using exponentially weighted mean per domain."""

    def __init__(self) -> None:
        self._by_domain: dict[str, _EWMean] = {}

    def observe(self, domain: str, reward: float) -> None:
        domain_key = domain or "unknown"
        clamped = min(1.0, max(0.0, reward))
        metric = self._by_domain.get(domain_key)
        if metric is None:
            metric = _EWMean(fading_factor=0.2)
            self._by_domain[domain_key] = metric
            while len(self._by_domain) > _MAX_TRACKED_DOMAINS:
                # Bound the tracker; high-cardinality domains would otherwise
                # accumulate one _EWMean each for the life of the process.
                self._by_domain.pop(next(iter(self._by_domain)))
        metric.update(clamped)

    def prior(self, domain: str) -> float:
        domain_key = domain or "unknown"
        metric = self._by_domain.get(domain_key)
        if metric is None:
            return 0.5
        return metric.get()


class ContextReuseCapability:
    """
    Retrieves and ranks relevant past reasoning blocks for injection into
    the current agent context.

    Ranking pipeline:
    1. BM25 rank  +  FTS rank  +  base-retriever rank  → fused via RRF
    2. Rescue boost for blocks whose failure_signals match current errors
    3. Bayesian-smoothed success rate (Beta prior avoids 0/1 extremes)
    4. Recency decay (half-life ~87 days)
    5. MMR diversity filter so injected blocks cover different strategies
    6. Token-budget gate so total injection stays under _DEFAULT_TOKEN_BUDGET

    Signature: __init__(store, root) — matches engine.py constructor call.
    """

    def __init__(self, store: StoreBundle, root: Path) -> None:
        self._store = store
        self._root = Path(root)
        self._dead_ends = DeadEndTracker()
        self._adaptive_priors = _AdaptivePriorTracker()
        self._embedder = get_embedder()
        self._last_retrieval_trace: dict[str, Any] | None = None
        # Savings tracker for finalize() reporting
        self._avoided_failures = 0
        self._avoided_tool_calls = 0
        self._rescue_procedures = 0

    # ------------------------------------------------------------------
    # Internal block collection
    # ------------------------------------------------------------------

    def _domain_blocks(self) -> list[Playbook]:
        from lemoncrow.core.domains import DomainManager

        manager = DomainManager(self._root)
        blocks: list[Playbook] = []
        seen: set[str] = set()
        for block in manager.all_playbooks():
            if block.status in ("quarantined", "deprecated"):
                continue
            if block.id in seen:
                continue
            seen.add(block.id)
            blocks.append(block)
        return blocks

    def _all_active_blocks(self) -> list[Playbook]:
        learned = self._store.knowledge.list_blocks()
        active = [b for b in learned if b.status not in ("quarantined", "deprecated")]
        domain_seen = {b.id for b in active}
        for b in self._domain_blocks():
            if b.id not in domain_seen:
                active.append(b)
                domain_seen.add(b.id)
        return active

    def last_retrieval_trace(self) -> dict[str, Any] | None:
        return self._last_retrieval_trace

    def _query_vector(self, *, task: str, errors: list[str] | None) -> list[float]:
        query_text = (
            task.strip()
            if not errors
            else _retrieval_query_text(
                task=task,
                domain=None,
                files=None,
                tools=None,
                errors=errors,
            )
        )
        if not query_text:
            return []

        try:
            vectors = self._embedder.embed([query_text])
        except Exception:
            logging.exception("Recovered from broad exception handler")
            vectors = [[]]

        if vectors and vectors[0]:
            return vectors[0]
        return _hash_vector(_tokenise(query_text))

    def _block_vectors(self, blocks: list[Playbook]) -> dict[str, list[float]]:
        embedder_name = getattr(self._embedder, "name", self._embedder.__class__.__name__)
        vectors_by_id: dict[str, list[float]] = {}
        uncached: list[tuple[Playbook, str, str]] = []

        for block in blocks:
            rendered = render_block_for_agent(block)
            cache_key = vector_cache_key(block.id, rendered)
            cached = get_cached_embedding(self._root, cache_key=cache_key, embedder_name=embedder_name)
            if cached is not None:
                vectors_by_id[block.id] = cached
                continue
            uncached.append((block, rendered, cache_key))

        embedded: list[list[float]] = []
        if uncached:
            try:
                embedded = self._embedder.embed([rendered for _block, rendered, _cache_key in uncached])
            except Exception:
                logging.exception("Recovered from broad exception handler")
                embedded = [[] for _ in uncached]

        for idx, (block, rendered, cache_key) in enumerate(uncached):
            vector = embedded[idx] if idx < len(embedded) else []
            if not vector:
                vector = _hash_vector(_tokenise(rendered))
            vectors_by_id[block.id] = vector
            put_cached_embedding(self._root, cache_key=cache_key, embedder_name=embedder_name, vector=vector)

        return vectors_by_id

    # ------------------------------------------------------------------
    # Primary ranking: Reciprocal Rank Fusion + rescue boost + Bayesian score
    # ------------------------------------------------------------------

    def rank_reusable_procedures(
        self,
        *,
        task: str,
        domain: str | None = None,
        files: list[str] | None = None,
        tools: list[str] | None = None,
        errors: list[str] | None = None,
        limit: int = 5,
        token_budget: int | None = _DEFAULT_TOKEN_BUDGET,
        dedup: bool = True,
        monitor_composite: float = 0.0,
        fsm_skip_etraces: bool = False,
    ) -> list[Any]:
        """
        Rank blocks using Reciprocal Rank Fusion of BM25 + FTS + base retriever.

        Steps:
        1. Score each block with BM25, FTS, and base retriever independently
        2. Fuse ranks via RRF (scale-invariant, no manual weight tuning)
        3. Apply rescue boost for blocks matching current errors
        4. Multiply by Bayesian-smoothed success x recency
        5. Apply MMR to select diverse top-k (avoids injecting redundant blocks)
        6. Enforce token budget gate
        """
        ctx = TaskContext(
            task=task,
            domain=domain,
            files=files or [],
            tools=tools or [],
            errors=errors or [],
        )
        trace_enabled = os.environ.get(_TRACE_ENV, "").lower() in ("1", "true", "yes")
        self._last_retrieval_trace = None

        all_blocks = self._all_active_blocks()
        if not all_blocks:
            return []

        query_text = _retrieval_query_text(
            task=task,
            domain=domain,
            files=files,
            tools=tools,
            errors=errors,
        )
        query_vector = self._query_vector(task=task, errors=errors)
        block_vectors = self._block_vectors(all_blocks)
        vector_scores = {
            block_id: cosine_similarity(query_vector, vector)
            for block_id, vector in block_vectors.items()
            if query_vector and vector and len(vector) == len(query_vector)
        }

        trace_blocks: dict[str, Playbook] = {}
        trace_reasons: dict[str, set[str]] = {}
        base_probe_scores: dict[str, float] = {}
        bm25_score_lookup: dict[str, float] = {}
        base_probe_candidate_ids: set[str] = set()

        if trace_enabled:
            for block in self._store.knowledge.list_blocks():
                trace_blocks.setdefault(block.id, block)
            for block in self._domain_blocks():
                trace_blocks.setdefault(block.id, block)

            for block_id, block in trace_blocks.items():
                trace_reasons[block_id] = set()
                if block.status == "quarantined":
                    trace_reasons[block_id].add("quarantined")

            base_query = query_text
            base_candidates: list[Playbook] = []
            if base_query:
                base_candidates.extend(self._store.knowledge.search_blocks(base_query, limit=50))
            if ctx.domain:
                base_candidates.extend(self._store.knowledge.list_blocks(domain=ctx.domain))

            seen_probe: set[str] = set()
            unique_probe: list[Playbook] = []
            for block in base_candidates:
                if block.id in seen_probe:
                    continue
                if block.status in ("quarantined", "deprecated"):
                    continue
                seen_probe.add(block.id)
                unique_probe.append(block)

            base_probe = [score_block(block, ctx) for block in unique_probe]
            base_probe_candidate_ids = {entry.block.id for entry in base_probe}
            base_probe_scores = {entry.block.id: entry.score for entry in base_probe}
            for block_id, score in base_probe_scores.items():
                if score < _BASE_RETRIEVER_MIN_SCORE:
                    trace_reasons.setdefault(block_id, set()).add("sub_min_score")

        error_text = " ".join(errors or [])
        query_tokens = _tokenise(query_text)

        # BM25 scoring
        doc_tokens_map = {b.id: _tokenise(_bm25_document_text(b)) for b in all_blocks}
        avg_len = sum(len(v) for v in doc_tokens_map.values()) / max(1, len(doc_tokens_map))
        idf = _build_idf(list(doc_tokens_map.values()))

        bm25_scored = sorted(
            [(b.id, _bm25(query_tokens, doc_tokens_map[b.id], idf, avg_len=avg_len)) for b in all_blocks],
            key=lambda x: x[1],
            reverse=True,
        )
        bm25_score_lookup = {bid: score for bid, score in bm25_scored}
        bm25_rank_trace: dict[str, int] = {bid: rank for rank, (bid, _) in enumerate(bm25_scored, start=1)}

        # FTS scoring (reciprocal rank from store)
        fts_blocks = self._store.knowledge.search_blocks(
            _fts_query_text(task=task, errors=errors, domain=domain, files=files, tools=tools),
            limit=max(limit * 5, 30),
        )
        fts_rank: dict[str, int] = {b.id: rank for rank, b in enumerate(fts_blocks)}
        fts_rank_trace: dict[str, int] = {b.id: rank for rank, b in enumerate(fts_blocks, start=1)}

        # Base retriever scoring (domain / trigger matching)
        learned = retrieve(
            self._store,
            ctx,
            limit=max(limit * 3, 20),
            min_score=_BASE_RETRIEVER_MIN_SCORE,
            vector_scores=vector_scores,
            use_vector_weights=True,
            dedup=dedup,
            token_budget=token_budget,
            monitor_composite=monitor_composite,
            fsm_skip_etraces=fsm_skip_etraces,
        )
        base_scores: dict[str, float] = {item.block.id: item.score for item in learned}
        direct_match_scores = {block.id: score_block(block, ctx).score for block in all_blocks}
        base_ranked = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        base_rank: dict[str, int] = {bid: rank for rank, (bid, _) in enumerate(base_ranked)}
        base_rank_trace: dict[str, int] = {bid: rank for rank, (bid, _) in enumerate(base_ranked, start=1)}
        # RRF fusion — merge three rank lists
        block_map: dict[str, Playbook] = {b.id: b for b in all_blocks}
        rrf_scores: dict[str, float] = {}
        for rank, (bid, _) in enumerate(bm25_scored):
            rrf_scores[bid] = rrf_scores.get(bid, 0.0) + 1.0 / (_RRF_K + rank)
        for bid, rank in fts_rank.items():
            if bid in block_map:
                rrf_scores[bid] = rrf_scores.get(bid, 0.0) + 1.0 / (_RRF_K + rank)
        for bid, rank in base_rank.items():
            if bid in block_map:
                rrf_scores[bid] = rrf_scores.get(bid, 0.0) + 1.0 / (_RRF_K + rank)

        error_text_lower = error_text.lower()

        results: list[_HybridResult] = []
        for bid, rrf in rrf_scores.items():
            matched_block = block_map.get(bid)
            if matched_block is None:
                continue

            is_rescue = bool(
                errors
                and matched_block.failure_signals
                and any(fs.lower() in error_text_lower for fs in matched_block.failure_signals)
            )

            # Combine: RRF (rank quality) x recency x Bayesian success
            # + rescue boost when error signals match
            quality = 0.8 + (0.4 * _bayesian_success(matched_block) * _recency_score(matched_block))
            final = rrf * quality
            if is_rescue:
                final = min(final + _RESCUE_BOOST, 1.0)

            results.append(
                _HybridResult(
                    block=matched_block,
                    fts_score=1.0 / (_RRF_K + fts_rank.get(bid, len(all_blocks))),
                    bm25_score=bm25_score_lookup.get(bid, 0.0),
                    recency_score=_recency_score(matched_block),
                    success_score=_bayesian_success(matched_block),
                    final_score=final,
                    match_score=direct_match_scores.get(bid, 0.0),
                    rescue=is_rescue,
                    dead_ends=list(matched_block.dead_ends),
                )
            )
        # Sort by final score before MMR selection
        results.sort(key=lambda r: r.final_score, reverse=True)

        self._apply_adaptive_priors(results)
        self._apply_graph_propagation(results)
        self._apply_ann_reranking(results, query_vector=query_vector, block_vectors=block_vectors)
        results.sort(key=lambda r: r.final_score, reverse=True)
        results = [r for r in results if r.match_score > _MIN_CONTEXT_MATCH_SCORE]
        effective_limit = min(limit, _MAX_CONTEXT_BLOCKS)

        if dedup:
            results = deduplicate_by_playbook(results, lambda item: item.block)

        # MMR diversity selection — covers different procedures after exact near-dup filtering.
        selected: list[_HybridResult] = []
        candidates = list(results)
        selection_pool_limit = len(results) if token_budget is not None else effective_limit
        while candidates and len(selected) < selection_pool_limit:
            if not selected:
                best = candidates.pop(0)
            else:
                # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                selected_tokens = [doc_tokens_map.get(s.block.id, []) for s in selected]
                best_score = -1.0
                best_idx = 0
                for idx, cand in enumerate(candidates):
                    cand_tokens = doc_tokens_map.get(cand.block.id, [])
                    max_sim = max(
                        (_jaccard(cand_tokens, st) for st in selected_tokens),
                        default=0.0,
                    )
                    mmr = _MMR_LAMBDA * cand.final_score - (1.0 - _MMR_LAMBDA) * max_sim
                    if mmr > best_score:
                        best_score = mmr
                        best_idx = idx
                best = candidates.pop(best_idx)

            selected.append(best)

        mmr_selected = list(selected)
        selected = pack_by_playbook_token_budget(
            selected,
            lambda item: item.block,
            limit=effective_limit,
            token_budget=token_budget,
        )

        if trace_enabled:
            mmr_selected_ids = {item.block.id for item in mmr_selected}
            final_ids = [item.block.id for item in selected]
            final_rank = {block_id: rank for rank, block_id in enumerate(final_ids, start=1)}
            result_rank = {item.block.id: rank for rank, item in enumerate(results, start=1)}
            for item in results:
                if item.block.id not in mmr_selected_ids:
                    trace_reasons.setdefault(item.block.id, set()).add("mmr_suppressed")
            for item in mmr_selected:
                if item.block.id not in final_rank:
                    trace_reasons.setdefault(item.block.id, set()).add("token_budget_evicted")

            if ctx.domain:
                domain_prefix = ctx.domain.split(".")[0]
                for block_id, block in trace_blocks.items():
                    if block.status == "quarantined":
                        continue
                    if block_id in fts_rank or block_id in base_rank:
                        continue
                    if block.domain == ctx.domain or block.domain.startswith(domain_prefix):
                        continue
                    trace_reasons.setdefault(block_id, set()).add("wrong_domain")

            trace_candidates: list[dict[str, Any]] = []
            for block_id, block in trace_blocks.items():
                bm25_rank_value = bm25_rank_trace.get(block_id)
                fts_rank_value = fts_rank_trace.get(block_id)
                base_rank_value = base_rank_trace.get(block_id)
                trace_candidates.append(
                    {
                        "block_id": block_id,
                        "domain": block.domain,
                        "status": block.status,
                        "final_rank": final_rank.get(block_id),
                        "result_rank": result_rank.get(block_id),
                        "bm25_rank": bm25_rank_value,
                        "fts_rank": fts_rank_value,
                        "base_rank": base_rank_value,
                        "bm25_score": round(float(bm25_score_lookup.get(block_id, 0.0)), 6),
                        "base_probe_score": round(float(base_probe_scores.get(block_id, 0.0)), 6),
                        "match_score": round(float(direct_match_scores.get(block_id, 0.0)), 6),
                        "rrf": round(float(rrf_scores.get(block_id, 0.0)), 6),
                        "rrf_contributions": {
                            "bm25": round(
                                0.0 if bm25_rank_value is None else 1.0 / (_RRF_K + bm25_rank_value - 1),
                                6,
                            ),
                            "fts": round(
                                0.0 if fts_rank_value is None else 1.0 / (_RRF_K + fts_rank_value - 1),
                                6,
                            ),
                            "base": round(
                                0.0 if base_rank_value is None else 1.0 / (_RRF_K + base_rank_value - 1),
                                6,
                            ),
                        },
                        "drop_reasons": sorted(trace_reasons.get(block_id, set())),
                    }
                )

            trace_candidates.sort(
                key=lambda item: (
                    item["final_rank"] is None,
                    item["final_rank"] or item["result_rank"] or 10_000,
                    -(item["rrf"]),
                    item["block_id"],
                )
            )

            self._last_retrieval_trace = {
                "retriever_version": _RETRIEVER_VERSION,
                "query": {
                    "task": task,
                    "domain": domain,
                    "files": list(files or []),
                    "tools": list(tools or []),
                    "errors": list(errors or []),
                },
                "candidate_count": len(all_blocks),
                "base_probe_candidate_count": len(base_probe_candidate_ids),
                "final_block_ids": final_ids,
                "candidates": trace_candidates,
            }

        for picked in selected:
            self._adaptive_priors.observe(picked.block.domain, picked.success_score)

        return selected

    def _apply_adaptive_priors(self, results: list[Any]) -> None:
        if not results:
            return
        for item in results:
            prior = self._adaptive_priors.prior(item.block.domain)
            item.adaptive_prior = prior
            multiplier = _ADAPTIVE_MIN_MULTIPLIER + ((_ADAPTIVE_MAX_MULTIPLIER - _ADAPTIVE_MIN_MULTIPLIER) * prior)
            item.final_score *= multiplier

    def _apply_graph_propagation(self, results: list[Any]) -> None:
        if not results:
            return
        graph = _Graph()
        for item in results:
            graph.add_node(item.block.id)
        for i, left in enumerate(results):
            for right in results[i + 1 :]:
                shared = len(set(left.block.triggers) & set(right.block.triggers))
                same_domain = 1 if left.block.domain == right.block.domain else 0
                shared_dead_ends = len(set(left.block.dead_ends) & set(right.block.dead_ends))
                weight = (0.2 * shared) + (0.3 * same_domain) + (0.1 * shared_dead_ends)
                if weight > 0:
                    graph.add_edge(left.block.id, right.block.id, weight=weight)

        if graph.number_of_edges() == 0:
            return

        rescue_nodes = [r.block.id for r in results if r.rescue]
        if rescue_nodes:
            base = 1.0 / len(rescue_nodes)
            personalization = {node: (base if node in rescue_nodes else 0.001) for node in graph.nodes}
        else:
            personalization = {node: 1.0 / max(graph.number_of_nodes(), 1) for node in graph.nodes}

        scores = _pagerank(graph, alpha=0.85, personalization=personalization, weight="weight")
        max_score = max(scores.values(), default=1.0) or 1.0
        for item in results:
            norm = scores.get(item.block.id, 0.0) / max_score
            item.graph_score = norm
            item.final_score += _GRAPH_BOOST * norm

    def _apply_ann_reranking(
        self,
        results: list[Any],
        *,
        query_vector: list[float],
        block_vectors: dict[str, list[float]],
    ) -> None:
        if not results or not query_vector:
            return
        candidate_count = min(len(results), 80)
        candidates = results[:candidate_count]

        if HNSW is not None:
            index = HNSW(distance_func=_cosine_distance)
            vectors_by_id: dict[str, list[float]] = {}
            for item in candidates:
                item.ann_score = 0.0
                vector = block_vectors.get(item.block.id)
                if not vector:
                    continue
                vectors_by_id[item.block.id] = vector
                index.insert(item.block.id, vectors_by_id[item.block.id])

            if not vectors_by_id:
                return

            nearest = index.query(query_vector, k=min(len(vectors_by_id), 25))
            max_sim = 0.0
            sim_map: dict[str, float] = {}
            for candidate_id, dist in nearest:
                sim = max(0.0, 1.0 - float(dist))
                sim_map[str(candidate_id)] = sim
                max_sim = max(max_sim, sim)

            denom = max_sim or 1.0
            for item in candidates:
                sim = sim_map.get(item.block.id, 0.0)
                ann_norm = sim / denom
                item.ann_score = ann_norm
                item.final_score += _ANN_BOOST * ann_norm
            return

        scored: list[tuple[float, Any]] = []
        for item in candidates:
            vec = block_vectors.get(item.block.id)
            # Mirror the vector_scores guard: a hash-fallback block vector
            # (_VECTOR_DIM) against a real query embedding (768/1536) would raise
            # ValueError("Vector dimension mismatch") in cosine_similarity.
            if not vec or len(vec) != len(query_vector):
                continue
            sim = cosine_similarity(query_vector, vec)
            scored.append((sim, item))
        max_sim = max((s for s, _ in scored), default=1.0) or 1.0
        for sim, item in scored:
            ann_norm = sim / max_sim
            item.ann_score = ann_norm
            item.final_score += _ANN_BOOST * ann_norm

    # ------------------------------------------------------------------
    # Engine API: retrieve → list[ScoredBlock]
    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        task: str,
        domain: str | None = None,
        files: list[str] | None = None,
        tools: list[str] | None = None,
        errors: list[str] | None = None,
        limit: int = 5,
        token_budget: int | None = _DEFAULT_TOKEN_BUDGET,
        dedup: bool = True,
        monitor_composite: float = 0.0,
        fsm_skip_etraces: bool = False,
    ) -> list[ScoredBlock]:
        """Return ScoredBlock list for engine.get_context."""
        ranked = self.rank_reusable_procedures(
            task=task,
            domain=domain,
            files=files,
            tools=tools,
            errors=errors,
            limit=limit,
            token_budget=token_budget,
            dedup=dedup,
            monitor_composite=monitor_composite,
            fsm_skip_etraces=fsm_skip_etraces,
        )
        return [
            ScoredBlock(
                block=r.block,
                score=r.match_score,
                breakdown={
                    "fts": r.fts_score,
                    "bm25": r.bm25_score,
                    "recency": r.recency_score,
                    "success": r.success_score,
                    "adaptive": r.adaptive_prior,
                    "graph": r.graph_score,
                    "ann": r.ann_score,
                    "rank": r.final_score,
                },
            )
            for r in ranked
        ]

    def inject_runtime_context(
        self,
        *,
        task: str,
        domain: str | None = None,
        files: list[str] | None = None,
        tools: list[str] | None = None,
        errors: list[str] | None = None,
        max_blocks: int = 5,
        token_budget: int | None = _DEFAULT_TOKEN_BUDGET,
        dedup: bool = True,
    ) -> dict[str, Any]:
        """Return structured injection payload for engine.inject_context."""
        ranked = self.rank_reusable_procedures(
            task=task,
            domain=domain,
            files=files,
            tools=tools,
            errors=errors,
            limit=max_blocks,
            token_budget=token_budget,
            dedup=dedup,
        )

        procedures: list[dict[str, Any]] = []
        dead_ends: list[str] = []
        rescue_strategies: list[str] = []
        required_validations: list[str] = []

        for r in ranked:
            proc: dict[str, Any] = {
                "id": r.block.id,
                "title": r.block.title,
                "domain": r.block.domain,
                "score": round(r.match_score, 3),
                "rescue": r.rescue,
            }
            if r.block.procedure:
                proc["procedure"] = r.block.procedure
            procedures.append(proc)
            dead_ends.extend(r.dead_ends)
            if r.rescue and r.block.procedure:
                rescue_strategies.extend(r.block.procedure)
            if r.block.required_rubrics:
                required_validations.extend(r.block.required_rubrics)

        # Update savings counters
        rescue_count = sum(1 for r in ranked if r.rescue)
        self._avoided_failures += rescue_count
        self._avoided_tool_calls += max(0, len(ranked) - 1)
        self._rescue_procedures += len(rescue_strategies)

        return {
            "procedures": procedures,
            "dead_ends": sorted(set(dead_ends)),
            "rescue_strategies": rescue_strategies[:3],
            "rescue_chains": self._rescue_chains(ranked),
            "required_validations": sorted(set(required_validations)),
            "savings": {
                "avoided_failures": self._avoided_failures,
                "avoided_tool_calls": self._avoided_tool_calls,
                "rescue_procedures": self._rescue_procedures,
            },
        }

    def _rescue_chains(self, ranked: list[Any]) -> list[dict[str, Any]]:
        if nx is None or not ranked:
            return []
        rescue_nodes = [r.block.id for r in ranked if r.rescue]
        if not rescue_nodes:
            return []
        graph = nx.Graph()
        id_to_title = {r.block.id: r.block.title for r in ranked}
        for r in ranked:
            graph.add_node(r.block.id)
        for i, left in enumerate(ranked):
            for right in ranked[i + 1 :]:
                shared = len(set(left.block.triggers) & set(right.block.triggers))
                if shared > 0 or left.block.domain == right.block.domain:
                    graph.add_edge(left.block.id, right.block.id)

        chains: list[dict[str, Any]] = []
        for node in rescue_nodes:
            neighbors = list(graph.neighbors(node))[:3]
            chains.append(
                {
                    "root_id": node,
                    "root_title": id_to_title.get(node, node),
                    "neighbors": [id_to_title.get(n, n) for n in neighbors],
                }
            )
        return chains

    def rescue_candidates(self, *, task: str, error: str, limit: int = 3) -> list[ScoredBlock]:
        """Rank active blocks by raw BM25 evidence for an explicit rescue call.

        Deliberately bypasses the RRF/ANN/MMR injection pipeline: rank-based
        fusion is flat on a small block set and hash-vector cosine is noisy,
        while raw bm25 cleanly separates true matches (13-22 on the seed
        blocks) from unrelated errors (<=4.3). The caller applies the
        evidence floor and falls back honestly below it.
        """
        blocks = self._all_active_blocks()
        if not blocks:
            return []
        query_tokens = _tokenise(f"{task} {error}")
        doc_tokens_map = {b.id: _tokenise(_bm25_document_text(b)) for b in blocks}
        avg_len = sum(len(v) for v in doc_tokens_map.values()) / max(1, len(doc_tokens_map))
        idf = _build_idf(list(doc_tokens_map.values()))
        scored = [
            ScoredBlock(
                block=b,
                score=_bm25(query_tokens, doc_tokens_map[b.id], idf, avg_len=avg_len),
                breakdown={},
            )
            for b in blocks
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def savings_estimate(self) -> dict[str, int]:
        return {
            "avoided_failures": self._avoided_failures,
            "avoided_tool_calls": self._avoided_tool_calls,
            "rescue_procedures": self._rescue_procedures,
        }

    # ------------------------------------------------------------------
    # Dead-end management
    # ------------------------------------------------------------------

    def mark_dead_end(self, approach: str) -> None:
        self._dead_ends.mark_dead_end(approach)

    def is_dead_end(self, approach: str) -> bool:
        return self._dead_ends.is_dead_end(approach)


# ---------------------------------------------------------------------------
# Internal result type
# ---------------------------------------------------------------------------


class _HybridResult:
    """Internal result type for rank_reusable_procedures."""

    __slots__ = (
        "adaptive_prior",
        "ann_score",
        "block",
        "bm25_score",
        "dead_ends",
        "final_score",
        "fts_score",
        "graph_score",
        "match_score",
        "recency_score",
        "rescue",
        "success_score",
    )

    def __init__(
        self,
        *,
        block: Playbook,
        fts_score: float,
        bm25_score: float,
        recency_score: float,
        success_score: float,
        final_score: float,
        match_score: float,
        rescue: bool,
        dead_ends: list[str],
    ) -> None:
        self.block = block
        self.fts_score = fts_score
        self.bm25_score = bm25_score
        self.recency_score = recency_score
        self.success_score = success_score
        self.final_score = final_score
        self.match_score = match_score
        self.rescue = rescue
        self.dead_ends = dead_ends
        self.adaptive_prior = 0.5
        self.graph_score = 0.0
        self.ann_score = 0.0
