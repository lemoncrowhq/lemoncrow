"""Pull algorithm for scoped pull-context (M4).

Deterministic and side-effect-free (except the trace id). Seeds candidates
from multiple retrieval channels, drops excluded/dead-end candidates, ranks
with a keyword/affected-path boost, and packs within the subtask budget.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from lemoncrow.pro.capabilities.code_context.budget import BudgetPacker
from lemoncrow.pro.capabilities.context_reuse.dead_ends import DeadEndTracker
from lemoncrow.pro.capabilities.repo_map.budget import count_tokens

from .line_skimmer import build_goal_text, is_line_skim_enabled, skim_chunks
from .models import ContextChunk, ScopedContext
from .prune import apply_exclusions

_ESSENTIAL_KEYS = ("path", "symbol", "kind", "provenance", "commit_sha")
_OPTIONAL_DROP_ORDER = ("snippet", "signature", "qualified_name")
_HISTORY_HINT_TOKENS = frozenset(
    {
        "commit",
        "history",
        "historical",
        "prior",
        "previous",
        "regression",
        "introduced",
        "origin",
        "root",
        "cause",
        "why",
        "when",
        "revert",
    }
)
_HISTORY_HINT_PHRASES = (
    "root cause",
    "which commit",
    "when did",
    "why did",
    "introduced by",
    "history of",
)


def _candidate(rec: Any, channel: str, position: int) -> dict[str, Any]:
    score = 1.0 / (1 + position)
    return {
        "path": str(getattr(rec, "file_path", "") or ""),
        "symbol": str(getattr(rec, "symbol_name", "") or ""),
        "qualified_name": str(getattr(rec, "qualified_name", "") or ""),
        "kind": str(getattr(rec, "kind", "") or ""),
        "language": str(getattr(rec, "language", "") or ""),
        "signature": str(getattr(rec, "signature", "") or ""),
        "snippet": str(getattr(rec, "snippet", "") or ""),
        "channel": channel,
        "score": score,
        "provenance": str(getattr(rec, "provenance", "") or ""),
        "commit_sha": str(getattr(rec, "commit_sha", "") or ""),
    }


def _tokens(value: str) -> set[str]:
    if not value:
        return set()
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return {token.lower() for token in re.findall(r"[A-Za-z0-9]+", normalized) if len(token) >= 2}


def _boost(cand: dict[str, Any], query_tokens: set[str], affected: list[str]) -> float:
    score = float(cand.get("score", 0.0) or 0.0)
    path_tokens = _tokens(str(cand.get("path", "")))
    text_tokens = _tokens(f"{cand.get('symbol', '')} {cand.get('qualified_name', '')} {cand.get('signature', '')}")
    path_overlap = len(query_tokens & path_tokens)
    text_overlap = len(query_tokens & text_tokens)
    if path_overlap:
        score += min(1.2, 0.4 * path_overlap)
    if text_overlap:
        score += min(0.45, 0.15 * text_overlap)
    if affected and cand.get("path") in affected:
        score += 0.5
    return score


def _history_intent(subtask: Any) -> bool:
    text = " ".join([subtask.description, *list(subtask.keywords)]).lower()
    if any(phrase in text for phrase in _HISTORY_HINT_PHRASES):
        return True
    return bool(_tokens(text) & _HISTORY_HINT_TOKENS)


def _seed(subtask: Any, engine: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # Channel A: hybrid retrieval on the natural-language description.
    for i, rec in enumerate(
        engine.search_symbols(
            subtask.description,
            limit=50,
            mode="auto",
            snippet="head",
            provenance_filter="local",
        )
    ):
        candidates.append(_candidate(rec, "description", i))
    # Channel B: lexical retrieval on explicit keywords.
    if subtask.keywords:
        kw_query = " ".join(k for k in subtask.keywords if k)
        if kw_query:
            for i, rec in enumerate(
                engine.search_symbols(
                    kw_query,
                    limit=20,
                    mode="lexical",
                    snippet="head",
                    provenance_filter="local",
                )
            ):
                candidates.append(_candidate(rec, "keyword", i))
    # Channel C: in-file symbols for explicitly affected paths.
    for path in subtask.affected_paths:
        for i, rec in enumerate(
            engine.search_symbols(
                subtask.description,
                limit=10,
                mode="lexical",
                snippet="head",
                file_glob=path,
                provenance_filter="local",
            )
        ):
            candidates.append(_candidate(rec, "affected_path", i))
    if _history_intent(subtask):
        for i, rec in enumerate(
            engine.search_symbols(
                subtask.description,
                limit=12,
                mode="auto",
                snippet="head",
                provenance_filter="commit",
            )
        ):
            candidates.append(_candidate(rec, "history", i))
        if subtask.keywords:
            kw_query = " ".join(k for k in subtask.keywords if k)
            if kw_query:
                for i, rec in enumerate(
                    engine.search_symbols(
                        kw_query,
                        limit=8,
                        mode="lexical",
                        snippet="head",
                        provenance_filter="commit",
                    )
                ):
                    candidates.append(_candidate(rec, "history_keyword", i))
    return candidates


def _dedup(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in candidates:
        key = (cand.get("path", ""), cand.get("symbol", ""))
        prior = best.get(key)
        if prior is None or float(cand.get("score", 0.0)) > float(prior.get("score", 0.0)):
            best[key] = cand
    return list(best.values())


def pull(
    subtask: Any,
    *,
    engine: Any,
    dead_ends: DeadEndTracker,
    packer: BudgetPacker,
) -> ScopedContext:
    """Run the scoped pull and return a packed, debuggable :class:`ScopedContext`."""
    trace_id = uuid.uuid4().hex
    seeded = _dedup(_seed(subtask, engine))
    query_tokens = _tokens(" ".join([subtask.description, *list(subtask.keywords)]))
    kept, excluded = apply_exclusions(seeded, excluded_paths=list(subtask.excluded_paths), dead_ends=dead_ends)
    for cand in kept:
        cand["score"] = round(_boost(cand, query_tokens, list(subtask.affected_paths)), 6)
    kept.sort(key=lambda c: float(c.get("score", 0.0)), reverse=True)

    packed, dropped, tokens = packer.pack(
        kept,
        subtask.budget_tokens,
        essential_keys=_ESSENTIAL_KEYS,
        optional_keys_in_drop_order=_OPTIONAL_DROP_ORDER,
    )
    chunks = [ContextChunk.from_dict(item) for item in packed]

    # T12: goal-conditioned per-line skim (default-off via LEMONCROW_LINE_SKIM).
    # Runs after candidate chunks are assembled; prunes each chunk body to its
    # goal-relevant lines + anchors, then recomputes the reported token count.
    # With the flag off this block is a no-op and chunk output is unchanged.
    skim_note = ""
    if chunks and is_line_skim_enabled():
        goal = build_goal_text(subtask)
        if goal:
            before = tokens
            chunks = skim_chunks(chunks, goal=goal, ranker=getattr(engine, "_semantic_ranker", None))
            tokens = count_tokens(
                json.dumps(
                    [c.to_dict() for c in chunks],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                )
            )
            if tokens < before:
                skim_note = f" line-skim: {before}->{tokens} tokens."

    if chunks:
        top = chunks[0]
        rationale = (
            f"{len(chunks)} chunk(s) for subtask; top: {top.symbol or top.path} "
            f"({top.path}) score={top.score:.3f}; {len(excluded)} excluded, "
            f"{dropped} field/item drops for budget {subtask.budget_tokens}.{skim_note}"
        )
    else:
        rationale = f"no chunks matched subtask within budget {subtask.budget_tokens}; {len(excluded)} excluded."

    return ScopedContext(
        chunks=chunks,
        rationale=rationale,
        excluded=excluded,
        trace_id=trace_id,
        total_tokens=tokens,
        dropped_for_budget=dropped,
        provenance="fresh",
    )
