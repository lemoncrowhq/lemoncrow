"""Manual sleep-time consolidation worker."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from lemoncrow.core.foundation.models import ConsolidationCandidate, Playbook
from lemoncrow.infra.internal_llm import InternalLLMError, chat
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.pro.foundation.curator import MIN_EVIDENCE, REMOVE_MIN_FAILURES, REMOVE_SUCCESS_RATE

_DEFAULT_STALE_AFTER = timedelta(days=180)


@dataclass(frozen=True)
class ConsolidationReport:
    duplicates: int
    stale: int
    quarantined: int
    llm_suggestions: int
    written: int

    def to_dict(self) -> dict[str, int]:
        return {
            "duplicates": self.duplicates,
            "stale": self.stale,
            "quarantined": self.quarantined,
            "llm_suggestions": self.llm_suggestions,
            "written": self.written,
        }


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z0-9_]+", text)}


def _block_tokens(b: Playbook) -> set[str]:
    """Pre-compute the token set for a Playbook."""
    return _tokens(" ".join([b.title, b.situation, *b.procedure, *b.failure_signals]))


def _token_set_similarity(left: set[str], right: set[str]) -> float:
    """Jaccard similarity between two pre-computed token sets."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _draft_merge(blocks: list[Playbook]) -> tuple[str | None, bool]:
    payload = [block.model_dump(mode="json") for block in blocks]
    try:
        response = chat(
            [
                {
                    "role": "system",
                    "content": "Return JSON with duplicate:boolean and proposed_body:string.",
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
                },
            ],
            json_schema={"type": "object"},
        )
    except InternalLLMError:
        return None, False
    if isinstance(response, dict) and response.get("duplicate") is not False:
        return str(response.get("proposed_body", "") or "").strip() or None, True
    return None, True


def _should_quarantine(block: Playbook) -> bool:
    total = block.success_count + block.failure_count
    return (
        block.status == "active"
        and total >= MIN_EVIDENCE
        and block.failure_count >= REMOVE_MIN_FAILURES
        and block.success_rate() <= REMOVE_SUCCESS_RATE
    )


def consolidate(
    store: StoreBundle,
    *,
    since: timedelta = _DEFAULT_STALE_AFTER,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Find duplicate/stale knowledge rows and write human-reviewed candidates."""
    blocks = store.knowledge.list_blocks(include_deprecated=False)
    blocks_to_quarantine = [block for block in blocks if _should_quarantine(block)]
    quarantined_ids = {block.id for block in blocks_to_quarantine}
    candidate_blocks = [block for block in blocks if block.id not in quarantined_ids]

    # Pre-compute token sets once — O(N) instead of O(N²) tokenizations
    token_sets = [_block_tokens(b) for b in candidate_blocks]

    candidates: list[ConsolidationCandidate] = []
    used: set[str] = set()
    llm_suggestions = 0

    for idx, block in enumerate(candidate_blocks):
        if block.id in used:
            continue
        cluster = [block]
        left_tokens = token_sets[idx]
        for j in range(idx + 1, len(candidate_blocks)):
            if candidate_blocks[j].id in used:
                continue
            if _token_set_similarity(left_tokens, token_sets[j]) >= 0.75:
                cluster.append(candidate_blocks[j])
        if len(cluster) < 2:
            continue
        for item in cluster:
            used.add(item.id)
        proposed_body, used_llm = _draft_merge(cluster)
        if used_llm:
            llm_suggestions += 1
        candidates.append(
            ConsolidationCandidate(
                kind="duplicate_cluster",
                affected_block_ids=[item.id for item in cluster],
                proposed_action="merge",
                proposed_body=proposed_body,
                evidence={"method": "internal_llm" if used_llm else "deterministic_only"},
            )
        )

    cutoff = datetime.now(UTC) - since
    stale_blocks = [block for block in candidate_blocks if block.id not in used and block.updated_at < cutoff]
    for block in stale_blocks:
        candidates.append(
            ConsolidationCandidate(
                kind="stale_candidate",
                affected_block_ids=[block.id],
                proposed_action="deprecate",
                evidence={
                    "block_id": block.id,
                    "method": "deterministic_only",
                    "source": "playbook",
                },
            )
        )

    stale_lessons = [
        item for item in store.lessons.list_lesson_candidates(status="inbox", limit=500) if item.created_at < cutoff
    ]
    for lesson in stale_lessons:
        candidates.append(
            ConsolidationCandidate(
                kind="stale_candidate",
                affected_block_ids=lesson.evidence_trace_ids,
                proposed_action="deprecate",
                evidence={"lesson_id": lesson.id, "method": "deterministic_only"},
            )
        )

    if not dry_run:
        quarantined_at = datetime.now(UTC)
        for block in blocks_to_quarantine:
            store.knowledge.upsert_block(
                block.model_copy(update={"status": "quarantined", "updated_at": quarantined_at})
            )
        for candidate in candidates:
            store.lessons.upsert_consolidation_candidate(candidate)

    duplicate_count = sum(1 for candidate in candidates if candidate.kind == "duplicate_cluster")
    stale_count = sum(1 for candidate in candidates if candidate.kind == "stale_candidate")
    return ConsolidationReport(
        duplicates=duplicate_count,
        stale=stale_count,
        quarantined=len(blocks_to_quarantine),
        llm_suggestions=llm_suggestions,
        written=0 if dry_run else len(candidates),
    )


__all__ = ["ConsolidationReport", "consolidate"]
