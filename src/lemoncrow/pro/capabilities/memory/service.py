"""Host-neutral memory operations shared by MCP, CLI, SDK, and API surfaces."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from lemoncrow.core.capabilities.memory_contract import (
    FactScope as FactScope,
)
from lemoncrow.core.capabilities.memory_contract import (
    MemoryFactResult as MemoryFactResult,
)
from lemoncrow.core.capabilities.memory_contract import (
    MemoryRecallPassage as MemoryRecallPassage,
)
from lemoncrow.core.capabilities.memory_contract import (
    MemoryRecallResult as MemoryRecallResult,
)
from lemoncrow.core.capabilities.memory_contract import (
    MemoryVoteResult as MemoryVoteResult,
)
from lemoncrow.core.capabilities.memory_contract import (
    VoteDirection as VoteDirection,
)
from lemoncrow.core.foundation.memory_models import MemoryBlock
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.storage.memory_store import MemoryStore
from lemoncrow.pro.capabilities.archival_recall import ArchivalRecallCapability
from lemoncrow.pro.capabilities.memory_arbitration import arbitrate
from lemoncrow.pro.capabilities.memory_arbitration.arbiter import _similar_blocks


class MemoryService:
    """Canonical memory lifecycle operations, independent of transport."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        embedder: Embedder,
        redactor: Callable[[str], str],
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._redactor = redactor

    def recall(
        self,
        *,
        agent_id: str | None,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
        since: str | None = None,
    ) -> MemoryRecallResult:
        """Recall relevant archival memory passages."""
        since_dt = datetime.fromisoformat(since) if since else None
        passages, _ = ArchivalRecallCapability(self._store, self._embedder, redactor=self._redactor).recall(
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            tags=tags or None,
            since=since_dt,
        )
        return MemoryRecallResult(
            passages=[
                MemoryRecallPassage(
                    id=passage.id,
                    text=passage.text,
                    source_ref=passage.source_ref,
                    tags=passage.tags,
                )
                for passage in passages
            ]
        )

    def store_fact(
        self,
        *,
        agent_id: str | None,
        subject: str,
        fact: str,
        scope: str,
        citations: str = "",
        reason: str = "",
    ) -> MemoryFactResult:
        """Store or update a durable fact."""
        clean_subject = self._clean_required(subject, "subject")
        clean_fact = self._clean_required(fact, "fact")
        clean_citations = self._redactor(citations).strip()
        clean_reason = self._redactor(reason).strip()
        clean_scope = self._fact_scope(scope)
        target_agent = agent_id or "shared"

        existing = self._find_fact_block(target_agent, clean_fact, scope=clean_scope)
        if existing is None:
            subject_slug = re.sub(r"[^a-z0-9]+", "-", clean_subject.lower()).strip("-") or "memory"
            digest = sha256(f"{clean_scope}:{clean_subject}:{clean_fact}".encode()).hexdigest()[:12]
            label = f"memory-fact/{clean_scope}/{subject_slug}/{digest}"
            metadata: dict[str, Any] = {
                "kind": "memory_fact",
                "subject": clean_subject,
                "fact": clean_fact,
                "citations": clean_citations,
                "reason": clean_reason,
                "scope": clean_scope,
                "fact_scope": clean_scope,
                "votes": {"upvote": 0, "downvote": 0},
                "vote_history": [],
            }
            stored = self._upsert_block(
                MemoryBlock(
                    agent_id=target_agent,
                    label=label,
                    value=clean_fact,
                    metadata=metadata,
                    pinned=True,
                )
            )
        else:
            metadata = self._fact_metadata(
                existing,
                subject=clean_subject,
                fact=clean_fact,
                citations=clean_citations,
                reason=clean_reason,
                scope=clean_scope,
            )
            stored = self._upsert_block(
                existing.model_copy(update={"value": clean_fact, "metadata": metadata, "pinned": True})
            )

        return MemoryFactResult(
            id=stored.id,
            subject=clean_subject,
            fact=clean_fact,
            scope=clean_scope,
            citations=clean_citations,
            reason=clean_reason,
        )

    def vote_fact(
        self,
        *,
        agent_id: str | None,
        fact: str,
        direction: str,
        reason: str,
        scope: str | None = None,
    ) -> MemoryVoteResult:
        """Vote on an existing fact by exact fact text."""
        clean_fact = self._clean_required(fact, "fact")
        clean_reason = self._clean_required(reason, "reason")
        clean_direction = self._vote_direction(direction)
        clean_scope = self._fact_scope(scope) if scope else None
        target_agent = agent_id or "shared"
        if clean_scope is None:
            matched_scopes = {
                str((block.metadata or {}).get("fact_scope") or (block.metadata or {}).get("scope") or "")
                for block in self._store.list_blocks(target_agent, include_tombstoned=False, limit=500)
                if (block.metadata or {}).get("kind") == "memory_fact"
                and str((block.metadata or {}).get("fact", "")) == clean_fact
            }
            if len(matched_scopes) > 1:
                raise ValueError("fact exists in multiple scopes; specify scope to disambiguate vote_fact")
        match = self._find_fact_block(target_agent, clean_fact, scope=clean_scope)
        if match is None:
            raise ValueError("no matching stored fact found for vote_fact")

        metadata = dict(match.metadata or {})
        fact_scope = self._fact_scope(str(metadata.get("fact_scope") or metadata.get("scope") or "repository"))
        votes = dict(metadata.get("votes") or {})
        up = int(votes.get("upvote", 0) or 0)
        down = int(votes.get("downvote", 0) or 0)
        if clean_direction == "upvote":
            up += 1
        else:
            down += 1
        history = list(metadata.get("vote_history") or [])
        voted_at = datetime.now(UTC).isoformat()
        history.append({"direction": clean_direction, "reason": clean_reason, "at": voted_at})
        metadata["votes"] = {"upvote": up, "downvote": down}
        metadata["vote_history"] = history[-20:]
        metadata["last_vote"] = {"direction": clean_direction, "reason": clean_reason, "at": voted_at}
        metadata["fact_scope"] = fact_scope

        stored = self._upsert_block(match.model_copy(update={"metadata": metadata}), dedup=False)
        return MemoryVoteResult(
            id=stored.id,
            fact=clean_fact,
            scope=fact_scope,
            direction=clean_direction,
            reason=clean_reason,
        )

    def list_facts(self, *, agent_id: str | None = None, limit: int = 500) -> list[MemoryFactResult]:
        """List stored facts from LemonCrow memory blocks."""
        return [
            self._result_from_block(block)
            for block in self._store.list_blocks(agent_id or "shared", include_tombstoned=False, limit=limit)
            if (block.metadata or {}).get("kind") == "memory_fact"
        ]

    def get_fact(self, *, fact_id: str, agent_id: str | None = None) -> MemoryFactResult | None:
        """Get one stored fact by memory block id."""
        for block in self._store.list_blocks(agent_id or "shared", include_tombstoned=False, limit=500):
            if block.id == fact_id and (block.metadata or {}).get("kind") == "memory_fact":
                return self._result_from_block(block)
        return None

    def share_fact(
        self,
        *,
        agent_id: str | None,
        fact_id: str,
        workspace_id: str,
        shared_by_user_id: str,
    ) -> MemoryFactResult:
        """Mark a stored fact as workspace-shared without changing its fact scope."""
        target_agent = agent_id or "shared"
        for block in self._store.list_blocks(target_agent, include_tombstoned=False, limit=500):
            if block.id != fact_id or (block.metadata or {}).get("kind") != "memory_fact":
                continue
            metadata = dict(block.metadata or {})
            metadata["visibility_scope"] = "shared"
            metadata["workspace_id"] = workspace_id
            metadata["shared_by_user_id"] = shared_by_user_id
            stored = self._upsert_block(block.model_copy(update={"metadata": metadata}))
            return self._result_from_block(stored)
        raise ValueError(f"memory fact not found: {fact_id}")

    def _clean_required(self, value: str, name: str) -> str:
        clean = self._redactor(value).strip()
        if not clean:
            raise ValueError(f"{name} is required")
        return clean

    def _fact_scope(self, scope: str | None) -> FactScope:
        clean = (scope or "").strip().lower()
        if clean not in {"repository", "user"}:
            raise ValueError("scope must be one of: repository, user")
        return clean  # type: ignore[return-value]

    def _vote_direction(self, direction: str) -> VoteDirection:
        clean = direction.strip().lower()
        if clean not in {"upvote", "downvote"}:
            raise ValueError("direction must be one of: upvote, downvote")
        return clean  # type: ignore[return-value]

    def _find_fact_block(self, agent_id: str, fact: str, *, scope: str | None) -> MemoryBlock | None:
        for block in self._store.list_blocks(agent_id, include_tombstoned=False, limit=500):
            metadata = block.metadata or {}
            if metadata.get("kind") != "memory_fact":
                continue
            if str(metadata.get("fact", "")) != fact:
                continue
            block_scope = str(metadata.get("fact_scope") or metadata.get("scope") or "")
            if scope and block_scope != scope:
                continue
            return block
        return None

    def _fact_metadata(
        self,
        block: MemoryBlock,
        *,
        subject: str,
        fact: str,
        citations: str,
        reason: str,
        scope: FactScope,
    ) -> dict[str, Any]:
        metadata = dict(block.metadata or {})
        votes = dict(metadata.get("votes") or {})
        metadata.update(
            {
                "kind": "memory_fact",
                "subject": subject,
                "fact": fact,
                "citations": citations,
                "reason": reason,
                "scope": scope,
                "fact_scope": scope,
                "votes": {
                    "upvote": int(votes.get("upvote", 0) or 0),
                    "downvote": int(votes.get("downvote", 0) or 0),
                },
                "vote_history": list(metadata.get("vote_history") or []),
            }
        )
        return metadata

    def _result_from_block(self, block: MemoryBlock) -> MemoryFactResult:
        metadata = block.metadata or {}
        scope = self._fact_scope(str(metadata.get("fact_scope") or metadata.get("scope") or "repository"))
        return MemoryFactResult(
            id=block.id,
            subject=str(metadata.get("subject") or ""),
            fact=str(metadata.get("fact") or block.value),
            scope=scope,
            citations=str(metadata.get("citations") or ""),
            reason=str(metadata.get("reason") or ""),
        )

    def _upsert_block(self, block: MemoryBlock, *, dedup: bool = True) -> MemoryBlock:
        if not dedup:
            # Caller already resolved an exact existing block (e.g. vote_fact),
            # so routing through arbitrate() would let a NOOP verdict silently
            # drop the update or an UPDATE verdict reapply it to a different
            # similar block. Persist the resolved block directly (the store's
            # optimistic version check still guards concurrent writers).
            return self._store.upsert_block(block, actor=f"agent:{block.agent_id}", reason="direct-update")
        candidates = {item.id: item for item in _similar_blocks(block, self._store, k=5)}
        decision = arbitrate(block, self._store, self._embedder)
        target = None
        if decision.target_block_id:
            target = candidates.get(decision.target_block_id)

        actor = f"agent:{block.agent_id}"
        if decision.op == "NOOP" and target is not None:
            return target
        if decision.op == "UPDATE" and target is not None:
            return self._store.upsert_block(
                target.model_copy(
                    update={
                        "value": decision.merged_value or block.value,
                        "metadata": {**(target.metadata or {}), **(block.metadata or {})},
                    }
                ),
                actor=actor,
                reason=decision.reason,
            )
        if decision.op == "DELETE" and target is not None:
            self._store.tombstone_block(
                target.id,
                deprecated_by_block_id=block.id,
                reason=decision.reason,
            )
        return self._store.upsert_block(block, actor=actor, reason=decision.reason)
