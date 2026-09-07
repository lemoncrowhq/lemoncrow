"""ContextCompressionCapability — token-budget-aware context compression."""

from __future__ import annotations

import contextlib
import hashlib
import logging
from typing import TYPE_CHECKING, Any

from .deduplication import deduplicate_tool_outputs
from .models import CompressionResult, DroppedContext
from .scoring import score_events
from .sleeptime import SleeptimeChunk, SleeptimeUnavailable, summarize_ledger

if TYPE_CHECKING:
    from lemoncrow.infra.runtime.run_ledger import RunLedger

# Approximate chars-per-token ratio
_CHARS_PER_TOKEN = 4
_log = logging.getLogger(__name__)


class ContextCompressionCapability:
    """
    Intelligent context compression using TF-IDF event scoring,
    recency weighting, and semantic deduplication.

    Strategies applied in order:
    1. Semantic deduplication (collapse near-identical events)
    2. TF-IDF + recency scoring (rank remaining events by importance)
    3. Budget-aware truncation (keep highest-scoring events up to token budget)
    """

    def _compress(
        self,
        ledger: RunLedger,
        *,
        token_budget: int,
        task: str,
    ) -> tuple[CompressionResult, list[dict[str, Any]]]:
        """Run the dedup + score + budget pipeline once.

        Returns the provenance ``CompressionResult`` and the raw dropped event
        dicts (dedup drops followed by budget drops) so callers that need the
        evicted events do not have to re-run the whole pipeline.
        """
        raw_events: list[Any] = []
        with contextlib.suppress(Exception):
            raw_events = list(getattr(ledger, "events", []) or [])

        # Normalise LedgerEvent Pydantic models to plain dicts
        events: list[dict[str, Any]] = [_normalise_event(ev) for ev in raw_events]

        chars_before = sum(len(str(ev.get("summary", ""))) + len(str(ev.get("payload", ""))) for ev in events)

        # Step 1: Deduplicate near-identical outputs
        events_deduped, dropped_events = deduplicate_tool_outputs(events)
        dropped: list[DroppedContext] = [
            DroppedContext(
                kind=str(ev.get("kind", "unknown")),
                summary=str(ev.get("summary", ""))[:200],
                original_chars=len(str(ev.get("summary", ""))),
            )
            for ev in dropped_events
        ]

        # Step 2: Score remaining events
        scored = score_events(events_deduped, task=task)
        scored.sort(key=lambda x: x.score, reverse=True)

        # Step 3: Greedy token-budget selection (keep highest-scoring until budget used).
        # Keystone-protected events (decision-flipping facts) are always kept even when
        # they fall below the budget cutoff — that is the guarantee score_events promises.
        budget_chars = token_budget * _CHARS_PER_TOKEN
        selected: list[dict[str, Any]] = []
        used_chars = 0
        budget_dropped: list[DroppedContext] = []
        budget_dropped_raw: list[dict[str, Any]] = []

        for es in scored:
            ev = es.event
            ev_chars = len(str(ev.get("summary", ""))) + len(str(ev.get("payload", "")))
            if es.keystone_protected or used_chars + ev_chars <= budget_chars:
                selected.append(ev)
                used_chars += ev_chars
            else:
                budget_dropped.append(
                    DroppedContext(
                        kind=str(ev.get("kind", "unknown")),
                        summary=str(ev.get("summary", ""))[:200],
                        original_chars=ev_chars,
                    )
                )
                budget_dropped_raw.append(ev)

        dropped += budget_dropped

        chars_after = used_chars
        reduction_pct = round(100.0 * (chars_before - chars_after) / chars_before, 1) if chars_before > 0 else 0.0
        token_savings = (chars_before - chars_after) // _CHARS_PER_TOKEN

        # Build preserved_facts from selected events (highest importance first)
        preserved_facts = [f"[{ev.get('kind', '?')}] {str(ev.get('summary', ''))[:200]}" for ev in selected[:20]]

        result = CompressionResult(
            chars_before=chars_before,
            chars_after=chars_after,
            reduction_pct=reduction_pct,
            preserved_facts=preserved_facts,
            dropped=dropped,
            token_savings=token_savings,
        )
        return result, dropped_events + budget_dropped_raw

    def compress_with_provenance(
        self,
        ledger: RunLedger,
        *,
        token_budget: int = 8000,
        task: str = "",
    ) -> CompressionResult:
        """
        Compress the ledger's context and return a full provenance record.

        Args:
            ledger:       The run ledger to compress.
            token_budget: Maximum allowed tokens after compression.
            task:         Optional task description for task-conditioned scoring.
                          Events whose summaries overlap with task terms score higher.
        """
        from lemoncrow.bench.mode import is_off as _bench_is_off
        from lemoncrow.core.capabilities import licensing

        if _bench_is_off() or not licensing.has_feature("context_compression"):
            return CompressionResult.passthrough()
        result, _dropped = self._compress(ledger, token_budget=token_budget, task=task)
        return result

    def compress_with_sleeptime(
        self,
        ledger: RunLedger,
        *,
        token_budget: int = 8000,
        agent_id: str = "lemoncrow",
        task: str = "",
    ) -> CompressionResult:
        """Like ``compress_with_provenance`` but also:

        * Converts each evicted event into a ``SleeptimeChunk`` paraphrase.
        * Archives each chunk as an ``ArchivalPassage`` in the memory store.
        * Writes a ``RunMemoryFrame`` row with tokens_pre/post and strategy.

        The original ``compress_with_provenance`` is unchanged.
        """
        from lemoncrow.bench.mode import is_off as _bench_is_off
        from lemoncrow.core.capabilities import licensing

        if _bench_is_off() or not licensing.has_feature("context_compression"):
            return CompressionResult.passthrough()
        result, all_dropped_events = self._compress(ledger, token_budget=token_budget, task=task)

        # Use a real sleeptime summarizer if available; otherwise skip the lever.
        chunks: list[SleeptimeChunk] = []
        strategy = "tfidf"
        try:
            chunks = summarize_ledger(all_dropped_events)
            strategy = "llm_summarizer"
        except SleeptimeUnavailable as exc:
            _log.warning("Sleeptime summarizer unavailable; skipping archival summary: %s", exc)
            chunks = []

        # Archive each chunk as an ArchivalPassage
        archived_ids: list[str] = []
        try:
            import os
            from pathlib import Path

            from lemoncrow.core.foundation.memory_models import ArchivalPassage
            from lemoncrow.core.foundation.paths import default_store_root, resolve_workspace_root
            from lemoncrow.infra.storage.sqlite_memory_store import SqliteMemoryStore

            root = Path(os.environ.get("LEMONCROW_ROOT", str(default_store_root())))
            store = SqliteMemoryStore(root)
            session_id = getattr(ledger, "session_id", "unknown")
            for chunk in chunks:
                dedup_hash = hashlib.sha1(chunk.paraphrase.encode()).hexdigest()
                passage = ArchivalPassage(
                    agent_id=agent_id,
                    text=chunk.paraphrase,
                    source="block_evict",
                    source_ref=f"run:{session_id}",
                    dedup_hash=dedup_hash,
                )
                saved = store.insert_passage(passage)
                archived_ids.append(saved.id)
        except Exception as exc:  # pragma: no cover
            logging.exception("Recovered from broad exception handler")
            _log.warning("Failed to archive sleeptime passages: %s", exc)

        # Write RunMemoryFrame
        try:
            import os
            from pathlib import Path

            from lemoncrow.core.foundation.memory_models import RunMemoryFrame
            from lemoncrow.core.foundation.paths import default_store_root, resolve_workspace_root
            from lemoncrow.infra.storage.sqlite_memory_store import SqliteMemoryStore

            root = Path(os.environ.get("LEMONCROW_ROOT", str(default_store_root())))
            store = SqliteMemoryStore(root)
            session_id = getattr(ledger, "session_id", "unknown")
            frame = RunMemoryFrame(
                session_id=session_id,
                workspace_path=str(resolve_workspace_root(root)),
                pinned_blocks=[],
                recalled_passages=[],
                summarized_events=[c.paraphrase for c in chunks],
                tokens_pre_summary=result.chars_before // _CHARS_PER_TOKEN,
                tokens_post_summary=result.chars_after // _CHARS_PER_TOKEN,
                compaction_strategy=strategy,  # type: ignore[arg-type]
            )
            store.write_run_frame(frame)
        except Exception as exc:  # pragma: no cover
            logging.exception("Recovered from broad exception handler")
            _log.warning("Failed to write RunMemoryFrame: %s", exc)

        return result

    def context_report(self, ledger: RunLedger) -> dict[str, Any]:
        """Return a dict summary of current context compression state."""
        result = self.compress_with_provenance(ledger)
        return result.to_dict()

    # Legacy alias
    def compress(self, ledger: RunLedger) -> dict[str, Any]:
        """Legacy single-pass compress; returns same dict as context_report."""
        return self.context_report(ledger)


def _normalise_event(ev: Any) -> dict[str, Any]:
    """Convert a LedgerEvent Pydantic model or plain dict to a plain dict."""
    if isinstance(ev, dict):
        import typing

        return typing.cast(dict[str, Any], ev)
    import typing

    # Pydantic model — use model_dump if available (Pydantic v2), else __dict__
    if hasattr(ev, "model_dump"):
        return typing.cast(dict[str, Any], ev.model_dump())
    if hasattr(ev, "dict"):
        return typing.cast(dict[str, Any], ev.dict())
    # Fallback: attribute access via known LedgerEvent fields
    return {
        "kind": getattr(ev, "kind", "unknown"),
        "summary": getattr(ev, "summary", ""),
        "payload": getattr(ev, "payload", {}),
    }
