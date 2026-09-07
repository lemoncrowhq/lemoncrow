"""Sleeptime summarizer for context compression."""

from __future__ import annotations

import logging
import os
from typing import Any

from lemoncrow.core.capabilities.context_compression_contract import (
    SleeptimeChunk as SleeptimeChunk,
)
from lemoncrow.core.capabilities.context_compression_contract import (
    SleeptimeUnavailable as SleeptimeUnavailable,
)
from lemoncrow.infra.internal_llm import InternalLLMError, summarize
from lemoncrow.infra.storage.factory import _memory_backend

logger = logging.getLogger(__name__)


def summarize_ledger(
    dropped_events: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> list[SleeptimeChunk]:
    if not dropped_events:
        return []

    text = _events_text(dropped_events)
    try:
        summary = summarize(text, max_tokens=256)
        return [
            SleeptimeChunk(
                start_event_index=start_index,
                end_event_index=start_index + len(dropped_events) - 1,
                paraphrase=summary.strip(),
            )
        ]
    except InternalLLMError:
        logger.warning(
            "Suppressed exception at sleeptime.py:43",
            exc_info=True,
        )

    try:
        from pathlib import Path

        from lemoncrow.core.foundation.paths import default_store_root

        backend = _memory_backend(
            Path(os.environ.get("LEMONCROW_ROOT", str(default_store_root()))),
            prefer=None,
        )
        if backend == "letta":
            from lemoncrow.infra.memory_bridges.letta_adapter import LettaAdapter

            chunks = LettaAdapter().summarize_run(dropped_events)
            return [SleeptimeChunk(**chunk) for chunk in chunks]
    except Exception as exc:
        raise SleeptimeUnavailable("Internal LLM and Letta sleeptime summarizers are unavailable") from exc

    raise SleeptimeUnavailable("Internal LLM and Letta sleeptime summarizers are unavailable")


def local_summarize(
    dropped_events: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> list[SleeptimeChunk]:
    """Backward-compatible alias for the real sleeptime summarizer."""
    return summarize_ledger(dropped_events, start_index=start_index)


def _events_text(dropped_events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, event in enumerate(dropped_events):
        kind = str(event.get("kind", "unknown"))
        summary = str(event.get("summary", ""))
        payload = str(event.get("payload", ""))[:1000]
        lines.append(f"[{idx}] kind={kind}\nsummary={summary}\npayload={payload}")
    return "\n\n".join(lines)


def deterministic_group_summary(
    dropped_events: list[dict[str, Any]],
    *,
    start_index: int = 0,
) -> list[SleeptimeChunk]:
    """Deterministic test helper; not used as a production fallback."""
    if not dropped_events:
        return []

    chunks: list[SleeptimeChunk] = []
    group_start = 0
    group_kind = str(dropped_events[0].get("kind", "unknown"))
    group_events = [dropped_events[0]]

    def _emit(g_start: int, g_end: int, events: list[dict[str, Any]], kind: str) -> SleeptimeChunk:
        n = len(events)
        last_summary = str(events[-1].get("summary", ""))[:200]
        label = f"{kind}s" if not kind.endswith("s") else kind
        paraphrase = f"[{n} {label}] {last_summary}".strip()
        return SleeptimeChunk(
            start_event_index=start_index + g_start,
            end_event_index=start_index + g_end,
            paraphrase=paraphrase,
        )

    for i, ev in enumerate(dropped_events[1:], start=1):
        kind = str(ev.get("kind", "unknown"))
        if kind == group_kind:
            group_events.append(ev)
        else:
            chunks.append(_emit(group_start, group_start + len(group_events) - 1, group_events, group_kind))
            group_start = i
            group_kind = kind
            group_events = [ev]

    chunks.append(_emit(group_start, group_start + len(group_events) - 1, group_events, group_kind))
    return chunks


__all__ = [
    "SleeptimeChunk",
    "SleeptimeUnavailable",
    "deterministic_group_summary",
    "local_summarize",
    "summarize_ledger",
]
