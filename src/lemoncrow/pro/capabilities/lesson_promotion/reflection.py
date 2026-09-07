"""Reflection drafting for lesson promotion clusters."""

from __future__ import annotations

import logging

from lemoncrow.core.foundation.models import Trace
from lemoncrow.infra.internal_llm import summarize
from lemoncrow.infra.internal_llm.ollama_client import OllamaUnavailable as InternalLLMError

_log = logging.getLogger(__name__)

_REFLECTION_PROMPT = (
    "From these failed attempts and the eventual fix, write a one-paragraph procedural "
    "reflection: what was the dead-end, what worked, and when does this apply?\n\n"
)


def _trace_excerpt(trace: Trace) -> str:
    commands = [str(item if isinstance(item, str) else item.command) for item in trace.commands_run]
    return "\n".join(
        [
            f"Task: {trace.task}",
            f"Commands: {'; '.join(commands[:5])}",
            f"Errors: {'; '.join(trace.errors_seen[:5])}",
            f"Diff: {trace.diff_summary}",
            f"Output: {trace.output_summary}",
        ]
    ).strip()


def fallback_lesson_body(cluster_traces: list[Trace]) -> str:
    """Return the deterministic fallback body used when local summarization is unavailable."""
    parts: list[str] = []
    for trace in cluster_traces[:5]:
        errors = "; ".join(trace.errors_seen[:3]) or "no explicit error recorded"
        diff = trace.diff_summary or trace.output_summary or trace.task
        parts.append(f"Dead-end: {errors}. Worked/fix signal: {diff}")
    return " ".join(parts).strip()


def draft_lesson_body(cluster_traces: list[Trace]) -> str:
    """Draft a procedural reflection for a promoted lesson cluster."""
    text = _REFLECTION_PROMPT + "\n\n---\n\n".join(_trace_excerpt(trace) for trace in cluster_traces)
    try:
        body = summarize(text, max_tokens=220)
    except InternalLLMError:
        _log.info("Internal LLM unavailable for lesson reflection; using deterministic fallback")
        return fallback_lesson_body(cluster_traces)
    return body.strip() or fallback_lesson_body(cluster_traces)


__all__ = ["draft_lesson_body", "fallback_lesson_body"]
