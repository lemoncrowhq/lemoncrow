"""LLMReflector — tiered Playbook extraction from a trace.

The tier is selected by the same ``LEMONCROW_LLM_BACKEND`` switch the internal
LLM subsystem already uses:

    none / unset           -> heuristic only (extractor.extract_candidate).
                              Offline, zero LLM calls. Default + local tier.
    ollama                 -> local Ollama model enriches the candidate.
    openai / openai_compatible -> OpenAI-compatible endpoint enriches it.

The LLM path *enriches* the heuristic candidate rather than replacing it: it
may sharpen ``situation`` and add/refine ``procedure`` / ``dead_ends`` /
``verification`` / ``failure_signals`` by contrasting what failed against what
worked. The heuristic block is always the floor, so an unavailable or
misbehaving model can never yield an invalid or empty block — on any error we
return the heuristic candidate unchanged.
"""

from __future__ import annotations

import os
from typing import Any

from lemoncrow.core.foundation.models import Playbook, Trace
from lemoncrow.infra.internal_llm import InternalLLMError, chat
from lemoncrow.pro.foundation.extractor import CandidateBlock, extract_candidate

_LLM_REFLECT_BONUS = 0.10
# Non-None json_schema only signals JSON mode to the internal LLM clients.
_REFLECT_SCHEMA: dict[str, Any] = {"type": "object"}


def _backend() -> str:
    return os.environ.get("LEMONCROW_LLM_BACKEND", "none").lower().strip()


def reflect(trace: Trace, *, backend: str | None = None) -> CandidateBlock:
    """Produce a candidate Playbook from *trace*, LLM-enriched when enabled.

    ``backend`` overrides the ``LEMONCROW_LLM_BACKEND`` env switch (mainly for
    tests). ``none``/unset keeps the pure heuristic path.
    """
    candidate = extract_candidate(trace)
    resolved = backend if backend is not None else _backend()
    if resolved in ("", "none"):
        return candidate
    try:
        enriched = _llm_enrich(trace, candidate.block)
    except (InternalLLMError, ValueError, KeyError, TypeError):
        return candidate
    if enriched is None:
        return candidate
    confidence = min(1.0, candidate.confidence + _LLM_REFLECT_BONUS)
    reasons = [*candidate.reasons, f"llm-reflected ({resolved}) (+{_LLM_REFLECT_BONUS:.2f})"]
    return CandidateBlock(block=enriched, confidence=confidence, reasons=reasons)


def _llm_enrich(trace: Trace, base: Playbook) -> Playbook | None:
    raw = chat(_build_messages(trace, base), json_schema=_REFLECT_SCHEMA)
    if not isinstance(raw, dict):
        return None
    return _merge_llm_fields(base, raw)


def _build_messages(trace: Trace, base: Playbook) -> list[dict[str, str]]:
    errors = "\n".join(f"- {e}" for e in trace.errors_seen[:10]) or "(none recorded)"
    repeated = "\n".join(f"- {rf.signature} (x{rf.count})" for rf in trace.repeated_failures[:10]) or "(none)"
    passed = [v.name for v in trace.validation_results if v.passed]
    failed = [v.name for v in trace.validation_results if not v.passed]
    system = (
        "You distill one durable engineering procedure from a single agent run. "
        "Contrast what failed against what finally worked. Be concrete and terse. "
        "Return ONLY a JSON object with keys: situation (string), "
        "procedure (array of imperative steps), dead_ends (array of strings), "
        "verification (array of strings), failure_signals (array of strings), "
        "when_not_to_apply (string)."
    )
    user = (
        f"Task: {trace.task}\n"
        f"Outcome: {trace.status}\n"
        f"Errors seen:\n{errors}\n"
        f"Repeated failures:\n{repeated}\n"
        f"Validations passed: {', '.join(passed) or '(none)'}\n"
        f"Validations failed: {', '.join(failed) or '(none)'}\n"
        f"Change summary: {trace.diff_summary or '(none)'}\n\n"
        "Current draft procedure (improve, do not discard real steps):\n"
        + "\n".join(f"- {step}" for step in base.procedure)
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _merge_llm_fields(base: Playbook, raw: dict[str, Any]) -> Playbook:
    situation = _clean_str(raw.get("situation")) or base.situation
    when_not = _clean_str(raw.get("when_not_to_apply")) or base.when_not_to_apply
    return base.model_copy(
        update={
            "situation": situation,
            "procedure": _union(base.procedure, _as_str_list(raw.get("procedure")), 12),
            "dead_ends": _union(base.dead_ends, _as_str_list(raw.get("dead_ends")), 8),
            "verification": _union(base.verification, _as_str_list(raw.get("verification")), 8),
            "failure_signals": _union(base.failure_signals, _as_str_list(raw.get("failure_signals")), 8),
            "when_not_to_apply": when_not,
        }
    )


def _clean_str(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _union(existing: list[str], incoming: list[str], cap: int) -> list[str]:
    out: list[str] = []
    for item in [*existing, *incoming]:
        if item and item not in out:
            out.append(item)
    return out[:cap]


__all__ = ["reflect"]
