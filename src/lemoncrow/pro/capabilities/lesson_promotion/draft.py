"""Drafting heuristics for lesson promotion candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from lemoncrow.core.foundation.lesson_models import LessonCandidate
from lemoncrow.core.foundation.models import Playbook, Trace

_COMMON_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
# Cap the contiguous word-span search so a huge multi-line stack trace cannot
# turn the O(words^2) candidate enumeration into a hot-path bottleneck.
_MAX_SHARED_WORDS = 40


def _command_text(command: str | Any) -> str:
    if isinstance(command, str):
        return command
    return str(getattr(command, "command", ""))


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _COMMON_WORD_RE.finditer(text)}


def _shared_error_substring(errors: list[str], min_len: int = 12) -> str | None:
    if len(errors) < 2:
        return None
    base = min((e for e in errors if e), key=len, default="")
    if len(base) < min_len:
        return None

    words = [word for word in re.split(r"\s+", base.strip()) if word][:_MAX_SHARED_WORDS]
    for span in range(len(words), 0, -1):
        for start in range(0, len(words) - span + 1):
            sub = " ".join(words[start : start + span]).strip(" ,.:;!()[]{}\"'")
            if len(sub) < min_len:
                continue
            if all(sub in err for err in errors):
                return sub
    return None


def _same_failed_rubric_check(traces: list[Trace]) -> str | None:
    failed_names: list[str] = []
    for trace in traces:
        failed = [r.name for r in trace.validation_results if not r.passed and r.name]
        if not failed:
            return None
        failed_names.extend(failed)
    counts = Counter(failed_names)
    check, count = counts.most_common(1)[0]
    return check if count >= len(traces) else None


def _synthesized_rubric_name(domain: str, check: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", check.lower()).strip("_") or "recurring_failure"
    dom = re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_") or "general"
    return f"lesson_{dom}_{clean}"


def _overlap_score(block: Playbook, errors: list[str]) -> float:
    block_text = " ".join(block.dead_ends + block.failure_signals + [block.situation])
    block_tokens = _tokenize(block_text)
    err_tokens = _tokenize(" ".join(errors))
    if not block_tokens or not err_tokens:
        return 0.0
    return len(block_tokens & err_tokens) / max(1, len(block_tokens | err_tokens))


def _investigation_procedure(traces: list[Trace], shared: str | None) -> list[str]:
    verbs = ["triage", "inspect", "stabilize", "verify"]
    components: list[str] = []
    for trace in traces:
        for cmd in trace.commands_run:
            text = _command_text(cmd)
            if not text:
                continue
            token = text.split()[0]
            components.append(token)
    component = Counter(components).most_common(1)[0][0] if components else "pipeline"
    verb = verbs[min(len(components), len(verbs) - 1)]
    detail = shared or "the recurring failure signature"
    return [f"Investigate {verb} in {component} for '{detail[:48]}'"]


def draft_lesson_candidate(
    *,
    traces: list[Trace],
    domain: str,
    cluster_fingerprint: str,
    embedding: list[float] | None,
    existing_blocks: list[Playbook],
) -> LessonCandidate:
    """Draft a lesson candidate for a cluster of related failed traces."""
    evidence = [t.id for t in traces]
    errors: list[str] = []
    for trace in traces:
        errors.extend([e for e in trace.errors_seen if e])

    same_check = _same_failed_rubric_check(traces)
    if same_check:
        return LessonCandidate(
            domain=domain,
            cluster_fingerprint=cluster_fingerprint,
            kind="new_rubric_check",
            proposed_rubric_check=_synthesized_rubric_name(domain, same_check),
            evidence_trace_ids=evidence,
            embedding=embedding,
            confidence=0.82,
        )

    shared_error = _shared_error_substring(errors)
    if not shared_error and len(cluster_fingerprint.strip()) >= 12:
        shared_error = cluster_fingerprint
    if shared_error:
        block = Playbook(
            id=Playbook.make_id(f"lesson {shared_error[:32]}", domain),
            title=f"Investigate recurring failure: {shared_error[:48]}",
            domain=domain,
            triggers=["recurring_failure", domain],
            situation="Repeated failures share a stable error signature.",
            dead_ends=[shared_error],
            procedure=_investigation_procedure(traces, shared_error),
            verification=["Re-run the failing scenario and confirm the error signature is absent."],
            failure_signals=[shared_error],
        )
        return LessonCandidate(
            domain=domain,
            cluster_fingerprint=cluster_fingerprint,
            kind="new_block",
            proposed_block=block,
            evidence_trace_ids=evidence,
            embedding=embedding,
            confidence=0.78,
        )

    target_id = None
    best_score = 0.0
    for block in existing_blocks:
        score = _overlap_score(block, errors)
        if score > best_score:
            best_score = score
            target_id = block.id

    return LessonCandidate(
        domain=domain,
        cluster_fingerprint=cluster_fingerprint,
        kind="edit_block",
        target_id=target_id,
        evidence_trace_ids=evidence,
        embedding=embedding,
        confidence=0.65 if target_id else 0.5,
        decision_reason="suggested dead-end addition from recurring failures",
    )
