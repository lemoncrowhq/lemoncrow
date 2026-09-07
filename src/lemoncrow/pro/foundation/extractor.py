"""Extractor — produce a candidate Playbook from a recorded trace.

Heuristic, not LLM-based. The output is always a *candidate* — humans (or
a future auto-rule) decide whether to accept it.

Confidence formula:
    base = 0.40
    + 0.20 if status == "success"
    + 0.10 per validation result that passed (cap 0.20)
    + 0.10 if at least one repeated_failure was overcome
    + 0.10 if both files_touched and validation_results are non-empty
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lemoncrow.core.foundation.models import (
    CommandRecord,
    FileEditRecord,
    Playbook,
    ToolCall,
    Trace,
    ValidationResult,
    slugify,
)


@dataclass
class CandidateBlock:
    block: Playbook
    confidence: float
    reasons: list[str]


_TRIGGER_STOPWORDS = {
    "with",
    "from",
    "into",
    "this",
    "that",
    "when",
    "where",
    "what",
    "your",
    "have",
    "been",
    "will",
    "does",
    "just",
}
_COMMAND_WRAPPER_TOKENS = {
    "uv",
    "run",
    "python",
    "python3",
    "env",
    "bash",
    "sh",
    "time",
    "nohup",
}


def extract_candidate(trace: Trace) -> CandidateBlock:
    title = _derive_title(trace)
    domain = trace.domain or "coding"
    block_id = Playbook.make_id(title, domain)

    dead_ends = _derive_dead_ends(trace)
    procedure = _derive_procedure(trace)
    verification = _derive_verification(trace)
    failure_signals = list({_short(e) for e in trace.errors_seen if e})[:8]

    confidence, reasons = _score_confidence(trace)

    block = Playbook(
        id=block_id,
        title=title,
        domain=domain,
        task_types=[slugify(domain)],
        triggers=_derive_triggers(trace),
        file_patterns=_derive_file_patterns(trace),
        tool_patterns=_derive_tool_patterns(trace),
        situation=_derive_situation(trace),
        dead_ends=dead_ends,
        procedure=procedure or ["(procedure could not be extracted; manual edit required)"],
        verification=verification,
        failure_signals=failure_signals,
        when_not_to_apply="",
        status="active",
    )
    return CandidateBlock(block=block, confidence=confidence, reasons=reasons)


# --------------------------------------------------------------------------- #
# Derivations                                                                 #
# --------------------------------------------------------------------------- #


def _derive_title(trace: Trace) -> str:
    base = trace.task.strip().rstrip(".")
    if len(base) > 80:
        base = base[:77] + "..."
    return base or "Untitled procedure"


def _derive_situation(trace: Trace) -> str:
    if trace.diff_summary:
        return f"When working on: {trace.task}. Context: {trace.diff_summary}"
    return f"When working on: {trace.task}"


def _derive_triggers(trace: Trace) -> list[str]:
    parts: list[str] = [trace.task, *trace.errors_seen]
    parts.extend(_trace_file_path(item) for item in trace.files_touched)
    parts.extend(_trace_tool_name(item) for item in trace.tools_called)
    parts.extend(_trace_command_text(item) for item in trace.commands_run)
    blob = " ".join(part for part in parts if part)
    blob = re.sub(r"([a-z])([A-Z])", r"\1 \2", blob)
    tokens = re.findall(r"[a-z0-9_./-]+", blob.lower())
    filtered: list[str] = []
    for token in tokens:
        for piece in re.split(r"[^a-z0-9]+", token):
            if len(piece) < 3 or piece in _TRIGGER_STOPWORDS:
                continue
            filtered.append(piece)
    return list(dict.fromkeys(filtered))[:16]


def _derive_file_patterns(trace: Trace) -> list[str]:
    out: list[str] = []
    for f in trace.files_touched:
        path = _trace_file_path(f)
        if not path:
            continue
        parts = path.split("/")
        pattern = "/".join(parts[:-1]) + "/**" if len(parts) >= 2 else parts[0]
        if pattern not in out:
            out.append(pattern)
        file_path = "/".join(parts)
        if file_path not in out:
            out.append(file_path)
    return out[:12]


def _derive_tool_patterns(trace: Trace) -> list[str]:
    patterns = [item.name for item in trace.tools_called if item.name]
    for command in trace.commands_run:
        text = _trace_command_text(command).strip()
        if not text:
            continue
        exe = _infer_command_tool(text)
        if exe:
            patterns.append(exe)
    return list(dict.fromkeys(patterns))[:12]


def _derive_dead_ends(trace: Trace) -> list[str]:
    dead: list[str] = []
    for rf in trace.repeated_failures:
        dead.append(f"Repeated failure pattern: {rf.signature}")
    return dead[:5]


def _derive_procedure(trace: Trace) -> list[str]:
    """Best-effort procedure derivation from successful actions."""
    steps: list[str] = []
    if trace.diff_summary:
        steps.append(f"Apply change: {trace.diff_summary}")
    for cmd in trace.commands_run:
        cmd_text = _trace_command_text(cmd).strip()
        if not cmd_text:
            continue
        suffix = ""
        if isinstance(cmd, CommandRecord) and cmd.exit_code is not None:
            suffix = f" (exit {cmd.exit_code})"
        steps.append(f"Run: {cmd_text}{suffix}")
    for validation in trace.validation_results:
        steps.append(f"Verify: {_trace_validation_text(validation)}")
    if trace.output_summary:
        steps.append(f"Confirm: {trace.output_summary}")
    return list(dict.fromkeys(steps))[:10]


def _trace_file_path(item: str | FileEditRecord) -> str:
    return item if isinstance(item, str) else item.path


def _trace_command_text(item: str | CommandRecord) -> str:
    return item if isinstance(item, str) else item.command


def _trace_tool_name(item: ToolCall) -> str:
    return item.name


def _trace_validation_text(item: ValidationResult) -> str:
    state = "passed" if item.passed else "failed"
    detail = f" ({item.detail})" if item.detail else ""
    return f"{item.name} {state}{detail}"


def _infer_command_tool(command: str) -> str:
    tokens = [token for token in re.split(r"\s+", command.strip()) if token]
    for token in tokens:
        base = Path(token).name
        if not base or base.startswith("-"):
            continue
        if base in _COMMAND_WRAPPER_TOKENS:
            continue
        return base
    return ""


def _derive_verification(trace: Trace) -> list[str]:
    out: list[str] = []
    for v in trace.validation_results:
        if v.passed:
            out.append(v.name)
    return out[:8]


def _short(text: str, n: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _score_confidence(trace: Trace) -> tuple[float, list[str]]:
    score = 0.40
    reasons = ["base score 0.40"]
    if trace.status == "success":
        score += 0.20
        reasons.append("trace status = success (+0.20)")
    passed = sum(1 for v in trace.validation_results if v.passed)
    val_bonus = min(0.20, 0.10 * passed)
    if val_bonus:
        score += val_bonus
        reasons.append(f"{passed} validations passed (+{val_bonus:.2f})")
    if trace.repeated_failures and trace.status == "success":
        score += 0.10
        reasons.append("recovered from repeated failure (+0.10)")
    if trace.files_touched and trace.validation_results:
        score += 0.10
        reasons.append("both files touched and validations present (+0.10)")
    return min(1.0, score), reasons
