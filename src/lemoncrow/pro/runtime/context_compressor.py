"""Compresses a run ledger into a tiny state packet for the next turn.

The compressor is the compact-state reducer in the spec. Instead of feeding
the next agent turn the entire raw transcript, we feed it:

  - the files changed (with most recent action per file)
  - the unique error fingerprints seen
  - the monitor alerts at >= medium severity
  - the current blocker, computed as the latest unresolved alert or the
    last failed command

This is enough for the next turn to make a coherent decision without
re-reading 50k tokens of tool output.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lemoncrow.infra.runtime.run_ledger import RunLedger

# ---------------------------------------------------------------------------
# Token-budget constants for recent-turns selection
# ---------------------------------------------------------------------------

# Task-type multipliers from route(op="decide") signals.
# debug/refactor sessions need more preserved context; explain/docs need less.
_TASK_TYPE_BUDGET_MULTIPLIER: dict[str, float] = {
    "debug": 1.6,
    "refactor": 1.4,
    "ops": 1.2,
    "test": 1.2,
    "feature": 1.0,
    "review": 0.75,
    "explain": 0.6,
    "docs": 0.5,
}

_RISK_LEVEL_BUDGET_MULTIPLIER: dict[str, float] = {
    "high": 1.3,
    "medium": 1.0,
    "low": 0.8,
}


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _load_compression_hints() -> dict[str, Any]:
    """Read model-provided compression hints from session state (fail-open)."""
    try:
        import os

        workspace = os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd())
        from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

        state_path = resolve_workspace_store_dir(workspace_root=Path(workspace)) / "session_state.json"
        if not state_path.exists():
            return {}
        state = json.loads(state_path.read_text("utf-8"))
        hints = state.get("compression_hints")
        return hints if isinstance(hints, dict) else {}
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}


@dataclass
class CompactState:
    files_changed: dict[str, str] = field(default_factory=dict)
    """Mapping of file path -> last action ('edit' or 'revert')."""
    error_fingerprints: list[str] = field(default_factory=list)
    high_severity_alerts: list[str] = field(default_factory=list)
    current_blocker: str | None = None
    tool_call_count: int = 0
    total_tool_output_chars: int = 0
    recent_turns: list[str] = field(default_factory=list)
    pinned_playbooks: list[str] = field(default_factory=list)
    claude_md_hash: str | None = None

    def to_prompt_block(self) -> str:
        lines: list[str] = ["## LemonCrow compact state"]
        if self.files_changed:
            lines.append("Files touched:")
            for path, action in self.files_changed.items():
                lines.append(f"  - {action}: {path}")
        if self.error_fingerprints:
            lines.append("Distinct errors seen:")
            for fp in self.error_fingerprints:
                lines.append(f"  - {fp}")
        if self.high_severity_alerts:
            lines.append("Active alerts:")
            for msg in self.high_severity_alerts:
                lines.append(f"  - {msg}")
        if self.current_blocker:
            lines.append(f"Current blocker: {self.current_blocker}")
        if self.pinned_playbooks:
            lines.append("Pinned Playbooks:")
            for block_id in self.pinned_playbooks:
                lines.append(f"  - {block_id}")
        if self.claude_md_hash:
            lines.append(f"CLAUDE.md sha256: {self.claude_md_hash}")
        if self.recent_turns:
            lines.append("Recent raw turns:")
            for turn in self.recent_turns:
                lines.append(f"  - {turn}")
        lines.append(f"Stats: tool_calls={self.tool_call_count} output_chars={self.total_tool_output_chars}")
        return "\n".join(lines)


@dataclass
class HandoverPacket:
    session_id: str
    goal: str
    progress: str
    decisions_made: list[str] = field(default_factory=list)
    files_changed: dict[str, str] = field(default_factory=dict)
    active_errors: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)

    @classmethod
    def from_ledger(
        cls,
        ledger: RunLedger,
        compact_state: CompactState,
        *,
        workspace_root: Path | None = None,
    ) -> HandoverPacket:
        decisions = _dedupe_preserve_order(
            [
                *ledger.verified_facts,
                *(f"hypothesis accepted: {h}" for h in ledger.hypotheses_tried[-5:]),
                *(f"hypothesis rejected: {h}" for h in ledger.hypotheses_rejected[-5:]),
            ]
        )
        next_steps = _extract_next_steps(ledger)
        context = _handover_context(ledger, compact_state, workspace_root=workspace_root)
        return cls(
            session_id=ledger.session_id,
            goal=ledger.task or "Continue the current LemonCrow session.",
            progress=compact_state.to_prompt_block(),
            decisions_made=decisions,
            files_changed=compact_state.files_changed,
            active_errors=compact_state.error_fingerprints,
            next_steps=next_steps,
            context=context,
        )

    def to_markdown(self, *, max_chars: int = 20_000) -> str:
        lines = [
            f"## Session Handover - {self.session_id}",
            f"### Goal: {self.goal}",
            "",
            "### Progress",
            _truncate(self.progress, 4_000),
            "",
            "### Decisions made",
            *_markdown_list(self.decisions_made),
            "",
            "### Files changed",
            *_markdown_list([f"{action}: {path}" for path, action in self.files_changed.items()]),
            "",
            "### Active errors",
            *_markdown_list(self.active_errors),
            "",
            "### Next steps",
            *_markdown_list(self.next_steps),
            "",
            "### Context",
            *_markdown_list(self.context),
        ]
        return _truncate("\n".join(lines).rstrip() + "\n", max_chars)


class ContextCompressor:
    def compress(
        self,
        ledger: RunLedger,
        *,
        preserve_last_n_turns: int = 10,
        workspace_root: Path | None = None,
    ) -> CompactState:
        files: dict[str, str] = {}
        errors: list[str] = []
        seen_errors: set[str] = set()
        alerts: list[str] = []
        last_failed_cmd: str | None = None

        for event in ledger.events:
            if event.kind in ("file_edit", "file_revert"):
                path = str(event.payload.get("path", ""))
                action = "revert" if event.kind == "file_revert" else "edit"
                if path:
                    files[path] = action
            elif event.kind == "command_result":
                ok = bool(event.payload.get("ok"))
                err = str(event.payload.get("error_signature", "")).strip()
                if not ok:
                    last_failed_cmd = event.summary
                    if err and err not in seen_errors:
                        seen_errors.add(err)
                        errors.append(err)
            elif event.kind == "watchdog_alert":
                sev = str(event.payload.get("severity", ""))
                if sev in ("medium", "high"):
                    alerts.append(event.summary)

        blocker: str | None = None
        if alerts:
            blocker = alerts[-1]
        elif last_failed_cmd:
            blocker = f"last failed command: {last_failed_cmd}"

        tool_calls = [e for e in ledger.events if e.kind == "tool_call"]
        total_chars = sum(int(e.payload.get("output_chars", 0)) for e in tool_calls)

        return CompactState(
            files_changed=files,
            error_fingerprints=errors,
            high_severity_alerts=alerts,
            current_blocker=blocker,
            tool_call_count=len(tool_calls),
            total_tool_output_chars=total_chars,
            recent_turns=_recent_raw_turns(ledger, preserve_last_n_turns),
            pinned_playbooks=list(dict.fromkeys(ledger.active_playbooks)),
            claude_md_hash=_claude_md_hash(workspace_root),
        )


def _event_dict(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        raw = event.model_dump(mode="json")
        return raw if isinstance(raw, dict) else {}
    if isinstance(event, dict):
        return event
    return {"summary": str(event)}


_RECENT_TURNS_TOKEN_BUDGET = 12_000  # default target for compact recent-turn retention
_RECENT_TURNS_TOKEN_BUDGET_MIN = 8_000  # floor for read-only sessions
_RECENT_TURNS_TOKEN_BUDGET_MAX = 24_000  # ceiling for heavy edit/debug sessions
_CHARS_PER_TOKEN = 4  # rough estimate used for budget enforcement

# How far back we look when scoring session complexity (in events)
_COMPLEXITY_WINDOW = 40

# Per-event value scores used by the dynamic budget and priority selection.
# Higher score = more important to preserve verbatim.
_EVENT_SCORES: dict[str, float] = {
    "file_edit": 3.0,  # code change - essential
    "file_revert": 2.5,  # revert - also important
    "test_result": 2.0,  # test outcome
    "command_result": 1.0,  # adjusted further by ok/fail below
    "reasoning": 1.5,  # model reasoning - valuable
    "agent_message": 1.5,  # model output - valuable
    "tool_result": 0.8,
    "tool_call": 0.5,
    "note": 1.2,
    "validation": 1.8,
    "watchdog_alert": 2.0,
    "model_recommendation": 0.2,
}


def _score_event(event: dict[str, object]) -> float:
    """Return a 0-4 value score for a single ledger event dict.

    Higher = more important to preserve during compaction.
    """
    kind = str(event.get("kind", ""))
    base = _EVENT_SCORES.get(kind, 0.5)

    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        if kind == "command_result" and not payload.get("ok", True):
            base = 2.8  # failed command = high-value debugging context
        if kind == "test_result" and not payload.get("passed", True):
            base = 2.5  # failing test = important
        # Long summaries carry more reasoning content
        output_chars = int(payload.get("output_chars", 0))
        if output_chars > 3_000:
            base += 0.5
        elif output_chars > 800:
            base += 0.2

    # Penalise very verbose low-signal events (e.g. huge read outputs)
    summary_len = len(str(event.get("summary", "")))
    if summary_len > 1_000 and base < 1.5:
        base *= 0.5  # halve score for bulky cheap events

    return base


def _dynamic_turn_budget(
    events: list[dict[str, object]],
    hints: dict[str, Any] | None = None,
) -> int:
    """Compute a session-adaptive token budget for preserving recent turns.

    The budget scales with session complexity so that:
    - Heavy edit / debugging sessions get more preserved context (up to 40 K).
    - Read-only or quiet sessions use a tighter budget (~8-12 K).

    Factors
    -------
    edit_density:
        Fraction of the recent window that are ``file_edit``/``file_revert``
        events.  High edit density -> more history needed (model must know
        what it already changed).
    error_density:
        Fraction of ``command_result`` and ``test_result`` events that
        failed.  High error density -> debugging requires more context.
    avg_event_verbosity:
        Mean chars per event summary in the window.  Very verbose events
        contain richer content; the budget should scale up to preserve them.

    LLM signals (from session state via ``hints``):
        task_type: scales by _TASK_TYPE_BUDGET_MULTIPLIER (debug=1.6 -> docs=0.5).
        risk_level: scales by _RISK_LEVEL_BUDGET_MULTIPLIER (high=1.3, low=0.8).
        model_complexity: 0.0-1.0 -> maps to 0.7x-1.5x multiplier.

    Formula
    -------
    budget = base(10K)
           + edit_bonus   (0-18K  scaled by edit_density)
           + error_bonus  (0-8K   scaled by error_density)
           + verbose_bonus(0-4K   scaled by avg event verbosity)
    Then scaled by LLM signals.
    Clamped to [_RECENT_TURNS_TOKEN_BUDGET_MIN, _RECENT_TURNS_TOKEN_BUDGET_MAX].
    """
    if not events:
        return _RECENT_TURNS_TOKEN_BUDGET

    window = events[-_COMPLEXITY_WINDOW:]

    # Edit density
    edit_count = sum(1 for e in window if str(e.get("kind", "")) in {"file_edit", "file_revert"})
    edit_density = edit_count / len(window)

    # Error density
    outcome_events = [e for e in window if str(e.get("kind", "")) in {"command_result", "test_result"}]
    if outcome_events:

        def _is_failure(ev: dict[str, object]) -> bool:
            raw = ev.get("payload")
            p: dict[str, object] = raw if isinstance(raw, dict) else {}
            ok = p.get("ok")
            passed = p.get("passed")
            if ok is not None:
                return not ok
            if passed is not None:
                return not passed
            return False

        fail_count = sum(1 for e in outcome_events if _is_failure(e))
        error_density = fail_count / len(outcome_events)
    else:
        error_density = 0.0

    # Verbosity: average chars per event
    avg_chars = sum(len(str(e.get("summary", ""))) for e in window) / len(window)

    base = 8_000
    edit_bonus = int(edit_density * 10_000)
    error_bonus = int(error_density * 4_000)
    verbose_bonus = int(min(avg_chars / 400, 1.0) * 2_000)

    budget = base + edit_bonus + error_bonus + verbose_bonus

    # ── LLM-signal multipliers ────────────────────────────────────────────
    applied_hints = hints if hints is not None else _load_compression_hints()

    task_type = str(applied_hints.get("task_type", "feature")).lower()
    task_mult = _TASK_TYPE_BUDGET_MULTIPLIER.get(task_type, 1.0)

    risk_level = str(applied_hints.get("risk_level", "medium")).lower()
    risk_mult = _RISK_LEVEL_BUDGET_MULTIPLIER.get(risk_level, 1.0)

    # model_complexity: 0.0-1.0 -> maps to 0.7x - 1.5x budget multiplier
    raw_complexity = applied_hints.get("model_complexity")
    complexity_mult = 0.7 + float(raw_complexity) * 0.8 if raw_complexity is not None else 1.0

    budget = int(budget * task_mult * risk_mult * complexity_mult)
    return max(_RECENT_TURNS_TOKEN_BUDGET_MIN, min(budget, _RECENT_TURNS_TOKEN_BUDGET_MAX))


def _recent_raw_turns(ledger: RunLedger, limit: int) -> list[str]:
    """Return recent turn-like events as readable ``[kind] summary`` strings.

    The selection is **complexity-adaptive**:

    1. *Dynamic budget* - token budget scales with session complexity
       (edit density, error rate, verbosity) and LLM-provided signals
       (task_type, risk_level, model_complexity from session state) so
       debugging/editing sessions preserve more history than quiet read-only
       sessions.

    2. *Priority filtering* - when the budget is nearly exhausted, cheap
       bulky events (low-value Read/Bash outputs) are skipped so that
       high-value events (edits, failures, reasoning) are never displaced.

    3. *Extended lookback* - the candidate window grows with the budget
       so complex sessions can draw on a deeper history.

    4. *Must-keep boosting* - events whose summary matches a keyword stored
       by the model via ``compact(op="score")`` always pass through regardless
       of budget pressure (score boosted to 4.0).
    """
    if limit <= 0:
        return []
    raw_events = [_event_dict(event) for event in ledger.events]

    # Load hints once - used for budget and must_keep boosting
    hints = _load_compression_hints()
    must_keep_lower = [kw.lower() for kw in (hints.get("must_keep") or [])]

    turn_like = [
        event
        for event in raw_events
        if str(event.get("kind", "")) in {"agent_message", "reasoning", "command_result", "test_result", "tool_result"}
    ]
    # Extend lookback proportionally: complex sessions need a deeper window
    dyn_budget = _dynamic_turn_budget(raw_events, hints=hints)
    extended_limit = max(limit, int(limit * dyn_budget / _RECENT_TURNS_TOKEN_BUDGET))
    candidates = (turn_like or raw_events)[-extended_limit:]

    char_budget = dyn_budget * _CHARS_PER_TOKEN
    lines: list[str] = []
    chars_used = 0

    for event in reversed(candidates):
        kind = str(event.get("kind", "event"))
        summary = str(event.get("summary", "")).strip()
        payload = event.get("payload", {})
        if kind == "command_result":
            ok = payload.get("ok", "?") if isinstance(payload, dict) else "?"
            summary = f"{'✓' if ok else '✗'} {summary}".strip()
        elif kind == "test_result":
            passed = payload.get("passed", "?") if isinstance(payload, dict) else "?"
            summary = f"{'✓' if passed else '✗'} {summary}".strip()
        line = f"[{kind}] {summary}" if summary else f"[{kind}]"
        line_chars = len(line)

        # Must-keep boosting: if model tagged this event's topic as essential,
        # guarantee inclusion regardless of budget pressure.
        score = _score_event(event)
        if must_keep_lower:
            summary_lower = summary.lower()
            if any(kw in summary_lower for kw in must_keep_lower):
                score = 4.0  # always passes the gate below

        # Priority gate: when >70% of budget is used, skip low-value bulky events
        # (verbose read/command outputs that aren't failures).
        # High-value events (score >= 2.0) always pass through.
        if score < 4.0 and chars_used > char_budget * 0.7 and score < 1.5 and line_chars > 300:
            continue

        # Must-keep events always included; others respect the budget cap.
        if score < 4.0:
            chars_used += line_chars
            if chars_used > char_budget:
                break
        lines.append(line)

    lines.reverse()
    return lines


def _claude_md_hash(workspace_root: Path | None) -> str | None:
    roots: list[Path] = []
    if workspace_root is not None:
        roots.append(workspace_root)
    roots.append(Path.cwd())
    for root in roots:
        path = root / "CLAUDE.md"
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return None


def _claude_md_excerpt(workspace_root: Path | None) -> str | None:
    roots: list[Path] = []
    if workspace_root is not None:
        roots.append(workspace_root)
    roots.append(Path.cwd())
    for root in roots:
        path = root / "CLAUDE.md"
        if path.is_file():
            return f"CLAUDE.md excerpt:\n{_truncate(path.read_text(encoding='utf-8', errors='replace'), 800)}"
    return None


def _extract_next_steps(ledger: RunLedger) -> list[str]:
    if ledger.next_required_validation:
        return [ledger.next_required_validation]
    for event in reversed(ledger.events):
        data = _event_dict(event)
        text = " ".join([str(data.get("summary", "")), json.dumps(data.get("payload", {}), default=str)])
        lowered = text.lower()
        if "next" in lowered or "todo" in lowered:
            return [_truncate(text, 500)]
    return ["Continue from the current compact state and resolve any active errors first."]


def _handover_context(
    ledger: RunLedger,
    compact_state: CompactState,
    *,
    workspace_root: Path | None,
) -> list[str]:
    context: list[str] = []
    if compact_state.claude_md_hash:
        context.append(f"CLAUDE.md sha256: {compact_state.claude_md_hash}")
    excerpt = _claude_md_excerpt(workspace_root)
    if excerpt:
        context.append(excerpt)
    file_edit_context: list[str] = []
    for event in reversed(ledger.events):
        data = _event_dict(event)
        if str(data.get("kind", "")) != "file_edit":
            continue
        payload = data.get("payload", {})
        if isinstance(payload, dict) and payload.get("diff"):
            file_edit_context.append(
                _truncate(
                    f"Snippet for {payload.get('path', 'unknown')}:\n{payload.get('diff', '')}",
                    800,
                )
            )
        if len(file_edit_context) >= 2:
            break
    context.extend(reversed(file_edit_context))
    context.extend(f"Recent turn: {turn}" for turn in compact_state.recent_turns[-2:])
    return context


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _markdown_list(items: list[str]) -> list[str]:
    if not items:
        return ["- None recorded."]
    return [f"- {_truncate(item, 1_000)}" for item in items]


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    head = value[: max_chars // 2].rstrip()
    tail = value[-max_chars // 2 :].lstrip()
    return f"{head}\n...<truncated>...\n{tail}"
