"""Runtime watchdogs that watch agent execution and trigger rescue logic.

Six watchdogs are implemented:

1. RepeatedCommandFailure — same command/test fails twice with the same
   error signature.
2. RepeatedToolCall — same tool called 3+ times with similar args.
3. KnownDeadEnd — agent plan or tool args contain a known dead-end phrase.
4. SkippedVerification — agent attempts to mark success without verification.
5. ContextBloat — trace contains repeated logs / large stale tool output.
6. HighRiskAction — high-risk tool used without a matching rubric.

Each watchdog implements `check(state) -> WatchdogAlert | None`.
The watchdogs are deliberately stateless w.r.t. each other; the caller
maintains the running session state and feeds it to all watchdogs.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from lemoncrow.core.foundation.models import Playbook, Severity


def _step_matches_phrase(step: str, phrase: str) -> bool:
    """True if step text matches a phrase (case-insensitive fuzzy match)."""
    # Simple normalization: lower, strip punctuation
    s = re.sub(r"[^\w\s]", "", step.lower())
    p = re.sub(r"[^\w\s]", "", phrase.lower())
    return p in s


# --------------------------------------------------------------------------- #
# Session state                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class SessionState:
    """In-flight state used by watchdogs during a single agent run."""

    domain: str | None = None
    plan: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    command_results: list[tuple[str, bool, str]] = field(default_factory=list)
    """Tuples of (command, succeeded, error_signature)."""
    tool_calls: list[tuple[str, str]] = field(default_factory=list)
    """Tuples of (tool_name, args_signature)."""
    tool_call_args: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)
    """Tuples of (tool_name, args_dict)."""
    tool_call_results: list[tuple[str, str]] = field(default_factory=list)
    """Tuples of (tool_name, result_summary)."""
    tool_outputs_chars: int = 0
    rubric_run: bool = False
    declared_success: bool = False
    validation_passed: bool = False
    file_events: list[tuple[str, str]] = field(default_factory=list)
    """Tuples of (path, action) where action in {'edit', 'revert'}."""
    file_diffs: list[tuple[str, str, str]] = field(default_factory=list)
    """Tuples of (path, action, diff_text)."""
    command_outputs: list[tuple[str, int | None, str, str]] = field(default_factory=list)
    """Tuples of (command, exit_code, stdout, stderr)."""
    estimated_tokens: int = 0
    budget_max_tool_calls: int | None = None
    budget_max_repeated_commands: int | None = None
    budget_max_estimated_tokens: int | None = None


@dataclass
class WatchdogAlert:
    watchdog: str
    severity: Severity
    message: str
    suggestion: str = ""


@dataclass(frozen=True)
class WatchdogDefinition:
    key: str
    title: str
    description: str
    default_weight: float
    severity: Severity


@dataclass(frozen=True)
class WatchdogProfileDefinition:
    id: str
    label: str
    description: str
    weights: dict[str, float]


# --------------------------------------------------------------------------- #
# Protocol                                                                    #
# --------------------------------------------------------------------------- #


class Watchdog(Protocol):
    name: str

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None: ...


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def error_signature(text: str) -> str:
    """Stable signature for an error string (volatile parts stripped)."""
    norm = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    norm = re.sub(r"\b\d+\b", "N", norm)
    norm = re.sub(r"\s+", " ", norm).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def args_signature(args: dict[str, Any] | str | None) -> str:
    if not args:
        return "()"
    if not isinstance(args, dict):
        return str(args)
    pairs = sorted((k, str(v)) for k, v in args.items())
    blob = "|".join(f"{k}={v}" for k, v in pairs)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Watchdogs                                                                    #
# --------------------------------------------------------------------------- #


class RepeatedCommandFailure:
    name = "repeated_command_failure"

    def __init__(self, failure_threshold: int = 2) -> None:
        self.failure_threshold = failure_threshold

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        sigs = [sig for _, ok, sig in state.command_results if not ok]
        counts = Counter(sigs)
        for sig, n in counts.items():
            if n >= self.failure_threshold:
                return WatchdogAlert(
                    watchdog=self.name,
                    severity="high",
                    message=f"Same command failed {n}x with error signature {sig}.",
                    suggestion=(
                        "Stop retrying. Search Playbooks for this failure mode "
                        "and adjust the approach before re-running."
                    ),
                )
        return None


class RepeatedToolCall:
    name = "repeated_tool_call"

    def __init__(self, repeat_threshold: int = 3) -> None:
        self.repeat_threshold = repeat_threshold

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        counts = Counter(state.tool_calls)
        for (tool, sig), n in counts.items():
            if n >= self.repeat_threshold:
                return WatchdogAlert(
                    watchdog=self.name,
                    severity="medium",
                    message=f"Tool {tool!r} called {n}x with same args signature {sig}.",
                    suggestion=(
                        "Probable tight loop. Summarize the invariant the agent "
                        "is fighting and request a different procedure."
                    ),
                )
        return None


class KnownDeadEnd:
    name = "known_dead_end"

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        for block in blocks:
            for dead in block.dead_ends:
                for step in state.plan:
                    if _step_matches_phrase(step, dead):
                        return WatchdogAlert(
                            watchdog=self.name,
                            severity="high",
                            message=f"Plan contains known dead end: {dead!r}",
                            suggestion=(f"Apply procedure from Playbook '{block.title}' instead."),
                        )
        return None


class SkippedVerification:
    name = "skipped_verification"

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        if state.declared_success and not state.validation_passed:
            return WatchdogAlert(
                watchdog=self.name,
                severity="high",
                message="Agent declared success without verified validation.",
                suggestion=("Run the rubric gate before accepting the result. No success without validation."),
            )
        return None


class ContextBloat:
    name = "context_bloat"

    def __init__(self, threshold_chars: int = 50_000) -> None:
        self.threshold_chars = threshold_chars

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        if state.tool_outputs_chars > self.threshold_chars:
            return WatchdogAlert(
                watchdog=self.name,
                severity="medium",
                message=(f"Tool outputs accumulated {state.tool_outputs_chars} chars. Likely stale repeated logs."),
                suggestion=("Compress trace to: files changed, errors seen, assumptions tested, current blocker."),
            )
        return None


HIGH_RISK_TOOLS = {
    "shopify.update_metafield",
    "shopify.publish",
    "shopify.product.update",
    "schema.validate",
    "tracker.classify",
    "catalog.write",
    "pdp.publish",
}


class HighRiskAction:
    name = "high_risk_action"

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        for tool, _ in state.tool_calls:
            if tool in HIGH_RISK_TOOLS and not state.rubric_run:
                return WatchdogAlert(
                    watchdog=self.name,
                    severity="high",
                    message=f"High-risk tool {tool!r} used without a rubric gate.",
                    suggestion="Run the matching rubric gate before accepting this action.",
                )
        return None


class SecondGuessing:
    """Detect patch-revert-repatch cycles on the same file without new evidence."""

    name = "second_guessing"

    def __init__(self, cycle_threshold: int = 1) -> None:
        self.cycle_threshold = cycle_threshold

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        per_file: dict[str, list[str]] = {}
        for path, action in state.file_events:
            per_file.setdefault(path, []).append(action)
        for path, actions in per_file.items():
            # edit -> revert -> edit on same file
            cycles = 0
            for i in range(len(actions) - 2):
                if actions[i] == "edit" and actions[i + 1] == "revert" and actions[i + 2] == "edit":
                    cycles += 1
                    if cycles >= self.cycle_threshold:
                        return WatchdogAlert(
                            watchdog=self.name,
                            severity="medium",
                            message=f"File {path!r} edited, reverted, edited again.",
                            suggestion=(
                                "Reset hypothesis. State the current assumption, the rejected "
                                "assumptions, and the next distinct strategy before editing again."
                            ),
                        )
        return None


class BudgetExhaustion:
    """Fire when configured budgets are exceeded."""

    name = "budget_exhaustion"

    def check(self, state: SessionState, blocks: Sequence[Playbook]) -> WatchdogAlert | None:
        if state.budget_max_tool_calls is not None and len(state.tool_calls) > state.budget_max_tool_calls:
            return WatchdogAlert(
                watchdog=self.name,
                severity="high",
                message=(f"Tool call count {len(state.tool_calls)} exceeds budget {state.budget_max_tool_calls}."),
                suggestion="Summarize-and-plan before continuing.",
            )
        if state.budget_max_repeated_commands is not None:
            counts = Counter(c for c, _, _ in state.command_results)
            for cmd, n in counts.items():
                if n > state.budget_max_repeated_commands:
                    return WatchdogAlert(
                        watchdog=self.name,
                        severity="high",
                        message=(f"Command {cmd!r} repeated {n}x exceeds budget {state.budget_max_repeated_commands}."),
                        suggestion="Summarize-and-plan before continuing.",
                    )
        if state.budget_max_estimated_tokens is not None and state.estimated_tokens > state.budget_max_estimated_tokens:
            return WatchdogAlert(
                watchdog=self.name,
                severity="high",
                message=(
                    f"Estimated tokens {state.estimated_tokens} exceeds budget {state.budget_max_estimated_tokens}."
                ),
                suggestion="Summarize-and-plan before continuing.",
            )
        return None


WATCHDOG_LIBRARY: tuple[WatchdogDefinition, ...] = (
    WatchdogDefinition(
        key="repeated_command_failure",
        title="Repeated command failure",
        description="Same command or test fails repeatedly with the same error signature.",
        default_weight=0.30,
        severity="high",
    ),
    WatchdogDefinition(
        key="repeated_tool_call",
        title="Repeated tool call",
        description="Same tool is called repeatedly with effectively the same args.",
        default_weight=0.18,
        severity="medium",
    ),
    WatchdogDefinition(
        key="known_dead_end",
        title="Known dead end",
        description="A plan step overlaps with a known dead end from the reasoning library.",
        default_weight=0.22,
        severity="high",
    ),
    WatchdogDefinition(
        key="skipped_verification",
        title="Skipped verification",
        description="The agent declares success without verified validation.",
        default_weight=0.24,
        severity="high",
    ),
    WatchdogDefinition(
        key="context_bloat",
        title="Context bloat",
        description="Tool output or stale log volume grows large enough to dilute useful context.",
        default_weight=0.12,
        severity="medium",
    ),
    WatchdogDefinition(
        key="high_risk_action",
        title="High-risk action",
        description="A risky tool path is used without a matching rubric gate.",
        default_weight=0.28,
        severity="high",
    ),
    WatchdogDefinition(
        key="second_guessing",
        title="Second guessing",
        description="A file gets edited, reverted, then edited again without new evidence.",
        default_weight=0.16,
        severity="medium",
    ),
    WatchdogDefinition(
        key="budget_exhaustion",
        title="Budget exhaustion",
        description="Configured tool-call or token budgets are exceeded.",
        default_weight=0.14,
        severity="medium",
    ),
)


WATCHDOG_PROFILES: tuple[WatchdogProfileDefinition, ...] = (
    WatchdogProfileDefinition(
        id="coding",
        label="Coding",
        description="Prioritize command retries, verification gaps, and risky edits.",
        weights={
            "repeated_command_failure": 0.30,
            "repeated_tool_call": 0.16,
            "known_dead_end": 0.16,
            "skipped_verification": 0.20,
            "context_bloat": 0.08,
            "high_risk_action": 0.14,
            "second_guessing": 0.12,
            "budget_exhaustion": 0.08,
        },
    ),
    WatchdogProfileDefinition(
        id="review",
        label="Review",
        description="Bias toward dead ends, skipped verification, and unsafe approval paths.",
        weights={
            "repeated_command_failure": 0.16,
            "repeated_tool_call": 0.10,
            "known_dead_end": 0.24,
            "skipped_verification": 0.24,
            "context_bloat": 0.08,
            "high_risk_action": 0.20,
            "second_guessing": 0.08,
            "budget_exhaustion": 0.06,
        },
    ),
    WatchdogProfileDefinition(
        id="qa",
        label="QA",
        description="Push verification and command stability to the top.",
        weights={
            "repeated_command_failure": 0.28,
            "repeated_tool_call": 0.08,
            "known_dead_end": 0.14,
            "skipped_verification": 0.28,
            "context_bloat": 0.06,
            "high_risk_action": 0.18,
            "second_guessing": 0.06,
            "budget_exhaustion": 0.08,
        },
    ),
    WatchdogProfileDefinition(
        id="research",
        label="Research",
        description="Allow broader exploration while still catching loops and late sprawl.",
        weights={
            "repeated_command_failure": 0.18,
            "repeated_tool_call": 0.22,
            "known_dead_end": 0.14,
            "skipped_verification": 0.10,
            "context_bloat": 0.16,
            "high_risk_action": 0.08,
            "second_guessing": 0.14,
            "budget_exhaustion": 0.14,
        },
    ),
)


def watchdog_library() -> list[WatchdogDefinition]:
    return list(WATCHDOG_LIBRARY)


def builtin_watchdog_profiles() -> list[WatchdogProfileDefinition]:
    return [
        WatchdogProfileDefinition(
            id=profile.id,
            label=profile.label,
            description=profile.description,
            weights=dict(profile.weights),
        )
        for profile in WATCHDOG_PROFILES
    ]


def default_watchdog_profile_id() -> str:
    return WATCHDOG_PROFILES[0].id


def default_watchdog_weights() -> dict[str, float]:
    return {definition.key: definition.default_weight for definition in WATCHDOG_LIBRARY}


def normalize_watchdog_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    normalized = default_watchdog_weights()
    if weights is None:
        return normalized
    for key in normalized:
        value = weights.get(key)
        if isinstance(value, (int, float)):
            normalized[key] = min(1.0, max(0.0, float(value)))
    return normalized


def build_watchdogs(weights: Mapping[str, float] | None = None) -> list[Watchdog]:
    normalized = normalize_watchdog_weights(weights)
    ordered: list[tuple[float, int, Watchdog]] = []
    for index, definition in enumerate(WATCHDOG_LIBRARY):
        weight = normalized.get(definition.key, definition.default_weight)
        if weight <= 0:
            continue
        ordered.append((weight, index, _build_watchdog(definition.key, weight)))
    ordered.sort(key=lambda item: (-item[0], item[1]))
    return [watchdog for _, _, watchdog in ordered]


def default_watchdogs(weights: Mapping[str, float] | None = None) -> list[Watchdog]:
    return build_watchdogs(weights)


def _build_watchdog(key: str, weight: float) -> Watchdog:
    if key == "repeated_command_failure":
        return RepeatedCommandFailure(failure_threshold=2 if weight >= 0.16 else 3)
    if key == "repeated_tool_call":
        if weight >= 0.18:
            threshold = 2
        elif weight <= 0.09:
            threshold = 4
        else:
            threshold = 3
        return RepeatedToolCall(repeat_threshold=threshold)
    if key == "known_dead_end":
        return KnownDeadEnd()
    if key == "skipped_verification":
        return SkippedVerification()
    if key == "context_bloat":
        if weight >= 0.14:
            threshold_chars = 40_000
        elif weight <= 0.08:
            threshold_chars = 75_000
        else:
            threshold_chars = 50_000
        return ContextBloat(threshold_chars=threshold_chars)
    if key == "high_risk_action":
        return HighRiskAction()
    if key == "second_guessing":
        return SecondGuessing(cycle_threshold=1 if weight >= 0.10 else 2)
    if key == "budget_exhaustion":
        return BudgetExhaustion()
    raise KeyError(f"unknown watchdog key: {key}")


def run_watchdogs(
    state: SessionState,
    blocks: Sequence[Playbook],
    watchdogs: Sequence[Watchdog] | None = None,
) -> list[WatchdogAlert]:
    watchdogs = watchdogs or default_watchdogs()
    alerts: list[WatchdogAlert] = []
    for m in watchdogs:
        alert = m.check(state, blocks)
        if alert is not None:
            alerts.append(alert)
    return alerts
