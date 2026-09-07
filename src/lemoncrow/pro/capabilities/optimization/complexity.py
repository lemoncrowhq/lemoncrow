"""Complexity scoring for optimization routing decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from lemoncrow.core.foundation.models import CommandRecord, FileEditRecord, Trace

ComplexityLabel = Literal["simple", "medium", "hard"]


@dataclass(frozen=True)
class ComplexitySignals:
    task_type_weight: float
    repo_context_size: float
    user_intent_risk: float
    test_failure_signal: float
    dependency_graph_depth: float
    ambiguity_score: float
    prior_failure_count: float
    required_tool_count: float

    def to_dict(self) -> dict[str, float]:
        return {
            "task_type_weight": self.task_type_weight,
            "repo_context_size": self.repo_context_size,
            "user_intent_risk": self.user_intent_risk,
            "test_failure_signal": self.test_failure_signal,
            "dependency_graph_depth": self.dependency_graph_depth,
            "ambiguity_score": self.ambiguity_score,
            "prior_failure_count": self.prior_failure_count,
            "required_tool_count": self.required_tool_count,
        }


@dataclass(frozen=True)
class ComplexityScore:
    score: float
    label: ComplexityLabel
    components: ComplexitySignals

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "components": self.components.to_dict(),
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _task_type_weight(text: str) -> float:
    lowered = text.lower()
    if any(word in lowered for word in ("migration", "migrate", "schema", "security", "auth", "permission")):
        return 0.95
    if any(word in lowered for word in ("refactor", "rewrite", "architecture", "redesign")):
        return 0.70
    if any(word in lowered for word in ("bug", "fix", "failing", "failed", "broken", "regression")):
        return 0.60
    if any(word in lowered for word in ("explain", "summarize", "summary", "docs", "document")):
        return 0.20
    return 0.45


def _risk(text: str) -> float:
    lowered = text.lower()
    keywords = (
        "production",
        "security",
        "migration",
        "urgent",
        "payment",
        "auth",
        "data loss",
        "delete",
    )
    hits = sum(1 for keyword in keywords if keyword in lowered)
    return _clamp(hits / 3.0)


def _path_string(item: str | FileEditRecord) -> str:
    return item.path if isinstance(item, FileEditRecord) else str(item)


def _command_failed(command: str | CommandRecord) -> bool:
    if isinstance(command, CommandRecord):
        return command.exit_code is not None and command.exit_code != 0
    lowered = str(command).lower()
    return "failed" in lowered or "traceback" in lowered or "error" in lowered


def _label(score: float) -> ComplexityLabel:
    if score < 0.33:
        return "simple"
    if score < 0.67:
        return "medium"
    return "hard"


def score_complexity(
    *,
    task: str,
    files_touched: int = 0,
    distinct_modules: int = 0,
    failed_tests_or_commands: int = 0,
    prior_failures: int = 0,
    required_tools: int = 0,
) -> ComplexityScore:
    question_marks = task.count("?")
    signals = ComplexitySignals(
        task_type_weight=_task_type_weight(task),
        repo_context_size=_clamp(files_touched / 50.0),
        user_intent_risk=_risk(task),
        test_failure_signal=1.0 if failed_tests_or_commands > 0 else 0.0,
        dependency_graph_depth=_clamp(distinct_modules / 6.0),
        ambiguity_score=_clamp((len(task.strip()) / 800.0) + (question_marks * 0.10)),
        prior_failure_count=_clamp(prior_failures / 5.0),
        required_tool_count=_clamp(required_tools / 10.0),
    )
    score = _clamp(
        (0.20 * signals.task_type_weight)
        + (0.15 * signals.repo_context_size)
        + (0.10 * signals.user_intent_risk)
        + (0.15 * signals.test_failure_signal)
        + (0.10 * signals.dependency_graph_depth)
        + (0.10 * signals.ambiguity_score)
        + (0.10 * signals.prior_failure_count)
        + (0.10 * signals.required_tool_count)
    )
    return ComplexityScore(score=round(score, 4), label=_label(score), components=signals)


def score_trace_complexity(trace: Trace) -> ComplexityScore:
    paths = [_path_string(item) for item in trace.files_touched]
    modules = {path.split("/", 1)[0] for path in paths if path}
    failed_validations = sum(1 for validation in trace.validation_results if not validation.passed)
    failed_commands = sum(1 for command in trace.commands_run if _command_failed(command))
    tool_names = {tool.name for tool in trace.tools_called}
    return score_complexity(
        task=trace.task,
        files_touched=len(paths),
        distinct_modules=len(modules),
        failed_tests_or_commands=failed_validations + failed_commands + len(trace.errors_seen),
        prior_failures=len(trace.repeated_failures) + len(trace.errors_seen),
        required_tools=len(tool_names) + len(trace.commands_run),
    )
