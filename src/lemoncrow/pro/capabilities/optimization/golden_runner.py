"""Golden optimization corpus loader.

The v0 runner validates that the committed corpus is present and well-formed.
It deliberately does not invoke paid model replays; the optimizer can surface
the corpus status without spending money during an advisory CLI call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.optimization.policy import Policy


@dataclass(frozen=True)
class GoldenSuiteResult:
    total: int
    passed: int
    score: float
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "score": self.score,
            "failures": list(self.failures),
        }


def default_golden_dir() -> Path:
    return Path(__file__).resolve().parents[5] / "tests" / "golden" / "optimization"


def _validate_case(payload: object, source: Path, line_number: int) -> str | None:
    if not isinstance(payload, dict):
        return f"{source}:{line_number}: expected object"
    for key in ("task_id", "complexity_label", "messages", "success_criteria"):
        if key not in payload:
            return f"{source}:{line_number}: missing {key}"
    if payload["complexity_label"] not in {"simple", "medium", "hard"}:
        return f"{source}:{line_number}: invalid complexity_label"
    if not isinstance(payload["messages"], list) or not payload["messages"]:
        return f"{source}:{line_number}: messages must be a non-empty list"
    if not isinstance(payload["success_criteria"], dict):
        return f"{source}:{line_number}: success_criteria must be an object"
    return None


def run_golden_suite(policy: Policy, corpus_dir: Path | None = None) -> GoldenSuiteResult:
    del policy
    root = corpus_dir or default_golden_dir()
    if not root.exists():
        return GoldenSuiteResult(total=0, passed=0, score=0.0, failures=[f"{root} does not exist"])
    failures: list[str] = []
    total = 0
    for path in sorted(root.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            total += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
                continue
            failure = _validate_case(payload, path, line_number)
            if failure is not None:
                failures.append(failure)
    passed = total - len(failures)
    score = (passed / total) if total else 0.0
    return GoldenSuiteResult(total=total, passed=passed, score=round(score, 4), failures=failures)
