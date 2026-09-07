"""Verifier capability orchestrator (M3).

Runs the deterministic checks over the files an agent touched and returns
structured counterexamples. Host-agnostic: it does not drive a retry loop — it
produces the signal the host (or M5's PostToolUse choreography) feeds back.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .checks import CommandRunner, run_lint, run_semantic_review, run_tests, run_typecheck
from .counterexample import Counterexample

_DEFAULT_CHECKS = ("lint", "typecheck", "tests")


class SemanticReviewRunner(Protocol):
    def __call__(self, files: list[str], task_intent: str, *, cwd: Path) -> list[Counterexample]: ...


def _default_run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True, check=False, timeout=120)


def _is_test_file_path(file_path: str) -> bool:
    path = Path(file_path)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "test" in parts or "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _typecheck_targets(files: list[str]) -> list[str]:
    """Distinct touched Python source files to typecheck exactly (excluding tests)."""
    targets: list[str] = []
    for f in files:
        if not f.endswith(".py") or _is_test_file_path(f):
            continue
        if f not in targets:
            targets.append(f)
    return targets


class VerifierCapability:
    """Run scoped deterministic checks and surface structured counterexamples."""

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        run: CommandRunner | None = None,
        task_intent: str | None = None,
        semantic_review: SemanticReviewRunner | None = None,
    ) -> None:
        self._cwd = Path(cwd or ".")
        self._run: CommandRunner = run or _default_run
        self._task_intent = task_intent.strip() if task_intent else ""
        self._semantic_review = semantic_review or run_semantic_review

    def run(
        self,
        *,
        scope_files: list[str],
        checks: Sequence[str] = _DEFAULT_CHECKS,
    ) -> list[Counterexample]:
        results: list[Counterexample] = []
        if "lint" in checks:
            results.extend(run_lint(scope_files, cwd=self._cwd, run=self._run))
        if "typecheck" in checks:
            results.extend(run_typecheck(_typecheck_targets(scope_files), cwd=self._cwd, run=self._run))
        if "tests" in checks:
            results.extend(run_tests(scope_files, cwd=self._cwd, run=self._run))
        if "semantic" in checks and self._task_intent:
            results.extend(self._semantic_review(scope_files, self._task_intent, cwd=self._cwd))
        return results

    @staticmethod
    def format_counterexamples(counterexamples: Sequence[Counterexample]) -> str:
        """Render counterexamples as a single TURN-channel feedback block."""
        return "\n".join(c.to_prompt_block() for c in counterexamples)
