"""Deterministic checks for the verification capability (M3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess

from .lint import run_lint
from .semantic_review import run_semantic_review
from .tests import run_tests
from .typecheck import run_typecheck

CommandRunner = Callable[[Sequence[str], Path], CompletedProcess[str]]

__all__ = ["CommandRunner", "run_lint", "run_semantic_review", "run_tests", "run_typecheck"]
