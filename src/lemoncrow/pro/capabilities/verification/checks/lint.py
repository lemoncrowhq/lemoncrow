"""Lint check (ruff) -> counterexamples (M3)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from ..counterexample import Counterexample

_Runner = Callable[[Sequence[str], Path], CompletedProcess[str]]


def run_lint(files: list[str], *, cwd: Path, run: _Runner) -> list[Counterexample]:
    """Run ruff over *files* and parse JSON diagnostics into counterexamples."""
    py_files = [f for f in files if f.endswith(".py")]
    if not py_files:
        return []
    try:
        proc = run(["ruff", "check", "--output-format=json", *py_files], cwd)
        items: list[dict[str, Any]] = json.loads(proc.stdout or "[]")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []  # fail-open
    out: list[Counterexample] = []
    for item in items:
        loc = item.get("location") or {}
        code = str(item.get("code") or "")
        message = str(item.get("message") or "")
        filename = item.get("filename")
        out.append(
            Counterexample(
                check="lint",
                severity="error",
                file_path=str(filename) if filename else None,
                line=int(loc["row"]) if loc.get("row") is not None else None,
                diagnostic=f"{code} {message}".strip(),
                repro_command=f"ruff check {filename}" if filename else "ruff check",
            )
        )
    return out
