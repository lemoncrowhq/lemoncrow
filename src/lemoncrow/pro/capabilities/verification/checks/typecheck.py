"""Typecheck (mypy) -> counterexamples (M3)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess

from ..counterexample import Counterexample

_Runner = Callable[[Sequence[str], Path], CompletedProcess[str]]
_LINE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?:\d+:)?\s*(?P<sev>error|warning|note):\s*(?P<msg>.*)$")


def _repro_command(targets: Sequence[str]) -> str:
    return "uv run mypy " + " ".join(targets)


def run_typecheck(targets: list[str], *, cwd: Path, run: _Runner) -> list[Counterexample]:
    """Run mypy over exact touched-file *targets* and parse diagnostics."""
    if not targets:
        return []
    try:
        proc = run(["uv", "run", "mypy", *targets], cwd)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []  # fail-open
    out: list[Counterexample] = []
    repro_command = _repro_command(targets)
    for raw in text.splitlines():
        m = _LINE.match(raw.strip())
        if not m:
            continue
        sev = m.group("sev")
        if sev == "note":
            continue
        out.append(
            Counterexample(
                check="typecheck",
                severity="error" if sev == "error" else "warn",
                file_path=m.group("file"),
                line=int(m.group("line")),
                diagnostic=m.group("msg").strip(),
                repro_command=repro_command,
            )
        )
    return out
