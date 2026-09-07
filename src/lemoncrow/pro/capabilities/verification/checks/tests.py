"""Test check (pytest) -> counterexamples (M3)."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from subprocess import CompletedProcess

from ..counterexample import Counterexample

_Runner = Callable[[Sequence[str], Path], CompletedProcess[str]]
_FAILED = re.compile(r"^FAILED\s+(?P<node>\S+)(?:\s+-\s+(?P<msg>.*))?$")


def run_tests(test_files: list[str], *, cwd: Path, run: _Runner) -> list[Counterexample]:
    """Run pytest over scoped *test_files* and parse FAILED lines."""
    scope = [f for f in test_files if "test" in Path(f).name]
    if not scope:
        return []
    try:
        proc = run(["pytest", "-q", "--no-header", *scope], cwd)
        text = proc.stdout or ""
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []  # fail-open
    out: list[Counterexample] = []
    for raw in text.splitlines():
        m = _FAILED.match(raw.strip())
        if not m:
            continue
        node = m.group("node")
        file_path = node.split("::", 1)[0]
        out.append(
            Counterexample(
                check="tests",
                severity="error",
                file_path=file_path,
                line=None,
                diagnostic=(m.group("msg") or "test failed").strip(),
                repro_command=f"pytest -q {node}",
            )
        )
    return out
