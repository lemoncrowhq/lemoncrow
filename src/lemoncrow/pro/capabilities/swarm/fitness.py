"""Project-agnostic measured fitness for the swarm ``best`` reducer.

A :class:`FitnessSpec` turns "optimize <objective>" into a real measurement: a
shell command run inside a candidate's worktree whose output is parsed into a
number, an optional correctness gate that must pass, and a baseline to beat.
Nothing here is LemonCrow-specific -- the project (or the eliciting skill) supplies
the commands; the engine just runs and compares them.

Parsers (``metric_parse``):
    ``json:<dotted.key>``  -- ``json.loads`` stdout (or its last JSON line) then
                              walk a dotted path (list indices allowed).
    ``regex:<pattern>``    -- first match; group(1) if present else group(0).
    ``stdout_float``       -- the last numeric token in stdout (default).
    ``exit_code``          -- the command's exit code as a float.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lemoncrow.core.capabilities.swarm_contract import (
    FitnessDirection as FitnessDirection,
)
from lemoncrow.core.capabilities.swarm_contract import (
    FitnessSpec as FitnessSpec,
)


@dataclass(slots=True)
class CandidateFitness:
    gate_passed: bool
    gate_detail: str
    metric: float | None
    parse_error: str
    raw_stdout: str
    raw_stderr: str
    exit_code: int


def _run(command: str, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout if timeout and timeout > 0 else None,
        check=False,
    )


def _last_float(text: str) -> tuple[float | None, str]:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None, "no numeric token found in stdout"
    try:
        return float(matches[-1]), ""
    except ValueError as exc:  # pragma: no cover - regex guarantees parseability
        return None, str(exc)


def _loads_last_json(text: str) -> object | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        whole: object = json.loads(stripped)
        return whole
    except json.JSONDecodeError:
        pass
    for raw_line in reversed(stripped.splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed: object = json.loads(line)
            return parsed
        except json.JSONDecodeError:
            continue
    return None


def _json_dotted(text: str, dotted: str) -> tuple[float | None, str]:
    payload = _loads_last_json(text)
    if payload is None:
        return None, "stdout is not JSON"
    cursor: object = payload
    for part in [p for p in dotted.split(".") if p]:
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        elif isinstance(cursor, list) and part.lstrip("-").isdigit():
            try:
                cursor = cursor[int(part)]
            except IndexError:
                return None, f"index {part} out of range"
        else:
            return None, f"key {part!r} not found in JSON"
    try:
        return float(cursor), ""  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, f"value at {dotted!r} is not numeric: {cursor!r}"


def _regex_group(text: str, pattern: str) -> tuple[float | None, str]:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return None, f"bad regex: {exc}"
    match = rx.search(text)
    if not match:
        return None, "regex did not match stdout"
    group = match.group(1) if match.groups() else match.group(0)
    try:
        return float(group), ""
    except ValueError as exc:
        return None, f"matched text is not numeric: {exc}"


def parse_metric(stdout: str, exit_code: int, metric_parse: str) -> tuple[float | None, str]:
    """Parse ``stdout`` (or ``exit_code``) into a float per ``metric_parse``."""

    spec = metric_parse.strip()
    if spec == "exit_code":
        return float(exit_code), ""
    if spec in ("stdout_float", ""):
        return _last_float(stdout)
    if spec.startswith("json:"):
        return _json_dotted(stdout, spec[len("json:") :].strip())
    if spec.startswith("regex:"):
        return _regex_group(stdout, spec[len("regex:") :])
    return None, f"unknown metric_parse {metric_parse!r}"


def run_gate(spec: FitnessSpec, worktree: Path) -> tuple[bool, str]:
    """Run the correctness gate (if any); ``True`` iff it exits 0."""

    if not spec.gate_command:
        return True, "no gate command"
    proc = _run(spec.gate_command, worktree, spec.timeout_seconds)
    if proc.returncode == 0:
        return True, "gate passed"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (tail[-1] if tail else f"gate exited {proc.returncode}")


def measure(spec: FitnessSpec, worktree: Path) -> tuple[float | None, str, str]:
    """Run ``metric_command`` and parse it; returns (value, parse_error, stdout)."""

    proc = _run(spec.metric_command, worktree, spec.timeout_seconds)
    value, err = parse_metric(proc.stdout, proc.returncode, spec.metric_parse)
    return value, err, proc.stdout


def evaluate_candidate(spec: FitnessSpec, worktree: Path) -> CandidateFitness:
    """Gate then measure a single candidate worktree."""

    gate_passed, gate_detail = run_gate(spec, worktree)
    if not gate_passed:
        return CandidateFitness(False, gate_detail, None, "", "", "", 0)
    proc = _run(spec.metric_command, worktree, spec.timeout_seconds)
    value, err = parse_metric(proc.stdout, proc.returncode, spec.metric_parse)
    return CandidateFitness(True, gate_detail, value, err, proc.stdout, proc.stderr, proc.returncode)


def build_fitness_spec(
    *,
    metric_command: str,
    metric_parse: str = "stdout_float",
    direction: FitnessDirection = "min",
    gate_command: str | None = None,
    baseline: float | str = "auto",
    improve_margin: float = 0.0,
    objective: str = "",
) -> FitnessSpec | None:
    """Build a :class:`FitnessSpec` from loose (e.g. CLI) inputs.

    Returns ``None`` when no metric command is supplied (i.e. not an optimize
    job). ``baseline`` accepts the string ``"auto"`` or a number.
    """
    if not metric_command or not metric_command.strip():
        return None
    parsed_baseline: float | Literal["auto"]
    if isinstance(baseline, str):
        text = baseline.strip().lower()
        parsed_baseline = "auto" if text in ("", "auto") else float(baseline)
    else:
        parsed_baseline = float(baseline)
    return FitnessSpec(
        objective=objective,
        metric_command=metric_command,
        metric_parse=metric_parse or "stdout_float",
        direction=direction,
        gate_command=gate_command or None,
        baseline=parsed_baseline,
        improve_margin=improve_margin,
    )


def resolve_baseline(spec: FitnessSpec) -> float | None:
    """The numeric baseline, or ``None`` if it is still ``"auto"`` (unmeasured)."""

    return float(spec.baseline) if isinstance(spec.baseline, (int, float)) else None


def measure_baseline(spec: FitnessSpec, worktree: Path) -> float:
    """Measure the baseline metric on a base snapshot worktree."""

    value, err, _ = measure(spec, worktree)
    if value is None:
        raise RuntimeError(f"baseline metric did not parse: {err}")
    return value


def improvement(spec: FitnessSpec, metric: float, baseline: float) -> float:
    """Signed improvement of ``metric`` over ``baseline`` (positive == better)."""

    return (baseline - metric) if spec.direction == "min" else (metric - baseline)


def beats_baseline(spec: FitnessSpec, metric: float, baseline: float) -> bool:
    """True iff ``metric`` improves on ``baseline`` by at least ``improve_margin``."""

    return improvement(spec, metric, baseline) >= spec.improve_margin


def rank_key(spec: FitnessSpec, metric: float) -> float:
    """Descending sort key (bigger == better) when no baseline is available."""

    return -metric if spec.direction == "min" else metric


__all__ = [
    "CandidateFitness",
    "FitnessDirection",
    "FitnessSpec",
    "beats_baseline",
    "build_fitness_spec",
    "evaluate_candidate",
    "improvement",
    "measure",
    "measure_baseline",
    "parse_metric",
    "rank_key",
    "resolve_baseline",
    "run_gate",
]
