"""Command-aware Bash output profiles for compilers and build systems.

Generic compression can recognize repetition, JSON, tables, and progress lines
without knowing the command. Profiles add the missing semantic layer: they know
which lines are routine build phases and which lines form actionable diagnostic
records. Every reducer is deterministic and post-hoc; commands have already run
exactly once, and lossy results remain recoverable through the caller's spill or
managed-log path.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from lemoncrow.pro.capabilities.tool_supervision.bash_output_compression import (
    CompressionResult,
    native_compression_enabled,
)

Reducer = Callable[[str, int, int | None], CompressionResult]


@dataclass(frozen=True)
class OutputProfile:
    name: str
    command_re: re.Pattern[str]
    reducer: Reducer


_DIAGNOSTIC_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:ruff|mypy|pyright|tsc|eslint|biome|golangci-lint|rubocop|"
    r"gcc|g\+\+|clang|clang\+\+|cc|c\+\+|javac)(?:\s|$)"
    r"|(?:^|\s)dotnet\s+(?:build|format)(?:\s|$)"
    r"|(?:^|\s)cargo\s+(?:check|clippy|build)(?:\s|$)",
    re.IGNORECASE,
)
_DIAGNOSTIC_LINE_RE = re.compile(
    r"(?:\b(?:fatal error|error|warning|note|help)\b(?:\[[^]]+\])?\s*:"
    r"|\b(?:error|warning)\s+(?:TS|CS|BC|CA|MSB)\d+\s*:"
    r"|\b[CEFNRW]\d{3,5}\b"
    r"|\bTS\d{3,5}\b"
    r"|\bCS\d{3,5}\b"
    r"|^\s*\d+:\d+\s+(?:error|warning)\s+)",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(r"\b(?:fatal error|error|failed|failure|panic)\b", re.IGNORECASE)
_WARNING_RE = re.compile(r"\b(?:warning|warn|deprecated)\b", re.IGNORECASE)
_SUMMARY_RE = re.compile(
    r"(?:\bfound\s+\d+\s+errors?\b|\b\d+\s+(?:errors?|warnings?|problems?)\b|"
    r"\berror:\s+aborting\b|\bcompilation failed\b|\bbuild failed\b|"
    r"\bfinished with\s+\d+\s+errors?\b)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"\b(?:TS|CS|BC|CA|MSB)?[A-Z]?\d{3,5}\b")


def _result(original: str, compacted: str, *, omitted: int, method: str) -> CompressionResult:
    if original.endswith("\n") and compacted and not compacted.endswith("\n"):
        compacted += "\n"
    saved = len(original) - len(compacted)
    if saved <= 0:
        return CompressionResult(original)
    return CompressionResult(compacted, saved, max(0, omitted), True, (method,))


def _merge(current: CompressionResult, result: CompressionResult) -> CompressionResult:
    if result.text == current.text:
        return current
    return CompressionResult(
        result.text,
        current.chars_saved + result.chars_saved,
        current.lines_omitted + result.lines_omitted,
        current.lossy or result.lossy,
        tuple(dict.fromkeys((*current.methods, *result.methods))),
    )


def _windows(indices: list[int], total: int, *, before: int = 1, after: int = 3) -> set[int]:
    kept: set[int] = set()
    for index in indices:
        kept.update(range(max(0, index - before), min(total, index + after + 1)))
    return kept


def _reduce_diagnostics(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    """Group compiler/linter diagnostics while retaining error source context."""
    lines = text.splitlines()
    hits = [index for index, line in enumerate(lines) if _DIAGNOSTIC_LINE_RE.search(line)]
    if len(hits) < 10 and len(text) <= budget:
        return CompressionResult(text)

    error_hits = [index for index in hits if _ERROR_RE.search(lines[index])]
    warning_hits = [index for index in hits if index not in error_hits and _WARNING_RE.search(lines[index])]
    other_hits = [index for index in hits if index not in error_hits and index not in warning_hits]

    selected_errors = error_hits if len(error_hits) <= 60 else [*error_hits[:45], *error_hits[-15:]]
    selected_warnings = warning_hits if len(warning_hits) <= 24 else [*warning_hits[:18], *warning_hits[-6:]]
    selected_other = other_hits if len(other_hits) <= 16 else [*other_hits[:12], *other_hits[-4:]]
    selected = [*selected_errors, *selected_warnings, *selected_other]
    kept = _windows(selected, len(lines))
    kept.update(index for index, line in enumerate(lines) if _SUMMARY_RE.search(line))
    kept.update(range(min(2, len(lines))))
    kept.update(range(max(0, len(lines) - 3), len(lines)))

    if len(kept) >= len(lines) - 3:
        return CompressionResult(text)

    codes = Counter(code.upper() for index in hits for code in _CODE_RE.findall(lines[index]))
    code_summary = ", ".join(f"{code}x{count}" for code, count in codes.most_common(8))
    omitted = len(lines) - len(kept)
    header = (
        f"[lc diagnostics: errors={len(error_hits)}, warnings={len(warning_hits)}, "
        f"other={len(other_hits)}, omitted_lines={omitted}" + (f"; codes={code_summary}" if code_summary else "") + "]"
    )
    body = [header]
    previous = -2
    for index in sorted(kept):
        if index > previous + 1:
            body.append(f"... ({index - previous - 1} diagnostic-adjacent lines omitted) ...")
        body.append(lines[index])
        previous = index
    if previous < len(lines) - 1:
        body.append(f"... ({len(lines) - previous - 1} trailing lines omitted) ...")
    return _result(text, "\n".join(body), omitted=omitted, method="profile:diagnostics")


def _reduce_phases(
    text: str,
    pattern: re.Pattern[str],
    *,
    label: str,
    minimum: int = 10,
) -> CompressionResult:
    """Aggregate routine phase lines by verb while leaving all other lines."""
    lines = text.splitlines()
    matches: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match and not _ERROR_RE.search(line) and not _WARNING_RE.search(line):
            phase = match.group(1).lower()
            if phase.startswith("[") and "/" in phase:
                phase = "progress"
            matches.append((index, phase))
    if len(matches) < minimum:
        return CompressionResult(text)

    positions: dict[str, list[int]] = {}
    for index, phase in matches:
        positions.setdefault(phase, []).append(index)
    removable: set[int] = set()
    replacement: dict[int, str] = {}
    for phase, indexes in positions.items():
        if len(indexes) < 4:
            continue
        keep = {indexes[0], indexes[-1]}
        removable.update(index for index in indexes if index not in keep)
        first = lines[indexes[0]].strip()
        last = lines[indexes[-1]].strip()
        replacement[indexes[0]] = f"[lc {label}: {phase}x{len(indexes)}; first={first}; last={last}]"
    if not removable:
        return CompressionResult(text)
    body: list[str] = []
    for index, line in enumerate(lines):
        if index in removable:
            continue
        body.append(replacement.get(index, line))
    return _result(text, "\n".join(body), omitted=len(removable), method=f"profile:{label}")


_CARGO_PHASE_RE = re.compile(
    r"^\s*(Compiling|Checking|Fresh|Downloading|Downloaded|Updating|Locking)\b",
    re.IGNORECASE,
)
_MAVEN_PHASE_RE = re.compile(
    r"^\s*\[INFO\]\s+(---|Building|Downloading|Downloaded|Copying|Compiling|Changes detected)(?:\s|$)",
    re.IGNORECASE,
)
_NATIVE_PHASE_RE = re.compile(
    r"^\s*(\[\d+/\d+\]|Scanning|Generating|Building|Linking|Compiling)(?:\s|$)",
    re.IGNORECASE,
)


def _reduce_cargo(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    return _reduce_phases(text, _CARGO_PHASE_RE, label="cargo")


def _reduce_maven(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    return _reduce_phases(text, _MAVEN_PHASE_RE, label="maven")


def _reduce_native_build(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    return _reduce_phases(text, _NATIVE_PHASE_RE, label="native-build")


_GO_SUCCESS_RE = re.compile(r"^(ok|\?)\s+(\S+)(?:\s+.*)?$")


def _reduce_go(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    lines = text.splitlines()
    successes = [(index, match) for index, line in enumerate(lines) if (match := _GO_SUCCESS_RE.match(line))]
    if len(successes) < 12:
        return CompressionResult(text)
    removable = {index for index, _ in successes[2:-2]}
    if not removable:
        return CompressionResult(text)
    ok_count = sum(1 for _, match in successes if match.group(1) == "ok")
    empty_count = len(successes) - ok_count
    first_index = successes[0][0]
    body: list[str] = []
    for index, line in enumerate(lines):
        if index == first_index:
            body.append(f"[lc go packages: ok={ok_count}, no-tests={empty_count}, shown=4/{len(successes)}]")
        if index not in removable:
            body.append(line)
    return _result(text, "\n".join(body), omitted=len(removable), method="profile:go")


_GRADLE_TASK_RE = re.compile(r"^\s*> Task\s+(\S+)(?:\s+(UP-TO-DATE|FROM-CACHE|SKIPPED|NO-SOURCE|FAILED))?\s*$")


def _reduce_gradle(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    lines = text.splitlines()
    tasks = [(index, match) for index, line in enumerate(lines) if (match := _GRADLE_TASK_RE.match(line))]
    if len(tasks) < 14:
        return CompressionResult(text)
    statuses = Counter((match.group(2) or "EXECUTED") for _, match in tasks)
    keep_positions = {index for index, match in tasks if match.group(2) == "FAILED"}
    keep_positions.update(index for index, _ in tasks[:3])
    keep_positions.update(index for index, _ in tasks[-3:])
    removable = {index for index, _ in tasks if index not in keep_positions}
    if not removable:
        return CompressionResult(text)
    summary = ", ".join(f"{status.lower()}={count}" for status, count in statuses.most_common())
    first_index = tasks[0][0]
    body: list[str] = []
    for index, line in enumerate(lines):
        if index == first_index:
            body.append(f"[lc gradle tasks: total={len(tasks)}, {summary}, shown={len(keep_positions)}]")
        if index not in removable:
            body.append(line)
    return _result(text, "\n".join(body), omitted=len(removable), method="profile:gradle")


_DOCKER_STEP_RE = re.compile(r"^#(\d+)\s+(.+)$")


def _reduce_docker_build(text: str, budget: int, exit_code: int | None) -> CompressionResult:
    lines = text.splitlines()
    steps: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        match = _DOCKER_STEP_RE.match(line)
        if match and not _ERROR_RE.search(line):
            steps.setdefault(match.group(1), []).append(index)
    total = sum(len(indexes) for indexes in steps.values())
    if total < 20:
        return CompressionResult(text)
    removable: set[int] = set()
    replacements: dict[int, str] = {}
    for step, indexes in steps.items():
        if len(indexes) < 3:
            continue
        removable.update(indexes[1:-1])
        replacements[indexes[0]] = (
            f"[lc docker step #{step}: {len(indexes)} log lines; "
            f"first={lines[indexes[0]].strip()}; last={lines[indexes[-1]].strip()}]"
        )
    if not removable:
        return CompressionResult(text)
    body = [replacements.get(index, line) for index, line in enumerate(lines) if index not in removable]
    return _result(text, "\n".join(body), omitted=len(removable), method="profile:docker-build")


_PROFILES: tuple[OutputProfile, ...] = (
    OutputProfile("cargo", re.compile(r"(?:^|\s)cargo\s+(?:build|check|clippy|test)(?:\s|$)", re.I), _reduce_cargo),
    OutputProfile("go", re.compile(r"(?:^|\s)go\s+(?:test|vet|build)(?:\s|$)", re.I), _reduce_go),
    OutputProfile("gradle", re.compile(r"(?:^|\s)(?:\.?/?gradlew|gradle)(?:\s|$)", re.I), _reduce_gradle),
    OutputProfile("maven", re.compile(r"(?:^|\s)mvn(?:w)?(?:\s|$)", re.I), _reduce_maven),
    OutputProfile(
        "native-build",
        re.compile(r"(?:^|\s)(?:make|ninja|cmake|meson|next|vite|webpack|esbuild)(?:\s|$)", re.I),
        _reduce_native_build,
    ),
    OutputProfile(
        "docker-build", re.compile(r"(?:^|\s)docker(?:\s+compose)?\s+build(?:\s|$)", re.I), _reduce_docker_build
    ),
    OutputProfile("diagnostics", _DIAGNOSTIC_COMMAND_RE, _reduce_diagnostics),
)


def compact_profiled_output(
    command: str,
    text: str,
    *,
    budget: int,
    exit_code: int | None,
) -> CompressionResult:
    """Apply every matching command profile, then return one accumulated result."""
    if not text or not native_compression_enabled():
        return CompressionResult(text)
    current = CompressionResult(text)
    for profile in _PROFILES:
        if profile.command_re.search(command):
            current = _merge(current, profile.reducer(current.text, budget, exit_code))
    return current


__all__ = ["OutputProfile", "compact_profiled_output"]
