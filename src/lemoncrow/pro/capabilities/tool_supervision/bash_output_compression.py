"""Deterministic semantic compression for Bash stdout/stderr.

RTK and similar wrappers can request compact output from supported read-only
commands before execution. This module is the universal post-hoc layer: it runs
after every command exactly once, including unsupported CLIs, mutators that must
never be wrapped, deferred commands, and interactive deltas.

Lossy transforms always include a visible omission marker. The caller is
responsible for retaining the full stream through its spill/log recovery path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_ENV_ENABLED = "LEMONCROW_BASH_NATIVE_COMPRESSION"
_MIN_CHARS = 1200
_MIN_LINES = 24

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ISO_TIME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+Z?\b")
_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-f]{8,64}\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|s|sec|secs|seconds?|m|min|mins|minutes?|h|hrs?|hours?)\b",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:[kmgtpe]?i?b|bytes?)\b", re.IGNORECASE)
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")

_DIAGNOSTIC_RE = re.compile(
    r"\b(error|exception|traceback|fatal|panic|warning|deprecated|denied|refused|"
    r"failed|failure|segfault|deadlock|cannot|can't|unable to|undefined reference|"
    r"not found|timed out|timeout|out of memory|oom|killed|permission|conflict)\b",
    re.IGNORECASE,
)
_LOGISH_RE = re.compile(
    r"^\s*(?:\[?(?:trace|debug|info|notice|warn(?:ing)?|error|fatal)\]?[: ]|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|"
    r"(?:downloading|downloaded|fetching|fetched|resolving|resolved|compiling|compiled|"
    r"building|built|installing|installed|linking|bundling|transpiling|generating|generated|"
    r"checking|checked|copying|extracting|unpacking|uploading|uploaded|processing|processed)\b)",
    re.IGNORECASE,
)
_PROGRESS_RE = re.compile(
    r"^\s*(?:\[[^]]+\]\s*)?(downloading|fetching|resolving|compiling|building|installing|"
    r"linking|bundling|transpiling|generating|checking|copying|extracting|unpacking|"
    r"uploading|processing)\b",
    re.IGNORECASE,
)
_TABLE_SEPARATOR_RE = re.compile(r"^\s*(?:[-=+|:]\s*){4,}$")


@dataclass(frozen=True)
class CompressionResult:
    text: str
    chars_saved: int = 0
    lines_omitted: int = 0
    lossy: bool = False
    methods: tuple[str, ...] = ()


def native_compression_enabled() -> bool:
    return os.environ.get(_ENV_ENABLED, "1").strip().lower() not in {"0", "false", "no", "off"}


def terminal_visible_text(text: str) -> str:
    """Apply terminal redraw semantics after ANSI escapes have been removed.

    Carriage-return progress bars leave only their final frame visible, and
    backspaces edit the current line instead of becoming transcript bytes.
    """
    if "\b" in text:
        out: list[str] = []
        for char in text:
            if char == "\b":
                if out and out[-1] != "\n":
                    out.pop()
            else:
                out.append(char)
        text = "".join(out)
    if "\r" in text:
        text = text.replace("\r\n", "\n")
        text = "\n".join(part.split("\r")[-1] for part in text.split("\n"))
    return _CONTROL_RE.sub("", text)


def _newline_like(original: str, compacted: str) -> str:
    if original.endswith("\n") and compacted and not compacted.endswith("\n"):
        return compacted + "\n"
    return compacted


def _result(original: str, compacted: str, *, omitted: int, method: str, lossy: bool) -> CompressionResult:
    compacted = _newline_like(original, compacted)
    saved = len(original) - len(compacted)
    if saved <= 0:
        return CompressionResult(original)
    return CompressionResult(compacted, saved, omitted, lossy, (method,))


def _diagnostic(value: Any) -> bool:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError):
        rendered = str(value)
    return bool(_DIAGNOSTIC_RE.search(rendered))


def _compact_json_document(text: str, budget: int) -> CompressionResult | None:
    """Minify JSON losslessly; sample large arrays/objects as valid JSON."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return None
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, RecursionError):
        return None
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(compact) <= budget:
        return _result(text, compact, omitted=0, method="json-minify", lossy=False)

    if isinstance(value, list) and len(value) >= 24:
        selected = {0, 1, 2, len(value) - 2, len(value) - 1}
        selected.update(index for index, row in enumerate(value) if _diagnostic(row))
        ordered = sorted(selected)
        if len(ordered) > 40:
            return _result(text, compact, omitted=0, method="json-minify", lossy=False)
        keys: list[str] = []
        for row in value:
            if isinstance(row, dict):
                for key in row:
                    key_text = str(key)
                    if key_text not in keys:
                        keys.append(key_text)
                    if len(keys) >= 24:
                        break
            if len(keys) >= 24:
                break
        omitted = len(value) - len(ordered)
        sampled = {
            "_lemoncrow": {
                "kind": "json-array-sample",
                "items": len(value),
                "shown": len(ordered),
                "omitted": omitted,
                "keys": keys,
            },
            "items": [value[index] for index in ordered],
        }
        return _result(
            text,
            json.dumps(sampled, ensure_ascii=False, separators=(",", ":")),
            omitted=omitted,
            method="json-array-sample",
            lossy=True,
        )

    if isinstance(value, dict) and len(value) >= 20:
        keys = list(value)
        selected_keys = [*keys[:8], *keys[-4:]]
        selected_keys.extend(key for key, item in value.items() if _diagnostic(item))
        selected_keys = list(dict.fromkeys(selected_keys))
        if len(selected_keys) > 40:
            return _result(text, compact, omitted=0, method="json-minify", lossy=False)
        omitted = len(keys) - len(selected_keys)
        sampled = {
            "_lemoncrow": {
                "kind": "json-object-sample",
                "keys": len(keys),
                "shown": len(selected_keys),
                "omitted": omitted,
                "key_names": keys[:40],
            },
            "sample": {key: value[key] for key in selected_keys},
        }
        return _result(
            text,
            json.dumps(sampled, ensure_ascii=False, separators=(",", ":")),
            omitted=omitted,
            method="json-object-sample",
            lossy=True,
        )
    return _result(text, compact, omitted=0, method="json-minify", lossy=False)


def _compact_ndjson(text: str, budget: int) -> CompressionResult:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(text) <= budget or len(lines) < 24:
        return CompressionResult(text)
    rows: list[Any] = []
    try:
        rows = [json.loads(line) for line in lines]
    except (json.JSONDecodeError, RecursionError):
        return CompressionResult(text)
    if not all(isinstance(row, (dict, list)) for row in rows):
        return CompressionResult(text)
    selected = {0, 1, 2, len(lines) - 2, len(lines) - 1}
    selected.update(index for index, row in enumerate(rows) if _diagnostic(row))
    ordered = sorted(selected)
    if len(ordered) > 40:
        return CompressionResult(text)
    omitted = len(lines) - len(ordered)
    body = [f"... ({omitted} NDJSON records omitted; {len(ordered)}/{len(lines)} shown) ..."]
    body.extend(lines[index] for index in ordered)
    return _result(text, "\n".join(body), omitted=omitted, method="ndjson-sample", lossy=True)


def _signature(line: str) -> str:
    value = _ISO_TIME_RE.sub("<time>", line)
    value = _CLOCK_RE.sub("<time>", value)
    value = _UUID_RE.sub("<id>", value)
    value = _HEX_RE.sub("<id>", value)
    value = _DURATION_RE.sub("<duration>", value)
    value = _SIZE_RE.sub("<size>", value)
    value = _PERCENT_RE.sub("<percent>", value)
    value = _NUMBER_RE.sub("<n>", value)
    return re.sub(r"\s+", " ", value).strip()


def _foldable(line: str) -> bool:
    return bool(_LOGISH_RE.search(line) or _PROGRESS_RE.search(line)) and not bool(_DIAGNOSTIC_RE.search(line))


def _collapse_similar_runs(text: str, minimum: int = 4) -> CompressionResult:
    """Fold adjacent logs that differ only in timestamps, IDs, sizes, or counts."""
    lines = text.splitlines()
    if len(lines) < minimum:
        return CompressionResult(text)
    out: list[str] = []
    omitted = 0
    index = 0
    while index < len(lines):
        signature = _signature(lines[index])
        end = index + 1
        while end < len(lines) and signature and _signature(lines[end]) == signature:
            end += 1
        count = end - index
        if count >= minimum and _foldable(lines[index]):
            out.extend((lines[index], f"... ({count - 2} similar log lines omitted) ...", lines[end - 1]))
            omitted += count - 2
        else:
            out.extend(lines[index:end])
        index = end
    return _result(text, "\n".join(out), omitted=omitted, method="similar-lines", lossy=True)


def _collapse_progress_phases(text: str, minimum: int = 8) -> CompressionResult:
    """Keep first/last samples from long adjacent build/download phases."""
    lines = text.splitlines()
    if len(lines) < minimum:
        return CompressionResult(text)
    out: list[str] = []
    omitted = 0
    index = 0
    while index < len(lines):
        match = _PROGRESS_RE.search(lines[index])
        phase = match.group(1).lower() if match and not _DIAGNOSTIC_RE.search(lines[index]) else ""
        end = index + 1
        while end < len(lines):
            next_match = _PROGRESS_RE.search(lines[end])
            next_phase = next_match.group(1).lower() if next_match and not _DIAGNOSTIC_RE.search(lines[end]) else ""
            if not phase or next_phase != phase:
                break
            end += 1
        count = end - index
        if phase and count >= minimum:
            out.extend(lines[index : index + 2])
            out.append(f"... ({count - 4} {phase} progress lines omitted) ...")
            out.extend(lines[end - 2 : end])
            omitted += count - 4
        else:
            out.extend(lines[index:end])
        index = end
    return _result(text, "\n".join(out), omitted=omitted, method="progress-phases", lossy=True)


def _collapse_recurring_messages(text: str, minimum: int = 8) -> CompressionResult:
    """Aggregate non-adjacent heartbeat/retry messages by volatile signature."""
    lines = text.splitlines()
    if len(lines) < minimum:
        return CompressionResult(text)
    positions: dict[str, list[int]] = {}
    for index, line in enumerate(lines):
        if _foldable(line):
            positions.setdefault(_signature(line), []).append(index)
    recurring = {key: indexes for key, indexes in positions.items() if key and len(indexes) >= minimum}
    if not recurring:
        return CompressionResult(text)
    out: list[str] = []
    omitted = 0
    for index, line in enumerate(lines):
        indexes = recurring.get(_signature(line)) if _foldable(line) else None
        if indexes is None:
            out.append(line)
        elif index in {indexes[0], indexes[-1]}:
            out.append(line)
        elif index == indexes[1]:
            count = len(indexes) - 2
            out.append(f"... (message recurred {count} intermediate times) ...")
            omitted += count
    return _result(text, "\n".join(out), omitted=omitted, method="recurring-messages", lossy=True)


def _collapse_repeated_blocks(text: str) -> CompressionResult:
    """Fold exact consecutive multi-line blocks, common in retry stack dumps."""
    lines = text.splitlines()
    if len(lines) < 6:
        return CompressionResult(text)
    out: list[str] = []
    omitted = 0
    index = 0
    while index < len(lines):
        matched = False
        for width in range(2, 7):
            block = lines[index : index + width]
            if len(block) < width or any(_DIAGNOSTIC_RE.search(line) for line in block):
                continue
            repeats = 1
            while lines[index + repeats * width : index + (repeats + 1) * width] == block:
                repeats += 1
            if repeats >= 3:
                out.extend(block)
                hidden = (repeats - 1) * width
                out.append(f"... ({repeats - 1} repeated {width}-line blocks; {hidden} lines omitted) ...")
                omitted += hidden
                index += repeats * width
                matched = True
                break
        if not matched:
            out.append(lines[index])
            index += 1
    return _result(text, "\n".join(out), omitted=omitted, method="repeated-blocks", lossy=True)


def _table_shape(line: str) -> tuple[str, int] | None:
    if not line.strip() or _TABLE_SEPARATOR_RE.match(line):
        return None
    if line.count("|") >= 2:
        return ("pipe", line.count("|") + 1)
    if line.count("\t") >= 2:
        return ("tab", line.count("\t") + 1)
    columns = re.split(r"\s{2,}", line.strip())
    return ("spaces", len(columns)) if len(columns) >= 3 else None


def _sample_table(text: str, budget: int) -> CompressionResult:
    """Sample homogeneous tabular output while retaining header and tail rows."""
    lines = text.splitlines()
    if len(text) <= budget or len(lines) < 32 or any(_DIAGNOSTIC_RE.search(line) for line in lines):
        return CompressionResult(text)
    counts: dict[tuple[str, int], int] = {}
    for line in lines:
        shape = _table_shape(line)
        if shape is not None:
            counts[shape] = counts.get(shape, 0) + 1
    if not counts:
        return CompressionResult(text)
    shape, count = max(counts.items(), key=lambda item: item[1])
    if count < int(len(lines) * 0.75):
        return CompressionResult(text)
    head, tail = 10, 6
    omitted = len(lines) - head - tail
    if omitted <= 8:
        return CompressionResult(text)
    body = [*lines[:head], f"... ({omitted} table rows omitted; shape={shape[1]} cols) ...", *lines[-tail:]]
    return _result(text, "\n".join(body), omitted=omitted, method="table-sample", lossy=True)


def _compact_long_lines(text: str, budget: int) -> CompressionResult:
    """Bound pathological single lines with a stable digest and both endpoints."""
    limit = max(800, min(4000, budget // 2))
    lines = text.splitlines()
    changed = False
    saved = 0
    out: list[str] = []
    for line in lines:
        if len(line) <= limit:
            out.append(line)
            continue
        digest = hashlib.sha256(line.encode("utf-8", errors="replace")).hexdigest()[:12]
        head = int(limit * 0.58)
        tail = int(limit * 0.30)
        omitted = len(line) - head - tail
        compacted = f"{line[:head]} [... {omitted} chars omitted; sha256={digest} ...] {line[-tail:]}"
        out.append(compacted)
        saved += len(line) - len(compacted)
        changed = True
    if not changed:
        return CompressionResult(text)
    compacted_text = _newline_like(text, "\n".join(out))
    return CompressionResult(compacted_text, max(0, saved), 0, True, ("long-lines",))


def _apply(current: CompressionResult, transform: Callable[[str], CompressionResult]) -> CompressionResult:
    result = transform(current.text)
    if result.text == current.text:
        return current
    return CompressionResult(
        result.text,
        current.chars_saved + result.chars_saved,
        current.lines_omitted + result.lines_omitted,
        current.lossy or result.lossy,
        tuple(dict.fromkeys((*current.methods, *result.methods))),
    )


def compact_bash_stream(text: str, *, budget: int) -> CompressionResult:
    """Apply all safe deterministic tiers to one cleaned output stream."""
    if not text or not native_compression_enabled():
        return CompressionResult(text)

    json_result = _compact_json_document(text, budget)
    if json_result is not None:
        return json_result

    current = CompressionResult(text)
    current = _apply(current, lambda value: _compact_ndjson(value, budget))
    if len(current.text) >= _MIN_CHARS or len(current.text.splitlines()) >= _MIN_LINES:
        for transform in (
            _collapse_repeated_blocks,
            _collapse_progress_phases,
            _collapse_similar_runs,
            _collapse_recurring_messages,
        ):
            current = _apply(current, transform)
    current = _apply(current, lambda value: _sample_table(value, budget))
    current = _apply(current, lambda value: _compact_long_lines(value, budget))
    return current


__all__ = [
    "CompressionResult",
    "compact_bash_stream",
    "native_compression_enabled",
    "terminal_visible_text",
]
