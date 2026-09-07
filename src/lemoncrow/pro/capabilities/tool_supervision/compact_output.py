"""Threshold-triggered tool-output compaction.

Head+tail compression keeps the high-signal start and end of an oversized tool
output and elides the repetitive middle, substantially cutting input tokens on
long tool results.

Key design choices:
- Char-based threshold (1800 chars) instead of token-based — predictable and fast
- Asymmetric head/tail split: head gets more budget (start has command, first error,
  context; tail has final result/status — middle is usually repetitive output)
- LLM summarization is opt-in only; head+tail alone achieves the savings
- keep_recent_tool_messages exempts the last N messages from compression, so the
  agent always sees its active step at full fidelity
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from lemoncrow.core.capabilities.tool_supervision_contract import (
    CompactMethod as CompactMethod,
)
from lemoncrow.core.capabilities.tool_supervision_contract import (
    CompactResult as CompactResult,
)
from lemoncrow.core.capabilities.tool_supervision_contract import (
    GateResult as GateResult,
)
from lemoncrow.infra.internal_llm import InternalLLMError, summarize
from lemoncrow.pro.capabilities.prompt_compilation.tokens import count_tokens

ContentType = Literal["file", "grep", "bash", "tool_output", "unknown"]

# Char-based compaction threshold: outputs longer than this get head+tail elided.
DEFAULT_COMPRESS_THRESHOLD_CHARS = 1800
DEFAULT_HEAD_KEEP_CHARS = 900  # ~56% of budget — head has more signal
DEFAULT_TAIL_KEEP_CHARS = 700  # ~44% of budget — tail has final result/status


# --------------------------------------------------------------------------- #
# Stats tracker                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class TokenSavingStats:
    """Aggregate token-saving counters for a session or benchmark run."""

    compressions: int = 0
    chars_saved: int = 0
    early_exits: int = 0
    tokens_saved: int = 0  # approx from tiktoken cl100k_base
    messages_compressed: list[int] = field(default_factory=list)  # chars saved per message

    def record(self, original_chars: int, compacted_chars: int, original_tokens: int, compacted_tokens: int) -> None:
        """Record one compression event."""
        if original_chars > compacted_chars:
            self.compressions += 1
            saved_chars = original_chars - compacted_chars
            self.chars_saved += saved_chars
            self.tokens_saved += max(0, original_tokens - compacted_tokens)
            self.messages_compressed.append(saved_chars)

    @property
    def compression_ratio(self) -> float:
        """Fraction of chars removed (0 = no savings, 1 = all removed)."""
        if not self.messages_compressed:
            return 0.0
        total_original = self.chars_saved + sum(DEFAULT_COMPRESS_THRESHOLD_CHARS for _ in self.messages_compressed)
        return self.chars_saved / max(1, total_original + self.chars_saved)


def compress_tool_output(
    content: str,
    *,
    threshold_chars: int = DEFAULT_COMPRESS_THRESHOLD_CHARS,
    head_chars: int = DEFAULT_HEAD_KEEP_CHARS,
    tail_chars: int = DEFAULT_TAIL_KEEP_CHARS,
) -> str:
    """Head+tail compress a single tool output string.

    Returns the content unchanged when it is within the threshold.
    When above the threshold, returns head + omission notice + tail.

    This is a standalone helper usable outside the compact MCP tool lifecycle.

    Args:
        content:         The tool output string.
        threshold_chars: Minimum length before compression is applied.
        head_chars:      Characters to keep from the start (default 900 — more
                         signal: command, first error, initial context).
        tail_chars:      Characters to keep from the end (default 700 — final
                         result, return value, last error).
    """
    if len(content) <= threshold_chars:
        return content
    elided = len(content) - head_chars - tail_chars
    return f"{content[:head_chars]}\n\n[… {elided} chars omitted …]\n\n{content[-tail_chars:]}"


def _head_tail(text: str, *, max_chars: int) -> str:
    """Legacy helper — kept for backward compatibility with existing callers.

    Uses asymmetric split: 60% head / 40% tail.
    """
    if len(text) <= max_chars:
        return text
    head = max(1, int(max_chars * 0.6))
    tail = max(1, max_chars - head)
    elided = len(text) - head - tail
    return f"{text[:head]}\n... ({elided} chars elided) ...\n{text[-tail:]}"


# Matches `path:lineno:` grep prefixes (filename then a numeric line number),
# so only real match lines start a new file bucket — separators, blank lines,
# and context lines stay attached to the file they belong to.
_GREP_PREFIX = re.compile(r"^(?P<path>.+?):\d+:")


def _compact_grep(content: str, budget_tokens: int = 80) -> str:
    """Group grep output by file, keeping a budget-scaled number of hits each.

    Lines are bucketed by the `path:lineno:` prefix; lines without it (group
    separators, context lines) attach to the current file instead of scattering
    into pseudo-files. The per-file keep count scales with the token budget so a
    file's 4th+ hit is only elided when the budget is genuinely exhausted.
    """
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    current = "unknown"
    for line in content.splitlines():
        match = _GREP_PREFIX.match(line)
        if match:
            current = match.group("path")
        if current not in grouped:
            grouped[current] = []
            order.append(current)
        grouped[current].append(line)
    # Budget ~= budget_tokens*4 chars; keep at least 3 hits per file so small
    # budgets still show real signal, and more when the budget allows.
    per_file_keep = max(3, (budget_tokens * 4) // max(1, len(order)) // 80)
    parts: list[str] = []
    for file_name in order:
        lines = grouped[file_name]
        parts.extend(lines[:per_file_keep])
        remaining = len(lines) - per_file_keep
        if remaining > 0:
            parts.append(f"... and {remaining} more in {file_name}")
    return "\n".join(parts)


def _compact_bash(content: str, budget_chars: int = 8000) -> str:
    """Compress bash output keeping head and tail by char budget.

    Preserves stderr context even after truncation — extracts it first,
    then re-attaches it if the truncated body lost it.
    """
    stderr_match = re.search(r"stderr:\s*(.+?)(?:\n\n|\Z)", content, flags=re.IGNORECASE | re.DOTALL)
    stderr = stderr_match.group(1).strip() if stderr_match else ""

    if len(content) <= budget_chars:
        return content

    head_chars = int(budget_chars * 0.6)
    tail_chars = budget_chars - head_chars
    compacted = compress_tool_output(
        content, threshold_chars=budget_chars, head_chars=head_chars, tail_chars=tail_chars
    )
    if stderr and stderr not in compacted:
        return f"{compacted}\n\nFull stderr:\n{stderr}"
    return compacted


def _compact_json(content: str) -> str | None:
    # Emit compact JSON (no indent whitespace): this helper exists to reduce
    # oversized tool output, so pretty-printing would be self-defeating.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        list_sample = data[:2]
        return json.dumps(
            {"type": "list", "len": len(data), "sample": list_sample},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if isinstance(data, dict):
        keys = sorted(data.keys())
        dict_sample = {key: data[key] for key in keys[:10]}
        return json.dumps(
            {"type": "object", "keys": keys, "sample": dict_sample},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return json.dumps(
        {"type": type(data).__name__, "value": data},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deterministic_truncate(content: str, content_type: str, budget_tokens: int) -> str:
    if content_type == "grep":
        return _compact_grep(content, budget_tokens=budget_tokens)
    if content_type == "bash":
        return _compact_bash(content, budget_chars=max(200, budget_tokens * 4))
    if content_type == "tool_output":
        compact_json = _compact_json(content)
        if compact_json is not None:
            return compact_json
    max_chars = max(200, budget_tokens * 4)
    return _head_tail(content, max_chars=max_chars)


def compact(
    content: str,
    content_type: str = "unknown",
    budget_tokens: int = 500,
    *,
    recovery_hint: str | None = None,
    enable_llm: bool = False,
) -> CompactResult:
    """Compact tool output using char-based threshold + head/tail compression.

    Uses a char-based threshold (1800 chars by default) rather than token-based
    for predictability. LLM summarization is opt-in only — head+tail alone
    achieves the bulk of the token savings.

    Args:
        content:       Tool output to compact.
        content_type:  One of file, grep, bash, tool_output, unknown.
        budget_tokens: Target token budget for the compacted result.
        recovery_hint: How to get the full output if needed.
        enable_llm:    If True, attempt LLM summarization for large outputs
                       when Internal LLM is available. Adds latency; off by default.
    """
    original_tokens = count_tokens(content)
    hint = recovery_hint or "Re-run the original tool call or request the full output by path/range."

    # Passthrough: under the validated char threshold — no compression needed
    if len(content) <= DEFAULT_COMPRESS_THRESHOLD_CHARS:
        return CompactResult(
            compacted=content,
            original_tokens=original_tokens,
            compacted_tokens=original_tokens,
            recovery_hint=hint,
            method="passthrough",
            content_type=content_type,
        )

    method: CompactMethod = "deterministic_truncate"
    compacted = deterministic_truncate(content, content_type, budget_tokens)

    if enable_llm and original_tokens > 2000 and content_type != "grep":
        try:
            prompt = f"Recovery hint: {hint}\n\nOutput to summarize:\n{content}"
            compacted = summarize(prompt, max_tokens=budget_tokens)
            method = "llm_summary"
        except InternalLLMError:
            method = "deterministic_truncate"

    compacted_tokens = count_tokens(compacted)
    return CompactResult(
        compacted=compacted,
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        recovery_hint=hint,
        method=method,
        content_type=content_type,
    )


def compress_history(
    messages: list[dict[str, str]],
    *,
    keep_recent: int = 2,
    threshold_chars: int = DEFAULT_COMPRESS_THRESHOLD_CHARS,
    head_chars: int = DEFAULT_HEAD_KEEP_CHARS,
    tail_chars: int = DEFAULT_TAIL_KEEP_CHARS,
    stats: TokenSavingStats | None = None,
) -> list[dict[str, str]]:
    """Head+tail compress stale tool-output messages in a history list.

    The most recent ``keep_recent`` tool messages are exempt — the agent must
    see its active step at full fidelity (default of 2).

    Each message is expected to be a dict with at least a ``"role"`` key and
    a ``"content"`` key (LangChain / OpenAI message shape). Only messages with
    ``role == "tool"`` are candidates for compression.

    Args:
        messages:         The full message history list (modified in a new list).
        keep_recent:      How many of the most recent tool messages to exempt.
        threshold_chars:  Minimum content length before compression is applied.
        head_chars:       Characters to keep from the start of a tool message.
        tail_chars:       Characters to keep from the end of a tool message.
        stats:            Optional stats tracker — records each compression event.

    Returns:
        A new list of messages with stale tool outputs compressed.
    """
    # Identify tool-message indices in order.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    # The last `keep_recent` tool messages are exempt.
    exempt = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()

    result: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool" or i in exempt:
            result.append(msg)
            continue
        content = msg.get("content", "")
        compressed = compress_tool_output(
            content,
            threshold_chars=threshold_chars,
            head_chars=head_chars,
            tail_chars=tail_chars,
        )
        if stats is not None and compressed != content:
            orig_tokens = count_tokens(content)
            comp_tokens = count_tokens(compressed)
            stats.record(len(content), len(compressed), orig_tokens, comp_tokens)
        result.append({**msg, "content": compressed})
    return result


# --------------------------------------------------------------------------- #
# N6 — Savings gate for compact encoding                                       #
# --------------------------------------------------------------------------- #

# Default savings floor: only ship a compact form when it removes at least this
# fraction of the original JSON length. Below the floor, the original JSON is
# emitted unchanged — this guarantees compaction never inflates small or
# low-redundancy payloads.
DEFAULT_SAVINGS_THRESHOLD = 0.15


def savings_ratio(original: str, compact_form: str) -> float:
    """Fraction of characters removed by *compact_form* vs *original*.

    Returns 0.0 (never negative) when the compact form is not smaller, so an
    inflating encoding can never appear to "save".
    """
    original_len = len(original)
    if original_len <= 0:
        return 0.0
    saved = original_len - len(compact_form)
    if saved <= 0:
        return 0.0
    return saved / original_len


def gate_compact(
    original: str,
    compact_form: str,
    *,
    threshold: float = DEFAULT_SAVINGS_THRESHOLD,
) -> GateResult:
    """Pick the compact form only when it beats *original* by *threshold*.

    ``(len(original) - len(compact)) / len(original) >= threshold`` ships the
    compact form; otherwise the original is returned unchanged. This is the
    safety guard that makes aggressive encoding safe to enable by default —
    compaction NEVER inflates small / low-redundancy payloads.
    """
    ratio = savings_ratio(original, compact_form)
    use_compact = ratio >= threshold
    return GateResult(
        chosen=compact_form if use_compact else original,
        used_compact=use_compact,
        original_chars=len(original),
        compact_chars=len(compact_form),
        savings_ratio=ratio,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# N7 — Schema-driven columnar + string-intern encoding                        #
# --------------------------------------------------------------------------- #

# Self-describing header so the consumer / model can interpret the encoding.
COLUMNAR_FORMAT = "lemoncrow-columnar-v1"


def _row_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Stable union of keys across all rows, first-seen order preserved."""
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def columnar_encode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Encode a list of homogeneous-ish row dicts columnar with a string legend.

    Repeated string values (file paths, FQNs, etc.) are interned into a
    ``legend`` list; each column holds either the raw value or a ``{"$": idx}``
    reference into the legend. The result is self-describing (``format`` header
    plus ``columns``) and lossless: ``columnar_decode`` reconstructs the exact
    original rows including missing keys (encoded as a JSON null and decoded as
    an absent key only when it was absent originally — see below).

    Missing keys are preserved exactly: a column carries one entry per row, and
    a per-column ``present`` bitmap records which rows actually had the key, so
    a stored ``None`` value is distinguished from an absent key on decode.
    """
    keys = _row_keys(rows)
    legend: list[str] = []
    legend_index: dict[str, int] = {}
    # Count string occurrences so we only intern values that repeat — interning
    # a once-seen string would add legend bytes without removing redundancy.
    string_counts: dict[str, int] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str):
                string_counts[value] = string_counts.get(value, 0) + 1

    def _intern(value: str) -> int:
        idx = legend_index.get(value)
        if idx is None:
            idx = len(legend)
            legend.append(value)
            legend_index[value] = idx
        return idx

    columns: dict[str, list[Any]] = {}
    present: dict[str, list[int]] = {}
    for key in keys:
        col: list[Any] = []
        present_col: list[int] = []
        for row in rows:
            has_key = key in row
            present_col.append(1 if has_key else 0)
            value = row.get(key)
            if isinstance(value, str) and string_counts.get(value, 0) > 1:
                col.append({"$": _intern(value)})
            else:
                col.append(value)
        columns[key] = col
        present[key] = present_col

    return {
        "format": COLUMNAR_FORMAT,
        "n": len(rows),
        "keys": keys,
        "legend": legend,
        "columns": columns,
        "present": present,
    }


def columnar_decode(encoded: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct the exact original rows from :func:`columnar_encode` output."""
    if encoded.get("format") != COLUMNAR_FORMAT:
        raise ValueError(f"unsupported columnar format: {encoded.get('format')!r}")
    n = int(encoded.get("n") or 0)
    keys = encoded.get("keys") or []
    legend = encoded.get("legend") or []
    columns = encoded.get("columns") or {}
    present = encoded.get("present") or {}

    def _resolve(value: Any) -> Any:
        if isinstance(value, dict) and set(value.keys()) == {"$"}:
            return legend[int(value["$"])]
        return value

    rows: list[dict[str, Any]] = []
    for i in range(n):
        row: dict[str, Any] = {}
        for key in keys:
            present_col = present.get(key) or []
            if i < len(present_col) and not present_col[i]:
                continue  # key was absent in the original row
            col = columns.get(key) or []
            if i < len(col):
                row[key] = _resolve(col[i])
        rows.append(row)
    return rows


def columnar_encode_json(rows: list[dict[str, Any]]) -> str:
    """Compact-JSON serialise the columnar encoding of *rows*."""
    return json.dumps(columnar_encode(rows), ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "COLUMNAR_FORMAT",
    "DEFAULT_SAVINGS_THRESHOLD",
    "CompactResult",
    "GateResult",
    "TokenSavingStats",
    "columnar_decode",
    "columnar_encode",
    "columnar_encode_json",
    "compact",
    "compress_history",
    "compress_tool_output",
    "deterministic_truncate",
    "gate_compact",
    "savings_ratio",
]
