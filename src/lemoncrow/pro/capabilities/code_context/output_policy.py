"""Shared output policy profiles for code-context response shaping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

TRUNCATION_MARKER = "... [truncated]"


@dataclass(frozen=True)
class OutputPolicy:
    include_code: bool
    include_docstring: bool
    include_snippet: bool
    include_edges: bool
    max_results: int
    max_related_symbols: int
    max_code_blocks: int
    max_code_block_chars: int
    max_total_tokens: int
    max_symbols_per_file: int
    container_outline_only: bool


SEARCH_COMPACT = OutputPolicy(
    include_code=False,
    include_docstring=False,
    include_snippet=False,
    include_edges=False,
    max_results=8,
    max_related_symbols=0,
    max_code_blocks=0,
    max_code_block_chars=0,
    max_total_tokens=1400,
    max_symbols_per_file=2,
    container_outline_only=True,
)

RELATION_COMPACT = OutputPolicy(
    include_code=False,
    include_docstring=False,
    include_snippet=False,
    include_edges=True,
    max_results=12,
    max_related_symbols=12,
    max_code_blocks=0,
    max_code_block_chars=0,
    max_total_tokens=1700,
    max_symbols_per_file=3,
    container_outline_only=True,
)

CONTEXT_COMPACT = OutputPolicy(
    include_code=True,
    include_docstring=False,
    include_snippet=True,
    include_edges=True,
    max_results=16,
    max_related_symbols=8,
    max_code_blocks=3,
    max_code_block_chars=2000,  # per-section; total bounded by budget_tokens in engine
    max_total_tokens=5000,
    max_symbols_per_file=4,
    container_outline_only=True,
)

NODE_CODE_COMPACT = OutputPolicy(
    include_code=True,
    include_docstring=False,
    include_snippet=True,
    include_edges=False,
    max_results=1,
    max_related_symbols=0,
    max_code_blocks=1,
    max_code_block_chars=1800,
    max_total_tokens=1800,
    max_symbols_per_file=0,
    container_outline_only=False,
)

_POLICY_BY_OPERATION = {
    "search": SEARCH_COMPACT,
    "relation": RELATION_COMPACT,
    "context": CONTEXT_COMPACT,
    "node": NODE_CODE_COMPACT,
}


def resolve_output_policy(operation: str) -> OutputPolicy:
    return _POLICY_BY_OPERATION.get(operation, SEARCH_COMPACT)


def hard_cap_chars(
    text: str,
    max_chars: int,
    *,
    file_path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Cap *text* to *max_chars*, appending a truncation marker.

    When *start_line* and *end_line* are provided the marker includes the line
    range so the reader knows where the missing content sits::

        ... [truncated — L42-L80]

    The file path is intentionally omitted from the marker — it is already
    present in the surrounding section header, and repeating it (with the word
    "read") caused agents to reflexively issue read calls even when the
    skeleton they had was sufficient.
    """
    if max_chars <= 0:
        return TRUNCATION_MARKER
    if len(text) <= max_chars:
        return text
    marker = TRUNCATION_MARKER
    if max_chars <= len(marker):
        return marker
    if start_line is not None and end_line is not None:
        marker = f"... [truncated — L{start_line}-L{end_line}]"
    available_chars = max_chars - len(marker)
    cut = text[:available_chars]
    newline_floor = int(available_chars * 0.8)
    last_newline = cut.rfind("\n")
    if last_newline >= newline_floor:
        cut = cut[:last_newline]
    cut = cut.rstrip()
    if not cut:
        return marker
    return f"{cut}\n{marker}"


def cap_source_by_tokens(
    text: str,
    max_tokens: int,
    estimate_tokens: Callable[[str], int],
    *,
    start_line: int,
    end_line: int,
) -> str:
    """Cap line-numbered *text* to ~*max_tokens*, cutting on line boundaries.

    Tokens -- not chars -- because tokens are the context unit the caller pays:
    a char cap over-counts cheap line-number prefixes and clips dense bodies at
    the tail, where a function's actual behavior (its delegated call, its
    ``return``) lives. Whole lines are kept until the budget is spent; the
    marker then names the last line kept and the symbol's end line so any
    follow-up read is exact instead of guesswork::

        ... [truncated after L9547 — L9552]

    Long-line caveat: a single line whose own token cost exceeds the whole
    budget (minified/generated source, a giant literal) would either return an
    empty body -- when it is the first line -- or silently blow the budget. Such
    a first line is hard-cut mid-line with an inline ellipsis so its head
    survives and the cap holds; an over-long line further down simply becomes
    the cut point, with the lines before it returned whole.
    """
    if max_tokens <= 0:
        return TRUNCATION_MARKER
    if estimate_tokens(text) <= max_tokens:
        return text
    lines = text.split("\n")
    kept: list[str] = []
    used = 0
    last_line = start_line - 1  # line number of the last line kept (whole or partial)
    for idx, line in enumerate(lines):
        line_tokens = estimate_tokens(line)
        if used + line_tokens <= max_tokens:
            kept.append(line)
            used += line_tokens
            last_line = start_line + idx
            continue
        remaining = max_tokens - used
        if not kept and remaining > 0 and line_tokens > 0:
            # First line alone overflows the budget: hard-cut it mid-line so we
            # return its head instead of an empty body.
            keep_chars = max(1, len(line) * remaining // line_tokens)
            clipped = line[:keep_chars].rstrip()
            if clipped:
                kept.append(f"{clipped} ...")
                last_line = start_line + idx
        break
    marker = f"... [truncated after L{last_line} — L{end_line}]"
    if not kept:
        return marker
    return "\n".join(kept) + "\n" + marker


__all__ = [
    "CONTEXT_COMPACT",
    "NODE_CODE_COMPACT",
    "RELATION_COMPACT",
    "SEARCH_COMPACT",
    "TRUNCATION_MARKER",
    "OutputPolicy",
    "cap_source_by_tokens",
    "hard_cap_chars",
    "resolve_output_policy",
]
