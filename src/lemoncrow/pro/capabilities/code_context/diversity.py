"""Filetype-diversity pass for ranked search surfaces.

A ranked window (fused file order, related_symbols, candidate_files) can be
monopolized by a single filetype -- in practice documentation (.md/.rst/...)
whose indexed headings match prose-y queries far more readily than code does,
crowding out the code files the caller actually wants. This module demotes the
*excess* same-type entries below the ranked window instead of dropping them:
a stable permutation, never a filter.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

# Documentation-class suffixes: prose files whose indexed headings/lines match
# natural-language queries far more readily than code symbols do.
DOC_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".rst", ".adoc", ".txt", ".ipynb"})

_DOC_QUERY_RE = re.compile(
    r"\b(?:docs?|documentation|readme|changelog|blog|markdown|guide|tutorial)\b" r"|\.(?:md|mdx|rst|adoc)\b",
    re.IGNORECASE,
)


def query_wants_docs(query: str) -> bool:
    """True when the query itself asks for documentation -- doc capping is skipped."""
    return bool(_DOC_QUERY_RE.search(query))


def is_doc_path(path: str) -> bool:
    dot = path.rfind(".")
    return dot >= 0 and path[dot:].lower() in DOC_SUFFIXES


def doc_cap(window: int) -> int:
    """Default share of a ranked window documentation may occupy (quarter, >=1)."""
    return max(1, window // 4)


def demote_doc_overflow[T](
    items: Sequence[T],
    *,
    window: int,
    cap: int | None = None,
    path_of: Callable[[T], str] = str,
) -> list[T]:
    """Stable permutation of *items*: within the head ``window`` keep at most
    ``cap`` documentation files (default ``doc_cap(window)``); excess docs are
    demoted below the window, never dropped. Relative order is preserved both
    inside the head and among the demoted/tail entries.
    """
    ordered = list(items)
    if window <= 0 or len(ordered) <= 1:
        return ordered
    limit = doc_cap(window) if cap is None else cap
    head: list[int] = []
    docs_in_head = 0
    for index, item in enumerate(ordered):
        if len(head) >= window:
            break
        if is_doc_path(path_of(item)):
            if docs_in_head >= limit:
                continue
            docs_in_head += 1
        head.append(index)
    if head == list(range(len(head))):
        return ordered  # nothing demoted
    picked = set(head)
    return [ordered[i] for i in head] + [item for i, item in enumerate(ordered) if i not in picked]


__all__ = ["DOC_SUFFIXES", "demote_doc_overflow", "doc_cap", "is_doc_path", "query_wants_docs"]
