"""Token budget fitting for repo maps.

Both counters delegate to the canonical ``prompt_compilation.tokens`` module;
the names are kept re-exported here for existing importers.
"""

from __future__ import annotations

from collections.abc import Callable

from lemoncrow.pro.capabilities.prompt_compilation.tokens import (
    approx_tokens as _approx_tokens,
)
from lemoncrow.pro.capabilities.prompt_compilation.tokens import (
    count_tokens as _count_tokens,
)


def count_tokens(text: str) -> int:
    """Exact tiktoken BPE count (delegates to canonical ``count_tokens``)."""
    return _count_tokens(text)


def estimate_tokens(text: str) -> int:
    """Fast char-based token approximation for budget *gating* (delegates to
    canonical ``approx_tokens``). Unified from the former ceil(len/3.6) to
    ``approx_tokens``' len/4 -- a deliberate minor change: the binary search in
    ``fit_to_budget`` gates on the exact ``count_tokens`` and stays correct."""
    return _approx_tokens(text)


def fit_to_budget(
    ranked_files: list[str], render: Callable[[list[str]], str], budget_tokens: int
) -> tuple[list[str], str]:
    lo = 0
    hi = len(ranked_files)
    best_files: list[str] = []
    best_text = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        current_files = ranked_files[:mid]
        text = str(render(current_files))
        if count_tokens(text) <= budget_tokens:
            best_files = current_files
            best_text = text
            lo = mid + 1
        else:
            hi = mid - 1
    if not best_files and ranked_files:
        best_files = ranked_files[:1]
        best_text = str(render(best_files))
    return best_files, best_text


__all__ = ["count_tokens", "estimate_tokens", "fit_to_budget"]
