from __future__ import annotations

import re

_BLANK_LINE = re.compile(r"\n\s*\n")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)


def minify_file_content(content: str) -> str:
    """Return a whitespace-stripped version of *content* for Survey/Plan phases.

    - Strips trailing whitespace from each line
    - Collapses 2+ blank lines to one blank line
    Preserves indentation and structure (not a tokenizer minifier).
    """
    content = _TRAILING_WHITESPACE.sub("", content)
    content = _BLANK_LINE.sub("\n\n", content)
    return content.strip()


def exact_file_content(content: str) -> str:
    """Return content unchanged (for Implement/edit phase)."""
    return content


__all__ = ["exact_file_content", "minify_file_content"]
