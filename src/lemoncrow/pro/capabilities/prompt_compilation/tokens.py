"""Canonical token counting for the whole repo.

Three functions, by intent:

* ``count_tokens``   -- EXACT tiktoken BPE. The single counter for anything
  billed/measured (cost, ledger, compaction accounting). Falls back to char/4
  only when tiktoken is unavailable.
* ``approx_tokens``  -- FAST char-only estimate (``len(strip())//4``). No
  tiktoken: for hot-path budget *gating* that must not pay the BPE cost.
* ``estimate_tokens`` -- accurate general estimate (tiktoken cl100k_base with a
  char/4 fallback). Reserved-``model`` signature for future per-provider
  tokenizers.

All three share the module-cached ``_get_encoding()`` where they use tiktoken.
Pattern is consistent with core/capabilities/repo_map/budget.py.
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["approx_tokens", "count_tokens", "estimate_tokens"]


class _Encoder(Protocol):
    def encode(self, text: str) -> list[int]: ...


class _UnavailableEncoding:
    pass


_UNAVAILABLE = _UnavailableEncoding()
_ENCODING: _Encoder | _UnavailableEncoding | None = None


def _get_encoding() -> _Encoder | None:
    """Return the tiktoken encoding, loading it once on first call."""
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except (AttributeError, ImportError, KeyError, ValueError):  # pragma: no cover
            _ENCODING = _UNAVAILABLE
    if isinstance(_ENCODING, _UnavailableEncoding):
        return None
    return _ENCODING


def estimate_tokens(text: str, model: str | None = None) -> int:
    """Estimate the token count of *text*.

    Args:
        text: The text to estimate.
        model: Ignored in this implementation (reserved for future per-provider
               tokenizers). The cl100k_base encoding is used regardless.

    Returns:
        An integer token estimate ≥ 0.
    """
    del model
    if not text:
        return 0
    enc = _get_encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Char/4 fallback — within ~15% for English prose and code.
    return max(1, len(text) // 4)


def count_tokens(text: str) -> int:
    """Exact tiktoken BPE token count — the single counter for billing/cost.

    Uses the module-cached cl100k_base encoder. Falls back to ``len(text)//4``
    (0 for empty) only when the encoder is unavailable.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def approx_tokens(text: str) -> int:
    """Fast char-only token estimate for hot-path budget *gating*.

    Deliberately does NOT use tiktoken (its BPE encode dominates retrieval hot
    paths). Mirrors the strip()+empty→0 behavior of the former
    ``session_parsers._common.char_tokens`` helper: whitespace-only ⇒ 0,
    otherwise ``max(1, len(text.strip()) // 4)``.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)
