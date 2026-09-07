"""T11 fine pruning: entropy/perplexity-guided line/token pruning.

The *fine* stage of perplexity/entropy-guided compression. T10 selects which
chunks to retain under a budget; T11 then trims the *inside* of an oversized
retained block, dropping the lowest-signal lines while keeping the high-signal
spans, until the block fits a target token budget.

Signal per line:

* **Structural entropy** (:func:`lemoncrow.infra.internal_llm.chunk_entropy`)
  scaled by line length — information content of the line. Used by default
  and whenever no backend is configured (headless v1).
* **Real perplexity** from per-token log-probabilities when a backend is
  configured *and* ``LEMONCROW_PERPLEXITY_COMPRESSION`` is on.

Keystone protection: lines matching decision/constraint/negation cues
(``only if``, ``unless``, ``must not``, ``return``, ``raise`` …) are pinned
and never dropped, mirroring the keystone concept used elsewhere in this
package so pruning never silently flips correctness.

Fully deterministic and model-free by default: with
``LEMONCROW_LLM_BACKEND=none`` only the structural entropy signal is used — no
model, no network.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

from lemoncrow.infra.internal_llm import chunk_entropy
from lemoncrow.infra.internal_llm.logprobs import logprobs

__all__ = ["LineScore", "PruneResult", "perplexity_compression_enabled", "prune_block"]

_FLAG = "LEMONCROW_PERPLEXITY_COMPRESSION"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")

# Lines whose removal can flip behaviour/meaning are pinned (never dropped),
# regardless of their entropy. Mirrors the keystone concept in scoring.py.
_KEYSTONE_RE = re.compile(
    r"\b(only\s+(if|when|for|on)|unless|except|must\s+not|should\s+not|cannot|"
    r"never|always|return|raise|assert|yield|if|elif|else|with|try|finally)\b",
    re.I,
)


def perplexity_compression_enabled() -> bool:
    """True when the default-off ``LEMONCROW_PERPLEXITY_COMPRESSION`` flag is set."""
    raw = os.environ.get(_FLAG)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


@dataclass
class LineScore:
    """Per-line scoring record used by the pruner."""

    index: int
    text: str
    tokens: int
    signal: float
    protected: bool


@dataclass
class PruneResult:
    """Outcome of pruning a single block.

    Attributes:
        text:            The pruned block text (kept lines in original order).
        kept_lines:      Original 0-based indices of retained lines.
        dropped_lines:   Original 0-based indices of removed lines.
        original_tokens: Token count before pruning.
        pruned_tokens:   Token count after pruning.
        source:          Signal provenance: ``'entropy'`` or ``'perplexity'``.
    """

    text: str
    kept_lines: list[int]
    dropped_lines: list[int]
    original_tokens: int
    pruned_tokens: int
    source: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.pruned_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "original_tokens": self.original_tokens,
            "pruned_tokens": self.pruned_tokens,
            "saved_tokens": self.saved_tokens,
            "kept": len(self.kept_lines),
            "dropped": len(self.dropped_lines),
            "source": self.source,
        }


def _line_signal_entropy(line: str) -> float:
    """Structural information content of a line (entropy scaled by length)."""
    stripped = line.strip()
    if not stripped:
        return 0.0
    # chunk_entropy rewards varied/rare tokens; scale by token count so a long
    # dense line outranks a short one of equal per-token entropy.
    return chunk_entropy(stripped) * math.log2(_token_count(stripped) + 2)


def _line_signal_perplexity(line: str, model: str | None) -> float | None:
    """Real per-line perplexity from logprobs; ``None`` if unavailable."""
    stripped = line.strip()
    if not stripped:
        return 0.0
    lps = logprobs(stripped, model)
    if not lps:
        return None
    mean_lp = sum(lps) / len(lps)
    return math.exp(-mean_lp) * math.log2(_token_count(stripped) + 2)


def _score_lines(lines: list[str], model: str | None) -> tuple[list[LineScore], str]:
    use_ppl = perplexity_compression_enabled()
    source = "entropy"
    scores: list[LineScore] = []
    for idx, line in enumerate(lines):
        signal: float | None = None
        if use_ppl:
            signal = _line_signal_perplexity(line, model)
            if signal is not None:
                source = "perplexity"
        if signal is None:
            signal = _line_signal_entropy(line)
        protected = bool(_KEYSTONE_RE.search(line))
        scores.append(
            LineScore(
                index=idx,
                text=line,
                tokens=_token_count(line),
                signal=signal,
                protected=protected,
            )
        )
    return scores, source


def prune_block(
    text: str,
    target_tokens: int,
    *,
    model: str | None = None,
) -> PruneResult:
    """Prune *text* down toward *target_tokens*, keeping high-signal lines.

    Coarse-to-fine: lines are scored by entropy (or real perplexity when the
    flag is on and a backend is configured), then the lowest-signal
    *unprotected* lines are dropped greedily until the block fits
    *target_tokens*. Keystone lines (decision/constraint/control-flow cues)
    are never dropped. Surviving lines are returned in original order.

    If the block already fits, it is returned unchanged. If even the protected
    lines exceed the target, all protected lines plus the highest-signal
    unprotected lines that still fit are kept (the target is a soft floor that
    never discards correctness-critical spans).

    Headless-safe: under ``LEMONCROW_LLM_BACKEND=none`` only structural entropy
    drives pruning — no model, no network.
    """
    lines = text.splitlines()
    original_tokens = _token_count(text)
    scores, source = _score_lines(lines, model)

    if original_tokens <= target_tokens or not lines:
        return PruneResult(
            text=text,
            kept_lines=list(range(len(lines))),
            dropped_lines=[],
            original_tokens=original_tokens,
            pruned_tokens=original_tokens,
            source=source,
        )

    # Always keep protected lines; fill remaining budget with the highest-signal
    # unprotected lines (densest spans first).
    kept: set[int] = {s.index for s in scores if s.protected}
    used = sum(scores[i].tokens for i in kept)

    unprotected = sorted(
        (s for s in scores if not s.protected),
        key=lambda s: (s.signal, -s.index),
        reverse=True,
    )
    for s in unprotected:
        if used + s.tokens <= target_tokens:
            kept.add(s.index)
            used += s.tokens

    kept_sorted = sorted(kept)
    dropped = [i for i in range(len(lines)) if i not in kept]
    pruned_text = "\n".join(lines[i] for i in kept_sorted)

    return PruneResult(
        text=pruned_text,
        kept_lines=kept_sorted,
        dropped_lines=dropped,
        original_tokens=original_tokens,
        pruned_tokens=_token_count(pruned_text),
        source=source,
    )
