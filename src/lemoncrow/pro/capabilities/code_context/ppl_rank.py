"""T10 coarse ranking: perplexity/entropy relevance of code chunks.

This is the *coarse* stage of perplexity/entropy-guided compression. Given a
blob of source code and an instruction, it:

1. Splits the code into chunks at natural function/class boundaries.
2. Scores each chunk with a per-chunk **utility** in ``[0, 1]`` that blends
   two signals:

   * **Information density** — structural Shannon entropy / token rarity of
     the chunk (:func:`lemoncrow.infra.internal_llm.chunk_entropy`), or real
     model perplexity when a backend is configured.
   * **Instruction relevance** — lexical token overlap between the chunk and
     the instruction, so on-task chunks outrank generic dense ones.

The whole stage is headless by default: with ``LEMONCROW_LLM_BACKEND=none`` it
uses only the structural entropy fallback — no model, no network. Real model
log-probabilities are used only when a backend is configured *and* the
``LEMONCROW_PERPLEXITY_COMPRESSION`` flag is on.

The per-chunk utility scores produced here are designed to feed the budget
optimizer's knapsack as an external utility source (T10 -> budget) and the
fine-grained token pruner (T11).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

from lemoncrow.infra.internal_llm import chunk_entropy, logprobs

__all__ = ["CodeChunk", "perplexity_compression_enabled", "rank_code_chunks"]

_FLAG = "LEMONCROW_PERPLEXITY_COMPRESSION"

# A new top-level definition (optionally decorated/async) starts a chunk. We
# treat any non-indented ``def``/``class`` as a boundary; decorators and
# leading blank/comment lines attach to the definition that follows.
_BOUNDARY_RE = re.compile(r"^(?:async\s+def|def|class)\b")
_DECORATOR_RE = re.compile(r"^@\S")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "in",
        "it",
        "to",
        "of",
        "and",
        "or",
        "for",
        "on",
        "with",
        "this",
        "that",
        "be",
        "as",
        "at",
        "by",
        "if",
        "def",
        "self",
        "return",
        "class",
    }
)


def perplexity_compression_enabled() -> bool:
    """True when the default-off ``LEMONCROW_PERPLEXITY_COMPRESSION`` flag is set."""
    raw = os.environ.get(_FLAG)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CodeChunk:
    """A function/class-boundary chunk of source code with its utility score.

    Attributes:
        id:          Stable identifier (``chunk{index}``) for the chunk.
        text:        The chunk source text.
        start_line:  1-based start line within the original blob.
        end_line:    1-based inclusive end line within the original blob.
        utility:     Blended relevance/density score in ``[0, 1]``.
        entropy:     Raw structural entropy/density signal for the chunk.
        relevance:   Instruction lexical-overlap signal in ``[0, 1]``.
        source:      Score provenance: ``'entropy'`` or ``'perplexity'``.
    """

    id: str
    text: str
    start_line: int
    end_line: int
    utility: float
    entropy: float
    relevance: float
    source: str
    metadata: dict[str, object] = field(default_factory=dict)

    def token_cost(self) -> int:
        """Rough token-count proxy (word-ish tokens) for budget weighting."""
        return max(1, len(_TOKEN_RE.findall(self.text)))


def _tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN_RE.findall(text)) if t not in _STOPWORDS and len(t) >= 3]


def split_code_chunks(code: str) -> list[tuple[str, int, int]]:
    """Split *code* at top-level function/class boundaries.

    Returns ``(text, start_line, end_line)`` tuples (1-based, inclusive).
    Leading decorators and the blank/comment lines immediately above a
    definition attach to that definition. Code before the first boundary
    (imports, module docstring, constants) forms its own leading chunk.
    """
    lines = code.splitlines()
    if not lines:
        return []

    # Identify boundary line indices (0-based), pulling decorators/comments up.
    starts: list[int] = []
    for i, line in enumerate(lines):
        if _BOUNDARY_RE.match(line):
            j = i
            while j - 1 >= 0:
                prev = lines[j - 1]
                stripped = prev.strip()
                if _DECORATOR_RE.match(prev) or stripped.startswith("#") or stripped == "":
                    j -= 1
                else:
                    break
            starts.append(j)

    starts = sorted(set(starts))
    if not starts or starts[0] != 0:
        starts = [0, *starts]

    chunks: list[tuple[str, int, int]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        if end <= start:
            continue
        text = "\n".join(lines[start:end])
        if not text.strip():
            continue
        chunks.append((text, start + 1, end))
    return chunks


def _perplexity_density(text: str, model: str | None) -> tuple[float, str]:
    """Return ``(density, source)`` using real logprobs when available.

    With a backend configured and the flag on, mean per-token perplexity
    (``exp(-mean_logprob)``) is the density signal. Otherwise the structural
    entropy fallback is used. ``source`` records which path ran.
    """
    if perplexity_compression_enabled():
        lps = logprobs(text, model)
        if lps:
            mean_lp = sum(lps) / len(lps)
            # Perplexity = exp(-mean logprob); higher => more surprising/dense.
            return math.exp(-mean_lp), "perplexity"
    return chunk_entropy(text), "entropy"


def rank_code_chunks(
    code: str,
    instruction: str = "",
    *,
    model: str | None = None,
    relevance_weight: float = 0.5,
) -> list[CodeChunk]:
    """Chunk *code* and score each chunk's utility for *instruction*.

    The utility blends normalized information density (entropy or perplexity)
    with lexical relevance to *instruction*::

        utility = (1 - relevance_weight) * density_norm
                  + relevance_weight * relevance

    When *instruction* is empty, utility is driven entirely by density. Both
    components are min-max normalized across the returned chunks so the
    utility lands in ``[0, 1]`` and is directly consumable by the knapsack.

    Works fully headless (entropy fallback) under ``LEMONCROW_LLM_BACKEND=none``.
    """
    raw = split_code_chunks(code)
    if not raw:
        return []

    instr_tokens = set(_tokenize(instruction))

    densities: list[float] = []
    relevances: list[float] = []
    source = "entropy"
    for text, _start, _end in raw:
        density, src = _perplexity_density(text, model)
        source = src
        densities.append(density)
        if instr_tokens:
            chunk_tokens = set(_tokenize(text))
            overlap = len(instr_tokens & chunk_tokens) / len(instr_tokens)
        else:
            overlap = 0.0
        relevances.append(overlap)

    density_norm = _min_max(densities)
    rel_norm = relevances if instr_tokens else [0.0] * len(relevances)
    weight = relevance_weight if instr_tokens else 0.0

    chunks: list[CodeChunk] = []
    for idx, (text, start, end) in enumerate(raw):
        utility = (1.0 - weight) * density_norm[idx] + weight * rel_norm[idx]
        chunks.append(
            CodeChunk(
                id=f"chunk{idx}",
                text=text,
                start_line=start,
                end_line=end,
                utility=round(utility, 6),
                entropy=round(densities[idx], 6),
                relevance=round(rel_norm[idx], 6),
                source=source,
            )
        )
    return chunks


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [1.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]
