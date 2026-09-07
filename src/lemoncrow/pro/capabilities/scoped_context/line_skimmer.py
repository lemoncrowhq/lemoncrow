"""Goal-conditioned per-line skimmer for scoped pull-context (T12).

The scoped-context pull (``pull.py``) retrieves goal-relevant *chunks* and
symbols and neurally reranks them, but never prunes a chunk body down to the
lines that actually matter for the subtask. This module closes that gap: given a
goal (the :class:`Subtask` description + keywords + affected paths) and a chunk
body, it scores each *line* by relevance and drops lines the subtask does not
need, while always keeping structural anchors (``def``/``class``/decorator/
signature lines) so the surviving text stays parseable.

Design (pragmatic headless v1 -- no bespoke model):

* **Lexical** scoring (token overlap against the goal) is the always-available
  baseline and the fallback when no embedding backend is configured.
* **Embedding** scoring reuses the existing
  :class:`~lemoncrow.pro.capabilities.code_context.embedding.SemanticSearchRanker`
  (cosine similarity between the goal embedding and each line embedding) at line
  granularity. When the ranker is unavailable (``ranker.available`` is False, or
  none is supplied) the skimmer degrades cleanly to lexical-only.

The whole feature is **default-off**: callers gate on :func:`is_line_skim_enabled`
(env ``LEMONCROW_LINE_SKIM``). With the flag off, chunk output is byte-for-byte
unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lemoncrow.core.environment import bool_env
from lemoncrow.infra.internal_llm.exceptions import OllamaUnavailable
from lemoncrow.infra.storage.vector import cosine_similarity

# Embedding backends fail in narrow, known ways (backend down, malformed vector);
# we degrade to lexical on any of these rather than propagate.
_EMBED_ERRORS = (OllamaUnavailable, OSError, ValueError, TypeError)

_LINE_SKIM_ENV = "LEMONCROW_LINE_SKIM"

# Default keep threshold: a line survives on its own merit when its relevance
# score is at least this fraction of the best-scoring line in the chunk.
_DEFAULT_KEEP_RATIO = 0.35
# Lines on either side of a kept line that we also keep, so a relevant statement
# does not lose the line that sets up or consumes it.
_DEFAULT_NEIGHBOR_WINDOW = 1
# Skip skimming tiny bodies: there is nothing to save and anchors dominate.
_MIN_LINES_TO_SKIM = 6

# Structural anchors: lines that define the parse skeleton. Always kept so the
# output remains syntactically navigable even after aggressive pruning.
_ANCHOR_PREFIXES = (
    "def ",
    "async def ",
    "class ",
    "@",  # decorators
    "import ",
    "from ",
)
# Anchor keywords detected anywhere a stripped line starts with them (covers
# common non-Python signature/structural lines too).
_ANCHOR_KEYWORDS = (
    "def ",
    "async def ",
    "class ",
    "interface ",
    "struct ",
    "func ",
    "function ",
    "public ",
    "private ",
    "protected ",
)

_TOKEN_SPLIT_RE = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def is_line_skim_enabled(env: dict[str, str] | None = None) -> bool:
    """Return True when the per-line skimmer is enabled (default-off)."""
    return bool_env(_LINE_SKIM_ENV, default=False, env=env)


def _tokens(value: str) -> set[str]:
    """Tokenize identifiers and prose into a lowercase token set.

    Mirrors ``pull.py._tokens`` (camelCase split, alnum runs, len>=2) so the
    line skimmer and the chunk-level boost speak the same vocabulary.
    """
    if not value:
        return set()
    normalized = _TOKEN_SPLIT_RE.sub(r"\1 \2", value)
    return {token.lower() for token in _TOKEN_RE.findall(normalized) if len(token) >= 2}


def build_goal_text(subtask: Any) -> str:
    """Compose the goal string from a subtask's description + keywords + paths."""
    parts: list[str] = [str(getattr(subtask, "description", "") or "")]
    parts.extend(str(k) for k in getattr(subtask, "keywords", []) or [])
    parts.extend(str(p) for p in getattr(subtask, "affected_paths", []) or [])
    return " ".join(part for part in parts if part).strip()


def is_structural_anchor(line: str) -> bool:
    """Return True when *line* is a structural anchor that must be preserved."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(_ANCHOR_PREFIXES):
        return True
    return stripped.startswith(_ANCHOR_KEYWORDS)


@dataclass
class SkimResult:
    """Outcome of skimming one chunk body."""

    text: str
    kept_lines: int
    total_lines: int
    dropped_lines: int
    used_embedding: bool
    elision_marker: str = ""

    @property
    def changed(self) -> bool:
        return self.dropped_lines > 0


@dataclass
class LineSkimmer:
    """Score chunk lines against a goal and keep only the relevant ones.

    ``ranker`` is any object exposing ``available: bool``, ``embed_query(str) ->
    list[float]`` and (optionally) embedding for arbitrary text -- in production
    the engine's :class:`SemanticSearchRanker`. When it is ``None`` or reports
    ``available is False`` the skimmer scores lexically only.
    """

    ranker: Any = None
    keep_ratio: float = _DEFAULT_KEEP_RATIO
    neighbor_window: int = _DEFAULT_NEIGHBOR_WINDOW
    min_lines: int = _MIN_LINES_TO_SKIM
    elision_marker: str = ""
    _goal_cache: dict[str, tuple[set[str], list[float]]] = field(default_factory=dict, repr=False)

    @property
    def _embedding_available(self) -> bool:
        return bool(self.ranker is not None and getattr(self.ranker, "available", False))

    def _goal_signal(self, goal: str) -> tuple[set[str], list[float]]:
        cached = self._goal_cache.get(goal)
        if cached is not None:
            return cached
        goal_tokens = _tokens(goal)
        goal_vec: list[float] = []
        if self._embedding_available:
            try:
                goal_vec = list(self.ranker.embed_query(goal) or [])
            except _EMBED_ERRORS:
                goal_vec = []
        signal = (goal_tokens, goal_vec)
        self._goal_cache[goal] = signal
        return signal

    def _lexical_score(self, line: str, goal_tokens: set[str]) -> float:
        if not goal_tokens:
            return 0.0
        line_tokens = _tokens(line)
        if not line_tokens:
            return 0.0
        overlap = goal_tokens & line_tokens
        if not overlap:
            return 0.0
        # Normalize by the line's own size so a short, on-topic line is not
        # penalised against a long line that merely mentions one goal token.
        return len(overlap) / float(len(line_tokens))

    def _embedding_score(self, line: str, goal_vec: list[float]) -> float | None:
        if not goal_vec:
            return None
        embed_text = getattr(self.ranker, "embed_text", None)
        if not callable(embed_text):
            embed_text = getattr(self.ranker, "embed_query", None)
        if not callable(embed_text):
            return None
        try:
            line_vec = list(embed_text(line) or [])
        except _EMBED_ERRORS:
            return None
        if not line_vec or len(line_vec) != len(goal_vec):
            return None
        return float(cosine_similarity(goal_vec, line_vec))

    def skim(self, body: str, goal: str) -> SkimResult:
        """Prune *body* to the goal-relevant lines + anchors + neighbor window."""
        if not body:
            return SkimResult(text=body, kept_lines=0, total_lines=0, dropped_lines=0, used_embedding=False)
        lines = body.splitlines()
        total = len(lines)
        goal_stripped = goal.strip()
        # Nothing to do: trivial body, or an empty goal would drop everything.
        if total < self.min_lines or not goal_stripped:
            return SkimResult(
                text=body,
                kept_lines=total,
                total_lines=total,
                dropped_lines=0,
                used_embedding=False,
            )

        goal_tokens, goal_vec = self._goal_signal(goal_stripped)
        used_embedding = False
        scores: list[float] = []
        for line in lines:
            lex = self._lexical_score(line, goal_tokens)
            emb = self._embedding_score(line, goal_vec)
            if emb is None:
                scores.append(lex)
            else:
                used_embedding = True
                # Blend: embedding leads, lexical guarantees exact-keyword hits.
                scores.append(max(emb, lex))

        best = max(scores) if scores else 0.0
        # With no positive signal anywhere, do not gut the chunk -- keep it whole
        # rather than collapse to anchors only.
        if best <= 0.0:
            return SkimResult(
                text=body,
                kept_lines=total,
                total_lines=total,
                dropped_lines=0,
                used_embedding=used_embedding,
            )

        threshold = best * self.keep_ratio
        keep = [False] * total
        for idx, (line, score) in enumerate(zip(lines, scores, strict=True)):
            if is_structural_anchor(line):
                keep[idx] = True
                continue
            if score >= threshold and score > 0.0:
                keep[idx] = True

        # Expand a small neighbor window around each on-merit / anchor keep.
        if self.neighbor_window > 0:
            seed = [i for i, k in enumerate(keep) if k]
            for i in seed:
                lo = max(0, i - self.neighbor_window)
                hi = min(total - 1, i + self.neighbor_window)
                for j in range(lo, hi + 1):
                    keep[j] = True

        kept_text_lines: list[str] = []
        dropped = 0
        prev_kept = True
        for idx, line in enumerate(lines):
            if keep[idx]:
                kept_text_lines.append(line)
                prev_kept = True
            else:
                dropped += 1
                if self.elision_marker and prev_kept:
                    kept_text_lines.append(self.elision_marker)
                prev_kept = False

        if dropped == 0:
            return SkimResult(
                text=body,
                kept_lines=total,
                total_lines=total,
                dropped_lines=0,
                used_embedding=used_embedding,
            )

        return SkimResult(
            text="\n".join(kept_text_lines),
            kept_lines=total - dropped,
            total_lines=total,
            dropped_lines=dropped,
            used_embedding=used_embedding,
            elision_marker=self.elision_marker,
        )


def skim_chunks(
    chunks: list[Any],
    *,
    goal: str,
    ranker: Any = None,
    keep_ratio: float = _DEFAULT_KEEP_RATIO,
    neighbor_window: int = _DEFAULT_NEIGHBOR_WINDOW,
) -> list[Any]:
    """Skim each chunk's ``snippet`` in place against *goal*; return *chunks*.

    Mutates the ``snippet`` attribute of each chunk that carries one. Chunks
    without a non-empty snippet are left untouched. Caller is responsible for
    gating on :func:`is_line_skim_enabled`.
    """
    if not goal.strip():
        return chunks
    skimmer = LineSkimmer(ranker=ranker, keep_ratio=keep_ratio, neighbor_window=neighbor_window)
    for chunk in chunks:
        snippet = getattr(chunk, "snippet", "") or ""
        if not snippet:
            continue
        result = skimmer.skim(snippet, goal)
        if result.changed:
            chunk.snippet = result.text
    return chunks


__all__ = [
    "LineSkimmer",
    "SkimResult",
    "build_goal_text",
    "is_line_skim_enabled",
    "is_structural_anchor",
    "skim_chunks",
]
