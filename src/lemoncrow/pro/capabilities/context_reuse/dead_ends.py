"""Dead-end approach tracker for reasoning reuse."""

from __future__ import annotations

import re


def _normalise_approach(text: str) -> str:
    """Normalise an approach description for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    return " ".join(text.split())


_MAX_DEAD_ENDS = 2000


class DeadEndTracker:
    """
    Tracks approaches that have been tried and failed.

    When ranking procedures, dead-end approaches are penalised so the agent
    does not retry strategies that are known to not work.
    """

    def __init__(self) -> None:
        # Insertion-ordered set (dict keys) so the tracker can evict its oldest
        # entries once it exceeds _MAX_DEAD_ENDS instead of growing unbounded for
        # the life of the capability instance.
        self._dead_ends: dict[str, None] = {}

    def mark_dead_end(self, approach: str) -> None:
        """Mark an approach as a dead end."""
        key = _normalise_approach(approach)
        self._dead_ends.pop(key, None)
        self._dead_ends[key] = None
        while len(self._dead_ends) > _MAX_DEAD_ENDS:
            self._dead_ends.pop(next(iter(self._dead_ends)))

    def is_dead_end(self, approach: str) -> bool:
        """Return True if the approach is known to be a dead end."""
        normalised = _normalise_approach(approach)
        if normalised in self._dead_ends:
            return True
        # Fuzzy check: Jaccard overlap with a known dead-end. Single-token
        # dead-ends require an exact match (handled above) so they cannot
        # poison every approach that happens to share that one token.
        tokens = set(normalised.split())
        for de in self._dead_ends:
            de_tokens = set(de.split())
            if len(de_tokens) < 2:
                continue
            union = tokens | de_tokens
            if not union:
                continue
            if len(tokens & de_tokens) / len(union) >= 0.6:
                return True
        return False

    def all_dead_ends(self) -> list[str]:
        return sorted(self._dead_ends)

    def dead_end_penalty(self, approach: str) -> float:
        """Return 0.0 (no penalty) or 0.8 (heavy penalty) for dead ends."""
        return 0.8 if self.is_dead_end(approach) else 0.0
