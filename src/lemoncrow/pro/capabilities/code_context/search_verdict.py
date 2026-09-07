"""Search verdicts: turn an empty search/grep result into an honest signal.

A bare empty result is ambiguous -- the agent cannot tell "the answer is not in
this tree" from "I have not looked hard enough," so on hard tasks it keeps
searching/spawning without bound. Stamping every result with a *verdict* (plus a
one-line ``next`` hint) closes that gap:

- ``found``  -- at least one hit.
- ``dark``   -- empty, but a channel that could have contributed was off
                (semantic embedder unset / zoekt daemon down); the empty is not
                trustworthy until the channel is enabled.
- ``missed`` -- empty, every live channel ran, but this is the first phrasing of
                the concept; reformulate before concluding anything.
- ``absent`` -- empty, every live channel ran, and >= ``threshold`` *distinct*
                phrasings of the same concept were all empty; safe to stop
                searching and derive from code already read.

``absent`` is deliberately gated behind repeated, distinct reformulations: a
single empty almost always means the query missed, not that the symbol is
absent, and pairing a single-empty "stop" with a derive-from-memory persona
turns a search spiral into a hallucination. Requiring reformulation is what makes
the verdict safe to act on. The threshold is tunable (env override) so benchmark
calibration can move it without a code change.

This module is pure: it computes verdicts from a hit count, channel health, and a
per-session memory of prior empty queries. Wiring (result packing, channel
probes, session storage) lives in the engine and MCP layers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

Verdict = Literal["found", "missed", "absent", "dark"]

_DEFAULT_REFORMULATION_THRESHOLD = 2
_DEFAULT_BREAKER_THRESHOLD = 6
_MAX_TRACKED_EMPTIES = 24
# Two query token-sets count as the same "area" when they share at least one
# token, and as distinct phrasings when their Jaccard similarity is below this cap
# (so a near-identical re-run is NOT counted as a fresh reformulation).
_DISTINCT_JACCARD_MAX = 0.6

_WORD_RE = re.compile(r"\w+")


def reformulation_threshold() -> int:
    """Distinct empty phrasings of one area required before a verdict of ``absent``.

    Tunable via ``LEMONCROW_SEARCH_REFORMULATION_THRESHOLD`` so benchmark
    calibration can move it without a code change; falls back to the default on a
    missing or malformed value.
    """
    raw = os.environ.get("LEMONCROW_SEARCH_REFORMULATION_THRESHOLD", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_REFORMULATION_THRESHOLD
        if value >= 1:
            return value
    return _DEFAULT_REFORMULATION_THRESHOLD


def breaker_threshold() -> int:
    """Consecutive unproductive searches before the circuit-breaker note fires.

    Phase 2 backstop: even when the agent ignores ``missed``/``absent``, a run of
    fruitless searches trips a soft "act on current evidence" notice. Tunable via
    ``LEMONCROW_SEARCH_BREAKER_THRESHOLD``; <= 0 disables the breaker.
    """
    raw = os.environ.get("LEMONCROW_SEARCH_BREAKER_THRESHOLD", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            return _DEFAULT_BREAKER_THRESHOLD
    return _DEFAULT_BREAKER_THRESHOLD


BREAKER_NOTE = "budget reached -- change from current evidence"


def normalize_query(query: str) -> frozenset[str]:
    """Lower-cased word-token set -- the unit of query identity and distinctness."""
    return frozenset(_WORD_RE.findall(query.lower()))


def _same_area(a: frozenset[str], b: frozenset[str]) -> bool:
    """Two queries concern the same concept when they share at least one token."""
    return bool(a & b)


def _distinct(a: frozenset[str], b: frozenset[str]) -> bool:
    """True when ``a`` is a genuinely different phrasing of ``b`` (not a re-run)."""
    if not a or not b:
        return a != b
    if a == b:
        return False
    union = len(a | b)
    if union == 0:
        return False
    return (len(a & b) / union) < _DISTINCT_JACCARD_MAX


@dataclass(frozen=True)
class ChannelHealth:
    """Liveness of the retrieval channels for one call.

    ``fts`` and literal ``grep`` are always live, so they are not tracked. For the
    optional channels, ``True`` = ran, ``False`` = the query wanted it but it was
    off (dark), ``None`` = not applicable to this query/mode (e.g. an identifier
    lookup never wanted semantic, so its absence is not "dark").
    """

    semantic: bool | None = None
    zoekt: bool | None = None

    def dark(self) -> tuple[str, ...]:
        """Channels the query wanted that were off -- the reason an empty is untrusted."""
        out: list[str] = []
        if self.semantic is False:
            out.append("semantic")
        if self.zoekt is False:
            out.append("zoekt")
        return tuple(out)


# Module-level singleton default (frozen, so shareable) -- avoids a call in the
# compute_verdict argument defaults (ruff B008).
_DEFAULT_CHANNELS = ChannelHealth()


@dataclass(frozen=True)
class VerdictResult:
    verdict: Verdict
    next: str

    def as_payload(self) -> dict[str, str]:
        """Model-facing shape: ``verdict`` always, ``next`` only when non-empty."""
        payload: dict[str, str] = {"verdict": self.verdict}
        if self.next:
            payload["next"] = self.next
        return payload


# A plain empty result, like vanilla `grep -rn`: state the absence, don't nudge a
# retry. The model decides whether to reformulate. (The escalation from `missed`
# to `absent` is kept for the breaker backstop -- repeated empties still trip
# `breaker_note` -- but neither carries a per-miss "reformulate" suggestion.)
_MISSED = VerdictResult("missed", "")
_ABSENT = VerdictResult("absent", "")


def compute_verdict(
    *,
    hit_count: int,
    query: str,
    channels: ChannelHealth = _DEFAULT_CHANNELS,
    prior_empties: tuple[frozenset[str], ...] = (),
    threshold: int | None = None,
) -> VerdictResult:
    """Classify a search/grep result. See module docstring for the four verdicts.

    ``prior_empties`` is this session's memory of earlier empty-result query
    token-sets (most recent last); the current query is *not* included.
    """
    if hit_count > 0:
        return VerdictResult("found", "")
    dark = channels.dark()
    if dark:
        label = " + ".join(dark)
        return VerdictResult("dark", f"{label} off -- empty unreliable")
    limit = reformulation_threshold() if threshold is None else threshold
    tokens = normalize_query(query)
    distinct_area_empties = sum(1 for prior in prior_empties if _same_area(prior, tokens) and _distinct(prior, tokens))
    # +1 accounts for the current empty phrasing itself.
    if distinct_area_empties + 1 >= limit:
        return _ABSENT
    return _MISSED


@dataclass
class SearchHistory:
    """Per-session memory of empty-result query phrasings (reformulation tracking).

    Mirror the MCP server's per-session monitor map: one instance per session id,
    bounded so a long-lived process cannot leak. A successful hit in an area
    prunes that area's empty streak so a later unrelated empty is not wrongly
    escalated to ``absent``.
    """

    empties: list[frozenset[str]] = field(default_factory=list)
    unproductive_streak: int = 0

    def prior_empties(self) -> tuple[frozenset[str], ...]:
        return tuple(self.empties)

    def record(self, query: str, *, found: bool) -> None:
        tokens = normalize_query(query)
        if found:
            # A hit ends the streak and clears the area's empty memory so a later
            # unrelated empty in the same area is not wrongly escalated.
            self.unproductive_streak = 0
            if tokens:
                self.empties = [e for e in self.empties if not _same_area(e, tokens)]
            return
        if not tokens:
            return
        self.unproductive_streak += 1
        self.empties.append(tokens)
        if len(self.empties) > _MAX_TRACKED_EMPTIES:
            del self.empties[: len(self.empties) - _MAX_TRACKED_EMPTIES]

    def breaker_tripped(self, threshold: int | None = None) -> bool:
        """True when the consecutive-unproductive run has reached the breaker."""
        limit = breaker_threshold() if threshold is None else threshold
        return limit > 0 and self.unproductive_streak >= limit
