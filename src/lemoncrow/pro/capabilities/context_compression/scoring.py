"""TF-IDF, recency, and keystone-aware event importance scoring.

Keystone preservation: events containing semantically critical spans
(conditionals, negations, exceptions, corrections) receive a boost so they
are never accidentally pruned by the budget-selection pass.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .models import EventScore

# Per-kind base importance weights
_KIND_WEIGHTS: dict[str, float] = {
    "error": 3.0,
    "exception": 3.0,
    "test_fail": 2.5,
    "validation_fail": 2.5,
    "file_edit": 2.0,
    "file_write": 2.0,
    "patch": 2.0,
    "tool_call": 1.0,
    "search": 0.8,
    "read_file": 0.7,
    "smart_read": 0.7,
    "observation": 1.2,
}

# Per-kind recency half-life (in event-count units).
# Errors/edits decay slowly — they remain relevant longer.
# Reads/searches decay quickly — they are cheap to redo.
_KIND_HALF_LIFE: dict[str, float] = {
    "error": 50.0,
    "exception": 50.0,
    "test_fail": 40.0,
    "validation_fail": 40.0,
    "file_edit": 30.0,
    "file_write": 30.0,
    "patch": 30.0,
    "tool_call": 10.0,
    "search": 5.0,
    "read_file": 5.0,
    "smart_read": 5.0,
    "observation": 15.0,
}

_DEFAULT_HALF_LIFE = 10.0

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
        "at",
        "by",
        "for",
        "on",
        "with",
        "from",
        "that",
        "this",
    }
)

# Penalty factor for repeated events of the same kind with similar summaries
_REPEAT_PENALTY = 0.5

# ---------------------------------------------------------------------------
# Keystone span detection
# ---------------------------------------------------------------------------
# These phrases signal decision branches, constraints, and corrections —
# tokens whose removal can flip the correct answer (Playbooks "keystone"
# concept).  Any event whose text matches gets a multiplicative boost so the
# greedy budget pass never silently drops it.

_KEYSTONE_PATTERNS: list[re.Pattern[str]] = [
    p
    for p in (
        # Conditional constraints: "only on weekdays", "only for admins", "only if flag set"
        re.compile(r"\bonly\b.{0,40}?\b(if|when|on|for|with|after|before|during|while|in|applies|works)\b", re.I),
        re.compile(r"\b(unless|except\s+when|except\s+if|except\s+for)\b", re.I),
        re.compile(r"\b(does\s+not\s+apply|not\s+applicable|disabled\s+for|skipped\s+for)\b", re.I),
        re.compile(r"\b(never|always\s+except|must\s+not|should\s+not|cannot|do\s+not)\b", re.I),
        re.compile(r"\b(however|but\s+(not|only|if|when)|instead\s+of|rather\s+than)\b", re.I),
        re.compile(r"\b(actually|in\s+fact|contrary\s+to|reverted|rolled\s+back|undone)\b", re.I),
        re.compile(r"\b(before\s+this\s+works|requires\s+first|depends\s+on|blocked\s+by)\b", re.I),
        re.compile(r"\b(at\s+most|at\s+least|exactly\s+\d|no\s+more\s+than|no\s+less\s+than)\b", re.I),
        re.compile(r"\b(fixed\s+(by|in|with)|introduced\s+(by|in)|caused\s+by|root\s+cause)\b", re.I),
        re.compile(r"\b(temporary|temporary\s+fix|workaround|fallback|regression)\b", re.I),
        re.compile(r"\b(not\s+applicable|disabled\s+when|enabled\s+only|restricted\s+to)\b", re.I),
    )
]

# Keystone boost tiers: each matching pattern adds a fixed increment
_KEYSTONE_BOOST_PER_MATCH = 0.5
_KEYSTONE_BOOST_MAX = 3.0  # cap at 3x to avoid drowning out everything else


def _keystone_boost(text: str) -> tuple[float, bool]:
    """Return (multiplicative_boost, is_protected) for a text span.

    boost is 1.0 (no effect) when no patterns match, up to _KEYSTONE_BOOST_MAX.
    is_protected is True when any pattern matched.
    """
    matches = sum(1 for p in _KEYSTONE_PATTERNS if p.search(text))
    if matches == 0:
        return 1.0, False
    boost = min(1.0 + matches * _KEYSTONE_BOOST_PER_MATCH, _KEYSTONE_BOOST_MAX)
    return boost, True


def _task_relevance_score(tokens: list[str], task_tokens: set[str]) -> float:
    """Token overlap between an event's terms and the current task query.

    Returns a score in [0, 1]: fraction of task tokens that appear in the
    event summary.  Provides light task-conditioning without an LLM.
    """
    if not task_tokens:
        return 0.0
    matched = sum(1 for t in task_tokens if t in tokens)
    return matched / len(task_tokens)


def _tokenise(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_]*", text.lower()) if t not in _STOPWORDS and len(t) >= 3]


def _build_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    if N == 0:
        return {}
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}


def score_events(
    events: list[dict[str, Any]],
    *,
    task: str = "",
) -> list[EventScore]:
    """
    Assign an importance score to each event.

    Score components:
    1. Kind weight (errors/edits > searches/reads)
    2. TF-IDF rarity of event summary tokens across the whole event log
    3. Recency: kind-specific exponential decay (errors decay slowly, reads fast)
    4. Error-chain boost: events immediately preceding an error get +50% weight
    5. Repeat penalty: second occurrence of same-kind with similar summary gets 0.5x
    6. Keystone boost: events containing critical decision spans get up to 3x boost
       and are marked keystone_protected — these are never silently dropped
    7. Task relevance: token overlap with the current task query adds up to +0.3

    Args:
        events: Ordered list of ledger event dicts.
        task:   Optional free-text description of the current task.  When
                provided, events whose summaries overlap with task terms score
                higher, improving task-conditioned context retention.
    """
    if not events:
        return []

    summaries = [str(ev.get("summary", "")) for ev in events]
    corpus = [_tokenise(s) for s in summaries]
    idf = _build_idf(corpus)
    N = len(events)

    # Pre-compute task tokens for task-relevance scoring
    task_tokens: set[str] = set(_tokenise(task)) if task else set()

    # Identify error-preceding indices for chain boosting
    error_precede: set[int] = set()
    for i, ev in enumerate(events):
        kind = str(ev.get("kind", "")).lower()
        if any(kind.startswith(k) for k in ("error", "exception", "test_fail", "validation_fail")) and i > 0:
            error_precede.add(i - 1)

    # Track seen (kind, summary_prefix) pairs to apply repeat penalties
    seen_kinds: dict[str, int] = {}  # key → first-seen index

    scored: list[EventScore] = []
    for idx, (ev, tokens) in enumerate(zip(events, corpus, strict=False)):
        kind = str(ev.get("kind", "")).lower()

        # Kind weight
        kind_w = 1.0
        matched_kind = ""
        for prefix, weight in _KIND_WEIGHTS.items():
            if kind.startswith(prefix):
                kind_w = weight
                matched_kind = prefix
                break

        # Kind-specific recency half-life
        half_life = _KIND_HALF_LIFE.get(matched_kind, _DEFAULT_HALF_LIFE)
        recency = math.exp(-(N - 1 - idx) / half_life)

        # TF-IDF rarity score
        tfidf = sum(idf.get(t, 0.0) for t in tokens)

        # Error-chain boost
        chain_boost = 1.5 if idx in error_precede else 1.0

        # Repeat penalty: same kind + same first 80 chars of summary → penalise repeat
        dedup_key = f"{kind}:{str(ev.get('summary', ''))[:80]}"
        repeat_factor = 1.0
        if dedup_key in seen_kinds:
            repeat_factor = _REPEAT_PENALTY
        else:
            seen_kinds[dedup_key] = idx

        # Keystone boost: check summary + first 400 chars of payload for critical spans
        summary_text = str(ev.get("summary", ""))
        payload_text = str(ev.get("payload", ""))[:400]
        k_boost, k_protected = _keystone_boost(f"{summary_text} {payload_text}")

        # Task relevance: small additive bonus for task-aligned events
        task_rel = _task_relevance_score(tokens, task_tokens)

        score = kind_w * chain_boost * repeat_factor * k_boost * (1.0 + tfidf + 0.3 * task_rel) * (0.5 + 0.5 * recency)
        reason = (
            f"kind_w={kind_w:.1f} recency={recency:.2f} tfidf={tfidf:.2f} "
            f"chain={chain_boost:.1f} repeat={repeat_factor:.1f} "
            f"keystone={k_boost:.1f} task_rel={task_rel:.2f}"
        )
        scored.append(EventScore(event=ev, score=score, reason=reason, keystone_protected=k_protected))

    return scored
