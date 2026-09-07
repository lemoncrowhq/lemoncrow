"""Six trajectory monitors for agent health assessment.

Pure-Python implementations of six trajectory-health monitors. Each monitor
scores a single dimension in [0, 1]
where 1 = maximum badness.

Monitors:
    semantic_loop         — reworded repetition in recent outputs      (weight: 0.25)
    verification_skip     — concluding without checking                 (weight: 0.15)
    claim_contradiction   — self-contradictory conclusions              (weight: 0.30)
    cyclic_compression    — context churn (re-stating earlier context)  (weight: 0.10)
    late_sprawl           — late-run scope creep                        (weight: 0.10)
    silent_topic_drift    — quiet off-task drift                        (weight: 0.10)

The composite score is a weighted sum.  A composite > 0.15 suggests the
agent needs steering.  A composite > 0.35 suggests a significant failure mode.

Usage::

    from lemoncrow.pro.capabilities.monitors.suite import evaluate_all, DEFAULT_WEIGHTS

    result = evaluate_all(
        steps=agent_step_texts,       # list[str] — one str per agent turn
        task=original_task_text,      # str
    )
    if result.composite > 0.15:
        print(result.fired)           # list of monitor names that reached threshold
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Weights (coding profile)                #
# --------------------------------------------------------------------------- #

DEFAULT_WEIGHTS: dict[str, float] = {
    "claim_contradiction": 0.30,
    "semantic_loop": 0.25,
    "verification_skip": 0.15,
    "cyclic_compression": 0.10,
    "late_sprawl": 0.10,
    "silent_topic_drift": 0.10,
}

FIRE_THRESHOLD: float = 0.40  # score ≥ this → monitor fires


# --------------------------------------------------------------------------- #
# Result dataclass                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class MonitorResult:
    """Aggregate result of evaluating all six trajectory monitors.

    Attributes:
        scores:     Per-monitor score in [0, 1].
        composite:  Weighted sum across all monitors.
        fired:      Names of monitors whose score reached FIRE_THRESHOLD.
        failure_type: Highest-weight fired monitor name, or None.
    """

    scores: dict[str, float]
    composite: float
    fired: list[str] = field(default_factory=list)
    failure_type: str | None = None


# --------------------------------------------------------------------------- #
# Individual monitor implementations                                           #
# --------------------------------------------------------------------------- #


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _semantic_loop(steps: Sequence[str], window: int = 4) -> float:
    """Detect reworded repetition in the last `window` steps.

    Returns the max pairwise bigram Jaccard similarity among recent step pairs,
    excluding identical duplicates (those are covered by cyclic_compression).
    """
    recent = list(steps[-window:])
    if len(recent) < 2:
        return 0.0
    bigrams = [_bigrams(_tokenize(s)) for s in recent]
    max_sim = 0.0
    for i in range(len(bigrams)):
        for j in range(i + 1, len(bigrams)):
            if recent[i] == recent[j]:
                continue  # exact duplicate handled by cyclic_compression
            sim = _jaccard(bigrams[i], bigrams[j])
            max_sim = max(max_sim, sim)
    return min(1.0, max_sim)


# Patterns that indicate the agent is verifying / checking its work.
# Note: avoid matching noun uses like "null check", "type check" — require
# verb context (checking/ran/run or subject "I").
_VERIFY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(test|verify|confirm|assert|validate)\b", re.IGNORECASE),
    re.compile(r"\b(ran|run) (the )?tests?\b", re.IGNORECASE),
    re.compile(r"\b(I |let me )(check|confirm|verify|validate)\b", re.IGNORECASE),
    re.compile(r"\b(make sure|double.check|ensure)\b", re.IGNORECASE),
    re.compile(r"\bresult[s]?\s+(show|indicate|confirm)\b", re.IGNORECASE),
]

# Patterns that indicate a final conclusion without verification
_CONCLUSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(done|complete[d]?|finished|fixed|solved|resolved)\b", re.IGNORECASE),
    re.compile(r"\b(the (answer|solution|fix) is)\b", re.IGNORECASE),
    re.compile(r"\b(therefore|thus|so the)\b", re.IGNORECASE),
]


def _verification_skip(steps: Sequence[str], window: int = 3) -> float:
    """Detect concluding without checking in the last `window` steps.

    Returns 0.8 if a conclusion keyword appears but no verification keyword
    appears in the same window.  Scales down as the window grows older.
    """
    recent = steps[-window:]
    text = " ".join(recent)
    has_conclusion = any(p.search(text) for p in _CONCLUSION_PATTERNS)
    has_verification = any(p.search(text) for p in _VERIFY_PATTERNS)
    if has_conclusion and not has_verification:
        return 0.8
    return 0.0


# Pairs of contradictory phrases (first seen → second seen = contradiction)
_CONTRADICTION_PAIRS: list[tuple[re.Pattern[str], re.Pattern[str]]] = [
    (
        re.compile(r"\b(it works|working correctly|no error)\b", re.IGNORECASE),
        re.compile(r"\b(error|fail|broken|doesn.t work)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(does not exist|missing|not found)\b", re.IGNORECASE),
        re.compile(r"\b(exists|found|present)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\bI (will|would|should|can)\b", re.IGNORECASE),
        re.compile(r"\bI (cannot|can.t|won.t|shouldn.t|must not)\b", re.IGNORECASE),
    ),
    (
        re.compile(r"\b(always|never|all|none)\b", re.IGNORECASE),
        re.compile(r"\b(sometimes|occasionally|some|few)\b", re.IGNORECASE),
    ),
]


def _claim_contradiction(steps: Sequence[str]) -> float:
    """Detect self-contradictory conclusions across the full step history.

    Counts pairs of contradictory patterns where the first pattern appeared
    before the second.  Each confirmed contradiction contributes 0.4.
    """
    texts = list(steps)
    score = 0.0
    for pat_a, pat_b in _CONTRADICTION_PAIRS:
        positions_a = [i for i, t in enumerate(texts) if pat_a.search(t)]
        positions_b = [i for i, t in enumerate(texts) if pat_b.search(t)]
        for pa in positions_a:
            for pb in positions_b:
                if pa < pb:
                    score += 0.4
                    break
            else:
                continue
            break
    return min(1.0, score)


def _cyclic_compression(steps: Sequence[str], window: int = 6) -> float:
    """Detect context churn — re-stating information from much earlier steps.

    Computes bigram overlap between the current step and the oldest step in
    the window.  High overlap means the agent is going in circles.
    """
    recent = list(steps[-window:])
    if len(recent) < 3:
        return 0.0
    oldest = _bigrams(_tokenize(recent[0]))
    newest = _bigrams(_tokenize(recent[-1]))
    return min(1.0, _jaccard(oldest, newest) * 1.5)  # amplify slightly


_SCOPE_CREEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(also|additionally|while I.m at it|and also|furthermore)\b", re.IGNORECASE),
    re.compile(r"\b(I noticed|I see that|by the way|incidentally)\b", re.IGNORECASE),
    re.compile(r"\b(refactor|clean up|improve|optimize|restructure)\b", re.IGNORECASE),
]


def _late_sprawl(steps: Sequence[str], late_fraction: float = 0.5) -> float:
    """Detect late-run scope creep.

    Scope-creep patterns (adding tangential work) that appear only in the
    second half of the step history are more alarming than early ones.
    """
    n = len(steps)
    if n < 2:
        return 0.0
    cutoff = max(1, int(n * late_fraction))
    late_steps = steps[cutoff:]
    early_steps = steps[:cutoff]
    late_text = " ".join(late_steps)
    early_text = " ".join(early_steps)
    late_hits = sum(1 for p in _SCOPE_CREEP_PATTERNS if p.search(late_text))
    early_hits = sum(1 for p in _SCOPE_CREEP_PATTERNS if p.search(early_text))
    if late_hits > early_hits:
        return min(1.0, 0.3 * (late_hits - early_hits))
    return 0.0


def _silent_topic_drift(steps: Sequence[str], task: str, window: int = 3) -> float:
    """Detect quiet off-task drift.

    Measures the Jaccard similarity between task keywords and the recent steps.
    Low overlap ⟹ the agent has drifted off-task without announcing it.
    """
    if not task.strip():
        return 0.0
    task_tokens = set(_tokenize(task))
    if not task_tokens:
        return 0.0
    recent = steps[-window:]
    recent_tokens = set(_tokenize(" ".join(recent)))
    if not recent_tokens:
        return 0.0
    overlap = len(task_tokens & recent_tokens) / len(task_tokens)
    # Low overlap = drifted. Scale: 0 overlap → score 0.9, full overlap → 0.0.
    return min(1.0, max(0.0, (1.0 - overlap) * 0.9))


# --------------------------------------------------------------------------- #
# Aggregate evaluator                                                          #
# --------------------------------------------------------------------------- #


def evaluate_all(
    steps: Sequence[str],
    task: str = "",
    *,
    weights: dict[str, float] | None = None,
    fire_threshold: float = FIRE_THRESHOLD,
) -> MonitorResult:
    """Evaluate all six trajectory monitors and return an aggregate result.

    Args:
        steps:          List of agent step texts (one string per turn).
                        The most recent step is steps[-1].
        task:           Original task description.  Used by silent_topic_drift.
        weights:        Override per-monitor weights.  Unspecified monitors keep
                        DEFAULT_WEIGHTS values.
        fire_threshold: Score threshold above which a monitor is considered fired.

    Returns:
        MonitorResult with scores, composite, fired list, and failure_type.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    scores: dict[str, float] = {
        "semantic_loop": _semantic_loop(steps),
        "verification_skip": _verification_skip(steps),
        "claim_contradiction": _claim_contradiction(steps),
        "cyclic_compression": _cyclic_compression(steps),
        "late_sprawl": _late_sprawl(steps),
        "silent_topic_drift": _silent_topic_drift(steps, task),
    }

    composite = sum(scores[name] * w.get(name, 0.0) for name in scores)
    fired = [name for name, score in scores.items() if score >= fire_threshold]

    # failure_type = highest-weight fired monitor (primary failure signal)
    failure_type: str | None = None
    if fired:
        failure_type = max(fired, key=lambda n: w.get(n, 0.0))

    return MonitorResult(
        scores=scores,
        composite=composite,
        fired=fired,
        failure_type=failure_type,
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "FIRE_THRESHOLD",
    "MonitorResult",
    "evaluate_all",
]
