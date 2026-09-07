"""DifficultyFSM — trajectory difficulty finite state machine.

Tracks how hard the agent is working over the last several steps and
transitions between six states:

    INIT   → Starting state before any step is scored.
    FAST   → Consistently easy; skip E-trace pipeline, longer monitor cooldown.
    NORMAL → Default operating state; full pipeline.
    SLOW   → Consistently hard; full pipeline; shorter monitor cooldown.
    SKIP   → Extended SLOW; same routing as SLOW; signals prolonged stall.
    END    → Reserved for future use; unreachable through built-in transitions.

The heuristic step score (a float in [0, 1]) is computed from four signals:
    - hedging_density:  fraction of tokens that are hedging words
    - response_length:  length relative to a reference length
    - error_language:   presence of error-related words
    - entity_density:   ratio of capitalised tokens (proxy for specificity)

A sigmoid squash brings the weighted sum to [0, 1] and is then passed to
``DifficultyFSM.transition``.

Usage::

    from lemoncrow.pro.capabilities.monitors.fsm import DifficultyFSM, score_step

    fsm = DifficultyFSM()
    for step_text in agent_steps:
        step_score = score_step(step_text)
        state = fsm.transition(step_score)
        print(state, fsm.current_state)
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum


class FSMState(Enum):
    """States of the DifficultyFSM."""

    INIT = "INIT"
    FAST = "FAST"
    NORMAL = "NORMAL"
    SLOW = "SLOW"
    SKIP = "SKIP"
    END = "END"  # Reserved — never reached by built-in transitions.


# --------------------------------------------------------------------------- #
# Step scorer                                                                  #
# --------------------------------------------------------------------------- #

_HEDGE_WORDS = frozenset(
    {
        "maybe",
        "perhaps",
        "might",
        "could",
        "should",
        "probably",
        "possibly",
        "unclear",
        "unsure",
        "uncertain",
        "seems",
        "appears",
        "like",
        "think",
        "believe",
        "guess",
        "assume",
        "not sure",
        "not certain",
    }
)

_ERROR_WORDS = frozenset(
    {
        "error",
        "exception",
        "fail",
        "failed",
        "failure",
        "crash",
        "broken",
        "bug",
        "issue",
        "problem",
        "traceback",
        "stderr",
        "panic",
        "abort",
        "undefined",
        "null",
        "none",
        "nan",
        "invalid",
        "wrong",
    }
)

_REF_LENGTH = 200  # characters — steps shorter than this score higher on length


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_step(text: str, ref_length: int = _REF_LENGTH) -> float:
    """Score one agent step as a difficulty float in [0, 1].

    The heuristic combines four signals:
        hedging_density — fraction of tokens that are hedge words   (→ harder)
        response_length — longer responses indicate more work       (→ longer = harder)
        error_language  — presence of error words                   (→ harder)
        entity_density  — ratio of Title-case tokens (proxy for specificity, → easier)

    Short completion messages like "All tests pass." score LOW (easy). Long
    error-heavy messages with hedging score HIGH (hard).

    Args:
        text:       The agent step text to score.
        ref_length: Reference length used to normalise the length score.

    Returns:
        Float in [0, 1] where 1.0 = maximum difficulty.
    """
    tokens = re.findall(r"[a-zA-Z']+", text)
    if not tokens:
        return 0.5  # unknown → neutral

    lower = [t.lower() for t in tokens]
    n = len(lower)

    hedging = sum(1 for t in lower if t in _HEDGE_WORDS) / n
    # Longer responses = more complex work = harder (short clean steps = easy)
    length_score = min(1.0, len(text) / max(1, ref_length * 3))
    error_hits = sum(1 for t in lower if t in _ERROR_WORDS) / n
    entity = sum(1 for t in tokens if t[0].isupper()) / n
    specificity = entity  # high entity density → more concrete → easier

    raw = 2.5 * hedging + 1.0 * length_score + 3.0 * error_hits - 1.5 * specificity
    return round(_sigmoid(raw), 4)


# --------------------------------------------------------------------------- #
# FSM                                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class DifficultyFSM:
    """Finite state machine that tracks agent trajectory difficulty.

    Attributes:
        fast_threshold:   Score ≤ this counts as "easy" (default 0.2).
        slow_threshold:   Score ≥ this counts as "hard" (default 0.6).
        skip_threshold:   Score ≥ this counts toward the SKIP gate (default 0.85).
        hysteresis_margin: Buffer for leaving FAST/SLOW (default 0.1).
        fast_window:      Consecutive easy steps needed to enter FAST (default 6).
        slow_window:      Consecutive hard steps needed to enter SLOW (default 5).
        skip_window:      Consecutive very-hard steps needed to enter SKIP (default 35).
    """

    fast_threshold: float = 0.2
    slow_threshold: float = 0.6
    skip_threshold: float = 0.85
    hysteresis_margin: float = 0.1
    fast_window: int = 6
    slow_window: int = 5
    skip_window: int = 35

    current_state: FSMState = field(default=FSMState.INIT, init=False)
    history: list[float] = field(default_factory=list, init=False)

    def transition(self, score: float) -> FSMState:
        """Record a step score and advance the FSM.

        Args:
            score: Difficulty score for the current step, float in [0, 1].

        Returns:
            The new FSMState after the transition.
        """
        self.history.append(score)
        s = self.current_state

        if s == FSMState.INIT:
            self.current_state = FSMState.NORMAL

        elif s == FSMState.NORMAL:
            if self._all_recent_below(self.fast_window, self.fast_threshold):
                self.current_state = FSMState.FAST
            elif self._all_recent_above(self.slow_window, self.slow_threshold):
                self.current_state = FSMState.SLOW

        elif s == FSMState.FAST:
            if score > self.fast_threshold + self.hysteresis_margin:
                self.current_state = FSMState.NORMAL

        elif s in (FSMState.SLOW, FSMState.SKIP):
            if score < self.slow_threshold - self.hysteresis_margin:
                self.current_state = FSMState.NORMAL
            elif s == FSMState.SLOW and self._all_recent_above(self.skip_window, self.skip_threshold):
                self.current_state = FSMState.SKIP

        return self.current_state

    # ---------------------------------------------------------------------- #
    # Properties derived from the current state                               #
    # ---------------------------------------------------------------------- #

    @property
    def skip_etraces(self) -> bool:
        """True when the E-trace pipeline should be skipped (FAST state)."""
        return self.current_state == FSMState.FAST

    @property
    def monitor_cooldown_steps(self) -> int:
        """Steps between monitor injections given the current state."""
        return {
            FSMState.INIT: 1,
            FSMState.FAST: 5,
            FSMState.NORMAL: 3,
            FSMState.SLOW: 2,
            FSMState.SKIP: 2,
            FSMState.END: 999,
        }[self.current_state]

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _all_recent_below(self, window: int, threshold: float) -> bool:
        recent = self.history[-window:]
        return len(recent) == window and all(s < threshold for s in recent)

    def _all_recent_above(self, window: int, threshold: float) -> bool:
        recent = self.history[-window:]
        return len(recent) == window and all(s > threshold for s in recent)

    def reset(self) -> None:
        """Reset to INIT state, clearing history."""
        self.current_state = FSMState.INIT
        self.history.clear()

    def summary(self) -> dict[str, object]:
        """Return a compact dict summary for telemetry."""
        return {
            "state": self.current_state.value,
            "steps_recorded": len(self.history),
            "last_score": self.history[-1] if self.history else None,
            "skip_etraces": self.skip_etraces,
            "monitor_cooldown": self.monitor_cooldown_steps,
        }


# Convenience re-export: call with a Sequence of step texts to get a final state
def advance_many(
    fsm: DifficultyFSM,
    steps: Sequence[str],
) -> FSMState:
    """Score and advance the FSM for each step in sequence.

    Useful for initialising an FSM from a history of steps.

    Returns:
        The FSMState after processing the last step.
    """
    state = fsm.current_state
    for text in steps:
        state = fsm.transition(score_step(text))
    return state


def make_signals_fn(
    fsm: DifficultyFSM,
) -> Callable[[Sequence[str]], dict[str, float]]:
    """Build a ``signals_fn`` for trajectory early-exit signalling.

    Returns ``{streak, hedge, diversity}`` in [0, 1] for the early-exit condition:

        streak > 0.7  OR  (hedge > 0.6 AND diversity > 0.5)

    * ``streak``    — fraction of last 10 steps above ``fsm.slow_threshold``
    * ``hedge``     — scaled mean hedging density over last 5 steps
    * ``diversity`` — 1 - mean pairwise bigram-Jaccard over last 4 steps
    """

    def _bigram_set(text: str) -> set[tuple[str, str]]:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        return set(itertools.pairwise(tokens)) if len(tokens) >= 2 else set()

    def _jaccard(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
        if not a and not b:
            return 1.0
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    slow_threshold = fsm.slow_threshold

    def signals_fn(steps: Sequence[str]) -> dict[str, float]:
        if not steps:
            return {"streak": 0.0, "hedge": 0.0, "diversity": 0.0}

        recent_scores = [score_step(s) for s in steps[-10:]]
        streak = sum(1 for s in recent_scores if s >= slow_threshold) / len(recent_scores)

        last5 = list(steps[-5:])
        # Hedge: fraction of hedge words, scaled up (typical max ~20% of tokens)
        hedge_vals = []
        for text in last5:
            tokens = re.findall(r"[a-zA-Z']+", text.lower())
            if tokens:
                hedge_vals.append(sum(1 for t in tokens if t in _HEDGE_WORDS) / len(tokens))
        hedge = min(1.0, (sum(hedge_vals) / max(1, len(hedge_vals))) * 5.0)

        last4 = list(steps[-4:])
        bigrams = [_bigram_set(s) for s in last4]
        pairs = [_jaccard(bigrams[i], bigrams[j]) for i in range(len(bigrams)) for j in range(i + 1, len(bigrams))]
        diversity = 1.0 - (sum(pairs) / max(1, len(pairs)))

        return {
            "streak": round(streak, 4),
            "hedge": round(hedge, 4),
            "diversity": round(diversity, 4),
        }

    return signals_fn


__all__ = [
    "DifficultyFSM",
    "FSMState",
    "advance_many",
    "make_signals_fn",
    "score_step",
]
