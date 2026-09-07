"""Trajectory monitors and difficulty FSM for agent health assessment.

Six pure-Python monitors for detecting failure patterns mid-run:
    - semantic_loop
    - verification_skip
    - claim_contradiction
    - cyclic_compression
    - late_sprawl
    - silent_topic_drift

DifficultyFSM tracks trajectory difficulty across six states and gates
E-trace retrieval and monitor injection cooldowns accordingly.

Usage::

    from lemoncrow.pro.capabilities.monitors import evaluate_all, MonitorResult
    from lemoncrow.pro.capabilities.monitors import DifficultyFSM, score_step
"""

from lemoncrow.pro.capabilities.monitors.fsm import (
    DifficultyFSM,
    FSMState,
    advance_many,
    make_signals_fn,
    score_step,
)
from lemoncrow.pro.capabilities.monitors.suite import (
    DEFAULT_WEIGHTS,
    FIRE_THRESHOLD,
    MonitorResult,
    evaluate_all,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "FIRE_THRESHOLD",
    "DifficultyFSM",
    "FSMState",
    "MonitorResult",
    "advance_many",
    "evaluate_all",
    "make_signals_fn",
    "score_step",
]
