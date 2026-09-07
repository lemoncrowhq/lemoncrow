"""Pluggable swarm reducers.

``merge`` (the LLM wave-evaluator) is the default and reproduces today's
behavior. ``best`` (heuristic / measured fitness) is additive. ``union`` and
``vote`` arrive in Phase 4.

Reducer modules must not import ``capability`` at module-load time (they use lazy
imports inside ``reduce``) so this package can be imported from ``capability``
itself without a cycle.
"""

from __future__ import annotations

from lemoncrow.pro.capabilities.swarm.reducers.base import (
    REDUCERS,
    Reducer,
    WaveContext,
    get_reducer,
    register_reducer,
)
from lemoncrow.pro.capabilities.swarm.reducers.best import BestReducer
from lemoncrow.pro.capabilities.swarm.reducers.merge import MergeReducer
from lemoncrow.pro.capabilities.swarm.reducers.union import UnionReducer
from lemoncrow.pro.capabilities.swarm.reducers.vote import VoteReducer

register_reducer(MergeReducer())
register_reducer(BestReducer())
register_reducer(UnionReducer())
register_reducer(VoteReducer())

__all__ = [
    "REDUCERS",
    "BestReducer",
    "MergeReducer",
    "Reducer",
    "UnionReducer",
    "VoteReducer",
    "WaveContext",
    "get_reducer",
    "register_reducer",
]
