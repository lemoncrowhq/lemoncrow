"""Scoped pull-context capability (M4).

Provides an explicit "given this subtask, return the minimal scoped context
required to act on it" entry point over the existing code-context engine,
dead-end tracker, and token-budget packer.
"""

from __future__ import annotations

from .capability import ScopedContextCapability
from .models import ContextChunk, ExclusionRecord, ScopedContext, Subtask

__all__ = [
    "ContextChunk",
    "ExclusionRecord",
    "ScopedContext",
    "ScopedContextCapability",
    "Subtask",
]
