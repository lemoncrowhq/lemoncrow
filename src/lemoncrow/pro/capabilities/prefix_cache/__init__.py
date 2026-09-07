"""Prefix cache planning and diagnostics.

Tracks stable-prefix stability across agent turns and surfaces
cache_hit_ratio, prefix_invalidated_reason, and token split diagnostics.
"""

from .diagnostics import PrefixCacheDiagnostics, PrefixTurnRecord
from .planner import PrefixCachePlan, PrefixCachePlanner

__all__ = [
    "PrefixCacheDiagnostics",
    "PrefixCachePlan",
    "PrefixCachePlanner",
    "PrefixTurnRecord",
]
