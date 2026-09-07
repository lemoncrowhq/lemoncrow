"""Data models for tool supervision capability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CircuitState(StrEnum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls (too many failures)
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ToolObservation:
    key: str
    payload: dict[str, Any]
    cache_hit: bool
    timestamp: float = 0.0
    cost_estimate: float = 0.0  # estimated token cost for this call


@dataclass
class AnomalyAlert:
    tool: str
    severity: str  # 'warning' | 'critical'
    message: str
    z_score: float = 0.0


@dataclass
class SupervisionMetrics:
    total_tool_calls: int = 0
    avoided_tool_calls: int = 0
    cache_hit_rate: float = 0.0
    token_savings: int = 0
    chars_saved: int = 0
    avg_cost_per_call: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tool_calls": self.total_tool_calls,
            "avoided_tool_calls": self.avoided_tool_calls,
            "cache_hit_rate": self.cache_hit_rate,
            "token_savings": self.token_savings,
            "chars_saved": self.chars_saved,
            "avg_cost_per_call": self.avg_cost_per_call,
        }
