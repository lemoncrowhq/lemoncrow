"""Caller-owned sticky routing state primitives."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_STICKINESS_WINDOW = 3


@dataclass(frozen=True)
class StickyRoutingState:
    """Small state object for callers that keep sticky routing between tool calls."""

    remaining_tool_calls: int = 0


def start_stickiness(window: int = DEFAULT_STICKINESS_WINDOW) -> StickyRoutingState:
    """Initialize a sticky window after a recommendation is chosen."""
    return StickyRoutingState(remaining_tool_calls=max(0, int(window)))


def decrement_stickiness(state: StickyRoutingState | int) -> StickyRoutingState:
    """Consume one sticky tool call."""
    remaining = state.remaining_tool_calls if isinstance(state, StickyRoutingState) else int(state)
    return StickyRoutingState(remaining_tool_calls=max(0, remaining - 1))


def reset_stickiness() -> StickyRoutingState:
    """Reset stickiness at a user-visible response boundary."""
    return StickyRoutingState(remaining_tool_calls=0)


def stickiness_remaining(state: StickyRoutingState | int | None) -> int:
    """Normalize caller-owned sticky state into a non-negative remaining count."""
    if state is None:
        return 0
    if isinstance(state, StickyRoutingState):
        return max(0, int(state.remaining_tool_calls))
    return max(0, int(state))
