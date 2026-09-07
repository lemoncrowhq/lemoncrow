"""Per-tool circuit breaker (closed → open → half-open state machine)."""

from __future__ import annotations

import time

from .models import CircuitState

# Thresholds
_FAILURE_THRESHOLD = 5  # consecutive failures to open circuit
_RECOVERY_TIMEOUT = 60.0  # seconds in OPEN before trying HALF_OPEN
_PROBE_SUCCESSES_NEEDED = 2  # successes in HALF_OPEN to close again


class CircuitBreaker:
    """
    Per-tool-name circuit breaker.

    States:
    - CLOSED: normal operation, all calls allowed
    - OPEN:   too many failures, calls are rejected
    - HALF_OPEN: one probe call allowed to test recovery
    """

    def __init__(self) -> None:
        # Maps tool_name → (state, consecutive_failures, opened_at, probe_successes)
        self._state: dict[str, tuple[str, int, float, int]] = {}

    def _get(self, tool: str) -> tuple[CircuitState, int, float, int]:
        entry = self._state.get(tool)
        if entry is None:
            return CircuitState.CLOSED, 0, 0.0, 0
        return (
            CircuitState(entry[0]),
            entry[1],
            entry[2],
            entry[3],
        )

    def _set(
        self,
        tool: str,
        state: CircuitState,
        failures: int,
        opened_at: float,
        probe_successes: int,
    ) -> None:
        self._state[tool] = (state.value, failures, opened_at, probe_successes)

    def should_allow(self, tool: str) -> bool:
        """Return True if the call should be allowed, False if circuit is OPEN."""
        state, failures, opened_at, _probes = self._get(tool)
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.OPEN:
            if time.time() - opened_at >= _RECOVERY_TIMEOUT:
                self._set(tool, CircuitState.HALF_OPEN, failures, opened_at, 0)
                return True  # allow the probe
            return False
        # HALF_OPEN: allow one probe
        return True

    def record_success(self, tool: str) -> None:
        state, failures, opened_at, probes = self._get(tool)
        if state == CircuitState.CLOSED:
            self._set(tool, CircuitState.CLOSED, 0, 0.0, 0)
        elif state == CircuitState.HALF_OPEN:
            probes += 1
            if probes >= _PROBE_SUCCESSES_NEEDED:
                self._set(tool, CircuitState.CLOSED, 0, 0.0, 0)
            else:
                self._set(tool, CircuitState.HALF_OPEN, failures, opened_at, probes)

    def record_failure(self, tool: str) -> None:
        state, failures, _opened_at, _probes = self._get(tool)
        failures += 1
        if state == CircuitState.HALF_OPEN or failures >= _FAILURE_THRESHOLD:
            self._set(tool, CircuitState.OPEN, failures, time.time(), 0)
        else:
            self._set(tool, CircuitState.CLOSED, failures, 0.0, 0)

    def all_states(self) -> dict[str, str]:
        return {tool: CircuitState(entry[0]).value for tool, entry in self._state.items()}
