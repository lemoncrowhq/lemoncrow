"""Retry budget tracker for the verification loop (M3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetryBudget:
    """Caps verification retries per subtask (default 3, per the M3 plan)."""

    max_attempts: int = 3
    used: int = 0
    attempts_by_key: dict[str, int] = field(default_factory=dict)

    def consume(self, key: str | None = None) -> int:
        self.used += 1
        if key is None:
            return self.used
        current = self.attempts_by_key.get(key, 0) + 1
        self.attempts_by_key[key] = current
        return current

    def exhausted(self, key: str | None = None) -> bool:
        return self.used_for(key) >= self.max_attempts

    def remaining(self, key: str | None = None) -> int:
        return max(0, self.max_attempts - self.used_for(key))

    def used_for(self, key: str | None = None) -> int:
        if key is None:
            return self.used
        return self.attempts_by_key.get(key, 0)

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self.used = 0
            self.attempts_by_key.clear()
            return
        self.attempts_by_key.pop(key, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "used": self.used,
            "attempts_by_key": dict(self.attempts_by_key),
        }

    @classmethod
    def from_mapping(cls, payload: Any, *, max_attempts: int = 3) -> RetryBudget:
        if not isinstance(payload, dict):
            return cls(max_attempts=max_attempts)
        attempts_raw = payload.get("attempts_by_key")
        attempts = (
            {str(key): int(value) for key, value in attempts_raw.items()} if isinstance(attempts_raw, dict) else {}
        )
        used = payload.get("used")
        if not isinstance(used, int):
            used = sum(attempts.values())
        return cls(
            max_attempts=int(payload.get("max_attempts", max_attempts)),
            used=used,
            attempts_by_key=attempts,
        )
