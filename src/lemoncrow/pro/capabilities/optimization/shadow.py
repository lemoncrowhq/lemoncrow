"""Local shadow-run state and cost guardrails for Optimization Advisor v0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowRunState:
    status: str
    policy: str
    days: int
    started_at: str
    baseline_weekly_cost_usd: float
    estimated_weekly_spend_usd: float
    max_daily_spend_usd: float
    spend_usd: float = 0.0
    stopped_at: str | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "policy": self.policy,
            "days": self.days,
            "started_at": self.started_at,
            "baseline_weekly_cost_usd": self.baseline_weekly_cost_usd,
            "estimated_weekly_spend_usd": self.estimated_weekly_spend_usd,
            "max_daily_spend_usd": self.max_daily_spend_usd,
            "spend_usd": self.spend_usd,
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
        }


def shadow_state_path(root: Path) -> Path:
    return Path(root) / "optimization_shadow.json"


def load_shadow_state(root: Path) -> dict[str, Any] | None:
    path = shadow_state_path(root)
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"shadow state at {path} must be a mapping")
    return loaded


def save_shadow_state(root: Path, state: ShadowRunState | dict[str, Any]) -> Path:
    payload = state.to_dict() if isinstance(state, ShadowRunState) else state
    path = shadow_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_shadow_state(
    *,
    policy: str,
    days: int,
    baseline_weekly_cost_usd: float,
    max_daily_spend_usd: float | None = None,
) -> ShadowRunState:
    baseline_daily = baseline_weekly_cost_usd / 7.0
    default_daily_cap = baseline_daily * 0.10
    maximum_allowed = baseline_daily * 0.25
    requested_cap = default_daily_cap if max_daily_spend_usd is None else max_daily_spend_usd
    if requested_cap > maximum_allowed:
        raise ValueError("shadow daily spend cap cannot exceed 25% of the trailing 7-day daily baseline")
    daily_cap = max(0.0, requested_cap)
    estimated_weekly = min(baseline_weekly_cost_usd * 0.10, daily_cap * max(1, days))
    return ShadowRunState(
        status="running",
        policy=policy,
        days=days,
        started_at=datetime.now(UTC).isoformat(),
        baseline_weekly_cost_usd=round(baseline_weekly_cost_usd, 6),
        estimated_weekly_spend_usd=round(estimated_weekly, 6),
        max_daily_spend_usd=round(daily_cap, 6),
    )


def stop_shadow(root: Path, *, reason: str = "user_stopped") -> dict[str, Any]:
    state = load_shadow_state(root)
    if state is None:
        return {"status": "not_running"}
    updated = dict(state)
    updated["status"] = "stopped"
    updated["stopped_at"] = datetime.now(UTC).isoformat()
    updated["stop_reason"] = reason
    save_shadow_state(root, updated)
    return updated
