"""Public-facing advisor wrapper for the cross-vendor routing core."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from lemoncrow.core.capabilities.pricing import active_model
from lemoncrow.pro.capabilities.lesson_promotion.store import TypedLessonStore

from .configuration import (
    RouteConfig,
    RouteConfigError,
    detect_configured_vendors,
    load_route_config,
    save_route_config,
)
from .router import CrossVendorRouter


class CrossVendorRouteAdvisor:
    """Configure, inspect, and recommend from the cross-vendor router."""

    def __init__(self, root: Path | str, *, env: Mapping[str, str] | None = None) -> None:
        self._root = Path(root).expanduser().resolve()
        self._env = env

    def configure(
        self,
        *,
        enabled_vendors: list[str] | None = None,
        risk_class: Literal["low", "medium", "high"] = "low",
    ) -> dict[str, Any]:
        configured = list(detect_configured_vendors(self._env))
        selected = enabled_vendors or configured
        if not selected:
            raise RouteConfigError(
                "no routable vendors detected; install a supported host CLI or set vendor API keys before running route configure"
            )
        config = RouteConfig(enabled_vendors=selected, risk_class=risk_class)
        path = save_route_config(self._root, config)
        return {
            "configured": True,
            "path": str(path),
            "enabled_vendors": config.enabled_vendors,
            "configured_vendors": configured,
            "risk_class": config.risk_class,
        }

    def recommend(
        self,
        *,
        tool_name: str,
        task_text: str,
        session_state: Mapping[str, Any] | None = None,
        actual_vendor: str | None = None,
    ) -> dict[str, Any]:
        config = load_route_config(self._root)
        lesson_store = TypedLessonStore(self._root, create=False)
        router = CrossVendorRouter(config, env=self._env, lesson_store=lesson_store)
        recommendation = router.recommend(
            tool_name=tool_name,
            task_text=task_text,
            session_state=session_state,
            actual_vendor=actual_vendor,
        )
        if recommendation is None:
            return {"configured": False, "bench_off": True}
        actual_model = active_model()
        actual_vendor_name = actual_vendor or _vendor_for_model(actual_model)
        recommendation_followed = _normalize_model(actual_model) == _normalize_model(recommendation.model)
        alternatives = [
            {
                "vendor": candidate.vendor,
                "model": candidate.model,
                "tier": candidate.tier,
                "estimated_cost_usd": candidate.estimated_cost_usd,
            }
            for candidate in recommendation.alternatives
        ]
        payload = {
            "configured": True,
            "vendor": recommendation.vendor,
            "model": recommendation.model,
            "tier": recommendation.tier,
            "predicted_cost_usd": recommendation.estimated_cost_usd,
            "alternatives": alternatives,
            "fallback": alternatives[1]["model"] if len(alternatives) > 1 else None,
            "reason": "; ".join(recommendation.reasons),
            "actual_model": actual_model,
            "actual_vendor": actual_vendor_name,
            "recommendation_followed": recommendation_followed,
            "enabled_vendors": config.enabled_vendors,
            "applied_lessons": list(recommendation.applied_lessons),
            "cost_cap_triggered": recommendation.cost_cap_triggered,
            "cost_cap_limit_usd_per_session": recommendation.cost_cap_limit_usd_per_session,
            "projected_session_cost_usd": recommendation.projected_session_cost_usd,
        }
        return payload

    def status(self) -> dict[str, Any]:
        config = load_route_config(self._root)
        route_event_count = 0
        estimated_savings = 0.0
        lesson_application_count = 0
        cost_cap_trigger_count = 0
        path = self._root / "live_savings_events.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("kind") != "model_recommendation":
                    continue
                if payload.get("configured") is False:
                    continue
                route_event_count += 1
                estimated_savings += float(payload.get("cost_saved_usd") or 0.0)
                if payload.get("applied_lessons"):
                    lesson_application_count += 1
                if payload.get("cost_cap_triggered"):
                    cost_cap_trigger_count += 1
        lesson_store = TypedLessonStore(self._root, create=False)
        return {
            "configured": True,
            "enabled_vendors": config.enabled_vendors,
            "configured_vendors": list(detect_configured_vendors(self._env)),
            "risk_class": config.risk_class,
            "recommendation_count": route_event_count,
            "estimated_savings_usd": round(estimated_savings, 6),
            "active_lesson_count": len(
                [
                    lesson
                    for lesson in lesson_store.list_lessons()
                    if lesson.is_active_at(datetime.now(UTC), scope=lesson.scope)
                ]
            ),
            "lesson_application_count": lesson_application_count,
            "cost_cap_trigger_count": cost_cap_trigger_count,
        }


def _normalize_model(model_id: str) -> str:
    return model_id.strip().lower().replace(".", "-")


def _vendor_for_model(model_id: str) -> str:
    normalized = _normalize_model(model_id)
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("gemini"):
        return "google"
    return "unknown"


__all__ = ["CrossVendorRouteAdvisor"]
