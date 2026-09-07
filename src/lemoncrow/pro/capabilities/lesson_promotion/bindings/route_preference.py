"""Apply route-preference lessons to ranked routing candidates."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from lemoncrow.pro.capabilities.lesson_promotion.models import TypedLesson

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.cross_vendor_routing.router import RankedCandidate


def session_phase(session_state: dict[str, Any]) -> str:
    raw = str(session_state.get("session_phase") or "").strip().lower()
    normalized = {
        "explore": "explore",
        "exploration": "explore",
        "execute": "execute",
        "execution": "execute",
        "transition": "transition",
    }.get(raw)
    if normalized:
        return normalized
    turn_number = int(session_state.get("turn_number") or 0)
    return "execute" if turn_number > 5 else "explore"


def apply_route_preferences(
    ranked: list[RankedCandidate],
    *,
    lessons: list[TypedLesson],
    tool_name: str,
    session_state: dict[str, Any],
    now: datetime | None = None,
) -> tuple[list[RankedCandidate], list[str]]:
    current = now or datetime.now(UTC)
    phase = session_phase(session_state)
    matching_lessons = [
        lesson for lesson in lessons if _matches_lesson(lesson, tool_name=tool_name, phase=phase, at=current)
    ]
    if not matching_lessons:
        return ranked, []

    adjusted: list[tuple[tuple[int, float, str, str], RankedCandidate, list[str], set[str]]] = []
    for candidate in ranked:
        priority = 1
        adjusted_cost = candidate.estimated_cost_usd
        reasons = list(candidate.reasons)
        applied_ids: set[str] = set()
        for lesson in matching_lessons:
            preferred_vendor = str(lesson.prefer.get("vendor") or "").strip().lower()
            preferred_model = str(lesson.prefer.get("model") or "").strip().lower()
            if not preferred_vendor or not preferred_model:
                raise ValueError(f"route-preference lesson {lesson.id} is missing prefer.vendor or prefer.model")
            if candidate.vendor.lower() == preferred_vendor and candidate.model.lower() == preferred_model:
                applied_ids.add(lesson.id)
                if lesson.applies_without_tiebreaker_at(current):
                    priority = 0
                else:
                    adjusted_cost -= round(0.05 * lesson.effective_confidence_at(current), 6)
                reasons.append(f"lesson={lesson.id}: route preference matched {tool_name}/{phase}")
        adjusted.append(((priority, adjusted_cost, candidate.vendor, candidate.model), candidate, reasons, applied_ids))

    adjusted.sort(key=lambda item: item[0])
    reordered = [replace(candidate, reasons=tuple(reasons)) for _, candidate, reasons, _ in adjusted]
    applied = sorted({lesson_id for _, _, _, lesson_ids in adjusted for lesson_id in lesson_ids})
    return reordered, applied


def _matches_lesson(lesson: TypedLesson, *, tool_name: str, phase: str, at: datetime) -> bool:
    if lesson.kind != "route-preference":
        return False
    if not lesson.is_active_at(at, scope=lesson.scope):
        return False
    expected_tool = str(lesson.match.get("tool") or "").strip().lower()
    expected_phase = str(lesson.match.get("phase") or "").strip().lower()
    if not expected_tool or not expected_phase:
        raise ValueError(f"route-preference lesson {lesson.id} is missing match.tool or match.phase")
    normalized_phase = session_phase({"session_phase": phase})
    return expected_tool == tool_name.strip().lower() and expected_phase == normalized_phase
