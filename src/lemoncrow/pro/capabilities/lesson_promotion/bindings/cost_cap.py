"""Apply cost-cap lessons to ranked routing candidates."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from lemoncrow.pro.capabilities.lesson_promotion.models import TypedLesson

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.cross_vendor_routing.router import RankedCandidate

_TIER_ORDER = {"cheap": 0, "medium": 1, "high": 2, "expensive": 2}


def apply_cost_cap(
    ranked: list[RankedCandidate],
    *,
    lessons: list[TypedLesson],
    session_state: dict[str, Any],
    now: datetime | None = None,
) -> tuple[list[RankedCandidate], list[str], bool, float | None, float | None]:
    current = now or datetime.now(UTC)
    active = [lesson for lesson in lessons if lesson.kind == "cost-cap" and lesson.is_active_at(current)]
    if not active or not ranked:
        return ranked, [], False, None, None

    lesson = min(active, key=lambda item: float(item.limit_usd_per_session or 0.0))
    if lesson.limit_usd_per_session is None or lesson.on_breach is None:
        raise ValueError(f"cost-cap lesson {lesson.id} is missing limit_usd_per_session or on_breach")

    session_cost_usd = float(session_state.get("session_cost_usd") or 0.0)
    projected_total = round(session_cost_usd + ranked[0].estimated_cost_usd, 6)
    if projected_total <= lesson.limit_usd_per_session:
        return ranked, [], False, lesson.limit_usd_per_session, projected_total

    if lesson.on_breach == "block":
        raise ValueError(f"cost-cap lesson {lesson.id} would block projected session spend {projected_total:.6f}")

    if lesson.on_breach == "warn":
        warned = replace(
            ranked[0],
            reasons=tuple([*ranked[0].reasons, f"lesson={lesson.id}: cost cap warning at ${projected_total:.6f}"]),
        )
        return [warned, *ranked[1:]], [lesson.id], True, lesson.limit_usd_per_session, projected_total

    current_tier = _TIER_ORDER.get(ranked[0].tier, 99)
    downgraded = next(
        (candidate for candidate in ranked[1:] if _TIER_ORDER.get(candidate.tier, 99) < current_tier), None
    )
    if downgraded is None:
        warned = replace(
            ranked[0],
            reasons=tuple([*ranked[0].reasons, f"lesson={lesson.id}: cost cap breached but no cheaper tier exists"]),
        )
        return [warned, *ranked[1:]], [lesson.id], True, lesson.limit_usd_per_session, projected_total

    selected = replace(
        downgraded,
        reasons=tuple(
            [*downgraded.reasons, f"lesson={lesson.id}: cost cap downgraded route at ${projected_total:.6f}"]
        ),
    )
    remaining = [
        candidate
        for candidate in ranked
        if candidate.model != downgraded.model or candidate.vendor != downgraded.vendor
    ]
    return [selected, *remaining], [lesson.id], True, lesson.limit_usd_per_session, projected_total
