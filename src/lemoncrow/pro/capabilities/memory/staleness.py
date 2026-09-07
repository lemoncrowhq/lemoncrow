"""Change-driven calibrated memory invalidation (N13).

When a code change touches the subject of a stored memory, that memory may have
gone stale. This module maps a *change type* to a *calibrated staleness action*
so a downstream policy can decide whether to invalidate the memory, queue it for
review, or leave it. The mapping is a pure, unit-testable function; wiring it to
actually mutate stored records (auto-invalidation) is opt-in and lives at the
call site, not here.

Calibration encodes confidence that the change really invalidated the memory:
a Deleted subject is certain (confidence 1.0 -> invalidate), a SignatureChanged
subject is likely (0.9 -> review), a Moved subject is plausible (0.6 -> review),
and a no-op change leaves the memory alone (keep).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ChangeType(StrEnum):
    """Kinds of code change that can stale a memory about a subject."""

    DELETED = "deleted"
    SIGNATURE_CHANGED = "signature_changed"
    BODY_CHANGED = "body_changed"
    MOVED = "moved"
    RENAMED = "renamed"
    UNCHANGED = "unchanged"


class StalenessAction(StrEnum):
    """What to do with a memory given a change to its subject."""

    INVALIDATE = "invalidate"
    REVIEW = "review"
    KEEP = "keep"


@dataclass(frozen=True)
class CalibratedAction:
    """A calibrated staleness verdict: the action plus its confidence (0..1)."""

    change_type: ChangeType
    action: StalenessAction
    confidence: float


# Calibrated mapping: change type -> (action, confidence). Confidence is the
# probability that the change genuinely staled the memory; the action is the
# bounded response. Tuned conservatively -- only a Deleted subject auto-
# invalidates; ambiguous changes (signature/move/rename) route to human/agent
# review rather than silently dropping recall.
_CALIBRATION: dict[ChangeType, tuple[StalenessAction, float]] = {
    ChangeType.DELETED: (StalenessAction.INVALIDATE, 1.0),
    ChangeType.SIGNATURE_CHANGED: (StalenessAction.REVIEW, 0.9),
    ChangeType.RENAMED: (StalenessAction.REVIEW, 0.7),
    ChangeType.MOVED: (StalenessAction.REVIEW, 0.6),
    ChangeType.BODY_CHANGED: (StalenessAction.REVIEW, 0.4),
    ChangeType.UNCHANGED: (StalenessAction.KEEP, 0.0),
}


def calibrate_staleness(change_type: ChangeType) -> CalibratedAction:
    """Map a ``ChangeType`` to its calibrated :class:`CalibratedAction`.

    Pure and total over the enum: an unknown member (only possible if the enum
    grows without updating ``_CALIBRATION``) defaults to conservative review at
    confidence 0.5 so a missing calibration never silently keeps a stale memory.
    """
    action, confidence = _CALIBRATION.get(change_type, (StalenessAction.REVIEW, 0.5))
    return CalibratedAction(change_type=change_type, action=action, confidence=confidence)


def should_auto_invalidate(action: CalibratedAction, *, min_confidence: float = 1.0) -> bool:
    """Opt-in gate: True only when the calibrated action is INVALIDATE at/above
    ``min_confidence``.

    Default ``min_confidence=1.0`` means *only* a certain (Deleted) change auto-
    invalidates; lowering it lets a policy auto-invalidate on weaker signals. The
    actual record mutation is the caller's responsibility -- this only decides.
    """
    return action.action is StalenessAction.INVALIDATE and action.confidence >= min_confidence


__all__ = [
    "CalibratedAction",
    "ChangeType",
    "StalenessAction",
    "calibrate_staleness",
    "should_auto_invalidate",
]
