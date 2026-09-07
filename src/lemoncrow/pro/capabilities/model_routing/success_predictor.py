"""Learned routing-confidence layer (T9) over the heuristic tier router.

This is a *pragmatic, headless v1*.  There is **no training, no GPU, and no
network**.  It computes an outcome-frequency table over a handful of coarse
buckets from the routing outcomes LemonCrow already persists, then answers a
single question for the router:

    Given the coarse shape of this turn, how often did a *weak* (downgraded)
    model previously succeed on similar turns?

``p_weak_succeeds(features) -> float | None`` returns that frequency, or
``None`` when there is not enough evidence (``< min_samples`` rows in the
bucket).  ``None`` means "uncertain" -- the router falls back to the existing
heuristic step-down rule.

Success signal
--------------
The authoritative outcome is the quality_router verification envelope persisted
in the SQLite ``verification_envelope`` table (migration ``v2_004_routing.sql``)
joined to its ``route_decision`` row.  ``VerificationEnvelope.outcome == "pass"``
is a success; ``fail`` / ``escalate`` are failures; ``warn`` is ambiguous and
is dropped from the table (it is neither a clean success nor a clean failure).

Buckets
-------
Four coarse buckets, derived identically from persisted rows (training side) and
from live ``session_state`` (inference side) so a lookup keys the same cell that
was counted:

* ``verb``        -- task verb class (read / edit / design / ...)
* ``tool_type``   -- read / edit / agent / other
* ``prior_error`` -- none / some / many
* ``size_bucket`` -- xs / s / m / l  (changed-file count)

The module never writes to the store; persistence is owned elsewhere and read
here read-only.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from lemoncrow.core.environment import bool_env

# --------------------------------------------------------------------------- #
# Flag + threshold convention (mirrors model_routing.complexity tier routing)  #
# --------------------------------------------------------------------------- #

LEARNED_ROUTING_ENV_VAR = "LEMONCROW_LEARNED_ROUTING"
LEARNED_ROUTING_THRESHOLD_ENV_VAR = "LEMONCROW_LEARNED_ROUTING_THRESHOLD"
LEARNED_ROUTING_MIN_SAMPLES_ENV_VAR = "LEMONCROW_LEARNED_ROUTING_MIN_SAMPLES"

#: Default confidence required before the learned layer authorises a downgrade.
DEFAULT_THRESHOLD = 0.9
#: Minimum rows in a bucket before its frequency is trusted.
DEFAULT_MIN_SAMPLES = 20


def learned_routing_enabled(state: Mapping[str, Any] | None = None) -> bool:
    """Whether the learned routing-confidence layer is opted in for this decision.

    Opt-in via an explicit ``learned_routing`` flag in ``session_state`` *or* the
    ``LEMONCROW_LEARNED_ROUTING`` environment variable.  Default is off, so the
    baseline (heuristic) routing decision is unchanged -- mirrors
    ``tier_routing_enabled``.
    """
    if state is not None and state.get("learned_routing"):
        return True
    return bool_env(LEARNED_ROUTING_ENV_VAR, default=False)


def learned_routing_threshold(state: Mapping[str, Any] | None = None) -> float:
    """Confidence threshold for authorising a learned downgrade.

    Resolution order: ``session_state["learned_routing_threshold"]`` ->
    ``LEMONCROW_LEARNED_ROUTING_THRESHOLD`` env -> ``DEFAULT_THRESHOLD``.  Values
    outside ``(0, 1]`` (or unparseable) fall back to the default.
    """
    if state is not None:
        override = _coerce_threshold(state.get("learned_routing_threshold"))
        if override is not None:
            return override
    override = _coerce_threshold(os.environ.get(LEARNED_ROUTING_THRESHOLD_ENV_VAR))
    if override is not None:
        return override
    return DEFAULT_THRESHOLD


def learned_routing_min_samples(state: Mapping[str, Any] | None = None) -> int:
    """Minimum bucket sample count before a frequency is trusted.

    Resolution order: ``session_state["learned_routing_min_samples"]`` ->
    ``LEMONCROW_LEARNED_ROUTING_MIN_SAMPLES`` env -> ``DEFAULT_MIN_SAMPLES``.
    """
    if state is not None:
        override = _coerce_min_samples(state.get("learned_routing_min_samples"))
        if override is not None:
            return override
    override = _coerce_min_samples(os.environ.get(LEARNED_ROUTING_MIN_SAMPLES_ENV_VAR))
    if override is not None:
        return override
    return DEFAULT_MIN_SAMPLES


# --------------------------------------------------------------------------- #
# Coarse feature buckets                                                        #
# --------------------------------------------------------------------------- #

_DESIGN_VERBS = frozenset({"design", "architect", "plan", "strategy", "migrate", "rewrite", "feature", "refactor"})
_EDIT_VERBS = frozenset({"implement", "fix", "add", "update", "change", "test", "verify", "edit", "debug", "docs"})
_READ_VERBS = frozenset(
    {"explain", "show", "list", "summarize", "summarise", "read", "find", "search", "inspect", "review"}
)

_READ_TOOLS = frozenset(
    {"read", "smart_read", "search", "smart_search", "grep", "glob", "ls", "list", "context", "memory"}
)
_EDIT_TOOLS = frozenset(
    {"edit", "smart_edit", "write", "multiedit", "apply_patch", "notebookedit", "verify", "compact"}
)
_AGENT_TOOLS = frozenset({"agent", "task", "spawn", "delegate", "architect"})


@dataclass(frozen=True)
class RouteFeatures:
    """Coarse, hashable bucket key shared by training and inference sides."""

    verb: str
    tool_type: str
    prior_error: str
    size_bucket: str

    def key(self) -> tuple[str, str, str, str]:
        return (self.verb, self.tool_type, self.prior_error, self.size_bucket)


def _verb_bucket(text: str | None) -> str:
    """Classify a task verb / task_type string into a coarse verb bucket."""
    words = {w for w in (text or "").lower().replace("-", "_").split() if w}
    if words & _DESIGN_VERBS:
        return "design"
    if words & _EDIT_VERBS:
        return "edit"
    if words & _READ_VERBS:
        return "read"
    # Single-token persisted task_type (e.g. "feature") that did not split.
    token = (text or "").strip().lower().replace("-", "_")
    if token in _DESIGN_VERBS:
        return "design"
    if token in _EDIT_VERBS:
        return "edit"
    if token in _READ_VERBS:
        return "read"
    return "other"


def _tool_bucket(tool_name: str | None) -> str:
    normalized = (tool_name or "").strip().lower().replace("-", "_")
    if not normalized:
        return "other"
    if normalized in _AGENT_TOOLS or "agent" in normalized:
        return "agent"
    if normalized in _EDIT_TOOLS or "edit" in normalized or "write" in normalized:
        return "edit"
    if normalized in _READ_TOOLS or normalized.startswith(("read", "search")):
        return "read"
    return "other"


def _prior_error_bucket(count: int) -> str:
    if count <= 0:
        return "none"
    if count >= 3:
        return "many"
    return "some"


def _size_bucket(changed_files: int) -> str:
    if changed_files <= 0:
        return "xs"
    if changed_files <= 2:
        return "s"
    if changed_files <= 8:
        return "m"
    return "l"


def features_from_state(
    tool_name: str,
    task_text: str,
    state: Mapping[str, Any] | None = None,
) -> RouteFeatures:
    """Build coarse buckets from a live routing decision (inference side).

    Pulls the same signals the heuristic router already reads from
    ``session_state``: the task verb, the tool, prior-error count, and the
    changed-file count as a size proxy.
    """
    s = state or {}
    changed = _count(s.get("changed_files")) or _count(s.get("files"))
    return RouteFeatures(
        verb=_verb_bucket(task_text),
        tool_type=_tool_bucket(tool_name),
        prior_error=_prior_error_bucket(_safe_int(s.get("prior_errors"))),
        size_bucket=_size_bucket(changed),
    )


# --------------------------------------------------------------------------- #
# Outcome rows + frequency table                                               #
# --------------------------------------------------------------------------- #

#: ``verb`` is parsed out of the persisted ``route_decision.reason`` which the
#: quality_router policy formats as ``"risk=..., task=<task_type>, tier=..."``.
_TASK_TOKEN_PREFIX = "task="


@dataclass(frozen=True)
class OutcomeRow:
    """One persisted route outcome reduced to its bucket key + success bit."""

    features: RouteFeatures
    success: bool


@dataclass(frozen=True)
class _Cell:
    successes: int
    total: int


@dataclass(frozen=True)
class SuccessTable:
    """Outcome-frequency table over coarse buckets.  Immutable, no training."""

    cells: dict[tuple[str, str, str, str], _Cell]

    def probability(self, features: RouteFeatures, *, min_samples: int) -> float | None:
        """P(success) for ``features``' bucket, or ``None`` below ``min_samples``."""
        cell = self.cells.get(features.key())
        if cell is None or cell.total < max(1, min_samples):
            return None
        return cell.successes / cell.total

    def __len__(self) -> int:
        return len(self.cells)


def build_success_table(rows: Iterable[OutcomeRow]) -> SuccessTable:
    """Aggregate outcome rows into a per-bucket frequency table."""
    successes: dict[tuple[str, str, str, str], int] = {}
    totals: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = row.features.key()
        totals[key] = totals.get(key, 0) + 1
        if row.success:
            successes[key] = successes.get(key, 0) + 1
    cells = {key: _Cell(successes=successes.get(key, 0), total=total) for key, total in totals.items()}
    return SuccessTable(cells=cells)


def _verb_from_reason(reason: str | None) -> str:
    """Extract the ``task=<task_type>`` token from a persisted route reason."""
    for part in (reason or "").split(","):
        token = part.strip()
        if token.startswith(_TASK_TOKEN_PREFIX):
            return _verb_bucket(token[len(_TASK_TOKEN_PREFIX) :])
    return "other"


def outcome_row_from_persisted(
    *,
    outcome: str,
    reason: str | None,
    step_type: str | None,
    escalation_trigger: str | None,
    changed_files_count: int,
) -> OutcomeRow | None:
    """Reduce one persisted ``verification_envelope`` row (+ its route_decision)
    to an :class:`OutcomeRow`.

    Returns ``None`` for ambiguous (``warn``) rows so they neither inflate nor
    deflate any bucket's frequency.
    """
    normalized = (outcome or "").strip().lower()
    if normalized == "pass":
        success = True
    elif normalized in ("fail", "escalate"):
        success = False
    else:
        # "warn", "not_run", or anything unexpected -> ambiguous, drop.
        return None

    # An escalation trigger is a prior-error proxy on the persisted side.
    prior = "many" if escalation_trigger else "none"
    features = RouteFeatures(
        verb=_verb_from_reason(reason),
        tool_type=_tool_bucket(step_type),
        prior_error=prior,
        size_bucket=_size_bucket(changed_files_count),
    )
    return OutcomeRow(features=features, success=success)


def load_outcome_rows(source: sqlite3.Connection | str | Path) -> list[OutcomeRow]:
    """Read persisted route outcomes from the SQLite store (read-only).

    ``source`` may be an open :class:`sqlite3.Connection` or a path to the
    store's ``lemoncrow.db``.  Joins ``verification_envelope`` to its
    ``route_decision`` and reduces each row to an :class:`OutcomeRow`.  Missing
    tables / unreadable stores yield an empty list (the predictor then returns
    ``None`` and the router stays heuristic).
    """
    if isinstance(source, sqlite3.Connection):
        conn = source
        owns_conn = False
    else:
        conn = sqlite3.connect(str(source))
        owns_conn = True
    try:
        cursor = conn.execute("""
            SELECT ve.outcome AS outcome,
                   rd.reason AS reason,
                   rd.step_type AS step_type,
                   rd.escalation_trigger AS escalation_trigger,
                   ve.changed_files AS changed_files
            FROM verification_envelope AS ve
            JOIN route_decision AS rd ON rd.id = ve.route_decision_id
            """)
        raw = cursor.fetchall()
    except sqlite3.Error:
        logging.exception("Recovered from broad exception handler")
        return []
    finally:
        if owns_conn:
            conn.close()

    rows: list[OutcomeRow] = []
    for record in raw:
        outcome, reason, step_type, escalation_trigger, changed_files = (
            record[0],
            record[1],
            record[2],
            record[3],
            record[4],
        )
        row = outcome_row_from_persisted(
            outcome=str(outcome or ""),
            reason=None if reason is None else str(reason),
            step_type=None if step_type is None else str(step_type),
            escalation_trigger=None if escalation_trigger is None else str(escalation_trigger),
            changed_files_count=_count(changed_files),
        )
        if row is not None:
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Public predictor                                                             #
# --------------------------------------------------------------------------- #


def p_weak_succeeds(
    features: RouteFeatures,
    *,
    table: SuccessTable | None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> float | None:
    """Probability a weak (downgraded) model succeeds on this turn's bucket.

    Returns ``None`` when there is no table, an empty table, or fewer than
    ``min_samples`` observations in the matching bucket -- i.e. "uncertain",
    which the router treats as a heuristic fallback.
    """
    if table is None or len(table) == 0:
        return None
    return table.probability(features, min_samples=min_samples)


# --------------------------------------------------------------------------- #
# Small coercion helpers (kept local; no router coupling)                       #
# --------------------------------------------------------------------------- #


def _count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            return max(0, int(text))
        except ValueError:
            # A JSON-encoded list (persisted changed_files) -> count its items.
            if text.startswith("[") and text.endswith("]"):
                import json

                try:
                    parsed = json.loads(text)
                except ValueError:
                    return 0
                return len(parsed) if isinstance(parsed, list) else 0
            return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _coerce_threshold(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if not 0.0 < parsed <= 1.0:
        return None
    return parsed


def _coerce_min_samples(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float):
        return int(value) if value >= 1 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = int(text)
        except ValueError:
            return None
        return parsed if parsed >= 1 else None
    return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        cleaned = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            cleaned = float(text)
        except ValueError:
            return None
    else:
        return None
    if not isfinite(cleaned):
        return None
    return cleaned


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_THRESHOLD",
    "LEARNED_ROUTING_ENV_VAR",
    "LEARNED_ROUTING_MIN_SAMPLES_ENV_VAR",
    "LEARNED_ROUTING_THRESHOLD_ENV_VAR",
    "OutcomeRow",
    "RouteFeatures",
    "SuccessTable",
    "build_success_table",
    "features_from_state",
    "learned_routing_enabled",
    "learned_routing_min_samples",
    "learned_routing_threshold",
    "load_outcome_rows",
    "outcome_row_from_persisted",
    "p_weak_succeeds",
]
