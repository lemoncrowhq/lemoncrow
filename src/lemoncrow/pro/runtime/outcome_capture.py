"""Outcome capture for route and compact decisions (Spec 01).

Captures the observable consequence of every routing and compaction decision
within a rolling window of subsequent turns. Capture is O(1) per turn: no
background threads, no file locks beyond the existing session_state.json writer.

Workflow
--------
1. At decision time:   ``schedule(decision_id, kind, ...)`` writes a *pending*
   entry to session_state.json with ``outcome_window: null``.
2. After every subsequent ledger event: ``advance(session_id, ledger)`` walks
   all pending outcomes, increments ``turns_observed``, and fills
   ``outcome_window`` once the target window size is reached.
3. At session close:   call ``advance`` one final time; pending entries are
   written with ``turns_observed < N`` (still valid, just less data).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.lesson_models import LessonCandidate
from lemoncrow.infra.storage.bundle import StoreBundle

# Window sizes (hardcoded for v1, configurable later per spec)
_ROUTE_WINDOW = 5
_COMPACT_WINDOW = 10


# --------------------------------------------------------------------------- #
# Data classes                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class RouteOutcome:
    decision_id: str
    at: str
    kind: str  # always "route"
    tool: str
    recommended_vendor: str | None
    recommended_tier: str
    recommended_model: str
    actual_vendor: str | None
    actual_model: str | None
    recommendation_followed: bool
    scored_state: dict[str, Any] = field(default_factory=dict)
    applied_lessons: list[str] = field(default_factory=list)
    cost_cap_triggered: bool = False
    cost_cap_limit_usd_per_session: float | None = None
    # mutable tracking fields (not serialised directly)
    _turns_observed: int = field(default=0, repr=False)
    _model_errors: int = field(default=0, repr=False)
    _env_errors: int = field(default=0, repr=False)
    _retries_same_tool: int = field(default=0, repr=False)
    _extra_reads: int = field(default=0, repr=False)
    outcome_window: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "decision_id": self.decision_id,
            "at": self.at,
            "kind": self.kind,
            "tool": self.tool,
            "recommended_vendor": self.recommended_vendor,
            "recommended_tier": self.recommended_tier,
            "recommended_model": self.recommended_model,
            "actual_vendor": self.actual_vendor,
            "actual_model": self.actual_model,
            "recommendation_followed": self.recommendation_followed,
            "applied_lessons": self.applied_lessons,
            "cost_cap_triggered": self.cost_cap_triggered,
            "cost_cap_limit_usd_per_session": self.cost_cap_limit_usd_per_session,
            "scored_state": self.scored_state,
            "outcome_window": self.outcome_window,
        }
        return d


@dataclass
class CompactOutcome:
    decision_id: str
    at: str
    kind: str  # always "compact"
    trigger: str
    tokens_before: int
    tokens_after: int
    must_keep_keywords: list[str]
    # mutable tracking fields
    _turns_observed: int = field(default=0, repr=False)
    _errors_before: int = field(default=0, repr=False)
    _errors_in_window: int = field(default=0, repr=False)
    _extra_reads: int = field(default=0, repr=False)
    _must_keep_violations: int = field(default=0, repr=False)
    outcome_window: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "at": self.at,
            "kind": self.kind,
            "trigger": self.trigger,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "must_keep_keywords": self.must_keep_keywords,
            "outcome_window": self.outcome_window,
        }


# --------------------------------------------------------------------------- #
# In-process registry (one entry per session_id)                              #
# --------------------------------------------------------------------------- #

_pending_route: dict[str, list[RouteOutcome]] = {}
_pending_compact: dict[str, list[CompactOutcome]] = {}


# --------------------------------------------------------------------------- #
# Score formulae                                                               #
# --------------------------------------------------------------------------- #


def _route_score(
    retries_same_tool: int,
    model_errors_in_window: int,
    extra_reads: int,
) -> float:
    score = (
        1.0
        - 0.4 * (1 if retries_same_tool > 0 else 0)
        - 0.3 * min(1.0, model_errors_in_window / 2)
        - 0.2 * min(1.0, extra_reads / 5)
    )
    return max(0.0, min(1.0, score))


def _compact_score(
    error_drift: float,
    extra_read_rate: float,
    must_keep_violations: int,
) -> float:
    score = 1.0 - 2.0 * max(0.0, error_drift) - 0.5 * extra_read_rate - 1.0 * (1 if must_keep_violations > 0 else 0)
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def schedule_route(
    *,
    session_id: str,
    tool: str,
    recommended_tier: str,
    recommended_model: str,
    recommendation_followed: bool,
    scored_state: dict[str, Any],
    recommended_vendor: str | None = None,
    actual_vendor: str | None = None,
    actual_model: str | None = None,
    applied_lessons: list[str] | None = None,
    cost_cap_triggered: bool = False,
    cost_cap_limit_usd_per_session: float | None = None,
    writer: _StateWriter | None = None,
) -> str:
    """Register a pending route outcome. Returns the decision_id."""
    decision_id = uuid.uuid4().hex
    entry = RouteOutcome(
        decision_id=decision_id,
        at=datetime.now(UTC).isoformat(),
        kind="route",
        tool=tool,
        recommended_vendor=recommended_vendor,
        recommended_tier=recommended_tier,
        recommended_model=recommended_model,
        actual_vendor=actual_vendor,
        actual_model=actual_model,
        recommendation_followed=recommendation_followed,
        applied_lessons=list(applied_lessons or []),
        cost_cap_triggered=cost_cap_triggered,
        cost_cap_limit_usd_per_session=cost_cap_limit_usd_per_session,
        scored_state=dict(scored_state),
    )
    _pending_route.setdefault(session_id, []).append(entry)
    _flush(session_id, writer=writer)
    return decision_id


def schedule_compact(
    *,
    session_id: str,
    trigger: str,
    tokens_before: int,
    tokens_after: int,
    must_keep_keywords: list[str],
    errors_before: int = 0,
    writer: _StateWriter | None = None,
) -> str:
    """Register a pending compact outcome. Returns the decision_id."""
    decision_id = uuid.uuid4().hex
    entry = CompactOutcome(
        decision_id=decision_id,
        at=datetime.now(UTC).isoformat(),
        kind="compact",
        trigger=trigger,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        must_keep_keywords=list(must_keep_keywords),
        _errors_before=errors_before,
    )
    _pending_compact.setdefault(session_id, []).append(entry)
    _flush(session_id, writer=writer)
    return decision_id


def advance(
    session_id: str,
    *,
    tool_name: str = "",
    is_error: bool = False,
    is_env_error: bool = False,
    is_read_tool: bool = False,
    errors_total: int = 0,
    writer: _StateWriter | None = None,
) -> None:
    """Advance all pending outcomes for *session_id* by one turn.

    Call this once per ledger event (tool call, command result, …).
    When an outcome reaches its window size, ``outcome_window`` is filled.
    """
    _advance_routes(
        session_id,
        tool_name=tool_name,
        is_error=is_error,
        is_env_error=is_env_error,
        is_read_tool=is_read_tool,
    )
    _advance_compacts(session_id, is_error=is_error, is_read_tool=is_read_tool, errors_total=errors_total)
    _flush_commit(session_id, writer=writer)


def close_session(
    session_id: str,
    *,
    errors_total: int = 0,
    writer: _StateWriter | None = None,
) -> None:
    """Finalise all still-pending outcomes at session close."""
    _finalise_routes(session_id)
    _finalise_compacts(session_id, errors_total=errors_total)
    _flush_commit(session_id, writer=writer)


def get_outcomes(session_id: str) -> dict[str, list[dict[str, Any]]]:
    """Return the current in-process outcome lists for *session_id*."""
    route_list = [e.to_dict() for e in _pending_route.get(session_id, [])]
    compact_list = [e.to_dict() for e in _pending_compact.get(session_id, [])]
    return {"route_outcomes": route_list, "compact_outcomes": compact_list}


def emit_typed_lesson_candidate(
    store: StoreBundle,
    *,
    kind: str,
    domain: str,
    route_outcomes: list[dict[str, Any]] | None = None,
    breach_count: int = 0,
    limit_usd_per_session: float | None = None,
    source_session_id: str | None = None,
    now: datetime | None = None,
    min_occurrences: int = 3,
) -> LessonCandidate | None:
    captured_at = now or datetime.now(UTC)
    if kind == "route-preference":
        outcomes = route_outcomes or []
        recurring = [
            outcome
            for outcome in outcomes
            if outcome.get("recommended_vendor")
            and outcome.get("recommended_model")
            and not outcome.get("recommendation_followed")
        ]
        if len(recurring) < min_occurrences:
            return None
        first = recurring[0]
        tool = str(first.get("tool") or "Read")
        phase = str((first.get("scored_state") or {}).get("session_phase") or "explore")
        vendor = str(first.get("recommended_vendor") or "")
        model = str(first.get("recommended_model") or "")
        confidence = min(0.95, 0.55 + len(recurring) * 0.1)
        candidate = LessonCandidate(
            domain=domain,
            cluster_fingerprint=f"route-preference:{tool}:{phase}:{vendor}:{model}",
            kind="route-preference",
            evidence_trace_ids=[],
            body=f"Recurring route preference detected for {tool}/{phase}: prefer {vendor}:{model}",
            confidence=confidence,
            created_at=captured_at,
            evidence={
                "source_session_id": source_session_id,
                "typed_lesson": {
                    "kind": "route-preference",
                    "scope": "user",
                    "match": {"tool": tool, "phase": phase},
                    "prefer": {"vendor": vendor, "model": model},
                    "confidence": confidence,
                    "source_session_id": source_session_id,
                    "captured_at": captured_at.isoformat(),
                    "decay_half_life_days": 30,
                },
                "route_outcomes": recurring,
            },
        )
        store.lessons.upsert_lesson_candidate(candidate)
        return candidate

    if kind == "cost-cap":
        if breach_count < min_occurrences or limit_usd_per_session is None:
            return None
        candidate = LessonCandidate(
            domain=domain,
            cluster_fingerprint=f"cost-cap:{limit_usd_per_session}",
            kind="cost-cap",
            evidence_trace_ids=[],
            body=f"Recurring session cost pressure suggests a ${limit_usd_per_session:.2f} session cap.",
            confidence=1.0,
            created_at=captured_at,
            evidence={
                "source_session_id": source_session_id,
                "typed_lesson": {
                    "kind": "cost-cap",
                    "scope": "user",
                    "limit_usd_per_session": limit_usd_per_session,
                    "on_breach": "downgrade-one-tier",
                    "confidence": 1.0,
                    "source_session_id": source_session_id,
                    "captured_at": captured_at.isoformat(),
                    "decay_half_life_days": None,
                },
                "breach_count": breach_count,
            },
        )
        store.lessons.upsert_lesson_candidate(candidate)
        return candidate

    raise ValueError(f"unsupported typed lesson candidate kind: {kind}")


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _advance_routes(
    session_id: str,
    *,
    tool_name: str,
    is_error: bool,
    is_env_error: bool,
    is_read_tool: bool,
) -> None:
    _READ_TOOLS = {"Read", "View", "cat", "view", "read_file", "search_read"}
    pending = _pending_route.get(session_id, [])
    for entry in pending:
        if entry.outcome_window is not None:
            continue
        entry._turns_observed += 1
        if is_error and not is_env_error:
            entry._model_errors += 1
        if is_env_error:
            entry._env_errors += 1
        if tool_name == entry.tool:
            entry._retries_same_tool += 1
        if is_read_tool or tool_name in _READ_TOOLS:
            entry._extra_reads += 1
        if entry._turns_observed >= _ROUTE_WINDOW:
            _fill_route_window(entry)


def _advance_compacts(
    session_id: str,
    *,
    is_error: bool,
    is_read_tool: bool,
    errors_total: int,
) -> None:
    pending = _pending_compact.get(session_id, [])
    for entry in pending:
        if entry.outcome_window is not None:
            continue
        entry._turns_observed += 1
        if is_error:
            entry._errors_in_window += 1
        if is_read_tool:
            entry._extra_reads += 1
        if entry._turns_observed >= _COMPACT_WINDOW:
            _fill_compact_window(entry, errors_total=errors_total)


def _finalise_routes(session_id: str) -> None:
    for entry in _pending_route.get(session_id, []):
        if entry.outcome_window is None:
            _fill_route_window(entry)


def _finalise_compacts(session_id: str, *, errors_total: int) -> None:
    for entry in _pending_compact.get(session_id, []):
        if entry.outcome_window is None:
            _fill_compact_window(entry, errors_total=errors_total)


def _fill_route_window(entry: RouteOutcome) -> None:
    score = _route_score(
        retries_same_tool=entry._retries_same_tool,
        model_errors_in_window=entry._model_errors,
        extra_reads=entry._extra_reads,
    )
    entry.outcome_window = {
        "captured_at": datetime.now(UTC).isoformat(),
        "turns_observed": entry._turns_observed,
        "model_errors_in_window": entry._model_errors,
        "env_errors_in_window": entry._env_errors,
        "retries_same_tool": entry._retries_same_tool,
        "extra_reads": entry._extra_reads,
        "outcome_score": round(score, 4),
    }


def _fill_compact_window(entry: CompactOutcome, *, errors_total: int) -> None:
    turns = max(1, entry._turns_observed)
    error_drift = (entry._errors_in_window - entry._errors_before) / turns
    extra_read_rate = entry._extra_reads / turns
    score = _compact_score(
        error_drift=error_drift,
        extra_read_rate=extra_read_rate,
        must_keep_violations=entry._must_keep_violations,
    )
    entry.outcome_window = {
        "captured_at": datetime.now(UTC).isoformat(),
        "turns_observed": entry._turns_observed,
        "error_drift": round(error_drift, 4),
        "extra_read_rate": round(extra_read_rate, 4),
        "must_keep_violations": entry._must_keep_violations,
        "session_continued": True,
        "outcome_score": round(score, 4),
    }


# --------------------------------------------------------------------------- #
# Persistence helper                                                           #
# --------------------------------------------------------------------------- #


class _StateWriter:
    """Thin protocol: write a dict of updates to the session-state store."""

    def write(self, updates: dict[str, Any]) -> None:  # pragma: no cover
        raise NotImplementedError

    def commit(self) -> None:  # pragma: no cover
        """Drain any buffered writes to the store. No-op for writers that
        persist eagerly in ``write()``."""


# In-process cache of parsed outcome-state files, keyed by resolved path.
# Maps path -> ((st_mtime_ns, st_size), parsed_full_state_dict). This collapses
# the per-tool-call disk re-read + JSON-parse of the outcome state file (paid on
# every _handle call via _route_outcome_calibration and FileStateWriter.write)
# into a single parse that is reused until the file changes. The (mtime_ns, size)
# stamp invalidates the entry if any *other* process rewrites the file, so the
# cache never serves stale cross-process data; this process's own writes refresh
# the entry in lockstep with the on-disk bytes.
_state_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _read_state_cached(path: Path) -> dict[str, Any]:
    """Return the parsed full state dict for *path*, reusing the in-process cache.

    A fresh ``os.stat`` decides cache validity: if the file's (mtime_ns, size)
    matches the cached stamp the parsed dict is returned without re-reading or
    re-parsing. The returned dict is the cached object and MUST NOT be mutated by
    callers.
    """
    key = str(path)
    try:
        st = path.stat()
    except OSError:
        _state_cache.pop(key, None)
        return {}
    stamp = (st.st_mtime_ns, st.st_size)
    cached = _state_cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        parsed = json.loads(path.read_text("utf-8"))
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}
    state = parsed if isinstance(parsed, dict) else {}
    _state_cache[key] = (stamp, state)
    return state


# Module-level write buffer keyed by path string. FileStateWriter.write()
# merges updates here without touching disk; commit() drains the buffer with a
# single read+merge+write_text. This coalesces the two per-tool-call flushes
# (schedule_route → _flush, then advance → _flush) into ONE disk write.
_write_buffer: dict[str, dict[str, Any]] = {}


class FileStateWriter(_StateWriter):
    """Write outcomes directly to a session_state.json file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._key = str(path)

    def write(self, updates: dict[str, Any]) -> None:
        """Buffer updates in memory; call commit() to flush to disk."""
        buf = _write_buffer.setdefault(self._key, {})
        buf.update(updates)

    def commit(self) -> None:
        """Drain the write buffer to disk in a single read+merge+write_text."""
        updates = _write_buffer.pop(self._key, None)
        if updates is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        merged = {**_read_state_cached(self._path), **updates}
        self._path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        try:
            st = self._path.stat()
            _state_cache[self._key] = ((st.st_mtime_ns, st.st_size), merged)
        except OSError:
            _state_cache.pop(self._key, None)


def _flush(session_id: str, *, writer: _StateWriter | None) -> None:
    """Buffer current outcomes in the writer (no disk I/O).

    Call _flush_commit() at the terminal point of a tool call to drain
    the buffer to disk in a single write.
    """
    if writer is None:
        return
    writer.write(
        {
            "route_outcomes": [e.to_dict() for e in _pending_route.get(session_id, [])],
            "compact_outcomes": [e.to_dict() for e in _pending_compact.get(session_id, [])],
        }
    )


def _flush_commit(session_id: str, *, writer: _StateWriter | None) -> None:
    """Buffer current outcomes then commit (one disk write per tool call)."""
    _flush(session_id, writer=writer)
    if writer is not None:
        writer.commit()


# --------------------------------------------------------------------------- #
# CLI query helpers (used by lc outcomes show/summary)                   #
# --------------------------------------------------------------------------- #


def load_outcomes_from_state(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load outcomes stored in a session_state.json file."""
    state = _read_state_cached(path)
    return {
        "route_outcomes": list(state.get("route_outcomes") or []),
        "compact_outcomes": list(state.get("compact_outcomes") or []),
    }


def summarise_outcomes(
    outcomes: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate outcome_scores by (kind, tool) — matches spec 01 summary format."""
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)

    for entry in outcomes.get("route_outcomes", []):
        ow = entry.get("outcome_window")
        if ow is None:
            continue
        key = ("route", str(entry.get("tool", "")))
        buckets[key].append(float(ow.get("outcome_score", 0.0)))

    for entry in outcomes.get("compact_outcomes", []):
        ow = entry.get("outcome_window")
        if ow is None:
            continue
        key = ("compact", str(entry.get("trigger", "")))
        buckets[key].append(float(ow.get("outcome_score", 0.0)))

    return [
        {
            "kind": k,
            "tool": t,
            "count": len(scores),
            "avg_outcome_score": round(sum(scores) / len(scores), 4),
        }
        for (k, t), scores in sorted(buckets.items())
    ]
