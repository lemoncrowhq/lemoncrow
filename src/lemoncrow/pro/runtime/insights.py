"""Spec 04 — ``lc insights``

Aggregates ``SessionReport`` objects over a time window into an
``InsightsWindow`` with vendor/tool breakdowns, outcomes summary, and
opportunity detection.

Usage::

    from lemoncrow.pro.runtime.insights import build_insights, render_text
    window = build_insights(root, since=since_dt, until=until_dt)
    print(render_text(window))
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lemoncrow.infra.runtime.session_report import (
    SessionReport,
    _model_vendor,
    list_run_files,
)
from lemoncrow.pro.runtime.outcome_capture import load_outcomes_from_state

# Approximate Gemini Flash 1.5 input price (public list: $0.075/1M tokens).
_GEMINI_FLASH_INPUT_PER_TOKEN: float = 0.075 / 1_000_000

# Read-class tool names (case-insensitive prefix match).
_READ_TOOLS: frozenset[str] = frozenset({"read", "grep", "glob", "view", "search"})

# Minimum estimated savings to surface an opportunity.
_MIN_OPPORTUNITY_SAVINGS: float = 0.50


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class SessionSummary:
    session_id: str
    cost_usd: float
    label: str
    duration_seconds: float


@dataclass
class OutcomesSummary:
    route_decisions: int
    route_avg_score: float
    compact_events: int
    compact_avg_score: float
    sessions_with_high_extra_reads: list[str]


@dataclass
class Opportunity:
    kind: str
    message: str
    estimated_savings_usd: float
    sessions_affected: int


@dataclass
class InsightsWindow:
    since: datetime
    until: datetime
    session_count: int
    total_duration_seconds: float
    total_cost_usd: float
    total_lemoncrow_savings_usd: float
    cost_by_vendor: dict[str, float]
    cost_by_tool: dict[str, float]
    cost_by_model: dict[str, float]
    top_sessions: list[SessionSummary]
    outcomes_summary: OutcomesSummary
    opportunities: list[Opportunity]


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _cost_by_vendor_model(
    calls: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return ``(by_vendor, by_model)`` dicts from a cost.calls list."""
    by_vendor: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    for call in calls:
        model = str(call.get("model") or "")
        cost = float(call.get("cost_usd") or 0.0)
        by_vendor[_model_vendor(model)] += cost
        if model:
            by_model[model] += cost
    return dict(by_vendor), dict(by_model)


def _session_label(snap: dict[str, Any]) -> str:
    """Return a short human label for a session from its snapshot."""
    task = str(snap.get("task") or "").strip()
    if task:
        return task[:40]
    sid = str(snap.get("session_id") or snap.get("run_id") or "")
    return sid[:8] if sid else "(unknown)"


def _read_tool_cost_fraction(report: SessionReport) -> float:
    """Fraction of tool cost that goes to read-class tools (0.0-1.0)."""
    if report.total_cost_usd <= 0:
        return 0.0
    read_cost = sum(cost for tool_name, _count, cost in report.top_tools_by_cost if tool_name.lower() in _READ_TOOLS)
    return read_cost / report.total_cost_usd


def _load_outcomes_for_session(
    session_id: str,
    root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load outcomes from the session's ``outcomes.json`` (host-agnostic lookup)."""
    from lemoncrow.core.foundation.paths import find_session_dir

    session_path = find_session_dir(root, session_id)
    if session_path is None:
        return {"route_outcomes": [], "compact_outcomes": []}
    return load_outcomes_from_state(session_path / "outcomes.json")


# --------------------------------------------------------------------------- #
# Opportunity detection                                                        #
# --------------------------------------------------------------------------- #


def _rule_cross_vendor_route(
    reports: list[SessionReport],
) -> Opportunity | None:
    """Fire if >5 sessions have >30% read-tool cost AND primary vendor is not Google."""
    affected_sessions: list[SessionReport] = []
    total_read_cost = 0.0
    for r in reports:
        if r.vendor in ("Google", "Unknown"):
            continue
        frac = _read_tool_cost_fraction(r)
        if frac > 0.30:
            affected_sessions.append(r)
            total_read_cost += r.total_cost_usd * frac

    if len(affected_sessions) <= 5:
        return None

    # Estimate: Gemini Flash is ~10x cheaper on input than Anthropic/OpenAI.
    # Use a conservative 50% savings on the read portion.
    estimated_savings = total_read_cost * 0.50
    if estimated_savings < _MIN_OPPORTUNITY_SAVINGS:
        return None

    return Opportunity(
        kind="cross_vendor_route",
        message=(
            f"{len(affected_sessions)} sessions had 30%+ read turns"
            f" — Gemini Flash would cost ~${estimated_savings:.2f} less"
        ),
        estimated_savings_usd=round(estimated_savings, 2),
        sessions_affected=len(affected_sessions),
    )


def _rule_compact_aggression(
    outcomes_by_session: dict[str, dict[str, list[dict[str, Any]]]],
) -> Opportunity | None:
    """Fire if avg compact extra_read_rate > 0.15 across sessions."""
    rates: list[float] = []
    for data in outcomes_by_session.values():
        for evt in data.get("compact_outcomes") or []:
            rate = float((evt.get("outcome_window") or {}).get("extra_read_rate") or 0.0)
            rates.append(rate)

    if not rates:
        return None

    avg_rate = sum(rates) / len(rates)
    if avg_rate <= 0.15:
        return None

    sessions_affected = sum(
        1
        for data in outcomes_by_session.values()
        if any(
            float((e.get("outcome_window") or {}).get("extra_read_rate") or 0.0) > 0.15
            for e in (data.get("compact_outcomes") or [])
        )
    )

    return Opportunity(
        kind="compact_aggression",
        message=(f"avg compact extra_read_rate {avg_rate:.0%} — consider tuning down compact aggression"),
        estimated_savings_usd=0.0,
        sessions_affected=sessions_affected,
    )


def _rule_sync_value(
    snaps: list[dict[str, Any]],
) -> Opportunity | None:
    """Fire if sessions were run on multiple machines."""
    machines = {str(s.get("machine_id") or "").strip() for s in snaps}
    machines.discard("")
    if len(machines) <= 1:
        return None

    return Opportunity(
        kind="sync_value",
        message=(f"{len(machines)} distinct machines detected — sync would help during travel"),
        estimated_savings_usd=0.0,
        sessions_affected=len(snaps),
    )


def _rule_error_pattern(
    snaps: list[dict[str, Any]],
    reports: list[SessionReport],
) -> Opportunity | None:
    """Fire if >10% of sessions in window have errors and a dominant error tool exists."""
    if not snaps:
        return None
    error_sessions = [s for s in snaps if int(s.get("errors_seen") or 0) > 0]
    if not error_sessions:
        return None
    rate = len(error_sessions) / len(snaps)
    if rate <= 0.10:
        return None

    # Find most-errored tool from per-report top tools (rough proxy).
    tool_errors: dict[str, int] = defaultdict(int)
    for report in reports:
        for snap in snaps:
            if snap.get("session_id") == report.session_id and int(snap.get("errors_seen") or 0) > 0:
                for tool_name, _count, _cost in report.top_tools_by_cost[:1]:
                    tool_errors[tool_name] += 1

    if not tool_errors:
        return None

    top_tool = max(tool_errors, key=lambda k: tool_errors[k])
    affected = len(error_sessions)

    if affected < 5:
        return None

    return Opportunity(
        kind="error_pattern",
        message=(
            f"{top_tool} has high error rate across {affected} sessions — consider stronger routing for that tool"
        ),
        estimated_savings_usd=0.0,
        sessions_affected=affected,
    )


def _detect_opportunities(
    reports: list[SessionReport],
    snaps: list[dict[str, Any]],
    outcomes_by_session: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[Opportunity]:
    """Run all opportunity rules and return at most 5, sorted by savings desc."""
    candidates: list[Opportunity] = []

    opp = _rule_cross_vendor_route(reports)
    if opp is not None:
        candidates.append(opp)

    opp = _rule_compact_aggression(outcomes_by_session)
    if opp is not None:
        candidates.append(opp)

    opp = _rule_sync_value(snaps)
    if opp is not None:
        candidates.append(opp)

    opp = _rule_error_pattern(snaps, reports)
    if opp is not None:
        candidates.append(opp)

    candidates.sort(key=lambda o: o.estimated_savings_usd, reverse=True)
    return candidates[:5]


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #


def build_insights(
    root: Path,
    since: datetime,
    until: datetime,
) -> InsightsWindow:
    """Aggregate all sessions in ``[since, until)`` into an ``InsightsWindow``."""
    # Load raw snapshots and build reports in one pass.
    files = list_run_files(root, since=since)
    snaps: list[dict[str, Any]] = []
    reports: list[SessionReport] = []
    # Parallel list: snap_by_index[i] is the raw snapshot for reports[i]
    snap_by_index: list[dict[str, Any]] = []
    outcomes_by_session: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for f in files:
        try:
            snap: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("Recovered from broad exception handler")
            continue

        from lemoncrow.infra.runtime.session_report import build_report

        try:
            report = build_report(snap, root)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            continue

        if report.started_at > until:
            continue
        if report.started_at < since:
            continue

        snaps.append(snap)
        snap_by_index.append(snap)
        reports.append(report)

        # Use session_id from report; fall back to run_id from snapshot for outcomes lookup.
        sid = report.session_id or str(snap.get("run_id") or "")
        outcomes_by_session[sid] = _load_outcomes_for_session(sid, root)

    # Aggregate totals.
    total_cost = sum(r.total_cost_usd for r in reports)
    total_savings = sum(r.total_lemoncrow_savings_usd for r in reports)
    total_duration = sum(r.duration_seconds for r in reports)

    # Cost by vendor and model (from raw cost.calls).
    agg_by_vendor: dict[str, float] = defaultdict(float)
    agg_by_model: dict[str, float] = defaultdict(float)
    for snap in snaps:
        calls: list[dict[str, Any]] = list((snap.get("cost") or {}).get("calls") or [])
        if not calls:
            # Fall back: attribute total cost to session's primary vendor.

            session_report = next(
                (r for r in reports if r.session_id == snap.get("session_id")),
                None,
            )
            if session_report and session_report.total_cost_usd > 0:
                agg_by_vendor[session_report.vendor] += session_report.total_cost_usd
        else:
            bv, bm = _cost_by_vendor_model(calls)
            for v, c in bv.items():
                agg_by_vendor[v] += c
            for m, c in bm.items():
                agg_by_model[m] += c

    # Cost by tool (sum across all sessions' top_tools_by_cost).
    agg_by_tool: dict[str, float] = defaultdict(float)
    for r in reports:
        for tool_name, _count, cost in r.top_tools_by_cost:
            agg_by_tool[tool_name] += cost

    # Top sessions by cost (up to 5).
    sorted_pairs = sorted(zip(reports, snap_by_index, strict=True), key=lambda rs: rs[0].total_cost_usd, reverse=True)
    top_sessions = [
        SessionSummary(
            session_id=r.session_id or str(s.get("run_id") or "")[:8],
            cost_usd=r.total_cost_usd,
            label=_session_label(s),
            duration_seconds=r.duration_seconds,
        )
        for r, s in sorted_pairs[:5]
    ]

    # Outcomes summary.
    all_route: list[dict[str, Any]] = []
    all_compact: list[dict[str, Any]] = []
    for data in outcomes_by_session.values():
        all_route.extend(data.get("route_outcomes") or [])
        all_compact.extend(data.get("compact_outcomes") or [])

    route_scores = [float((e.get("outcome_window") or {}).get("outcome_score") or 0.0) for e in all_route]
    compact_scores = [float((e.get("outcome_window") or {}).get("outcome_score") or 0.0) for e in all_compact]

    high_extra_read_sessions = [
        sid
        for sid, data in outcomes_by_session.items()
        if any(
            float((e.get("outcome_window") or {}).get("extra_read_rate") or 0.0) > 0.20
            for e in (data.get("compact_outcomes") or [])
        )
    ]

    outcomes_summary = OutcomesSummary(
        route_decisions=len(all_route),
        route_avg_score=round(sum(route_scores) / len(route_scores), 4) if route_scores else 0.0,
        compact_events=len(all_compact),
        compact_avg_score=round(sum(compact_scores) / len(compact_scores), 4) if compact_scores else 0.0,
        sessions_with_high_extra_reads=high_extra_read_sessions,
    )

    # Opportunities.
    opportunities = _detect_opportunities(reports, snaps, outcomes_by_session)

    return InsightsWindow(
        since=since,
        until=until,
        session_count=len(reports),
        total_duration_seconds=total_duration,
        total_cost_usd=round(total_cost, 4),
        total_lemoncrow_savings_usd=round(total_savings, 4),
        cost_by_vendor=dict(sorted(agg_by_vendor.items(), key=lambda kv: kv[1], reverse=True)),
        cost_by_tool=dict(sorted(agg_by_tool.items(), key=lambda kv: kv[1], reverse=True)),
        cost_by_model=dict(sorted(agg_by_model.items(), key=lambda kv: kv[1], reverse=True)),
        top_sessions=top_sessions,
        outcomes_summary=outcomes_summary,
        opportunities=opportunities,
    )


# --------------------------------------------------------------------------- #
# Renderers                                                                    #
# --------------------------------------------------------------------------- #


def _bar(fraction: float, width: int = 20) -> str:
    """Return a Unicode bar of *width* chars with filled/empty blocks."""
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h {m}m"


def _localdt(dt: datetime) -> str:
    """Format *dt* in local timezone, YYYY-MM-DD."""
    try:
        local = dt.astimezone()
        return local.strftime("%Y-%m-%d")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return dt.strftime("%Y-%m-%d")


def render_text(window: InsightsWindow, *, no_color: bool = False) -> str:
    """Render a human-readable insights summary (80 cols, Unicode bars)."""
    lines: list[str] = []

    since_str = _localdt(window.since)
    until_str = _localdt(window.until)
    lines.append(f"Weekly insights · {since_str} to {until_str}")
    lines.append("─" * 49)

    avg_min = (window.total_duration_seconds / window.session_count / 60) if window.session_count else 0.0
    total_h, total_rem = divmod(int(window.total_duration_seconds), 3600)
    total_m = total_rem // 60
    savings_pct = (
        (window.total_lemoncrow_savings_usd / window.total_cost_usd * 100) if window.total_cost_usd > 0 else 0.0
    )
    lines.append(
        f"Sessions:         {window.session_count}"
        + (f" (avg {int(avg_min)} min, total {total_h}h {total_m}m)" if window.session_count else "")
    )
    lines.append(f"AI spend:         ${window.total_cost_usd:.2f}")
    lines.append(
        f"LemonCrow savings:  ${window.total_lemoncrow_savings_usd:.2f}"
        + (f"  ({savings_pct:.1f}% of total)" if window.total_cost_usd > 0 else "")
    )

    # Cost by vendor.
    if window.cost_by_vendor:
        lines.append("")
        lines.append("Cost by vendor")
        for vendor, cost in list(window.cost_by_vendor.items())[:5]:
            pct = cost / window.total_cost_usd if window.total_cost_usd > 0 else 0.0
            bar = _bar(pct)
            lines.append(f"  {vendor:<12}  ${cost:>7.2f}  {pct * 100:>3.0f}%  {bar}")

    # Cost by tool (top 5).
    if window.cost_by_tool:
        lines.append("")
        lines.append("Cost by tool (top 5)")
        top_tools = list(window.cost_by_tool.items())[:5]
        for tool_name, cost in top_tools:
            pct = cost / window.total_cost_usd if window.total_cost_usd > 0 else 0.0
            lines.append(f"  {tool_name:<14}  ${cost:>7.2f}  {pct * 100:>3.0f}%")

    # Top spending sessions.
    if window.top_sessions:
        lines.append("")
        lines.append("Top spending sessions")
        for i, s in enumerate(window.top_sessions, start=1):
            sid_short = s.session_id[:6] if s.session_id else "??????"
            dur = _fmt_duration(s.duration_seconds)
            label = s.label[:40]
            lines.append(f"  {i}. {sid_short}    ${s.cost_usd:>6.2f}   {label!r:<44}  {dur}")

    # Outcomes.
    oc = window.outcomes_summary
    lines.append("")
    lines.append("Outcomes")
    lines.append(
        f"  Route decisions: {oc.route_decisions}"
        + (f" (avg outcome_score {oc.route_avg_score:.2f})" if oc.route_decisions else "")
    )
    lines.append(
        f"  Compact events:  {oc.compact_events}"
        + (f" (avg outcome_score {oc.compact_avg_score:.2f})" if oc.compact_events else "")
    )
    if oc.sessions_with_high_extra_reads:
        n = len(oc.sessions_with_high_extra_reads)
        lines.append(f'  Sessions hitting "extra_reads > 0.2": {n} — review compact aggression')

    # Opportunities.
    if window.opportunities:
        lines.append("")
        lines.append("Opportunities")
        for opp in window.opportunities:
            lines.append(f"  * {opp.message}")

    return "\n".join(lines)


def render_json(window: InsightsWindow) -> str:
    """Render the insights window as JSON."""

    def _default(obj: object) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def _to_dict(w: InsightsWindow) -> dict[str, Any]:
        return {
            "since": w.since.isoformat(),
            "until": w.until.isoformat(),
            "session_count": w.session_count,
            "total_duration_seconds": w.total_duration_seconds,
            "total_cost_usd": w.total_cost_usd,
            "total_lemoncrow_savings_usd": w.total_lemoncrow_savings_usd,
            "cost_by_vendor": w.cost_by_vendor,
            "cost_by_tool": w.cost_by_tool,
            "cost_by_model": w.cost_by_model,
            "top_sessions": [
                {
                    "session_id": s.session_id,
                    "cost_usd": s.cost_usd,
                    "label": s.label,
                    "duration_seconds": s.duration_seconds,
                }
                for s in w.top_sessions
            ],
            "outcomes_summary": {
                "route_decisions": w.outcomes_summary.route_decisions,
                "route_avg_score": w.outcomes_summary.route_avg_score,
                "compact_events": w.outcomes_summary.compact_events,
                "compact_avg_score": w.outcomes_summary.compact_avg_score,
                "sessions_with_high_extra_reads": w.outcomes_summary.sessions_with_high_extra_reads,
            },
            "opportunities": [
                {
                    "kind": o.kind,
                    "message": o.message,
                    "estimated_savings_usd": o.estimated_savings_usd,
                    "sessions_affected": o.sessions_affected,
                }
                for o in w.opportunities
            ],
        }

    return json.dumps(_to_dict(window), indent=2, default=_default)
