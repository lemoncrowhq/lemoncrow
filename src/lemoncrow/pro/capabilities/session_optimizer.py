"""Session cost optimizer guidance and recommendation helpers.

This module keeps CodeBurn-style budget guardrails host-neutral so every
LemonCrow host can surface the same operating posture without copying rules into
host-specific code.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from lemoncrow.core.capabilities.pricing import get_model_pricing, usage_cost_usd
from lemoncrow.core.foundation.models import ToolCall, Trace
from lemoncrow.gateway.hosts.session_parsers.registry import SUPPORTED_SESSION_IMPORT_HOSTS

SUPPORTED_OPTIMIZER_HOSTS = SUPPORTED_SESSION_IMPORT_HOSTS


def effective_input_tokens(trace: Trace) -> int:
    return (
        int(trace.input_tokens or 0) + int(trace.cached_input_tokens or 0) + int(trace.cache_creation_input_tokens or 0)
    )


def trace_cost_usd(trace: Trace) -> float:
    return usage_cost_usd(
        trace.model or "_default",
        input_tokens=int(trace.input_tokens or 0),
        output_tokens=int(trace.output_tokens or 0),
        cache_read_tokens=int(trace.cached_input_tokens or 0),
        cache_write_tokens=int(trace.cache_creation_input_tokens or 0),
        thinking_tokens=int(trace.thinking_tokens or 0),
    )


def _tool_names(trace: Trace) -> set[str]:
    return {tool.name.lower() for tool in trace.tools_called}


def _has_delivery_signal(trace: Trace) -> bool:
    if trace.files_touched:
        return True
    tool_names = _tool_names(trace)
    if tool_names & {"edit", "write", "multiedit", "apply_patch", "mcp__lc__edit"}:
        return True
    for command in trace.commands_run:
        text = command.command if hasattr(command, "command") else str(command)
        lowered = text.lower()
        if "apply_patch" in lowered or "git commit" in lowered:
            return True
    return False


def _trace_created_after(trace: Trace, cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    created = trace.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created >= cutoff


def _trace_project_key(trace: Trace) -> str:
    payload_key = trace.domain or trace.host or trace.agent or "unknown"
    return str(payload_key)


def _make_recommendation(
    *,
    rule_id: str,
    title: str,
    severity: str,
    sessions: list[dict[str, Any]],
    action: str,
    estimated_tokens_saved: int,
    estimated_usd_saved: float,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "title": title,
        "severity": severity,
        "sessions": sessions,
        "session_count": len(sessions),
        "action": action,
        "estimated_tokens_saved": int(max(0, estimated_tokens_saved)),
        "estimated_usd_saved": round(max(0.0, estimated_usd_saved), 6),
    }


def build_trace_optimization_report(
    traces: Iterable[Trace],
    *,
    days: int = 7,
    host: str | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Build CodeBurn-style optimization recommendations from LemonCrow traces."""
    cutoff = datetime.now(UTC) - timedelta(days=max(1, days))
    # Trace.host is an open-ended field (SDK-recorded traces can carry any
    # provider/host string), so filter on the literal value given (empty -> no
    # filter). Bucketing an explicit-but-unknown --host would silently turn it
    # into a match-everything wildcard instead of zero matches.
    host_filter = (host or "").strip().lower() or None
    filtered = [
        trace
        for trace in traces
        if _trace_created_after(trace, cutoff)
        and (host_filter is None or (trace.host or trace.agent or "").lower() == host_filter)
    ]

    by_project: dict[str, list[Trace]] = defaultdict(list)
    costs: dict[str, float] = {}
    for trace in filtered:
        costs[trace.id] = trace_cost_usd(trace)
        by_project[_trace_project_key(trace)].append(trace)

    outliers: list[dict[str, Any]] = []
    outlier_tokens = 0
    outlier_usd = 0.0
    for project_traces in by_project.values():
        if len(project_traces) < 2:
            continue
        for trace in project_traces:
            peer_costs = [costs[other.id] for other in project_traces if other.id != trace.id]
            peer_avg = sum(peer_costs) / len(peer_costs) if peer_costs else 0.0
            cost = costs[trace.id]
            if peer_avg > 0 and cost > peer_avg * 2:
                effective_tokens = effective_input_tokens(trace) + int(trace.output_tokens or 0)
                outlier_tokens += int(effective_tokens * 0.35)
                outlier_usd += max(0.0, cost - peer_avg)
                outliers.append(
                    {
                        "trace_id": trace.id,
                        "host": trace.host or trace.agent,
                        "project": _trace_project_key(trace),
                        "cost_usd": round(cost, 6),
                        "peer_average_usd": round(peer_avg, 6),
                        "multiple": round(cost / peer_avg, 2),
                    }
                )

    context_heavy: list[dict[str, Any]] = []
    context_tokens = 0
    context_usd = 0.0
    previous_effective_by_project: dict[str, int] = {}
    for trace in sorted(filtered, key=lambda item: item.created_at):
        effective = effective_input_tokens(trace)
        output = max(1, int(trace.output_tokens or 0))
        ratio = effective / output
        previous = previous_effective_by_project.get(_trace_project_key(trace), 0)
        previous_effective_by_project[_trace_project_key(trace)] = effective
        previous_multiple = (effective / previous) if previous else 0.0
        if effective >= 250_000 and (ratio >= 30.0 or previous_multiple >= 3.0):
            saved = int(effective * 0.20)
            context_tokens += saved
            context_usd += get_model_pricing(trace.model or "_default").tokens_to_usd(saved, "input")
            context_heavy.append(
                {
                    "trace_id": trace.id,
                    "host": trace.host or trace.agent,
                    "effective_input_tokens": effective,
                    "output_tokens": int(trace.output_tokens or 0),
                    "input_output_ratio": round(ratio, 2),
                    "previous_input_multiple": round(previous_multiple, 2) if previous_multiple else None,
                }
            )

    low_worth: list[dict[str, Any]] = []
    low_tokens = 0
    low_usd = 0.0
    for trace in filtered:
        cost = costs.get(trace.id, trace_cost_usd(trace))
        effective = effective_input_tokens(trace) + int(trace.output_tokens or 0)
        if cost >= 1.0 and not _has_delivery_signal(trace):
            low_tokens += int(effective * 0.50)
            low_usd += cost * 0.50
            low_worth.append(
                {
                    "trace_id": trace.id,
                    "host": trace.host or trace.agent,
                    "cost_usd": round(cost, 6),
                    "tools": sorted(_tool_names(trace)),
                    "reason": "no edit or file-delivery signal",
                }
            )

    recommendations = []
    if outliers:
        recommendations.append(
            _make_recommendation(
                rule_id="high-cost-session-outliers",
                title="High-cost session outliers",
                severity="high",
                sessions=sorted(outliers, key=lambda item: item["cost_usd"], reverse=True)[:limit],
                action="Investigate the cost driver and narrow scope; keep loaded context minimal.",
                estimated_tokens_saved=outlier_tokens,
                estimated_usd_saved=outlier_usd,
            )
        )
    if context_heavy:
        recommendations.append(
            _make_recommendation(
                rule_id="context-heavy-sessions",
                title="Context-heavy sessions",
                severity="high",
                sessions=sorted(context_heavy, key=lambda item: item["effective_input_tokens"], reverse=True)[:limit],
                action="Bound context before editing: load only the current goal, relevant files, and failing output.",
                estimated_tokens_saved=context_tokens,
                estimated_usd_saved=context_usd,
            )
        )
    if low_worth:
        recommendations.append(
            _make_recommendation(
                rule_id="low-worth-expensive-sessions",
                title="Possibly low-worth expensive sessions",
                severity="medium",
                sessions=sorted(low_worth, key=lambda item: item["cost_usd"], reverse=True)[:limit],
                action="Name the deliverable; after two failed attempts, change approach or stop.",
                estimated_tokens_saved=low_tokens,
                estimated_usd_saved=low_usd,
            )
        )

    total_tokens = sum(int(item["estimated_tokens_saved"]) for item in recommendations)
    total_usd = sum(float(item["estimated_usd_saved"]) for item in recommendations)
    return {
        "window_days": days,
        "host": host_filter,
        "hosts_supported": list(SUPPORTED_OPTIMIZER_HOSTS),
        "trace_count": len(filtered),
        "recommendations": recommendations,
        "estimated_tokens_saved": total_tokens,
        "estimated_usd_saved": round(total_usd, 6),
        "estimation": {
            "source": "stored_lemoncrow_traces",
            "basis": "heuristic_patterns_for_outliers_context_heavy_and_low_delivery_sessions",
            "savings_are_estimates": True,
            "limitations": [
                "does not replay sessions",
                "does not prove the same task would complete with fewer tokens",
                "unknown model pricing can understate dollar estimates",
            ],
        },
    }


def mark_session_optimizer_notice(stats: dict[str, Any], notice_id: str) -> dict[str, Any]:
    updated = dict(stats)
    notices = dict(updated.get("optimizer_notices") or {})
    notices[notice_id] = True
    updated["optimizer_notices"] = notices
    return updated


def tool_is_edit(tool_name: str) -> bool:
    lowered = tool_name.strip().lower()
    return lowered.endswith("edit") or lowered in {"edit", "write", "multiedit", "apply_patch"}


def summarize_tool_calls(tools: Iterable[ToolCall]) -> list[str]:
    return sorted({tool.name for tool in tools})
