"""Routing cost-quality eval metrics (WP-28)."""

from __future__ import annotations

from dataclasses import dataclass

from lemoncrow.core.foundation.routing_models import RoutingEvalSummary


@dataclass(frozen=True)
class RoutingEvalCase:
    """Observed routing outcome for one attempted patch flow."""

    tier: str
    accepted: bool
    cost_usd: float
    escalated: bool = False
    regression: bool = False


def summarize_routing_evals(session_id: str, cases: list[RoutingEvalCase]) -> RoutingEvalSummary:
    """Compute routing quality metrics from observed eval cases.

    Metrics focus on accepted outcomes. Failed cheap attempts still count toward
    both cost and cheap success denominator.
    """

    total = len(cases)
    if total == 0:
        return RoutingEvalSummary(
            session_id=session_id,
            cost_per_accepted_patch=0.0,
            premium_call_rate=0.0,
            cheap_success_rate=0.0,
            escalation_success_rate=0.0,
            routing_regression_rate=0.0,
        )

    accepted_count = sum(1 for c in cases if c.accepted)
    total_cost = sum(max(0.0, c.cost_usd) for c in cases)

    premium_calls = sum(1 for c in cases if c.tier == "premium")
    cheap_cases = [c for c in cases if c.tier == "cheap"]
    cheap_accepted = sum(1 for c in cheap_cases if c.accepted)

    escalated_cases = [c for c in cases if c.escalated]
    escalated_accepted = sum(1 for c in escalated_cases if c.accepted)

    regressions = sum(1 for c in cases if c.regression)

    cost_per_accepted_patch = total_cost / accepted_count if accepted_count > 0 else total_cost

    return RoutingEvalSummary(
        session_id=session_id,
        cost_per_accepted_patch=round(cost_per_accepted_patch, 6),
        premium_call_rate=round(premium_calls / total, 6),
        cheap_success_rate=(round(cheap_accepted / len(cheap_cases), 6) if cheap_cases else 0.0),
        escalation_success_rate=(round(escalated_accepted / len(escalated_cases), 6) if escalated_cases else 0.0),
        routing_regression_rate=round(regressions / total, 6),
    )


__all__ = ["RoutingEvalCase", "summarize_routing_evals"]
