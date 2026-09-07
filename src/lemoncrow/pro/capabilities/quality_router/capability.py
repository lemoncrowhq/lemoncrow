"""Quality-aware router runtime capability (WP-26)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from lemoncrow.core.foundation.models import ValidationResult
from lemoncrow.core.foundation.routing_models import (
    AgentRequest,
    ContextBudgetPolicy,
    RouteDecision,
    StepType,
    TaskType,
    VerificationEnvelope,
)
from lemoncrow.pro.capabilities.quality_router.config import (
    RoutingPolicyConfig,
    load_routing_policy_config,
)
from lemoncrow.pro.capabilities.quality_router.execution_contract import (
    RouteExecutionContract,
    route_execution_contract,
)
from lemoncrow.pro.capabilities.quality_router.policy import (
    draft_route_decision,
    required_verifiers,
    selected_model_for_tier,
)
from lemoncrow.pro.capabilities.quality_router.verifier import verify_route
from lemoncrow.pro.foundation.retriever import TaskContext, retrieve

if TYPE_CHECKING:
    from lemoncrow.infra.runtime.run_ledger import RunLedger
    from lemoncrow.infra.storage.bundle import StoreBundle


class QualityRouterCapability:
    """Deterministic router integration for runtime, MCP, and CLI surfaces."""

    def __init__(
        self,
        store: StoreBundle,
        repo_root: str | Path,
    ) -> None:
        self.store = store
        self.repo_root = Path(repo_root)
        self.config = load_routing_policy_config(self.repo_root)

    def reload_config(self) -> RoutingPolicyConfig:
        self.config = load_routing_policy_config(self.repo_root)
        return self.config

    def decide(
        self,
        *,
        user_goal: str,
        repo_root: str,
        task_type: TaskType,
        risk_level: str,
        changed_files: list[str] | None = None,
        domain: str | None = None,
        step_type: StepType = "plan",
        step_index: int = 0,
        session_id: str | None = None,
        evidence_summary: Mapping[str, object] | None = None,
        ledger: RunLedger | None = None,
    ) -> RouteDecision:
        files = self._merge_changed_files(changed_files, ledger)
        request = AgentRequest(
            session_id=session_id or (ledger.session_id if ledger is not None else None),
            user_goal=user_goal,
            repo_root=repo_root,
            task_type=task_type,
            risk_level=risk_level,  # type: ignore[arg-type]
            changed_files=files,
        )

        budget = self._budget_policy(request)
        summary = self._build_evidence_summary(
            request=request,
            budget=budget,
            domain=domain,
            evidence_summary=evidence_summary,
            ledger=ledger,
        )

        decision = draft_route_decision(
            request=request,
            budget=budget,
            config=self.config,
            step_type=step_type,
            step_index=step_index,
            domain=domain,
            evidence_summary=summary,
        )

        self._apply_runtime_overrides(
            decision=decision,
            request=request,
            budget=budget,
            summary=summary,
            step_type=step_type,
        )
        return decision

    def verify(
        self,
        *,
        route_decision_id: str,
        session_id: str,
        changed_files: list[str] | None = None,
        validation_results: list[ValidationResult | dict[str, object]] | None = None,
        rubric_status: str = "not_run",
        required_verifiers: list[str] | None = None,
        protected_file_match: bool = False,
        repeated_failure_signatures: list[str] | None = None,
        diff_line_count: int = 0,
        human_accepted: bool | None = None,
        benchmark_accepted: bool | None = None,
    ) -> VerificationEnvelope:
        return verify_route(
            route_decision_id=route_decision_id,
            session_id=session_id,
            changed_files=changed_files,
            validation_results=validation_results,
            rubric_status=rubric_status,  # type: ignore[arg-type]
            required_verifiers=required_verifiers,
            protected_file_match=protected_file_match,
            repeated_failure_signatures=repeated_failure_signatures,
            diff_line_count=diff_line_count,
            human_accepted=human_accepted,
            benchmark_accepted=benchmark_accepted,
        )

    def contract(self, host: str) -> RouteExecutionContract:
        """Return the routing execution contract for the named host (WP-31).

        Raises ``ValueError`` for unknown hosts.
        The ``provider_enforced`` mode is always disabled.
        """
        return route_execution_contract(host)

    def _budget_policy(self, request: AgentRequest) -> ContextBudgetPolicy:
        max_tokens = max(4096, 4096 + 800 * len(request.changed_files))
        return self.config.budget_policy(max_input_tokens=max_tokens)

    def _build_evidence_summary(
        self,
        *,
        request: AgentRequest,
        budget: ContextBudgetPolicy,
        domain: str | None,
        evidence_summary: Mapping[str, object] | None,
        ledger: RunLedger | None,
    ) -> dict[str, object]:
        summary = dict(evidence_summary or {})

        raw_refs = summary.get("refs")
        refs = list(raw_refs) if isinstance(raw_refs, list) else []

        scored = retrieve(
            self.store,
            TaskContext(
                task=request.user_goal,
                domain=domain,
                files=request.changed_files,
                tools=(ledger.tools_called if ledger else []),
            ),
            limit=5,
            token_budget=2000,
            dedup=True,
        )
        playbook_refs = [f"block:{entry.block.id}" for entry in scored]
        refs.extend(playbook_refs)

        errors_seen = list(ledger.errors_seen) if ledger else []
        repeated_failures = len(ledger.repeated_failures) if ledger else 0

        estimated_tokens = summary.get("estimated_input_tokens")
        if not isinstance(estimated_tokens, int):
            estimated_tokens = self._latest_input_tokens(request.session_id) or self._estimate_input_tokens(
                request, len(playbook_refs)
            )

        base_confidence = self._confidence_from_runtime(
            playbook_count=len(playbook_refs),
            errors_seen=len(errors_seen),
            repeated_failures=repeated_failures,
        )
        supplied_confidence = summary.get("confidence")
        if isinstance(supplied_confidence, int | float):
            confidence = min(float(supplied_confidence), base_confidence)
        else:
            confidence = base_confidence

        memory_confidence = summary.get("memory_confidence")
        if isinstance(memory_confidence, int | float):
            confidence = min(confidence, float(memory_confidence))

        verifier_coverage = summary.get("verifier_coverage")
        if isinstance(verifier_coverage, int | float):
            summary["verifier_coverage"] = max(0.0, min(1.0, float(verifier_coverage)))
        else:
            summary["verifier_coverage"] = 1.0

        summary["confidence"] = max(0.05, min(1.0, confidence))
        summary["estimated_input_tokens"] = int(estimated_tokens)
        summary["playbook_count"] = len(playbook_refs)
        summary["errors_seen"] = len(errors_seen)
        summary["repeated_failures"] = repeated_failures
        summary["max_input_tokens"] = budget.max_input_tokens
        summary["refs"] = sorted({str(ref) for ref in refs})
        return summary

    def _apply_runtime_overrides(
        self,
        *,
        decision: RouteDecision,
        request: AgentRequest,
        budget: ContextBudgetPolicy,
        summary: Mapping[str, object],
        step_type: StepType,
    ) -> None:
        raw_repeated = summary.get("repeated_failures", 0)
        repeated_failures = int(raw_repeated) if isinstance(raw_repeated, int | float) else 0
        raw_coverage = summary.get("verifier_coverage", 1.0)
        verifier_coverage = float(raw_coverage) if isinstance(raw_coverage, int | float) else 1.0
        raw_confidence = summary.get("confidence", 1.0)
        confidence = float(raw_confidence) if isinstance(raw_confidence, int | float) else 1.0

        deterministic_step = step_type in {"classify", "compress", "retrieve", "summarize"}
        if (
            deterministic_step
            and request.risk_level == "low"
            and not decision.protected_file_match
            and repeated_failures == 0
            and confidence >= 0.90
        ):
            decision.tier = "deterministic"
            decision.selected_model = ""
            decision.escalation_trigger = None
            decision.reason += ", deterministic_step=true"
            return

        forced_trigger: str | None = None
        if repeated_failures > 0:
            forced_trigger = "repeated_failure"
        elif verifier_coverage < 0.50:
            forced_trigger = "verifier_gap"

        if forced_trigger is not None:
            decision.tier = "premium"
            decision.selected_model = selected_model_for_tier("premium", budget)
            if decision.escalation_trigger is None:
                decision.escalation_trigger = forced_trigger
            decision.verifier_required = required_verifiers(
                config=self.config,
                protected_file_match=decision.protected_file_match,
                high_risk=True,
                low_confidence=True,
            )
            decision.reason += f", forced_escalation={forced_trigger}"

        refs = summary.get("refs", [])
        if isinstance(refs, list):
            decision.evidence_refs = [str(ref) for ref in refs]

    def _merge_changed_files(self, changed_files: list[str] | None, ledger: RunLedger | None) -> list[str]:
        merged: list[str] = []
        for path in changed_files or []:
            if path not in merged:
                merged.append(path)
        if ledger is not None:
            for path in ledger.files_touched:
                if path not in merged:
                    merged.append(path)
        return merged

    def _latest_input_tokens(self, session_id: str | None) -> int | None:
        if not session_id:
            return None
        budgets = self.store.telemetry.list_context_budgets(session_id)
        if not budgets:
            return None
        latest = budgets[-1]
        return int(getattr(latest, "input_tokens", 0))

    @staticmethod
    def _estimate_input_tokens(request: AgentRequest, playbook_count: int) -> int:
        goal_tokens = max(128, len(request.user_goal) // 4)
        file_tokens = 400 * max(0, len(request.changed_files))
        reason_tokens = 200 * max(0, playbook_count)
        return goal_tokens + file_tokens + reason_tokens

    @staticmethod
    def _confidence_from_runtime(
        *,
        playbook_count: int,
        errors_seen: int,
        repeated_failures: int,
    ) -> float:
        confidence = 1.0
        if playbook_count == 0:
            confidence -= 0.15
        confidence -= min(0.30, errors_seen * 0.10)
        confidence -= min(0.30, repeated_failures * 0.15)
        return max(0.05, min(1.0, confidence))


__all__ = ["QualityRouterCapability"]
