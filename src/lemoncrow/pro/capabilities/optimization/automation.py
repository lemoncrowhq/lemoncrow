"""Optimization automation runner and proposal helpers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.paths import resolve_workspace_root
from lemoncrow.core.service.telemetry.emit import emit_product_local
from lemoncrow.core.service.telemetry.schema import hash_identifier
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.infra.storage.factory import create_store
from lemoncrow.pro.capabilities.optimization.non_inferiority import (
    evaluate_non_inferiority,
    load_terminalbench_records,
)
from lemoncrow.pro.capabilities.optimization.optimizer import (
    append_history,
    optimize_from_traces,
)
from lemoncrow.pro.capabilities.optimization.policy import (
    AutomationConfig,
    BenchmarkEvidence,
    load_automation_config,
    load_current_policy,
    save_automation_config,
)
from lemoncrow.pro.capabilities.session_optimizer import build_trace_optimization_report

PROPOSAL_ARTIFACT_PATH = Path("docs/plans/world-class-lemoncrow/results/optimization/latest.json")


def run_optimization_cycle(
    *,
    store_root: Path,
    host: str | None = None,
    days: int = 7,
    source: str = "manual",
    open_pr: bool = False,
    dry_run: bool = False,
    proposal_tokens_threshold: int | None = None,
    benchmark_evidence: BenchmarkEvidence | None = None,
    store: StoreBundle | None = None,
) -> dict[str, Any]:
    resolved_store_root = Path(store_root)
    repo_root = resolve_workspace_root(resolved_store_root)
    active_store = store if store is not None else create_store(resolved_store_root)
    # Both consumers below drop traces older than *days*; apply that cutoff in
    # SQL so out-of-window payloads are never deserialized.
    traces = active_store.history.list_traces(since=datetime.now(UTC) - timedelta(days=max(1, days)), limit=5000)
    current_policy = load_current_policy(resolved_store_root)
    advisor = optimize_from_traces(traces, current_policy=current_policy, days=max(1, days), host=host)
    append_history(resolved_store_root, advisor)
    legacy_report = build_trace_optimization_report(traces, days=max(1, days), host=host, limit=6)
    automation = load_automation_config(resolved_store_root)
    if proposal_tokens_threshold is not None:
        automation = AutomationConfig(
            enabled=automation.enabled,
            minimum_projected_tokens_saved=max(0, proposal_tokens_threshold),
            benchmark_evidence=automation.benchmark_evidence,
            last_proposal_fingerprint=automation.last_proposal_fingerprint,
            last_proposal_at=automation.last_proposal_at,
        )
    evidence = benchmark_evidence or automation.benchmark_evidence
    proposal = _evaluate_proposal(
        repo_root=repo_root,
        store_root=resolved_store_root,
        source=source,
        open_pr=open_pr,
        dry_run=dry_run,
        advisor=advisor.to_dict(),
        legacy_report=legacy_report,
        automation=automation,
        evidence=evidence,
    )
    return {
        "repo_root": str(repo_root),
        "advisor": advisor.to_dict(),
        "legacy_report": legacy_report,
        "automation": automation.to_dict(),
        "proposal": proposal,
    }


def _evaluate_proposal(
    *,
    repo_root: Path,
    store_root: Path,
    source: str,
    open_pr: bool,
    dry_run: bool,
    advisor: dict[str, Any],
    legacy_report: dict[str, Any],
    automation: AutomationConfig,
    evidence: BenchmarkEvidence,
) -> dict[str, Any]:
    repo_id = hash_identifier(str(repo_root))
    projected_tokens_saved = max(0, int(legacy_report.get("estimated_tokens_saved", 0) or 0))
    recommended = bool(advisor.get("has_recommendation"))
    has_evidence = evidence.configured()
    verdict_dict: dict[str, Any] | None = None
    passed = False
    reason = "no_recommendation"
    artifact_path: str | None = None
    pr_result: dict[str, Any] | None = None

    if not recommended:
        reason = "no_recommendation"
    elif projected_tokens_saved < automation.minimum_projected_tokens_saved:
        reason = "below_projected_token_threshold"
    elif not has_evidence:
        reason = "missing_non_inferiority_evidence"
    else:
        try:
            runs = load_terminalbench_records(Path(str(evidence.runs_path)))
            verdict = evaluate_non_inferiority(
                runs,
                baseline_cost_usd=float(evidence.baseline_cost_usd or 0.0),
                candidate_cost_usd=float(evidence.candidate_cost_usd or 0.0),
                confidence=evidence.confidence,
                margin=evidence.margin,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            reason = "invalid_non_inferiority_evidence"
        else:
            verdict_dict = verdict.to_dict()
            passed = verdict.passed
            reason = "passed" if passed else "non_inferior_failed"

    emit_product_local(
        "optimization_proposal_evaluated",
        source=source,
        repo_id=repo_id,
        has_recommendation=recommended,
        projected_tokens_saved=projected_tokens_saved,
        projected_weekly_savings_usd=float(advisor.get("weekly_savings_usd", 0.0) or 0.0),
        benchmark_evidence_present=has_evidence,
        ni_passed=passed,
        open_pr_requested=open_pr,
    )

    if passed:
        fingerprint = _proposal_fingerprint(
            advisor=advisor,
            verdict=verdict_dict or {},
            threshold=automation.minimum_projected_tokens_saved,
        )
        if source != "manual" and fingerprint == automation.last_proposal_fingerprint:
            reason = "duplicate_proposal"
        else:
            artifact_path = str(
                _write_proposal_artifact(
                    repo_root=repo_root,
                    source=source,
                    advisor=advisor,
                    legacy_report=legacy_report,
                    automation=automation,
                    verdict=verdict_dict or {},
                    fingerprint=fingerprint,
                )
            )
            updated = AutomationConfig(
                enabled=automation.enabled,
                minimum_projected_tokens_saved=automation.minimum_projected_tokens_saved,
                benchmark_evidence=automation.benchmark_evidence,
                last_proposal_fingerprint=fingerprint,
                last_proposal_at=datetime.now(UTC).isoformat(),
            )
            save_automation_config(store_root, updated)
            automation = updated
            if open_pr:
                pr_result = OptimizationProposalPrBot(repo_root).open(
                    artifact_path=Path(artifact_path),
                    advisor=advisor,
                    verdict=verdict_dict or {},
                    dry_run=dry_run,
                )
                reason = "pr_dry_run" if dry_run else "pr_opened"
            else:
                reason = "artifact_written"

    emit_product_local(
        "optimization_proposal_result",
        source=source,
        repo_id=repo_id,
        action=reason,
        benchmark_evidence_present=has_evidence,
        ni_passed=passed,
        open_pr_requested=open_pr,
    )
    return {
        "action": reason,
        "artifact_path": artifact_path,
        "ni_verdict": verdict_dict,
        "open_pr": pr_result,
        "projected_tokens_saved": projected_tokens_saved,
        "minimum_projected_tokens_saved": automation.minimum_projected_tokens_saved,
    }


def _proposal_fingerprint(*, advisor: dict[str, Any], verdict: dict[str, Any], threshold: int) -> str:
    payload = {
        "recommended_preset": _recommended_preset(advisor),
        "recommended_quality_floor": _recommended_policy_field(advisor, "quality_floor", "recommended_quality_floor"),
        "recommended_confidence_required": _recommended_policy_field(
            advisor,
            "confidence_required",
            "recommended_confidence_required",
        ),
        "weekly_savings_usd": advisor.get("weekly_savings_usd"),
        "delta_lower_bound": verdict.get("delta_lower_bound"),
        "threshold": threshold,
    }
    return hash_identifier(json.dumps(payload, sort_keys=True))


def _recommended_policy(advisor: dict[str, Any]) -> dict[str, Any]:
    policy = advisor.get("recommended_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def _recommended_policy_field(advisor: dict[str, Any], field: str, legacy_field: str) -> Any:
    policy = _recommended_policy(advisor)
    if field in policy:
        return policy[field]
    return advisor.get(legacy_field)


def _recommended_preset(advisor: dict[str, Any]) -> Any:
    candidate_id = advisor.get("recommended_candidate_id")
    if candidate_id:
        return candidate_id
    legacy = advisor.get("recommended_preset")
    if legacy:
        return legacy
    policy = _recommended_policy(advisor)
    return policy.get("preset")


def _write_proposal_artifact(
    *,
    repo_root: Path,
    source: str,
    advisor: dict[str, Any],
    legacy_report: dict[str, Any],
    automation: AutomationConfig,
    verdict: dict[str, Any],
    fingerprint: str,
) -> Path:
    artifact_path = repo_root / PROPOSAL_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "fingerprint": fingerprint,
        "proposal": {
            "recommended_candidate_id": advisor.get("recommended_candidate_id"),
            "recommended_preset": _recommended_preset(advisor),
            "recommended_policy": _recommended_policy(advisor),
            "quality_floor": _recommended_policy_field(advisor, "quality_floor", "recommended_quality_floor"),
            "confidence_required": _recommended_policy_field(
                advisor,
                "confidence_required",
                "recommended_confidence_required",
            ),
            "weekly_savings_usd": advisor.get("weekly_savings_usd"),
            "estimated_tokens_saved": legacy_report.get("estimated_tokens_saved"),
            "minimum_projected_tokens_saved": automation.minimum_projected_tokens_saved,
            "estimation": advisor.get("estimation"),
        },
        "non_inferiority": verdict,
        "advisor": advisor,
        "legacy_report": legacy_report,
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_path


class OptimizationProposalPrBot:
    def __init__(
        self,
        repo_root: Path,
        *,
        run_cmd: Callable[[list[str], Path], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self._run_cmd = run_cmd or _run_subprocess

    def open(
        self,
        *,
        artifact_path: Path,
        advisor: dict[str, Any],
        verdict: dict[str, Any],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        branch = f"lemoncrow/optimize-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        title = f"Optimize policy: {_recommended_preset(advisor) or 'balanced'}"
        body = self._body(artifact_path=artifact_path, advisor=advisor, verdict=verdict)
        if dry_run:
            return {"branch": branch, "title": title, "body": body}
        self._ensure_clean_worktree()
        original_ref = self._run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        self._run(["git", "switch", "-c", branch])
        try:
            self._run(["git", "add", str(artifact_path.relative_to(self.repo_root))])
            self._run(
                [
                    "git",
                    "commit",
                    "-m",
                    title,
                    "--",
                    str(artifact_path.relative_to(self.repo_root)),
                ]
            )
            created = self._run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--draft",
                    "--title",
                    title,
                    "--body",
                    body,
                ]
            )
        except RuntimeError:
            self._rollback(branch=branch, original_ref=original_ref)
            raise
        return {"branch": branch, "title": title, "url": created.stdout.strip()}

    def _rollback(self, *, branch: str, original_ref: str) -> None:
        # Best-effort restore: return to the original ref and drop the work branch.
        if original_ref and original_ref != "HEAD":
            self._run_cmd(["git", "switch", "--force", original_ref], self.repo_root)
        self._run_cmd(["git", "branch", "-D", branch], self.repo_root)

    def _body(self, *, artifact_path: Path, advisor: dict[str, Any], verdict: dict[str, Any]) -> str:
        return "\n".join(
            [
                "## Optimization proposal",
                f"- Artifact: `{artifact_path.relative_to(self.repo_root)}`",
                f"- Recommended preset: `{_recommended_preset(advisor) or 'balanced'}`",
                f"- Estimated weekly savings: `${float(advisor.get('weekly_savings_usd', 0.0) or 0.0):.2f}`",
                f"- NI delta lower bound: `{float(verdict.get('delta_lower_bound', 0.0) or 0.0):.4f}`",
            ]
        )

    def _ensure_clean_worktree(self) -> None:
        result = self._run(["git", "status", "--porcelain"])
        if result.stdout.strip():
            msg = "open-pr requires a clean git worktree"
            raise ValueError(msg)

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._run_cmd(command, self.repo_root)
        if completed.returncode != 0:
            msg = completed.stderr.strip() or completed.stdout.strip() or "command failed"
            raise RuntimeError(msg)
        return completed


def _run_subprocess(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


__all__ = [
    "PROPOSAL_ARTIFACT_PATH",
    "OptimizationProposalPrBot",
    "run_optimization_cycle",
]
