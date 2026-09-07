"""Verification-gated escalation policy for routing (WP-27)."""

from __future__ import annotations

from typing import Literal

from lemoncrow.core.foundation.models import ValidationResult
from lemoncrow.core.foundation.routing_models import VerificationEnvelope

RubricStatus = Literal["not_run", "pass", "warn", "fail"]


def verify_route(
    *,
    route_decision_id: str,
    session_id: str,
    changed_files: list[str] | None = None,
    validation_results: list[ValidationResult | dict[str, object]] | None = None,
    rubric_status: RubricStatus = "not_run",
    required_verifiers: list[str] | None = None,
    protected_file_match: bool = False,
    repeated_failure_signatures: list[str] | None = None,
    diff_line_count: int = 0,
    human_accepted: bool | None = None,
    benchmark_accepted: bool | None = None,
) -> VerificationEnvelope:
    """Convert observed verification signals into a serializable outcome envelope.

    This verifier consumes existing validation observations only. It does not execute
    shell commands and does not attempt to run tests/lint/type-check itself.
    """

    files = changed_files or []
    results = _normalize_validation_results(validation_results or [])
    required = [v.strip() for v in (required_verifiers or []) if v.strip()]
    repeated = [s for s in (repeated_failure_signatures or []) if s]

    failed = [r.name for r in results if not r.passed]
    covered = _covered_verifiers(results, rubric_status=rubric_status)
    missing = [v for v in required if v not in covered]

    escalation_reasons: list[str] = []
    warn_reasons: list[str] = []

    if repeated:
        escalation_reasons.append("repeated_failures")
    if protected_file_match and failed:
        escalation_reasons.append("protected_file_failed_validation")
    if diff_line_count >= 1200:
        escalation_reasons.append("unexpectedly_large_diff")

    if (rubric_status == "fail" or failed) and not escalation_reasons:
        # Failures on unprotected files can remain "fail" without forced escalation.
        warn_reasons.append("validation_failed")
    if missing:
        warn_reasons.append("missing_required_verification")
    if rubric_status == "warn":
        warn_reasons.append("rubric_warn")
    if 400 <= diff_line_count < 1200:
        warn_reasons.append("large_diff")
    if benchmark_accepted is False:
        warn_reasons.append("benchmark_rejected")

    if escalation_reasons:
        outcome: Literal["pass", "warn", "fail", "escalate"] = "escalate"
    elif rubric_status == "fail" or failed or human_accepted is False:
        outcome = "fail"
    elif missing or warn_reasons or human_accepted is None:
        # Unknown human outcome is a warning for manual follow-up.
        outcome = "warn"
    else:
        outcome = "pass"

    evidence = _compress_evidence(
        failed=failed,
        missing=missing,
        repeated_count=len(repeated),
        diff_line_count=diff_line_count,
        rubric_status=rubric_status,
        escalation_reasons=escalation_reasons,
        warn_reasons=warn_reasons,
        protected_file_match=protected_file_match,
    )

    return VerificationEnvelope(
        route_decision_id=route_decision_id,
        session_id=session_id,
        changed_files=files,
        validation_results=results,
        rubric_status=rubric_status,
        outcome=outcome,
        compressed_evidence=evidence,
        human_accepted=human_accepted,
    )


def _normalize_validation_results(
    raw: list[ValidationResult | dict[str, object]],
) -> list[ValidationResult]:
    normalized: list[ValidationResult] = []
    for item in raw:
        if isinstance(item, ValidationResult):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            name = str(item.get("name", "validation"))
            passed = bool(item.get("passed", False))
            detail = str(item.get("detail", ""))
            normalized.append(ValidationResult(name=name, passed=passed, detail=detail))
    return normalized


def _covered_verifiers(results: list[ValidationResult], *, rubric_status: RubricStatus) -> set[str]:
    covered: set[str] = set()
    for res in results:
        key = res.name.lower()
        if "test" in key or "pytest" in key:
            covered.add("tests")
        if "lint" in key or "ruff" in key:
            covered.add("lint")
        if "type" in key or "mypy" in key:
            covered.add("typecheck")
        if "review" in key:
            covered.add("review")
        if "bench" in key:
            covered.add("benchmark")
    if rubric_status in {"pass", "warn"}:
        covered.add("rubric")
    return covered


def _compress_evidence(
    *,
    failed: list[str],
    missing: list[str],
    repeated_count: int,
    diff_line_count: int,
    rubric_status: RubricStatus,
    escalation_reasons: list[str],
    warn_reasons: list[str],
    protected_file_match: bool,
) -> str:
    parts: list[str] = []
    parts.append(f"rubric={rubric_status}")
    if protected_file_match:
        parts.append("protected_file=true")
    if failed:
        parts.append("failed=" + ",".join(failed[:4]))
    if missing:
        parts.append("missing=" + ",".join(missing[:4]))
    if repeated_count > 0:
        parts.append(f"repeated_failures={repeated_count}")
    if diff_line_count > 0:
        parts.append(f"diff_lines={diff_line_count}")
    if escalation_reasons:
        parts.append("escalate=" + ",".join(escalation_reasons[:4]))
    if warn_reasons:
        parts.append("warn=" + ",".join(warn_reasons[:4]))
    return " | ".join(parts)[:800]


__all__ = ["verify_route"]
