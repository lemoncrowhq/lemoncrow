"""Fail-closed non-inferiority gate over current LemonCrow benchmark artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class TerminalBenchArmSummary:
    mode: str
    total: int
    passed: int
    failed: int
    error_like: int
    pass_rate: float
    wilson_lower: float
    wilson_upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "error_like": self.error_like,
            "pass_rate": self.pass_rate,
            "wilson_lower": self.wilson_lower,
            "wilson_upper": self.wilson_upper,
        }


@dataclass(frozen=True)
class NonInferiorityVerdict:
    candidate: TerminalBenchArmSummary
    baseline: TerminalBenchArmSummary
    margin: float
    confidence: float
    pass_rate_delta: float
    delta_lower_bound: float
    delta_upper_bound: float
    baseline_cost_usd: float
    candidate_cost_usd: float
    estimated_cost_delta_usd: float
    estimated_cost_savings_usd: float
    estimated_cost_savings_pct: float | None
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "baseline": self.baseline.to_dict(),
            "margin": self.margin,
            "confidence": self.confidence,
            "pass_rate_delta": self.pass_rate_delta,
            "delta_lower_bound": self.delta_lower_bound,
            "delta_upper_bound": self.delta_upper_bound,
            "baseline_cost_usd": self.baseline_cost_usd,
            "candidate_cost_usd": self.candidate_cost_usd,
            "estimated_cost_delta_usd": self.estimated_cost_delta_usd,
            "estimated_cost_savings_usd": self.estimated_cost_savings_usd,
            "estimated_cost_savings_pct": self.estimated_cost_savings_pct,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def load_terminalbench_records(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path)
    runs_path = candidate / "runs.jsonl" if candidate.is_dir() else candidate
    records: list[dict[str, Any]] = []
    for raw_line in runs_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _z_value(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(successes: int, total: int, *, confidence: float) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    z = _z_value(confidence)
    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    spread = z * (((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) ** 0.5) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def summarize_terminalbench_arm(
    records: list[dict[str, Any]], *, mode: str, confidence: float = 0.95
) -> TerminalBenchArmSummary:
    filtered = [record for record in records if str(record.get("mode") or "").lower() == mode.lower()]
    if not filtered:
        raise ValueError(f"no TerminalBench rows found for mode={mode!r}")

    passed = 0
    error_like = 0
    for record in filtered:
        verdict = str(record.get("grader_verdict") or "").lower()
        if verdict == "pass":
            passed += 1
        if bool(record.get("is_error")) or verdict in {"", "error"}:
            error_like += 1

    total = len(filtered)
    failed = total - passed
    lower, upper = wilson_interval(passed, total, confidence=confidence)
    return TerminalBenchArmSummary(
        mode=mode,
        total=total,
        passed=passed,
        failed=failed,
        error_like=error_like,
        pass_rate=passed / total,
        wilson_lower=lower,
        wilson_upper=upper,
    )


def evaluate_non_inferiority(
    records: list[dict[str, Any]],
    *,
    baseline_cost_usd: float,
    candidate_cost_usd: float,
    margin: float = 0.05,
    confidence: float = 0.95,
) -> NonInferiorityVerdict:
    candidate = summarize_terminalbench_arm(records, mode="on", confidence=confidence)
    baseline = summarize_terminalbench_arm(records, mode="off", confidence=confidence)
    if margin < 0.0:
        raise ValueError("margin must be non-negative")

    pass_rate_delta = candidate.pass_rate - baseline.pass_rate
    delta_lower = candidate.wilson_lower - baseline.wilson_upper
    delta_upper = candidate.wilson_upper - baseline.wilson_lower
    cost_delta = candidate_cost_usd - baseline_cost_usd
    savings = baseline_cost_usd - candidate_cost_usd
    savings_pct = (savings / baseline_cost_usd) if baseline_cost_usd > 0 else None

    reasons: list[str] = []
    if candidate_cost_usd >= baseline_cost_usd:
        reasons.append("candidate did not reduce estimated cost versus baseline")
    if delta_lower < -margin:
        reasons.append(
            f"lower confidence bound on pass-rate delta {delta_lower:.4f} is below allowed margin {-margin:.4f}"
        )

    return NonInferiorityVerdict(
        candidate=candidate,
        baseline=baseline,
        margin=margin,
        confidence=confidence,
        pass_rate_delta=pass_rate_delta,
        delta_lower_bound=delta_lower,
        delta_upper_bound=delta_upper,
        baseline_cost_usd=baseline_cost_usd,
        candidate_cost_usd=candidate_cost_usd,
        estimated_cost_delta_usd=cost_delta,
        estimated_cost_savings_usd=savings,
        estimated_cost_savings_pct=savings_pct,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def evaluate_non_inferiority_from_runs(
    path: str | Path,
    *,
    baseline_cost_usd: float,
    candidate_cost_usd: float,
    margin: float = 0.05,
    confidence: float = 0.95,
) -> NonInferiorityVerdict:
    return evaluate_non_inferiority(
        load_terminalbench_records(path),
        baseline_cost_usd=baseline_cost_usd,
        candidate_cost_usd=candidate_cost_usd,
        margin=margin,
        confidence=confidence,
    )
