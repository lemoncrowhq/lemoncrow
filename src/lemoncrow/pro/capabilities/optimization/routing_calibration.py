"""Calibrated expected-total-cost routing for LemonCrow-owned phase boundaries."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DEFAULT_MIN_SAMPLES = 20
_PRIOR_STRENGTH = 4.0


@dataclass(frozen=True)
class CalibratedRoute:
    provider: str
    model: str
    tier: str
    direct_cost_usd: float
    failure_probability: float
    escalation_cost_usd: float
    cache_break_cost_usd: float
    expected_total_cost_usd: float
    sample_count: int
    eligible: bool
    reason: str

    def trace(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "failure_probability": round(self.failure_probability, 4),
            "expected_total_cost_usd": round(self.expected_total_cost_usd, 8),
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class _Candidate:
    provider: str
    model: str
    tier: str
    direct_cost_usd: float


def _database_path(root: Path | str) -> Path:
    return Path(root).expanduser().resolve() / "optimization" / "routing-outcomes.sqlite3"


def _connect(root: Path | str) -> sqlite3.Connection:
    path = _database_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS route_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            observed_at REAL NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            phase TEXT NOT NULL,
            success INTEGER NOT NULL,
            cost_usd REAL NOT NULL
        )
        """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_route_outcome_bucket ON route_outcomes(model, phase)")
    connection.commit()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return connection


def canonical_model(model: str) -> str:
    normalized = model.strip()
    prefix, separator, remainder = normalized.partition("/")
    if separator and prefix in {"anthropic", "google", "openai"}:
        return remainder
    return normalized


def provider_for_model(model: str) -> str:
    lowered = model.lower()
    if "/" in lowered:
        prefix = lowered.split("/", 1)[0]
        if prefix in {
            "anthropic",
            "azure",
            "bedrock",
            "google",
            "groq",
            "mistral",
            "ollama",
            "openai",
            "openrouter",
            "together_ai",
            "vertex_ai",
        }:
            return prefix
    if "claude" in lowered:
        return "anthropic"
    if "gemini" in lowered:
        return "google"
    if any(marker in lowered for marker in ("gpt", "codex", "o1", "o3", "o4")):
        return "openai"
    return "unknown"


def record_route_outcome(
    root: Path | str,
    *,
    provider: str,
    model: str,
    phase: str,
    success: bool,
    cost_usd: float,
    now: datetime | None = None,
) -> None:
    try:
        with _connect(root) as connection:
            connection.execute(
                """
                INSERT INTO route_outcomes(observed_at, provider, model, phase, success, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (now or datetime.now(UTC)).timestamp(),
                    provider[:64],
                    canonical_model(model)[:160],
                    phase[:32],
                    int(bool(success)),
                    max(0.0, float(cost_usd)),
                ),
            )
    except sqlite3.Error:
        return


def _outcome_stats(root: Path | str, model: str, phase: str) -> tuple[int, int, float]:
    try:
        with _connect(root) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(success), 0), COALESCE(AVG(cost_usd), 0)
                FROM route_outcomes WHERE model = ? AND phase = ?
                """,
                (canonical_model(model), phase),
            ).fetchone()
        if row is None:
            return 0, 0, 0.0
        return int(row[0]), int(row[1]), float(row[2])
    except sqlite3.Error:
        return 0, 0, 0.0


def _prior_failure(tier: str, phase: str) -> float:
    base = 0.10 if tier == "high" else 0.24
    if phase == "execute":
        base += 0.05
    elif phase == "repair":
        base += 0.18
    elif phase == "finish":
        base -= 0.03
    return min(0.75, max(0.02, base))


def _failure_probability(samples: int, successes: int, *, tier: str, phase: str) -> float:
    prior = _prior_failure(tier, phase)
    prior_failures = prior * _PRIOR_STRENGTH
    failures = max(0, samples - successes)
    return (failures + prior_failures) / (samples + _PRIOR_STRENGTH)


def _minimum_samples() -> int:
    try:
        return max(1, int(os.environ.get("LEMONCROW_ROUTING_MIN_SAMPLES", str(_DEFAULT_MIN_SAMPLES))))
    except ValueError:
        return _DEFAULT_MIN_SAMPLES


def _hysteresis_fraction() -> float:
    try:
        value = float(os.environ.get("LEMONCROW_ROUTING_HYSTERESIS_PCT", "10")) / 100.0
    except ValueError:
        value = 0.10
    return min(0.50, max(0.0, value))


def _candidates(route_decision: Any) -> list[_Candidate]:
    alternatives = list(getattr(route_decision, "alternatives", ()) or ())
    selected_provider = str(getattr(route_decision, "provider", ""))
    selected_model = str(getattr(route_decision, "model", ""))
    selected_tier = str(getattr(route_decision, "tier", "high"))
    rows: list[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    for item in alternatives:
        provider = str(getattr(item, "provider", ""))
        model = str(getattr(item, "model", ""))
        if not provider or not model or (provider, canonical_model(model)) in seen:
            continue
        seen.add((provider, canonical_model(model)))
        rows.append(
            _Candidate(
                provider=provider,
                model=model,
                tier=str(getattr(item, "tier", "high")),
                direct_cost_usd=max(0.0, float(getattr(item, "estimated_cost_usd", 0.0) or 0.0)),
            )
        )
    selected_key = (selected_provider, canonical_model(selected_model))
    if selected_provider and selected_model and selected_key not in seen:
        projected = getattr(route_decision, "projected_session_cost_usd", None)
        fallback_cost = min((row.direct_cost_usd for row in rows), default=0.001)
        rows.append(
            _Candidate(
                provider=selected_provider,
                model=selected_model,
                tier=selected_tier,
                direct_cost_usd=max(0.0, float(projected if projected is not None else fallback_cost)),
            )
        )
    return rows


def estimate_cache_break_cost(model: str, context_tokens: int) -> float:
    if context_tokens <= 0:
        return 0.0
    try:
        from lemoncrow.core.capabilities.pricing import get_model_pricing

        pricing = get_model_pricing(model)
        return max(0.0, float(pricing.input) * context_tokens / 1_000_000 * 0.90)
    except Exception:
        return 0.0


def choose_calibrated_route(
    root: Path | str,
    *,
    route_decision: Any,
    phase: str,
    current_model: str,
    context_tokens: int,
    failure_count: int = 0,
) -> CalibratedRoute:
    candidates = _candidates(route_decision)
    if not candidates:
        provider = str(getattr(route_decision, "provider", ""))
        model = str(getattr(route_decision, "model", ""))
        return CalibratedRoute(
            provider=provider,
            model=model,
            tier=str(getattr(route_decision, "tier", "high")),
            direct_cost_usd=0.0,
            failure_probability=1.0,
            escalation_cost_usd=0.0,
            cache_break_cost_usd=0.0,
            expected_total_cost_usd=0.0,
            sample_count=0,
            eligible=False,
            reason="no comparable route alternatives",
        )

    high = [candidate for candidate in candidates if candidate.tier == "high"]
    repair = phase == "repair" or failure_count > 0
    considered = high if repair and high else candidates
    escalation_cost = min(
        (candidate.direct_cost_usd for candidate in high),
        default=max(candidate.direct_cost_usd for candidate in candidates),
    )
    scored: list[tuple[float, _Candidate, float, int, float, float]] = []
    for candidate in considered:
        samples, successes, observed_cost = _outcome_stats(root, candidate.model, phase)
        failure_probability = _failure_probability(
            samples,
            successes,
            tier=candidate.tier,
            phase=phase,
        )
        direct_cost = (
            observed_cost if samples >= _minimum_samples() and observed_cost > 0 else candidate.direct_cost_usd
        )
        cache_break = (
            0.0
            if not current_model or canonical_model(current_model) == canonical_model(candidate.model)
            else estimate_cache_break_cost(candidate.model, context_tokens)
        )
        expected = direct_cost + failure_probability * escalation_cost + cache_break
        scored.append((expected, candidate, failure_probability, samples, direct_cost, cache_break))

    selected_row = min(
        scored,
        key=lambda item: (item[0], item[1].provider, item[1].model),
    )
    reason = (
        "repair safety gate requires a high-tier route"
        if repair and high
        else "calibrated direct + failure*escalation + cache-break expected cost"
    )
    if current_model and not repair:
        current_row = next(
            (item for item in scored if canonical_model(item[1].model) == canonical_model(current_model)),
            None,
        )
        if current_row is not None and current_row is not selected_row:
            improvement = current_row[0] - selected_row[0]
            required = max(0.0, current_row[0]) * _hysteresis_fraction()
            if improvement <= required:
                selected_row = current_row
                reason = "warm cache lane retained because expected savings did not clear route hysteresis"

    expected, selected, probability, samples, direct_cost, cache_break = selected_row
    eligible = repair or samples >= _minimum_samples()
    return CalibratedRoute(
        provider=selected.provider,
        model=selected.model,
        tier=selected.tier,
        direct_cost_usd=direct_cost,
        failure_probability=probability,
        escalation_cost_usd=escalation_cost,
        cache_break_cost_usd=cache_break,
        expected_total_cost_usd=expected,
        sample_count=samples,
        eligible=eligible,
        reason=reason,
    )


def summarize_route_outcomes(root: Path | str) -> dict[str, Any]:
    try:
        with _connect(root) as connection:
            total, successes, cost = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(success), 0), COALESCE(SUM(cost_usd), 0) FROM route_outcomes"
            ).fetchone()
            buckets = connection.execute("""
                SELECT model, phase, COUNT(*), COALESCE(SUM(success), 0), COALESCE(AVG(cost_usd), 0)
                FROM route_outcomes GROUP BY model, phase ORDER BY COUNT(*) DESC, model, phase LIMIT 50
                """).fetchall()
    except (sqlite3.Error, TypeError):
        return {"outcomes": 0, "success_rate": None, "cost_usd": 0.0, "buckets": []}
    return {
        "outcomes": int(total),
        "success_rate": round(int(successes) / int(total), 4) if int(total) else None,
        "cost_usd": round(float(cost), 8),
        "buckets": [
            {
                "model": str(model),
                "phase": str(phase),
                "samples": int(samples),
                "success_rate": round(int(passed) / int(samples), 4) if int(samples) else None,
                "average_cost_usd": round(float(average), 8),
            }
            for model, phase, samples, passed, average in buckets
        ],
    }


__all__ = [
    "CalibratedRoute",
    "canonical_model",
    "choose_calibrated_route",
    "estimate_cache_break_cost",
    "provider_for_model",
    "record_route_outcome",
    "summarize_route_outcomes",
]
