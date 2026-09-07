"""Optimization audit helpers inspired by external token-audit tooling.

These helpers stay native to LemonCrow's own telemetry and knowledge layout.
They surface two missing views in the current product:

- a static audit of prompt-relevant context surfaces
- a multi-signal quality score over recent traces
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.models import Trace
from lemoncrow.pro.capabilities.session_optimizer import effective_input_tokens

_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_HOST_CONFIG_DIR = _PACKAGE_ROOT / "gateway" / "hosts" / "configs"
_BUNDLED_BLOCKS_DIR = _PACKAGE_ROOT / "infra" / "seed_playbooks"
_BUNDLED_RUBRICS_DIR = _PACKAGE_ROOT / "core" / "rubrics"

_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude": 1_000_000,
    "gpt-5": 400_000,
    "gpt-4.1": 400_000,
    "gemini": 2_000_000,
    "o3": 200_000,
    "o4": 200_000,
    "deepseek": 128_000,
    "qwen": 128_000,
    "mistral": 128_000,
    "grok": 131_000,
    "local": 128_000,
}


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def _read_file_tokens(path: Path) -> int:
    try:
        if not path.exists() or not path.is_file() or path.stat().st_size > 1_000_000:
            return 0
        return _estimate_tokens(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return 0


def _collect_paths(root: Path | None, patterns: Iterable[str]) -> list[Path]:
    if root is None or not root.exists():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        try:
            paths.extend(root.rglob(pattern))
        except OSError:
            # An unreadable subdirectory must not crash the advisory audit.
            continue
    return sorted({path for path in paths if path.is_file()})


def _knowledge_paths(primary_root: Path | None, fallback_root: Path, patterns: Iterable[str]) -> list[Path]:
    primary = _collect_paths(primary_root, patterns)
    return primary if primary else _collect_paths(fallback_root, patterns)


def _component(
    *,
    identifier: str,
    title: str,
    category: str,
    mode: str,
    optimizable: bool,
    paths: list[Path],
    notes: str,
) -> dict[str, Any] | None:
    tokens = sum(_read_file_tokens(path) for path in paths)
    if tokens <= 0:
        return None
    return {
        "id": identifier,
        "title": title,
        "category": category,
        "mode": mode,
        "estimated_tokens": int(tokens),
        "file_count": len(paths),
        "optimizable": optimizable,
        "notes": notes,
    }


def build_context_audit(
    *,
    project_root: Path,
    blocks_dir: Path | None = None,
    rubrics_dir: Path | None = None,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []

    repo_guidance = _component(
        identifier="repo_guidance",
        title="Repo guidance",
        category="always_on",
        mode="always_on",
        optimizable=True,
        paths=[path for path in [project_root / "AGENTS.md"] if path.exists()],
        notes="Repository-wide instructions that can become startup overhead when they grow without pruning.",
    )
    if repo_guidance:
        components.append(repo_guidance)

    host_templates = _component(
        identifier="host_instruction_templates",
        title="Host instruction templates",
        category="host_templates",
        mode="host_template",
        optimizable=True,
        paths=_collect_paths(_HOST_CONFIG_DIR, ("*.yaml", "*.yml")),
        notes="Packaged host templates. Only one is active per session, so this is an aggregate maintenance surface rather than cumulative runtime overhead.",
    )
    if host_templates:
        components.append(host_templates)

    reasoning_procedures = _component(
        identifier="reasoning_procedures",
        title="Reasoning procedures",
        category="retrieval",
        mode="retrieval_backed",
        optimizable=True,
        paths=_knowledge_paths(blocks_dir, _BUNDLED_BLOCKS_DIR, ("*.md", "*.yaml", "*.yml")),
        notes="Playbooks and seed procedures are selectively recalled. Keep triggers narrow so retrieval stays targeted.",
    )
    if reasoning_procedures:
        components.append(reasoning_procedures)

    quality_rubrics = _component(
        identifier="quality_rubrics",
        title="Quality rubrics",
        category="quality",
        mode="retrieval_backed",
        optimizable=True,
        paths=_knowledge_paths(rubrics_dir, _BUNDLED_RUBRICS_DIR, ("*.yaml", "*.yml")),
        notes="Rubrics are valuable when scoped tightly. Broad or duplicated checks add maintenance cost and can inflate guardrail prompts.",
    )
    if quality_rubrics:
        components.append(quality_rubrics)

    components.sort(key=lambda item: int(item["estimated_tokens"]), reverse=True)

    recommendations: list[str] = []
    repo_tokens = int(repo_guidance["estimated_tokens"]) if repo_guidance else 0
    host_tokens = int(host_templates["estimated_tokens"]) if host_templates else 0
    procedure_tokens = int(reasoning_procedures["estimated_tokens"]) if reasoning_procedures else 0
    rubric_tokens = int(quality_rubrics["estimated_tokens"]) if quality_rubrics else 0

    if repo_tokens > 5_000:
        recommendations.append(
            "AGENTS.md is large enough to tax every session. Split durable rules from optional reference material."
        )
    if host_tokens > 18_000:
        recommendations.append(
            "Host templates are heavy in aggregate. Keep shared optimizer guidance centralized to avoid drift across hosts."
        )
    if procedure_tokens > 16_000:
        recommendations.append(
            "Reasoning procedures are substantial. Tighten triggers and retire stale procedures so retrieval stays selective."
        )
    if rubric_tokens > 10_000:
        recommendations.append(
            "Rubrics are growing dense. Prefer domain-scoped gates over broad global checks when the same rule repeats."
        )
    if not recommendations:
        recommendations.append(
            "No major static context hotspots were detected in the currently audited LemonCrow surfaces."
        )

    audited_total = sum(int(item["estimated_tokens"]) for item in components)
    always_on_total = sum(int(item["estimated_tokens"]) for item in components if item["mode"] == "always_on")
    optimizable_total = sum(int(item["estimated_tokens"]) for item in components if item["optimizable"])

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "audited_tokens_total": audited_total,
        "always_on_tokens": always_on_total,
        "optimizable_tokens": optimizable_total,
        "component_count": len(components),
        "components": components,
        "recommendations": recommendations,
    }


def _normalize_model(model: str | None) -> str:
    return (model or "").strip().lower()


def context_window_for_model(model: str | None) -> int:
    normalized = _normalize_model(model)
    for prefix, window in _MODEL_CONTEXT_WINDOWS.items():
        if prefix in normalized:
            return window
    return 200_000


def _created_after(trace: Trace, cutoff: datetime) -> bool:
    created = trace.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created >= cutoff


def _trace_has_delivery_signal(trace: Trace) -> bool:
    if trace.files_touched:
        return True
    if any(bool(getattr(result, "passed", False)) for result in trace.validation_results):
        return True
    tool_names = {tool.name.lower() for tool in trace.tools_called}
    return bool(tool_names & {"edit", "write", "multiedit", "apply_patch", "mcp__lc__edit"})


def _trace_signature(trace: Trace) -> str:
    task = re.sub(r"\d+", "#", (trace.task or "").lower())
    task = " ".join(task.split()[:12])
    tools = ",".join(sorted(tool.name.lower() for tool in trace.tools_called)[:5])
    return "|".join(
        [
            (trace.host or trace.agent or "unknown").lower(),
            (trace.domain or "unknown").lower(),
            task,
            tools,
        ]
    )


def _grade(score: int) -> str:
    if score >= 90:
        return "S"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _build_signal(identifier: str, title: str, weight_pct: int, score: int, detail: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "title": title,
        "weight_pct": weight_pct,
        "score": max(0, min(100, int(score))),
        "detail": detail,
    }


def build_session_quality_summary(
    traces: Iterable[Trace],
    *,
    window_days: int,
    context_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cutoff = datetime.now(UTC) - timedelta(days=max(1, window_days))
    recent = [trace for trace in traces if _created_after(trace, cutoff)]

    if not recent:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "trace_count": 0,
            "score": 0,
            "grade": "N/A",
            "dominant_model": None,
            "dominant_context_window_tokens": 0,
            "signals": [],
            "recommendations": ["No recent traces were available for quality scoring."],
            "risk_flags": [],
        }

    model_counts = Counter(_normalize_model(trace.model or trace.host or trace.agent) for trace in recent)
    dominant_model = model_counts.most_common(1)[0][0] if model_counts else ""
    dominant_window = context_window_for_model(dominant_model)
    always_on_tokens = int((context_audit or {}).get("always_on_tokens", 0) or 0)

    fills: list[float] = []
    cache_ratios: list[float] = []
    message_efficiencies: list[float] = []
    delivery_count = 0
    failure_weight = 0.0
    duplicate_signatures = 0
    signatures: dict[str, int] = {}

    high_fill_count = 0
    expensive_low_delivery = 0
    failed_count = 0

    for trace in recent:
        window = context_window_for_model(trace.model or trace.host or trace.agent)
        effective_input = effective_input_tokens(trace)
        fill = (effective_input + always_on_tokens) / max(window, 1)
        fills.append(fill)
        if fill >= 0.6:
            high_fill_count += 1

        total_dynamic = effective_input + int(trace.output_tokens or 0)
        if total_dynamic > 0:
            message_efficiencies.append(int(trace.output_tokens or 0) / total_dynamic)

        cache_tokens = int(trace.cached_input_tokens or 0)
        if effective_input > 0:
            cache_ratios.append(cache_tokens / effective_input)

        if _trace_has_delivery_signal(trace):
            delivery_count += 1
        elif effective_input > 100_000:
            expensive_low_delivery += 1

        if trace.status == "success":
            failure_weight += 1.0
        elif trace.status == "partial":
            failure_weight += 0.6
        else:
            failed_count += 1

        signature = _trace_signature(trace)
        signatures[signature] = signatures.get(signature, 0) + 1

    duplicate_signatures = sum(count - 1 for count in signatures.values() if count > 1)

    avg_fill = sum(fills) / len(fills)
    fill_score = 10
    if avg_fill < 0.2:
        fill_score = 100
    elif avg_fill < 0.35:
        fill_score = 85
    elif avg_fill < 0.5:
        fill_score = 70
    elif avg_fill < 0.65:
        fill_score = 50
    elif avg_fill < 0.8:
        fill_score = 30

    delivery_ratio = delivery_count / len(recent)
    delivery_score = round(delivery_ratio * 100)

    outcome_score = round((failure_weight / len(recent)) * 100)

    if any(ratio > 0 for ratio in cache_ratios):
        avg_cache_ratio = sum(cache_ratios) / len(cache_ratios)
        if avg_cache_ratio >= 0.3:
            cache_score = 95
        elif avg_cache_ratio >= 0.15:
            cache_score = 80
        elif avg_cache_ratio >= 0.05:
            cache_score = 65
        else:
            cache_score = 45
        cache_detail = f"Average cache reuse: {avg_cache_ratio * 100:.1f}% of effective input tokens."
    else:
        avg_cache_ratio = 0.0
        cache_score = 60
        cache_detail = "No cache reuse was observed in this window, so cache leverage is treated as neutral."

    avg_efficiency = sum(message_efficiencies) / len(message_efficiencies) if message_efficiencies else 0.0
    if avg_efficiency >= 0.12:
        message_score = 90
    elif avg_efficiency >= 0.06:
        message_score = 75
    elif avg_efficiency >= 0.03:
        message_score = 60
    elif avg_efficiency >= 0.015:
        message_score = 40
    else:
        message_score = 20

    duplicate_ratio = duplicate_signatures / len(recent)
    if duplicate_ratio < 0.05:
        compression_score = 95
    elif duplicate_ratio < 0.12:
        compression_score = 80
    elif duplicate_ratio < 0.2:
        compression_score = 65
    elif duplicate_ratio < 0.35:
        compression_score = 45
    else:
        compression_score = 25

    signals = [
        _build_signal(
            "context_fill",
            "Context fill",
            25,
            fill_score,
            f"Average effective fill was {avg_fill * 100:.1f}% of the active model window. Dominant window: {dominant_window:,} tokens.",
        ),
        _build_signal(
            "delivery_signal",
            "Delivery signal",
            20,
            delivery_score,
            f"{delivery_count} of {len(recent)} traces ended with an edit or passing validation signal.",
        ),
        _build_signal(
            "outcome_health",
            "Outcome health",
            20,
            outcome_score,
            f"Success-weighted completion score across recent traces. Failed traces: {failed_count}.",
        ),
        _build_signal("cache_leverage", "Cache leverage", 15, cache_score, cache_detail),
        _build_signal(
            "message_efficiency",
            "Message efficiency",
            10,
            message_score,
            f"Average output share was {avg_efficiency * 100:.1f}% of total trace tokens.",
        ),
        _build_signal(
            "compression_opportunity",
            "Compression opportunity",
            10,
            compression_score,
            f"{duplicate_signatures} repeated work-shape traces were detected across {len(recent)} recent traces.",
        ),
    ]

    score = round(sum(signal["score"] * signal["weight_pct"] for signal in signals) / 100)

    risk_flags: list[str] = []
    if high_fill_count > 0:
        risk_flags.append(f"{high_fill_count} traces exceeded 60% of their model window.")
    if expensive_low_delivery > 0:
        risk_flags.append(f"{expensive_low_delivery} high-input traces had no edit or validation signal.")
    if duplicate_signatures > 0:
        risk_flags.append(f"{duplicate_signatures} traces repeated a similar work shape and may be compressible.")
    if failed_count > 0:
        risk_flags.append(f"{failed_count} traces ended in failure inside the current window.")

    recommendations: list[str] = []
    for signal in sorted(signals, key=lambda item: int(item["score"])):
        if signal["score"] >= 70:
            continue
        if signal["id"] == "context_fill":
            recommendations.append(
                "Trim startup guidance and compact earlier when sessions start spending more than half of the active context window."
            )
        elif signal["id"] == "delivery_signal":
            recommendations.append(
                "Stop long exploratory loops earlier. Name the expected deliverable before further searching if there is no edit or validation within the slice."
            )
        elif signal["id"] == "outcome_health":
            recommendations.append(
                "Inspect failed and partial traces first. Repeated failures should route into rescue or a narrower plan instead of another retry."
            )
        elif signal["id"] == "cache_leverage":
            recommendations.append(
                "Keep stable prompt prefixes and reuse narrower reads so provider caching and LemonCrow cached-read paths can fire more often."
            )
        elif signal["id"] == "message_efficiency":
            recommendations.append(
                "Ask for tighter, structured responses on routine work so the model stops spending tokens on narrative that does not advance the patch."
            )
        elif signal["id"] == "compression_opportunity":
            recommendations.append(
                "Repeated work shapes suggest reread or rediscovery churn. Favor targeted diffs, cached state, and reusable procedures over broad restarts."
            )

    if not recommendations:
        recommendations.append("Recent traces look healthy. No immediate quality intervention is indicated.")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "trace_count": len(recent),
        "score": score,
        "grade": _grade(score),
        "dominant_model": dominant_model or None,
        "dominant_context_window_tokens": dominant_window,
        "signals": signals,
        "recommendations": recommendations,
        "risk_flags": risk_flags,
    }


__all__ = ["build_context_audit", "build_session_quality_summary", "context_window_for_model"]
