"""Coordinator and child-run logic for the LemonCrow swarm harness."""

from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from lemoncrow.core.capabilities.host_runners import (
    list_runner_profiles as _list_runner_profiles,
)
from lemoncrow.core.capabilities.host_runners import (
    resolve_runner_metadata as _resolve_runner_metadata,
)
from lemoncrow.core.capabilities.host_runners import (
    resolve_swarm_runner_command as _resolve_swarm_runner_command,
)
from lemoncrow.infra.runtime.run_ledger import RunLedger
from lemoncrow.infra.storage.factory import create_store
from lemoncrow.pro.capabilities.swarm.fitness import FitnessSpec
from lemoncrow.pro.capabilities.swarm.models import (
    Finding,
    SwarmAcceptedCommit,
    SwarmArtifactRef,
    SwarmChildState,
    SwarmConvergenceVerdict,
    SwarmEvaluatorBackend,
    SwarmExecMode,
    SwarmPlanningMode,
    SwarmRunState,
    SwarmValidationCheck,
    SwarmWaveDecision,
    SwarmWaveEvaluation,
    SwarmWaveState,
)
from lemoncrow.pro.capabilities.swarm.reducers import WaveContext, get_reducer
from lemoncrow.pro.capabilities.swarm.reducers.best import (
    _has_non_structural_passing_validation,
    _score_child,
)
from lemoncrow.pro.capabilities.swarm.reducers.best import (
    rank_children as rank_children,  # re-exported via swarm/__init__.py
)
from lemoncrow.pro.runtime.swarm_worktree import (
    SwarmWorktreeManager,
    collect_dirty_paths,
    git_repo_root,
    read_head_ref,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


_PROVIDER_WORKER_STEP_BUDGETS: dict[str, int] = {
    "low": 6,
    "medium": 10,
    "high": 14,
}
_PROVIDER_WORKER_TIME_BUDGETS: dict[str, int] = {
    "low": 120,
    "medium": 240,
    "high": 420,
}
_PROVIDER_WORKER_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {
            "type": "string",
            "enum": ["context", "search", "read", "edit", "finish"],
        },
        "task": {"type": "string"},
        "files": {"type": "array", "items": {"type": "string"}},
        "query": {"type": "string"},
        "path": {"type": "string"},
        "range": {"type": "string"},
        "full": {"type": "boolean"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace": {"type": "boolean"},
                },
                "required": ["file_path", "new_string"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["action"],
}
_SWARM_EVALUATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "verdict": {
            "type": "string",
            "enum": ["continue", "converged", "stagnating", "blocked"],
        },
        "candidate_order": {"type": "array", "items": {"type": "string"}},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "child_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["accept", "reject", "defer"],
                    },
                    "rationale": {"type": "string"},
                    "conflicts_with": {"type": "array", "items": {"type": "string"}},
                    "duplicates": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["child_id", "verdict", "rationale"],
            },
        },
        "next_wave_directives": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "verdict", "candidate_order", "decisions", "next_wave_directives"],
}
_SWARM_APPROACH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "approaches": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One distinct implementation approach per child run, in child order.",
        }
    },
    "required": ["approaches"],
}


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
            tmp_path = handle.name
        Path(tmp_path).replace(path)
    except OSError:
        if tmp_path:
            with contextlib.suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)
        raise


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def load_swarm_state(path: Path) -> SwarmRunState:
    return SwarmRunState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_swarm_state(path: Path, state: SwarmRunState) -> None:
    state.updated_at = _utcnow()
    _write_json(path, state.model_dump(mode="json"))


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"Invalid swarm run_id: {run_id!r}")
    return run_id


def swarm_run_dir(root: Path, run_id: str) -> Path:
    return Path(root).resolve() / "swarm" / "runs" / _validate_run_id(run_id)


def resolve_state_path(root: Path, run_id: str) -> Path:
    return swarm_run_dir(root, run_id) / "state.json"


def _run_artifact_root(state: SwarmRunState) -> Path:
    if state.artifact_root:
        return Path(state.artifact_root)
    return Path(state.copied_spec_path).resolve().parent / "artifacts"


def _artifact_relative_path(run_dir: Path, path: Path) -> str:
    with contextlib.suppress(ValueError):
        return str(path.relative_to(run_dir))
    return str(path)


def _artifact_ref(
    run_dir: Path,
    path: Path,
    *,
    kind: str,
    label: str,
    metadata: dict[str, object] | None = None,
) -> SwarmArtifactRef:
    mime_type, _ = mimetypes.guess_type(path.name)
    exists = path.exists()
    return SwarmArtifactRef(
        artifact_id=f"{kind}:{_artifact_relative_path(run_dir, path)}",
        kind=kind,
        label=label,
        path=str(path),
        relative_path=_artifact_relative_path(run_dir, path),
        mime_type=mime_type or ("application/json" if path.suffix == ".json" else "text/plain"),
        size_bytes=path.stat().st_size if exists else 0,
        exists=exists,
        metadata=metadata or {},
    )


def list_swarm_runner_profiles() -> list[dict[str, Any]]:
    return _list_runner_profiles()


def resolve_swarm_spec_path(
    *,
    project_root: Path,
    spec_path: Path | str | None,
) -> tuple[Path, Literal["explicit", "default"], bool]:
    resolved_root = Path(project_root).expanduser().resolve()
    if spec_path is None or not str(spec_path).strip():
        candidate = resolved_root / "program.md"
        if not candidate.exists():
            candidate = resolved_root / "PROGRAM.md"
        if not candidate.exists():
            raise RuntimeError(f"default swarm spec not found: {candidate}")
        if not candidate.is_file():
            raise RuntimeError(f"default swarm spec is not a file: {candidate}")
        return candidate, "default", True

    raw_path = Path(spec_path).expanduser()
    candidate = (resolved_root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise RuntimeError(f"swarm spec must stay under the selected project root: {resolved_root}")
    if not candidate.exists():
        raise RuntimeError(f"swarm spec not found: {candidate}")
    if not candidate.is_file():
        raise RuntimeError(f"swarm spec is not a file: {candidate}")
    return candidate, "explicit", candidate.name in {"program.md", "PROGRAM.md"}


def resolve_swarm_child_command(
    *,
    runner: str | None,
    runner_model: str | None,
    runner_args: list[str] | tuple[str, ...],
    child_command: list[str] | tuple[str, ...],
    prompt_template: str,
) -> list[str]:
    return _resolve_swarm_runner_command(
        runner=runner,
        runner_model=runner_model,
        runner_args=runner_args,
        child_command=child_command,
        prompt_template=prompt_template,
    )


def resolve_swarm_runner_metadata(
    *,
    runner: str | None,
    runner_model: str | None,
    child_command: list[str] | tuple[str, ...],
) -> tuple[str, str]:
    return _resolve_runner_metadata(runner=runner, runner_model=runner_model, child_command=child_command)


def resolve_swarm_provider_command(provider: Literal["openai", "litellm"]) -> list[str]:
    if provider not in {"openai", "litellm"}:
        raise ValueError(f"unsupported provider-backed swarm worker: {provider}")
    return [
        *_python_cli_invocation(),
        "--root",
        "{lemoncrow_root}",
        "swarm",
        "_provider-worker",
    ]


def build_swarm_spec_payload(state: SwarmRunState) -> dict[str, Any]:
    content = _spec_text(state)
    excerpt = content[:4000]
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = lines[0] if lines else Path(state.copied_spec_path).name
    return {
        "source_path": state.spec_source_path or state.copied_spec_path,
        "copied_path": state.copied_spec_path,
        "resolution": state.spec_resolution,
        "used_program_md": state.used_program_md,
        "job_kind": state.job_kind,
        "reducer": state.reducer_name,
        "exec_mode": state.exec_mode,
        "search_space": list(state.search_space),
        "quorum": state.quorum,
        "fitness": (state.fitness_spec.model_dump(mode="json") if state.fitness_spec else None),
        "title": title[:160],
        "excerpt": excerpt,
        "truncated": len(content) > len(excerpt),
        "content": content,
    }


def _upsert_run_artifact(state: SwarmRunState, artifact: SwarmArtifactRef) -> None:
    for index, existing in enumerate(state.export_artifacts):
        if existing.path == artifact.path:
            state.export_artifacts[index] = artifact
            return
    state.export_artifacts.append(artifact)


def _spec_text(state: SwarmRunState) -> str:
    spec_path = Path(state.copied_spec_path)
    if not spec_path.exists():
        return ""
    return spec_path.read_text(encoding="utf-8", errors="replace")


def _trim_text(text: str, *, limit: int) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}\n...[truncated]..."


def _current_internal_llm_backend() -> str:
    return os.environ.get("LEMONCROW_LLM_BACKEND", "none").strip().lower()


def _coerce_decision_verdict(value: object) -> Literal["accept", "reject", "defer"]:
    verdict = str(value or "defer")
    if verdict in {"accept", "reject", "defer"}:
        return cast(Literal["accept", "reject", "defer"], verdict)
    return "defer"


def _coerce_convergence_verdict(value: object) -> SwarmConvergenceVerdict:
    verdict = str(value or "continue")
    if verdict in {"continue", "converged", "stagnating", "blocked"}:
        return cast(SwarmConvergenceVerdict, verdict)
    return "continue"


def _resolve_evaluator_backend(state: SwarmRunState) -> SwarmEvaluatorBackend | None:
    backend = state.evaluator_backend
    if backend == "disabled":
        return None
    if backend != "auto":
        return backend
    active = _current_internal_llm_backend()
    if active in {"ollama", "openai", "litellm"}:
        return cast(SwarmEvaluatorBackend, active)
    if state.launch_provider in {"openai", "litellm"}:
        return cast(SwarmEvaluatorBackend, state.launch_provider)
    return None


def _patch_digest(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_preview(path: Path, *, limit: int = 2400) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [
        line for line in text.splitlines() if not line.startswith(("index ", "new file mode ", "deleted file mode "))
    ]
    return _trim_text("\n".join(lines), limit=limit)


def _normalize_changed_file(entry: str) -> str:
    path_text = entry[3:] if len(entry) > 3 else entry
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[1]
    return path_text.strip()


def _changed_file_set(entries: list[str]) -> set[str]:
    return {_normalize_changed_file(path) for path in entries if _normalize_changed_file(path)}


def _validation_summary(validation_results: list[SwarmValidationCheck]) -> list[dict[str, object]]:
    return [
        {
            "name": item.name,
            "passed": item.passed,
            "detail": _trim_text(item.detail, limit=180),
        }
        for item in validation_results[:8]
    ]


def _build_relation_maps(
    children: list[SwarmChildState],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    duplicates: dict[str, set[str]] = {child.child_id: set() for child in children}
    conflicts: dict[str, set[str]] = {child.child_id: set() for child in children}
    for hint in _build_relation_hints(children):
        child_ids = hint.get("child_ids")
        if not isinstance(child_ids, list) or len(child_ids) != 2:
            continue
        left = str(child_ids[0])
        right = str(child_ids[1])
        if left not in duplicates or right not in duplicates:
            continue
        relation = str(hint.get("relation") or "")
        if relation == "duplicate":
            duplicates[left].add(right)
            duplicates[right].add(left)
        elif relation == "possible_conflict":
            conflicts[left].add(right)
            conflicts[right].add(left)
    return duplicates, conflicts


def _accepted_patch_digest_map(accepted_commits: list[SwarmAcceptedCommit]) -> dict[str, list[str]]:
    digests: dict[str, list[str]] = {}
    for accepted in accepted_commits:
        digest = _patch_digest(Path(accepted.patch_path))
        if not digest:
            continue
        digests.setdefault(digest, []).append(accepted.child_id)
    return digests


def _recent_accepted_file_sets(
    accepted_commits: list[SwarmAcceptedCommit],
    *,
    limit: int = 2,
) -> list[tuple[str, set[str]]]:
    recent: list[tuple[str, set[str]]] = []
    for accepted in accepted_commits[-limit:]:
        files = _changed_file_set(accepted.files_changed)
        if files:
            recent.append((accepted.child_id, files))
    return recent


def _build_relation_hints(children: list[SwarmChildState]) -> list[dict[str, object]]:
    hints: list[dict[str, object]] = []
    patch_digests: dict[str, str] = {}
    file_sets: dict[str, set[str]] = {}
    for child in children:
        patch_digests[child.child_id] = _patch_digest(Path(child.patch_path))
        file_sets[child.child_id] = _changed_file_set(child.files_changed)
    for index, left in enumerate(children):
        for right in children[index + 1 :]:
            left_digest = patch_digests[left.child_id]
            right_digest = patch_digests[right.child_id]
            if left_digest and left_digest == right_digest:
                hints.append(
                    {
                        "child_ids": [left.child_id, right.child_id],
                        "relation": "duplicate",
                        "reason": "Patch content is identical.",
                    }
                )
                continue
            overlap = sorted(file_sets[left.child_id] & file_sets[right.child_id])
            if overlap:
                hints.append(
                    {
                        "child_ids": [left.child_id, right.child_id],
                        "relation": "possible_conflict",
                        "reason": f"Both candidates modify overlapping files: {', '.join(overlap[:8])}",
                    }
                )
            else:
                hints.append(
                    {
                        "child_ids": [left.child_id, right.child_id],
                        "relation": "likely_independent",
                        "reason": "Candidates modify disjoint files.",
                    }
                )
    return hints[:24]


def _build_wave_evidence_payload(
    state: SwarmRunState,
    wave: SwarmWaveState,
    children: list[SwarmChildState],
) -> dict[str, object]:
    accepted_history = [
        {
            "child_id": item.child_id,
            "summary": _trim_text(item.summary, limit=240),
            "files_changed": item.files_changed[:10],
            "commit_ref": item.commit_ref,
        }
        for item in state.accepted_commits[-8:]
    ]
    child_payload: list[dict[str, object]] = []
    for child in children:
        child_payload.append(
            {
                "child_id": child.child_id,
                "status": child.status,
                "summary": _trim_text(child.summary or child.error or "No summary emitted.", limit=320),
                "files_changed": [_normalize_changed_file(path) for path in child.files_changed[:12]],
                "validation": _validation_summary(child.validation_results),
                "score": child.score,
                "score_breakdown": child.score_breakdown[:6],
                "duration_seconds": round(child.duration_seconds, 3),
                "cost_usd": round(child.cost_usd, 6),
                "patch_preview": _patch_preview(Path(child.patch_path)),
            }
        )
    return {
        "run_id": state.run_id,
        "wave_index": wave.wave_index,
        "base_task_spec": _trim_text(_spec_text(state), limit=5000),
        "accepted_history": accepted_history,
        "previous_convergence_summary": _trim_text(state.convergence_summary, limit=320),
        "previous_directives": state.next_wave_directives[:8],
        "candidate_relation_hints": _build_relation_hints(children),
        "children": child_payload,
    }


def _fallback_wave_evaluation(
    state: SwarmRunState,
    children: list[SwarmChildState],
    *,
    error: str = "",
) -> SwarmWaveEvaluation:
    ranked = rank_children(children)
    duplicate_map, conflict_map = _build_relation_maps(ranked)
    accepted_history_digests = _accepted_patch_digest_map(state.accepted_commits)
    recent_history_files = _recent_accepted_file_sets(state.accepted_commits)
    accepted: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    decisions: list[SwarmWaveDecision] = []
    next_wave_directives: list[str] = []
    accepted_set: set[str] = set()
    for child in ranked:
        verdict: Literal["accept", "reject", "defer"]
        eligible = (
            child.status == "success"
            and bool(child.files_changed)
            and not any(not check.passed for check in child.validation_results)
        )
        child_files = _changed_file_set(child.files_changed)
        duplicate_ids = sorted(duplicate_map.get(child.child_id, set()) & accepted_set)
        conflict_ids = sorted(conflict_map.get(child.child_id, set()) & accepted_set)
        history_duplicates = sorted(accepted_history_digests.get(_patch_digest(Path(child.patch_path)), []))
        revisit_ids = sorted(
            accepted_child_id
            for accepted_child_id, accepted_files in recent_history_files
            if child_files and child_files == accepted_files
        )
        if eligible and not duplicate_ids and not conflict_ids and not history_duplicates:
            if revisit_ids and not _has_non_structural_passing_validation(child):
                deferred.append(child.child_id)
                verdict = "defer"
                rationale = (
                    "Fallback deferred this candidate because it revisits recently accepted "
                    f"file set(s) without new non-structural validation evidence: {', '.join(revisit_ids)}."
                )
                directive = (
                    "Pursue a distinct angle or add non-structural validation evidence before revisiting "
                    "recently accepted files."
                )
                if directive not in next_wave_directives:
                    next_wave_directives.append(directive)
            else:
                accepted.append(child.child_id)
                accepted_set.add(child.child_id)
                verdict = "accept"
                rationale = (
                    "Fallback accepted this candidate because it succeeded, changed files, and passed child validation."
                )
        elif history_duplicates:
            rejected.append(child.child_id)
            verdict = "reject"
            rationale = (
                "Fallback rejected this candidate because it matches already accepted "
                f"candidate(s): {', '.join(history_duplicates)}."
            )
        elif duplicate_ids:
            rejected.append(child.child_id)
            verdict = "reject"
            rationale = (
                "Fallback rejected this candidate because it duplicates already accepted "
                f"candidate(s): {', '.join(duplicate_ids)}."
            )
        elif conflict_ids:
            rejected.append(child.child_id)
            verdict = "reject"
            rationale = (
                "Fallback conservatively rejected this candidate because it overlaps with already accepted "
                f"candidate(s): {', '.join(conflict_ids)}."
            )
        else:
            rejected.append(child.child_id)
            verdict = "reject"
            rationale = (
                "Fallback rejected this candidate because it failed, made no changes, or failed child validation."
            )
        decisions.append(
            SwarmWaveDecision(
                child_id=child.child_id,
                verdict=verdict,
                rationale=rationale,
                conflicts_with=[*conflict_ids, *revisit_ids],
                duplicates=[*history_duplicates, *duplicate_ids],
            )
        )
    return SwarmWaveEvaluation(
        status="fallback",
        evaluator_backend=state.evaluator_backend,
        evaluator_model=state.evaluator_model,
        summary="Used deterministic overlap-aware fallback evaluation because semantic evaluation was unavailable.",
        verdict="continue" if accepted or deferred else "stagnating",
        candidate_order=[child.child_id for child in ranked],
        accepted_child_ids=accepted,
        rejected_child_ids=rejected,
        deferred_child_ids=deferred,
        decisions=decisions,
        next_wave_directives=next_wave_directives,
        error=error,
        finished_at=_utcnow(),
    )


def _evaluate_wave(
    state: SwarmRunState,
    wave: SwarmWaveState,
    children: list[SwarmChildState],
) -> SwarmWaveEvaluation:
    backend = _resolve_evaluator_backend(state)
    if backend is None:
        return _fallback_wave_evaluation(state, children, error="Evaluator backend disabled.")

    from lemoncrow.infra import internal_llm

    evidence = _build_wave_evidence_payload(state, wave, children)
    messages = [
        {
            "role": "system",
            "content": (
                "You are the swarm evaluator. Judge candidate changes semantically for integration into the shared base. "
                "Accept multiple candidates only when they are compatible and independently valuable. Reject weaker duplicates "
                "or conflicting alternatives. Use 'defer' for promising but incomplete ideas that should guide later waves. "
                "Hard gates like child validation and patch application are enforced separately, so focus on semantic value, "
                "independence, duplication, contradiction, and what the next wave should explore."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(evidence, ensure_ascii=False),
        },
    ]
    env_key = "LEMONCROW_LLM_BACKEND"
    previous_backend = os.environ.get(env_key)
    os.environ[env_key] = backend
    try:
        response = internal_llm.chat(
            messages,
            model=state.evaluator_model or None,
            json_schema=_SWARM_EVALUATION_SCHEMA,
        )
    except internal_llm.InternalLLMError as exc:
        return _fallback_wave_evaluation(state, children, error=str(exc))
    finally:
        if previous_backend is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_backend

    if not isinstance(response, dict):
        return _fallback_wave_evaluation(state, children, error="Evaluator returned a non-object response.")

    decisions_payload = response.get("decisions")
    candidate_order = response.get("candidate_order")
    next_wave_directives = response.get("next_wave_directives")
    if not isinstance(decisions_payload, list) or not isinstance(candidate_order, list):
        return _fallback_wave_evaluation(state, children, error="Evaluator returned an invalid decision structure.")

    decisions: list[SwarmWaveDecision] = []
    accepted: list[str] = []
    rejected: list[str] = []
    deferred: list[str] = []
    for item in decisions_payload:
        if not isinstance(item, dict) or "child_id" not in item:
            continue
        decision = SwarmWaveDecision(
            child_id=str(item["child_id"]),
            verdict=_coerce_decision_verdict(item.get("verdict")),
            rationale=_trim_text(str(item.get("rationale") or ""), limit=240),
            conflicts_with=[str(value) for value in item.get("conflicts_with") or []][:8],
            duplicates=[str(value) for value in item.get("duplicates") or []][:8],
        )
        decisions.append(decision)
        if decision.verdict == "accept":
            accepted.append(decision.child_id)
        elif decision.verdict == "reject":
            rejected.append(decision.child_id)
        else:
            deferred.append(decision.child_id)
    summary = _trim_text(str(response.get("summary") or ""), limit=400)
    verdict = _coerce_convergence_verdict(response.get("verdict"))
    directives = [
        _trim_text(str(item), limit=240)
        for item in (next_wave_directives if isinstance(next_wave_directives, list) else [])
        if str(item).strip()
    ][:8]
    return SwarmWaveEvaluation(
        status="completed",
        evaluator_backend=backend,
        evaluator_model=state.evaluator_model,
        summary=summary,
        verdict=verdict,
        candidate_order=[str(item) for item in candidate_order][: len(children)],
        accepted_child_ids=accepted,
        rejected_child_ids=rejected,
        deferred_child_ids=deferred,
        decisions=decisions,
        next_wave_directives=directives,
        finished_at=_utcnow(),
    )


def _plan_wave_runs(state: SwarmRunState, wave_index: int) -> tuple[SwarmPlanningMode, int, str]:
    max_runs = max(state.max_runs or state.runs or 1, 1)
    if max_runs == 1:
        return "bounded", 1, "max_runs is 1, so the coordinator launches a single child."

    spec_text = _spec_text(state)
    lowered = spec_text.lower()
    file_hints = {
        match
        for match in re.findall(
            r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|sql|sh|yaml|yml|toml|css|scss)",
            spec_text,
        )
    }
    bullet_count = sum(1 for line in spec_text.splitlines() if re.match(r"\s*(?:[-*]|\d+\.)\s+", line))
    open_signals = [
        signal
        for signal in (
            "end-to-end",
            "dashboard",
            "frontend",
            "backend",
            "api",
            "ux",
            "open-ended",
            "explore",
            "across",
            "multiple",
            "service endpoints",
            "integration/export",
        )
        if signal in lowered
    ]
    bounded_signals = [
        signal
        for signal in (
            "rename",
            "one file",
            "single file",
            "small fix",
            "typo",
            "bounded",
            "narrow",
        )
        if signal in lowered
    ]
    previous_wave_accepts = len(state.waves[-1].accepted_child_ids) if state.waves else 0
    is_open_ended = bool(open_signals) or len(file_hints) >= 3 or bullet_count >= 4
    if wave_index > 1 and previous_wave_accepts > 1:
        is_open_ended = True
    if bounded_signals and not open_signals and len(file_hints) <= 2 and bullet_count <= 3:
        is_open_ended = False

    if is_open_ended:
        reason = (
            f"Open-ended search space detected from {len(file_hints)} file hints, "
            f"{bullet_count} task bullets, and signals: {', '.join(open_signals) or 'broad scope'}."
        )
        return "open-ended", max_runs, reason

    planned = min(max_runs, 2)
    if wave_index > 1 and previous_wave_accepts <= 1:
        planned = 1
    reason = (
        f"Bounded search space detected from {len(file_hints)} file hints, "
        f"{bullet_count} task bullets, and signals: {', '.join(bounded_signals) or 'narrow scope'}."
    )
    return "bounded", planned, reason


def _write_artifact_payload(
    run_dir: Path,
    relative_path: str,
    payload: dict[str, object],
    *,
    kind: str,
    label: str,
    metadata: dict[str, object] | None = None,
) -> SwarmArtifactRef:
    artifact_path = run_dir / relative_path
    _write_json(artifact_path, payload)
    return _artifact_ref(run_dir, artifact_path, kind=kind, label=label, metadata=metadata)


def _refresh_transplant_commands(state: SwarmRunState) -> None:
    commit_refs = [item.commit_ref for item in state.accepted_commits if item.commit_ref]
    commands: list[str] = []
    if commit_refs:
        commands.append(f"git cherry-pick {' '.join(commit_refs)}")
    for accepted in state.accepted_commits:
        if accepted.patch_path:
            commands.append(f"git apply {shlex.quote(accepted.patch_path)}")
    state.transplant_commands = commands


def _write_run_base_snapshot_manifest(state: SwarmRunState, run_dir: Path) -> None:
    state.base_snapshot_ref = state.base_snapshot_ref or state.integration_base_ref or state.base_ref
    artifact = _write_artifact_payload(
        run_dir,
        "artifacts/base-snapshot.json",
        {
            "run_id": state.run_id,
            "base_ref": state.base_ref,
            "base_snapshot_ref": state.base_snapshot_ref,
            "integration_base_ref": state.integration_base_ref,
            "dirty_paths": state.dirty_paths,
            "dirty_state_applied": bool(state.dirty_paths),
            "semantics": "base_snapshot_ref is the exact integration snapshot the swarm launched from, including synced dirty state when present.",
        },
        kind="base-snapshot",
        label="Base snapshot manifest",
        metadata={
            "base_ref": state.base_ref,
            "base_snapshot_ref": state.base_snapshot_ref,
            "dirty_state_applied": bool(state.dirty_paths),
        },
    )
    state.base_snapshot_artifact = artifact
    _upsert_run_artifact(state, artifact)


def _compose_wave_spec_text(
    state: SwarmRunState,
    wave: SwarmWaveState,
    *,
    child_index: int | None,
    planned_runs: int,
) -> str:
    base_spec = _spec_text(state).strip()
    lines = [
        base_spec,
        "",
        "## Swarm convergence context",
        f"- Wave: {wave.wave_index} of a continuous swarm run.",
    ]
    if state.accepted_commits:
        lines.append("- Accepted improvements already integrated into the current base:")
        for accepted in state.accepted_commits[-5:]:
            summary = accepted.summary or accepted.child_id
            lines.append(f"  - {accepted.child_id}: {summary}")
    else:
        lines.append("- No prior improvements have been accepted yet.")
    if state.convergence_summary:
        lines.append(f"- Latest evaluator summary: {state.convergence_summary}")
    # Carry forward rejected attempts so workers don't repeat them.
    rejected_entries: list[str] = []
    accepted_ids = {c.child_id for c in state.accepted_commits}
    for child in state.children:
        if child.child_id in accepted_ids:
            continue
        if child.metric is None:
            continue  # no fitness measured — not informative
        summary_snippet = (child.summary or "")[:120].replace("\n", " ")
        rejected_entries.append(f"  - {child.child_id} (metric={child.metric:.4f}): {summary_snippet}")
    if rejected_entries:
        lines.append("- Approaches already tried that did NOT beat the baseline — DO NOT repeat these:")
        lines.extend(rejected_entries[-12:])  # cap at last 12 to avoid bloat
    directives = state.next_wave_directives[:]
    if directives:
        if child_index is None:
            lines.append("- Evaluator-proposed directions for this wave:")
            for item in directives[:6]:
                lines.append(f"  - {item}")
        else:
            focus = directives[(child_index - 1) % len(directives)]
            lines.append(f"- Primary focus for this child: {focus}")
            secondary = [item for item in directives if item != focus][:3]
            if secondary:
                lines.append("- Secondary directions to consider if they compose cleanly:")
                for item in secondary:
                    lines.append(f"  - {item}")
    else:
        lines.append(
            "- Primary focus for this child: explore the best remaining improvement opportunity without duplicating already accepted work."
        )
    lines.extend(
        [
            (
                f"- This is candidate {child_index} of {planned_runs}; bias toward a distinct angle rather than repeating another likely attempt."
                if child_index is not None
                else f"- The coordinator will launch up to {planned_runs} candidates from this wave spec."
            ),
            "- Prefer independent improvements that can stack with already accepted changes.",
            "- If you detect a conflict with accepted work, choose a compatible alternative rather than reverting the base.",
        ]
    )
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def _write_wave_evaluation_manifest(state: SwarmRunState, wave: SwarmWaveState) -> None:
    if wave.evaluation is None:
        return
    run_dir = Path(state.copied_spec_path).resolve().parent
    artifact = _write_artifact_payload(
        run_dir,
        f"artifacts/waves/wave-{wave.wave_index:02d}-evaluation.json",
        {
            "run_id": state.run_id,
            "wave_index": wave.wave_index,
            "evaluation": wave.evaluation.model_dump(mode="json"),
        },
        kind="wave-evaluation",
        label=f"Wave {wave.wave_index} evaluation",
        metadata={
            "wave_index": wave.wave_index,
            "verdict": wave.evaluation.verdict,
            "accepted_count": len(wave.evaluation.accepted_child_ids),
        },
    )
    wave.evaluation.artifact = artifact
    _upsert_run_artifact(state, artifact)


def _write_wave_manifest(state: SwarmRunState, wave: SwarmWaveState) -> None:
    run_dir = Path(state.copied_spec_path).resolve().parent
    rejected_outcomes = [
        {
            "child_id": child.child_id,
            "status": child.status,
            "summary": child.summary,
            "error": child.error,
            "acceptance_note": child.acceptance_note,
        }
        for child in _children_for_wave(state, wave.wave_index)
        if child.child_id in wave.rejected_child_ids
    ]
    artifact = _write_artifact_payload(
        run_dir,
        f"artifacts/waves/wave-{wave.wave_index:02d}-manifest.json",
        {
            "run_id": state.run_id,
            "wave_index": wave.wave_index,
            "status": wave.status,
            "max_runs": wave.max_runs,
            "planned_runs": wave.planned_runs,
            "planning_mode": wave.planning_mode,
            "planning_reason": wave.planning_reason,
            "primary_winner_child_id": wave.primary_winner_child_id,
            "accepted_child_ids": wave.accepted_child_ids,
            "accepted_commits": [item.model_dump(mode="json") for item in wave.accepted_commits],
            "rejected_outcomes": rejected_outcomes,
            "evaluation": (wave.evaluation.model_dump(mode="json") if wave.evaluation is not None else None),
            "integration_validation_results": [
                item.model_dump(mode="json") for item in wave.integration_validation_results
            ],
            "synthesized_spec_path": wave.synthesized_spec_path,
            "summary": wave.summary,
        },
        kind="wave-manifest",
        label=f"Wave {wave.wave_index} manifest",
        metadata={
            "wave_index": wave.wave_index,
            "accepted_count": len(wave.accepted_commits),
            "rejected_count": len(rejected_outcomes),
        },
    )
    wave.manifest_artifact = artifact
    _upsert_run_artifact(state, artifact)


def _write_run_acceptance_manifest(state: SwarmRunState) -> None:
    run_dir = Path(state.copied_spec_path).resolve().parent
    artifact = _write_artifact_payload(
        run_dir,
        "artifacts/accepted-commits.json",
        {
            "run_id": state.run_id,
            "base_snapshot_ref": state.base_snapshot_ref,
            "integration_base_ref": state.integration_base_ref,
            "accepted_child_ids": state.accepted_child_ids,
            "accepted_commits": [item.model_dump(mode="json") for item in state.accepted_commits],
            "transplant_commands": state.transplant_commands,
        },
        kind="accepted-commits",
        label="Accepted commit manifest",
        metadata={"accepted_count": len(state.accepted_commits)},
    )
    _upsert_run_artifact(state, artifact)


def _child_run_dir(root: Path, run_id: str, child_id: str) -> Path:
    return swarm_run_dir(root, run_id) / "children" / child_id


def _relative_git_status(worktree: Path) -> list[str]:
    completed = _git("status", "--short", cwd=worktree)
    if completed.returncode != 0:
        return []
    filtered: list[str] = []
    for line in completed.stdout.splitlines():
        text = line.rstrip()
        if not text:
            continue
        path_text = text[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if path_text == ".lemoncrow/swarm/" or path_text.startswith(".lemoncrow/swarm/"):
            continue
        filtered.append(text)
    return filtered


def _load_child_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_validation_results(
    base: list[SwarmValidationCheck],
    payload: object,
) -> list[SwarmValidationCheck]:
    merged = list(base)
    if not isinstance(payload, list):
        return merged
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or f"child-metadata-{index}")
        passed = bool(item.get("passed"))
        detail = str(item.get("detail") or "")
        merged.append(
            SwarmValidationCheck(
                name=name,
                command=str(item.get("command") or name),
                passed=passed,
                exit_code=0 if passed else int(item.get("exit_code") or 1),
                detail=detail,
                stdout_path="",
                stderr_path="",
                duration_seconds=float(item.get("duration_seconds") or 0.0),
            )
        )
    return merged


def _summarize_output(stdout_path: Path, stderr_path: Path) -> str:
    for path in (stdout_path, stderr_path):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1][:400]
    return "No summary emitted."


def _collect_cost_and_tokens(lemoncrow_root: Path) -> tuple[int, float]:
    runs_dir = lemoncrow_root / "sessions"
    if not runs_dir.is_dir():
        return 0, 0.0
    total_tokens = 0
    total_cost = 0.0
    for path in runs_dir.glob("**/run.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total_tokens += int(payload.get("token_count") or 0)
        cost = payload.get("cost") or {}
        if isinstance(cost, dict):
            total_cost += float(cost.get("total_cost_usd") or 0.0)
    return total_tokens, round(total_cost, 6)


def _expand_command_tokens(child: SwarmChildState, command: list[str]) -> list[str]:
    spec_contents = Path(child.spec_path).read_text(encoding="utf-8").strip()
    replacements = {
        "{spec}": child.spec_path,
        "{spec_contents}": spec_contents,
        "{worktree}": child.worktree_path,
        "{child_id}": child.child_id,
        "{lemoncrow_root}": child.lemoncrow_root,
        "{result_path}": child.result_path,
        "{metadata_path}": child.metadata_path,
    }
    expanded: list[str] = []
    for token in command:
        updated = token
        for placeholder, value in replacements.items():
            updated = updated.replace(placeholder, value)
        expanded.append(updated)
    return expanded


def _coerce_int(value: object, fallback: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return int(value)
    return fallback


def _coerce_float(value: object, fallback: float) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return fallback


def _coerce_float_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value)
    return None


def _coerce_findings(value: object) -> list[Finding]:
    if not isinstance(value, list):
        return []
    findings: list[Finding] = []
    for item in value:
        if isinstance(item, dict):
            with contextlib.suppress(Exception):
                findings.append(Finding.model_validate(item))
    return findings


# `_score_child`, `rank_children`, and the structural-validation helpers now live
# in `swarm/reducers/best.py` (the heuristic `best` reducer) and are re-imported
# at the top of this module. They are referenced below unchanged.


def _ensure_snapshot_commit(worktree: Path, message: str) -> str:
    status = _relative_git_status(worktree)
    if not status:
        return read_head_ref(worktree)
    add_completed = _git("add", "-A", cwd=worktree)
    if add_completed.returncode != 0:
        raise RuntimeError(add_completed.stderr.strip() or "git add failed")
    commit_completed = subprocess.run(
        [
            "git",
            "-C",
            str(worktree),
            "-c",
            "user.name=LemonCrow Swarm",
            "-c",
            "user.email=swarm@lemoncrow.local",
            "commit",
            "--allow-empty",
            "--no-verify",
            "-m",
            message,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit_completed.returncode != 0:
        raise RuntimeError(commit_completed.stderr.strip() or "git commit failed")
    return read_head_ref(worktree)


def _tail_lines(path: Path, tail: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-tail:])


def _preview_file(path: Path, tail: int = 5) -> tuple[str, datetime | None]:
    if not path.exists():
        return "", None
    preview = _tail_lines(path, tail).strip()
    last_output_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return preview[-800:], last_output_at


def _update_child_activity(child: SwarmChildState) -> bool:
    stdout_preview, stdout_time = _preview_file(Path(child.stdout_path))
    stderr_preview, stderr_time = _preview_file(Path(child.stderr_path))
    latest_time = max(
        [item for item in (stdout_time, stderr_time) if item is not None],
        default=None,
    )
    activity = ""
    for preview in (stderr_preview, stdout_preview):
        lines = [line.strip() for line in preview.splitlines() if line.strip()]
        if lines:
            activity = lines[-1][:240]
            break
    updated = (
        child.stdout_preview != stdout_preview
        or child.stderr_preview != stderr_preview
        or child.current_activity != activity
        or child.last_output_at != latest_time
    )
    child.stdout_preview = stdout_preview
    child.stderr_preview = stderr_preview
    child.current_activity = activity
    child.last_output_at = latest_time
    return updated


def _children_for_wave(state: SwarmRunState, wave_index: int) -> list[SwarmChildState]:
    return [child for child in state.children if child.wave_index == wave_index]


def _latest_wave(state: SwarmRunState, wave_index: int) -> SwarmWaveState:
    for wave in state.waves:
        if wave.wave_index == wave_index:
            return wave
    raise RuntimeError(f"wave {wave_index} not found")


def _refresh_child_result(
    children: list[SwarmChildState],
    child_id: str,
) -> SwarmChildState | None:
    child = next((item for item in children if item.child_id == child_id), None)
    if child is None:
        return None
    result_path = Path(child.result_path)
    if not result_path.exists():
        return child
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    refreshed = SwarmChildState.model_validate(payload)
    index = children.index(child)
    children[index] = refreshed
    return refreshed


def _kill_pid(pid: int | None) -> None:
    if pid is None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def _python_cli_invocation() -> list[str]:
    return [sys.executable, "-m", "lemoncrow.gateway.cli"]


def _workspace_relative_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if not raw_path.strip():
        raise ValueError("path is required")
    if candidate.is_absolute():
        raise ValueError("paths must be workspace-relative")
    resolved = (workspace_root / candidate).resolve()
    if not resolved.is_relative_to(workspace_root):
        raise ValueError(f"path escapes the swarm worktree: {raw_path}")
    return resolved


def _validate_worker_files(workspace_root: Path, files: object, *, allow_missing: bool = True) -> list[str]:
    if files is None:
        return []
    if not isinstance(files, list):
        raise ValueError("files must be a list of relative paths")
    validated: list[str] = []
    for item in files[:8]:
        if not isinstance(item, str):
            raise ValueError("files entries must be strings")
        resolved = _workspace_relative_path(workspace_root, item)
        if resolved.exists() or allow_missing:
            validated.append(resolved.relative_to(workspace_root).as_posix())
    return validated


def _validate_provider_worker_action(
    workspace_root: Path,
    payload: object,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("worker action must be a JSON object")
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"context", "search", "read", "edit", "finish"}:
        raise ValueError(f"unsupported worker action: {action or '(missing)'}")
    if action == "context":
        task = str(payload.get("task") or "").strip()
        if not task:
            raise ValueError("context action requires task")
        files = _validate_worker_files(workspace_root, payload.get("files"))
        return {"action": action, "task": task, "files": files}
    if action == "search":
        query = str(payload.get("query") or "").strip()
        if not query:
            raise ValueError("search action requires query")
        path_text = str(payload.get("path") or ".").strip() or "."
        search_path = _workspace_relative_path(workspace_root, path_text).relative_to(workspace_root).as_posix()
        return {"action": action, "query": query, "path": search_path}
    if action == "read":
        path_text = str(payload.get("path") or "").strip()
        if not path_text:
            raise ValueError("read action requires path")
        file_path = _workspace_relative_path(workspace_root, path_text)
        if not file_path.exists() or not file_path.is_file():
            raise ValueError(f"file not found: {path_text}")
        normalized: dict[str, Any] = {
            "action": action,
            "path": file_path.relative_to(workspace_root).as_posix(),
            "full": bool(payload.get("full")),
        }
        if payload.get("range"):
            normalized["range"] = str(payload["range"])
        return normalized
    if action == "edit":
        edits = payload.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError("edit action requires at least one edit")
        normalized_edits: list[dict[str, Any]] = []
        for item in edits[:8]:
            if not isinstance(item, dict):
                raise ValueError("edit entries must be objects")
            file_path_text = str(item.get("file_path") or "").strip()
            new_string = item.get("new_string")
            if not file_path_text or not isinstance(new_string, str):
                raise ValueError("each edit requires file_path and new_string")
            resolved_path = _workspace_relative_path(workspace_root, file_path_text)
            normalized_edit: dict[str, Any] = {
                "file_path": resolved_path.relative_to(workspace_root).as_posix(),
                "new_string": new_string,
            }
            if "old_string" in item:
                old_string = item.get("old_string")
                if not isinstance(old_string, str):
                    raise ValueError("old_string must be a string when provided")
                normalized_edit["old_string"] = old_string
            if item.get("replace") or item.get("overwrite"):  # overwrite is the legacy name
                normalized_edit["replace"] = True
            normalized_edits.append(normalized_edit)
        return {"action": action, "edits": normalized_edits}
    summary = str(payload.get("summary") or "").strip()
    if not summary:
        raise ValueError("finish action requires summary")
    return {"action": action, "summary": summary[:800]}


def _redact_tool_payload(payload: object) -> object:
    if isinstance(payload, dict):
        redacted: dict[str, object] = {}
        for key, value in payload.items():
            if "path" in key.lower() and isinstance(value, str):
                with contextlib.suppress(Exception):
                    value = Path(value).name
            redacted[key] = _redact_tool_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_tool_payload(item) for item in payload[:12]]
    if isinstance(payload, str) and len(payload) > 6000:
        return f"{payload[:6000]}\n...[truncated]..."
    return payload


def _render_worker_observation(name: str, payload: object) -> str:
    return json.dumps({"tool": name, "result": _redact_tool_payload(payload)}, ensure_ascii=False)


def _run_structural_validation(
    *,
    workspace_root: Path,
    run_dir: Path,
    env: dict[str, str],
) -> SwarmValidationCheck:
    stdout_path = run_dir / "structural-diff-check.stdout.log"
    stderr_path = run_dir / "structural-diff-check.stderr.log"
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        proc = subprocess.run(
            ["git", "diff", "--check"],
            cwd=workspace_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
            timeout=30,
        )
    duration = round(time.perf_counter() - started, 3)
    detail = "pass" if proc.returncode == 0 else "git diff --check reported formatting or merge issues"
    return SwarmValidationCheck(
        name="structural-diff-check",
        command="git diff --check",
        passed=proc.returncode == 0,
        exit_code=proc.returncode,
        detail=detail,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        duration_seconds=duration,
    )


def run_provider_swarm_worker() -> int:
    from lemoncrow.gateway.sdk import mcp as gateway_mcp
    from lemoncrow.infra import internal_llm

    workspace_root = Path(
        os.environ.get("LEMONCROW_WORKSPACE_ROOT") or os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
    ).resolve()
    spec_path = Path(os.environ["LEMONCROW_SWARM_SPEC_PATH"]).resolve()
    metadata_path = Path(os.environ["LEMONCROW_SWARM_METADATA_PATH"]).resolve()
    provider = os.environ.get("LEMONCROW_SWARM_PROVIDER", "").strip().lower()
    model = os.environ.get("LEMONCROW_SWARM_MODEL", "").strip() or None
    step_budget = _coerce_int(os.environ.get("LEMONCROW_SWARM_STEP_BUDGET"), 10)
    time_budget = _coerce_int(os.environ.get("LEMONCROW_SWARM_TIME_BUDGET_SECONDS"), 240)
    if provider not in {"openai", "litellm"}:
        raise RuntimeError(f"provider-backed swarm worker requires openai/litellm, got {provider!r}")

    spec_text = spec_path.read_text(encoding="utf-8", errors="replace")
    client = gateway_mcp.MCPClient(root=Path(os.environ["LEMONCROW_ROOT"]).resolve())
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a bounded swarm child working inside a single git worktree. "
                "Return exactly one JSON object matching the schema. Use only relative paths. "
                "Allowed actions: context, search, read, edit, finish. "
                "Edits must use the rich edit format with file_path/new_string and optional old_string or replace=true. "
                "Stop as soon as the requested work is complete."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Implement the spec from {spec_path.name} inside this repository.\n"
                f"Provider: {provider}\n"
                f"Model: {model or '(provider default)'}\n"
                f"Step budget: {step_budget}\n"
                f"Time budget seconds: {time_budget}\n\n"
                f"SPEC:\n{spec_text}"
            ),
        },
    ]

    started = time.monotonic()
    invalid_actions = 0
    summary = ""
    final_error = ""
    validation_results: list[SwarmValidationCheck] = []

    for _step in range(1, max(step_budget, 1) + 1):
        if time.monotonic() - started >= max(time_budget, 30):
            final_error = f"Worker timed out after {time_budget}s."
            break
        try:
            response = internal_llm.chat(
                messages,
                model=model,
                json_schema=_PROVIDER_WORKER_ACTION_SCHEMA,
            )
            action = _validate_provider_worker_action(workspace_root, response)
        except (internal_llm.InternalLLMError, ValueError, KeyError) as exc:
            invalid_actions += 1
            if invalid_actions >= 3:
                final_error = str(exc)
                break
            messages.append(
                {
                    "role": "user",
                    "content": (f"Previous response was invalid: {exc}. Return a corrected JSON object only."),
                }
            )
            continue

        invalid_actions = 0
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})

        if action["action"] == "finish":
            summary = action["summary"]
            break

        try:
            if action["action"] == "context":
                result = client.get_context(
                    task=action["task"],
                    domain="swarm",
                    files=action.get("files") or None,
                    tools=["context", "search", "read", "edit"],
                    max_blocks=6,
                    token_budget=2200,
                ).model_dump(mode="json")
            elif action["action"] == "search":
                result = client.smart_search(
                    query=action["query"],
                    path=action["path"],
                    limit=8,
                )
            elif action["action"] == "read":
                result = client.transport.call_tool(
                    "read",
                    {
                        "path": action["path"],
                        "full": bool(action.get("full")),
                        **({"range": action["range"]} if action.get("range") else {}),
                    },
                )
            else:
                result = client.transport.call_tool(
                    "edit",
                    {
                        "edits": action["edits"],
                        "atomic": True,
                        "post_edit_hooks": False,
                    },
                )
        except (
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive path
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool execution failed: {exc}. Continue with another valid action.",
                }
            )
            continue

        messages.append(
            {
                "role": "user",
                "content": _render_worker_observation(action["action"], result),
            }
        )

    validation_results.append(
        _run_structural_validation(workspace_root=workspace_root, run_dir=metadata_path.parent, env=dict(os.environ))
    )
    if not summary and not final_error:
        summary = "Completed provider-backed swarm worker without an explicit finish summary."

    payload: dict[str, object] = {
        "summary": summary,
        "error": final_error,
        "validation_results": [item.model_dump(mode="json") for item in validation_results],
    }
    _write_json(metadata_path, payload)

    status_line = summary or final_error or "Provider-backed swarm worker finished."
    print(status_line)
    for check in validation_results:
        print(f"{check.name}: {'pass' if check.passed else 'fail'}")
    return 0 if not final_error else 1


def spawn_swarm_coordinator(
    root: Path,
    repo_root: Path,
    state_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, Path]:
    log_path = state_path.parent / "coordinator.log"
    env = dict(os.environ)
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                *_python_cli_invocation(),
                "--root",
                str(root),
                "swarm",
                "_run",
                "--state",
                str(state_path),
            ],
            cwd=repo_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
    return proc.pid, log_path


def list_swarm_runs(root: Path) -> list[SwarmRunState]:
    runs_root = Path(root) / "swarm" / "runs"
    if not runs_root.exists():
        return []
    states: list[SwarmRunState] = []
    for state_path in sorted(runs_root.glob("*/state.json")):
        with contextlib.suppress(OSError, json.JSONDecodeError, ValueError):
            states.append(load_swarm_state(state_path))
    return sorted(states, key=lambda item: item.created_at, reverse=True)


def _transcript_turn_text(entry: dict[str, Any]) -> str:
    """Extract a compact one-line summary from one Claude Code session transcript entry."""
    message = entry.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        text = content.strip()
        if text:
            parts.append(text)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
            elif block_type == "tool_use":
                parts.append(f"[{block.get('name', 'tool')}] {json.dumps(block.get('input', {}))}")
            elif block_type == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = " ".join(str(part.get("text", "")) for part in inner if isinstance(part, dict))
                text = str(inner or "").strip()
                if text:
                    parts.append(f"[result] {text}")
    return " | ".join(parts)


def _latest_claude_transcript_path(worktree_path: str) -> Path | None:
    """Most recently modified Claude Code session transcript for *worktree_path*,
    or None. Claude Code names each project's transcript directory after the
    cwd it was started in, with path separators replaced by ``-``."""
    encoded = str(Path(worktree_path)).replace(os.sep, "-")
    project_dir = Path.home() / ".claude" / "projects" / encoded
    if not project_dir.is_dir():
        return None
    transcripts = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return transcripts[0] if transcripts else None


def _find_codex_session_id(worktree_path: str, *, scan_limit: int = 500) -> str | None:
    """Session id of the most recent Codex rollout recorded for *worktree_path*.

    Unlike Claude Code, Codex stores every session under a flat
    ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` tree with no cwd in the
    path, so the cwd is recovered from each rollout's first (``session_meta``)
    line instead. Scans newest-first and stops at the first match, bounded by
    *scan_limit* so this stays cheap even with years of session history.
    """
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.is_dir():
        return None
    rollouts = sorted(sessions_root.glob("*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    target = str(Path(worktree_path))
    for path in rollouts[:scan_limit]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                first_line = handle.readline()
        except OSError:
            continue
        try:
            meta = json.loads(first_line)
        except json.JSONDecodeError:
            continue
        payload = meta.get("payload") if isinstance(meta, dict) else None
        if isinstance(payload, dict) and payload.get("cwd") == target:
            session_id = payload.get("id")
            return str(session_id) if session_id else None
    return None


def resolve_swarm_child_attach(runner_name: str, worktree_path: str) -> dict[str, str] | None:
    """Best-effort runner-native session id for *worktree_path*, plus the exact
    command to interactively attach/resume it — or None if not discoverable
    (e.g. an API-backed provider worker with no local CLI session)."""
    if "codex" in (runner_name or "").lower():
        session_id = _find_codex_session_id(worktree_path)
        if session_id is None:
            return None
        return {"session_id": session_id, "attach_command": f"codex exec resume {session_id}"}
    # claude / ollama-claude / other claude-print-based profiles all use the Claude Code CLI.
    transcript_path = _latest_claude_transcript_path(worktree_path)
    if transcript_path is None:
        return None
    session_id = transcript_path.stem
    return {"session_id": session_id, "attach_command": f"cd {worktree_path} && claude --resume {session_id}"}


def read_swarm_child_activity(child: SwarmChildState, *, turns: int = 2, max_chars: int = 300) -> str:
    """Best-effort tail of a child's live Claude Code session transcript.

    Children run non-interactively (``claude --print``), so ``stdout``/``stderr``
    are buffered until process exit and are empty for the child's entire
    lifetime. Each child does write an incremental session transcript under
    ``~/.claude/projects/<encoded-worktree-path>/*.jsonl`` while it works;
    this tails the most recent one and returns the last few meaningful turns
    (assistant text / tool calls / tool results), or "" if none is found.
    """
    transcript_path = _latest_claude_transcript_path(child.worktree_path)
    if transcript_path is None:
        return ""
    try:
        raw_lines = _tail_lines(transcript_path, 400).splitlines()
    except OSError:
        return ""
    summaries: list[str] = []
    for line in reversed(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        summary = _transcript_turn_text(entry)
        if summary:
            summaries.append(summary[:max_chars])
        if len(summaries) >= turns:
            break
    if not summaries:
        return ""
    idle_seconds = max(
        0, int((_utcnow() - datetime.fromtimestamp(transcript_path.stat().st_mtime, tz=UTC)).total_seconds())
    )
    lines = [f"live activity (idle {idle_seconds}s, transcript={transcript_path.name}):"]
    lines.extend(f"  {index}. {summary}" for index, summary in enumerate(summaries, start=1))
    return "\n".join(lines)


def read_swarm_log(
    root: Path,
    run_id: str,
    *,
    child_id: str | None = None,
    stderr: bool = False,
    tail: int = 40,
) -> str:
    state = load_swarm_state(resolve_state_path(root, run_id))
    if child_id is None:
        log_path = Path(state.coordinator_log_path or swarm_run_dir(root, run_id) / "coordinator.log")
        source_label = f"coordinator log: {log_path}"
        content = _tail_lines(log_path, tail).strip()
        if not content:
            return f"No log output yet at {log_path}"
        child_cmd = " ".join(state.child_command)
        return f"# {source_label}\n# command: {child_cmd}\n\n" + content

    child = next((item for item in state.children if item.child_id == child_id), None)
    if child is None:
        raise RuntimeError(f"unknown child id: {child_id}")

    # Non-interactive children buffer stdout/stderr until process exit, so a
    # running child's log files are empty for its whole lifetime. Prefer the
    # live session transcript in that case; fall back to the raw log once
    # the child has finished (or if no transcript is found).
    if child.status == "running" and not stderr:
        activity = read_swarm_child_activity(child, turns=5)
        if activity:
            return f"# {child_id}: live session transcript (child still running)\n\n{activity}"

    log_path = Path(child.stderr_path if stderr else child.stdout_path)
    stream_label = "stderr" if stderr else "stdout"
    source_label = f"{child_id} {stream_label}: {log_path}"
    content = _tail_lines(log_path, tail).strip()
    if not content:
        return f"No log output yet at {log_path}"
    child_cmd = " ".join(state.child_command)
    return f"# {source_label}\n# command: {child_cmd}\n\n" + content


def build_swarm_export_payload(state: SwarmRunState) -> dict[str, object]:
    return {
        "run_id": state.run_id,
        "status": state.status,
        "mode": state.mode,
        "runner_name": state.runner_name,
        "runner_model": state.runner_model,
        "evaluator_backend": state.evaluator_backend,
        "evaluator_model": state.evaluator_model,
        "base_ref": state.base_ref,
        "base_snapshot_ref": state.base_snapshot_ref,
        "integration_base_ref": state.integration_base_ref,
        "convergence_status": state.convergence_status,
        "convergence_summary": state.convergence_summary,
        "next_wave_directives": list(state.next_wave_directives),
        "artifact_root": state.artifact_root,
        "base_snapshot_artifact": (
            state.base_snapshot_artifact.model_dump(mode="json") if state.base_snapshot_artifact is not None else None
        ),
        "accepted_child_ids": list(state.accepted_child_ids),
        "accepted_commits": [item.model_dump(mode="json") for item in state.accepted_commits],
        "waves": [
            {
                "wave_index": wave.wave_index,
                "status": wave.status,
                "max_runs": wave.max_runs,
                "planned_runs": wave.planned_runs,
                "planning_mode": wave.planning_mode,
                "primary_winner_child_id": wave.primary_winner_child_id,
                "accepted_child_ids": list(wave.accepted_child_ids),
                "rejected_child_ids": list(wave.rejected_child_ids),
                "evaluation": (wave.evaluation.model_dump(mode="json") if wave.evaluation is not None else None),
                "integration_validation_results": [
                    item.model_dump(mode="json") for item in wave.integration_validation_results
                ],
                "manifest_artifact": (
                    wave.manifest_artifact.model_dump(mode="json") if wave.manifest_artifact is not None else None
                ),
            }
            for wave in state.waves
        ],
        "artifacts": [artifact.model_dump(mode="json") for artifact in state.export_artifacts],
        "transplant_commands": list(state.transplant_commands),
    }


def build_swarm_apply_payload(
    state: SwarmRunState,
    *,
    wave_index: int | None = None,
    child_id: str | None = None,
) -> dict[str, object]:
    selected_commits = list(state.accepted_commits)
    if wave_index is not None:
        selected_ids = {child_id for child_id in _latest_wave(state, wave_index).accepted_child_ids}
        selected_commits = [item for item in selected_commits if item.child_id in selected_ids]
    if child_id is not None:
        selected_commits = [item for item in selected_commits if item.child_id == child_id]
    cherry_pick_refs = [item.commit_ref for item in selected_commits if item.commit_ref]
    commands: list[str] = []
    if cherry_pick_refs:
        commands.append(f"git cherry-pick {' '.join(cherry_pick_refs)}")
    commands.extend(f"git apply {shlex.quote(item.patch_path)}" for item in selected_commits if item.patch_path)
    return {
        "run_id": state.run_id,
        "wave_index": wave_index,
        "child_id": child_id,
        "base_snapshot_ref": state.base_snapshot_ref,
        "integration_base_ref": state.integration_base_ref,
        "selected_commits": [item.model_dump(mode="json") for item in selected_commits],
        "commands": commands,
        "artifacts": [artifact.model_dump(mode="json") for item in selected_commits for artifact in item.artifacts],
    }


def initialize_swarm_run(
    *,
    root: Path,
    repo_root: Path,
    spec_path: Path,
    child_command: list[str],
    runs: int,
    validation_commands: list[str],
    keep_worktrees: bool,
    detached: bool,
    runner_name: str = "custom",
    runner_model: str = "",
    spec_source_path: str = "",
    spec_resolution: Literal["explicit", "default"] = "explicit",
    used_program_md: bool = False,
    launch_provider: Literal["cli", "openai", "litellm"] = "cli",
    launch_effort: str = "",
    evaluator_backend: SwarmEvaluatorBackend = "auto",
    evaluator_model: str = "",
    continuous: bool = False,
    max_waves: int = 0,
    max_evaluator_failures: int = 3,
    job_kind: str = "solve",
    reducer_name: str = "merge",
    exec_mode: SwarmExecMode = "edit",
    search_space: list[str] | None = None,
    fitness_spec: FitnessSpec | None = None,
    quorum: int = 0,
) -> tuple[SwarmRunState, Path]:
    root = Path(root).resolve()
    repo_root = Path(repo_root).resolve()
    spec_path = Path(spec_path).resolve()
    run_id = f"swarm-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = swarm_run_dir(root, run_id)
    spec_copy = run_dir / "PROGRAM.md"
    spec_copy.parent.mkdir(parents=True, exist_ok=True)
    spec_copy.write_text(spec_path.read_text(encoding="utf-8"), encoding="utf-8")
    worktree_pool = repo_root.parent / f"{repo_root.name}-swarm-worktrees" / run_id
    manager = SwarmWorktreeManager(repo_root=repo_root, pool_root=worktree_pool)
    dirty_paths = collect_dirty_paths(repo_root)
    integration_worktree = manager.create_worktree(
        run_id=run_id,
        child_id="integration",
        ref=read_head_ref(repo_root),
    )
    manager.sync_dirty_state(base_worktree=repo_root, child_worktree=integration_worktree)
    integration_base_ref = _ensure_snapshot_commit(
        integration_worktree,
        "[swarm] base snapshot",
    )
    artifact_root = run_dir / "artifacts"
    state = SwarmRunState(
        run_id=run_id,
        status="pending",
        mode="continuous" if continuous else "single",
        repo_root=str(repo_root),
        base_worktree=str(repo_root),
        base_ref=read_head_ref(repo_root),
        base_snapshot_ref=integration_base_ref,
        worktree_pool=str(worktree_pool),
        integration_worktree=str(integration_worktree),
        integration_base_ref=integration_base_ref,
        artifact_root=str(artifact_root),
        spec_source_path=str(spec_source_path or spec_path),
        copied_spec_path=str(spec_copy),
        spec_resolution=spec_resolution,
        used_program_md=used_program_md,
        runner_name=runner_name,
        runner_model=runner_model,
        launch_provider=launch_provider,
        launch_effort=launch_effort,
        evaluator_backend=evaluator_backend,
        evaluator_model=evaluator_model,
        job_kind=job_kind,
        reducer_name=reducer_name,
        exec_mode=exec_mode,
        search_space=list(search_space or []),
        fitness_spec=fitness_spec,
        quorum=quorum,
        child_command=list(child_command),
        validation_commands=list(validation_commands),
        runs=runs,
        max_runs=runs,
        max_waves=max(max_waves, 0),
        max_evaluator_failures=max(max_evaluator_failures, 1),
        keep_worktrees=keep_worktrees,
        detached=detached,
        dirty_paths=dirty_paths,
        limitations=[
            "Each child gets an isolated git worktree and LEMONCROW_ROOT.",
            "The agent command must consume the provided worktree/spec env vars to use LemonCrow MCP/runtime inside each child.",
            "Accepted child patches are merged onto an integration worktree in score order; later waves branch from that accepted base.",
            "max_runs is the per-wave cap; planned_runs records how many children the coordinator actually launched in that wave.",
            "Continuous mode now uses a semantic evaluator plus stagnation budgets to decide whether later waves should keep running.",
        ],
    )
    _write_run_base_snapshot_manifest(state, run_dir)
    state_path = resolve_state_path(root, run_id)
    save_swarm_state(state_path, state)
    return state, state_path


def build_child_env(child: SwarmChildState, state: SwarmRunState) -> dict[str, str]:
    effort = (state.launch_effort or "high").lower().strip()
    step_budget = _PROVIDER_WORKER_STEP_BUDGETS.get(effort, _PROVIDER_WORKER_STEP_BUDGETS["high"])
    time_budget = _PROVIDER_WORKER_TIME_BUDGETS.get(effort, _PROVIDER_WORKER_TIME_BUDGETS["high"])
    env = dict(os.environ)
    env.update(
        {
            "LEMONCROW_ROOT": child.lemoncrow_root,
            "LEMONCROW_WORKSPACE_ROOT": child.worktree_path,
            "CLAUDE_WORKSPACE_ROOT": child.worktree_path,
            "LEMONCROW_SWARM_RUN_ID": state.run_id,
            "LEMONCROW_SWARM_CHILD_ID": child.child_id,
            "LEMONCROW_SWARM_SPEC_PATH": child.spec_path,
            "LEMONCROW_SWARM_RESULT_PATH": child.result_path,
            "LEMONCROW_SWARM_METADATA_PATH": child.metadata_path,
            "LEMONCROW_SWARM_BASE_REF": state.integration_base_ref,
            "LEMONCROW_SWARM_PROVIDER": state.launch_provider,
            "LEMONCROW_SWARM_MODEL": state.runner_model,
            "LEMONCROW_SWARM_EFFORT": effort,
            "LEMONCROW_SWARM_STEP_BUDGET": str(step_budget),
            "LEMONCROW_SWARM_TIME_BUDGET_SECONDS": str(time_budget),
        }
    )
    if state.launch_provider == "openai":
        env["LEMONCROW_LLM_BACKEND"] = "openai"
        if state.runner_model:
            env["LEMONCROW_OPENAI_MODEL"] = state.runner_model
    elif state.launch_provider == "litellm":
        env["LEMONCROW_LLM_BACKEND"] = "litellm"
        if state.runner_model:
            env["LEMONCROW_LITELLM_MODEL"] = state.runner_model
    return env


def _plan_wave_approaches(state: SwarmRunState, planned_runs: int) -> list[str]:
    """Run a single planning LLM call to generate ``planned_runs`` orthogonal approaches.

    Returns one approach string per child so that ``state.next_wave_directives``
    can be seeded before wave-1 children are launched — preventing every run from
    independently converging on the same obvious solution.

    Falls back to an empty list on any error so the swarm continues normally.
    """
    if planned_runs <= 1:
        return []
    # Only provider-backed launches (openai/litellm) can return structured output.
    # CLI-mode children run as full claude subprocesses where we can't easily
    # parse structured output, so skip planning for that mode — the children
    # are capable enough to self-direct from the spec.
    if state.launch_provider not in {"openai", "litellm"}:
        return []
    backend = state.launch_provider

    from lemoncrow.infra import internal_llm

    spec_text = _spec_text(state).strip()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a swarm planning agent. Given a task specification and a requested number of "
                "parallel implementation runs, produce exactly that many distinct, non-overlapping "
                "implementation approaches. Each approach must differ meaningfully from the others: "
                "different algorithms, data structures, entry points, or architectural patterns. "
                "Avoid paraphrasing the same idea. Be concrete and specific — one or two sentences "
                "each. Return only the 'approaches' array."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"task": spec_text, "num_approaches": planned_runs},
                ensure_ascii=False,
            ),
        },
    ]
    env_key = "LEMONCROW_LLM_BACKEND"
    previous_backend = os.environ.get(env_key)
    os.environ[env_key] = backend
    try:
        response = internal_llm.chat(
            messages,
            model=state.runner_model or None,
            json_schema=_SWARM_APPROACH_SCHEMA,
        )
    except internal_llm.InternalLLMError:
        return []
    finally:
        if previous_backend is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_backend

    if not isinstance(response, dict):
        return []
    approaches = response.get("approaches")
    if not isinstance(approaches, list):
        return []
    return [str(a) for a in approaches if a][:planned_runs]


def _prepare_wave(state: SwarmRunState, root: Path, wave_index: int) -> SwarmWaveState:
    manager = SwarmWorktreeManager(
        repo_root=Path(state.repo_root),
        pool_root=Path(state.worktree_pool),
    )
    planning_mode, planned_runs, planning_reason = _plan_wave_runs(state, wave_index)
    state.max_runs = max(state.max_runs or state.runs or 1, 1)
    state.runs = state.max_runs
    state.planning_mode = planning_mode
    state.fan_out_reason = planning_reason
    wave = SwarmWaveState(
        wave_index=wave_index,
        max_runs=state.max_runs,
        planned_runs=planned_runs,
        planning_mode=planning_mode,
        planning_reason=planning_reason,
    )
    spec_copy = Path(state.copied_spec_path)
    run_dir = Path(state.copied_spec_path).resolve().parent
    synthesized_spec_path = run_dir / "artifacts" / "waves" / f"wave-{wave.wave_index:02d}-PROGRAM.md"
    synthesized_spec_path.parent.mkdir(parents=True, exist_ok=True)
    synthesized_spec_path.write_text(
        _compose_wave_spec_text(state, wave, child_index=None, planned_runs=wave.planned_runs),
        encoding="utf-8",
    )
    wave.synthesized_spec_path = str(synthesized_spec_path)
    wave.synthesized_spec_artifact = _artifact_ref(
        run_dir,
        synthesized_spec_path,
        kind="wave-spec",
        label=f"Wave {wave.wave_index} synthesized PROGRAM.md",
        metadata={"wave_index": wave.wave_index},
    )
    _upsert_run_artifact(state, wave.synthesized_spec_artifact)
    # Seed diverse approaches before children launch so wave-1 runs don't
    # independently converge on the same solution.  Only runs when no prior
    # evaluator feedback exists (i.e. the very first wave or a fresh swarm).
    if not state.next_wave_directives and planned_runs > 1:
        approaches = _plan_wave_approaches(state, planned_runs)
        if approaches:
            state.next_wave_directives = approaches
    for index in range(1, wave.planned_runs + 1):
        child_id = f"wave-{wave_index:02d}-run-{index:02d}"
        child_dir = _child_run_dir(root, state.run_id, child_id)
        child_dir.mkdir(parents=True, exist_ok=True)
        worktree_path = manager.create_worktree(
            run_id=state.run_id,
            child_id=child_id,
            ref=state.integration_base_ref,
        )
        child_spec_path = worktree_path / ".lemoncrow/swarm" / "PROGRAM.md"
        child_spec_path.parent.mkdir(parents=True, exist_ok=True)
        child_spec_path.write_text(
            (
                _compose_wave_spec_text(state, wave, child_index=index, planned_runs=wave.planned_runs)
                if state.mode == "continuous" or bool(state.next_wave_directives)
                else spec_copy.read_text(encoding="utf-8")
            ),
            encoding="utf-8",
        )
        child_root = child_dir / "lemoncrow-root"
        store = create_store(child_root)
        store.init()
        child = SwarmChildState(
            child_id=child_id,
            label=f"candidate-{index}",
            wave_index=wave_index,
            worktree_path=str(worktree_path),
            lemoncrow_root=str(child_root),
            run_dir=str(child_dir),
            spec_path=str(child_spec_path),
            result_path=str(child_dir / "result.json"),
            stdout_path=str(child_dir / "child.stdout.log"),
            stderr_path=str(child_dir / "child.stderr.log"),
            metadata_path=str(child_dir / "child-metadata.json"),
            patch_path=str(child_dir / "candidate.patch"),
        )
        state.children.append(child)
        wave.child_ids.append(child_id)
    state.waves.append(wave)
    state.current_wave = wave_index
    return wave


def _run_wave_children(root: Path, state_path: Path, wave_index: int) -> SwarmRunState:
    state = load_swarm_state(state_path)
    procs: dict[str, tuple[subprocess.Popen[str], contextlib.ExitStack]] = {}
    wave_children = _children_for_wave(state, wave_index)
    for child in wave_children:
        stack = contextlib.ExitStack()
        stdout_handle = stack.enter_context(Path(child.stdout_path).open("w", encoding="utf-8"))
        stderr_handle = stack.enter_context(Path(child.stderr_path).open("w", encoding="utf-8"))
        proc = subprocess.Popen(
            [
                *_python_cli_invocation(),
                "--root",
                child.lemoncrow_root,
                "swarm",
                "_child-run",
                "--state",
                str(state_path),
                "--child-id",
                child.child_id,
            ],
            cwd=child.worktree_path,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        procs[child.child_id] = (proc, stack)
        child.pid = proc.pid
        child.status = "running"
        child.started_at = _utcnow()
    save_swarm_state(state_path, state)

    while procs:
        state = load_swarm_state(state_path)
        if state.stop_requested:
            for child_id, (proc, stack) in procs.items():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                child = next(item for item in state.children if item.child_id == child_id)
                child.status = "stopped"
                child.finished_at = _utcnow()
                stack.close()
            procs.clear()
            wave = _latest_wave(state, wave_index)
            wave.status = "stopped"
            wave.finished_at = _utcnow()
            wave.summary = "Stopped by user."
            save_swarm_state(state_path, state)
            return state

        changed = False
        for child in _children_for_wave(state, wave_index):
            if child.child_id in procs:
                changed = _update_child_activity(child) or changed

        finished: list[str] = []
        for child_id, (proc, _stack) in procs.items():
            if proc.poll() is not None:
                finished.append(child_id)
        for child_id in finished:
            proc, stack = procs.pop(child_id)
            child_state = _refresh_child_result(state.children, child_id)
            if child_state is None:
                child_state = next(item for item in state.children if item.child_id == child_id)
                child_state.status = "failed" if proc.returncode else "success"
                child_state.exit_code = proc.returncode
                child_state.finished_at = _utcnow()
            _update_child_activity(child_state)
            stack.close()
            changed = True

        if changed:
            save_swarm_state(state_path, state)
        time.sleep(0.2)

    return load_swarm_state(state_path)


def _write_child_patch(child: SwarmChildState) -> Path | None:
    patch_path = Path(child.patch_path)
    worktree = Path(child.worktree_path)
    # Record intent-to-add for untracked files so `git diff` emits them as
    # additions; without this, children that only create new files produce an
    # empty diff and are falsely rejected as "No diff to apply."
    _git("add", "-N", "--", ".", cwd=worktree)
    completed = _git("diff", "--binary", cwd=worktree)
    if completed.returncode != 0:
        child.acceptance_note = completed.stderr.strip() or "Failed to build patch."
        return None
    if not completed.stdout.strip():
        child.acceptance_note = "No diff to apply."
        return None
    patch_path.write_text(completed.stdout, encoding="utf-8")
    return patch_path


def _can_apply_patch(worktree: Path, patch_path: Path) -> bool:
    completed = _git("apply", "--check", "--3way", str(patch_path), cwd=worktree)
    return completed.returncode == 0


def _apply_patch_to_worktree(worktree: Path, patch_path: Path) -> None:
    completed = _git("apply", "--3way", str(patch_path), cwd=worktree)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git apply failed")


def _recreate_integration_worktree(state: SwarmRunState) -> Path:
    manager = SwarmWorktreeManager(
        repo_root=Path(state.repo_root),
        pool_root=Path(state.worktree_pool),
    )
    current = Path(state.integration_worktree)
    manager.remove_worktree(current)
    recreated = manager.create_worktree(
        run_id=state.run_id,
        child_id="integration",
        ref=state.integration_base_ref,
    )
    state.integration_worktree = str(recreated)
    return recreated


def _run_integration_validation(
    *,
    state: SwarmRunState,
    wave: SwarmWaveState,
    child: SwarmChildState,
    integration: Path,
) -> list[SwarmValidationCheck]:
    validation_dir = (
        Path(state.copied_spec_path).resolve().parent
        / "artifacts"
        / "waves"
        / f"wave-{wave.wave_index:02d}"
        / child.child_id
    )
    validation_dir.mkdir(parents=True, exist_ok=True)
    env = build_child_env(child, state)
    results = [_run_structural_validation(workspace_root=integration, run_dir=validation_dir, env=env)]
    for ordinal, command in enumerate(state.validation_commands, start=1):
        log_base = validation_dir / f"integration-validation-{ordinal:02d}"
        stdout_path = log_base.with_suffix(".stdout.log")
        stderr_path = log_base.with_suffix(".stderr.log")
        started = time.perf_counter()
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            proc = subprocess.run(
                command,
                cwd=integration,
                env=env,
                shell=True,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        results.append(
            SwarmValidationCheck(
                name=f"integration-validation-{ordinal}",
                command=command,
                passed=proc.returncode == 0,
                exit_code=proc.returncode,
                detail="pass" if proc.returncode == 0 else "failed",
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )
    return results


def _apply_readonly_wave(
    state: SwarmRunState,
    wave_children: list[SwarmChildState],
    wave: SwarmWaveState,
) -> bool:
    """Reduce a readonly wave: no patches, transplant, or integration validation.

    Children produced ``findings`` / ``answer`` / ``metric`` instead of a diff.
    The reducer (``union`` / ``vote`` / ``merge``) selects which to keep and may
    synthesize ``merged_output``; acceptance is recorded without mutating any
    integration worktree.
    """
    reducer = get_reducer(state.reducer_name or "merge")
    evaluation = reducer.reduce(wave_children, WaveContext(state=state, wave=wave))
    wave.evaluation = evaluation
    decision_map = {item.child_id: item for item in evaluation.decisions}
    accepted_set = set(evaluation.accepted_child_ids)
    accepted: list[str] = []
    rejected: list[str] = []
    for child in wave_children:
        decision = decision_map.get(child.child_id)
        note = decision.rationale if decision is not None and decision.rationale else ""
        if child.child_id in accepted_set:
            child.accepted = True
            child.acceptance_note = note or "Accepted (readonly candidate)."
            accepted.append(child.child_id)
        else:
            child.accepted = False
            child.acceptance_note = note or "Not selected by the reducer."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
    wave.accepted_child_ids = accepted
    wave.rejected_child_ids = rejected
    wave.primary_winner_child_id = accepted[0] if accepted else None
    wave.finished_at = _utcnow()
    state.convergence_status = evaluation.verdict
    state.convergence_summary = evaluation.summary
    state.next_wave_directives = list(evaluation.next_wave_directives)
    if evaluation.status == "completed":
        state.consecutive_evaluator_failures = 0
    else:
        state.consecutive_evaluator_failures += 1
    for child_id in accepted:
        if child_id not in state.accepted_child_ids:
            state.accepted_child_ids.append(child_id)
    if accepted:
        state.primary_winner_child_id = accepted[0]
        state.winner_child_id = accepted[0]
        wave.status = "applied"
        wave.summary = evaluation.summary or f"Accepted {len(accepted)} readonly candidate(s)."
    else:
        wave.status = "no-improvement"
        wave.summary = evaluation.summary or "No readonly candidate was accepted."
    _write_wave_evaluation_manifest(state, wave)
    _write_wave_manifest(state, wave)
    _write_run_acceptance_manifest(state)
    return bool(accepted)


def apply_wave_candidates(
    state: SwarmRunState,
    wave_children: list[SwarmChildState],
    wave: SwarmWaveState,
) -> bool:
    if state.exec_mode == "readonly":
        return _apply_readonly_wave(state, wave_children, wave)
    integration = Path(state.integration_worktree)
    ranked = rank_children(wave_children)
    # Write candidate patches before evaluation so the evaluator evidence
    # (patch_preview, digest-based duplicate hints) and the deterministic
    # fallback evaluator can read real diffs instead of nonexistent files.
    patch_paths: dict[str, Path | None] = {child.child_id: _write_child_patch(child) for child in ranked}
    reducer = get_reducer(state.reducer_name or "merge")
    evaluation = reducer.reduce(ranked, WaveContext(state=state, wave=wave))
    wave.evaluation = evaluation
    accepted: list[str] = []
    rejected: list[str] = []
    accepted_commits: list[SwarmAcceptedCommit] = []
    run_dir = Path(state.copied_spec_path).resolve().parent
    decision_map = {item.child_id: item for item in evaluation.decisions}
    ranked_map = {child.child_id: child for child in ranked}
    accepted_history_digests = _accepted_patch_digest_map(state.accepted_commits)
    ordered_ids: list[str] = []
    accepted_set: set[str] = set()
    for child_id in evaluation.candidate_order:
        if child_id in ranked_map and child_id not in ordered_ids:
            ordered_ids.append(child_id)
    for child in ranked:
        if child.child_id not in ordered_ids:
            ordered_ids.append(child.child_id)

    for child_id in ordered_ids:
        child = ranked_map[child_id]
        decision = decision_map.get(child.child_id)
        if decision is not None:
            accepted_conflicts = sorted(conflict for conflict in decision.conflicts_with if conflict in accepted_set)
            accepted_duplicates = sorted(duplicate for duplicate in decision.duplicates if duplicate in accepted_set)
        else:
            accepted_conflicts = []
            accepted_duplicates = []
        if accepted_duplicates or accepted_conflicts:
            child.accepted = False
            assert decision is not None
            if accepted_duplicates:
                related = ", ".join(accepted_duplicates)
                child.acceptance_note = (
                    decision.rationale
                    or f"Evaluator marked this candidate as a duplicate of accepted candidate(s): {related}."
                )
            else:
                related = ", ".join(accepted_conflicts)
                child.acceptance_note = (
                    decision.rationale
                    or f"Evaluator marked this candidate as conflicting with accepted candidate(s): {related}."
                )
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if decision is not None and decision.verdict == "reject":
            child.accepted = False
            child.acceptance_note = decision.rationale or "Evaluator rejected this candidate."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if decision is not None and decision.verdict == "defer":
            child.accepted = False
            child.acceptance_note = decision.rationale or "Deferred for a later wave."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if child.status != "success":
            child.accepted = False
            child.acceptance_note = "Child command did not succeed."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if not child.files_changed:
            child.accepted = False
            child.acceptance_note = "Child produced no code changes."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if any(not check.passed for check in child.validation_results):
            child.accepted = False
            child.acceptance_note = "Validation failed."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        patch_path = patch_paths.get(child.child_id)
        if patch_path is None:
            child.accepted = False
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        history_duplicates = sorted(accepted_history_digests.get(_patch_digest(patch_path), []))
        if history_duplicates:
            child.accepted = False
            child.acceptance_note = f"Rejected because this patch duplicates already accepted candidate(s): {', '.join(history_duplicates)}."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        if not _can_apply_patch(integration, patch_path):
            child.accepted = False
            child.acceptance_note = "Patch conflicted with the already accepted integration state."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        try:
            _apply_patch_to_worktree(integration, patch_path)
        except RuntimeError:
            integration = _recreate_integration_worktree(state)
            child.accepted = False
            child.acceptance_note = "Patch conflicted with the already accepted integration state."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        previous_base_ref = state.integration_base_ref
        integration_validation = _run_integration_validation(
            state=state,
            wave=wave,
            child=child,
            integration=integration,
        )
        wave.integration_validation_results.extend(integration_validation)
        if any(not check.passed for check in integration_validation):
            state.integration_base_ref = previous_base_ref
            integration = _recreate_integration_worktree(state)
            child.accepted = False
            child.acceptance_note = "Rejected after integration validation failed."
            rejected.append(child.child_id)
            wave.rejected_child_notes[child.child_id] = child.acceptance_note
            continue
        child.accepted = True
        child.acceptance_note = (
            decision.rationale if decision is not None and decision.rationale else "Applied to integration base."
        )
        state.integration_base_ref = _ensure_snapshot_commit(
            integration,
            f"[swarm] wave {wave.wave_index:02d} accept {child.child_id}",
        )
        patch_artifact = _artifact_ref(
            run_dir,
            patch_path,
            kind="patch",
            label=f"{child.child_id} patch",
            metadata={
                "wave_index": wave.wave_index,
                "child_id": child.child_id,
                "commit_ref": state.integration_base_ref,
            },
        )
        apply_commands = [
            f"git cherry-pick {state.integration_base_ref}",
            f"git apply {shlex.quote(str(patch_path))}",
        ]
        accepted_commit = SwarmAcceptedCommit(
            order=len(state.accepted_commits) + len(accepted_commits) + 1,
            child_id=child.child_id,
            commit_ref=state.integration_base_ref,
            summary=child.summary,
            files_changed=list(child.files_changed),
            patch_path=str(patch_path),
            score=child.score,
            artifacts=[patch_artifact],
            apply_commands=apply_commands,
        )
        child.accepted_commit_ref = accepted_commit.commit_ref
        child.accepted_order = accepted_commit.order
        child.export_artifacts = [patch_artifact]
        child.apply_commands = apply_commands
        accepted_commits.append(accepted_commit)
        _upsert_run_artifact(state, patch_artifact)
        accepted.append(child.child_id)
        accepted_set.add(child.child_id)
        patch_digest = _patch_digest(patch_path)
        if patch_digest:
            accepted_history_digests.setdefault(patch_digest, []).append(child.child_id)

    wave.accepted_child_ids = accepted
    wave.rejected_child_ids = rejected
    wave.accepted_commits = accepted_commits
    wave.primary_winner_child_id = accepted[0] if accepted else None
    wave.finished_at = _utcnow()
    state.convergence_status = evaluation.verdict
    state.convergence_summary = evaluation.summary
    state.next_wave_directives = list(evaluation.next_wave_directives)
    if evaluation.status == "completed":
        state.consecutive_evaluator_failures = 0
    else:
        state.consecutive_evaluator_failures += 1
    if accepted:
        for child_id in accepted:
            if child_id not in state.accepted_child_ids:
                state.accepted_child_ids.append(child_id)
        state.accepted_commits.extend(accepted_commits)
        state.primary_winner_child_id = accepted[0]
        state.winner_child_id = accepted[0]
        winner = next(child for child in ranked if child.child_id == accepted[0])
        state.ranking_notes = winner.score_breakdown
        wave.status = "applied"
        wave.summary = evaluation.summary or f"Accepted {len(accepted)} child patch(es)."
        _refresh_transplant_commands(state)
        _write_wave_evaluation_manifest(state, wave)
        _write_wave_manifest(state, wave)
        _write_run_acceptance_manifest(state)
        return True
    wave.status = "no-improvement"
    wave.summary = evaluation.summary or "No child patch was accepted in this wave."
    _refresh_transplant_commands(state)
    _write_wave_evaluation_manifest(state, wave)
    _write_wave_manifest(state, wave)
    _write_run_acceptance_manifest(state)
    return False


def _ensure_fitness_baseline(state: SwarmRunState) -> None:
    """Auto-measure the fitness baseline on the base snapshot before wave 1.

    No-op unless the run carries a ``FitnessSpec`` whose ``baseline`` is still
    ``"auto"``. The measured value replaces ``"auto"`` and is persisted with the
    run so every wave compares against the same frozen number.
    """
    spec = state.fitness_spec
    if spec is None or spec.baseline != "auto":
        return
    from lemoncrow.pro.capabilities.swarm.fitness import measure_baseline

    worktree = Path(state.integration_worktree or state.base_worktree or state.repo_root)
    try:
        spec.baseline = measure_baseline(spec, worktree)
        state.ranking_notes.append(f"Auto-measured fitness baseline = {spec.baseline:g} on the base snapshot.")
    except Exception as exc:  # noqa: BLE001 - a bad fitness must not wedge the coordinator
        state.limitations.append(f"Fitness baseline auto-measure failed: {exc}")


def launch_swarm_children(root: Path, state_path: Path) -> SwarmRunState:
    state = load_swarm_state(state_path)
    state.status = "running"
    state.coordinator_pid = os.getpid()
    _ensure_fitness_baseline(state)
    save_swarm_state(state_path, state)

    def _terminate(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _terminate)
    try:
        while True:
            state = load_swarm_state(state_path)
            if state.stop_requested:
                state.status = "stopped"
                state.stop_reason = state.stop_reason or "Stopped by user."
                save_swarm_state(state_path, state)
                break

            wave_index = state.current_wave + 1
            wave = _prepare_wave(state, root, wave_index)
            save_swarm_state(state_path, state)

            state = _run_wave_children(root, state_path, wave_index)
            wave = _latest_wave(state, wave_index)
            if state.stop_requested:
                state.status = "stopped"
                state.stop_reason = state.stop_reason or "Stopped by user."
                save_swarm_state(state_path, state)
                break

            wave_children = _children_for_wave(state, wave_index)
            accepted_any = apply_wave_candidates(state, wave_children, wave)
            save_swarm_state(state_path, state)

            if state.mode == "single":
                state.status = "success" if accepted_any else "failed"
                state.stop_reason = "Single-wave swarm completed."
                save_swarm_state(state_path, state)
                break

            if state.max_waves > 0 and wave_index >= state.max_waves:
                state.status = "success" if state.accepted_child_ids else "failed"
                state.stop_reason = f"Stopped after wave {wave_index}: reached max_waves={state.max_waves}."
                save_swarm_state(state_path, state)
                break

            if state.consecutive_evaluator_failures >= max(state.max_evaluator_failures, 1):
                state.status = "failed"
                state.stop_reason = (
                    f"Stopped after wave {wave_index}: evaluator failed "
                    f"{state.consecutive_evaluator_failures} consecutive times."
                )
                save_swarm_state(state_path, state)
                break

            if state.convergence_status == "blocked":
                state.status = "failed"
                state.stop_reason = f"Stopped after wave {wave_index}: evaluator marked the run blocked."
                save_swarm_state(state_path, state)
                break

            if state.convergence_status == "converged":
                state.status = "success"
                state.stop_reason = f"Stopped after wave {wave_index}: evaluator marked the run converged."
                save_swarm_state(state_path, state)
                break

            if accepted_any:
                continue

            if (
                wave.evaluation is not None
                and wave.evaluation.deferred_child_ids
                and state.convergence_status == "continue"
            ):
                continue

            state.status = "success"
            state.stop_reason = f"Stopped after wave {wave_index}: no accepted improvements."
            if state.convergence_summary:
                state.stop_reason = f"Stopped after wave {wave_index}: {state.convergence_summary}"
            save_swarm_state(state_path, state)
            break

        state = load_swarm_state(state_path)
        if not state.keep_worktrees:
            cleanup_swarm_run(state)
            state.ranking_notes.append("Removed child worktrees because --cleanup was enabled.")
            save_swarm_state(state_path, state)
        return state
    except KeyboardInterrupt:
        stop_swarm_run(root=root, state_path=state_path, cleanup=False)
        state = load_swarm_state(state_path)
        state.status = "stopped"
        state.stop_reason = "Interrupted."
        save_swarm_state(state_path, state)
        raise
    except Exception as exc:  # noqa: BLE001
        # Git/worktree failures must not leave the run wedged in "running" with
        # leaked worktrees. Finalize state and reclaim worktrees instead.
        state = load_swarm_state(state_path)
        for child in state.children:
            if child.status == "running":
                _kill_pid(child.pid)
                child.status = "failed"
                child.finished_at = _utcnow()
        state.status = "failed"
        state.stop_reason = f"Coordinator failed: {exc}"
        save_swarm_state(state_path, state)
        with contextlib.suppress(Exception):
            cleanup_swarm_run(state)
        state.ranking_notes.append("Removed swarm worktrees after coordinator failure.")
        save_swarm_state(state_path, state)
        return state
    finally:
        signal.signal(signal.SIGTERM, old_term)


def stop_swarm_run(*, root: Path, state_path: Path, cleanup: bool) -> SwarmRunState:
    state = load_swarm_state(state_path)
    state.stop_requested = True
    if not state.stop_reason:
        state.stop_reason = "Stop requested by user."
    for child in state.children:
        if child.status == "running":
            _kill_pid(child.pid)
            child.status = "stopped"
            child.finished_at = _utcnow()
    if state.coordinator_pid and state.coordinator_pid != os.getpid():
        _kill_pid(state.coordinator_pid)
    state.status = "stopped"
    save_swarm_state(state_path, state)
    if cleanup:
        cleanup_swarm_run(state)
        state.ranking_notes.append("Removed swarm worktrees after stop/cleanup.")
        save_swarm_state(state_path, state)
    return state


def cleanup_swarm_run(state: SwarmRunState) -> None:
    manager = SwarmWorktreeManager(
        repo_root=Path(state.repo_root),
        pool_root=Path(state.worktree_pool),
    )
    # A single locked or contended worktree makes `git worktree remove --force`
    # exit non-zero; suppress per-worktree so one failure does not leak the rest.
    for child in state.children:
        with contextlib.suppress(Exception):
            manager.remove_worktree(Path(child.worktree_path))
    if state.integration_worktree:
        with contextlib.suppress(Exception):
            manager.remove_worktree(Path(state.integration_worktree))


def run_child_once(state_path: Path, child_id: str) -> SwarmChildState:
    state = load_swarm_state(state_path)
    child = next((item for item in state.children if item.child_id == child_id), None)
    if child is None:
        raise RuntimeError(f"unknown child id: {child_id}")

    child.status = "running"
    child.started_at = _utcnow()
    env = build_child_env(child, state)
    ledger = RunLedger(
        session_id=f"{state.run_id}-{child.child_id}",
        agent="swarm-child",
        root=Path(child.lemoncrow_root),
        task=f"Swarm child {child.child_id}",
        domain="swarm",
    )
    stdout_path = Path(child.stdout_path)
    stderr_path = Path(child.stderr_path)
    metadata_path = Path(child.metadata_path)
    result_path = Path(child.result_path)
    worktree = Path(child.worktree_path)
    active_proc: subprocess.Popen[str] | None = None

    def _terminate(signum: int, _frame: object) -> None:
        nonlocal active_proc
        if active_proc is not None and active_proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(active_proc.pid, signum)
        raise KeyboardInterrupt

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)
    started = time.perf_counter()
    try:
        command = _expand_command_tokens(child, state.child_command)
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            active_proc = subprocess.Popen(
                command,
                cwd=worktree,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                start_new_session=True,
            )
            exit_code = active_proc.wait()
        ledger.record_command(
            " ".join(shlex.quote(token) for token in command),
            ok=exit_code == 0,
        )
        validation_results: list[SwarmValidationCheck] = [
            _run_structural_validation(workspace_root=worktree, run_dir=Path(child.run_dir), env=env)
        ]
        for index, validate_command in enumerate(state.validation_commands, start=1):
            validation_results.append(
                _run_validation_command(
                    child=child,
                    command=validate_command,
                    env=env,
                    cwd=worktree,
                    ordinal=index,
                    ledger=ledger,
                )
            )

        metadata = _load_child_metadata(metadata_path)
        token_count, cost_usd = _collect_cost_and_tokens(Path(child.lemoncrow_root))
        files_changed = _relative_git_status(worktree)
        summary = str(metadata.get("summary") or _summarize_output(stdout_path, stderr_path))
        error = str(metadata.get("error") or "")
        silent_noop = (
            state.exec_mode == "edit"
            and exit_code == 0
            and not files_changed
            and summary == "No summary emitted."
            and not error
        )
        result = child.model_copy(
            update={
                "status": "failed" if silent_noop else ("success" if exit_code == 0 else "failed"),
                "exit_code": exit_code,
                "files_changed": files_changed,
                "validation_results": _merge_validation_results(
                    validation_results,
                    metadata.get("validation_results"),
                ),
                "summary": (
                    "Child runner exited successfully without output or code changes." if silent_noop else summary
                ),
                "error": ("Child runner exited successfully without output or code changes." if silent_noop else error),
                "findings": _coerce_findings(metadata.get("findings")),
                "answer": str(metadata.get("answer") or ""),
                "metric": _coerce_float_or_none(metadata.get("metric")),
                "token_count": _coerce_int(metadata.get("token_count"), token_count),
                "cost_usd": _coerce_float(metadata.get("cost_usd"), cost_usd),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "finished_at": _utcnow(),
            }
        )
        _update_child_activity(result)
        score, score_breakdown = _score_child(result)
        result.score = score
        result.score_breakdown = score_breakdown
        ledger.close("complete" if result.status == "success" else "failed")
        ledger.persist(Path(child.lemoncrow_root))
        _write_json(result_path, result.model_dump(mode="json"))
        return result
    except KeyboardInterrupt:
        child.status = "stopped"
        child.finished_at = _utcnow()
        child.duration_seconds = round(time.perf_counter() - started, 3)
        child.error = "Interrupted."
        _update_child_activity(child)
        ledger.close("partial")
        ledger.persist(Path(child.lemoncrow_root))
        _write_json(result_path, child.model_dump(mode="json"))
        return child
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _run_validation_command(
    *,
    child: SwarmChildState,
    command: str,
    env: dict[str, str],
    cwd: Path,
    ordinal: int,
    ledger: RunLedger,
) -> SwarmValidationCheck:
    log_base = Path(child.run_dir) / f"validation-{ordinal:02d}"
    stdout_path = log_base.with_suffix(".stdout.log")
    stderr_path = log_base.with_suffix(".stderr.log")
    started = time.perf_counter()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        proc = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            shell=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    duration = round(time.perf_counter() - started, 3)
    ledger.record_command(command, ok=proc.returncode == 0)
    return SwarmValidationCheck(
        name=f"validation-{ordinal}",
        command=command,
        passed=proc.returncode == 0,
        exit_code=proc.returncode,
        detail="pass" if proc.returncode == 0 else "failed",
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        duration_seconds=duration,
    )


def format_swarm_summary(state: SwarmRunState) -> str:
    current_wave = next((wave for wave in state.waves if wave.wave_index == state.current_wave), None)
    planned_runs = current_wave.planned_runs if current_wave is not None else len(state.accepted_child_ids)
    planned_total = current_wave.max_runs if current_wave is not None else len(state.children)
    running_children = sum(1 for child in state.children if child.status == "running")
    failed_children = sum(1 for child in state.children if child.status == "failed")

    evaluator_label: str = state.evaluator_backend
    if state.evaluator_model:
        evaluator_label += f" ({state.evaluator_model})"

    lines = [
        f"  run_id:      {state.run_id}",
        f"  status:      {state.status}",
        f"  mode:        {state.mode or 'single'}",
        f"  runner:      {state.runner_name}{' (' + state.runner_model + ')' if state.runner_model else ''}",
        f"  evaluator:   {evaluator_label}",
        f"  wave:        {state.current_wave}",
        f"  children:    {len(state.accepted_child_ids)} accepted / {len(state.children)} total / {failed_children} failed ({running_children} live)",
        f"  planned:     {planned_runs}/{planned_total}",
        f"  created:     {state.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    primary_winner = state.primary_winner_child_id or state.winner_child_id
    if primary_winner is None and state.accepted_child_ids:
        primary_winner = state.accepted_child_ids[0]
    if primary_winner:
        lines.append(f"  winner:      {primary_winner}")

    if state.base_ref:
        lines.append(f"  base:        {state.base_ref}")
    if state.integration_base_ref:
        lines.append(f"  int_base:    {state.integration_base_ref}")
    if state.integration_worktree:
        lines.append(f"  worktree:    {state.integration_worktree}")

    if state.stop_reason:
        lines.append(f"  stopped:     {state.stop_reason}")
    if state.convergence_status and state.convergence_status != "continue":
        lines.append(f"  converge:    {state.convergence_status}")
    if state.convergence_summary:
        lines.append(f"  converge_notes: {state.convergence_summary}")

    if state.coordinator_log_path:
        lines.append(f"  log:         {state.coordinator_log_path}")

    if state.accepted_commits:
        lines.append("")
        lines.append("  ACCEPTED COMMITS:")
        for accepted in state.accepted_commits:
            header = f"    {accepted.order}. {accepted.child_id} -> {accepted.commit_ref[:8]}"
            if accepted.score is not None:
                header += f" (score: {accepted.score:.1f})"
            lines.append(header)
            if accepted.summary:
                for s_line in accepted.summary.strip().splitlines():
                    lines.append(f"      {s_line}")
            if accepted.files_changed:
                lines.append(f"      files: {', '.join(accepted.files_changed)}")

    # Group children by wave
    children_by_wave: dict[int, list[SwarmChildState]] = {}
    for child in state.children:
        children_by_wave.setdefault(child.wave_index, []).append(child)

    if children_by_wave:
        lines.append("")
        lines.append("  WAVES & CHILDREN:")
        for wave_idx in sorted(children_by_wave.keys()):
            wave_state = next((w for w in state.waves if w.wave_index == wave_idx), None)
            wave_info = f"    Wave {wave_idx}"
            if wave_state:
                wave_info += f" [{wave_state.status}]"
                if wave_state.summary:
                    wave_info += f": {wave_state.summary}"
            lines.append(wave_info)

            for child in children_by_wave[wave_idx]:
                status_icon = (
                    "\u2713" if child.accepted else "\u2717" if child.status in ("failed", "stopped") else "\u25cf"
                )
                status_label: str = child.status
                if child.status == "running" and child.current_activity:
                    status_label = f"running ({child.current_activity})"

                info = f"      {status_icon} {child.child_id:<15} {status_label:<15}"
                if child.accepted_commit_ref:
                    info += f" commit={child.accepted_commit_ref[:8]}"
                if child.score is not None:
                    info += f" score={child.score:.1f}"
                lines.append(info)

                detail = child.summary or child.error or child.acceptance_note
                if detail:
                    for d_line in detail.strip().splitlines():
                        capped = d_line if len(d_line) < 120 else d_line[:117] + "..."
                        lines.append(f"        {capped}")

                attach = resolve_swarm_child_attach(state.runner_name, child.worktree_path)
                if attach is not None:
                    lines.append(f"        attach: {attach['attach_command']}  (session {attach['session_id']})")

    if state.next_wave_directives:
        lines.append("")
        lines.append("  NEXT WAVE DIRECTIVES:")
        for directive in state.next_wave_directives:
            lines.append(f"    - {directive}")

    if state.transplant_commands:
        lines.append("")
        lines.append("  TRANSPLANT COMMANDS:")
        for command in state.transplant_commands:
            lines.append(f"    {command}")

    return "\n".join(lines)


def discover_repo_root(cwd: Path) -> Path:
    return git_repo_root(cwd)
