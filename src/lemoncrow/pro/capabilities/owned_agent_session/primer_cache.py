"""Workspace-verified cache for deterministic task primers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.owned_agent_session.task_primer import build_task_primer

_MAX_ENTRIES = 128


@dataclass(frozen=True)
class PrimerResult:
    text: str
    hit: bool
    fingerprint: str
    optimization_mode: str = "shadow"
    base_primer_skipped: bool = False
    evidence_hits: int = 0
    invalidated_evidence: int = 0
    evidence_applied: bool = False
    local_retrieval_invoked: bool = False
    local_retrieval_cache_hit: bool = False
    local_retrieval_packet_ready: bool = False
    local_retrieval_applied: bool = False
    local_retrieval_model_calls: int = 0
    local_retrieval_turns: int = 0
    local_retrieval_spans: int = 0
    local_retrieval_confidence: float = 0.0
    local_retrieval_reason: str = ""

    def optimization_metadata(self) -> dict[str, Any]:
        return {
            "optimization_mode": self.optimization_mode,
            "base_primer_skipped": self.base_primer_skipped,
            "evidence_hits": self.evidence_hits,
            "invalidated_evidence": self.invalidated_evidence,
            "evidence_applied": self.evidence_applied,
            "local_retrieval_invoked": self.local_retrieval_invoked,
            "local_retrieval_cache_hit": self.local_retrieval_cache_hit,
            "local_retrieval_packet_ready": self.local_retrieval_packet_ready,
            "local_retrieval_applied": self.local_retrieval_applied,
            "local_retrieval_model_calls": self.local_retrieval_model_calls,
            "local_retrieval_turns": self.local_retrieval_turns,
            "local_retrieval_spans": self.local_retrieval_spans,
            "local_retrieval_confidence": round(self.local_retrieval_confidence, 4),
            "local_retrieval_reason": self.local_retrieval_reason[:128],
        }


def _git_output(workspace: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=workspace,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def workspace_fingerprint(workspace: Path) -> str:
    """Fingerprint HEAD plus dirty file metadata without reading file contents."""
    workspace = workspace.resolve()
    head = _git_output(workspace, "rev-parse", "HEAD")
    status = _git_output(workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    parts = [str(workspace), head, status]
    if status:
        for entry in status.split("\0")[:500]:
            relative = entry[3:] if len(entry) > 3 else ""
            if " -> " in relative:
                relative = relative.split(" -> ", 1)[1]
            path = workspace / relative
            try:
                stat = path.stat()
                parts.append(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                parts.append(f"{relative}:missing")
    elif not head:
        seen = 0
        for path in sorted(workspace.rglob("*")):
            if seen >= 500:
                break
            relative_path = path.relative_to(workspace)
            if not path.is_file() or any(part.startswith(".") for part in relative_path.parts):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            parts.append(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}")
            seen += 1
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"entries": {}}
    return data if isinstance(data, dict) and isinstance(data.get("entries"), dict) else {"entries": {}}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="primer-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def cached_task_primer(
    task: str,
    workspace: Path,
    store_root: Path,
    *,
    max_chars: int = 12_000,
    retrieval_mode: str | None = None,
    local_retrieval_model: str | None = None,
    optimization_mode: str | None = None,
) -> PrimerResult:
    """Return a deterministic primer plus policy-gated verified evidence."""
    from lemoncrow.pro.capabilities.optimization.runtime_decisions import normalize_optimization_mode

    mode = normalize_optimization_mode(optimization_mode)
    optimization_enabled = mode != "off"
    apply_optimized_evidence = mode == "enforce"
    workspace = workspace.resolve()
    from lemoncrow.pro.capabilities.optimization.local_retrieval import (
        normalize_retrieval_mode,
        task_has_explicit_source_path,
    )

    retrieval_policy = normalize_retrieval_mode(retrieval_mode)
    base_primer_skipped = retrieval_policy != "force" and task_has_explicit_source_path(task)
    if base_primer_skipped:
        fingerprint = hashlib.sha256(f"{workspace}\0explicit-path-primer-skipped".encode()).hexdigest()
        hit = False
        base_text = ""
    else:
        fingerprint = workspace_fingerprint(workspace)
        key_material = f"{workspace}\0{' '.join(task.split()).lower()}\0{max_chars}"
        key = hashlib.sha256(key_material.encode()).hexdigest()
        cache_path = store_root / "cache" / "task_primers.json"
        data = _load(cache_path)
        entry = data["entries"].get(key)
        hit = isinstance(entry, dict) and entry.get("fingerprint") == fingerprint
        if hit:
            base_text = str(entry.get("text", ""))
        else:
            base_text = build_task_primer(task, workspace, max_chars=max_chars)
            data["entries"][key] = {"fingerprint": fingerprint, "text": base_text}
            while len(data["entries"]) > _MAX_ENTRIES:
                data["entries"].pop(next(iter(data["entries"])))
            try:
                _save(cache_path, data)
            except OSError:
                pass

    evidence = None
    if optimization_enabled:
        try:
            from lemoncrow.pro.capabilities.optimization.evidence_reuse import load_verified_evidence

            evidence = load_verified_evidence(
                store_root,
                workspace,
                task=task,
                max_chars=max(2_000, max_chars // 2),
            )
        except Exception:
            evidence = None
    evidence_text = evidence.text if evidence is not None else ""

    local = None
    local_reason = "optimization mode off" if not optimization_enabled else ""
    if optimization_enabled and (evidence is None or evidence.hit_count == 0):
        try:
            from lemoncrow.pro.capabilities.optimization.local_retrieval import cached_local_evidence_packet

            local = cached_local_evidence_packet(
                store_root,
                task,
                workspace,
                mode=retrieval_mode,
                model=local_retrieval_model or os.environ.get("LEMONCROW_LOCAL_RETRIEVAL_MODEL", ""),
                max_chars=max(2_000, max_chars // 2),
            )
            local_reason = local.reason
        except Exception:
            local_reason = "local retrieval unavailable"
    elif optimization_enabled:
        local_reason = "verified evidence already available"
    local_text = local.text if local is not None else ""

    evidence_applied = apply_optimized_evidence and bool(evidence_text)
    local_packet_ready = bool(local_text)
    local_applied = apply_optimized_evidence and local_packet_ready
    extras = [
        item for item, applied in ((evidence_text, evidence_applied), (local_text, local_applied)) if item and applied
    ]
    if extras:
        extras_text = "\n\n".join(extras)
        base_budget = max(2_000, max_chars - len(extras_text) - 2)
        text = f"{base_text[:base_budget]}\n\n{extras_text}"[:max_chars]
    else:
        text = base_text
    return PrimerResult(
        text,
        bool(hit),
        fingerprint,
        optimization_mode=mode,
        base_primer_skipped=base_primer_skipped,
        evidence_hits=evidence.hit_count if evidence is not None else 0,
        invalidated_evidence=evidence.invalidated_count if evidence is not None else 0,
        evidence_applied=evidence_applied,
        local_retrieval_invoked=bool(local and local.invoked),
        local_retrieval_cache_hit=bool(local and local.cache_hit),
        local_retrieval_packet_ready=local_packet_ready,
        local_retrieval_applied=local_applied,
        local_retrieval_model_calls=local.model_calls if local is not None else 0,
        local_retrieval_turns=local.turns if local is not None else 0,
        local_retrieval_spans=local.span_count if local is not None else 0,
        local_retrieval_confidence=local.confidence if local is not None else 0.0,
        local_retrieval_reason=local_reason,
    )


__all__ = ["PrimerResult", "cached_task_primer", "workspace_fingerprint"]
