"""Verified, source-hashed cross-session reuse for deterministic retrieval evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import lemoncrow

_ALLOWED_TOOLS = frozenset({"code_search", "explore", "grep", "read", "symbols"})
_DEPENDENCY_FILES = (
    "Cargo.lock",
    "go.mod",
    "go.sum",
    "package-lock.json",
    "pnpm-lock.yaml",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "yarn.lock",
)
_MAX_RESULT_CHARS = 32_000
_MAX_SOURCE_FILES = 32
_MAX_SOURCE_BYTES = 2_000_000
_MAX_PACKETS = 512
_RESULT_PATH_RE = re.compile(r"(?m)^([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+):L?(\d+)(?:-L?(\d+))?")
_RANGE_RE = re.compile(r"^(.*?):L(\d+)(?:-L?(\d+))?$")
_VOLATILE_MARKERS = (
    "diff --git ",
    "traceback (most recent call last)",
    "short test summary info",
    "npm err!",
)


@dataclass(frozen=True)
class VerificationReceipt:
    kind: str
    command: str
    ok: bool
    output_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        normalized_command = " ".join(self.command.split())
        return {
            "kind": self.kind[:64],
            "command_sha256": hashlib.sha256(normalized_command.encode()).hexdigest(),
            "ok": bool(self.ok),
            "output_hash": self.output_hash[:128],
        }


@dataclass(frozen=True)
class EvidenceStageResult:
    staged: bool
    packet_key: str = ""
    reason: str = ""


@dataclass(frozen=True)
class EvidenceReuseResult:
    text: str
    hit_count: int
    invalidated_count: int


def _task_fingerprint(task: str) -> str:
    normalized = " ".join(task.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def dependency_fingerprint(workspace: Path) -> str:
    parts = [f"python:{sys.version_info.major}.{sys.version_info.minor}"]
    for name in _DEPENDENCY_FILES:
        path = workspace / name
        if not path.is_file():
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        parts.append(f"{name}:{hashlib.sha256(content).hexdigest()}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _workspace_fingerprint(workspace: Path) -> str:
    from lemoncrow.pro.capabilities.owned_agent_session.primer_cache import workspace_fingerprint

    return workspace_fingerprint(workspace)


def _tool_version() -> str:
    return str(getattr(lemoncrow, "__version__", "unknown"))


def _database_path(root: Path | str) -> Path:
    return Path(root).expanduser().resolve() / "cache" / "evidence-reuse.sqlite3"


def _connect(root: Path | str) -> sqlite3.Connection:
    path = _database_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS evidence_packets (
            packet_key TEXT PRIMARY KEY,
            task_fingerprint TEXT NOT NULL,
            workspace TEXT NOT NULL,
            workspace_fingerprint TEXT NOT NULL,
            dependency_fingerprint TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            args_fingerprint TEXT NOT NULL,
            result_text TEXT NOT NULL,
            result_hash TEXT NOT NULL,
            source_manifest TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            tool_version TEXT NOT NULL,
            verified_at REAL,
            receipt_json TEXT,
            invalidated_reason TEXT
        )
        """)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence_task ON evidence_packets(task_fingerprint, workspace, verified_at)"
    )
    connection.commit()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return connection


def _ttl_days() -> int:
    try:
        return max(1, int(os.environ.get("LEMONCROW_EVIDENCE_TTL_DAYS", "14")))
    except ValueError:
        return 14


def _normalized_args(args: dict[str, Any]) -> str:
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _candidate_specs(args: dict[str, Any], result: str) -> list[tuple[str, int | None, int | None]]:
    specs: list[tuple[str, int | None, int | None]] = []

    def add(value: Any, range_value: Any = None) -> None:
        if not isinstance(value, str) or not value:
            return
        match = _RANGE_RE.fullmatch(value)
        if match:
            specs.append((match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))))
            return
        start = end = None
        if isinstance(range_value, str):
            range_match = re.fullmatch(r"L?(\d+)(?:-L?(\d+))?", range_value)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2) or range_match.group(1))
        specs.append((value, start, end))

    add(args.get("path") or args.get("file_path"), args.get("range"))
    paths = args.get("paths")
    if isinstance(paths, str):
        add(paths)
    elif isinstance(paths, list):
        for item in paths:
            add(item)
    files = args.get("files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, str):
                add(item)
            elif isinstance(item, dict):
                add(item.get("path"), item.get("range"))

    for match in _RESULT_PATH_RE.finditer(result):
        specs.append((match.group(1), int(match.group(2)), int(match.group(3) or match.group(2))))

    combined: dict[str, tuple[int | None, int | None]] = {}
    for raw_path, start, end in specs:
        previous = combined.get(raw_path)
        if previous is None:
            combined[raw_path] = (start, end)
            continue
        starts = [value for value in (previous[0], start) if value is not None]
        ends = [value for value in (previous[1], end) if value is not None]
        combined[raw_path] = (min(starts) if starts else None, max(ends) if ends else None)
    return [(path, *line_range) for path, line_range in list(combined.items())[:_MAX_SOURCE_FILES]]


def _source_manifest(workspace: Path, args: dict[str, Any], result: str) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for raw_path, start, end in _candidate_specs(args, result):
        candidate = Path(raw_path)
        path = candidate if candidate.is_absolute() else workspace / candidate
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(workspace.resolve())
            stat = resolved.stat()
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or stat.st_size > _MAX_SOURCE_BYTES:
            continue
        try:
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except OSError:
            continue
        manifest.append(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "start_line": start,
                "end_line": end,
            }
        )
    return manifest


def _validate_manifest(workspace: Path, manifest: list[dict[str, Any]]) -> bool:
    if not manifest:
        return False
    for item in manifest:
        path = workspace / str(item.get("path") or "")
        try:
            resolved = path.resolve()
            resolved.relative_to(workspace.resolve())
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        except (OSError, ValueError):
            return False
        if digest != item.get("sha256"):
            return False
    return True


def stage_evidence_result(
    root: Path | str,
    workspace: Path,
    *,
    task: str,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    now: datetime | None = None,
) -> EvidenceStageResult:
    """Stage deterministic evidence; mutation/model/shell material is never accepted."""
    if tool_name not in _ALLOWED_TOOLS:
        return EvidenceStageResult(False, reason="tool is not deterministic retrieval")
    if not result or len(result) > _MAX_RESULT_CHARS:
        return EvidenceStageResult(False, reason="result is empty or over the evidence limit")
    lowered = result.lower()
    if any(marker in lowered for marker in _VOLATILE_MARKERS):
        return EvidenceStageResult(False, reason="volatile output is not reusable evidence")
    workspace = workspace.resolve()
    manifest = _source_manifest(workspace, args, result)
    if not manifest:
        return EvidenceStageResult(False, reason="no source file hash could be proven")

    args_rendered = _normalized_args(args)
    args_fingerprint = hashlib.sha256(args_rendered.encode()).hexdigest()
    result_hash = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()
    task_fingerprint = _task_fingerprint(task)
    packet_material = json.dumps(
        {
            "task": task_fingerprint,
            "tool": tool_name,
            "args": args_fingerprint,
            "sources": manifest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    packet_key = hashlib.sha256(packet_material.encode()).hexdigest()
    timestamp = (now or datetime.now(UTC)).timestamp()
    expires_at = timestamp + timedelta(days=_ttl_days()).total_seconds()
    try:
        with _connect(root) as connection:
            connection.execute(
                """
                INSERT INTO evidence_packets(
                    packet_key, task_fingerprint, workspace, workspace_fingerprint,
                    dependency_fingerprint, tool_name, args_fingerprint, result_text,
                    result_hash, source_manifest, created_at, expires_at, tool_version,
                    verified_at, receipt_json, invalidated_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                ON CONFLICT(packet_key) DO UPDATE SET
                    workspace_fingerprint = excluded.workspace_fingerprint,
                    dependency_fingerprint = excluded.dependency_fingerprint,
                    result_text = excluded.result_text,
                    result_hash = excluded.result_hash,
                    source_manifest = excluded.source_manifest,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at,
                    tool_version = excluded.tool_version,
                    verified_at = NULL,
                    receipt_json = NULL,
                    invalidated_reason = NULL
                """,
                (
                    packet_key,
                    task_fingerprint,
                    str(workspace),
                    _workspace_fingerprint(workspace),
                    dependency_fingerprint(workspace),
                    tool_name,
                    args_fingerprint,
                    result,
                    result_hash,
                    json.dumps(manifest, separators=(",", ":")),
                    timestamp,
                    expires_at,
                    _tool_version(),
                ),
            )
            connection.execute("DELETE FROM evidence_packets WHERE expires_at <= ?", (timestamp,))
            connection.execute(
                """
                DELETE FROM evidence_packets WHERE packet_key IN (
                    SELECT packet_key FROM evidence_packets
                    ORDER BY created_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (_MAX_PACKETS,),
            )
    except sqlite3.Error:
        return EvidenceStageResult(False, reason="evidence store unavailable")
    return EvidenceStageResult(True, packet_key=packet_key, reason="staged pending verification")


def finalize_task_evidence(
    root: Path | str,
    workspace: Path,
    *,
    task: str,
    receipt: VerificationReceipt,
    now: datetime | None = None,
) -> int:
    if not receipt.ok:
        return 0
    workspace = workspace.resolve()
    task_fingerprint = _task_fingerprint(task)
    timestamp = (now or datetime.now(UTC)).timestamp()
    finalized = 0
    try:
        with _connect(root) as connection:
            rows = connection.execute(
                """
                SELECT packet_key, source_manifest FROM evidence_packets
                WHERE task_fingerprint = ? AND workspace = ? AND verified_at IS NULL
                """,
                (task_fingerprint, str(workspace)),
            ).fetchall()
            if not rows:
                return 0
            current_workspace = _workspace_fingerprint(workspace)
            current_dependencies = dependency_fingerprint(workspace)
            for packet_key, manifest_json in rows:
                try:
                    manifest = json.loads(str(manifest_json))
                except ValueError:
                    connection.execute(
                        "UPDATE evidence_packets SET invalidated_reason = ? WHERE packet_key = ?",
                        ("corrupt source manifest", packet_key),
                    )
                    continue
                if not isinstance(manifest, list) or not _validate_manifest(workspace, manifest):
                    connection.execute(
                        "UPDATE evidence_packets SET invalidated_reason = ? WHERE packet_key = ?",
                        ("source hash changed before verification", packet_key),
                    )
                    continue
                connection.execute(
                    """
                    UPDATE evidence_packets SET
                        workspace_fingerprint = ?,
                        dependency_fingerprint = ?,
                        verified_at = ?,
                        receipt_json = ?,
                        invalidated_reason = NULL
                    WHERE packet_key = ?
                    """,
                    (
                        current_workspace,
                        current_dependencies,
                        timestamp,
                        json.dumps(receipt.to_dict(), separators=(",", ":")),
                        packet_key,
                    ),
                )
                finalized += 1
    except sqlite3.Error:
        return 0
    return finalized


def load_verified_evidence(
    root: Path | str,
    workspace: Path,
    *,
    task: str,
    max_chars: int = 12_000,
    now: datetime | None = None,
) -> EvidenceReuseResult:
    workspace = workspace.resolve()
    task_fingerprint = _task_fingerprint(task)
    timestamp = (now or datetime.now(UTC)).timestamp()
    sections: list[str] = []
    invalidated = 0
    hits = 0
    try:
        with _connect(root) as connection:
            rows = connection.execute(
                """
                SELECT packet_key, tool_name, result_text, result_hash, source_manifest,
                       workspace_fingerprint, dependency_fingerprint, tool_version, receipt_json
                FROM evidence_packets
                WHERE task_fingerprint = ? AND workspace = ? AND verified_at IS NOT NULL
                  AND expires_at > ?
                ORDER BY verified_at DESC LIMIT 16
                """,
                (task_fingerprint, str(workspace), timestamp),
            ).fetchall()
            if not rows:
                return EvidenceReuseResult("", 0, 0)
            current_workspace = _workspace_fingerprint(workspace)
            current_dependencies = dependency_fingerprint(workspace)
            for row in rows:
                packet_key, tool_name, result, result_hash, manifest_json, workspace_hash, deps, version, receipt = row
                reason = ""
                try:
                    manifest = json.loads(str(manifest_json))
                    receipt_data = json.loads(str(receipt))
                except (TypeError, ValueError):
                    manifest = []
                    receipt_data = {}
                    reason = "corrupt packet"
                if not reason and workspace_hash != current_workspace:
                    reason = "workspace fingerprint changed"
                if not reason and deps != current_dependencies:
                    reason = "dependency fingerprint changed"
                if not reason and version != _tool_version():
                    reason = "tool version changed"
                if not reason and hashlib.sha256(str(result).encode()).hexdigest() != result_hash:
                    reason = "result hash mismatch"
                if not reason and (not isinstance(manifest, list) or not _validate_manifest(workspace, manifest)):
                    reason = "source hash changed"
                if reason:
                    invalidated += 1
                    connection.execute(
                        "UPDATE evidence_packets SET invalidated_reason = ? WHERE packet_key = ?",
                        (reason, packet_key),
                    )
                    continue

                source_labels = []
                for item in manifest:
                    label = str(item.get("path") or "")
                    start = item.get("start_line")
                    end = item.get("end_line")
                    if start:
                        label += f":L{start}" + (f"-L{end}" if end and end != start else "")
                    source_labels.append(label)
                receipt_kind = str(receipt_data.get("kind") or "workspace_revalidation")[:64]
                receipt_hash = str(receipt_data.get("command_sha256") or "")[:12]
                receipt_label = f"{receipt_kind} receipt" + (f" {receipt_hash}" if receipt_hash else "")
                sections.append(
                    f"### {tool_name}: {', '.join(source_labels)}\n" f"Verified by: {receipt_label}\n\n{result!s}"
                )
                hits += 1
                if sum(len(section) for section in sections) >= max_chars:
                    break
    except sqlite3.Error:
        return EvidenceReuseResult("", 0, 0)

    if not sections:
        return EvidenceReuseResult("", 0, invalidated)
    text = (
        "## Verified cross-session evidence\n"
        "Source and dependency hashes match the current workspace. Treat this as read-only evidence; "
        "never replay a patch from it.\n\n" + "\n\n".join(sections)
    )
    return EvidenceReuseResult(text[:max_chars], hits, invalidated)


__all__ = [
    "EvidenceReuseResult",
    "EvidenceStageResult",
    "VerificationReceipt",
    "dependency_fingerprint",
    "finalize_task_evidence",
    "load_verified_evidence",
    "stage_evidence_result",
]
