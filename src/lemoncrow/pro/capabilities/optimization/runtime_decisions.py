"""Local, redacted decision traces for the LemonCode optimization control plane."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

OptimizationMode = Literal["off", "shadow", "enforce"]
_SCHEMA_VERSION = 1
_MAX_DECISIONS = 256
_MAX_PROVIDER_CALLS = 256
_WRITE_LOCK = threading.Lock()
_SENSITIVE_KEYS = frozenset(
    {
        "args",
        "arguments",
        "command",
        "content",
        "context",
        "diff",
        "file",
        "files",
        "path",
        "paths",
        "prompt",
        "query",
        "result",
        "task",
        "task_text",
        "text",
    }
)


def normalize_optimization_mode(value: str | None) -> OptimizationMode:
    normalized = (value or os.environ.get("LEMONCROW_OPTIMIZATION_MODE", "shadow")).strip().lower()
    if normalized in {"off", "disabled", "legacy"}:
        return "off"
    if normalized in {"on", "enabled", "enforce"}:
        return "enforce"
    return "shadow"


def runtime_decision_trace_path(root: Path | str) -> Path:
    return Path(root).expanduser().resolve() / "optimization" / "runtime-decisions.jsonl"


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:20]


def _safe_value(value: Any, *, key: str = "") -> Any:
    if key.lower() in _SENSITIVE_KEYS:
        return {"redacted_sha256": _fingerprint(str(value))}
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, Mapping):
        return {
            str(item_key)[:64]: _safe_value(item_value, key=str(item_key)) for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:64]]
    return str(value)[:256]


class OptimizationTraceRecorder:
    """Collect one owned-runtime turn and append one redacted JSONL record."""

    def __init__(
        self,
        root: Path | str,
        *,
        session_id: str,
        task_text: str,
        mode: str | None,
        now: datetime | None = None,
    ) -> None:
        self.mode = normalize_optimization_mode(mode)
        self._root = Path(root)
        self._started_monotonic = time.monotonic()
        started_at = now or datetime.now(UTC)
        self._finished = False
        self._payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": _fingerprint(f"{session_id}:{started_at.isoformat()}:{os.getpid()}"),
            "session_fingerprint": _fingerprint(session_id),
            "task_fingerprint": _fingerprint(" ".join(task_text.split()).lower()),
            "started_at": started_at.isoformat(),
            "mode": self.mode,
            "decisions": [],
            "provider_calls": [],
            "tool_calls": {},
            "broker_calls": 0,
            "route_switches": 0,
            "truncation_extensions": 0,
            "verification": {"count": 0, "passed": 0, "failed": 0},
            "tokens": {"fresh_input": 0, "cache_read": 0, "cache_write": 0, "output": 0},
            "cost_usd": 0.0,
        }

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def decision(
        self,
        kind: str,
        *,
        phase: str,
        proposed: Mapping[str, Any],
        actual: Mapping[str, Any],
        reason: str,
        eligible: bool = True,
    ) -> None:
        if not self.enabled or len(self._payload["decisions"]) >= _MAX_DECISIONS:
            return
        proposed_safe = _safe_value(proposed)
        actual_safe = _safe_value(actual)
        self._payload["decisions"].append(
            {
                "kind": kind[:64],
                "phase": phase[:32],
                "eligible": bool(eligible),
                "changed": proposed_safe != actual_safe,
                "proposed": proposed_safe,
                "actual": actual_safe,
                "reason": reason[:256],
            }
        )
        if kind == "route" and proposed_safe != actual_safe:
            self._payload["route_switches"] += 1

    def record_provider_call(
        self,
        *,
        phase: str,
        model: str,
        finish_reason: str,
        output_limit: int,
        reasoning_effort: str | None,
        fresh_input_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        if not self.enabled:
            return
        tokens = self._payload["tokens"]
        tokens["fresh_input"] += max(0, int(fresh_input_tokens))
        tokens["cache_read"] += max(0, int(cache_read_tokens))
        tokens["cache_write"] += max(0, int(cache_write_tokens))
        tokens["output"] += max(0, int(output_tokens))
        self._payload["cost_usd"] = round(float(self._payload["cost_usd"]) + max(0.0, cost_usd), 8)
        if len(self._payload["provider_calls"]) < _MAX_PROVIDER_CALLS:
            self._payload["provider_calls"].append(
                {
                    "phase": phase[:32],
                    "model": model[:128],
                    "finish_reason": finish_reason[:32],
                    "output_limit": max(0, int(output_limit)),
                    "reasoning_effort": reasoning_effort,
                }
            )

    def record_tool(self, name: str, *, ok: bool) -> None:
        if not self.enabled:
            return
        normalized = name[:128]
        row = self._payload["tool_calls"].setdefault(normalized, {"count": 0, "passed": 0, "failed": 0})
        row["count"] += 1
        row["passed" if ok else "failed"] += 1
        if normalized == "mcp_tool":
            self._payload["broker_calls"] += 1

    def record_verification(self, *, ok: bool) -> None:
        if not self.enabled:
            return
        verification = self._payload["verification"]
        verification["count"] += 1
        verification["passed" if ok else "failed"] += 1

    def record_truncation_extension(self) -> None:
        if self.enabled:
            self._payload["truncation_extensions"] += 1

    def finish(self, *, accepted: bool, error_code: str | None = None) -> Path | None:
        if not self.enabled or self._finished:
            return None
        self._finished = True
        self._payload["finished_at"] = datetime.now(UTC).isoformat()
        self._payload["duration_ms"] = round(max(0.0, time.monotonic() - self._started_monotonic) * 1000)
        self._payload["accepted"] = bool(accepted)
        self._payload["error_code"] = (error_code or "")[:64] or None
        path = runtime_decision_trace_path(self._root)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(self._payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with _WRITE_LOCK:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, line)
            finally:
                os.close(descriptor)
        return path


def load_runtime_decision_traces(
    root: Path | str,
    *,
    days: int = 7,
    limit: int = 10_000,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    path = runtime_decision_trace_path(root)
    if not path.exists() or limit <= 0:
        return []
    cutoff = (now or datetime.now(UTC)) - timedelta(days=max(1, days))
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    finished = datetime.fromisoformat(str(row.get("finished_at", "")))
                    if finished.tzinfo is None:
                        finished = finished.replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    continue
                if isinstance(row, dict) and finished >= cutoff:
                    rows.append(row)
    except OSError:
        return []
    return list(rows)


def summarize_runtime_decisions(
    root: Path | str,
    *,
    days: int = 7,
    limit: int = 10_000,
    now: datetime | None = None,
) -> dict[str, Any]:
    rows = load_runtime_decision_traces(root, days=days, limit=limit, now=now)
    accepted = sum(bool(row.get("accepted")) for row in rows)
    total_cost = round(sum(float(row.get("cost_usd", 0.0) or 0.0) for row in rows), 8)
    token_keys = ("fresh_input", "cache_read", "cache_write", "output")
    tokens = {key: sum(int(row.get("tokens", {}).get(key, 0) or 0) for row in rows) for key in token_keys}
    decisions = [item for row in rows for item in row.get("decisions", []) if isinstance(item, dict)]
    tool_calls = sum(
        int(stats.get("count", 0) or 0)
        for row in rows
        for stats in row.get("tool_calls", {}).values()
        if isinstance(stats, dict)
    )
    by_mode: dict[str, dict[str, Any]] = {}
    for row in rows:
        mode = str(row.get("mode") or "unknown")
        bucket = by_mode.setdefault(mode, {"runs": 0, "accepted": 0, "cost_usd": 0.0})
        bucket["runs"] += 1
        bucket["accepted"] += int(bool(row.get("accepted")))
        bucket["cost_usd"] = round(float(bucket["cost_usd"]) + float(row.get("cost_usd", 0.0) or 0.0), 8)
    return {
        "days": max(1, days),
        "runs": len(rows),
        "accepted_runs": accepted,
        "acceptance_rate": round(accepted / len(rows), 4) if rows else None,
        "cost_usd": total_cost,
        "cost_per_accepted_run_usd": round(total_cost / accepted, 8) if accepted else None,
        "tokens": tokens,
        "provider_calls": sum(len(row.get("provider_calls", [])) for row in rows),
        "tool_calls": tool_calls,
        "broker_calls": sum(int(row.get("broker_calls", 0) or 0) for row in rows),
        "truncation_extensions": sum(int(row.get("truncation_extensions", 0) or 0) for row in rows),
        "decisions": len(decisions),
        "proposed_changes": sum(bool(item.get("changed")) for item in decisions),
        "by_mode": by_mode,
        "trace_path": str(runtime_decision_trace_path(root)),
    }


__all__ = [
    "OptimizationMode",
    "OptimizationTraceRecorder",
    "load_runtime_decision_traces",
    "normalize_optimization_mode",
    "runtime_decision_trace_path",
    "summarize_runtime_decisions",
]
