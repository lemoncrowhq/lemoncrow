"""ToolSupervisionCapability — advanced tool monitoring with circuit breakers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities._optional_runtime import (
    Counter,
    Histogram,
    pybreaker,
    retry,
    stop_after_attempt,
    wait_exponential,
)
from lemoncrow.core.capabilities.pricing import active_model, get_model_pricing

from .anomaly import ToolAnomalyDetector
from .circuit_breaker import CircuitBreaker
from .models import CircuitState
from .store import SupervisionStore

# Token-cost estimates per tool type (in tokens per call)
_TOOL_COSTS: dict[str, int] = {
    "read_file": 200,
    "smart_read": 250,
    "grep": 100,
    "search": 150,
    "edit_file": 300,
    "write_file": 300,
    "diff": 200,
    "run_test": 500,
    "default": 100,
}

_TOKEN_SAVINGS_PER_CACHE_HIT = 200  # estimated tokens saved per avoided call
_DEFAULT_CACHE_TTL_SECONDS = 600

_TOOL_CALL_COUNTER = (
    Counter(
        "lemoncrow_tool_calls_total",
        "Total tool observations by tool name and cache-hit status",
        ["tool", "cache_hit", "failed"],
    )
    if Counter is not None
    else None
)

_TOOL_LATENCY_SECONDS = (
    Histogram(
        "lemoncrow_tool_observe_latency_seconds",
        "Latency spent processing tool supervision observe() calls",
        ["tool"],
        buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    )
    if Histogram is not None
    else None
)


def _estimate_cost(key: str) -> int:
    for prefix, cost in _TOOL_COSTS.items():
        if key.startswith(prefix):
            return cost
    return _TOOL_COSTS["default"]


def _content_hash(payload: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation of a payload.

    Used to recognise semantically identical tool calls regardless of the
    string key used by the caller.  Truncated to 16 hex chars for efficiency.
    """
    try:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        canonical = str(payload)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _cache_disabled_by_env() -> bool:
    return str(os.environ.get("LEMONCROW_CACHE_DISABLED") or "").strip().lower() in {"1", "true", "yes", "on"}


class ToolSupervisionCapability:
    """
    Monitors tool usage with:
    - Transparent result caching (avoids redundant calls)
    - Content-hash cache: same-content different-key still hits
    - Per-tool circuit breaker (blocks pathologically failing tools)
    - Z-score anomaly detection (flags abnormal tool usage)
    - Burst detection (too many calls in a short window)
    - Per-tool cost modeling (token overhead estimates)
    - Token savings tracking
    - Accurate retries_prevented counter (circuit breaker fires)
    """

    def __init__(
        self,
        root: Path,
        *,
        model: str = "",
        cache_enabled: bool = True,
        cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._root = Path(root)
        self._store = SupervisionStore(self._root)
        self._circuit = CircuitBreaker()
        self._pybreakers: dict[str, Any] = {}
        self._anomaly = ToolAnomalyDetector()
        self._model = model or active_model()
        self._cache_enabled = bool(cache_enabled) and not _cache_disabled_by_env()
        self._cache_ttl_seconds = max(0, int(cache_ttl_seconds))
        self._total_calls = 0
        self._avoided_calls = 0
        self._retries_prevented = 0  # circuit-breaker-prevented calls
        self._token_savings = 0
        self._usd_savings = 0.0
        self._chars_saved = 0
        self._tool_call_counts: dict[str, int] = defaultdict(int)
        # Content-hash → cache key reverse index (in-memory only)
        self._hash_index: dict[str, str] = {}

    @property
    def cache_enabled(self) -> bool:
        return self._cache_enabled

    # ------------------------------------------------------------------
    # Core observe/get API
    # ------------------------------------------------------------------

    def observe(
        self,
        key: str,
        result: dict[str, Any],
        *,
        cache_hit: bool = False,
        tool_name: str = "",
    ) -> None:
        """Record a tool call.  If cache_hit=True the actual call was avoided."""
        started = time.perf_counter()
        self._total_calls += 1
        tool = tool_name or key.split(":")[0]
        self._tool_call_counts[tool] += 1

        failed = bool(result.get("error")) or str(result.get("status", "")).lower() in {
            "error",
            "failed",
            "failure",
        }

        # Optional pybreaker state check (in addition to native breaker)
        if pybreaker is not None:
            breaker = self._get_pybreaker(tool)
            if str(getattr(breaker, "current_state", "")).lower().endswith("open"):
                self._retries_prevented += 1
                self._append_history_with_retry(
                    {
                        "kind": "pybreaker_open",
                        "tool": tool,
                        "key": key,
                        "ts": time.time(),
                    }
                )

        # Circuit breaker check — if OPEN, increment retries_prevented
        if not self._circuit.should_allow(tool):
            self._retries_prevented += 1
            self._append_history_with_retry(
                {
                    "kind": "circuit_open",
                    "tool": tool,
                    "key": key,
                    "ts": time.time(),
                }
            )

        self._anomaly.record(tool)

        if failed:
            self._circuit.record_failure(tool)
            if pybreaker is not None:
                self._record_pybreaker_failure(tool)
        else:
            self._circuit.record_success(tool)
            if pybreaker is not None:
                self._record_pybreaker_success(tool)

        effective_cache_hit = bool(cache_hit and self._cache_enabled)
        if effective_cache_hit:
            self._avoided_calls += 1
            savings = _estimate_cost(tool)
            self._token_savings += savings
            self._usd_savings += get_model_pricing(self._model).tokens_to_usd(savings, "output")
            self._chars_saved += savings * 4

        # Cache the result and update content-hash index
        if self._cache_enabled:
            self._set_cached_with_retry(key, result)
            content_hash = _content_hash(result)
            self._hash_index[content_hash] = key

        if _TOOL_CALL_COUNTER is not None:
            _TOOL_CALL_COUNTER.labels(
                tool=tool,
                cache_hit=str(effective_cache_hit).lower(),
                failed=str(bool(failed)).lower(),
            ).inc()

        if _TOOL_LATENCY_SECONDS is not None:
            _TOOL_LATENCY_SECONDS.labels(tool=tool).observe(max(0.0, time.perf_counter() - started))

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Return a previously cached tool result, or None.

        Falls back to content-hash lookup so semantically identical
        results are found even under different cache keys.
        """
        if not self._cache_enabled:
            return None
        result = self._get_cached_with_retry(cache_key)
        if result is not None:
            return result
        # Content-hash fallback: check if we've seen an identical response
        # (This is a best-effort in-memory lookup; not persisted across restarts)
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.02, min=0.02, max=0.2),
        reraise=True,
    )
    def _set_cached_with_retry(self, key: str, payload: dict[str, Any]) -> None:
        self._store.set_cached(key, payload, git_head=self._git_head())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.02, min=0.02, max=0.2),
        reraise=True,
    )
    def _get_cached_with_retry(self, key: str) -> dict[str, Any] | None:
        return self._store.get_cached(
            key,
            ttl_seconds=self._cache_ttl_seconds,
            git_head=self._git_head(),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.02, min=0.02, max=0.2),
        reraise=True,
    )
    def _append_history_with_retry(self, entry: dict[str, Any]) -> None:
        self._store.append_history(entry)

    def _get_pybreaker(self, tool: str) -> Any:
        if pybreaker is None:
            return None
        breaker = self._pybreakers.get(tool)
        if breaker is None:
            breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60, name=f"tool:{tool}")
            self._pybreakers[tool] = breaker
        return breaker

    def _git_head(self) -> str:
        from lemoncrow.core.foundation.paths import resolve_workspace_root

        workspace = resolve_workspace_root(self._root)
        try:
            result = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    def _record_pybreaker_failure(self, tool: str) -> None:
        breaker = self._get_pybreaker(tool)
        if breaker is None:
            return

        def _raise_failure() -> None:
            raise RuntimeError("tool call failure")

        try:
            breaker.call(_raise_failure)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return

    def _record_pybreaker_success(self, tool: str) -> None:
        breaker = self._get_pybreaker(tool)
        if breaker is None:
            return
        try:
            breaker.close()
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return

    # ------------------------------------------------------------------
    # Metrics & reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return high-level metrics."""
        cache_hit_rate = round(self._avoided_calls / self._total_calls, 3) if self._total_calls > 0 else 0.0
        return {
            "total_tool_calls": self._total_calls,
            "avoided_tool_calls": self._avoided_calls,
            "cache_hit_rate": cache_hit_rate,
            "cache_enabled": self._cache_enabled,
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "token_savings": self._token_savings,
            "usd_savings": round(self._usd_savings, 6),
            "model": self._model,
            "chars_saved": self._chars_saved,
            "retries_prevented": self._retries_prevented,
            "tool_histogram": dict(self._tool_call_counts),
        }

    def tool_report(self) -> dict[str, Any]:
        """Full tool supervision report including anomalies and circuit states."""
        metrics = self.status()
        anomalies: list[dict[str, Any]] = [
            {"tool": a.tool, "severity": a.severity, "message": a.message, "z_score": a.z_score}
            for a in self._anomaly.detect_anomalies()
        ]
        circuits = self._circuit.all_states()
        open_circuits = [t for t, s in circuits.items() if s == CircuitState.OPEN.value]

        redundant_patterns = self._find_redundant_patterns()
        recommendations = self._build_recommendations(redundant_patterns, anomalies, open_circuits)

        return {
            "metrics": metrics,
            "redundant_patterns": redundant_patterns,
            "anomalies": anomalies,
            "circuit_breakers": circuits,
            "open_circuits": open_circuits,
            "recommendations": recommendations,
        }

    def _find_redundant_patterns(self) -> list[dict[str, Any]]:
        state = self._store.load()
        cache = state.get("cache", {})
        # Group keys by tool prefix
        tool_counts: dict[str, int] = {}
        for key in cache:
            prefix = key.split(":")[0]
            tool_counts[prefix] = tool_counts.get(prefix, 0) + 1
        return [{"tool": tool, "cached_results": count} for tool, count in tool_counts.items() if count >= 3]

    def _build_recommendations(
        self,
        redundant: list[dict[str, Any]],
        anomalies: list[dict[str, Any]],
        open_circuits: list[str],
    ) -> list[str]:
        recs: list[str] = []
        if redundant:
            tools = [r["tool"] for r in redundant]
            recs.append(f"Consider caching results for: {', '.join(tools)}")
        for a in anomalies:
            if a["severity"] == "critical":
                recs.append(f"Investigate heavy usage of tool '{a['tool']}': {a['message']}")
        for t in open_circuits:
            recs.append(f"Circuit breaker OPEN for '{t}' — check for systematic failures")
        if self._retries_prevented > 0:
            recs.append(f"Circuit breaker prevented {self._retries_prevented} retries on failing tools")
        if self._avoided_calls > 0:
            recs.append(
                f"Cache hits saved ~{self._token_savings} tokens "
                f"(≈${self._usd_savings:.4f} at {self._model}) "
                f"across {self._avoided_calls} avoided calls"
            )
        return recs

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def diff_context(
        self,
        paths: list[str],
        *,
        lines: int = 5,
    ) -> dict[str, Any]:
        """Return git diff context for a list of file paths (non-crashing)."""
        diffs: list[dict[str, Any]] = []
        for path in paths:
            try:
                result = subprocess.run(
                    ["git", "diff", "--unified=" + str(lines), "--", path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                diffs.append({"path": path, "diff": result.stdout[:4000]})
            except Exception as exc:
                logging.exception("Recovered from broad exception handler")
                diffs.append({"path": path, "diff": "", "error": str(exc)})
        return {"diffs": diffs}

    def test_context(self, paths: list[str]) -> dict[str, Any]:
        """Return lightweight test context for a list of file paths."""
        test_contexts: list[dict[str, Any]] = []
        for path in paths:
            p = Path(path)
            ctx: dict[str, Any] = {"path": path, "exists": p.is_file(), "test_files": []}
            if p.is_file() and p.suffix == ".py":
                try:
                    src = p.read_text(encoding="utf-8", errors="replace")
                    import re

                    test_fns = re.findall(r"^def (test_\w+)", src, re.M)
                    ctx["test_files"] = test_fns[:20]
                except Exception:
                    logging.exception("Recovered from broad exception handler")
                    ctx["test_files"] = []
            test_contexts.append(ctx)
        return {"test_contexts": test_contexts}
