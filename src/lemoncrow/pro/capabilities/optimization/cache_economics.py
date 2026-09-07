"""Provider-aware cache TTL, stable-lane, and selective-write decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from lemoncrow.pro.capabilities.optimization.runtime_decisions import normalize_optimization_mode
from lemoncrow.pro.capabilities.owned_agent_session.phase_runner import _provider_cache_style
from lemoncrow.pro.capabilities.owned_agent_session.runtime_policy import normalize_cache_policy

CacheTier = Literal["off", "5m", "1h"]
_MAX_GAPS = 64
_MIN_ADAPTIVE_SAMPLES = 3
_LONG_TTL_THRESHOLD = 5 * 60
_LONG_TTL_CEILING = 60 * 60
_ANTHROPIC_READ_SAVING_RATIO = 0.90
_ANTHROPIC_EXTRA_1H_WRITE_RATIO = 0.75
_VOLATILE_MARKERS = (
    "diff --git ",
    "traceback (most recent call last)",
    "short test summary info",
    "npm err!",
    "command failed with exit",
    "[output truncated",
)
_STABLE_TOOL_NAMES = frozenset({"read", "grep", "explore", "symbols", "code_search"})
_TTL_RE = re.compile(r"^(?:300|3600)s$")


@dataclass(frozen=True)
class CacheBreakpoint:
    index: int
    reason: str


@dataclass(frozen=True)
class CacheDecision:
    requested_policy: str
    provider_style: str
    proposed_tier: CacheTier
    actual_tier: CacheTier
    prefix_hash: str
    lane_key: str
    breakpoint: CacheBreakpoint | None
    sample_count: int
    long_gap_probability: float
    expected_value_ratio: float
    reason: str

    @property
    def enabled(self) -> bool:
        return self.actual_tier != "off"

    def trace_proposed(self) -> dict[str, Any]:
        return {
            "tier": self.proposed_tier,
            "lane": self.lane_key,
            "breakpoint": self.breakpoint.reason if self.breakpoint else "system_only",
        }

    def trace_actual(self) -> dict[str, Any]:
        return {
            "tier": self.actual_tier,
            "lane": self.lane_key,
            "breakpoint": self.breakpoint.reason if self.breakpoint else "system_only",
        }


def cache_control_for_tier(tier: CacheTier) -> dict[str, str] | None:
    if tier == "off":
        return None
    control = {"type": "ephemeral"}
    if tier == "1h":
        control["ttl"] = "1h"
    return control


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def stable_system_text(messages: list[dict[str, Any]]) -> str:
    if not messages or messages[0].get("role") != "system":
        return ""
    return _content_text(messages[0].get("content"))


def stable_prefix_hash(provider_style: str, model: str, messages: list[dict[str, Any]]) -> str:
    material = f"{provider_style}\0{model}\0{stable_system_text(messages)}"
    return hashlib.sha256(material.encode("utf-8", errors="replace")).hexdigest()


def _tool_names_by_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            name = str(call.get("function", {}).get("name") or "")
            if call_id:
                names[call_id] = name
    return names


def _volatile(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _VOLATILE_MARKERS)


def select_cache_breakpoint(messages: list[dict[str, Any]]) -> CacheBreakpoint | None:
    """Select reusable user/read evidence; reject model prose, diffs, and logs."""
    tool_names = _tool_names_by_id(messages)
    for index in range(len(messages) - 1, 0, -1):
        message = messages[index]
        text = _content_text(message.get("content"))
        if not text or len(text) > 32_000 or _volatile(text):
            continue
        role = message.get("role")
        if role == "tool":
            tool_name = tool_names.get(str(message.get("tool_call_id") or ""), "")
            if tool_name and tool_name not in _STABLE_TOOL_NAMES:
                continue
            return CacheBreakpoint(index=index, reason=f"{tool_name or 'bounded_tool'}_evidence")
        if role == "user" and not text.startswith("[Output was truncated."):
            return CacheBreakpoint(index=index, reason="stable_user_turn")
    return None


def _database_path(root: Path | str) -> Path:
    return Path(root).expanduser().resolve() / "cache" / "cache-economics.sqlite3"


def _connect(root: Path | str) -> sqlite3.Connection:
    path = _database_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""
        CREATE TABLE IF NOT EXISTS cache_prefix_stats (
            prefix_hash TEXT PRIMARY KEY,
            last_access REAL NOT NULL,
            gaps_json TEXT NOT NULL,
            accesses INTEGER NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS provider_cache_handles (
            prefix_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            name TEXT NOT NULL,
            expires_at REAL NOT NULL,
            PRIMARY KEY (prefix_hash, model)
        )
        """)
    connection.commit()
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return connection


def _observe_prefix(root: Path | str, prefix_hash: str, now: datetime) -> tuple[list[float], int]:
    timestamp = now.timestamp()
    try:
        with _connect(root) as connection:
            row = connection.execute(
                "SELECT last_access, gaps_json, accesses FROM cache_prefix_stats WHERE prefix_hash = ?",
                (prefix_hash,),
            ).fetchone()
            gaps: list[float] = []
            accesses = 0
            if row is not None:
                accesses = max(0, int(row[2]))
                try:
                    loaded = json.loads(str(row[1]))
                    gaps = [float(item) for item in loaded if isinstance(item, int | float) and math.isfinite(item)]
                except (TypeError, ValueError):
                    gaps = []
                gap = timestamp - float(row[0])
                if gap >= 0:
                    gaps.append(gap)
            gaps = gaps[-_MAX_GAPS:]
            accesses += 1
            connection.execute(
                """
                INSERT INTO cache_prefix_stats(prefix_hash, last_access, gaps_json, accesses)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(prefix_hash) DO UPDATE SET
                    last_access = excluded.last_access,
                    gaps_json = excluded.gaps_json,
                    accesses = excluded.accesses
                """,
                (prefix_hash, timestamp, json.dumps(gaps, separators=(",", ":")), accesses),
            )
            return gaps, accesses
    except sqlite3.Error:
        return [], 0


def _adaptive_tier(provider_style: str, gaps: list[float]) -> tuple[CacheTier, float, float, str]:
    if provider_style == "none":
        return "off", 0.0, 0.0, "provider has no managed cache policy"
    if not gaps:
        return "5m", 0.0, -_ANTHROPIC_EXTRA_1H_WRITE_RATIO, "no observed reuse gap; conservative 5m"
    long_count = sum(_LONG_TTL_THRESHOLD < gap <= _LONG_TTL_CEILING for gap in gaps)
    probability = long_count / len(gaps)
    expected_value = probability * _ANTHROPIC_READ_SAVING_RATIO - _ANTHROPIC_EXTRA_1H_WRITE_RATIO
    if provider_style in {"anthropic", "gemini"} and len(gaps) >= _MIN_ADAPTIVE_SAMPLES and expected_value > 0:
        return "1h", probability, expected_value, "observed 5-60 minute reuse repays the longer write"
    if provider_style == "openai":
        return "5m", probability, expected_value, "provider-managed cache with a stable prompt cache key"
    return "5m", probability, expected_value, "long-tier expected value is not positive"


def choose_cache_decision(
    root: Path | str,
    *,
    requested_policy: str,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    optimization_mode: str,
    now: datetime | None = None,
) -> CacheDecision:
    style = _provider_cache_style(provider, model)
    prefix_hash = stable_prefix_hash(style, model, messages)
    lane_key = f"lc:{style}:{prefix_hash[:24]}"
    mode = normalize_optimization_mode(optimization_mode)
    gaps, _accesses = ([], 0) if mode == "off" else _observe_prefix(root, prefix_hash, now or datetime.now(UTC))
    breakpoint = select_cache_breakpoint(messages)
    explicit = (requested_policy or "auto").strip().lower() != "auto"

    if explicit:
        proposed = normalize_cache_policy(requested_policy)
        reason = "explicit cache policy"
        probability = 0.0
        expected_value = 0.0
    else:
        proposed, probability, expected_value, reason = _adaptive_tier(style, gaps)

    if style == "none":
        proposed = "off"
        actual: CacheTier = "off"
    elif explicit:
        actual = proposed
    elif mode == "enforce":
        actual = proposed
    else:
        actual = "5m"

    return CacheDecision(
        requested_policy=requested_policy,
        provider_style=style,
        proposed_tier=proposed,
        actual_tier=actual,
        prefix_hash=prefix_hash,
        lane_key=lane_key,
        breakpoint=breakpoint,
        sample_count=len(gaps),
        long_gap_probability=round(probability, 4),
        expected_value_ratio=round(expected_value, 4),
        reason=reason,
    )


def should_rewrite_compacted_prefix(
    *,
    old_tokens: int,
    compacted_tokens: int,
    expected_future_reads: float,
    write_multiplier: float = 1.25,
    read_multiplier: float = 0.10,
) -> bool:
    """Return whether one compacted rewrite is cheaper than future reads of the old prefix."""
    if old_tokens <= 0 or compacted_tokens <= 0 or compacted_tokens >= old_tokens:
        return False
    rewrite = compacted_tokens * write_multiplier
    keep = old_tokens * read_multiplier * max(0.0, expected_future_reads)
    compacted = compacted_tokens * read_multiplier * max(0.0, expected_future_reads)
    return rewrite + compacted < keep


def load_provider_cache_handle(
    root: Path | str,
    *,
    prefix_hash: str,
    model: str,
    now: datetime | None = None,
) -> str | None:
    try:
        with _connect(root) as connection:
            row = connection.execute(
                "SELECT name, expires_at FROM provider_cache_handles WHERE prefix_hash = ? AND model = ?",
                (prefix_hash, model),
            ).fetchone()
            if row is None or float(row[1]) <= (now or datetime.now(UTC)).timestamp():
                return None
            return str(row[0])
    except sqlite3.Error:
        return None


def save_provider_cache_handle(
    root: Path | str,
    *,
    prefix_hash: str,
    model: str,
    name: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> None:
    if not name:
        return
    expires_at = (now or datetime.now(UTC)).timestamp() + max(1, ttl_seconds)
    try:
        with _connect(root) as connection:
            connection.execute(
                """
                INSERT INTO provider_cache_handles(prefix_hash, model, name, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(prefix_hash, model) DO UPDATE SET
                    name = excluded.name,
                    expires_at = excluded.expires_at
                """,
                (prefix_hash, model, name, expires_at),
            )
    except sqlite3.Error:
        return


def valid_gemini_ttl(value: str) -> str:
    return value if _TTL_RE.fullmatch(value) else "300s"


__all__ = [
    "CacheBreakpoint",
    "CacheDecision",
    "CacheTier",
    "cache_control_for_tier",
    "choose_cache_decision",
    "load_provider_cache_handle",
    "save_provider_cache_handle",
    "select_cache_breakpoint",
    "should_rewrite_compacted_prefix",
    "stable_system_text",
    "valid_gemini_ttl",
]
