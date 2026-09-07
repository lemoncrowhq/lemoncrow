"""Deterministic cost controls for LemonCrow-owned coding sessions.

These policies are intentionally local: choosing a phase, trimming a schema, or
compacting old context must never require another model or MCP round trip.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

RuntimePhase = Literal["explore", "execute", "repair", "finish"]
_VERIFY_PATTERNS = (
    "pytest",
    "unittest",
    "npm test",
    "npm run test",
    "pnpm test",
    "yarn test",
    "cargo test",
    "go test",
    "make test",
    "make check",
    "ruff check",
    "mypy",
    "pyright",
    "tsc ",
    "tsc --",
)


def normalize_cache_policy(policy: str) -> Literal["off", "5m", "1h"]:
    normalized = (policy or "auto").strip().lower()
    if normalized in {"off", "fresh", "none", "disabled"}:
        return "off"
    return "1h" if normalized == "1h" else "5m"


def cache_control(policy: str) -> dict[str, str] | None:
    normalized = normalize_cache_policy(policy)
    if normalized == "off":
        return None
    control = {"type": "ephemeral"}
    if normalized == "1h":
        control["ttl"] = "1h"
    return control


@dataclass
class RuntimeTurnState:
    """Observable phase and verification receipt for an owned runtime turn."""

    primer_supplied: bool = False
    phase: RuntimePhase = "explore"
    edit_count: int = 0
    failure_count: int = 0
    verification_count: int = 0
    edited_paths: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    last_verification_ok: bool | None = None
    last_verification_output_hash: str = ""
    mutation_generation: int = 0
    verified_generation: int = -1

    def __post_init__(self) -> None:
        if self.primer_supplied:
            self.phase = "execute"

    @property
    def ready_for_receipt(self) -> bool:
        return (
            self.edit_count > 0
            and self.last_verification_ok is True
            and self.verified_generation == self.mutation_generation
        )

    def record(self, tool_name: str, ok: bool, args: dict[str, Any], result: str = "") -> None:
        verification = tool_name == "bash" and is_verification_command(args)
        if verification:
            self.verification_count += 1
            command = _render_command(args.get("command", ""))
            if command:
                self.verification_commands.append(command[:500])
            self.last_verification_ok = ok
            self.last_verification_output_hash = hashlib.sha256(result.encode("utf-8", errors="replace")).hexdigest()
            if not ok:
                self.failure_count += 1
                self.verified_generation = -1
                self.phase = "repair"
                return
            if self.edit_count:
                self.verified_generation = self.mutation_generation
                self.phase = "finish"
            else:
                self.phase = "execute"
            return

        if not ok:
            self.failure_count += 1
            self.last_verification_ok = False
            self.verified_generation = -1
            self.phase = "repair"
            return
        if tool_name in {"read", "grep", "explore", "symbols", "code_search"} and self.phase == "explore":
            self.phase = "execute"
        if tool_name == "edit":
            self.edit_count += 1
            self.mutation_generation += 1
            self.last_verification_ok = None
            self.verified_generation = -1
            for edit in args.get("edits", []):
                if not isinstance(edit, dict):
                    continue
                path = str(edit.get("file_path") or edit.get("path") or "").split("#", 1)[0]
                if path and path not in self.edited_paths:
                    self.edited_paths.append(path)
            self.phase = "execute"
        elif self.edit_count and (tool_name == "bash" or tool_name == "mcp_tool" or tool_name.startswith("mcp__")):
            # Unknown shell/MCP effects after an edit require a fresh verification.
            self.mutation_generation += 1
            self.last_verification_ok = None
            self.verified_generation = -1
            self.phase = "execute"


def _render_command(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(item) for item in command)
    return str(command)


def is_verification_command(args: dict[str, Any]) -> bool:
    rendered = _render_command(args.get("command", ""))
    lowered = f" {rendered.lower()} "
    return any(pattern in lowered for pattern in _VERIFY_PATTERNS)


def output_token_limit(budget: str, phase: RuntimePhase) -> int:
    override = os.environ.get("LEMONCROW_MAX_OUTPUT_TOKENS", "").strip()
    if override:
        try:
            return max(256, int(override))
        except ValueError:
            pass
    table = {
        "cheap": {"explore": 900, "execute": 2800, "repair": 3600, "finish": 900},
        "balanced": {"explore": 1200, "execute": 4096, "repair": 5200, "finish": 1200},
        "best": {"explore": 1800, "execute": 6144, "repair": 7600, "finish": 1600},
    }
    return table.get(budget, table["balanced"])[phase]


def reasoning_effort_for(model: str, budget: str, phase: RuntimePhase) -> str | None:
    lowered = model.lower()
    if not any(marker in lowered for marker in ("gpt-5", "codex", "/o1", "/o3", "/o4")):
        return None
    if phase in {"explore", "finish"}:
        return "low"
    if phase == "repair" or budget == "best":
        return "high"
    return "medium"


_OUTPUT_GOVERNOR_DIRECTIVE = (
    "\n\n## Runtime output governor\n"
    "During execution, call tools without progress narration. After an edit, run the relevant "
    "verification command. LemonCrow will generate the bounded final receipt after successful verification."
)
_MUTATION_WORDS = (
    r"add|build|change|create|debug|edit|fix|implement|make|migrate|modify|"
    r"patch|refactor|remove|rename|repair|resolve|update|write"
)
_MUTATION_INTENT = re.compile(rf"\b({_MUTATION_WORDS})\b", re.IGNORECASE)
_READ_ONLY_PREFIX = re.compile(
    r"^\s*(?:can you\s+)?(?:explain|find|how|investigate|locate|review|show|trace|understand|what|where|why)\b",
    re.IGNORECASE,
)
_CONJUNCTIVE_MUTATION = re.compile(
    rf"\b(?:and|also|then)\s+(?:please\s+)?({_MUTATION_WORDS})\b",
    re.IGNORECASE,
)


def task_requests_mutation(task: str) -> bool:
    if not _MUTATION_INTENT.search(task):
        return False
    if _READ_ONLY_PREFIX.search(task) and not _CONJUNCTIVE_MUTATION.search(task):
        return False
    return True


def required_tool_choice_supported(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in ("anthropic", "claude", "openai", "gpt", "codex", "/o1", "/o3"))


def recommended_tool_choice(model: str, phase: RuntimePhase, task: str) -> str:
    if phase in {"execute", "repair"} and task_requests_mutation(task) and required_tool_choice_supported(model):
        return "required"
    return "auto"


def output_governor_system_message(message: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode != "enforce":
        return message
    governed = dict(message)
    content = message.get("content")
    if isinstance(content, str):
        governed["content"] = content + _OUTPUT_GOVERNOR_DIRECTIVE
        return governed
    if isinstance(content, list):
        blocks = [dict(block) if isinstance(block, dict) else block for block in content]
        for block in blocks:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] += _OUTPUT_GOVERNOR_DIRECTIVE
                break
        governed["content"] = blocks
    return governed


def is_truncation_finish_reason(reason: str) -> bool:
    return reason.strip().lower() in {"length", "max_tokens", "max_output_tokens"}


def build_final_receipt(state: RuntimeTurnState) -> str:
    paths = state.edited_paths[:6]
    rendered_paths = ", ".join(paths) if paths else f"{state.edit_count} file(s)"
    if len(state.edited_paths) > len(paths):
        rendered_paths += f" (+{len(state.edited_paths) - len(paths)} more)"
    command = state.verification_commands[-1] if state.verification_commands else "verification"
    return f"Done: updated {rendered_paths}. Verified: `{command}` passed."


@dataclass(frozen=True)
class MCPExposure:
    tools: tuple[Any, ...]
    focused: bool
    reason: str


def _phrase(value: str) -> str:
    return re.sub(r"[_\-]+", " ", value.lower()).strip()


def _mcp_match_score(tool: Any, task: str) -> int:
    """Require an explicit name; description similarity caused prior false matches."""
    lowered = task.lower()
    server = _phrase(str(getattr(tool, "server_name", "")))
    name = _phrase(str(getattr(tool, "name", "")))
    score = 0
    if server and re.search(rf"\b{re.escape(server)}\b", lowered):
        score += 10
    if name and len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", lowered):
        score += 12
    return score


def choose_mcp_exposure(tools: list[Any], task: str, mode: str = "auto") -> MCPExposure:
    """Select a stable per-turn schema set without a mandatory discovery call."""
    ordered = sorted(tools, key=lambda item: (str(item.server_name), str(item.name)))
    if mode == "eager" or (mode == "auto" and len(ordered) <= 12):
        return MCPExposure(tuple(ordered), False, "small or explicitly eager catalog")
    scored = sorted(
        ((score, tool) for tool in ordered if (score := _mcp_match_score(tool, task)) > 0),
        key=lambda item: (-item[0], str(item[1].server_name), str(item[1].name)),
    )
    high_confidence = [tool for score, tool in scored if score >= 10][:12]
    if mode == "auto" and not high_confidence:
        return MCPExposure(tuple(ordered), False, "ambiguous task; eager avoids a broker round trip")
    selected = high_confidence if high_confidence else [tool for _, tool in scored[:12]]
    return MCPExposure(tuple(selected), True, "explicit MCP server/tool match")


def mcp_broker_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "mcp_tool",
            "description": (
                "Fallback for a focused MCP catalog. Use only when exposed MCP tools "
                "cannot do the task. Search is local; call invokes the named tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "call"]},
                    "query": {"type": "string"},
                    "server": {"type": "string"},
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["action"],
            },
        },
    }


def estimate_context_tokens(messages: list[dict[str, Any]]) -> int:
    try:
        rendered = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = repr(messages)
    return max(1, len(rendered) // 4)


def should_switch_route(
    messages: list[dict[str, Any]],
    current_model: str,
    candidate_model: str,
    phase: RuntimePhase,
) -> bool:
    if not candidate_model or candidate_model == current_model or phase == "finish":
        return False
    return estimate_context_tokens(messages) <= 30_000


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("text")
        )
    return str(content or "")


def compact_history(
    messages: list[dict[str, Any]],
    *,
    threshold_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Compact old completed turns between requests, never during a tool loop."""
    if threshold_tokens is None:
        try:
            threshold_tokens = int(os.environ.get("LEMONCROW_COMPACT_AT_TOKENS", "120000"))
        except ValueError:
            threshold_tokens = 120_000
    if threshold_tokens <= 0 or estimate_context_tokens(messages) <= threshold_tokens:
        return messages, False
    head = [messages[0]] if messages and messages[0].get("role") == "system" else []
    body = messages[len(head) :]
    if len(body) < 6:
        return messages, False
    keep_chars = max(8_000, threshold_tokens * 4 // 3)
    used = 0
    start = len(body)
    for index in range(len(body) - 1, -1, -1):
        used += len(_content_text(body[index])) + 200
        if used > keep_chars:
            break
        start = index
    while start < len(body) and body[start].get("role") != "user":
        start += 1
    if start <= 0 or start >= len(body):
        return messages, False

    summary_lines = ["[Compacted prior session evidence]"]
    for message in body[:start]:
        role = str(message.get("role", "unknown"))
        if message.get("tool_calls"):
            names = [
                str(call.get("function", {}).get("name", "tool"))
                for call in message.get("tool_calls", [])
                if isinstance(call, dict)
            ]
            summary_lines.append(f"assistant tools: {', '.join(names)}")
            continue
        text = " ".join(_content_text(message).split())
        if text:
            summary_lines.append(f"{role}: {text[:500]}")
        if sum(len(line) for line in summary_lines) >= 12_000:
            break
    compacted = [
        *head,
        {"role": "user", "content": "\n".join(summary_lines)[:12_000]},
        {"role": "assistant", "content": "Context restored from the deterministic compacted session record."},
        *body[start:],
    ]
    return compacted, True


__all__ = [
    "MCPExposure",
    "RuntimeTurnState",
    "build_final_receipt",
    "cache_control",
    "choose_mcp_exposure",
    "compact_history",
    "estimate_context_tokens",
    "is_truncation_finish_reason",
    "is_verification_command",
    "mcp_broker_schema",
    "normalize_cache_policy",
    "output_governor_system_message",
    "output_token_limit",
    "reasoning_effort_for",
    "recommended_tool_choice",
    "required_tool_choice_supported",
    "should_switch_route",
    "task_requests_mutation",
]
