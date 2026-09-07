"""Agentic (tool-using) reviewer loop — the model investigates before judging.

Unlike the single-shot runner, this lets the model call ``read_file`` / ``grep``
to investigate the diff before emitting its verdict via a ``finish`` tool — the
agentic reviewer pass. It runs against litellm directly (provider creds
from env), so it works inside the detached reviewer child. Fully fail-open:
any problem returns ``None`` and the caller falls back to the single-shot review.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_MAX_TURNS = 6
_READ_CAP = 4000
_GREP_CAP = 60
_FALLBACK_MODEL = "claude-sonnet-4-5"

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a repo file, optionally a 1-based line range, to investigate the change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the repo with a regex; returns matching path:line: text lines.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Emit the final review. Call exactly once when done investigating.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["DONE", "NEEDS_FIX"]},
                    "checklist": {"type": "string"},
                    "missing": {"type": "string"},
                    "findings": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["verdict"],
            },
        },
    },
]


def _read_file(repo_root: Path, path: str, start: int | None, end: int | None) -> str:
    try:
        target = (repo_root / path).resolve()
        target.relative_to(repo_root.resolve())  # contain within repo
    except (ValueError, OSError):
        return f"refused: {path} is outside the repo"
    try:
        text = target.read_text("utf-8")
    except OSError as exc:
        return f"error reading {path}: {exc}"
    if start is not None:
        lines = text.splitlines()
        lo = max(0, int(start) - 1)
        hi = int(end) if end is not None else lo + 40
        text = "\n".join(lines[lo:hi])
    return text[:_READ_CAP]


def _grep(repo_root: Path, pattern: str) -> str:
    try:
        result = subprocess.run(
            ["rg", "-n", "--no-heading", "-e", pattern, str(repo_root)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return "grep unavailable"
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()][:_GREP_CAP]
    return "\n".join(lines) or "no matches"


def _dispatch(name: str, args: Mapping[str, Any], repo_root: Path) -> str:
    if name == "read_file":
        return _read_file(repo_root, str(args.get("path") or ""), args.get("start"), args.get("end"))
    if name == "grep":
        return _grep(repo_root, str(args.get("pattern") or ""))
    return f"unknown tool: {name}"


def _finish_verdict(args: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(args.get("verdict") or "").strip().upper()
    findings = args.get("findings")
    return {
        "verdict": "DONE" if verdict == "DONE" else "NEEDS_FIX",
        "checklist": str(args.get("checklist") or ""),
        "missing": str(args.get("missing") or ""),
        "findings": findings if isinstance(findings, list) else [],
    }


def _json_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _litellm_completion(*, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
    # Route through the infra litellm boundary (keeps the litellm import out of
    # core/capabilities per the provider-confinement architecture test).
    from lemoncrow.infra.internal_llm.litellm_client import tool_completion

    return tool_completion(model=model, messages=messages, tools=tools)


def run_agentic_review(
    *,
    repo_root: str | Path,
    diffs: Mapping[str, str],
    contract: str,
    kb: str = "",
    model: str = "",
    max_turns: int = _MAX_TURNS,
    completion: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Run a bounded tool-using review loop. Returns a verdict dict, or None to fall back."""
    repo = Path(repo_root)
    call = completion or _litellm_completion
    model_id = model.split("/", 1)[-1] if model.startswith("anthropic/") else (model or _FALLBACK_MODEL)
    diff_text = "\n".join(f"### {path}\n```diff\n{diff}\n```" for path, diff in diffs.items())
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": contract + "\nInvestigate with read_file/grep as needed, then call finish exactly once.",
        },
        {"role": "user", "content": (kb + "\n" if kb else "") + "## Diffs under review\n" + diff_text},
    ]
    for _ in range(max_turns):
        try:
            response = call(model=model_id, messages=messages, tools=_TOOLS)
            message = response.choices[0].message
        except Exception:  # noqa: BLE001 - any transport error -> single-shot fallback
            return None
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return None  # no structured finish -> let the caller fall back
        messages.append(
            {
                "role": "assistant",
                "content": getattr(message, "content", "") or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            args = _json_args(tc.function.arguments)
            if tc.function.name == "finish":
                return _finish_verdict(args)
            output = _dispatch(tc.function.name, args, repo)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
    return None
