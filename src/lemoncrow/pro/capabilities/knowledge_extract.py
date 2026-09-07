"""Local knowledge extraction — distill durable review rules from .lemoncrow/lessons.

LemonCrow's local analog of baseline's KB learning loop, but on the user's own
infrastructure. It reads the repo's ``.lemoncrow/lessons`` blocks, asks a model to distill
short durable repo-specific review rules, and merges them into the review overlay
(``review_overlay.json`` -> notes) the live reviewer already applies.

The model backend is selectable — LemonCrow's owned agent-spawn (``auto``), or the
``claude`` / ``codex`` / ``ollama`` CLI — and every run is bounded by a spend cap
estimated before any tokens are spent.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.live_reviewer.knowledge import (
    ensure_repo_share_gitignore,
    load_overlay,
    load_repo_overlay,
    overlay_path,
    repo_overlay_path,
    write_overlay,
)

_MAX_RULES = 25
_DEFAULT_MAX_ITEMS = 20
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_MAX_SPEND_USD = 0.50
_OVERLAY_NOTES_CAP = 60
_FREE_HOSTS = frozenset({"ollama", "codex"})  # local / flat-rate subscription

_PROMPT_HEADER = (
    "You are distilling durable, repo-specific CODE REVIEW rules from a team's lessons notes. "
    "Output ONLY a JSON array of short imperative rules (strings), at most 25, each one line and "
    "directly actionable by a reviewer (e.g. 'New endpoints must check authorization'). "
    "No prose, no markdown fences — just the JSON array.\n\nLessons:\n"
)


def gather_sources(
    repo_root: str | Path,
    *,
    max_items: int = _DEFAULT_MAX_ITEMS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[str]:
    """Read up to *max_items* recent lessons blocks, bounded by *max_chars*."""
    blocks = Path(repo_root) / ".lemoncrow" / "lessons" / "blocks"
    if not blocks.is_dir():
        return []
    try:
        files = sorted(blocks.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    out: list[str] = []
    total = 0
    for path in files[:max_items]:
        try:
            text = path.read_text("utf-8").strip()
        except OSError:
            continue
        if not text:
            continue
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
        if text:
            out.append(text)
            total += len(text)
        if total >= max_chars:
            break
    return out


def preflight_cost_usd(prompt: str, host: str, model: str) -> float:
    """Rough pre-flight cost estimate for a prompt string. Local/subscription hosts are free.

    Distinct from ``savings_summary.estimate_cost_usd`` (which prices known token
    counts): this guesses tokens from the prompt length before the call is made.
    """
    if host in _FREE_HOSTS:
        return 0.0
    from lemoncrow.core.capabilities.pricing import get_model_pricing

    model_id = model.split("/", 1)[-1] if "/" in model else model
    pricing = get_model_pricing(model_id or "claude-sonnet-4-5")
    if pricing is None or not pricing.known:
        pricing = get_model_pricing("claude-sonnet-4-5")
    return round(pricing.request_cost_usd(input_tokens=len(prompt) // 4 + 200, output_tokens=800), 4)


def parse_rules(text: str) -> list[str]:
    """Parse a model's output into a deduped, capped list of rule strings."""
    text = (text or "").strip()
    rules: list[str] = []
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                rules = [str(item).strip() for item in arr if str(item).strip()]
        except json.JSONDecodeError:
            rules = []
    if not rules:
        for line in text.splitlines():
            stripped = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if len(stripped) > 8:
                rules.append(stripped)
    seen: set[str] = set()
    uniq: list[str] = []
    for rule in rules:
        rule = rule[:200]
        key = rule.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(rule)
    return uniq[:_MAX_RULES]


def overlay_target(root: str | Path, repo_root: str | Path | None, scope: str) -> Path:
    """Where extracted rules are written. ``repo`` = team-shared (committable)."""
    if scope == "repo" and repo_root is not None:
        return repo_overlay_path(repo_root)
    return overlay_path(root)


def merge_into_overlay(
    root: str | Path,
    rules: list[str],
    *,
    repo_root: str | Path | None = None,
    scope: str = "repo",
) -> int:
    """Append new rules to the target overlay's notes (deduped). Returns count added.

    Default scope ``repo`` writes the team overlay (<repo>/.lemoncrow/review.json) so
    committing it distributes the rules; ``personal`` writes the per-user overlay.
    """
    if not rules:
        return 0
    if scope == "repo" and repo_root is not None:
        overlay = load_repo_overlay(repo_root)
    else:
        overlay = load_overlay(root)
    existing = {note.lower() for note in overlay["notes"]}
    added = [rule for rule in rules if rule.lower() not in existing]
    if not added:
        return 0
    prior_count = len(overlay["notes"])
    overlay["notes"] = (overlay["notes"] + added)[:_OVERLAY_NOTES_CAP]
    # Report what actually persisted: notes past the cap are dropped, so
    # len(added) over-reports near the cap.
    persisted = max(0, len(overlay["notes"]) - prior_count)
    if persisted == 0:
        return 0
    if not write_overlay(overlay_target(root, repo_root, scope), overlay):
        return 0
    if scope == "repo" and repo_root is not None:
        ensure_repo_share_gitignore(repo_root)
    return persisted


def _run_owned(prompt: str, *, root: str | Path, model: str) -> str:
    from lemoncrow.pro.capabilities.owned_execution_lanes import execute_owned_prompt
    from lemoncrow.pro.capabilities.owned_execution_routing import OwnedRouteRequest, select_owned_route

    if "/" in model:
        provider, model_id = model.split("/", 1)
    else:
        provider, model_id = "", model
    if provider and model_id:
        request = OwnedRouteRequest(
            tool_name="agent",
            task_text=prompt,
            mode="explicit",
            provider=provider,
            model=model_id,
            host_agent="lemoncrow:knowledge",
        )
        allow_fallback = False
    else:
        request = OwnedRouteRequest(
            tool_name="agent",
            task_text=prompt,
            mode="auto",
            budget="cheap",
            model=model_id,
            host_agent="lemoncrow:knowledge",
        )
        allow_fallback = True
    decision = select_owned_route(root, request)
    return execute_owned_prompt(
        prompt,
        root=root,
        tool_name="agent",
        task_text=prompt,
        decision=decision,
        host_agent="lemoncrow:knowledge",
        allow_fallback=allow_fallback,
    ).output


def _run_host_cli(prompt: str, *, host: str, model: str) -> str:
    if host == "claude":
        cmd = ["claude", "--model", model, "-p", prompt] if model else ["claude", "-p", prompt]
    elif host == "codex":
        cmd = ["codex", "exec", prompt]
    elif host == "ollama":
        cmd = ["ollama", "run", model or "llama3.1", prompt]
    else:
        raise ValueError(f"unknown extraction host: {host}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.stdout or ""


def _run_backend(prompt: str, *, host: str, model: str, root: str | Path) -> str:
    if host in ("auto", "lemoncrow", "owned"):
        return _run_owned(prompt, root=root, model=model)
    return _run_host_cli(prompt, host=host, model=model)


def extract_rules(
    root: str | Path,
    repo_root: str | Path,
    *,
    host: str = "auto",
    model: str = "",
    max_items: int = _DEFAULT_MAX_ITEMS,
    max_chars: int = _DEFAULT_MAX_CHARS,
    max_spend_usd: float = _DEFAULT_MAX_SPEND_USD,
    dry_run: bool = False,
    scope: str = "repo",
    runner: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Distill review rules from .lemoncrow/lessons and merge them into the overlay.

    Spend is bounded: the cost is estimated from the prompt size before any call,
    and the run aborts (no tokens spent) when it would exceed *max_spend_usd*.
    """
    sources = gather_sources(repo_root, max_items=max_items, max_chars=max_chars)
    base: dict[str, Any] = {
        "rules": [],
        "applied": 0,
        "host": host,
        "model": model,
        "estimated_cost_usd": 0.0,
        "dry_run": dry_run,
        "sources": len(sources),
        "scope": scope,
        "overlay": str(overlay_target(root, repo_root, scope)),
    }
    if not sources:
        return {**base, "reason": "no .lemoncrow/lessons/blocks found"}
    prompt = _PROMPT_HEADER + "\n\n---\n\n".join(sources)
    estimate = preflight_cost_usd(prompt, host, model)
    base["estimated_cost_usd"] = estimate
    if estimate > max_spend_usd:
        return {
            **base,
            "reason": f"estimated ${estimate} exceeds cap ${max_spend_usd}; raise --max-spend or lower --max-items",
        }
    try:
        output = (runner or _run_backend)(prompt, host=host, model=model, root=root)
    except Exception as exc:  # noqa: BLE001 - extraction must never crash the caller
        return {**base, "error": str(exc)}
    rules = parse_rules(output)
    applied = 0 if dry_run else merge_into_overlay(root, rules, repo_root=repo_root, scope=scope)
    return {**base, "rules": rules, "applied": applied}
