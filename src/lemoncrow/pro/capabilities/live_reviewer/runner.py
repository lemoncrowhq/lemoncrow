"""Run a code review out-of-band via owned execution and parse the verdict."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.live_reviewer.duplication import find_duplications
from lemoncrow.pro.capabilities.live_reviewer.knowledge import collect_review_context
from lemoncrow.pro.capabilities.live_reviewer.settings import (
    ReviewerSettings,
    split_provider_model,
)
from lemoncrow.pro.capabilities.owned_execution_lanes import execute_owned_prompt
from lemoncrow.pro.capabilities.owned_execution_routing import (
    OwnedRouteRequest,
    select_owned_route,
)

_VERDICT_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

_REVIEW_CONTRACT = (
    "You are an adversarial code reviewer. Find what is wrong; do not validate "
    "that work was done. Apply the verification ladder (existence -> substantive "
    "-> wired -> data flow). Default to NEEDS_FIX: a DONE verdict requires positive "
    "proof every change is correct.\n\n"
    "Review ONLY the diff(s) below. End with exactly one fenced json block and nothing "
    "after it. The block is an object with: verdict (DONE or NEEDS_FIX), checklist (one "
    "line), missing (bulleted gaps, empty when DONE), and findings (a list).\n"
    "Each findings entry is an object with a type field:\n"
    "- type patch: fields file, old_string (verbatim current text), new_string (the fix), "
    "and reason. Use ONLY for high-confidence mechanical fixes (wrong constant, missing "
    "import, typo); old_string must match the file verbatim so it can be auto-applied.\n"
    "- type nudge: fields anchor (file and line), severity (Blocker or Warning), and reason "
    "(the concern plus a concrete fix). Use for anything needing human judgment.\n"
    "Omit silent entries; findings may be empty. If a '## Possible duplications' section is "
    "present, verify each candidate and flag any real duplication as a nudge."
)


def _git_diff(path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _branch_diffs(repo_root: str | Path, base: str) -> dict[str, str]:
    """Per-file diff of the branch vs *base* (merge-base via three-dot)."""
    try:
        names = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", f"{base}...HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return {}
    out: dict[str, str] = {}
    for file in (line for line in names.stdout.splitlines() if line.strip()):
        try:
            one = subprocess.run(
                ["git", "-C", str(repo_root), "diff", f"{base}...HEAD", "--", file],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if one.stdout.strip():
            out[file] = one.stdout.strip()
    return out


def _build_prompt(diffs: Mapping[str, str], kb: str = "", duplications: Sequence[str] | None = None) -> str:
    parts = [_REVIEW_CONTRACT]
    if kb:
        parts.append(kb)
    if duplications:
        parts.append("## Possible duplications (added primitives that may already exist)")
        parts.extend(f"- {note}" for note in duplications)
        parts.append("")
    parts.append("## Diffs under review\n")
    for path, diff in diffs.items():
        parts.append(f"### {path}\n```diff\n{diff}\n```\n")
    return "\n".join(parts)


def parse_verdict(text: str) -> dict[str, Any]:
    """Extract the final fenced JSON verdict. Safe default, never raises."""
    for block in reversed(_VERDICT_RE.findall(text or "")):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            verdict = str(obj.get("verdict") or "").strip().upper()
            findings = obj.get("findings")
            return {
                "verdict": "DONE" if verdict == "DONE" else "NEEDS_FIX",
                "checklist": str(obj.get("checklist") or ""),
                "missing": str(obj.get("missing") or ""),
                "findings": findings if isinstance(findings, list) else [],
            }
    return {"verdict": "ERROR", "checklist": "", "missing": "review output could not be parsed", "findings": []}


def run_review(
    session_id: str,
    mode: str,
    paths: Sequence[str],
    settings: ReviewerSettings,
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    base: str | None = None,
) -> dict[str, Any]:
    """Review a diff and return a verdict record.

    Working-tree diff of ``paths`` by default; when ``base`` is given, review the
    whole branch diff (``base...HEAD``) instead — the PR-style mode.
    """
    repo_root = os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
    if base:
        diffs = _branch_diffs(repo_root, base)
    else:
        diffs = {path: _git_diff(path) for path in paths if path}
        diffs = {path: diff for path, diff in diffs.items() if diff}
    record: dict[str, Any] = {"mode": mode, "paths": list(diffs.keys()), "session_id": session_id}
    if not diffs:
        return {**record, "verdict": "DONE", "checklist": "no diff to review", "missing": "", "findings": []}

    duplications = find_duplications(repo_root, diffs)
    kb = collect_review_context(root, repo_root)
    if settings.agentic:
        from lemoncrow.pro.capabilities.live_reviewer.agentic import run_agentic_review

        agentic_verdict = run_agentic_review(
            repo_root=repo_root,
            diffs=diffs,
            contract=_REVIEW_CONTRACT,
            kb=kb,
            model=settings.model_for(mode),
        )
        if agentic_verdict is not None:
            if duplications:
                agentic_verdict["duplications"] = duplications
            agentic_verdict["agentic"] = True
            return {**record, **agentic_verdict}
    prompt = _build_prompt(diffs, kb, duplications)
    provider, model_id = split_provider_model(settings.model_for(mode))
    if provider and model_id:
        request = OwnedRouteRequest(
            tool_name="agent",
            task_text=prompt,
            mode="explicit",
            provider=provider,
            model=model_id,
            host_agent="lemoncrow:review",
        )
        allow_fallback = False
    else:
        request = OwnedRouteRequest(
            tool_name="agent",
            task_text=prompt,
            mode="auto",
            budget="best" if mode == "deep" else "cheap",
            model=model_id,
            host_agent="lemoncrow:review",
        )
        allow_fallback = True

    decision = select_owned_route(root, request, env=env)
    result = execute_owned_prompt(
        prompt,
        root=root,
        tool_name="agent",
        task_text=prompt,
        decision=decision,
        host_agent="lemoncrow:review",
        allow_fallback=allow_fallback,
    )
    verdict = parse_verdict(result.output)
    if duplications:
        verdict["duplications"] = duplications
    return {**record, **verdict}
