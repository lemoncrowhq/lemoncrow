"""LLM-backed semantic intent review for scoped post-edit verification."""

from __future__ import annotations

import json
from pathlib import Path

from lemoncrow.infra.internal_llm import InternalLLMError, chat

from ..counterexample import Counterexample

_MAX_INTENT_CHARS = 2000
_MAX_FILES = 5
_MAX_FILE_CHARS = 4000


def _scoped_sources(files: list[str], *, cwd: Path) -> list[dict[str, str]]:
    snippets: list[dict[str, str]] = []
    for file_path in files:
        candidate = Path(file_path)
        resolved = candidate if candidate.is_absolute() else cwd / candidate
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            content = resolved.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        snippets.append(
            {
                "file_path": file_path,
                "content": content[:_MAX_FILE_CHARS],
            }
        )
        if len(snippets) >= _MAX_FILES:
            break
    return snippets


def run_semantic_review(files: list[str], task_intent: str, *, cwd: Path) -> list[Counterexample]:
    intent = task_intent.strip()
    if not intent:
        return []

    scoped_sources = _scoped_sources(files, cwd=cwd)
    if not scoped_sources:
        return []

    try:
        response = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON with mismatches: []. "
                        "Only report semantic mismatches when the edited files appear inconsistent "
                        "with the user's stated coding intent. Be conservative."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_intent": intent[:_MAX_INTENT_CHARS],
                            "files": scoped_sources,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            json_schema={"type": "object"},
        )
    except InternalLLMError:
        return []

    if not isinstance(response, dict):
        return []

    mismatches = response.get("mismatches")
    if not isinstance(mismatches, list):
        return []

    counterexamples: list[Counterexample] = []
    for raw in mismatches:
        if not isinstance(raw, dict):
            continue
        file_path = str(raw.get("file_path") or scoped_sources[0]["file_path"])
        line_raw = raw.get("line")
        line = line_raw if isinstance(line_raw, int) and line_raw > 0 else None
        diagnostic = str(raw.get("diagnostic") or "").strip()
        actual = str(raw.get("actual") or "").strip() or None
        if not diagnostic:
            continue
        counterexamples.append(
            Counterexample(
                check="semantic",
                severity="error",
                file_path=file_path,
                line=line,
                diagnostic=diagnostic,
                expected=intent,
                actual=actual,
            )
        )
    return counterexamples
