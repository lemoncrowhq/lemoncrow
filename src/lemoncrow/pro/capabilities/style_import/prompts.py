"""Fixed prompt template for style-guide imports."""

from __future__ import annotations

from typing import Any

STYLE_IMPORT_PROMPT_VERSION = "style-guide-import-v1"

STYLE_IMPORT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "procedural": {"type": "boolean"},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "triggers": {"type": "array", "items": {"type": "string"}},
        "procedure": {"type": "array", "items": {"type": "string"}},
        "verification": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["procedural", "title", "body", "procedure", "confidence"],
}

STYLE_IMPORT_SYSTEM_PROMPT = f"""\
You extract procedural engineering rules from Markdown for LemonCrow Playbooks.
Prompt version: {STYLE_IMPORT_PROMPT_VERSION}.

Return only JSON matching this contract:
- procedural: true only when the chunk tells engineers what to do, avoid, verify, or gate.
- title: concise rule title, no marketing copy.
- body: one paragraph in the project's Playbook style.
- triggers: short phrases that should cause retrieval.
- procedure: concrete steps an AI coding agent can follow.
- verification: observable checks, commands, or review evidence.
- confidence: 0.0 to 1.0.

If the chunk is narrative, status-only, install-only, or non-procedural, return
procedural=false with empty title/body/procedure/verification.
Do not invent company rules that are not supported by the chunk.
"""


def build_messages(*, file_path: str, chunk_range: tuple[int, int], text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STYLE_IMPORT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source file: {file_path}\nLine range: {chunk_range[0]}-{chunk_range[1]}\n\nMarkdown chunk:\n{text}"
            ),
        },
    ]


__all__ = [
    "STYLE_IMPORT_PROMPT_VERSION",
    "STYLE_IMPORT_RESPONSE_SCHEMA",
    "STYLE_IMPORT_SYSTEM_PROMPT",
    "build_messages",
]
