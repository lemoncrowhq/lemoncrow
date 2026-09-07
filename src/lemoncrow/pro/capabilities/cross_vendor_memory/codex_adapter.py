"""Memory adapter for OpenAI Codex.

Reads the following paths (all optional):

* ``~/.codex/memories/*.md``   consolidated session summaries
* ``~/.codex/memories/global.md``  global memory (if present)

Parse rules:
- Each ``## Heading`` block is one fact; heading text stored in
  ``raw_meta["heading"]``.
- Bullet points *directly under* a heading are emitted as sub-facts **only**
  if they are "standalone declarations": the line starts with a capital letter
  and ends with ``.`` or ``;``.
- All other content within a heading block is folded into the heading fact's
  ``content`` field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.cross_vendor_memory.base import (
    MemoryFact,
    _fact_id,
    _utcnow,
)

_VENDOR = "codex"


def _codex_root() -> Path:
    return Path.home() / ".codex"


def _candidate_files(codex_root: Path) -> list[Path]:
    memories = codex_root / "memories"
    if not memories.is_dir():
        return []
    return sorted(memories.glob("*.md"))


def _is_standalone_declaration(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    first_char = stripped[0]
    return first_char.isupper() and stripped[-1] in {".", ";"}


def _parse_codex_facts(text: str, path: Path) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    captured_at = _utcnow()
    source_kind = "codex-mem"

    current_heading: str | None = None
    heading_start: int | None = None
    heading_body_lines: list[str] = []

    def _flush_heading(end_lineno: int) -> None:
        nonlocal current_heading, heading_start, heading_body_lines
        if current_heading is None:
            return

        body = "\n".join(heading_body_lines).strip()
        content = current_heading if not body else f"{current_heading}\n{body}"
        meta: dict[str, Any] = {"heading": current_heading}
        facts.append(
            MemoryFact(
                fact_id=_fact_id(_VENDOR, content),
                vendor=_VENDOR,
                source_path=path,
                source_kind=source_kind,
                content=content,
                line_number=heading_start,
                captured_at=captured_at,
                raw_meta=meta,
            )
        )
        current_heading = None
        heading_start = None
        heading_body_lines = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if stripped.startswith("## "):
            _flush_heading(lineno - 1)
            current_heading = stripped[3:].strip()
            heading_start = lineno
            heading_body_lines = []
            continue

        if current_heading is None:
            continue

        # Bullet under heading: emit as sub-fact if it's a standalone declaration
        if stripped.startswith(("- ", "* ")):
            bullet_text = stripped[2:].strip()
            if _is_standalone_declaration(bullet_text):
                meta_sub: dict[str, Any] = {
                    "heading": current_heading,
                    "sub_fact": True,
                }
                facts.append(
                    MemoryFact(
                        fact_id=_fact_id(_VENDOR, bullet_text),
                        vendor=_VENDOR,
                        source_path=path,
                        source_kind=source_kind,
                        content=bullet_text,
                        line_number=lineno,
                        captured_at=captured_at,
                        raw_meta=meta_sub,
                    )
                )
                continue

        heading_body_lines.append(line)

    _flush_heading(0)
    return facts


class CodexAdapter:
    """Reads native Codex memory files from ``~/.codex/memories/``."""

    vendor: str = _VENDOR

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or _codex_root()

    def is_available(self) -> bool:
        return bool(_candidate_files(self._root))

    def source_paths(self) -> list[Path]:
        return _candidate_files(self._root)

    def list_facts(self) -> list[MemoryFact]:
        facts: list[MemoryFact] = []
        for path in _candidate_files(self._root):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            facts.extend(_parse_codex_facts(text, path))
        return facts


__all__ = ["CodexAdapter"]
