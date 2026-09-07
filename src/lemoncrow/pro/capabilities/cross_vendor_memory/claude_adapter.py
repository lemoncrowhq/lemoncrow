"""Memory adapter for Anthropic Claude (Code / Desktop).

Reads the following paths (all optional):

* ``~/.claude/CLAUDE.md``                            global instructions
* ``~/.claude/projects/<project>/CLAUDE.md``         per-project instructions
* ``~/.claude/projects/<project>/MEMORY.md``         auto-memory
* ``~/.claude/projects/<project>/memory/MEMORY.md``  alternate auto-memory path
* ``~/.claude/projects/<project>/session_memory/*.md``  session memory output

Parse rules:
- Bullet lines (``- text`` or ``* text``) → one fact per line.
- Headings are **not** facts; they update ``raw_meta["section"]``.
- Fenced code blocks (``` ... ```) → one multi-line fact.
- All other lines are skipped (plain prose / blank lines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.cross_vendor_memory.base import (
    MemoryFact,
    _fact_id,
    _utcnow,
)

_VENDOR = "claude"


def _claude_root() -> Path:
    return Path.home() / ".claude"


def _project_dirs(claude_root: Path) -> list[Path]:
    projects = claude_root / "projects"
    if not projects.is_dir():
        return []
    return [d for d in projects.iterdir() if d.is_dir()]


def _candidate_files(claude_root: Path) -> list[tuple[Path, str]]:
    """Return ``(path, source_kind)`` pairs for every Claude memory file candidate."""
    result: list[tuple[Path, str]] = []

    global_md = claude_root / "CLAUDE.md"
    result.append((global_md, "claude-md-global"))

    for project_dir in _project_dirs(claude_root):
        result.append((project_dir / "CLAUDE.md", "claude-md"))
        result.append((project_dir / "MEMORY.md", "auto-memory"))
        result.append((project_dir / "memory" / "MEMORY.md", "auto-memory"))
        session_mem = project_dir / "session_memory"
        if session_mem.is_dir():
            for f in sorted(session_mem.glob("*.md")):
                result.append((f, "session-memory"))

    return result


def _parse_markdown_facts(
    text: str,
    path: Path,
    source_kind: str,
) -> list[MemoryFact]:
    """Parse *text* from a markdown file into a list of ``MemoryFact``s."""
    facts: list[MemoryFact] = []
    captured_at = _utcnow()
    current_section: str | None = None
    in_code_block = False
    code_block_lines: list[str] = []
    code_block_start: int | None = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        # Toggle fenced code block
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_lines = []
                code_block_start = lineno
            else:
                # Close block — emit as one fact
                content = "\n".join(code_block_lines).strip()
                if content:
                    meta: dict[str, Any] = {}
                    if current_section:
                        meta["section"] = current_section
                    facts.append(
                        MemoryFact(
                            fact_id=_fact_id(_VENDOR, content),
                            vendor=_VENDOR,
                            source_path=path,
                            source_kind=source_kind,
                            content=content,
                            line_number=code_block_start,
                            captured_at=captured_at,
                            raw_meta=meta,
                        )
                    )
                in_code_block = False
                code_block_lines = []
                code_block_start = None
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Headings update current section context
        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
            continue

        # Bullet points → facts
        if stripped.startswith(("- ", "* ", "-\t", "*\t")):
            content = stripped[2:].strip()
            if content:
                meta = {}
                if current_section:
                    meta["section"] = current_section
                facts.append(
                    MemoryFact(
                        fact_id=_fact_id(_VENDOR, content),
                        vendor=_VENDOR,
                        source_path=path,
                        source_kind=source_kind,
                        content=content,
                        line_number=lineno,
                        captured_at=captured_at,
                        raw_meta=meta,
                    )
                )

    return facts


class ClaudeAdapter:
    """Reads native Claude memory files from ``~/.claude/``."""

    vendor: str = _VENDOR

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or _claude_root()

    def is_available(self) -> bool:
        return self._root.is_dir() and any(p.exists() for p, _ in _candidate_files(self._root))

    def source_paths(self) -> list[Path]:
        return [p for p, _ in _candidate_files(self._root)]

    def list_facts(self) -> list[MemoryFact]:
        facts: list[MemoryFact] = []
        for path, source_kind in _candidate_files(self._root):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            facts.extend(_parse_markdown_facts(text, path, source_kind))
        return facts


__all__ = ["ClaudeAdapter"]
