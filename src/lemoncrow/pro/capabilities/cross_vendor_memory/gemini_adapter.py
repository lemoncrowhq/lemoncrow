"""Memory adapter for Google Gemini CLI.

Reads GEMINI.md files in priority order (most specific wins for dedup, but
all are returned; callers decide override semantics):

1. ``~/.gemini/GEMINI.md``            global
2. ``<repo-root>/GEMINI.md``           project (walk up from cwd looking for .git/)
3. ``<cwd>/GEMINI.md``                 subdirectory (skipped when same as repo root)

Parse rules (same as Claude adapter):
- Bullet lines (``- text`` / ``* text``) → one fact per line.
- Headings update ``raw_meta["section"]``; not emitted as facts.
- Fenced code blocks → one multi-line fact.
- All other lines skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lemoncrow.pro.capabilities.cross_vendor_memory.base import (
    MemoryFact,
    _fact_id,
    _utcnow,
)

_VENDOR = "gemini"


def _gemini_global_root() -> Path:
    return Path.home() / ".gemini"


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from *start* until we find a ``.git`` directory."""
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _candidate_files(
    cwd: Path | None = None,
    global_root: Path | None = None,
) -> list[tuple[Path, str]]:
    """Return ``(path, source_kind)`` for each potential GEMINI.md file."""
    cwd = (cwd or Path.cwd()).resolve()
    global_root = global_root or _gemini_global_root()
    result: list[tuple[Path, str]] = []

    global_file = global_root / "GEMINI.md"
    result.append((global_file, "gemini-md-global"))

    repo_root = _find_repo_root(cwd)
    if repo_root is not None:
        project_file = repo_root / "GEMINI.md"
        result.append((project_file, "gemini-md-project"))

        subdir_file = cwd / "GEMINI.md"
        if cwd != repo_root and subdir_file != project_file:
            result.append((subdir_file, "gemini-md-subdir"))

    return result


def _parse_gemini_facts(
    text: str,
    path: Path,
    source_kind: str,
) -> list[MemoryFact]:
    """Parse *text* from a GEMINI.md file into a list of ``MemoryFact``s."""
    facts: list[MemoryFact] = []
    captured_at = _utcnow()
    current_section: str | None = None
    in_code_block = False
    code_block_lines: list[str] = []
    code_block_start: int | None = None

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_block_lines = []
                code_block_start = lineno
            else:
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

        if stripped.startswith("#"):
            current_section = stripped.lstrip("#").strip()
            continue

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


class GeminiAdapter:
    """Reads native Gemini CLI memory files (GEMINI.md hierarchy)."""

    vendor: str = _VENDOR

    def __init__(
        self,
        cwd: Path | None = None,
        global_root: Path | None = None,
    ) -> None:
        self._cwd = cwd
        self._global_root = global_root

    def is_available(self) -> bool:
        return any(p.exists() for p, _ in _candidate_files(self._cwd, self._global_root))

    def source_paths(self) -> list[Path]:
        return [p for p, _ in _candidate_files(self._cwd, self._global_root)]

    def list_facts(self) -> list[MemoryFact]:
        facts: list[MemoryFact] = []
        for path, source_kind in _candidate_files(self._cwd, self._global_root):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            facts.extend(_parse_gemini_facts(text, path, source_kind))
        return facts


__all__ = ["GeminiAdapter"]
