"""Deterministic workspace primer injected into the first owned-session user turn.

Front-loads a file tree and task-keyword-matched excerpts so the model reaches
its first evidence-backed edit with fewer discovery round trips. Pure file-system
work — no LLM calls, no index requirement.
"""

from __future__ import annotations

import re
from pathlib import Path

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}
_MAX_FILE_BYTES = 200_000
_MAX_TREE_ENTRIES = 200
_MAX_FILES_SCANNED = 400
_MAX_HITS_PER_FILE = 12
_MAX_EXCERPT_FILES = 8
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "your",
    "should",
    "must",
    "when",
    "then",
    "than",
    "them",
    "they",
    "what",
    "which",
    "into",
    "only",
    "also",
    "make",
    "made",
    "need",
    "needs",
    "task",
    "file",
    "files",
    "code",
    "test",
    "tests",
    "using",
    "use",
    "run",
    "add",
    "fix",
    "change",
    "update",
    "ensure",
    "please",
    "implement",
    "does",
    "each",
    "every",
    "after",
    "before",
    "where",
}


def _keywords(task: str, limit: int = 10) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", task):
        lowered = word.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        out.append(word)
        if len(out) >= limit:
            break
    return out


def _workspace_files(workspace: Path, cap: int = _MAX_FILES_SCANNED) -> list[Path]:
    files: list[Path] = []
    stack = [workspace]
    while stack and len(files) < cap:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in _SKIP_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                files.append(entry)
                if len(files) >= cap:
                    break
    return files


def build_task_primer(task: str, workspace: Path, *, max_chars: int = 12_000) -> str:
    """Return a primer block for *task*, or "" when nothing useful was found.

    Never raises: a primer is an optimization, not a contract — any filesystem
    surprise degrades to the un-primed behavior.
    """
    try:
        return _build(task, workspace, max_chars)
    except Exception:  # Primer failure must never break a run.
        return ""


def _build(task: str, workspace: Path, max_chars: int) -> str:
    files = _workspace_files(workspace)
    if not files:
        return ""
    rel_paths = sorted(str(f.relative_to(workspace)) for f in files)
    tree = "\n".join(rel_paths[:_MAX_TREE_ENTRIES])

    excerpts: list[tuple[int, str]] = []
    keywords = _keywords(task)
    if keywords:
        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
        for f in files:
            try:
                if f.stat().st_size > _MAX_FILE_BYTES:
                    continue
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            hits: list[str] = []
            for lineno, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"  {lineno}: {line.strip()[:200]}")
                    if len(hits) >= _MAX_HITS_PER_FILE:
                        break
            if hits:
                rel = f.relative_to(workspace)
                excerpts.append((len(hits), f"{rel}:\n" + "\n".join(hits)))
        excerpts.sort(key=lambda item: -item[0])

    sections = [
        "## Workspace primer (auto-generated)",
        "Pre-assembled context: use it to choose your first targeted reads and "
        "edits instead of re-listing or re-searching the workspace. Verify line "
        "numbers with a read before editing.",
        "### Files",
        tree,
    ]
    if excerpts:
        sections.append("### Task-keyword matches")
        sections.extend(snippet for _, snippet in excerpts[:_MAX_EXCERPT_FILES])
    return "\n\n".join(sections)[:max_chars]


__all__ = ["build_task_primer"]
