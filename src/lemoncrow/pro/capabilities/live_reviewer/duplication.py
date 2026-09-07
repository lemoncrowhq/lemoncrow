"""Find new top-level declarations in a diff that may duplicate existing code.

Heuristic and name-based: parse the symbols a diff ADDS, then search the repo for
other definitions of the same name outside the changed files — it catches the
choice-level bug of re-implementing something that already exists. Reuses ripgrep
when present; fully
fail-open (returns an empty list on any problem).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

# Added top-level declarations across common languages (matched on + lines).
_DEF_PATTERNS = [
    re.compile(r"^\+\s*(?:async\s+)?def\s+([A-Za-z_]\w+)"),
    re.compile(r"^\+\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w+)"),
    re.compile(r"^\+\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w+)"),
    re.compile(r"^\+\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w+)\s*=\s*(?:async\s+)?(?:\(|function)"),
    re.compile(r"^\+\s*(?:pub\s+)?fn\s+([A-Za-z_]\w+)"),
]
_COMMON = frozenset(
    {"main", "setup", "run", "init", "test", "handler", "new", "default", "build", "parse", "load", "render"}
)


def added_symbols(diffs: Mapping[str, str]) -> list[str]:
    """Distinct top-level symbol names introduced by the diff (added lines)."""
    names: list[str] = []
    seen: set[str] = set()
    for diff in diffs.values():
        for line in diff.splitlines():
            if not line.startswith("+"):
                continue
            for pattern in _DEF_PATTERNS:
                match = pattern.match(line)
                if match:
                    name = match.group(1)
                    if len(name) >= 4 and name.lower() not in _COMMON and name not in seen:
                        seen.add(name)
                        names.append(name)
                    break
    return names


def _rg_defs(repo_root: Path, name: str) -> list[str]:
    pattern = rf"(def|class|function|fn|const|let|var)\s+{re.escape(name)}\b"
    try:
        result = subprocess.run(
            ["rg", "-n", "--no-heading", "-e", pattern, str(repo_root)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode not in (0, 1):  # 1 == no matches
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def find_duplications(
    repo_root: str | Path,
    diffs: Mapping[str, str],
    *,
    max_symbols: int = 15,
    max_hits: int = 3,
) -> list[str]:
    """Human-readable notes for added symbols that already exist elsewhere."""
    repo = Path(repo_root)
    changed: set[str] = set()
    for path in diffs:
        try:
            changed.add(str((repo / path).resolve()))
        except OSError:
            continue
    notes: list[str] = []
    for name in added_symbols(diffs)[:max_symbols]:
        external: list[str] = []
        for hit in _rg_defs(repo, name):
            file = hit.split(":", 1)[0]
            try:
                resolved = str(Path(file).resolve())
            except OSError:
                continue
            if resolved not in changed:
                external.append(hit)
            if len(external) >= max_hits:
                break
        if external:
            notes.append(f"{name} (added here) may already exist: " + "; ".join(external))
    return notes
