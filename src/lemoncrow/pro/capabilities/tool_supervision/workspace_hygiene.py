"""Workspace hygiene snapshot/report for bench-style runs.

File-hygiene verifiers (e.g. ``os.listdir(dir) == [expected_file]``) fail on
leftover scratch outputs even when the solution is correct. Snapshot the tree
before solving, then report new files that look like build/debug residue.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

# Directories that are never agent scratch yet can be enormous (VCS internals,
# virtualenvs, dependency trees, tool caches). They are pruned from the walk so
# snapshot_workspace stays fast on a real checkout — rglob would otherwise
# descend into .venv/node_modules/.git and stat hundreds of thousands of files.
# NOTE: build/, __pycache__/, and *.egg-info/ are deliberately NOT skipped —
# they are the scratch-residue patterns scratch_leftovers must still detect.
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        ".venv-build",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
    }
)

_SCRATCH_PATTERNS: tuple[str, ...] = (
    "*.o",
    "*.obj",
    "*.pyc",
    "*.pyo",
    "*.tmp",
    "*.temp",
    "*.log",
    "*.swp",
    "*.bak",
    "a.out",
    "core",
    "__pycache__/*",
    "build/*",
    "*.egg-info/*",
)


def snapshot_workspace(root: Path) -> frozenset[str]:
    """Relative paths of all files under *root*, excluding VCS/env/cache dirs.

    Uses ``os.walk`` with in-place directory pruning so skip-dirs are never
    descended into (``rglob`` cannot prune and would stat every file under a
    .venv/node_modules tree). Relative paths are built with a single per-dir
    ``relpath`` plus string concat rather than a ``Path.relative_to`` per file.
    """
    if not root.is_dir():
        return frozenset()
    root_str = str(root)
    paths: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_str):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if dirpath == root_str:
            prefix = ""
        else:
            prefix = os.path.relpath(dirpath, root_str).replace(os.sep, "/") + "/"
        for name in filenames:
            paths.add(prefix + name)
    return frozenset(paths)


def scratch_leftovers(root: Path, before: frozenset[str]) -> list[str]:
    """New files since *before* that match scratch/build-residue patterns."""
    new_paths = snapshot_workspace(root) - before
    flagged = [
        rel
        for rel in new_paths
        if any(
            fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)
            for pattern in _SCRATCH_PATTERNS
        )
    ]
    return sorted(flagged)
