"""Shared path-safety constants and helpers for the edit module.

Centralises the protected-directory set that rich_edit enforces on every write.
"""

from __future__ import annotations

from pathlib import Path

#: Directory names that must never be modified by any edit tool.
PROTECTED_PARTS: frozenset[str] = frozenset({".git", ".lemoncrow", "node_modules", ".venv"})


def check_protected(path: Path, raw: str = "") -> None:
    """Raise :class:`ValueError` if *path* contains a protected directory component.

    Args:
        path: Resolved absolute path to check.
        raw:  Original user-supplied path string used in the error message.
              Falls back to ``str(path)`` when omitted.
    """
    label = raw or str(path)
    if any(part in PROTECTED_PARTS for part in path.parts):
        raise ValueError(f"protected path denied: {label}")
