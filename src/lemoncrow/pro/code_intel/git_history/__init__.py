"""Bootstrap helpers for the isolated git-history substrate."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from lemoncrow.core.capabilities.code_intel_contract import GitHistoryBootstrapError

try:
    _PYGIT2: ModuleType | None = import_module("pygit2")
except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
    _PYGIT2 = None
    _PYGIT2_IMPORT_ERROR: ImportError | None = exc
else:
    _PYGIT2_IMPORT_ERROR = None


def require_pygit2() -> ModuleType:
    """Return the pinned pygit2 module or raise a clear bootstrap error."""

    if _PYGIT2 is None:
        raise GitHistoryBootstrapError(
            "pygit2 is required for lemoncrow.pro.code_intel.git_history; "
            "install the pinned Phase 4 dependency and retry. "
            "GitPython and subprocess fallbacks are intentionally unsupported."
        ) from _PYGIT2_IMPORT_ERROR
    return _PYGIT2


__all__ = ["GitHistoryBootstrapError", "require_pygit2"]
