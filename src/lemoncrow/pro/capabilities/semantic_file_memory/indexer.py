"""Content-addressed file index using SHA-256.

Uses SHA-256 of file content (not mtime) so the cache survives git operations,
Docker volume remounts, touch commands, and any other mtime-invalidating events.
Provides atomic writes and optional LRU eviction.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

_MAX_CACHE_ENTRIES = 2000
_CACHE_FILENAME = "semantic_file_index.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: str) -> None:
    """Write to a temp file then rename for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".~tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class FileIndex:
    """
    SHA-256-keyed cache of semantic file analyses.

    Cache entries are keyed by file path; hit/miss is determined by comparing
    the stored ``content_hash`` against the current file's SHA-256.  This is
    fully reliable across git checkouts, Docker volume mounts, and rsync.
    """

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / _CACHE_FILENAME

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"v": 2, "files": {}}
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            # Upgrade v1 caches (mtime-based) to v2 (hash-based)
            if data.get("v", 1) < 2:
                return {"v": 2, "files": {}}
            import typing

            return typing.cast(dict[str, Any], data)
        except (OSError, json.JSONDecodeError):
            return {"v": 2, "files": {}}

    def _save(self, state: dict[str, Any]) -> None:
        files = state.get("files", {})
        if len(files) > _MAX_CACHE_ENTRIES:
            # Evict by write recency (oldest ``cached_at`` first); legacy
            # entries without the field are treated as oldest. This keeps
            # actively re-analyzed files cached regardless of path ordering.
            def _recency(k: str) -> float:
                entry = files[k]
                return float(entry.get("cached_at", 0.0)) if isinstance(entry, dict) else 0.0

            sorted_keys = sorted(files, key=_recency)
            to_evict = sorted_keys[: len(files) - _MAX_CACHE_ENTRIES]
            for k in to_evict:
                del files[k]
        _atomic_write(self._path, json.dumps(state, indent=2))

    def content_hash(self, path: Path) -> str:
        """Return SHA-256 hex digest of path's content (empty string if unreadable)."""
        try:
            return _sha256(path.read_bytes())
        except OSError:
            return ""

    def get(self, path: Path) -> dict[str, Any] | None:
        """Return cached entry if content hash matches, else None."""
        state = self._load()
        entry = state.get("files", {}).get(str(path))
        if not isinstance(entry, dict):
            return None
        stored_hash = entry.get("content_hash", "")
        if not stored_hash:
            return None
        current_hash = self.content_hash(path)
        if current_hash != stored_hash:
            return None
        return entry

    def put(self, path: Path, payload: dict[str, Any]) -> None:
        """Store payload for path, keyed by current content hash."""
        ch = self.content_hash(path)
        state = self._load()
        state.setdefault("files", {})[str(path)] = {**payload, "content_hash": ch, "cached_at": time.time()}
        self._save(state)

    def invalidate(self, path: Path) -> None:
        """Remove a single cached entry."""
        state = self._load()
        state.get("files", {}).pop(str(path), None)
        self._save(state)

    def all_entries(self) -> dict[str, dict[str, Any]]:
        """Return all valid cache entries (no freshness check — for search)."""
        return dict(self._load().get("files", {}))

    def build_reverse_deps(self) -> dict[str, list[str]]:
        """
        Build a reverse dependency map from the cache.

        Returns: { file_path: [list of files that import this file] }
        """
        entries = self.all_entries()
        rdeps: dict[str, list[str]] = {}
        for src_path, entry in entries.items():
            for dep in entry.get("dependency_map", []):
                rdeps.setdefault(dep, []).append(src_path)
        return rdeps
