"""Within-session content dedup for read-style MCP tools.

When a ``read`` / ``search`` / ``grep`` / ``explore`` result is byte-identical
to content already returned earlier in the *same* session, emit a short stub
pointer instead of re-paying to put the same bytes back into the context window.
The pointer is self-healing: it tells the caller to re-request with ``force=true``
when the content is not in its context. This matters for subagents — they run in
a *separate* context window but share this session id (one stdio MCP process, no
per-caller identity; see anthropics/claude-code#32514), so a subagent can receive
a pointer for content it never got; the force=true cue lets it recover.

For single-file ``read`` results that *changed* since the previous read (the
common re-read-after-edit case, where exact dedup can never fire), ``delta_for``
emits a unified diff against the text previously returned instead of the full
body — the model already holds the prior version in its transcript.

**Scope**: MCP tool-output level, within one session, exact SHA-256 hash match.
Do not confuse with ``context_compression.deduplication``, which runs inside
the compression pipeline and uses edit-distance / MinHash for *near*-duplicate
collapsing of tool outputs during sleeptime summarisation.

Correctness hinges on one invariant: in a Claude Code session, returned content
stays in context until a **compaction** (or /clear) drops it. So the only reset
signal we need is the session's ``compaction_epoch`` (bumped by the PostCompact
hook): when it changes we clear the seen-set, because the compacted summary may
no longer contain the deduped content.

Fail-open by construction: callers wrap usage in suppression and a stub is only
ever emitted on an exact hash match.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Not worth stubbing small results (also matches the cache_control threshold).
_MIN_DEDUP_CHARS = 4096
# Delta must be meaningfully smaller than a full re-emit to be worth the
# indirection; otherwise just resend the body.
_DELTA_MAX_RATIO = 0.5
# Bound per-session memory for stored read bodies (LRU-evicted).
_MAX_TRACKED_RESOURCES = 64
_MAX_TRACKED_CONTENT_CHARS = 2_000_000


@dataclass
class _SessionDedup:
    epoch: int = 0
    calls: int = 0
    seen: dict[str, int] = field(default_factory=dict)  # content hash -> call ordinal
    last_read: dict[str, str] = field(default_factory=dict)  # resource key -> last emitted text


class ContextDedup:
    """In-memory per-session dedup registry (one MCP server == one session)."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionDedup] = {}
        self._lock = threading.Lock()

    def _session(self, session_id: str, epoch: int) -> _SessionDedup:
        st = self._sessions.get(session_id)
        if st is None or st.epoch != epoch:
            st = _SessionDedup(epoch=epoch)  # epoch change == compaction == reset
            self._sessions[session_id] = st
        return st

    def _record(self, st: _SessionDedup, content_hash: str) -> int:
        ordinal = st.seen.get(content_hash)
        if ordinal is None:
            st.calls += 1
            st.seen[content_hash] = st.calls
            ordinal = st.calls
        return ordinal

    def stub_for(
        self,
        *,
        session_id: str,
        content: str,
        epoch: int,
        force: bool,
        salt: str = "",
    ) -> tuple[str, int] | None:
        """Return ``(stub_text, chars_saved)`` for a duplicate, else ``None``.

        Records the content either way (so a later non-forced identical call can
        dedup). Returns ``None`` — i.e. keep the original — when forced, when the
        content is too small to bother, or when this content is new this epoch.

        ``salt`` scopes the fingerprint to the request identity (e.g. the read's
        path+range): single-entry reads no longer embed the path in the rendered
        text, so two different files with identical bytes must not cross-dedup.
        """
        if not session_id or len(content) < _MIN_DEDUP_CHARS:
            return None
        content_hash = _hash(salt + "\x00" + content) if salt else _hash(content)
        with self._lock:
            st = self._session(session_id, epoch)
            seen_ordinal = st.seen.get(content_hash)
            if force or seen_ordinal is None:
                self._record(st, content_hash)
                return None
        stub = f"[dedup] =read #{seen_ordinal} ({len(content)} chars); not in context? re-read force=true"
        return stub, len(content) - len(stub)

    def delta_for(
        self,
        *,
        session_id: str,
        resource: str,
        content: str,
        epoch: int,
        force: bool,
    ) -> tuple[str, int] | None:
        """Return ``(delta_text, chars_saved)`` vs the last read of *resource*.

        Records *content* as the new baseline either way. Returns ``None`` (emit
        the full body) when forced, on first read, when content is small, or
        when the diff is not meaningfully smaller than the body itself.
        """
        if not session_id or not resource:
            return None
        with self._lock:
            st = self._session(session_id, epoch)
            previous = st.last_read.get(resource)
            if len(content) <= _MAX_TRACKED_CONTENT_CHARS:
                st.last_read.pop(resource, None)  # re-insert for LRU ordering
                st.last_read[resource] = content
                while len(st.last_read) > _MAX_TRACKED_RESOURCES:
                    st.last_read.pop(next(iter(st.last_read)))
            else:
                st.last_read.pop(resource, None)
        if force or previous is None or previous == content or len(content) < _MIN_DEDUP_CHARS:
            return None
        import difflib

        diff_lines = list(
            difflib.unified_diff(
                previous.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile="previous read",
                tofile="current",
                n=3,
            )
        )
        if not diff_lines:
            return None
        diff_text = "".join(diff_lines)
        header = f"[delta] {resource} ({len(content)} chars), diff vs last read; force=true for full\n"
        delta = header + diff_text
        if len(delta) > len(content) * _DELTA_MAX_RATIO:
            return None
        return delta, len(content) - len(delta)


_REGISTRY = ContextDedup()


def registry() -> ContextDedup:
    return _REGISTRY


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _session_state_path() -> Path | None:
    workspace = os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
    try:
        from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

        return resolve_workspace_store_dir(workspace_root=Path(workspace)) / "session_state.json"
    except OSError:
        return None


def current_epoch() -> int:
    """Read the session's compaction epoch from session_state (0 when absent)."""
    path = _session_state_path()
    if path is None or not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return int(data.get("compaction_epoch", 0) or 0)
    except (OSError, ValueError, TypeError):
        return 0
    return 0
