"""Session-local run-and-dedup of byte-identical command outputs.

Agents re-run the same inspection commands constantly (``git status`` after
every step, a test suite in a fix loop). When a re-run produces output
byte-identical to its previous run, every byte is already in the model's
context -- shipping it again is pure token waste. This module remembers a
digest of each command's last output so the bash lane can replace an
identical re-run with a one-line ``unchanged`` marker.

Crucially this is run-and-dedup, NOT caching: execution is never skipped, so
external-state commands (``docker ps``, anything with timestamps) stay
correct by construction -- any change in their output hashes differently and
ships in full. There is no invalidation problem because there is no cache.

Only exit-0 runs are ever reported unchanged: identical *failure* output
ships in full so the agent sees the error every time.

State is process-local and resets with the MCP server. Opt out with
``LEMONCROW_BASH_UNCHANGED_DELTA=0``.
"""

from __future__ import annotations

import hashlib
import os
import threading

_ENV_ENABLED = "LEMONCROW_BASH_UNCHANGED_DELTA"

# Below this combined output size the marker saves too little to be worth the
# indirection -- small outputs ship as-is.
_MIN_CHARS = 500
# Bounded memory: when the map fills, the stalest half is dropped.
_MAX_TRACKED = 256

_lock = threading.Lock()
# key -> (output digest, monotonic sequence of last sighting)
_last: dict[str, tuple[str, int]] = {}
_seq = 0


def enabled() -> bool:
    return os.environ.get(_ENV_ENABLED, "1").strip().lower() not in {"0", "false", "no", "off"}


def _key(command: str, cwd: str | None) -> str:
    # cwd is part of the identity: `git status` in two checkouts is two commands.
    return f"{cwd or ''}\x00{' '.join(command.split())}"


def _digest(stdout: str, stderr: str, exit_code: int) -> str:
    h = hashlib.sha256()
    h.update(stdout.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(stderr.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(str(exit_code).encode("ascii"))
    return h.hexdigest()


def observe(command: str, *, cwd: str | None, stdout: str, stderr: str, exit_code: int) -> bool:
    """Record this run's output digest; True when it repeats the previous run.

    Every run is recorded (so the next comparison is always against the
    latest output), but only an exit-0 run whose combined output exceeds
    ``_MIN_CHARS`` is ever reported unchanged.
    """
    global _seq
    if not enabled():
        return False
    key = _key(command, cwd)
    digest = _digest(stdout, stderr, exit_code)
    with _lock:
        _seq += 1
        prev = _last.get(key)
        if prev is None and len(_last) >= _MAX_TRACKED:
            for stale in sorted(_last, key=lambda k: _last[k][1])[: _MAX_TRACKED // 2]:
                del _last[stale]
        _last[key] = (digest, _seq)
    if prev is None or exit_code != 0:
        return False
    if len(stdout) + len(stderr) <= _MIN_CHARS:
        return False
    return prev[0] == digest


def reset() -> None:
    """Clear process-local state (tests)."""
    global _seq
    with _lock:
        _last.clear()
        _seq = 0


__all__ = ["enabled", "observe", "reset"]
