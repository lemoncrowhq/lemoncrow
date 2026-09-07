"""Machine-global smart_state.json persistence (atomic read/write + POSIX flock).

Foundational persistence substrate for cumulative counters shared across sibling
MCP processes. No ``mcp_server`` import; leaf module.

Extracted verbatim from ``mcp_server.py`` (behaviour-preserving); ``mcp_server``
re-exports these names for backward compatibility.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from lemoncrow.gateway.adapters.mcp.session_state import _lemoncrow_root

logger = logging.getLogger(__name__)

# Serializes read-modify-write of the machine-global counters across threads.
_STATE_LOCK = threading.RLock()

# Per-call token-savings thread-local: written by tool handlers (incl. bash),
# reset + read by the dispatcher's savings accounting. One shared instance.
_tool_call_tokens_saved: threading.local = threading.local()


def _smart_state_path() -> Path:
    return _lemoncrow_root() / "smart_state.json"


def _read_smart_state() -> dict[str, Any]:
    path = _smart_state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}


def _write_smart_state(state: dict[str, Any]) -> None:
    try:
        path = _smart_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a torn smart_state.json would corrupt cumulative
        # counters, so stage to a temp file and os.replace into place.
        tmp_path: str | None = None
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(state, handle, indent=2)
            tmp_path = handle.name
        Path(tmp_path).replace(path)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        logger.warning("Suppressed exception while writing smart_state", exc_info=True)


#: How long :func:`_acquire_smart_state_flock` waits for the machine-global lock
#: before giving up. The lock file lives under the LemonCrow root, so it is held
#: across every MCP process and daemon on the machine -- a blocking acquisition
#: makes one stalled holder park every tool call everywhere. Read fresh per call
#: so it can be tuned (or set to 0) without a restart.
_DEFAULT_FLOCK_TIMEOUT_S = 5.0


def _smart_state_flock_timeout() -> float:
    raw = os.environ.get("LEMONCROW_SMART_STATE_LOCK_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_FLOCK_TIMEOUT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_FLOCK_TIMEOUT_S


def _acquire_smart_state_flock() -> Any:
    """Best-effort POSIX exclusive flock on smart_state's sidecar lock file so a
    sibling MCP process can't lose-update the machine-global counters. Returns an
    open handle the caller must release, or None where flock is unavailable or
    the lock stayed busy past the timeout.

    Bounded on purpose. A blocking ``LOCK_EX`` waits forever, and this lock is
    machine-global: any process that stalls while holding it parks every tool
    call in every other MCP process behind it, with no error and no deadline --
    the call has already done its real work and is queued only to record a
    counter. Giving up degrades to the same unsynchronised path non-POSIX
    platforms already take: worst case a lost counter update, which is cheaper
    than an unbounded stall.
    """
    try:
        import fcntl
    except ImportError:
        return None
    try:
        p = _smart_state_path()
        lock_path = p.parent / (p.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "w", encoding="utf-8")
    except OSError:
        return None
    deadline = time.monotonic() + _smart_state_flock_timeout()
    delay = 0.005
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.debug("smart_state flock busy past timeout; proceeding unsynchronised")
                with contextlib.suppress(OSError):
                    handle.close()
                return None
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.1)


def _release_smart_state_flock(handle: Any) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass
    with contextlib.suppress(OSError):
        handle.close()
