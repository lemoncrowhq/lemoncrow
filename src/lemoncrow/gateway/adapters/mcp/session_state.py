"""MCP process identity + session-registration file + managed-bash ownership.

Foundational session-state substrate shared by the dispatch loop and the tool
handlers. No ``mcp_server`` import, so any tool module can depend on it.

Extracted verbatim from ``mcp_server.py`` (behaviour-preserving); ``mcp_server``
re-exports these names for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid as _uuid_mod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Per-process unique id; the SessionStart hook writes the Claude session UUID +
# model into the registration file named by it.
_MCP_ID: str = f"lemoncrow-{_uuid_mod.uuid4().hex[:16]}"


def _lemoncrow_root() -> Path:
    from lemoncrow.core.foundation.paths import default_store_root

    return Path(os.environ.get("LEMONCROW_ROOT", str(default_store_root())))


_MCP_SESSION_FILE_LOCK = threading.Lock()


def _mcp_session_file() -> Path:
    """Path to this MCP process's registration file.

    Written at startup; SessionStart hook writes claude_session_id + model into it.
    """
    return _lemoncrow_root() / "mcp_sessions" / f"{_MCP_ID}.json"


def pid_is_running(pid: int) -> bool:
    """True only for a process that is still executing — zombies are not.

    ``os.kill(pid, 0)`` succeeds for a zombie (exited, but its parent has not
    ``wait()``ed yet), so a signal probe alone reports leaked children of an
    agent host as live servers for as long as that host runs. On Linux the
    state field in ``/proc/<pid>/stat`` settles it; elsewhere the probe stands.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[-1].split(maxsplit=1)[0]
    except (OSError, IndexError):
        return True  # not Linux, or raced with exit: trust the signal probe
    return state != "Z"


def mcp_pid_is_live(pid: int) -> bool:
    """True only if ``pid`` is still *this* MCP server, not a recycled PID.

    Registrations outlive crashed servers, and the kernel reuses PIDs — a
    5-day-old registration was found pointing at a freshly started
    ``gitstatusd``, which a liveness probe alone reports as "running". The
    command line settles identity: only a ``lemoncrow``/``lc ... mcp`` process
    can be one of ours.
    """
    if not pid_is_running(pid):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return True  # not Linux: liveness is all we can check
    try:
        parts = [part for part in cmdline.read_bytes().split(b"\0") if part]
    except OSError:
        return False
    text = " ".join(part.decode("utf-8", errors="ignore") for part in parts)
    return ("lemoncrow" in text or "lc" in text) and "mcp" in text


def prune_stale_mcp_sessions(root: Path | None = None) -> int:
    """Delete registration files whose process is gone; return how many.

    A clean shutdown removes its own file, but a killed — or zombied — server
    never gets there, and the leftovers are what made ``lc mcp list`` show
    day-old "servers". Nobody can reap another process's zombie (only its
    parent can ``wait()``), so owning *our* registry is the part that is
    actually fixable: this reclaims it instead of filtering the corpses out of
    one view and leaving them on disk for every other reader.
    """
    sessions_dir = (Path(root) if root is not None else _lemoncrow_root()) / "mcp_sessions"
    if not sessions_dir.is_dir():
        return 0
    pruned = 0
    for entry in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = data.get("pid") if isinstance(data, dict) else None
        if isinstance(pid, int) and mcp_pid_is_live(pid):
            continue
        try:
            entry.unlink()
            pruned += 1
        except OSError:
            logger.debug("failed to prune stale mcp session file: %s", entry, exc_info=True)
    return pruned


def _mutate_mcp_managed_bash(*, record: dict[str, Any] | None = None, remove_id: str = "") -> None:
    """Atomically update live Bash ownership in this MCP registration."""
    path = _mcp_session_file()
    with _MCP_SESSION_FILE_LOCK:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            commands = [row for row in data.get("managed_bash", []) if isinstance(row, dict)]
            target_id = remove_id or str((record or {}).get("session_id") or "")
            if target_id:
                commands = [row for row in commands if str(row.get("session_id") or "") != target_id]
            if record is not None:
                commands.append(record)
            data["managed_bash"] = commands
            tmp = path.with_name(f".{path.name}.{_uuid_mod.uuid4().hex}.tmp")
            try:
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            logger.debug("MCP managed Bash registration update failed", exc_info=True)


def _record_mcp_managed_bash(started: dict[str, Any]) -> None:
    session_id = str(started.get("session_id") or "")
    pid = started.get("pid")
    if not session_id or not isinstance(pid, int):
        return
    record: dict[str, Any] = {
        "session_id": session_id,
        "pid": pid,
        "explicit_background": bool(started.get("explicit_background")),
        "started_at": time.time(),
    }
    for key in ("log_file", "log_file_stderr"):
        value = started.get(key)
        if isinstance(value, str) and value:
            record[key] = value
    _mutate_mcp_managed_bash(record=record)


def _forget_mcp_managed_bash(session_id: str) -> None:
    if session_id:
        _mutate_mcp_managed_bash(remove_id=session_id)


def live_managed_bash_ids() -> list[str]:
    """Session ids this MCP process currently owns, most recently started first.

    Reads the same registration file ``_record_mcp_managed_bash`` writes, so a
    poll against an id that does not exist can name the handles that would have
    worked instead of dead-ending on "unknown shell session".
    """
    try:
        data = json.loads(_mcp_session_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    rows = [row for row in data.get("managed_bash", []) if isinstance(row, dict)]
    rows.sort(key=lambda row: float(row.get("started_at") or 0.0), reverse=True)
    seen: list[str] = []
    for row in rows:
        session_id = str(row.get("session_id") or "")
        if session_id and session_id not in seen:
            seen.append(session_id)
    return seen
