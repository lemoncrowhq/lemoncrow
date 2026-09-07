"""Append-only audit log for cross-vendor memory facts.

See docs/specs/day30/08-memory-audit-viewer.md for the full spec.

Layout on disk:
  <root>/memory_audit.jsonl          — main event log (one JSON object per line)
  <root>/cross_vendor_memory.yaml    — user overrides / allow-list config
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from lemoncrow.pro.capabilities.cross_vendor_memory.models import AuditEvent

# ---------------------------------------------------------------------------
# Path helpers (used by sync_engine.py and serializer.py)
# ---------------------------------------------------------------------------


def audit_store_root(root: Path | str) -> Path:
    """Return the directory that contains audit JSONL files.

    Currently the same as ``root`` — audit files live at the top level of the
    LemonCrow store directory alongside other data files.
    """
    return Path(root).expanduser().resolve()


def audit_overrides_path(root: Path | str) -> Path:
    """Path to the cross-vendor memory user-override YAML file."""
    return audit_store_root(root) / "cross_vendor_memory.yaml"


def local_machine_id() -> str:
    """Return a stable, opaque, RANDOM-LOCAL identifier for this installation.

    Open-source runtime: used only to distinguish local installations within a
    local team workspace. It is a cryptographically-secure random id cached in
    the LemonCrow data directory (``machine_id``) — never derived from
    ``/etc/machine-id``, hostname, MAC address, or any other machine property.
    Reset it by deleting the file. See docs/maintenance-mode-transition.md.
    """
    from lemoncrow.core.foundation.paths import default_store_root

    path = default_store_root() / "machine_id"
    try:
        if path.exists():
            val = path.read_text(encoding="utf-8").strip()
            if val:
                return val
    except OSError:
        pass
    machine_id = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(machine_id, encoding="utf-8")
    except OSError:
        pass
    return machine_id


# ---------------------------------------------------------------------------
# MemoryAuditLog
# ---------------------------------------------------------------------------

_LOG_FILENAME = "memory_audit.jsonl"

# Retention pruning is housekeeping, not a per-append obligation: rewriting
# the whole log on every append() would make the audit trail O(events) per
# write. Once per interval per log path keeps the file bounded without extra
# I/O on the hot path (mirrors LocalTelemetryStore.write_event's prune gate).
_RETENTION_DAYS = 90
_PRUNE_INTERVAL_S = 6 * 60 * 60
_last_prune_ts: dict[str, float] = {}


class MemoryAuditLog:
    """Append-only log of ``AuditEvent`` records.

    Usage::

        log = MemoryAuditLog(root)
        log.append(AuditEvent(vendor="claude", event="added", ...))
        for event in log.read(since=yesterday):
            print(event.fact_id, event.content)
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._path = self._root / _LOG_FILENAME

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def append(self, event: AuditEvent) -> None:
        """Append *event* to the log (creates the file if absent)."""
        self._root.mkdir(parents=True, exist_ok=True)
        record = event.to_public_record()
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
        now = time.time()
        key = str(self._path)
        if now - _last_prune_ts.get(key, 0.0) >= _PRUNE_INTERVAL_S:
            _last_prune_ts[key] = now
            self._prune(now)

    def _prune(self, now: float) -> None:
        """Drop events older than the retention window (best-effort, fail-open)."""
        if not self._path.exists():
            return
        cutoff = datetime.fromtimestamp(now, tz=UTC) - timedelta(days=_RETENTION_DAYS)
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        kept: list[str] = []
        changed = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                event_at = AuditEvent.model_validate(json.loads(stripped)).at.astimezone(UTC)
            except (json.JSONDecodeError, ValueError):
                kept.append(stripped)  # unparsable line: keep rather than lose data
                continue
            if event_at < cutoff:
                changed = True
                continue
            kept.append(stripped)
        if not changed:
            return
        try:
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            tmp.replace(self._path)
        except OSError:
            pass  # fail-open

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    def read(self, *, since: datetime | None = None) -> list[AuditEvent]:
        """Return all events, optionally filtered to those at or after *since*."""
        if not self._path.exists():
            return []

        since_utc = since.astimezone(UTC) if since is not None else None
        results: list[AuditEvent] = []

        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            event = AuditEvent.model_validate(record)

            if since_utc is not None:
                event_time = event.at.astimezone(UTC)
                if event_time < since_utc:
                    continue

            results.append(event)

        return results

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.read())

    def clear(self) -> None:
        """Delete the log file (useful in tests)."""
        if self._path.exists():
            self._path.unlink()


__all__ = [
    "AuditEvent",
    "MemoryAuditLog",
    "audit_overrides_path",
    "audit_store_root",
    "local_machine_id",
]
