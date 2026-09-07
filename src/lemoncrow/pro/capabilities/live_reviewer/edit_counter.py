"""Pure helpers for counting ``file_edit`` events in a RunLedger run file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_events(run_file: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(Path(run_file).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = data.get("events") if isinstance(data, dict) else None
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def count_file_edits(run_file: str | Path) -> int:
    """Number of ``file_edit`` events recorded so far. Fail-open to 0."""
    return sum(1 for event in _load_events(run_file) if event.get("kind") == "file_edit")
