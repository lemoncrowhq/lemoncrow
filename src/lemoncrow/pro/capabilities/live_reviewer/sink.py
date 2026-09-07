"""Append-only review verdict log under ``reviews/<session_id>.jsonl``.

Each line is one verdict record. The Stop hook reads unconsumed records to
surface them once, then flips their ``consumed`` flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def reviews_path(root: str | Path, session_id: str) -> Path:
    return Path(root) / "reviews" / f"{session_id}.jsonl"


def append_verdict(root: str | Path, session_id: str, verdict: dict[str, Any]) -> None:
    path = reviews_path(root, session_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(verdict, ensure_ascii=False) + "\n")
    except OSError:
        pass  # fail-open


def read_verdicts(root: str | Path, session_id: str) -> list[dict[str, Any]]:
    path = reviews_path(root, session_id)
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text("utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def latest_verdict(root: str | Path, session_id: str) -> dict[str, Any] | None:
    rows = read_verdicts(root, session_id)
    return rows[-1] if rows else None


def latest_unconsumed(root: str | Path, session_id: str) -> list[dict[str, Any]]:
    return [row for row in read_verdicts(root, session_id) if not row.get("consumed")]


def mark_consumed(root: str | Path, session_id: str) -> None:
    path = reviews_path(root, session_id)
    rows = read_verdicts(root, session_id)
    if not rows:
        return
    changed = False
    for row in rows:
        if not row.get("consumed"):
            row["consumed"] = True
            changed = True
    if not changed:
        return
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        pass  # fail-open
