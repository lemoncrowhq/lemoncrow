from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.paths import default_store_root


@dataclass
class Checkpoint:
    id: str
    session_id: str
    label: str
    created_at: str
    message_count: int
    snapshot_path: str


def save_checkpoint(
    session_id: str,
    messages: list[dict[str, Any]],
    label: str = "",
    root: Path | None = None,
) -> Checkpoint:
    store = root or default_store_root()
    cp_dir = store / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)

    cp_id = f"cp-{uuid.uuid4().hex[:8]}"
    cp_path = cp_dir / f"{session_id}_{cp_id}.json"
    created_at = datetime.utcnow().isoformat()
    resolved_label = label or f"checkpoint-{len(messages)}-messages"
    cp_path.write_text(
        json.dumps(
            {
                "id": cp_id,
                "session_id": session_id,
                "label": resolved_label,
                "created_at": created_at,
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )

    return Checkpoint(
        id=cp_id,
        session_id=session_id,
        label=resolved_label,
        created_at=created_at,
        message_count=len(messages),
        snapshot_path=str(cp_path),
    )


def load_checkpoint(cp_id: str, session_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    store = root or default_store_root()
    cp_dir = store / "checkpoints"
    path = cp_dir / f"{session_id}_{cp_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {cp_id} not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["messages"])


def list_checkpoints(session_id: str, root: Path | None = None) -> list[Checkpoint]:
    store = root or default_store_root()
    cp_dir = store / "checkpoints"
    if not cp_dir.exists():
        return []
    results = []
    for f in sorted(
        cp_dir.glob(f"{session_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append(
                Checkpoint(
                    id=data["id"],
                    session_id=data["session_id"],
                    label=data["label"],
                    created_at=data["created_at"],
                    message_count=len(data["messages"]),
                    snapshot_path=str(f),
                )
            )
        except Exception:  # noqa: BLE001 - skip corrupt checkpoint files
            pass
    return results


__all__ = ["Checkpoint", "list_checkpoints", "load_checkpoint", "save_checkpoint"]
