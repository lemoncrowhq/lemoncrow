"""Per-user usage rollups from persisted run ledgers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .workspace import TeamWorkspaceManager


def summarize_workspace_usage(
    root: Path | str,
    *,
    manager: TeamWorkspaceManager | None = None,
    since: datetime | None = None,
) -> dict[str, Any]:
    store_root = Path(root).expanduser().resolve()
    workspace_manager = manager or TeamWorkspaceManager(store_root)
    workspace = workspace_manager.load_workspace()
    totals: dict[str, dict[str, Any]] = {}
    sessions_dir = store_root / "sessions"
    session_count = 0
    total_cost_usd = 0.0
    if sessions_dir.exists():
        for path in sorted(sessions_dir.glob("**/run.json")):
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            updated_at_raw = snapshot.get("updated_at") or snapshot.get("created_at")
            updated_at = _parse_datetime(updated_at_raw) if updated_at_raw else None
            if since is not None and updated_at is not None and updated_at < since.astimezone(UTC):
                continue
            agent_settings = snapshot.get("agent_settings") or {}
            user_id = str(agent_settings.get("user_id") or agent_settings.get("user_email") or "unknown").lower()
            cost = float((snapshot.get("cost") or {}).get("total_cost_usd") or 0.0)
            entry = totals.setdefault(user_id, {"user_id": user_id, "session_count": 0, "total_cost_usd": 0.0})
            entry["session_count"] += 1
            entry["total_cost_usd"] = round(float(entry["total_cost_usd"]) + cost, 6)
            session_count += 1
            total_cost_usd = round(total_cost_usd + cost, 6)
    members = {member.user_id: member.role for member in workspace.members}
    users = []
    for user_id, payload in sorted(totals.items()):
        users.append({**payload, "role": members.get(user_id, "unknown")})
    return {
        "workspace_id": workspace.id,
        "account_id": workspace.account_id,
        "session_count": session_count,
        "total_cost_usd": total_cost_usd,
        "users": users,
    }


def _parse_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
