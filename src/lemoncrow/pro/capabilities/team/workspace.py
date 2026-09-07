"""Local team workspace management, invite flow, and audit trail."""

from __future__ import annotations

import contextlib
import json
import secrets
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities.team_contract import (
    TeamPermissionError as TeamPermissionError,
)
from lemoncrow.core.capabilities.team_contract import (
    TeamWorkspaceError as TeamWorkspaceError,
)
from lemoncrow.core.capabilities.team_contract import (
    TeamWorkspaceNotInitializedError as TeamWorkspaceNotInitializedError,
)
from lemoncrow.pro.capabilities.cross_vendor_memory.audit_log import local_machine_id

from .models import TeamAuditEvent, TeamInvite, TeamMember, TeamRole, TeamWorkspace


def workspace_path(root: Path) -> Path:
    return root / "team_workspace.json"


def audit_path(root: Path) -> Path:
    return root / "team_audit.jsonl"


def signing_key_path(root: Path) -> Path:
    return root / "team_signing_key.txt"


def _workspace_lock_path(root: Path) -> Path:
    path = workspace_path(root)
    return path.parent / (path.name + ".lock")


# Serialize workspace read-modify-write critical sections across worker threads
# sharing this process (the FastAPI sync-handler threadpool serves concurrent
# /v1/team/* requests here); a sidecar flock additionally guards sibling
# processes (e.g. simultaneous `lc team` invocations).
_WORKSPACE_MUTATION_LOCK = threading.Lock()


class TeamWorkspaceManager:
    """Persist and query local workspace state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()

    def exists(self) -> bool:
        return workspace_path(self.root).exists()

    def load_workspace(self) -> TeamWorkspace:
        path = workspace_path(self.root)
        if not path.exists():
            raise TeamWorkspaceNotInitializedError("team workspace not initialized")
        return TeamWorkspace.model_validate_json(path.read_text(encoding="utf-8"))

    def save_workspace(self, workspace: TeamWorkspace) -> TeamWorkspace:
        self.root.mkdir(parents=True, exist_ok=True)
        workspace_path(self.root).write_text(
            workspace.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return workspace

    @contextlib.contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        """Serialize a load_workspace->modify->save_workspace critical section so
        concurrent mutations cannot lose-update each other. Guards worker threads
        sharing this process (_WORKSPACE_MUTATION_LOCK) and, best-effort, sibling
        processes sharing the same team_workspace.json (POSIX flock)."""
        with _WORKSPACE_MUTATION_LOCK:
            handle = self._acquire_flock()
            try:
                yield
            finally:
                self._release_flock(handle)

    def _acquire_flock(self) -> Any:
        try:
            import fcntl
        except ImportError:
            return None
        try:
            lock_path = _workspace_lock_path(self.root)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "w", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return handle
        except OSError:
            return None

    def _release_flock(self, handle: Any) -> None:
        if handle is None:
            return
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        with contextlib.suppress(OSError):
            handle.close()

    def init_workspace(self, *, name: str, admin_email: str = "admin@local") -> TeamWorkspace:
        if self.exists():
            raise TeamWorkspaceError("team workspace already initialized")
        admin_user_id = admin_email.strip().lower()
        workspace = TeamWorkspace(
            name=name,
            current_user_id=admin_user_id,
            members=[
                TeamMember(
                    user_id=admin_user_id,
                    email=admin_email.strip().lower(),
                    role="admin",
                    auth_provider="google",
                )
            ],
            machine_bindings={local_machine_id(): admin_user_id},
        )
        self.save_workspace(workspace)
        self.get_signing_secret()
        self.append_audit_event(
            TeamAuditEvent(
                action="workspace.init",
                actor_user_id=admin_user_id,
                details={"workspace_id": workspace.id, "name": workspace.name},
            )
        )
        return workspace

    def get_member(self, user_id: str, *, workspace: TeamWorkspace | None = None) -> TeamMember | None:
        current = workspace or self.load_workspace()
        normalized = user_id.strip().lower()
        for member in current.members:
            if member.user_id == normalized:
                return member
        return None

    def require_member(self, user_id: str | None = None, *, workspace: TeamWorkspace | None = None) -> TeamMember:
        current = workspace or self.load_workspace()
        resolved_user = (user_id or current.current_user_id or "").strip().lower()
        member = self.get_member(resolved_user, workspace=current)
        if member is None:
            raise TeamPermissionError(f"user is not a workspace member: {resolved_user!r}")
        return member

    def require_admin(self, user_id: str | None = None, *, workspace: TeamWorkspace | None = None) -> TeamMember:
        member = self.require_member(user_id, workspace=workspace)
        if member.role != "admin":
            raise TeamPermissionError("admin role required")
        return member

    def invite_members(
        self,
        emails: list[str],
        *,
        role: TeamRole,
        actor_user_id: str | None = None,
        expires_in_days: int = 7,
    ) -> list[TeamInvite]:
        with self._mutation_lock():
            workspace = self.load_workspace()
            actor = self.require_admin(actor_user_id, workspace=workspace)
            invites: list[TeamInvite] = []
            for email in emails:
                normalized = email.strip().lower()
                invite = TeamInvite(
                    code=secrets.token_urlsafe(10),
                    email=normalized,
                    role=role,
                    expires_at=datetime.now(UTC) + timedelta(days=expires_in_days),
                )
                workspace.invites.append(invite)
                invites.append(invite)
            self.save_workspace(workspace)
        for invite in invites:
            self.append_audit_event(
                TeamAuditEvent(
                    action="workspace.invite",
                    actor_user_id=actor.user_id,
                    details={"email": invite.email, "role": invite.role, "invite_id": invite.id},
                )
            )
        return invites

    def join_workspace(
        self,
        invite_code: str,
        *,
        user_id: str | None = None,
        auth_provider: str = "google",
    ) -> TeamMember:
        workspace = self.load_workspace()
        normalized_code = invite_code.strip()
        invite = next((item for item in workspace.invites if item.code == normalized_code), None)
        if invite is None:
            raise TeamWorkspaceError("invite code not found")
        now = datetime.now(UTC)
        if invite.used_at is not None:
            raise TeamWorkspaceError("invite code has already been used")
        if invite.expires_at < now:
            raise TeamWorkspaceError("invite code has expired")
        # The member identity is bound to the invite's email; the optional
        # user_id may only re-state that binding, never select a different one.
        invite_user_id = invite.email.strip().lower()
        requested_user_id = (user_id or invite.email).strip().lower()
        if requested_user_id != invite_user_id:
            raise TeamWorkspaceError("user_id does not match the invited email")
        member_user_id = invite_user_id
        if self.get_member(member_user_id, workspace=workspace) is not None:
            # A join may only create the invited member, never overwrite an
            # existing one (which would let an invite hijack another account).
            raise TeamWorkspaceError("a member with this identity already exists")
        member = TeamMember(
            user_id=member_user_id,
            email=invite.email,
            role=invite.role,
            auth_provider=auth_provider,
        )
        workspace.members.append(member)
        updated_invite = invite.model_copy(update={"used_by_user_id": member.user_id, "used_at": now})
        workspace.invites = [updated_invite if item.id == invite.id else item for item in workspace.invites]
        workspace.current_user_id = member.user_id
        workspace.machine_bindings[local_machine_id()] = member.user_id
        self.save_workspace(workspace)
        self.append_audit_event(
            TeamAuditEvent(
                action="workspace.join",
                actor_user_id=member.user_id,
                details={"email": member.email, "role": member.role, "invite_id": updated_invite.id},
            )
        )
        return member

    def set_role(self, user_id: str, role: TeamRole, *, actor_user_id: str | None = None) -> TeamMember:
        workspace = self.load_workspace()
        actor = self.require_admin(actor_user_id, workspace=workspace)
        target = self.get_member(user_id, workspace=workspace)
        if target is None:
            raise TeamWorkspaceError(f"team member not found: {user_id}")
        if target.role == "admin" and role != "admin":
            admin_count = sum(1 for member in workspace.members if member.role == "admin")
            if admin_count <= 1:
                raise TeamWorkspaceError("cannot remove the final workspace admin")
        updated = target.model_copy(update={"role": role})
        workspace.members = [updated if item.user_id == target.user_id else item for item in workspace.members]
        self.save_workspace(workspace)
        self.append_audit_event(
            TeamAuditEvent(
                action="workspace.role",
                actor_user_id=actor.user_id,
                details={"user_id": updated.user_id, "role": updated.role},
            )
        )
        return updated

    def list_audit_events(self, *, since: datetime | None = None) -> list[TeamAuditEvent]:
        path = audit_path(self.root)
        if not path.exists():
            return []
        records: list[TeamAuditEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = TeamAuditEvent.model_validate(json.loads(line))
            if since is not None and event.at < since.astimezone(UTC):
                continue
            records.append(event)
        records.sort(key=lambda item: item.at)
        return records

    def append_audit_event(self, event: TeamAuditEvent) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with audit_path(self.root).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def get_signing_secret(self) -> str:
        path = signing_key_path(self.root)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        self.root.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_hex(32)
        path.write_text(secret, encoding="utf-8")
        return secret
