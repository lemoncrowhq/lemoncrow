"""Team workspace capability."""

from .attribution import summarize_workspace_usage
from .models import TeamAuditEvent, TeamInvite, TeamMember, TeamRole, TeamWorkspace
from .rbac import can_write_shared_memory, ensure_shared_memory_write, visible_memory_blocks
from .sso import begin_google_oidc, finish_google_oidc
from .workspace import (
    TeamPermissionError,
    TeamWorkspaceError,
    TeamWorkspaceManager,
    TeamWorkspaceNotInitializedError,
    audit_path,
    signing_key_path,
    workspace_path,
)

__all__ = [
    "TeamAuditEvent",
    "TeamInvite",
    "TeamMember",
    "TeamPermissionError",
    "TeamRole",
    "TeamWorkspace",
    "TeamWorkspaceError",
    "TeamWorkspaceManager",
    "TeamWorkspaceNotInitializedError",
    "audit_path",
    "begin_google_oidc",
    "can_write_shared_memory",
    "ensure_shared_memory_write",
    "finish_google_oidc",
    "signing_key_path",
    "summarize_workspace_usage",
    "visible_memory_blocks",
    "workspace_path",
]
