"""RBAC and visibility helpers for the local workspace."""

from __future__ import annotations

from lemoncrow.core.foundation.memory_models import MemoryBlock

from .models import TeamMember
from .workspace import TeamPermissionError, TeamWorkspaceManager


def can_write_shared_memory(member: TeamMember) -> bool:
    return member.role in {"admin", "member"}


def ensure_shared_memory_write(member: TeamMember) -> None:
    if not can_write_shared_memory(member):
        raise TeamPermissionError("shared workspace memory requires admin or member role")


def memory_block_is_visible(
    block: MemoryBlock,
    *,
    viewer: TeamMember,
    workspace_id: str,
    shared_only: bool = False,
) -> bool:
    metadata = block.metadata or {}
    block_workspace_id = str(metadata.get("workspace_id") or workspace_id)
    if block_workspace_id != workspace_id:
        return False
    scope = str(metadata.get("visibility_scope") or metadata.get("scope") or "private")
    owner_user_id = str(metadata.get("owner_user_id") or "")
    if shared_only:
        return scope == "shared"
    if viewer.role == "admin":
        return True
    if scope == "shared":
        return True
    return owner_user_id == viewer.user_id


def visible_memory_blocks(
    blocks: list[MemoryBlock],
    *,
    manager: TeamWorkspaceManager,
    user_id: str | None = None,
    shared_only: bool = False,
) -> list[MemoryBlock]:
    workspace = manager.load_workspace()
    viewer = manager.require_member(user_id, workspace=workspace)
    return [
        block
        for block in blocks
        if memory_block_is_visible(block, viewer=viewer, workspace_id=workspace.id, shared_only=shared_only)
    ]
