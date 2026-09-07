"""Workspace config parsing for multi-repo code-intel routing."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceRepoConfig:
    name: str
    path: str
    repo_root: Path


@dataclass(frozen=True)
class WorkspaceConfig:
    workspace_id: str
    workspace_root: Path
    repos: tuple[WorkspaceRepoConfig, ...]

    def repo_by_name(self, repo_name: str) -> WorkspaceRepoConfig | None:
        for repo in self.repos:
            if repo.name == repo_name:
                return repo
        return None


def load_workspace_config(repo_root: str | Path) -> WorkspaceConfig | None:
    workspace_root = Path(repo_root).resolve()
    config_path = workspace_root / ".lemoncrow" / "workspace.toml"
    if not config_path.exists():
        return None

    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    workspace_payload = payload.get("workspace")
    if not isinstance(workspace_payload, dict):
        raise ValueError("workspace.toml must define a [workspace] table")
    workspace_id = workspace_payload.get("id")
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        raise ValueError("workspace.toml must define workspace.id")

    repos_payload = workspace_payload.get("repos")
    if not isinstance(repos_payload, list) or not repos_payload:
        raise ValueError("workspace.toml must define at least one [[workspace.repos]] entry")

    allowed_root = workspace_root.parent.resolve()
    repos: list[WorkspaceRepoConfig] = []
    seen_names: set[str] = set()
    for entry in repos_payload:
        if not isinstance(entry, dict):
            raise ValueError("workspace repo entries must be tables")
        name = entry.get("name")
        rel_path = entry.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("workspace repo entries require a non-empty name")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError(f"workspace repo '{name}' requires a non-empty path")
        if Path(rel_path).is_absolute():
            raise ValueError(f"workspace repo '{name}' path must be relative")
        if name in seen_names:
            raise ValueError(f"workspace repo '{name}' is duplicated")
        resolved = (workspace_root / rel_path).resolve()
        if resolved != allowed_root and allowed_root not in resolved.parents:
            raise ValueError(f"workspace repo '{name}' resolves outside the allowed workspace scope")
        repos.append(WorkspaceRepoConfig(name=name, path=rel_path, repo_root=resolved))
        seen_names.add(name)

    return WorkspaceConfig(
        workspace_id=workspace_id.strip(),
        workspace_root=workspace_root,
        repos=tuple(repos),
    )
