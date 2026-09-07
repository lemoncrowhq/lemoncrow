"""Workspace-aware routing for supported read-only code-intel operations."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities.code_context_contract import (
    UnsupportedWorkspaceOperationError as UnsupportedWorkspaceOperationError,
)
from lemoncrow.pro.capabilities.code_context.workspace_config import (
    WorkspaceConfig,
    load_workspace_config,
)

SUPPORTED_WORKSPACE_OPS = frozenset({"search", "symbol"})


class WorkspaceCodeRouter:
    """Fan out supported code-intel calls across configured workspace repos."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        engine_factory: Callable[[Path], Any],
        config: WorkspaceConfig | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.engine_factory = engine_factory
        self.config = config if config is not None else load_workspace_config(self.repo_root)

    @property
    def is_configured(self) -> bool:
        return self.config is not None

    def route(self, op: str, *, repo: str | None = None, **kwargs: Any) -> dict[str, Any]:
        if op not in SUPPORTED_WORKSPACE_OPS:
            raise UnsupportedWorkspaceOperationError(f"Unsupported workspace operation: {op}")
        targets = self._target_repo_roots(repo)
        if op == "search":
            return self._route_search(targets, **kwargs)
        return self._route_symbol(targets, **kwargs)

    def _target_repo_roots(self, repo: str | None) -> list[Path]:
        if self.config is None:
            if repo is not None:
                raise ValueError("Unknown workspace repo: workspace config is not available")
            return [self.repo_root]
        if repo is None:
            return [entry.repo_root for entry in self.config.repos]
        selected = self.config.repo_by_name(repo)
        if selected is None:
            raise ValueError(f"Unknown workspace repo: {repo}")
        return [selected.repo_root]

    def _route_search(self, targets: list[Path], **kwargs: Any) -> dict[str, Any]:
        merged_items: list[dict[str, Any]] = []
        total_tokens = 0
        tokens_saved = 0
        cache_hit = True
        provenance = "local"
        provenance_breakdown: dict[str, int] = {}
        mode: str | None = None
        for repo_root, payload in self._search_payloads(targets, **kwargs):
            repo_name = self._repo_name_for_root(repo_root)
            merged_items.extend(self._annotate_items(list(payload.get("items", [])), repo_name=repo_name))
            total_tokens += int(payload.get("total_tokens", 0))
            tokens_saved += int(payload.get("tokens_saved", 0))
            cache_hit = cache_hit and bool(payload.get("cache_hit", False))
            provenance = str(payload.get("provenance", provenance))
            mode = str(payload.get("mode", mode or "")) or mode
            payload_breakdown = payload.get("provenance_breakdown")
            if isinstance(payload_breakdown, dict):
                for key, value in payload_breakdown.items():
                    provenance_breakdown[str(key)] = provenance_breakdown.get(str(key), 0) + int(value)
            elif payload.get("items"):
                provenance_breakdown[provenance] = provenance_breakdown.get(provenance, 0) + len(
                    list(payload.get("items", []))
                )
        result = {
            "items": merged_items,
            "cache_hit": cache_hit,
            "provenance": provenance,
            "tokens_saved": tokens_saved,
            "total_tokens": total_tokens,
        }
        if mode is not None:
            result["mode"] = mode
        if provenance_breakdown:
            result["provenance_breakdown"] = provenance_breakdown
        return result

    def _search_payloads(self, targets: list[Path], **kwargs: Any) -> list[tuple[Path, dict[str, Any]]]:
        """Run ``tool_search`` per repo, concurrently for multi-repo workspaces.

        Results are returned in the original ``targets`` order so the merged
        union stays deterministic regardless of completion order.
        """
        if len(targets) <= 1:
            return [(repo_root, self.engine_factory(repo_root).tool_search(**kwargs)) for repo_root in targets]
        with ThreadPoolExecutor(max_workers=len(targets)) as executor:
            payloads = list(
                executor.map(lambda repo_root: self.engine_factory(repo_root).tool_search(**kwargs), targets)
            )
        return list(zip(targets, payloads, strict=True))

    def _route_symbol(self, targets: list[Path], **kwargs: Any) -> dict[str, Any]:
        isolate_failures = len(targets) > 1
        last_error: dict[str, Any] | None = None
        for repo_root in targets:
            try:
                payload = self.engine_factory(repo_root).tool_symbol(**kwargs)
            except LookupError:
                last_error = {"error": "symbol_not_found"}
                continue
            except Exception:
                # One repo's engine failing (bad index, IO error, ...) must not
                # abort the multi-repo lookup; a later repo may still resolve it.
                if not isolate_failures:
                    raise
                last_error = {"error": "symbol_not_found"}
                continue
            if "error" not in payload:
                return self._annotate_item(payload, repo_name=self._repo_name_for_root(repo_root))
            last_error = payload
        return last_error or {"error": "symbol_not_found"}

    def _repo_name_for_root(self, repo_root: Path) -> str | None:
        if self.config is None:
            return None
        for entry in self.config.repos:
            if entry.repo_root == repo_root:
                return entry.name
        return None

    def _annotate_items(self, items: list[dict[str, Any]], *, repo_name: str | None) -> list[dict[str, Any]]:
        return [self._annotate_item(item, repo_name=repo_name) for item in items]

    def _annotate_item(self, item: dict[str, Any], *, repo_name: str | None) -> dict[str, Any]:
        if repo_name is None:
            return dict(item)
        annotated = dict(item)
        annotated["repo_name"] = repo_name
        return annotated
