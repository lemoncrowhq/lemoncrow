from __future__ import annotations

from typing import Any

from lemoncrow.pro.capabilities.tool_supervision.smart_search import IndexedSearch, smart_search


def _match_paths(matches: list[dict[str, Any]]) -> list[str]:
    return [str(match.get("path")) for match in matches if isinstance(match.get("path"), str) and match.get("path")]


def _discovery_calls_saved(matches: list[dict[str, Any]]) -> int:
    return max(0, len(_match_paths(matches)) - 1)


def _context_follow_up(*, task: str, files: list[str], mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool": "context", "mode": mode, "task": task, "files": files}
    if mode == "procedures":
        payload["recall"] = True
    return payload


def search_first(
    *,
    query: str,
    task: str,
    path: str = ".",
    max_files: int = 8,
    max_chars_per_file: int = 1600,
    include_outline: bool = True,
    budget_tokens: int = 2000,
    indexed_search: IndexedSearch | None = None,
) -> dict[str, Any]:
    payload = smart_search(
        query=query,
        path=path,
        mode="chunks",
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        include_outline=include_outline,
        budget_tokens=budget_tokens,
        indexed_search=indexed_search,
    )
    matches = [match for match in payload.get("matches", []) if isinstance(match, dict)]
    match_paths = [path for path in payload.get("match_paths", []) if isinstance(path, str) and path]
    if not match_paths:
        match_paths = _match_paths(matches)

    enriched_matches: list[dict[str, Any]] = []
    for match in matches:
        match_path = str(match.get("path") or "")
        enriched_matches.append(
            {
                **match,
                "follow_up": {
                    "read": {"tool": "read", "path": match_path},
                    "context": _context_follow_up(task=task, files=[match_path], mode="symbols"),
                },
            }
        )

    return {
        "query": query,
        "task": task,
        "path": path,
        "mode": "chunks",
        "discovery": {"tool": "search", "mode": "chunks"},
        "matches": enriched_matches,
        "match_paths": match_paths,
        "calls_saved": _discovery_calls_saved(enriched_matches),
        "handoff": {
            "read": {"tool": "read"},
            "context": _context_follow_up(task=task, files=match_paths, mode="symbols"),
            "memory": _context_follow_up(task=task, files=match_paths, mode="procedures"),
            "relations": {"tool": "grep", "relation": "usages", "symbol": query},
        },
        "backend": str(payload.get("backend") or "ripgrep"),
        "index_age_seconds": payload.get("index_age_seconds"),
        "cache_hit": bool(payload.get("cache_hit", False)),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
        "tokens_saved": int(payload.get("tokens_saved", 0) or 0),
        **({"fallback": payload["fallback"]} if isinstance(payload.get("fallback"), dict) else {}),
    }


__all__ = ["_discovery_calls_saved", "search_first"]
