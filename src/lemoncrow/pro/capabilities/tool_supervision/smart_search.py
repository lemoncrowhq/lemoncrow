"""Smart search capability for the consolidated MCP surface."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from lemoncrow.core.capabilities.native_read_baseline import claude_read_baseline_text
from lemoncrow.core.foundation.paths import confine_to_root
from lemoncrow.infra.code_intel.zoekt.adapter import get_zoekt_supervisor
from lemoncrow.infra.embeddings.factory import get_embedder
from lemoncrow.infra.storage.vector import cosine_similarity
from lemoncrow.pro.capabilities.repo_map import build_repo_map
from lemoncrow.pro.capabilities.repo_map.graph import (
    build_reference_graph,
    should_skip_path,
)
from lemoncrow.pro.capabilities.repo_map.pagerank import personalized_pagerank
from lemoncrow.pro.capabilities.tool_supervision.search_read import search_read, search_read_to_dict

SearchMode = Literal["chunks", "full", "map"]
IndexedSearch = Callable[..., dict[str, Any]]

_CLAUDE_GREP_FILE_LIMIT = 100
_SHELL_METACHARS_RE = re.compile(r"[;&|`$<>()\n\r]")
_LEADING_DASH_RE = re.compile(r"^-")
_TEXT_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".sh",
    ".css",
    ".html",
}


def _assert_safe_query(query: str, path: str) -> None:
    if _SHELL_METACHARS_RE.search(query):
        raise ValueError("smart_search rejected: shell metacharacters not allowed in query")
    if _LEADING_DASH_RE.match(query):
        raise ValueError("smart_search rejected: query must not start with '-'")
    if _SHELL_METACHARS_RE.search(path):
        raise ValueError("smart_search rejected: shell metacharacters not allowed in path")


def _repo_root() -> Path:
    return Path(os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd())).resolve()


def _resolve_path(repo_root: Path, path: str) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else repo_root / raw
    # Confine the agent-controlled path to the workspace root so smart_search
    # cannot be coerced into reading files outside it. Raises ValueError on
    # escape (including via symlinks, which confine_to_root resolves).
    return confine_to_root(resolved, repo_root)


def _iter_text_files(root: Path, *, limit: int = 500) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in _TEXT_SUFFIXES else []
    if not root.exists():
        return []
    files: list[Path] = []
    for item in root.rglob("*"):
        if len(files) >= limit:
            break
        if not item.is_file():
            continue
        if should_skip_path(item, repo_root=root):
            continue
        if item.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            if item.stat().st_size > 512_000:
                continue
        except OSError:
            continue
        files.append(item)
    return sorted(files)


def _safe_fts_query(query: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(terms[:8])


def _fts_rank(repo_root: Path, search_path: Path, query: str, *, max_files: int) -> dict[str, float]:
    fts_query = _safe_fts_query(query)
    if not fts_query:
        return {}
    files = _iter_text_files(search_path)
    if not files:
        return {}
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(path UNINDEXED, content)")
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(file_path.relative_to(repo_root)) if file_path.is_relative_to(repo_root) else str(file_path)
            conn.execute("INSERT INTO docs(path, content) VALUES(?, ?)", (rel, content[:200_000]))
        rows = conn.execute(
            "SELECT path, bm25(docs) AS rank FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, max_files),
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        with contextlib.suppress(Exception):
            if conn is not None:
                conn.close()
    scores: dict[str, float] = {}
    for path, rank in rows:
        scores[str(path)] = 1.0 / (1.0 + abs(float(rank)))
    return scores


def _semantic_rank(repo_root: Path, paths: list[str], query: str) -> dict[str, float]:
    if not paths:
        return {}
    embedder = get_embedder()
    if embedder.dim <= 0:
        return {}

    # Embed query
    try:
        _qvecs = embedder.embed([query])
        if not _qvecs or not _qvecs[0]:
            return {}
        query_vector = _qvecs[0]
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}

    # Read all files in one pass, tracking which paths succeeded
    valid_paths: list[str] = []
    contents: list[str] = []
    for path in paths:
        try:
            content = (repo_root / path).read_text(encoding="utf-8", errors="replace")[:20_000]
            valid_paths.append(path)
            contents.append(content)
        except Exception:
            logging.exception("Recovered from broad exception handler")

    if not contents:
        return {}

    # Batch-embed all file contents in a single call (one round-trip for network backends)
    try:
        file_vecs = embedder.embed(contents)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}

    scores: dict[str, float] = {}
    for path, vec in zip(valid_paths, file_vecs, strict=False):
        if vec:
            scores[path] = max(0.0, cosine_similarity(query_vector, vec))
    return scores


def _graph_rank(repo_root: Path, seed_files: list[str]) -> dict[str, float]:
    try:
        graph, _tags = build_reference_graph(repo_root)
        return personalized_pagerank(graph, seed_files)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}


def _cache_key(payload: dict[str, Any], search_path: Path) -> str:
    stat_bits: list[str] = []
    with contextlib.suppress(Exception):
        if search_path.exists():
            stat = search_path.stat()
            stat_bits.append(f"{search_path}:{stat.st_size}:{stat.st_mtime_ns}")
    raw = json.dumps(payload, sort_keys=True) + "\n" + "\n".join(stat_bits)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def _state_path(repo_root: Path) -> Path:
    from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

    return resolve_workspace_store_dir(workspace_root=repo_root) / "smart_state.json"


def _load_cache(repo_root: Path) -> dict[str, Any]:
    path = _state_path(repo_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}
    cache = data.get("smart_search") if isinstance(data, dict) else None
    return cache if isinstance(cache, dict) else {}


def _save_cache(repo_root: Path, cache: dict[str, Any]) -> None:
    path = _state_path(repo_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        logging.exception("Recovered from broad exception handler")
        data = {}
    if not isinstance(data, dict):
        data = {}
    if len(cache) > 100:
        for key in list(cache.keys())[: len(cache) - 100]:
            cache.pop(key, None)
    data["smart_search"] = cache
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    highest = max(abs(value) for value in scores.values()) or 1.0
    return {key: value / highest for key, value in scores.items() if math.isfinite(value)}


def _search_with_backend(
    *,
    repo_root: Path,
    search_path: Path,
    query: str,
    max_files: int,
    max_chars_per_file: int,
    include_outline: bool,
) -> dict[str, Any] | None:
    supervisor = get_zoekt_supervisor(repo_root)
    if not supervisor.should_route(search_path):
        return None
    health = supervisor.health()
    if not health.ok:
        return None
    result = supervisor.search(
        query=query,
        search_path=search_path,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        include_outline=include_outline,
    )
    payload = search_read_to_dict(result, include_metadata=False)
    payload["backend"] = result.backend
    payload["index_age_seconds"] = result.index_age_seconds
    payload["total_tokens"] = result.total_tokens
    return payload


def _naive_bytes_for_matches(matches: list[dict[str, Any]], *, mode: SearchMode = "chunks") -> int:
    """Bytes in the closest Claude Code built-in baseline for these matches."""
    if mode != "full":
        paths = [str(match.get("path", "")) for match in matches[:_CLAUDE_GREP_FILE_LIMIT] if match.get("path")]
        return len("\n".join(paths))

    total = 0
    for match in matches[:_CLAUDE_GREP_FILE_LIMIT]:
        raw = str(match.get("path", ""))
        content = match.get("content")
        if isinstance(content, str) and content:
            total += len(claude_read_baseline_text(content))
            continue
        if not raw:
            continue
        try:
            source = Path(raw).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        total += len(claude_read_baseline_text(source))
    return total


def _rendered_bytes_for_matches(matches: list[dict[str, Any]]) -> int:
    """Approximate bytes returned to the agent: JSON-serialize the matches."""
    try:
        return len(json.dumps(matches, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


def _fallback_terms(query: str, *, limit: int = 6) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", query):
        if len(term) < 3 or term.lower() in {"and", "for", "from", "the", "this", "with"}:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _indexed_matches(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    max_files: int,
    max_chars_per_file: int,
) -> dict[str, Any] | None:
    matches_by_path: dict[str, dict[str, Any]] = {}
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        raw_path = str(item.get("file_path") or "")
        if not raw_path:
            continue
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = repo_root / resolved
        path = str(resolved.resolve())
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            try:
                snippet = resolved.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
            except OSError:
                snippet = ""
        entry = matches_by_path.setdefault(
            path,
            {
                "path": path,
                "lang": str(item.get("language") or ""),
                "snippets": [],
                "outline": None,
            },
        )
        snippets = entry["snippets"]
        if isinstance(snippets, list) and snippet:
            snippets.append(
                {
                    "line_start": int(item.get("start_line") or 1),
                    "line_end": int(item.get("end_line") or item.get("start_line") or 1),
                    "text": snippet[:max_chars_per_file],
                }
            )
        if len(matches_by_path) >= max_files:
            break
    if not matches_by_path:
        return None
    matches = list(matches_by_path.values())
    return {
        "matches": matches,
        "match_paths": list(matches_by_path),
        "backend": "code_index",
        "index_age_seconds": None,
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
        "fallback": {
            "reason": "empty_primary_result",
            "strategy": "indexed_hybrid",
        },
    }


def _search_index(
    indexed_search: IndexedSearch | None,
    *,
    query: str,
    path: str,
    max_files: int,
    max_chars_per_file: int,
    budget_tokens: int,
    repo_root: Path,
) -> dict[str, Any] | None:
    if indexed_search is None:
        return None
    try:
        payload = indexed_search(
            query=query,
            path=path,
            max_files=max_files,
            budget_tokens=budget_tokens,
        )
    except Exception:
        logging.exception("Indexed smart-search fallback failed")
        return None
    return _indexed_matches(
        payload,
        repo_root=repo_root,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
    )


def _relaxed_search(
    *,
    query: str,
    search_path: Path,
    max_files: int,
    max_chars_per_file: int,
    include_outline: bool,
) -> dict[str, Any] | None:
    matches_by_path: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    attempted_terms: list[str] = []
    for term in _fallback_terms(query):
        attempted_terms.append(term)
        result = search_read(
            query=re.escape(term),
            path=str(search_path),
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
            include_outline=include_outline,
        )
        total_tokens += result.total_tokens
        for match in search_read_to_dict(result, include_metadata=False).get("matches", []):
            if isinstance(match, dict) and match.get("path"):
                matches_by_path.setdefault(str(match["path"]), match)
        if len(matches_by_path) >= max_files:
            break
    if not matches_by_path:
        return None
    matches = list(matches_by_path.values())[:max_files]
    return {
        "matches": matches,
        "match_paths": [str(match["path"]) for match in matches],
        "backend": "ripgrep",
        "index_age_seconds": None,
        "total_tokens": total_tokens,
        "fallback": {
            "reason": "empty_primary_result",
            "strategy": "query_terms",
            "terms": attempted_terms,
        },
    }


def _file_preview(
    *,
    search_path: Path,
    max_chars_per_file: int,
    budget_tokens: int,
) -> dict[str, Any] | None:
    if not search_path.is_file() or search_path.suffix.lower() not in _TEXT_SUFFIXES:
        return None
    try:
        content = search_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    content = content[: min(max_chars_per_file, max(1, budget_tokens) * 4)]
    return {
        "matches": [{"path": str(search_path), "content": content, "snippets": []}],
        "match_paths": [str(search_path)],
        "backend": "file_preview",
        "index_age_seconds": None,
        "total_tokens": max(1, len(content) // 4),
        "fallback": {"reason": "empty_relaxed_result", "strategy": "scoped_file_preview"},
    }


def _cap_fields(returned: int, max_files: int) -> dict[str, Any]:
    """Truncation signal: 'capped' is True when the result hit the max_files
    limit, so the caller (LLM) knows it may be partial and can raise max_files
    to page instead of assuming these are all the matches.
    """
    return {
        "returned": returned,
        "capped": returned >= max_files,
        "cap_param": "max_files",
        "cap_limit": max_files,
    }


def smart_search(
    *,
    query: str,
    path: str = ".",
    mode: SearchMode = "chunks",
    max_files: int = 10,
    max_chars_per_file: int = 2000,
    include_outline: bool = True,
    seed_files: list[str] | None = None,
    budget_tokens: int = 2000,
    indexed_search: IndexedSearch | None = None,
) -> dict[str, Any]:
    """Search with lexical, semantic, and graph ranking signals."""
    _assert_safe_query(query, path)
    repo_root = _repo_root()
    search_path = _resolve_path(repo_root, path)
    seeds = seed_files or []

    if mode == "map":
        result = build_repo_map(repo_root, seed_files=seeds, budget_tokens=budget_tokens)
        map_result = result.model_dump(mode="json")
        map_result["mode"] = "map"
        return map_result

    cache_payload = {
        "query": query,
        "path": str(search_path),
        "mode": mode,
        "max_files": max_files,
        "max_chars_per_file": max_chars_per_file,
        "include_outline": include_outline,
        "seed_files": seeds,
        "budget_tokens": budget_tokens,
        "response_schema": 2,
    }
    cache_key = _cache_key(cache_payload, search_path)
    if os.environ.get("LEMONCROW_CACHE_DISABLED") != "1":
        cache = _load_cache(repo_root)
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            cached_matches = [match for match in cached.get("matches", []) if isinstance(match, dict)]
            cached_sliced = cached_matches[:max_files]
            return {
                "matches": cached_sliced,
                "match_paths": [path for path in cached.get("match_paths", []) if isinstance(path, str)][:max_files],
                "mode": mode,
                "backend": str(cached.get("backend") or "ripgrep"),
                "index_age_seconds": cached.get("index_age_seconds"),
                "cache_hit": True,
                "total_tokens": int(cached.get("total_tokens", 0) or 0),
                "tokens_saved": int(cached.get("tokens_saved", 0) or 0),
                **_cap_fields(len(cached_sliced), max_files),
                **({"fallback": cached["fallback"]} if isinstance(cached.get("fallback"), dict) else {}),
            }
    else:
        cache = {}

    payload: dict[str, Any] | None = _search_with_backend(
        repo_root=repo_root,
        search_path=search_path,
        query=query,
        max_files=max_files,
        max_chars_per_file=max_chars_per_file,
        include_outline=include_outline,
    )
    if payload is None:
        chunk_result = search_read(
            query=query,
            path=str(search_path),
            max_files=max_files,
            max_chars_per_file=max_chars_per_file,
            include_outline=include_outline,
        )
        payload = search_read_to_dict(chunk_result, include_metadata=False)
        payload["match_paths"] = [match.path for match in chunk_result.matches]
        payload["total_tokens"] = chunk_result.total_tokens
    if not payload.get("matches"):
        payload = (
            _search_index(
                indexed_search,
                query=query,
                path=path,
                max_files=max_files,
                max_chars_per_file=max_chars_per_file,
                budget_tokens=budget_tokens,
                repo_root=repo_root,
            )
            or _relaxed_search(
                query=query,
                search_path=search_path,
                max_files=max_files,
                max_chars_per_file=max_chars_per_file,
                include_outline=include_outline,
            )
            or _file_preview(
                search_path=search_path,
                max_chars_per_file=max_chars_per_file,
                budget_tokens=budget_tokens,
            )
            or payload
        )
    backend = str(payload.get("backend") or "ripgrep")
    matches = [match for match in payload.get("matches", []) if isinstance(match, dict)]
    if backend == "zoekt":
        if mode == "full":
            full_matches: list[dict[str, Any]] = []
            for match in matches[:max_files]:
                raw_path = str(match.get("path", ""))
                try:
                    content = Path(raw_path).read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
                except OSError:
                    content = ""
                full_matches.append({**match, "content": content, "snippets": []})
            matches = full_matches
        zoekt_matches = matches[:max_files]
        zoekt_naive = _naive_bytes_for_matches(zoekt_matches, mode=mode)
        zoekt_rendered = _rendered_bytes_for_matches(zoekt_matches)
        response = {
            "matches": zoekt_matches,
            "match_paths": [str(match.get("path", "")) for match in zoekt_matches if match.get("path")],
            "mode": mode,
            "backend": backend,
            "index_age_seconds": payload.get("index_age_seconds"),
            "total_tokens": payload.get("total_tokens", 0),
            "cache_hit": False,
            "tokens_saved": max(0, (zoekt_naive - zoekt_rendered) // 4),
            **_cap_fields(len(zoekt_matches), max_files),
            **({"fallback": payload["fallback"]} if isinstance(payload.get("fallback"), dict) else {}),
        }
        if os.environ.get("LEMONCROW_CACHE_DISABLED") != "1":
            cache[cache_key] = response
            _save_cache(repo_root, cache)
        return response
    paths = [str(match.get("path", "")) for match in payload.get("matches", []) if isinstance(match, dict)]
    rel_paths = [
        str(Path(item).resolve().relative_to(repo_root)) if Path(item).resolve().is_relative_to(repo_root) else item
        for item in paths
    ]
    fts_scores = _normalize_scores(_fts_rank(repo_root, search_path, query, max_files=max_files * 2))
    semantic_scores = _normalize_scores(_semantic_rank(repo_root, rel_paths, query))
    graph_scores = _normalize_scores(_graph_rank(repo_root, seeds or rel_paths[:1]))

    def score(match: dict[str, Any]) -> float:
        raw_path = str(match.get("path", ""))
        try:
            rel = str(Path(raw_path).resolve().relative_to(repo_root))
        except ValueError:
            rel = raw_path
        snippet_score = 0.0
        snippets = match.get("snippets")
        if isinstance(snippets, list) and snippets:
            snippet_score = max(float(item.get("score", 0.0)) for item in snippets if isinstance(item, dict))
        return (
            snippet_score
            + 0.35 * fts_scores.get(rel, fts_scores.get(raw_path, 0.0))
            + 0.25 * semantic_scores.get(rel, 0.0)
            + 0.40 * graph_scores.get(rel, 0.0)
        )

    matches = [match for match in payload.get("matches", []) if isinstance(match, dict)]
    matches.sort(key=lambda item: (-score(item), str(item.get("path", ""))))
    if mode == "full":
        fm: list[dict[str, Any]] = []
        for match in matches[:max_files]:
            raw_path = str(match.get("path", ""))
            try:
                content = Path(raw_path).read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
            except OSError:
                content = ""
            fm.append({**match, "content": content, "snippets": []})
        matches = fm
    final_matches = matches[:max_files]
    final_naive = _naive_bytes_for_matches(final_matches, mode=mode)
    final_rendered = _rendered_bytes_for_matches(final_matches)
    response = {
        "matches": final_matches,
        "match_paths": [str(match.get("path", "")) for match in final_matches if match.get("path")],
        "mode": mode,
        "backend": backend,
        "index_age_seconds": payload.get("index_age_seconds"),
        "total_tokens": int(payload.get("total_tokens", 0) or 0),
        "cache_hit": False,
        "tokens_saved": max(0, (final_naive - final_rendered) // 4),
        **_cap_fields(len(final_matches), max_files),
        **({"fallback": payload["fallback"]} if isinstance(payload.get("fallback"), dict) else {}),
    }

    if os.environ.get("LEMONCROW_CACHE_DISABLED") != "1":
        cache[cache_key] = response
        _save_cache(repo_root, cache)
    return response


__all__ = ["IndexedSearch", "SearchMode", "smart_search"]
