"""LemonCrow production service API.

Creates a FastAPI application exposing the LemonCrow reasoning runtime
over HTTP with optional Bearer auth.

Usage (import-safe — no server starts on import)::

    from lemoncrow.core.service.api import create_app
    app = create_app(store_root="/path/to/data")

Run via CLI::

    lc runtime start

"""

from __future__ import annotations

import contextlib
import json
import logging
import mimetypes
import os
import re
import shlex
import sqlite3
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from lemoncrow.core.capabilities.host_router_bridge import evaluate_host_router_request
from lemoncrow.core.capabilities.pricing import usage_cost_usd
from lemoncrow.core.capabilities.workflow_runtime_state import (
    pause_workflow_runtime,
    stop_workflow_runtime,
    workflow_runtime_detail,
)
from lemoncrow.core.foundation.models import Trace, coerce_trace_json, to_jsonable
from lemoncrow.core.foundation.paths import (
    confine_to_root,
    flat_session_dir,
    resolve_session_state_path,
    resolve_workspace_root,
    safe_segment,
)
from lemoncrow.core.service.auth import verify_api_key
from lemoncrow.core.service.config import cfg
from lemoncrow.core.service.schemas import (
    ContextRequest,
    ContextResponse,
)
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.pro.capabilities.swarm import (
    build_swarm_apply_payload,
    build_swarm_export_payload,
    build_swarm_spec_payload,
    discover_repo_root,
    initialize_swarm_run,
    list_swarm_runner_profiles,
    list_swarm_runs,
    load_swarm_state,
    read_swarm_log,
    resolve_state_path,
    resolve_swarm_child_command,
    resolve_swarm_provider_command,
    resolve_swarm_runner_metadata,
    resolve_swarm_spec_path,
    save_swarm_state,
    spawn_swarm_coordinator,
    stop_swarm_run,
)

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.optimization.optimizer import Candidate, OptimizationResult

logger = logging.getLogger(__name__)


def _host_family(host: str | None) -> str:
    host_name = str(host or "").strip() or "unknown"
    if host_name == "cursor-agent":
        return "cursor"
    return host_name


_HOST_LABEL_OVERRIDES: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "opencode": "opencode",
    "lemoncode": "LemonCode",
    "copilot": "Copilot / VS Code",
    "antigravity": "Antigravity",
    "cursor": "Cursor IDE",
    "hermes": "Hermes Agent",
}

_HOST_DESCRIPTION_OVERRIDES: dict[str, str] = {
    "claude": "Generated AGENTS surface, MCP wrapper, and Claude plugin hooks.",
    "codex": "Codex MCP registration with generated instructions and shared telemetry.",
    "opencode": "OpenCode MCP config with imported session support and local agents.",
    "lemoncode": "LemonCrow's controlled OpenCode fork: managed host binary, local agents, and session import.",
    "copilot": "VS Code / Copilot MCP config with custom instructions and shared telemetry.",
    "antigravity": "Antigravity MCP config plus generated AGENTS guidance and agy companion flow.",
    "cursor": "Cursor MCP config with project rules and MCP-first guidance.",
    "hermes": "Global-only Hermes MCP registration through ~/.hermes/config.yaml.",
}

_HOST_ORDER: tuple[str, ...] = (
    "lemoncode",
    "claude",
    "codex",
    "opencode",
    "copilot",
    "antigravity",
    "cursor",
    "hermes",
)


def _host_import_stats(store: StoreBundle) -> dict[str, dict[str, Any]]:
    """Per-host last-import time and imported-session count for /hosts.

    ``Trace.created_at`` (the ``traces.created_at`` column) is seeded from
    the *session's own* first-message timestamp at import time (see e.g.
    session_parsers/claude.py's ``created_at`` walk over the transcript, or
    ``_common.py``'s ``source_mtime`` for the other host importers) -- i.e.
    it reflects when the session *started*, not when LemonCrow imported it.
    ``RawArtifact.created_at`` is the only field stamped with wall-clock
    ``utcnow()`` at the moment an importer actually processed the file (see
    ``session_parsers/claude.py`` and the shared ``record_normalized_session``
    helper), so it is the most honest signal available for "when did we last
    import from this host" without adding a new store.py column.

    Two plain aggregate queries -- not ContextStore.list_traces()'s per-row
    pydantic parse, and not RawArtifact.model_validate per artifact -- so
    this stays cheap regardless of history size.
    """
    with sqlite3.connect(store.history.db_path) as conn:
        session_counts = dict(
            conn.execute(
                "SELECT host, COUNT(*) FROM traces "
                "WHERE task != 'session-auto-record' AND host IS NOT NULL "
                "GROUP BY host"
            ).fetchall()
        )
        last_imports = dict(
            conn.execute("SELECT source, MAX(created_at) FROM raw_artifacts GROUP BY source").fetchall()
        )
    hosts = set(session_counts) | set(last_imports)
    return {
        host: {
            "imported_session_count": int(session_counts.get(host, 0)),
            "last_import_at": last_imports.get(host),
        }
        for host in hosts
    }


def _build_trace_fts_query(query: str) -> str:
    """Build an FTS5 MATCH expression from free-text search input.

    Mirrors ContextStore._build_trace_search_query() -- small, self-contained
    regex logic duplicated here rather than reaching into a "private" store
    method, since store.py is off-limits for this change.
    """
    clauses: list[str] = []
    for phrase, token in re.findall(r'"([^"]+)"|(\S+)', query):
        term = (phrase or token).strip().lower()
        if not term:
            continue
        if phrase:
            escaped_term = term.replace('"', '""')
            clauses.append(f'"{escaped_term}"')
            continue
        pieces = [piece for piece in re.split(r"[^0-9a-z_]+", term) if piece]
        clauses.extend(f"{piece}*" for piece in pieces)
    if clauses:
        return " AND ".join(clauses)
    escaped = query.strip().replace('"', '""')
    return f'"{escaped}"'


def _list_traces_filtered(
    store: StoreBundle,
    *,
    domain: str | None,
    status: str | None,
    agent: str | None,
    host: str | None,
    workspace: str | None,
    query: str | None,
    since: datetime | None,
    limit: int,
    offset: int,
) -> list[Trace]:
    """Like ContextStore.list_traces(), plus a workspace_path filter.

    ContextStore.list_traces() has no workspace parameter and store.py can't
    be touched here, so when a workspace filter is requested this selects
    matching ids directly (mirroring list_traces()'s own SQL, plus the extra
    predicate) and hydrates each through the public get_trace() -- reusing
    the store's own Trace-parsing/fallback logic instead of duplicating it.
    """
    if not workspace:
        return store.history.list_traces(
            domain=domain,
            status=status,
            agent=agent,
            host=host,
            query=query,
            since=since,
            limit=limit,
            offset=offset,
        )

    sql = "SELECT id FROM traces WHERE task != 'session-auto-record' AND workspace_path = ?"
    params: list[Any] = [workspace]
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    if status:
        sql += " AND status = ?"
        params.append(status)
    if agent:
        sql += " AND agent = ?"
        params.append(agent)
    if host:
        sql += " AND host = ?"
        params.append(host)
    if since:
        sql += " AND created_at >= ?"
        params.append(since.isoformat())
    if query and query.strip():
        sql += " AND id IN (SELECT id FROM traces_fts WHERE traces_fts MATCH ?)"
        params.append(_build_trace_fts_query(query))
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.append(limit)
    params.append(offset)

    with sqlite3.connect(store.history.db_path) as conn:
        ids = [row[0] for row in conn.execute(sql, params).fetchall()]
    traces = [store.history.get_trace(trace_id) for trace_id in ids]
    return [trace for trace in traces if trace is not None]


def _distinct_workspaces(
    store: StoreBundle,
    *,
    domain: str | None,
    agent: str | None,
    host: str | None,
    since: datetime | None,
) -> list[str]:
    """Distinct workspace_path values for the /traces facets.

    Full history, not just the loaded page -- mirrors get_traces_metrics()'s
    existing "hosts" facet.
    """
    sql = (
        "SELECT DISTINCT workspace_path FROM traces "
        "WHERE task != 'session-auto-record' AND workspace_path IS NOT NULL AND workspace_path != ''"
    )
    params: list[Any] = []
    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    if agent:
        sql += " AND agent = ?"
        params.append(agent)
    if host:
        sql += " AND host = ?"
        params.append(host)
    if since:
        sql += " AND created_at >= ?"
        params.append(since.isoformat())
    with sqlite3.connect(store.history.db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return sorted({row[0] for row in rows if row[0]})


def _bulk_raw_artifact_fingerprints(store: StoreBundle, artifact_ids: set[str]) -> dict[str, tuple[str, int]]:
    """(source_file_mtime_iso, byte_count_original) for many artifacts in one query.

    ContextStore.get_raw_artifact() opens a fresh sqlite3 connection per call
    outside of batch_mode(), so calling it once per raw-artifact id (as the
    imported-session cache fingerprint used to) means N_sessions x
    N_artifacts fresh connections on every /v1/sessions poll. One bulk
    lookup replaces that with a single query regardless of history size.
    """
    if not artifact_ids:
        return {}
    # Chunk into batches well under SQLITE_MAX_VARIABLE_NUMBER (999 on older
    # SQLite): a single ``IN (?,?,...)`` over up to limit*5 (~5000) ids raises
    # OperationalError, 500-ing the whole /v1/sessions dashboard poll.
    result: dict[str, tuple[str, int]] = {}
    ids = list(artifact_ids)
    with sqlite3.connect(store.history.db_path) as conn:
        for start in range(0, len(ids), 900):
            batch = ids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT id, source_file_mtime, byte_count_original FROM raw_artifacts WHERE id IN ({placeholders})",
                batch,
            ).fetchall()
            # ``row[2] is not None`` (not ``or -1``) so a genuine 0-byte original
            # maps to 0, not the -1 sentinel — else its import cache never hits.
            result.update({row[0]: (row[1] or "", int(row[2]) if row[2] is not None else -1) for row in rows})
    return result


_SWARM_FILE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}


class SwarmLaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_root: str
    spec_path: str | None = None
    spec_mode: Literal["existing", "inline"] = "existing"
    spec_content: str | None = None
    provider: Literal["cli", "openai", "litellm"] = "cli"
    runner: str | None = "claude"
    runner_model: str | None = None
    model: str | None = None
    runner_options: str = ""
    runs: int = Field(default=3, ge=1)
    continuous: bool = True
    evaluator_backend: Literal["auto", "disabled", "ollama", "openai", "litellm"] = "auto"
    evaluator_model: str | None = None
    max_waves: int = Field(default=5, ge=0)
    max_evaluator_failures: int = Field(default=3, ge=1)
    keep_worktrees: bool = True
    effort: str = "high"
    provider_api_key: str | None = None
    provider_base_url: str | None = None
    provider_env: dict[str, str] = Field(default_factory=dict)


class WorkflowSnapshotActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str | None = None


class HostRouterEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "/router-preset/claudecode/auto"
    model: str = ""
    system: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    session_state: dict[str, Any] = Field(default_factory=dict)
    mode: Literal["disabled", "shadow", "enforced"] | None = None


_SWARM_PROVIDER_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SWARM_DEFAULT_SPEC_NAMES = ("PROGRAM.md", "program.md")
_SWARM_PROVIDER_ENV_BLOCKLIST = {
    "LEMONCROW_ROOT",
    "LEMONCROW_WORKSPACE_ROOT",
    "CLAUDE_WORKSPACE_ROOT",
    "PATH",
    "PYTHONPATH",
}


def _sanitize_swarm_provider_env(raw_env: dict[str, str] | None) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in (raw_env or {}).items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "")
        if not key:
            continue
        if not _SWARM_PROVIDER_ENV_KEY.fullmatch(key):
            raise ValueError(f"invalid provider env key: {key!r}")
        if key in _SWARM_PROVIDER_ENV_BLOCKLIST or key.startswith("LEMONCROW_SWARM_"):
            raise ValueError(f"provider env key is reserved: {key}")
        if "\x00" in value:
            raise ValueError(f"provider env value contains NUL byte: {key}")
        sanitized[key] = value
    return sanitized


def _build_swarm_provider_env(payload: SwarmLaunchRequest) -> dict[str, str]:
    env = _sanitize_swarm_provider_env(payload.provider_env)
    if payload.provider == "openai":
        if payload.provider_api_key:
            env["LEMONCROW_OPENAI_API_KEY"] = payload.provider_api_key
        if payload.provider_base_url:
            env["LEMONCROW_OPENAI_BASE_URL"] = payload.provider_base_url
    return env


def _workflow_session_state_path() -> Path:
    return resolve_session_state_path(resolve_workspace_root())


def _read_workflow_session_state() -> dict[str, Any]:
    path = _workflow_session_state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_workflow_session_state(state: dict[str, Any]) -> None:
    path = _workflow_session_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _default_swarm_spec_name(project_root: Path) -> str:
    for name in _SWARM_DEFAULT_SPEC_NAMES:
        if (project_root / name).is_file():
            return name
    return _SWARM_DEFAULT_SPEC_NAMES[0]


def _swarm_candidate_project_roots(root: Path) -> list[Path]:
    seen: set[Path] = set()
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            return
        seen.add(resolved)
        candidates.append(resolved)

    with contextlib.suppress(RuntimeError):
        add_candidate(discover_repo_root(Path.cwd()))
    for state in list_swarm_runs(root):
        with contextlib.suppress(OSError, RuntimeError):
            add_candidate(Path(state.repo_root))
    return candidates


def _coerce_project_root(root: Path, project_root: str | None) -> Path:
    candidates = _swarm_candidate_project_roots(root)
    if project_root:
        requested = Path(project_root).expanduser().resolve()
        if requested in candidates:
            return requested
        try:
            discovered = discover_repo_root(requested)
        except RuntimeError as exc:
            raise ValueError(f"unknown swarm project root: {project_root}") from exc
        resolved = discovered.resolve()
        if resolved != requested:
            raise ValueError(f"project root must point at a repository root: {project_root}")
        return resolved
    if candidates:
        return candidates[0]
    raise ValueError("no swarm project roots available")


def _iter_swarm_spec_candidates(project_root: Path, limit: int = 80) -> list[str]:
    preferred = [
        "PROGRAM.md",
        "program.md",
        "spec.md",
        "SPEC.md",
        "prompt.md",
        "PROMPT.md",
        "README.md",
    ]
    ordered: list[str] = []
    seen: set[str] = set()

    def add_relative(path: Path) -> None:
        relative = path.relative_to(project_root).as_posix()
        if relative not in seen:
            seen.add(relative)
            ordered.append(relative)

    for name in preferred:
        candidate = project_root / name
        if candidate.exists() and candidate.is_file():
            add_relative(candidate)

    for current_root, dirnames, filenames in os.walk(project_root):
        current_path = Path(current_root)
        dirnames[:] = [name for name in dirnames if name not in _SWARM_FILE_SKIP_DIRS and not name.startswith(".")]
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in {".md", ".txt"}:
                continue
            candidate = current_path / filename
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            if size > 128_000:
                continue
            add_relative(candidate)
            if len(ordered) >= limit:
                return ordered
    return ordered


def _read_swarm_preview(path: Path, limit: int = 4000) -> tuple[str, bool]:
    content = path.read_text(encoding="utf-8")
    return content[:limit], len(content) > limit


def _resolve_swarm_spec_target(project_root: Path, spec_path: str | None) -> tuple[Path, str]:
    default_spec = _default_swarm_spec_name(project_root)
    requested = (spec_path or default_spec).strip() or default_spec
    raw = Path(requested).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    if not candidate.is_relative_to(project_root):
        raise ValueError(f"swarm spec must stay under the selected project root: {project_root}")
    if candidate.exists() and not candidate.is_file():
        raise ValueError(f"swarm spec is not a file: {candidate}")
    return candidate, candidate.relative_to(project_root).as_posix()


def _load_swarm_spec_document(project_root: Path, spec_path: str | None) -> dict[str, Any]:
    candidate, relative_path = _resolve_swarm_spec_target(project_root, spec_path)
    exists = candidate.is_file()
    content = candidate.read_text(encoding="utf-8") if exists else ""
    return {
        "path": relative_path,
        "content": content,
        "exists": exists,
        "is_default": relative_path == _default_swarm_spec_name(project_root),
    }


def _materialize_swarm_spec(
    *,
    project_root: Path,
    spec_path: str | None,
    spec_mode: Literal["existing", "inline"],
    spec_content: str | None,
) -> tuple[Path, str, Literal["explicit", "default"], bool]:
    if spec_mode == "inline" and spec_content is None:
        raise ValueError("inline swarm specs require spec_content")
    target, relative_path = _resolve_swarm_spec_target(project_root, spec_path)
    # Only an *inline* request writes the spec. Gating on ``spec_content is not
    # None`` alone let a ``spec_mode="existing"`` request with an empty/bound
    # textarea value overwrite (blank out) the user's on-disk spec.
    if spec_mode == "inline" and spec_content is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(spec_content, encoding="utf-8")
    resolved_spec_path, spec_resolution, used_program_md = resolve_swarm_spec_path(
        project_root=project_root,
        spec_path=relative_path,
    )
    return resolved_spec_path, relative_path, spec_resolution, used_program_md


def _extract_enum_params(input_schema: dict[str, Any]) -> list[dict[str, Any]]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return []

    priority = {"op": 0, "action": 1, "task_type": 2, "budget": 3}
    enum_params: list[dict[str, Any]] = []
    for param_name, raw_spec in properties.items():
        if not isinstance(raw_spec, dict):
            continue
        raw_enum = raw_spec.get("enum")
        if not isinstance(raw_enum, list) or len(raw_enum) <= 1:
            continue
        options = [str(item) for item in raw_enum if item is not None]
        if len(options) > 20:
            continue
        enum_params.append(
            {
                "name": str(param_name),
                "options": options,
                "description": str(raw_spec.get("description") or ""),
            }
        )

    enum_params.sort(key=lambda item: (priority.get(item["name"], 50), item["name"]))
    return enum_params


# --------------------------------------------------------------------------- #
# Savings helpers                                                             #
# --------------------------------------------------------------------------- #


def _normalize_lever(operation: str) -> str:
    op = (operation or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    if "search_read" in op or "smart_read" in op:
        return "search_read"
    if "batch_edit" in op:
        return "batch_edit"
    if "session_compact" in op or ("compact" in op and "session" in op):
        return "session_compaction"
    if "model_recommend" in op or "model_routing" in op or "routing" in op:
        return "model_routing"
    if "ast" in op and ("trunc" in op or "outline" in op):
        return "ast_truncation"
    if "sleep" in op:
        return "sleeptime"
    if "cache" in op:
        return "cached_read"
    if "recall" in op:
        return "scoped_recall"
    if "compact" in op:
        return "compact_lifecycle"
    if "playbook" in op or "inject" in op:
        return "playbook_inject"
    return op


_BILLABLE_ANALYTICS_EVENTS = {
    "prompt",
    "cached_prompt",
    "cache_create",
    "result",
    "thinking",
    "tool_charge",
}
_NATIVE_ANALYTICS_TOOLS = {
    "Read",
    "Bash",
    "Edit",
    "Grep",
    "Glob",
    "Write",
    "Agent",
    "ListDir",
    "bash",
    "run_shell_command",
    "exec_command",
    "shell",
    "read_file",
    "replace",
    "apply_patch",
    "view",
}


def _llm_usage_cost(
    model_id: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    thinking_tokens: int = 0,
) -> float:
    return usage_cost_usd(
        model_id or "_default",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        thinking_tokens=thinking_tokens,
    )


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    return sum(
        int(usage.get(field) or 0)
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
            "thinking_tokens",
            # reasoning_output_tokens is a subset of output_tokens.
        )
    )


def _model_usage_cost(usage: dict[str, Any]) -> float:
    return _llm_usage_cost(
        str(usage.get("model") or "_default"),
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=int(usage.get("cached_input_tokens") or 0),
        cache_write_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        thinking_tokens=int(usage.get("thinking_tokens") or 0),
    )


def _normalize_trace_usage_entry(raw_entry: Any, *, fallback_model: str = "") -> dict[str, Any] | None:
    if not isinstance(raw_entry, dict):
        return None

    kind = str(raw_entry.get("kind") or "llm").strip().lower()
    if kind == "tool":
        tool_name = str(raw_entry.get("tool_name") or raw_entry.get("name") or "").strip()
        input_tokens = int(raw_entry.get("input_tokens") or 0)
        output_tokens = int(raw_entry.get("output_tokens") or 0)
        cost_usd = float(raw_entry.get("cost_usd") or raw_entry.get("cost") or 0.0)
        if not tool_name and input_tokens == 0 and output_tokens == 0 and cost_usd == 0.0:
            return None
        return {
            "kind": "tool",
            "tool_name": tool_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "source_type": str(raw_entry.get("source_type") or ""),
            "source_id": str(raw_entry.get("source_id") or ""),
        }

    model_id = str(raw_entry.get("model") or fallback_model or "").strip()
    entry = {
        "kind": "llm",
        "model": model_id,
        "input_tokens": int(raw_entry.get("input_tokens") or 0),
        "output_tokens": int(raw_entry.get("output_tokens") or 0),
        "reasoning_output_tokens": min(
            int(raw_entry.get("reasoning_output_tokens") or 0),
            int(raw_entry.get("output_tokens") or 0),
        ),
        "thinking_tokens": int(raw_entry.get("thinking_tokens") or 0),
        "cached_input_tokens": int(raw_entry.get("cached_input_tokens") or 0),
        "cache_creation_input_tokens": int(raw_entry.get("cache_creation_input_tokens") or 0),
        "source_type": str(raw_entry.get("source_type") or ""),
        "source_id": str(raw_entry.get("source_id") or ""),
    }
    token_fields = (
        entry["input_tokens"],
        entry["output_tokens"],
        entry["reasoning_output_tokens"],
        entry["thinking_tokens"],
        entry["cached_input_tokens"],
        entry["cache_creation_input_tokens"],
    )
    if not model_id and not any(token_fields):
        return None
    return entry


def _trace_usage_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    fallback_model = str(payload.get("model") or "")
    normalized: list[dict[str, Any]] = []

    raw_entries = payload.get("usage_entries")
    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            entry = _normalize_trace_usage_entry(raw_entry, fallback_model=fallback_model)
            if entry is not None:
                normalized.append(entry)
        if normalized:
            return normalized

    raw_usages = payload.get("model_usages")
    if isinstance(raw_usages, list):
        for raw_usage in raw_usages:
            entry = _normalize_trace_usage_entry(raw_usage, fallback_model=fallback_model)
            if entry is not None:
                normalized.append(entry)
        if normalized:
            return normalized

    entry = _normalize_trace_usage_entry(
        {
            "kind": "llm",
            "model": fallback_model,
            "input_tokens": payload.get("input_tokens") or 0,
            "output_tokens": payload.get("output_tokens") or 0,
            "reasoning_output_tokens": payload.get("reasoning_output_tokens") or 0,
            "thinking_tokens": payload.get("thinking_tokens") or 0,
            "cached_input_tokens": payload.get("cached_input_tokens") or 0,
            "cache_creation_input_tokens": payload.get("cache_creation_input_tokens") or 0,
        }
    )
    return [entry] if entry is not None else []


def _trace_model_usages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for raw_entry in _trace_usage_entries(payload):
        if raw_entry.get("kind") == "tool":
            continue
        model_id = str(raw_entry.get("model") or "").strip()
        usage: dict[str, Any] = {
            "input_tokens": int(raw_entry.get("input_tokens") or 0),
            "output_tokens": int(raw_entry.get("output_tokens") or 0),
            "reasoning_output_tokens": int(raw_entry.get("reasoning_output_tokens") or 0),
            "thinking_tokens": int(raw_entry.get("thinking_tokens") or 0),
            "cached_input_tokens": int(raw_entry.get("cached_input_tokens") or 0),
            "cache_creation_input_tokens": int(raw_entry.get("cache_creation_input_tokens") or 0),
        }
        if not model_id and not any(usage.values()):
            continue
        bucket = aggregated.setdefault(model_id, {"model": model_id, **{key: 0 for key in usage}})
        for field, value in usage.items():
            bucket[field] += value

    usages = list(aggregated.values())
    usages.sort(key=lambda usage: (_model_usage_cost(usage), _usage_total_tokens(usage)), reverse=True)
    return usages


def _trace_session_model(
    payload: dict[str, Any],
    usage_entries: list[dict[str, Any]] | None = None,
    model_usages: list[dict[str, Any]] | None = None,
) -> str:
    entries = usage_entries if usage_entries is not None else _trace_usage_entries(payload)
    unique_models = {
        str(entry.get("model") or "").strip()
        for entry in entries
        if entry.get("kind") != "tool" and str(entry.get("model") or "").strip()
    }
    if len(unique_models) == 1:
        return next(iter(unique_models))
    if len(unique_models) > 1:
        return ""

    usages = model_usages if model_usages is not None else _trace_model_usages(payload)
    usage_models = {str(usage.get("model") or "").strip() for usage in usages if str(usage.get("model") or "").strip()}
    if len(usage_models) == 1:
        return next(iter(usage_models))
    if usage_models:
        return ""
    return str(payload.get("model") or "").strip()


def _trace_cost_from_payload(payload: dict[str, Any]) -> float:
    usage_entries = _trace_usage_entries(payload)
    if not usage_entries:
        return 0.0

    total_cost = 0.0
    for entry in usage_entries:
        if entry.get("kind") == "tool":
            total_cost += float(entry.get("cost_usd") or 0.0)
            continue
        total_cost += _llm_usage_cost(
            str(entry.get("model") or "_default"),
            input_tokens=int(entry.get("input_tokens") or 0),
            output_tokens=int(entry.get("output_tokens") or 0),
            cache_read_tokens=int(entry.get("cached_input_tokens") or 0),
            cache_write_tokens=int(entry.get("cache_creation_input_tokens") or 0),
            thinking_tokens=int(entry.get("thinking_tokens") or 0),
        )
    return round(total_cost, 8)


def _trace_cost_breakdown_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    usage_entries = _trace_usage_entries(payload)
    if not usage_entries:
        return {
            "input_token_cost_usd": 0.0,
            "output_token_cost_usd": 0.0,
            "cache_read_cost_usd": 0.0,
            "cache_write_cost_usd": 0.0,
        }

    input_token_cost_usd = 0.0
    output_token_cost_usd = 0.0
    cache_read_cost_usd = 0.0
    cache_write_cost_usd = 0.0
    for entry in usage_entries:
        if entry.get("kind") == "tool":
            continue
        model_id = str(entry.get("model") or "_default")
        input_token_cost_usd += _llm_usage_cost(model_id, input_tokens=int(entry.get("input_tokens") or 0))
        output_token_cost_usd += _llm_usage_cost(model_id, output_tokens=int(entry.get("output_tokens") or 0))
        cache_read_cost_usd += _llm_usage_cost(model_id, cache_read_tokens=int(entry.get("cached_input_tokens") or 0))
        cache_write_cost_usd += _llm_usage_cost(
            model_id,
            cache_write_tokens=int(entry.get("cache_creation_input_tokens") or 0),
        )
    return {
        "input_token_cost_usd": round(input_token_cost_usd, 8),
        "output_token_cost_usd": round(output_token_cost_usd, 8),
        "cache_read_cost_usd": round(cache_read_cost_usd, 8),
        "cache_write_cost_usd": round(cache_write_cost_usd, 8),
    }


def _trace_models_used_from_payload(
    payload: dict[str, Any],
    usage_entries: list[dict[str, Any]] | None = None,
    model_usages: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    entries = usage_entries if usage_entries is not None else _trace_usage_entries(payload)
    models_used: dict[str, int] = {}
    for entry in entries:
        if entry.get("kind") == "tool":
            continue
        model_id = str(entry.get("model") or "").strip()
        if model_id:
            models_used[model_id] = models_used.get(model_id, 0) + 1
    if models_used:
        return models_used

    usages = model_usages if model_usages is not None else _trace_model_usages(payload)
    for usage in usages:
        model_id = str(usage.get("model") or "").strip()
        if model_id:
            models_used[model_id] = 1
    if models_used:
        return models_used

    started_model = _trace_session_model(payload, entries, usages).strip()
    return {started_model: 1} if started_model else {}


def _analytics_event_cost(model_id: str | None, event_type: str, input_tokens: int, output_tokens: int) -> float:
    if event_type == "prompt":
        return _llm_usage_cost(model_id, input_tokens=input_tokens)
    if event_type == "cached_prompt":
        return _llm_usage_cost(model_id, cache_read_tokens=input_tokens)
    if event_type == "cache_create":
        return _llm_usage_cost(model_id, cache_write_tokens=input_tokens)
    if event_type == "result":
        return _llm_usage_cost(model_id, output_tokens=output_tokens)
    if event_type == "thinking":
        return _llm_usage_cost(model_id, thinking_tokens=output_tokens)
    if event_type == "tool_call":
        # Tool outputs re-enter the next prompt window and use input pricing.
        return _llm_usage_cost(model_id, input_tokens=output_tokens)
    return 0.0


def _filter_analytics_rows(
    rows: list[dict[str, Any]],
    *,
    agent: str | None = None,
    model: str | None = None,
    category: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    agent_match = (agent or "").strip().lower()
    model_match = (model or "").strip().lower()
    search_match = (search or "").strip().lower()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_agent = str(row.get("agent") or "").strip().lower()
        row_model = str(row.get("model") or "").strip().lower()
        if agent_match and row_agent != agent_match:
            continue
        if model_match and row_model != model_match:
            continue
        if category and row.get("category") != category:
            continue
        if search_match:
            tool_name = str(row.get("tool_name") or "").lower()
            sub_command = str(row.get("sub_command") or "").lower()
            if search_match not in tool_name and search_match not in sub_command:
                continue
        filtered.append(row)
    return filtered


# ~100 years — comfortably beyond any real analytics window, yet small enough
# that ``datetime.now() - timedelta(days=_MAX_WINDOW_DAYS)`` stays in range. A
# user-supplied ``days=1000000000`` otherwise raises OverflowError -> HTTP 500.
_MAX_WINDOW_DAYS = 36600


def _clamp_days(value: int) -> int:
    """Clamp a user-supplied day/window count to a sane, non-overflowing range."""
    return max(1, min(int(value), _MAX_WINDOW_DAYS))


def _build_analytics_summary(rows: list[dict[str, Any]], *, days: int | None) -> dict[str, Any]:
    total_output_tokens = sum(
        int(row.get("output_tokens") or 0)
        for row in rows
        if row.get("event_type") in {"result", "thinking", "tool_call"}
    )
    user_input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows if row.get("event_type") == "user_string")
    tool_calls = sum(int(row.get("call_count") or 1) for row in rows if row.get("event_type") == "tool_call")
    unique_tools = len(
        {
            str(row.get("tool_name") or "")
            for row in rows
            if row.get("event_type") == "tool_call" and row.get("tool_name")
        }
    )
    cached_prompt_tokens = sum(
        int(row.get("input_tokens") or 0) for row in rows if row.get("event_type") == "cached_prompt"
    )
    model_response_tokens = sum(int(row.get("output_tokens") or 0) for row in rows if row.get("event_type") == "result")
    model_thinking_tokens = sum(
        int(row.get("output_tokens") or 0) for row in rows if row.get("event_type") == "thinking"
    )
    tool_input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows if row.get("event_type") == "tool_call")
    tool_output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows if row.get("event_type") == "tool_call")
    total_cost = round(
        sum(float(row.get("cost") or 0.0) for row in rows if row.get("event_type") in _BILLABLE_ANALYTICS_EVENTS),
        6,
    )
    if days is not None:
        effective_days = max(1, days)
    else:
        # No explicit window: ``days or 1`` collapsed the span to 1 day, turning
        # the projection into (all-time cost) x 30. Derive the real span from the
        # row timestamps instead so the monthly estimate is meaningful.
        stamps: list[datetime] = []
        for row in rows:
            for key in ("first_seen", "last_seen"):
                raw = str(row.get(key) or "")
                if raw:
                    with contextlib.suppress(ValueError):
                        stamps.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        span_days = (max(stamps) - min(stamps)).days + 1 if stamps else 1
        effective_days = max(1, span_days)
    estimated_monthly_cost = round(total_cost * (30 / effective_days), 6)

    tool_costs: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        tool_costs[str(row.get("tool_name") or "—")] += float(row.get("cost") or 0.0)
    top_cost_driver = max(tool_costs.items(), key=lambda item: item[1])[0] if tool_costs else "—"

    return {
        "total_cost": total_cost,
        "estimated_monthly_cost": estimated_monthly_cost,
        "top_cost_driver": top_cost_driver,
        "user_input_tokens": user_input_tokens,
        "model_thinking_tokens": model_thinking_tokens,
        "llm_output_tokens": model_response_tokens + tool_input_tokens,
        "tool_output_tokens": tool_output_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "tool_calls": tool_calls,
        "unique_tools": unique_tools,
        "total_output_tokens": total_output_tokens,
        "row_count": len(rows),
    }


def _tool_category(tool_name: str) -> str:
    return "Native / Unoptimized" if tool_name in _NATIVE_ANALYTICS_TOOLS else "LemonCrow Optimized"


def _trace_analytics_events(
    trace_id: str,
    agent: str,
    host: str | None,
    created_at: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    usage_entries = _trace_usage_entries(payload)
    model_usages = _trace_model_usages(payload)
    session_model = _trace_session_model(payload, usage_entries, model_usages)
    session_host = host or agent

    def _append_event(
        *,
        model: str,
        event_type: str,
        tool_name: str,
        category: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_count: int = 1,
        sub_command: str | None = None,
        cost_override: float | None = None,
    ) -> None:
        events.append(
            {
                "trace_id": trace_id,
                "host": session_host,
                "agent": agent,
                "model": model,
                "event_type": event_type,
                "tool_name": tool_name,
                "sub_command": sub_command,
                "category": category,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "first_seen": created_at,
                "last_seen": created_at,
                "call_count": call_count,
                "session_count": 1,
                "cost": (
                    float(cost_override)
                    if cost_override is not None
                    else _analytics_event_cost(model, event_type, input_tokens, output_tokens)
                ),
            }
        )

    user_prompt_tokens = int(payload.get("user_prompt_tokens") or 0)
    if user_prompt_tokens > 0:
        _append_event(
            model=session_model,
            event_type="user_string",
            tool_name="User Entered String",
            category="User Activity",
            input_tokens=user_prompt_tokens,
        )

    for entry in usage_entries:
        if entry.get("kind") == "tool":
            tool_name = str(entry.get("tool_name") or "unknown")
            cost_usd = float(entry.get("cost_usd") or 0.0)
            if cost_usd > 0:
                _append_event(
                    model="",
                    event_type="tool_charge",
                    tool_name=tool_name,
                    category="Tool Billing",
                    cost_override=cost_usd,
                )
            continue

        model_id = str(entry.get("model") or session_model)
        input_tokens = int(entry.get("input_tokens") or 0)
        cached_input_tokens = int(entry.get("cached_input_tokens") or 0)
        cache_write_tokens = int(entry.get("cache_creation_input_tokens") or 0)
        output_tokens = int(entry.get("output_tokens") or 0)
        thinking_tokens = int(entry.get("thinking_tokens") or 0)

        if input_tokens > 0:
            _append_event(
                model=model_id,
                event_type="prompt",
                tool_name="Context Window (Base)",
                category="LLM Context",
                input_tokens=input_tokens,
            )
        if cached_input_tokens > 0:
            _append_event(
                model=model_id,
                event_type="cached_prompt",
                tool_name="Cached Prompt (Cache Read)",
                category="LLM Context (Cache Read)",
                input_tokens=cached_input_tokens,
            )
        if cache_write_tokens > 0:
            _append_event(
                model=model_id,
                event_type="cache_create",
                tool_name="Cache Write",
                category="LLM Context (Cache Write)",
                input_tokens=cache_write_tokens,
            )
        if output_tokens > 0:
            _append_event(
                model=model_id,
                event_type="result",
                tool_name="Assistant Response",
                category="LLM Generation",
                output_tokens=output_tokens,
            )
        if thinking_tokens > 0:
            _append_event(
                model=model_id,
                event_type="thinking",
                tool_name="Thinking",
                category="LLM Generation",
                output_tokens=thinking_tokens,
            )

    for raw_tool in payload.get("tools_called") or []:
        if not isinstance(raw_tool, dict):
            continue
        tool_name = str(raw_tool.get("name") or "unknown")
        _append_event(
            model=session_model,
            event_type="tool_call",
            tool_name=tool_name,
            category=_tool_category(tool_name),
            input_tokens=int(raw_tool.get("input_tokens") or 0),
            output_tokens=int(raw_tool.get("output_tokens") or 0),
            call_count=int(raw_tool.get("count") or 1),
        )

    return events


def _group_analytics_rows(events: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    session_ids: dict[tuple[Any, ...], set[str]] = {}

    for event in events:
        key = (
            event["host"],
            event["agent"],
            event["model"],
            event["event_type"],
            event["tool_name"],
            event["sub_command"],
            event["category"],
        )
        row = grouped_rows.get(key)
        if row is None:
            row = dict(event)
            grouped_rows[key] = row
            session_ids[key] = {str(event["trace_id"])}
            continue

        row["input_tokens"] += int(event.get("input_tokens") or 0)
        row["output_tokens"] += int(event.get("output_tokens") or 0)
        row["call_count"] += int(event.get("call_count") or 0)
        row["cost"] = round(float(row.get("cost") or 0.0) + float(event.get("cost") or 0.0), 8)
        row["first_seen"] = min(str(row.get("first_seen") or ""), str(event.get("first_seen") or ""))
        row["last_seen"] = max(str(row.get("last_seen") or ""), str(event.get("last_seen") or ""))
        session_ids[key].add(str(event["trace_id"]))

    rows: list[dict[str, Any]] = []
    for key, row in grouped_rows.items():
        row["session_count"] = len(session_ids[key])
        row.pop("trace_id", None)
        rows.append(row)

    rows.sort(key=lambda row: str(row.get("last_seen") or ""), reverse=True)
    return rows[:limit]


def _query_analytics_rows(
    db_path: str | Path,
    *,
    grouped: bool,
    days: int | None,
    limit: int,
    agent: str | None = None,
    model: str | None = None,
    category: str | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    start_day = None
    if days is not None:
        window_days = _clamp_days(int(days))
        start_day = (datetime.now().astimezone().date() - timedelta(days=window_days - 1)).isoformat()

    params: list[Any] = []

    sql = f"""
        SELECT id, agent, host, payload, created_at
        FROM traces
        WHERE 1=1 {"AND date(datetime(created_at, 'localtime')) >= ?" if start_day else ""}
        ORDER BY created_at DESC
    """

    if start_day:
        params.append(start_day)

    events: list[dict[str, Any]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(sql, params).fetchall():
            try:
                payload = json.loads(row["payload"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            events.extend(
                _trace_analytics_events(
                    str(row["id"]),
                    str(row["agent"] or ""),
                    str(row["host"] or ""),
                    str(row["created_at"] or ""),
                    payload,
                )
            )

    # Filter BEFORE applying the row limit: filtering the newest-N slice (as the
    # callers used to) silently dropped matching rows that fell outside the most
    # recent `limit` events, understating or zeroing filtered results.
    events = _filter_analytics_rows(events, agent=agent, model=model, category=category, search=search)

    if grouped:
        return _group_analytics_rows(events, limit=limit)

    events.sort(key=lambda event: str(event.get("last_seen") or ""), reverse=True)
    rows = [dict(event) for event in events[:limit]]
    for row in rows:
        row.pop("trace_id", None)
    return rows


def _iter_live_savings_events(root: Path) -> list[dict[str, Any]]:
    path = root / "live_savings_events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _latest_savings_benchmark(root: Path) -> dict[str, Any] | None:
    bench_dir = root / "benchmarks" / "savings"

    # ── Paired command benchmark (savings/latest.json) ─────────────────────
    paired: dict[str, Any] = {}
    paired_path = bench_dir / "latest.json"
    if paired_path.exists():
        try:
            payload = json.loads(paired_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                paired_keys = {
                    "session_id",
                    "model",
                    "n_prompts",
                    "total_tokens_baseline",
                    "total_tokens_lemoncrow",
                    "tokens_saved",
                    "reduction_pct",
                    "total_cost_baseline_usd",
                    "total_cost_lemoncrow_usd",
                    "cost_saved_usd",
                    "total_time_baseline_ms",
                    "total_time_lemoncrow_ms",
                    "time_saved_ms",
                    "baseline_success_rate",
                    "lemoncrow_success_rate",
                }
                paired = {k: payload[k] for k in paired_keys if k in payload}
        except (OSError, json.JSONDecodeError):
            pass

    # ── Compact replay benchmark (compact_latest.json) ────────────────────
    compact: dict[str, Any] = {}
    compact_path = bench_dir / "compact_latest.json"
    if compact_path.exists():
        try:
            payload = json.loads(compact_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                compact = {
                    "sessions_benchmarked": payload.get("sessions_benchmarked", 0),
                    "avg_native_freed_tokens_measured": payload.get("avg_native_freed_tokens_measured", 0),
                    "avg_lemoncrow_freed_tokens_est": payload.get("avg_lemoncrow_freed_tokens_est", 0),
                    "avg_delta_tokens": payload.get("avg_delta_tokens", 0),
                    "lemoncrow_vs_native_delta_pct": payload.get("lemoncrow_vs_native_delta_pct", 0.0),
                    "total_cost_saved_usd": payload.get("total_cost_saved_usd", 0.0),
                    "generated_at": payload.get("generated_at", ""),
                    "note": payload.get("note", ""),
                }
        except (OSError, json.JSONDecodeError):
            pass

    # ── Routing replay benchmark (routing_latest.json) ────────────────────
    routing: dict[str, Any] = {}
    routing_path = bench_dir / "routing_latest.json"
    if routing_path.exists():
        try:
            payload = json.loads(routing_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                routing = {
                    "sessions_benchmarked": payload.get("sessions_benchmarked", 0),
                    "total_turns_analyzed": payload.get("total_turns_analyzed", 0),
                    "total_downtiered_turns": payload.get("total_downtiered_turns", 0),
                    "downtiered_pct": payload.get("downtiered_pct", 0.0),
                    "total_cost_saved_usd": payload.get("total_cost_saved_usd", 0.0),
                    "avg_cost_saved_usd_per_session": payload.get("avg_cost_saved_usd_per_session", 0.0),
                    "by_tier": payload.get("by_tier", {}),
                    "generated_at": payload.get("generated_at", ""),
                    "note": payload.get("note", ""),
                }
        except (OSError, json.JSONDecodeError):
            pass

    if not paired and not compact and not routing:
        return None

    result: dict[str, Any] = {}
    if paired:
        result.update(paired)
    if compact:
        result["compact_bench"] = compact
    if routing:
        result["routing_bench"] = routing
    return result


_OBSERVED_OPTIMIZATION_TITLES = {
    "search_read": "Search/read compaction",
    "batch_edit": "Batch edit",
    "compact_lifecycle": "Tool-output compaction",
    "session_compaction": "Session compaction",
    "model_routing": "Model routing (tier downgrade)",
    "scoped_recall": "Scoped recall",
    "playbook_inject": "Playbook injection",
    "cached_read": "Cached reuse",
    "delta_read": "Delta read",
    "structure_map": "Structure map",
}


def _trace_created_at(trace: Trace) -> datetime:
    created = trace.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created


def _trace_run_key(trace: Trace) -> str:
    return str(trace.session_id or trace.id)


def _trace_total_tokens(trace: Trace) -> int:
    from lemoncrow.pro.capabilities.session_optimizer import effective_input_tokens

    return effective_input_tokens(trace) + int(trace.output_tokens or 0)


def _trace_cache_leverage(trace: Trace) -> float:
    from lemoncrow.pro.capabilities.session_optimizer import effective_input_tokens

    effective_input = effective_input_tokens(trace)
    if effective_input <= 0:
        return 0.0
    return round(int(trace.cached_input_tokens or 0) / effective_input, 4)


def _recent_traces(store: StoreBundle, *, window_days: int) -> list[Trace]:
    cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(window_days))
    traces = store.history.list_traces(limit=5000)
    recent = [trace for trace in traces if _trace_created_at(trace) >= cutoff]
    return sorted(recent, key=_trace_created_at)


def _tracked_saved_tokens(store: StoreBundle, trace: Trace) -> tuple[int, int]:
    try:
        rows = store.telemetry.list_context_budgets(_trace_run_key(trace))
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        logger.warning("Failed to load context budgets for trace %s: %s", trace.id, exc)
        return 0, 0

    saved_tokens = 0
    tracked_turns = 0
    for row in rows:
        tracked_turns += int(row.tool_calls or 0)
        lever_keys = [str(key or "") for key in row.lever_savings]
        non_marker_keys = [key for key in lever_keys if not key.startswith("tool:")]
        if non_marker_keys and all(key.startswith("compact_tool_output:") for key in non_marker_keys):
            continue
        actual_tokens = (
            int(row.input_tokens or 0)
            + int(row.cache_read_tokens or 0)
            + int(row.cache_write_tokens or 0)
            + int(row.output_tokens or 0)
        )
        naive_tokens = int(row.naive_input_tokens or 0)
        saved_tokens += max(0, naive_tokens - actual_tokens)
    return saved_tokens, tracked_turns


def _window_metrics(store: StoreBundle, traces: list[Trace]) -> dict[str, Any]:
    from lemoncrow.pro.capabilities.session_optimizer import trace_cost_usd

    entries: list[dict[str, Any]] = []
    for trace in traces:
        saved_tokens, tracked_turns = _tracked_saved_tokens(store, trace)
        entries.append(
            {
                "trace_id": trace.id,
                "tokens": _trace_total_tokens(trace),
                "cost_usd": trace_cost_usd(trace),
                "cache_leverage": _trace_cache_leverage(trace),
                "saved_tokens": saved_tokens,
                "tracked_turns": tracked_turns,
                "created_at": _trace_created_at(trace).isoformat(),
            }
        )

    count = len(entries)
    if count == 0:
        return {
            "trace_count": 0,
            "avg_tokens": 0,
            "avg_cost_usd": 0.0,
            "avg_cache_leverage": 0.0,
            "avg_saved_tokens": 0,
            "tracked_turns": 0,
            "from": None,
            "to": None,
        }

    return {
        "trace_count": count,
        "avg_tokens": round(sum(int(item["tokens"]) for item in entries) / count),
        "avg_cost_usd": round(sum(float(item["cost_usd"]) for item in entries) / count, 6),
        "avg_cache_leverage": round(sum(float(item["cache_leverage"]) for item in entries) / count, 4),
        "avg_saved_tokens": round(sum(int(item["saved_tokens"]) for item in entries) / count),
        "tracked_turns": sum(int(item["tracked_turns"]) for item in entries),
        "from": entries[0]["created_at"],
        "to": entries[-1]["created_at"],
    }


def _pct_change(before: float, after: float) -> float:
    if before <= 0:
        if after <= 0:
            return 0.0
        return 1.0
    return round((after - before) / before, 4)


def _impact_verdict(tokens_delta: float, cost_delta: float, cache_delta: float) -> str:
    improvements = 0
    regressions = 0

    if tokens_delta <= -0.05:
        improvements += 1
    elif tokens_delta >= 0.05:
        regressions += 1

    if cost_delta <= -0.05:
        improvements += 1
    elif cost_delta >= 0.05:
        regressions += 1

    if cache_delta >= 0.05:
        improvements += 1
    elif cache_delta <= -0.05:
        regressions += 1

    if improvements >= 2 and regressions == 0:
        return "improved"
    if regressions >= 2 and improvements == 0:
        return "regressed"
    if improvements == 0 and regressions == 0:
        return "no_change"
    return "mixed"


def _build_impact_validation(
    store: StoreBundle,
    traces: list[Trace],
    *,
    window_days: int,
) -> dict[str, Any]:
    if len(traces) < 2:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "window_days": window_days,
            "strategy": "chronological_halves",
            "verdict": "insufficient_data",
            "before": _window_metrics(store, []),
            "after": _window_metrics(store, []),
            "deltas": {
                "tokens_pct": 0.0,
                "cost_pct": 0.0,
                "cache_leverage_pct": 0.0,
                "saved_tokens_pct": 0.0,
            },
            "notes": ["Need at least two traces in the selected window to compare before vs after behavior."],
        }

    midpoint = max(1, len(traces) // 2)
    before = _window_metrics(store, traces[:midpoint])
    after = _window_metrics(store, traces[midpoint:])

    tokens_delta_pct = _pct_change(float(before["avg_tokens"]), float(after["avg_tokens"]))
    cost_delta_pct = _pct_change(float(before["avg_cost_usd"]), float(after["avg_cost_usd"]))
    cache_delta_pct = _pct_change(float(before["avg_cache_leverage"]), float(after["avg_cache_leverage"]))
    saved_tokens_delta_pct = _pct_change(float(before["avg_saved_tokens"]), float(after["avg_saved_tokens"]))

    notes: list[str] = []
    if after["avg_tokens"] < before["avg_tokens"]:
        notes.append("Average token load fell in the later half of the window.")
    if after["avg_cost_usd"] < before["avg_cost_usd"]:
        notes.append("Average per-trace cost dropped in the later half of the window.")
    if after["avg_cache_leverage"] > before["avg_cache_leverage"]:
        notes.append("Cache leverage improved in the later half of the window.")
    if not notes:
        notes.append("Later traces look materially similar to earlier traces in this window.")

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "strategy": "chronological_halves",
        "verdict": _impact_verdict(tokens_delta_pct, cost_delta_pct, cache_delta_pct),
        "before": before,
        "after": after,
        "deltas": {
            "tokens_pct": tokens_delta_pct,
            "cost_pct": cost_delta_pct,
            "cache_leverage_pct": cache_delta_pct,
            "saved_tokens_pct": saved_tokens_delta_pct,
        },
        "notes": notes,
    }


def _lever_title(lever: str) -> str:
    return _OBSERVED_OPTIMIZATION_TITLES.get(lever, lever.replace("_", " ").title())


def _build_auto_optimizations(
    savings_payload: dict[str, Any],
    live_events: list[dict[str, Any]],
    *,
    window_days: int,
) -> list[dict[str, Any]]:
    start_day = datetime.now(UTC).date() - timedelta(days=_clamp_days(window_days) - 1)
    per_lever = {
        str(key): int(value or 0)
        for key, value in (savings_payload.get("per_lever") or {}).items()
        if isinstance(key, str)
    }
    observed: dict[str, dict[str, Any]] = {}

    for event in live_events:
        tokens_saved = int(event.get("tokens_saved", 0) or 0)
        cost_saved_usd = float(event.get("cost_saved_usd", 0.0) or 0.0)
        if tokens_saved <= 0 and cost_saved_usd <= 0:
            continue
        try:
            at_date = datetime.fromisoformat(str(event.get("at", "")).replace("Z", "+00:00")).date()
        except Exception as exc:
            logging.exception("Recovered from broad exception handler")
            logger.debug("Bad timestamp %r in savings event, using today: %s", event.get("at"), exc)
            at_date = datetime.now(UTC).date()
        if at_date < start_day:
            continue
        lever = _normalize_lever(str(event.get("lever") or event.get("tool_name") or "plugin_live"))
        item = observed.setdefault(
            lever,
            {
                "id": lever,
                "title": _lever_title(lever),
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "calls_saved": 0,
                "sessions": set(),
                "tools": set(),
            },
        )
        item["tokens_saved"] += tokens_saved
        item["cost_saved_usd"] += cost_saved_usd
        item["calls_saved"] += int(event.get("calls_saved", 0) or 0)
        session_id = str(event.get("session_id") or "")
        if session_id:
            item["sessions"].add(session_id)
        tool_name = str(event.get("tool_name") or "")
        if tool_name:
            item["tools"].add(tool_name)

    rows: list[dict[str, Any]] = []
    for lever in set(per_lever) | set(observed):
        tokens_saved = int(per_lever.get(lever, 0) or 0)
        item = observed.get(
            lever,
            {
                "id": lever,
                "title": _lever_title(lever),
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "calls_saved": 0,
                "sessions": set(),
                "tools": set(),
            },
        )
        item["tokens_saved"] = max(int(item["tokens_saved"]), tokens_saved)
        if int(item["tokens_saved"]) <= 0 and float(item["cost_saved_usd"]) <= 0:
            continue
        rows.append(
            {
                "id": item["id"],
                "title": item["title"],
                "tokens_saved": int(item["tokens_saved"]),
                "cost_saved_usd": round(float(item["cost_saved_usd"]), 6),
                "calls_saved": int(item["calls_saved"]),
                "session_count": len(item["sessions"]),
                "tools": sorted(item["tools"]),
            }
        )

    return sorted(
        rows,
        key=lambda row: (int(row["tokens_saved"]), float(row["cost_saved_usd"])),
        reverse=True,
    )[:8]


def _reread_kind(event: dict[str, Any]) -> str | None:
    lever = _normalize_lever(str(event.get("lever") or ""))
    if lever in {"delta_read", "structure_map"}:
        return lever
    mode = str(event.get("read_mode") or "").strip().lower()
    if mode == "range":
        return "delta_read"
    if mode == "outline":
        return "structure_map"
    return None


def _build_reread_telemetry(root: Path, *, window_days: int) -> dict[str, Any]:
    start_day = datetime.now(UTC).date() - timedelta(days=_clamp_days(window_days) - 1)
    by_kind: dict[str, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    total_tokens_saved = 0
    total_cost_saved = 0.0
    event_count = 0

    for event in _iter_live_savings_events(root):
        tokens_saved = int(event.get("tokens_saved", 0) or 0)
        if tokens_saved <= 0:
            continue
        try:
            at = datetime.fromisoformat(str(event.get("at", "")).replace("Z", "+00:00"))
        except Exception as exc:
            logging.exception("Recovered from broad exception handler")
            logger.debug("Bad timestamp %r in reread event, using now: %s", event.get("at"), exc)
            at = datetime.now(UTC)
        if at.date() < start_day:
            continue
        kind = _reread_kind(event)
        if kind is None:
            continue

        event_count += 1
        total_tokens_saved += tokens_saved
        total_cost_saved += float(event.get("cost_saved_usd", 0.0) or 0.0)

        kind_row = by_kind.setdefault(
            kind,
            {
                "id": kind,
                "title": _lever_title(kind),
                "event_count": 0,
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "path_count": 0,
                "last_seen_at": None,
            },
        )
        kind_row["event_count"] += 1
        kind_row["tokens_saved"] += tokens_saved
        kind_row["cost_saved_usd"] += float(event.get("cost_saved_usd", 0.0) or 0.0)
        kind_row["last_seen_at"] = at.isoformat()

        path = str(event.get("path") or "(unknown file)")
        path_row = by_path.setdefault(
            path,
            {
                "path": path,
                "event_count": 0,
                "tokens_saved": 0,
                "kinds": set(),
            },
        )
        path_row["event_count"] += 1
        path_row["tokens_saved"] += tokens_saved
        path_row["kinds"].add(kind)

    for kind_row in by_kind.values():
        matching_paths = [row for row in by_path.values() if kind_row["id"] in row["kinds"]]
        kind_row["path_count"] = len(matching_paths)
        kind_row["cost_saved_usd"] = round(float(kind_row["cost_saved_usd"]), 6)

    top_paths = []
    for path_row in sorted(by_path.values(), key=lambda row: int(row["tokens_saved"]), reverse=True)[:5]:
        top_paths.append(
            {
                "path": path_row["path"],
                "event_count": int(path_row["event_count"]),
                "tokens_saved": int(path_row["tokens_saved"]),
                "kinds": sorted(path_row["kinds"]),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "event_count": event_count,
        "total_tokens_saved": total_tokens_saved,
        "total_cost_saved_usd": round(total_cost_saved, 6),
        "kinds": sorted(by_kind.values(), key=lambda row: int(row["tokens_saved"]), reverse=True),
        "top_paths": top_paths,
    }


def _cheaper_model_for(model: str | None) -> str | None:
    normalized = (model or "").strip().lower()
    if not normalized:
        return None
    mappings = [
        (("claude-opus", "claude-sonnet"), "claude-haiku-4-5"),
        (("gpt-5.5", "gpt-5.4-pro", "chat-latest"), "gpt-5.4-mini"),
        (("gpt-5.4", "gpt-5.3-codex"), "gpt-5.4-mini"),
        (("o3",), "o4-mini-deep-research"),
        (("gemini-3.1-pro", "gemini-2.5-pro"), "gemini-2.5-flash"),
        (("gemini-3-flash", "gemini-2.5-flash"), "gemini-2.5-flash-lite"),
        (("grok-4.3", "grok-4.20"), "grok-4-1-fast-non-reasoning"),
        (("deepseek-v4-pro", "deepseek-reasoner"), "deepseek-chat"),
    ]
    for prefixes, target in mappings:
        if any(prefix in normalized for prefix in prefixes):
            return target
    return None


def _routine_trace_reason(trace: Trace) -> str | None:
    tool_calls = sum(int(tool.count or 0) for tool in trace.tools_called)
    file_count = len(trace.files_touched)
    total_tokens = _trace_total_tokens(trace)
    output_tokens = int(trace.output_tokens or 0)

    if trace.status != "success":
        return None
    if trace.errors_seen or trace.repeated_failures:
        return None
    if total_tokens > 120_000:
        return None
    if output_tokens > 8_000:
        return None
    if tool_calls > 4:
        return None
    if file_count > 2:
        return None
    if any(not bool(result.passed) for result in trace.validation_results):
        return None
    return "Success with bounded tokens, limited tools, and no visible recovery work."


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _live_event_datetime(event: dict[str, Any]) -> datetime | None:
    try:
        return datetime.fromisoformat(str(event.get("at") or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def _recent_live_model_recommendations(live_events: list[dict[str, Any]], *, window_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(window_days))
    rows: list[dict[str, Any]] = []
    for event in live_events:
        if event.get("kind") != "model_recommendation":
            continue
        at_raw = str(event.get("at") or "")
        at = _live_event_datetime(event)
        if at is not None and at < cutoff:
            continue
        rows.append(
            {
                "at": at_raw,
                "session_id": event.get("session_id") or "",
                "agent": event.get("agent") or "",
                "tool_name": event.get("tool_name") or "",
                "tier": event.get("tier") or "",
                "model": event.get("model") or "",
                "score": _coerce_int(event.get("score") or 0),
                "cache_affinity_model": event.get("cache_affinity_model"),
                "cost_saved_usd": round(_coerce_float(event.get("cost_saved_usd") or 0.0), 6),
                "vs_model": event.get("vs_model") or "",
                "estimated_input_tokens": _coerce_int(event.get("estimated_input_tokens") or 0),
                "reasons": [str(reason) for reason in event.get("reasons", []) if isinstance(reason, str)],
            }
        )
    rows.sort(key=lambda row: str(row["at"]), reverse=True)
    return rows[:10]


def _build_actual_routing_savings(live_events: list[dict[str, Any]], *, window_days: int) -> dict[str, Any]:
    cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(window_days))
    calls_downtiered = 0
    total_cost_saved = 0.0
    by_tier: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    recent_events: list[dict[str, Any]] = []

    for event in live_events:
        lever = _normalize_lever(str(event.get("lever") or event.get("kind") or ""))
        if lever != "model_routing":
            continue
        at = _live_event_datetime(event)
        if at is not None and at < cutoff:
            continue
        cost_saved_usd = _coerce_float(event.get("cost_saved_usd") or 0.0)
        if cost_saved_usd <= 0:
            continue
        tier = str(event.get("tier") or "unknown")
        model = str(event.get("model") or "unknown")
        calls_downtiered += 1
        total_cost_saved += cost_saved_usd

        tier_row = by_tier.setdefault(
            tier,
            {"tier": tier, "calls_downtiered": 0, "cost_saved_usd": 0.0, "models": set()},
        )
        tier_row["calls_downtiered"] += 1
        tier_row["cost_saved_usd"] += cost_saved_usd
        tier_row["models"].add(model)

        model_row = by_model.setdefault(
            model,
            {
                "model": model,
                "tier": tier,
                "calls_downtiered": 0,
                "cost_saved_usd": 0.0,
                "sessions": set(),
            },
        )
        model_row["calls_downtiered"] += 1
        model_row["cost_saved_usd"] += cost_saved_usd
        session_id = str(event.get("session_id") or "")
        if session_id:
            model_row["sessions"].add(session_id)

        recent_events.append(
            {
                "at": str(event.get("at") or ""),
                "session_id": session_id,
                "agent": str(event.get("agent") or ""),
                "tool_name": str(event.get("tool_name") or ""),
                "tier": tier,
                "model": model,
                "cost_saved_usd": round(cost_saved_usd, 6),
                "vs_model": str(event.get("vs_model") or ""),
                "reasons": [str(reason) for reason in event.get("reasons", []) if isinstance(reason, str)],
            }
        )

    by_tier_rows = [
        {
            "tier": row["tier"],
            "calls_downtiered": int(row["calls_downtiered"]),
            "cost_saved_usd": round(float(row["cost_saved_usd"]), 6),
            "models": sorted(str(model) for model in row["models"]),
        }
        for row in by_tier.values()
    ]
    by_tier_rows.sort(key=lambda row: float(row["cost_saved_usd"]), reverse=True)

    by_model_rows = [
        {
            "model": row["model"],
            "tier": row["tier"],
            "calls_downtiered": int(row["calls_downtiered"]),
            "cost_saved_usd": round(float(row["cost_saved_usd"]), 6),
            "session_count": len({str(session_id) for session_id in row["sessions"]}),
        }
        for row in by_model.values()
    ]
    by_model_rows.sort(key=lambda row: float(row["cost_saved_usd"]), reverse=True)

    recent_events.sort(key=lambda row: str(row["at"]), reverse=True)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "calls_downtiered": calls_downtiered,
        "cost_saved_usd": round(total_cost_saved, 6),
        "by_tier": by_tier_rows,
        "by_model": by_model_rows,
        "recent_events": recent_events[:10],
    }


def _build_compact_session_history(live_events: list[dict[str, Any]], *, window_days: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(window_days))
    rows: list[dict[str, Any]] = []
    for event in live_events:
        lever = _normalize_lever(str(event.get("lever") or event.get("kind") or ""))
        if lever != "session_compaction":
            continue
        at = _live_event_datetime(event)
        if at is not None and at < cutoff:
            continue
        tokens_freed = _coerce_int(event.get("tokens_freed") or event.get("tokens_saved") or 0)
        cost_saved_usd = _coerce_float(event.get("cost_saved_usd") or 0.0)
        if tokens_freed <= 0 and cost_saved_usd <= 0:
            continue
        rows.append(
            {
                "at": str(event.get("at") or ""),
                "session_id": str(event.get("session_id") or ""),
                "agent": str(event.get("agent") or ""),
                "tokens_freed": tokens_freed,
                "cost_saved_usd": round(cost_saved_usd, 6),
                "utilisation_pct": round(_coerce_float(event.get("utilisation_pct") or 0.0), 1),
                "reason": str(event.get("reason") or ""),
                "trigger": str(event.get("trigger") or ""),
                "tokens_before": _coerce_int(event.get("tokens_before") or 0),
                "tokens_after_estimate": _coerce_int(event.get("tokens_after_estimate") or 0),
            }
        )
    rows.sort(key=lambda row: str(row["at"]), reverse=True)
    return rows[:10]


def _build_model_routing_simulation(
    traces: list[Trace], *, window_days: int, live_events: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    total_current_cost = 0.0
    total_simulated_cost = 0.0
    rerouted_tokens = 0

    for trace in traces:
        target_model = _cheaper_model_for(trace.model)
        if target_model is None:
            continue
        reason = _routine_trace_reason(trace)
        if reason is None:
            continue

        from lemoncrow.core.capabilities.pricing import get_model_pricing
        from lemoncrow.pro.capabilities.session_optimizer import trace_cost_usd

        current_cost = trace_cost_usd(trace)
        simulated_cost = get_model_pricing(target_model).cost_usd(
            input_tokens=int(trace.input_tokens or 0),
            output_tokens=int(trace.output_tokens or 0),
            cache_read_tokens=int(trace.cached_input_tokens or 0),
            cache_write_tokens=int(trace.cache_creation_input_tokens or 0),
        )
        if simulated_cost >= current_cost:
            continue

        total_current_cost += current_cost
        total_simulated_cost += simulated_cost
        rerouted_tokens += _trace_total_tokens(trace)
        candidates.append(
            {
                "trace_id": trace.id,
                "task": trace.task,
                "current_model": trace.model or "_default",
                "target_model": target_model,
                "current_cost_usd": round(current_cost, 6),
                "simulated_cost_usd": round(simulated_cost, 6),
                "estimated_cost_saved_usd": round(current_cost - simulated_cost, 6),
                "total_tokens": _trace_total_tokens(trace),
                "reason": reason,
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "candidate_count": len(candidates),
        "estimated_cost_saved_usd": round(total_current_cost - total_simulated_cost, 6),
        "current_cost_usd": round(total_current_cost, 6),
        "simulated_cost_usd": round(total_simulated_cost, 6),
        "total_tokens_rerouted": rerouted_tokens,
        "heuristic": "Conservative routine-trace filter: success only, no errors, <=120K total tokens, <=4 tool calls, <=2 files touched.",
        "candidates": sorted(candidates, key=lambda row: float(row["estimated_cost_saved_usd"]), reverse=True)[:8],
        "live_recommendations": _recent_live_model_recommendations(live_events or [], window_days=window_days),
        "actual_savings": _build_actual_routing_savings(live_events or [], window_days=window_days),
    }


def _savings_summary_payload(
    root: Path,
    *,
    window_days: int,
    store: StoreBundle | None = None,
) -> dict[str, Any]:
    from lemoncrow.core.capabilities.savings_summary import aggregate_window_savings
    from lemoncrow.infra.runtime.cost_tracker import CostTracker, load_cost_history

    window_days = max(1, min(window_days, 30))
    window_summary = aggregate_window_savings(root, days=window_days)
    history = load_cost_history(root)
    ops = history.get("operations", {}) if isinstance(history, dict) else {}
    today = datetime.now(UTC).date()
    start_day = today - timedelta(days=_clamp_days(window_days) - 1)

    by_day_seed: dict[str, dict[str, int | str]] = {}
    for i in range(window_days):
        day = (start_day + timedelta(days=i)).isoformat()
        by_day_seed[day] = {"day": day, "naive": 0, "actual": 0}

    total_naive = 0
    total_actual = 0
    per_lever: defaultdict[str, int] = defaultdict(int)
    live_cost_saved = 0.0
    live_calls_saved = 0
    live_time_saved_ms = 0
    source_totals: dict[tuple[str, str], dict[str, Any]] = {}

    for entry in ops.values() if isinstance(ops, dict) else []:
        calls = entry.get("calls", []) if isinstance(entry, dict) else []
        for call in calls:
            if not isinstance(call, dict):
                continue
            at_raw = str(call.get("at", ""))
            try:
                at_date = datetime.fromisoformat(at_raw.replace("Z", "+00:00")).date()
            except Exception as exc:
                logging.exception("Recovered from broad exception handler")
                logger.debug("Bad timestamp %r in savings call, using today: %s", at_raw, exc)
                at_date = today
            # Skip calls outside the requested window BEFORE accumulating totals —
            # otherwise the headline total_naive/total_actual/per_lever cover all
            # history while the by_day chart is windowed, so they contradict each
            # other (mirrors the live-events loop's own `at_date < start_day` guard).
            if at_date < start_day:
                continue
            input_tokens = int(call.get("input_tokens", 0) or 0)
            output_tokens = int(call.get("output_tokens", 0) or 0)
            cache_read_tokens = int(call.get("cache_read_tokens", 0) or 0)
            actual = input_tokens + output_tokens
            naive = actual + cache_read_tokens
            total_naive += naive
            total_actual += actual
            per_lever[_normalize_lever(str(call.get("operation", "unknown")))] += max(0, naive - actual)
            day_key = at_date.isoformat()
            if day_key in by_day_seed:
                by_day_seed[day_key]["naive"] = int(by_day_seed[day_key]["naive"]) + naive
                by_day_seed[day_key]["actual"] = int(by_day_seed[day_key]["actual"]) + actual

    live_events_window: list[dict[str, Any]] = []
    for event in _iter_live_savings_events(root):
        live_tokens_saved = int(event.get("live_tokens_saved", 0) or 0)
        tokens_saved = live_tokens_saved or int(event.get("tokens_saved", 0) or 0)
        cost_saved_usd = float(event.get("cost_saved_usd", 0.0) or 0.0)
        if tokens_saved <= 0 and cost_saved_usd <= 0:
            continue
        at_raw = str(event.get("at", ""))
        try:
            at_date = datetime.fromisoformat(at_raw.replace("Z", "+00:00")).date()
        except Exception as exc:
            logging.exception("Recovered from broad exception handler")
            logger.debug("Bad timestamp %r in live savings event, using today: %s", at_raw, exc)
            at_date = today
        if at_date < start_day:
            continue
        live_events_window.append(event)
        lever = _normalize_lever(str(event.get("lever") or event.get("tool_name") or "plugin_live"))
        tool_name = str(event.get("tool_name") or lever)
        per_lever.setdefault(lever, 0)
        if tokens_saved > 0:
            total_naive += tokens_saved
            per_lever[lever] += tokens_saved
        live_cost_saved += cost_saved_usd
        live_calls_saved += int(event.get("calls_saved", 0) or 0)
        live_time_saved_ms += int(event.get("time_saved_ms", 0) or 0)
        day_key = at_date.isoformat()
        if tokens_saved > 0 and day_key in by_day_seed:
            by_day_seed[day_key]["naive"] = int(by_day_seed[day_key]["naive"]) + tokens_saved
        key = (lever, tool_name)
        source = source_totals.setdefault(
            key,
            {
                "lever": lever,
                "tool_name": tool_name,
                "calls_saved": 0,
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "time_saved_ms": 0,
            },
        )
        source["calls_saved"] += int(event.get("calls_saved", 0) or 0)
        source["tokens_saved"] += tokens_saved
        source["cost_saved_usd"] += cost_saved_usd
        source["time_saved_ms"] += int(event.get("time_saved_ms", 0) or 0)

    reduction_pct = round((1.0 - (total_actual / total_naive)) * 100.0, 1) if total_naive > 0 else 0.0
    sorted_levers = dict(sorted(per_lever.items(), key=lambda kv: kv[1], reverse=True))
    cost_summary = CostTracker(root).total_savings()
    # Headline = the per-session ledger rule (_price_savings_row) — the SAME
    # figure the statusline, stop hook, and `lc savings` CLI show, so
    # every surface agrees on realized savings. The CostTracker + live-events
    # composite that used to headline here is a different spend domain
    # (internal-LLM ops + routing/compaction analytics); it stays visible as
    # ``ops_saved_usd``, never summed into the headline.
    ops_saved_usd = round(float(cost_summary["saved_usd"]) + live_cost_saved, 6)
    saved_usd = round(window_summary.saved_usd, 6)
    would_have_cost = round(window_summary.would_have_cost_usd, 6)
    actually_cost = round(window_summary.spend_usd, 6)
    saved_pct = window_summary.saved_pct
    top_sources = sorted(source_totals.values(), key=lambda row: float(row["cost_saved_usd"]), reverse=True)
    for src in top_sources:
        src["cost_saved_usd"] = round(float(src["cost_saved_usd"]), 6)
    cost_only_sources = [
        dict(source)
        for source in top_sources
        if int(source.get("tokens_saved", 0) or 0) <= 0 and float(source.get("cost_saved_usd", 0.0) or 0.0) > 0
    ]

    if store is None:
        return {
            "window_days": window_days,
            "total_naive_tokens": total_naive,
            "total_actual_tokens": total_actual,
            "reduction_pct": reduction_pct,
            "per_lever": sorted_levers,
            "by_day": list(by_day_seed.values()),
            "saved_usd": saved_usd,
            "saved_pct": saved_pct,
            "would_have_cost_usd": would_have_cost,
            "actually_cost_usd": round(actually_cost, 6),
            "total_calls": cost_summary["total_calls"],
            "live_calls_saved": live_calls_saved,
            "live_time_saved_ms": live_time_saved_ms,
            "live_saved_usd": round(live_cost_saved, 6),
            "ops_saved_usd": ops_saved_usd,
            "carry_usd": window_summary.carry_usd,
            "carry_tokens": window_summary.carry_tokens,
            "read_saved_usd": round(window_summary.read_saved_usd, 6),
            "total_saved_usd": round(window_summary.total_saved_usd, 6),
            "ledger_tokens_saved": window_summary.tokens_saved,
            "ledger_calls_saved": window_summary.calls_saved,
            "routing_saved_usd": window_summary.routing_usd,
            "top_sources": top_sources[:10],
            "cost_only_sources": cost_only_sources[:10],
            "latest_benchmark": _latest_savings_benchmark(root),
            "tracked_tool_calls": 0,
            "cost_basis": "session_ledger",
            "tool_aggregates": [],
            "session_proof": [],
            "coverage_gaps": [],
            "verification": {},
        }

    def _parse_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.now(UTC)

    def _load_run_ledger(session_id: str) -> dict[str, Any] | None:
        from lemoncrow.core.foundation.paths import find_session_dir

        # Ledgers live in the canonical sessions/YYYY/MM/DD/<host>/<id>/run.json
        # tree (see session_dir/find_session_dir).
        session_path = find_session_dir(root, session_id)
        if session_path is None:
            return None
        run_path = session_path / "run.json"
        if not run_path.exists():
            return None
        with contextlib.suppress(Exception):
            payload = json.loads(run_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        return None

    def _nearest_ledger_tool_name(ledger: dict[str, Any] | None, turn_index: int) -> str | None:
        if not ledger:
            return None
        candidates: list[tuple[int, int, str]] = []
        for idx, event in enumerate(ledger.get("events") or []):
            if not isinstance(event, dict):
                continue
            if str(event.get("kind") or "") != "tool_result":
                continue
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            tool_name = str(payload.get("tool") or "").strip()
            if tool_name:
                candidates.append((abs(idx - turn_index), -idx, tool_name))
        if not candidates:
            return None
        return min(candidates)[2]

    live_by_session_tool: dict[tuple[str, str], dict[str, Any]] = {}
    live_agent_by_session: dict[str, str] = {}
    for event in live_events_window:
        session_id = str(event.get("session_id") or "").strip()
        tool_name = str(event.get("tool_name") or "").strip()
        if session_id and isinstance(event.get("agent"), str):
            live_agent_by_session[session_id] = str(event["agent"])
        if not session_id or not tool_name:
            continue
        bucket = live_by_session_tool.setdefault(
            (session_id, tool_name),
            {
                "calls_saved": 0,
                "time_saved_ms": 0,
                "cost_saved_usd": 0.0,
                "model": "",
            },
        )
        bucket["calls_saved"] += int(event.get("calls_saved", 0) or 0)
        bucket["time_saved_ms"] += int(event.get("time_saved_ms", 0) or 0)
        bucket["cost_saved_usd"] += float(event.get("cost_saved_usd", 0.0) or 0.0)
        if isinstance(event.get("model"), str) and event.get("model"):
            bucket["model"] = str(event["model"])

    traces = store.history.list_traces(limit=5000)
    recent_traces = [trace for trace in traces if _trace_created_at(trace).date() >= start_day]
    trace_by_session = {_trace_run_key(trace): trace for trace in recent_traces}

    from lemoncrow.core.capabilities.pricing import get_model_pricing

    proof_rows: list[dict[str, Any]] = []
    with sqlite3.connect(store.telemetry.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT session_id, turn_index, model, input_tokens, cache_read_tokens,
                   cache_write_tokens, output_tokens, naive_input_tokens,
                   lever_savings_json, tool_calls, created_at
            FROM context_budget
            WHERE date(datetime(created_at)) >= ?
            ORDER BY created_at ASC, turn_index ASC
            """,
            (start_day.isoformat(),),
        ).fetchall()

    for row in rows:
        session_id = str(row["session_id"] or "")
        lever_savings = json.loads(str(row["lever_savings_json"] or "{}"))
        if not isinstance(lever_savings, dict):
            lever_savings = {}
        non_marker_keys = [
            str(key) for key, value in lever_savings.items() if not str(key).startswith("tool:") and int(value or 0) > 0
        ]
        compact_keys = [key for key in non_marker_keys if key.startswith("compact_tool_output:")]
        compact_only = bool(non_marker_keys) and len(compact_keys) == len(non_marker_keys)
        lever_key = compact_keys[0] if compact_keys else (non_marker_keys[0] if non_marker_keys else "unattributed")
        lever = (
            _normalize_lever(lever_key.split(":", 1)[1])
            if lever_key.startswith("compact_tool_output:")
            else _normalize_lever(lever_key)
        )
        if lever == "unknown" and not non_marker_keys:
            lever = "unattributed"

        input_tokens = int(row["input_tokens"] or 0)
        cache_read_tokens = int(row["cache_read_tokens"] or 0)
        cache_write_tokens = int(row["cache_write_tokens"] or 0)
        output_tokens = int(row["output_tokens"] or 0)
        actual_tokens = input_tokens + cache_read_tokens + cache_write_tokens + output_tokens
        naive_tokens = int(row["naive_input_tokens"] or 0)
        saved_tokens = max(0, naive_tokens - actual_tokens)
        model = str(row["model"] or "_default")

        ledger = _load_run_ledger(session_id)
        matched_tool_name = _nearest_ledger_tool_name(ledger, int(row["turn_index"] or 0))
        if not matched_tool_name and len([key for (sid, key) in live_by_session_tool if sid == session_id]) == 1:
            matched_tool_name = next(key for (sid, key) in live_by_session_tool if sid == session_id)
        if not matched_tool_name:
            matched_tool_name = "unattributed"

        live_bucket = live_by_session_tool.get((session_id, matched_tool_name), {})
        pricing_model = str(live_bucket.get("model") or model or "claude-sonnet-4")
        if pricing_model in {"", "_default", "test-model"}:
            pricing_model = "claude-sonnet-4"
        pricing = get_model_pricing(pricing_model)
        actual_cost_usd = round(
            pricing.cost_usd(
                input_tokens=input_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                output_tokens=output_tokens,
            ),
            6,
        )
        baseline_cost_usd = (
            actual_cost_usd
            if saved_tokens <= 0
            else round(actual_cost_usd + pricing.cost_usd(input_tokens=saved_tokens), 6)
        )
        saved_cost_usd = round(max(0.0, baseline_cost_usd - actual_cost_usd), 6)
        trace = trace_by_session.get(session_id)
        agent = ""
        task = ""
        if ledger:
            agent = str(ledger.get("agent") or "")
            task = str(ledger.get("task") or "")
        if trace is not None:
            agent = agent or str(trace.agent or trace.host or "")
            task = task or str(trace.task or "")
        agent = agent or live_agent_by_session.get(session_id, "")

        proof_rows.append(
            {
                "session_id": session_id,
                "turn_index": int(row["turn_index"] or 0),
                "tool_name": matched_tool_name,
                "lever": lever if lever != "unknown" else "unattributed",
                "actual_tokens": actual_tokens,
                "naive_tokens": naive_tokens,
                "saved_tokens": saved_tokens,
                "actual_cost_usd": actual_cost_usd,
                "baseline_cost_usd": baseline_cost_usd,
                "saved_cost_usd": saved_cost_usd,
                "tool_calls": int(row["tool_calls"] or 0),
                "created_at": _parse_dt(row["created_at"]).isoformat(),
                "compact_only": compact_only,
                "agent": agent,
                "task": task,
                "ledger_backed": ledger is not None,
                "trace_linked": trace is not None,
                "live_calls_saved": int(live_bucket.get("calls_saved", 0) or 0),
                "live_time_saved_ms": int(live_bucket.get("time_saved_ms", 0) or 0),
                "live_saved_usd": round(float(live_bucket.get("cost_saved_usd", 0.0) or 0.0), 6),
            }
        )

    coverage_gaps: list[dict[str, Any]] = []
    proof_sessions = {row["session_id"] for row in proof_rows}
    live_sessions = {str(event.get("session_id") or "") for event in live_events_window}
    for trace in recent_traces:
        session_id = _trace_run_key(trace)
        host = _host_family(trace.host or trace.agent)
        if host != "copilot":
            continue
        if session_id in proof_sessions or session_id in live_sessions:
            continue
        coverage_gaps.append(
            {
                "session_id": session_id,
                "trace_id": trace.id,
                "agent": trace.agent,
                "task": trace.task,
                "status": trace.status,
                "trace_confidence": trace.trace_confidence,
                "created_at": _trace_created_at(trace).isoformat(),
                "reason": (
                    "Copilot trace/import exists, but no live MCP savings telemetry was captured for this run; "
                    "proof is limited to the imported trace/ledger surface."
                ),
                "missing_surfaces": list(trace.missing_surfaces or []),
            }
        )

    if proof_rows:
        eligible_rows = [row for row in proof_rows if not row["compact_only"]]
        total_naive = sum(int(row["naive_tokens"]) for row in eligible_rows)
        total_actual = sum(int(row["actual_tokens"]) for row in eligible_rows)
        reduction_pct = round((1.0 - (total_actual / total_naive)) * 100.0, 1) if total_naive > 0 else 0.0
        per_lever_totals: defaultdict[str, int] = defaultdict(int)
        for row in eligible_rows:
            per_lever_totals[str(row["lever"])] += int(row["saved_tokens"])
        sorted_levers = dict(sorted(per_lever_totals.items(), key=lambda kv: kv[1], reverse=True))
        saved_usd = round(sum(float(row["saved_cost_usd"]) for row in eligible_rows), 6)
        would_have_cost = round(sum(float(row["baseline_cost_usd"]) for row in eligible_rows), 6)
        actually_cost = round(sum(float(row["actual_cost_usd"]) for row in eligible_rows), 6)
        saved_pct = round(100.0 * saved_usd / would_have_cost, 2) if would_have_cost > 0 else 0.0
        tracked_tool_calls = sum(int(row["tool_calls"]) for row in eligible_rows)
        live_calls_saved = sum(int(row["live_calls_saved"]) for row in proof_rows)
        live_time_saved_ms = sum(int(row["live_time_saved_ms"]) for row in proof_rows)
        live_cost_saved = round(sum(float(row["live_saved_usd"]) for row in proof_rows), 6)

        aggregate_map: dict[tuple[str, str], dict[str, Any]] = {}
        for row in proof_rows:
            key = (str(row["tool_name"]), str(row["lever"]))
            bucket = aggregate_map.setdefault(
                key,
                {
                    "tool_name": row["tool_name"],
                    "lever": row["lever"],
                    "turns": 0,
                    "session_ids": set(),
                    "actual_tokens": 0,
                    "naive_tokens": 0,
                    "saved_tokens": 0,
                    "actual_cost_usd": 0.0,
                    "baseline_cost_usd": 0.0,
                    "saved_cost_usd": 0.0,
                    "live_calls_saved": 0,
                    "live_time_saved_ms": 0,
                    "live_saved_usd": 0.0,
                },
            )
            bucket["turns"] += 1
            bucket["session_ids"].add(str(row["session_id"]))
            bucket["actual_tokens"] += int(row["actual_tokens"])
            bucket["naive_tokens"] += int(row["naive_tokens"])
            bucket["saved_tokens"] += int(row["saved_tokens"])
            bucket["actual_cost_usd"] += float(row["actual_cost_usd"])
            bucket["baseline_cost_usd"] += float(row["baseline_cost_usd"])
            bucket["saved_cost_usd"] += float(row["saved_cost_usd"])
            bucket["live_calls_saved"] += int(row["live_calls_saved"])
            bucket["live_time_saved_ms"] += int(row["live_time_saved_ms"])
            bucket["live_saved_usd"] += float(row["live_saved_usd"])

        tool_aggregates = [
            {
                "tool_name": bucket["tool_name"],
                "lever": bucket["lever"],
                "turns": bucket["turns"],
                "session_count": len(bucket["session_ids"]),
                "actual_tokens": bucket["actual_tokens"],
                "naive_tokens": bucket["naive_tokens"],
                "saved_tokens": bucket["saved_tokens"],
                "actual_cost_usd": round(bucket["actual_cost_usd"], 6),
                "baseline_cost_usd": round(bucket["baseline_cost_usd"], 6),
                "saved_cost_usd": round(bucket["saved_cost_usd"], 6),
                "live_calls_saved": bucket["live_calls_saved"],
                "live_time_saved_ms": bucket["live_time_saved_ms"],
                "live_saved_usd": round(bucket["live_saved_usd"], 6),
            }
            for bucket in aggregate_map.values()
        ]
        tool_aggregates.sort(key=lambda row: (-row["saved_tokens"], row["tool_name"]))

        session_map: dict[str, dict[str, Any]] = {}
        for row in proof_rows:
            session = session_map.setdefault(
                str(row["session_id"]),
                {
                    "session_id": row["session_id"],
                    "agent": row["agent"],
                    "task": row["task"],
                    "saved_tokens": 0,
                    "saved_cost_usd": 0.0,
                    "items": [],
                },
            )
            if not session["agent"] and row["agent"]:
                session["agent"] = row["agent"]
            if not session["task"] and row["task"]:
                session["task"] = row["task"]
            session["saved_tokens"] += int(row["saved_tokens"])
            session["saved_cost_usd"] = round(float(session["saved_cost_usd"]) + float(row["saved_cost_usd"]), 6)
            session["items"].append(
                {
                    "session_id": row["session_id"],
                    "turn_index": row["turn_index"],
                    "tool_name": row["tool_name"],
                    "lever": row["lever"],
                    "actual_tokens": row["actual_tokens"],
                    "naive_tokens": row["naive_tokens"],
                    "saved_tokens": row["saved_tokens"],
                    "actual_cost_usd": row["actual_cost_usd"],
                    "baseline_cost_usd": row["baseline_cost_usd"],
                    "saved_cost_usd": row["saved_cost_usd"],
                    "created_at": row["created_at"],
                }
            )
        session_proof = sorted(session_map.values(), key=lambda row: (-row["saved_tokens"], row["session_id"]))
        for row in session_proof:
            row["items"].sort(key=lambda item: (-int(item["saved_tokens"]), int(item["turn_index"])))

        dominant_run = session_proof[0] if session_proof else None
        dominant_item = (
            max(proof_rows, key=lambda row: (int(row["saved_tokens"]), row["created_at"])) if proof_rows else None
        )
        total_saved_proof_tokens = sum(int(row["saved_tokens"]) for row in proof_rows) or 1
        dominant_run_share = (
            round((int(dominant_run["saved_tokens"]) / total_saved_proof_tokens) * 100.0, 1) if dominant_run else 0.0
        )
        dominant_item_share = (
            round((int(dominant_item["saved_tokens"]) / total_saved_proof_tokens) * 100.0, 1) if dominant_item else 0.0
        )
        compact_output_row_count = sum(1 for row in proof_rows if row["compact_only"])
        compact_output_saved_tokens = sum(int(row["saved_tokens"]) for row in proof_rows if row["compact_only"])
        if compact_output_row_count > 0:
            headline_explanation = (
                "This headline excludes compact-tool-output rows such as search_read naive-vs-compacted comparisons. "
                "Those rows remain in the proof tables below as tool-output compression evidence, "
                "but they do not count toward top-line token or cost savings. "
                "The headline is still a proof-oriented estimate, not audited provider billing."
            )
            warning = (
                f"{compact_output_row_count} compact-tool-output proof row(s) were excluded from the headline totals. "
                "One proof row dominates the estimated saved-token total. Inspect the leading session/item rows below before trusting the aggregate."
            )
        else:
            headline_explanation = (
                "These top-line totals come from headline-eligible context-budget proof rows and exclude live-estimate-only overlays. "
                "They are proof-oriented estimates, not audited provider billing."
            )
            warning = "One proof row dominates the estimated saved-token total. Inspect the leading session/item rows below before trusting the aggregate."

        top_sources = [
            {
                "lever": row["lever"],
                "tool_name": row["tool_name"],
                "calls_saved": row["live_calls_saved"],
                "tokens_saved": row["saved_tokens"],
                "cost_saved_usd": row["saved_cost_usd"],
                "time_saved_ms": row["live_time_saved_ms"],
            }
            for row in tool_aggregates[:10]
        ]

        return {
            "window_days": window_days,
            "total_naive_tokens": total_naive,
            "total_actual_tokens": total_actual,
            "reduction_pct": reduction_pct,
            "per_lever": sorted_levers,
            "by_day": list(by_day_seed.values()),
            "saved_usd": saved_usd,
            "saved_pct": saved_pct,
            "would_have_cost_usd": would_have_cost,
            "actually_cost_usd": actually_cost,
            "tracked_actual_cost_usd": actually_cost,
            "tracked_baseline_cost_usd": would_have_cost,
            "tracked_saved_cost_usd": saved_usd,
            "total_calls": cost_summary["total_calls"],
            "live_calls_saved": live_calls_saved,
            "live_time_saved_ms": live_time_saved_ms,
            "live_saved_usd": live_cost_saved,
            "ops_saved_usd": ops_saved_usd,
            "carry_usd": window_summary.carry_usd,
            "carry_tokens": window_summary.carry_tokens,
            "read_saved_usd": round(window_summary.read_saved_usd, 6),
            "total_saved_usd": round(window_summary.total_saved_usd, 6),
            "ledger_tokens_saved": window_summary.tokens_saved,
            "ledger_calls_saved": window_summary.calls_saved,
            "routing_saved_usd": window_summary.routing_usd,
            "top_sources": top_sources,
            "cost_only_sources": [],
            "latest_benchmark": _latest_savings_benchmark(root),
            "tracked_tool_calls": tracked_tool_calls,
            "cost_basis": "context_budget",
            "tool_aggregates": tool_aggregates,
            "session_proof": session_proof,
            "coverage_gaps": coverage_gaps,
            "verification": {
                "data_root": str(root),
                "headline_kind": "tracked_proof_reduction",
                "headline_explanation": headline_explanation,
                "tracked_row_count": len(proof_rows),
                "tracked_run_count": len({row["session_id"] for row in proof_rows}),
                "trace_linked_run_count": len({row["session_id"] for row in proof_rows if row["trace_linked"]}),
                "ledger_backed_run_count": len({row["session_id"] for row in proof_rows if row["ledger_backed"]}),
                "live_event_count": len(live_events_window),
                "coverage_gap_count": len(coverage_gaps),
                "compact_output_row_count": compact_output_row_count,
                "compact_output_saved_tokens": compact_output_saved_tokens,
                "dominant_run": (
                    {
                        "session_id": dominant_run["session_id"],
                        "agent": dominant_run["agent"],
                        "task": dominant_run["task"],
                        "saved_tokens": dominant_run["saved_tokens"],
                        "saved_cost_usd": dominant_run["saved_cost_usd"],
                    }
                    if dominant_run
                    else None
                ),
                "dominant_item": (
                    {
                        "session_id": dominant_item["session_id"],
                        "turn_index": dominant_item["turn_index"],
                        "tool_name": dominant_item["tool_name"],
                        "lever": dominant_item["lever"],
                        "actual_tokens": dominant_item["actual_tokens"],
                        "naive_tokens": dominant_item["naive_tokens"],
                        "saved_tokens": dominant_item["saved_tokens"],
                        "created_at": dominant_item["created_at"],
                    }
                    if dominant_item
                    else None
                ),
                "dominant_run_share_pct": dominant_run_share,
                "dominant_item_share_pct": dominant_item_share,
                "warning": warning,
            },
        }

    return {
        "window_days": window_days,
        "total_naive_tokens": total_naive,
        "total_actual_tokens": total_actual,
        "reduction_pct": reduction_pct,
        "per_lever": sorted_levers,
        "by_day": list(by_day_seed.values()),
        "saved_usd": saved_usd,
        "saved_pct": saved_pct,
        "would_have_cost_usd": would_have_cost,
        "actually_cost_usd": round(actually_cost, 6),
        "total_calls": cost_summary["total_calls"],
        "live_calls_saved": live_calls_saved,
        "live_time_saved_ms": live_time_saved_ms,
        "live_saved_usd": round(live_cost_saved, 6),
        "ops_saved_usd": ops_saved_usd,
        "carry_usd": window_summary.carry_usd,
        "carry_tokens": window_summary.carry_tokens,
        "read_saved_usd": round(window_summary.read_saved_usd, 6),
        "total_saved_usd": round(window_summary.total_saved_usd, 6),
        "ledger_tokens_saved": window_summary.tokens_saved,
        "ledger_calls_saved": window_summary.calls_saved,
        "routing_saved_usd": window_summary.routing_usd,
        "top_sources": top_sources[:10],
        "cost_only_sources": cost_only_sources[:10],
        "latest_benchmark": _latest_savings_benchmark(root),
        "tracked_tool_calls": 0,
        "cost_basis": "session_ledger",
        "tool_aggregates": [],
        "session_proof": [],
        "coverage_gaps": coverage_gaps,
        "verification": {},
    }


def _optimization_runtime_coverage() -> list[dict[str, Any]]:
    return [
        {
            "host": "claude",
            "mode": "runtime",
            "automatic_at_start": True,
            "automatic_mid_session": True,
            "advisory_only": False,
            "surfaces": [
                "SessionStart hook",
                "PostToolUse telemetry hook",
                "Host instructions",
            ],
            "notes": (
                "Live SessionStart guidance is emitted from the Claude plugin, and PostToolUse telemetry now "
                "emits one-shot nudges for no-edit drift, session-quality degradation, and loop-rescue intervention."
            ),
        },
        {
            "host": "codex",
            "mode": "runtime",
            "automatic_at_start": True,
            "automatic_mid_session": True,
            "advisory_only": False,
            "surfaces": [
                "SessionStart update hook",
                "Savings reporter hook",
                "Host instructions",
            ],
            "notes": (
                "Codex now gets live SessionStart budget guidance plus the same PostToolUse nudges for no-edit drift, "
                "session-quality degradation, and loop-rescue intervention."
            ),
        },
        {
            "host": "copilot",
            "mode": "task_wrapper",
            "automatic_at_start": True,
            "automatic_mid_session": False,
            "advisory_only": False,
            "surfaces": [
                "Copilot preflight task",
                "lc CLI shell task",
                "Copilot instructions",
                "LemonCrow agent",
            ],
            "notes": (
                "Copilot now has a task-driven preflight that runs lc CLI checks before chat work begins. "
                "Native Copilot chat remains instruction-backed if that task path is skipped."
            ),
        },
        {
            "host": "antigravity",
            "mode": "wrapper",
            "automatic_at_start": True,
            "automatic_mid_session": False,
            "advisory_only": False,
            "surfaces": [
                "agy companion CLI",
                "Antigravity MCP config",
                "Antigravity AGENTS surface",
            ],
            "notes": (
                "Antigravity can now load LemonCrow through its MCP config while agy provides a terminal companion flow. "
                "The host remains advisory because there is no first-class hook stream."
            ),
        },
        {
            "host": "opencode",
            "mode": "wrapper",
            "automatic_at_start": True,
            "automatic_mid_session": False,
            "advisory_only": False,
            "surfaces": [
                "OpenCode agent instruction",
            ],
            "notes": ("Agent-profile-only sessions still rely on installed instructions."),
        },
    ]


def _optimization_lever_tokens(
    per_lever: dict[str, int],
    *,
    exact: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> int:
    total = 0
    for lever, tokens in per_lever.items():
        if lever in exact or any(lever.startswith(prefix) for prefix in prefixes):
            total += int(tokens or 0)
    return total


def _optimization_lever_cost(live_events: list[dict[str, Any]], *, lever: str, window_days: int) -> float:
    cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(window_days))
    total = 0.0
    for event in live_events:
        normalized = _normalize_lever(str(event.get("lever") or event.get("kind") or ""))
        if normalized != lever:
            continue
        at = _live_event_datetime(event)
        if at is not None and at < cutoff:
            continue
        total += _coerce_float(event.get("cost_saved_usd") or 0.0)
    return round(total, 6)


def _optimization_lever_examples(
    top_sources: list[dict[str, Any]],
    *,
    exact: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> list[str]:
    examples: list[str] = []
    for source in top_sources:
        lever = str(source.get("lever") or "")
        if lever in exact or any(lever.startswith(prefix) for prefix in prefixes):
            tool_name = str(source.get("tool_name") or lever)
            if tool_name not in examples:
                examples.append(tool_name)
    return examples[:3]


def _implemented_optimization_catalog(
    savings_payload: dict[str, Any], live_events: list[dict[str, Any]], *, window_days: int
) -> list[dict[str, Any]]:
    from lemoncrow.pro.capabilities.session_optimizer import SUPPORTED_OPTIMIZER_HOSTS

    supported_hosts = list(SUPPORTED_OPTIMIZER_HOSTS)
    owned_runtime_hosts = ["lemoncode", "codex", "claude", "native"]
    per_lever = {
        str(key): int(value or 0)
        for key, value in (savings_payload.get("per_lever") or {}).items()
        if isinstance(key, str)
    }
    top_sources = [item for item in (savings_payload.get("top_sources") or []) if isinstance(item, dict)]

    catalog = [
        {
            "id": "session_budget_optimizer",
            "title": "Session budget optimizer",
            "category": "session_control",
            "automation": "Automatic on Claude/Codex and wrapper/task-enforced at start for Copilot/Gemini/OpenCode",
            "status": "active",
            "observed_tokens_saved": 0,
            "applies_to": supported_hosts,
            "notes": (
                "Injects smallest-plan, bounded-context, and delivery-or-stop guardrails. "
                "Also powers the trace-based optimization recommendations shown below."
            ),
            "examples": ["SessionStart guidance", "No-edit 10m nudge", "lc optimize"],
        },
        {
            "id": "optimization_decision_trace",
            "title": "Redacted optimization decision trace",
            "category": "measurement",
            "automation": "Automatic in shadow and enforce modes",
            "status": "active",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Records proposed versus actual policy, calls, tokens, cost, verification, "
                "and accepted outcomes without prompts, source, arguments, commands, or diffs."
            ),
            "examples": ["lc optimize decisions --json", "--optimization-mode off|shadow|enforce"],
        },
        {
            "id": "output_governor_v2",
            "title": "Output Governor V2",
            "category": "output_control",
            "automation": "Shadow by default; opt-in enforcement",
            "status": "implemented_shadow",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Suppresses retained execution narration, requires real truncation before extension, "
                "and finalizes verified mutations without another provider call."
            ),
            "examples": ["tool-only mutation phase", "deterministic verification receipt"],
        },
        {
            "id": "adaptive_cache_economics_v2",
            "title": "Adaptive cache economics V2",
            "category": "cache",
            "automation": "Shadow by default; explicit cache policy always wins",
            "status": "implemented_shadow",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Learns reuse gaps, chooses provider-aware TTLs and stable lanes, filters volatile "
                "breakpoints, and applies rewrite economics to compaction."
            ),
            "examples": ["--cache-policy auto|5m|1h|off", "route/cache hysteresis"],
        },
        {
            "id": "calibrated_phase_routing_v2",
            "title": "Calibrated phase routing V2",
            "category": "model_selection",
            "automation": "Shadow until the outcome sample floor; repair escalates immediately",
            "status": "implemented_shadow",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Minimizes direct cost plus calibrated failure escalation and cache-break cost, "
                "with a warm-lane hysteresis guard."
            ),
            "examples": ["20-sample default floor", "pinned-model rollback"],
        },
        {
            "id": "local_retrieval_firewall_v2",
            "title": "Bounded local retrieval firewall",
            "category": "context_pruning",
            "automation": "Auto-gated; prompt-visible only in enforce mode",
            "status": "implemented_shadow",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Returns bounded exact source-hashed spans. Explicit-file tasks skip before scanning; "
                "the optional planner accepts local model identifiers only."
            ),
            "examples": ["--local-retrieval auto|force|off", "--local-retrieval-model ollama/..."],
        },
        {
            "id": "hybrid_mcp_exposure",
            "title": "Hybrid MCP and tool exposure",
            "category": "tool_supervision",
            "automation": "Automatic eager core with deterministic focused fallback",
            "status": "adapted_complete",
            "observed_tokens_saved": 0,
            "applies_to": [*supported_hosts, "lemoncode", "native"],
            "notes": (
                "Keeps ordinary coding tools eager and never requires a search-first broker call; "
                "managed LemonCode exposes no redundant outer MCP schemas."
            ),
            "examples": ["LEMONCROW_MCP_TOOL_PROFILE=full", "--mcp-schema-mode auto"],
        },
        {
            "id": "verified_evidence_reuse_v2",
            "title": "Verified cross-session evidence reuse",
            "category": "context_reuse",
            "automation": "Collected in shadow; prompt-visible only in enforce mode",
            "status": "implemented_shadow",
            "observed_tokens_saved": 0,
            "applies_to": owned_runtime_hosts,
            "notes": (
                "Reuses deterministic source-hashed retrieval only after a matching workspace, "
                "dependency, tool-version, and verification receipt check."
            ),
            "examples": ["read-only revalidation receipt", "post-edit project-command receipt"],
        },
        {
            "id": "search_read",
            "title": "Search + read compaction",
            "category": "tool_supervision",
            "automation": "Automatic when smart search/read tools are used",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("search_read",)),
            "applies_to": supported_hosts,
            "notes": "Targets grep/read workflows into bounded chunks instead of feeding whole-file context.",
            "examples": _optimization_lever_examples(top_sources, exact=("search_read",)),
        },
        {
            "id": "batch_edit",
            "title": "Batch edit",
            "category": "tool_supervision",
            "automation": "Automatic when batch edit paths are used",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("batch_edit",)),
            "applies_to": supported_hosts,
            "notes": "Combines related edits into a single operation to reduce repeated edit context and tool chatter.",
            "examples": _optimization_lever_examples(top_sources, exact=("batch_edit",)),
        },
        {
            "id": "compact_tool_output",
            "title": "Tool-output compaction",
            "category": "output_control",
            "automation": "Automatic in LemonCrow MCP paths, optional in Claude host hook",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(
                per_lever,
                exact=("compact_lifecycle",),
                prefixes=("compact_tool_output:",),
            ),
            "applies_to": supported_hosts,
            "notes": "Reduces verbose tool output before it becomes future prompt context.",
            "examples": _optimization_lever_examples(
                top_sources,
                exact=("compact_lifecycle",),
                prefixes=("compact_tool_output:",),
            ),
        },
        {
            "id": "session_compaction",
            "title": "Session compaction",
            "category": "context_lifecycle",
            "automation": "Advisory - compact tool fires on utilisation >= 80%",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("session_compaction",)),
            "observed_cost_saved_usd": _optimization_lever_cost(
                live_events, lever="session_compaction", window_days=window_days
            ),
            "applies_to": supported_hosts,
            "notes": "Compresses the conversation into a smaller carry-forward state at task boundaries.",
            "examples": _optimization_lever_examples(top_sources, exact=("session_compaction",)),
        },
        {
            "id": "cached_read",
            "title": "Cached read reuse",
            "category": "cache",
            "automation": "Automatic when cached token paths are available",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("cached_read",)),
            "applies_to": supported_hosts,
            "notes": "Prefers cached or discounted context where the host/provider exposes it.",
            "examples": _optimization_lever_examples(top_sources, exact=("cached_read",)),
        },
        {
            "id": "scoped_recall",
            "title": "Scoped recall",
            "category": "memory",
            "automation": "Automatic when memory recall paths are used",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("scoped_recall",)),
            "applies_to": supported_hosts,
            "notes": "Pulls narrower memory slices instead of replaying broad historical context.",
            "examples": _optimization_lever_examples(top_sources, exact=("scoped_recall",)),
        },
        {
            "id": "playbook_inject",
            "title": "Playbook injection",
            "category": "context_reuse",
            "automation": "Automatic when matching playbook blocks are selected",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("playbook_inject",)),
            "applies_to": supported_hosts,
            "notes": "Reuses prior solved procedures instead of re-deriving them from scratch.",
            "examples": _optimization_lever_examples(top_sources, exact=("playbook_inject",)),
        },
        {
            "id": "ast_truncation",
            "title": "AST-aware truncation",
            "category": "context_pruning",
            "automation": "Automatic when structured truncation paths are used",
            "status": "active",
            "observed_tokens_saved": _optimization_lever_tokens(per_lever, exact=("ast_truncation",)),
            "applies_to": supported_hosts,
            "notes": "Keeps syntax-relevant structure while trimming low-value surrounding text.",
            "examples": _optimization_lever_examples(top_sources, exact=("ast_truncation",)),
        },
        {
            "id": "model_routing",
            "title": "Model routing (tier downgrade)",
            "category": "model_selection",
            "automation": "Advisory - route tool fires before every tool call",
            "status": "active",
            "observed_tokens_saved": 0,
            "observed_cost_saved_usd": _optimization_lever_cost(
                live_events, lever="model_routing", window_days=window_days
            ),
            "applies_to": supported_hosts,
            "notes": "Recommends a cheaper model tier than opus for routine tool calls and records the estimated dollar delta.",
            "examples": _optimization_lever_examples(top_sources, exact=("model_routing",)),
        },
        {
            "id": "loop_detection",
            "title": "Loop detection + rescue",
            "category": "control_plane",
            "automation": "Automatic detection, partial host-native rescue surfacing",
            "status": "active",
            "observed_tokens_saved": 0,
            "applies_to": supported_hosts,
            "notes": (
                "Detects repeated search/read or retry loops and redirects the agent toward rescue instead of repeated spend. "
                "This is a qualitative safeguard rather than a direct savings counter today."
            ),
            "examples": ["search_read_loop", "repeated bash failures"],
        },
    ]
    for item in catalog:
        observed_tokens = item.get("observed_tokens_saved")
        observed_cost = _coerce_float(item.get("observed_cost_saved_usd") or 0.0)
        if ((isinstance(observed_tokens, int) and observed_tokens > 0) or observed_cost > 0) and item[
            "status"
        ] == "active":
            item["status"] = "active_observed"
    return catalog


def _optimization_implementation_gaps() -> list[dict[str, Any]]:
    from lemoncrow.pro.capabilities.session_optimizer import SUPPORTED_OPTIMIZER_HOSTS

    supported_hosts = list(SUPPORTED_OPTIMIZER_HOSTS)
    return [
        {
            "id": "owned-runtime-controlled-ab",
            "priority": "high",
            "title": "Run controlled quality/cost acceptance gates for the V2 savings runtime",
            "hosts": ["lemoncode", "codex", "claude", "native"],
            "notes": (
                "All six runtime implementations are locally verified, but routing, output, cache, "
                "retrieval, and evidence reuse remain measurement-pending until frozen-corpus A/B "
                "reports show lower cost per accepted change without a quality regression."
            ),
        },
        {
            "id": "local-retrieval-recall-corpus",
            "priority": "high",
            "title": "Measure local retrieval recall, frontier tokens, and latency",
            "hosts": ["lemoncode", "codex", "claude", "native"],
            "notes": (
                "Compare deterministic-only and optional-local-planner packets against normal "
                "frontier retrieval; keep the feature shadow-only if exact-target recall regresses."
            ),
        },
        {
            "id": "wrapper-adoption-visibility",
            "priority": "medium",
            "title": "Distinguish wrapper-launched sessions from instruction-only sessions",
            "hosts": supported_hosts,
            "notes": (
                "These hosts now have live start-time wrapper/task entrypoints, but the dashboard still does not label which recorded runs actually used those enforced paths."
            ),
        },
        {
            "id": "mid-session-nudges-non-hook-hosts",
            "priority": "medium",
            "title": "Extend mid-session quality and loop nudges beyond Claude and Codex",
            "hosts": supported_hosts,
            "notes": (
                "Claude and Codex now emit no-edit, quality-drop, and loop-rescue nudges through runtime telemetry. "
                "Copilot, Gemini, and OpenCode still only get start-time automation today."
            ),
        },
        {
            "id": "host-native-output-compaction-rollout",
            "priority": "medium",
            "title": "Roll out host-native tool-output compaction more broadly",
            "hosts": supported_hosts,
            "notes": (
                "Tool-output compaction exists in LemonCrow's supervised paths and optional Claude hook surfaces, "
                "but the host-native rollout is not uniform yet."
            ),
        },
        {
            "id": "impact-validation-windows",
            "priority": "medium",
            "title": "Add before/after impact validation for optimization changes",
            "hosts": supported_hosts,
            "notes": (
                "LemonCrow now audits static context surfaces and recent trace quality, but it still does not compare pre-change and post-change windows to prove whether an optimization improved tokens, cost, or cache leverage."
            ),
        },
        {
            "id": "delta-read-and-structure-map-telemetry",
            "priority": "medium",
            "title": "Measure smart re-read savings at the file-structure level",
            "hosts": supported_hosts,
            "notes": (
                "LemonCrow has search/read compaction and AST-aware truncation, but it does not yet expose repeat-read telemetry comparable to delta-read and structure-map measurement."
            ),
        },
    ]


def _candidate_potential_breakdown(advisor: OptimizationResult, candidate: Candidate) -> dict[str, float]:
    """Read/Carry/Output/Routing/Total breakdown for one advisor candidate.

    Uses the advisor's clamped `weekly_savings_usd` as Total for the
    recommended candidate (matching the CLI's rendering); every other
    candidate's Total is its own raw delta against the baseline weekly cost.
    """
    from lemoncrow.pro.capabilities.optimization import potential_savings_breakdown

    if candidate.id == advisor.recommended_candidate_id:
        total_saved_usd = advisor.weekly_savings_usd
    else:
        total_saved_usd = advisor.baseline_weekly_cost_usd - candidate.weekly_cost_usd
    return potential_savings_breakdown(candidate, advisor.baseline_weekly_cost_usd, total_saved_usd)


def _optimizations_summary_payload(root: Path, store: StoreBundle, *, window_days: int) -> dict[str, Any]:
    from lemoncrow.pro.capabilities.optimization import (
        load_current_policy,
        load_history,
        optimize_from_traces,
    )
    from lemoncrow.pro.capabilities.session_optimizer import build_trace_optimization_report

    savings = _savings_summary_payload(root, window_days=window_days)
    traces = store.history.list_traces(limit=5000)
    recent_traces = _recent_traces(store, window_days=window_days)
    live_events = _iter_live_savings_events(root)
    recommendations = build_trace_optimization_report(traces, days=window_days)
    advisor = optimize_from_traces(
        traces,
        current_policy=load_current_policy(root),
        days=window_days,
    )
    advisor_dict = advisor.to_dict()
    for candidate_payload, candidate in zip(advisor_dict["candidates"], advisor.candidates, strict=True):
        candidate_payload.update(_candidate_potential_breakdown(advisor, candidate))
    advisor_history = load_history(root, limit=6)
    from lemoncrow.core.foundation.paths import resolve_workspace_root

    project_root_candidate = resolve_workspace_root(root)
    if not ((project_root_candidate / "src").exists() or (project_root_candidate / "AGENTS.md").exists()):
        project_root_candidate = Path.cwd()
    from lemoncrow.pro.capabilities.optimization import (
        build_context_audit,
        build_session_quality_summary,
    )

    context_audit = build_context_audit(
        project_root=project_root_candidate,
        blocks_dir=getattr(store.knowledge, "blocks_dir", None),
        rubrics_dir=getattr(store.knowledge, "rubrics_dir", None),
    )
    quality_score = build_session_quality_summary(traces, window_days=window_days, context_audit=context_audit)
    runtime_coverage = _optimization_runtime_coverage()
    implemented_levers = _implemented_optimization_catalog(savings, live_events, window_days=window_days)
    auto_optimizations = _build_auto_optimizations(savings, live_events, window_days=window_days)
    impact_validation = _build_impact_validation(store, recent_traces, window_days=window_days)
    reread_telemetry = _build_reread_telemetry(root, window_days=window_days)
    model_routing_simulation = _build_model_routing_simulation(
        recent_traces, window_days=window_days, live_events=live_events
    )
    compact_session_history = _build_compact_session_history(live_events, window_days=window_days)
    from lemoncrow.pro.capabilities.optimization.runtime_decisions import summarize_runtime_decisions

    runtime_decisions = summarize_runtime_decisions(root, days=window_days)
    runtime_decisions.pop("trace_path", None)

    automatic_hosts = sum(1 for item in runtime_coverage if item["automatic_at_start"])
    advisory_only_hosts = sum(1 for item in runtime_coverage if item["advisory_only"])
    observed_levers = sum(
        1
        for item in implemented_levers
        if (
            (isinstance(item.get("observed_tokens_saved"), int) and item["observed_tokens_saved"] > 0)
            or _coerce_float(item.get("observed_cost_saved_usd") or 0.0) > 0
        )
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "automatic_hosts": automatic_hosts,
        "advisory_only_hosts": advisory_only_hosts,
        "observed_levers": observed_levers,
        "runtime_coverage": runtime_coverage,
        "implemented_levers": implemented_levers,
        "implementation_gaps": _optimization_implementation_gaps(),
        "advisor": advisor_dict,
        "advisor_history": advisor_history,
        "recommendations": recommendations,
        "context_audit": context_audit,
        "quality_score": quality_score,
        "auto_optimizations": auto_optimizations,
        "impact_validation": impact_validation,
        "reread_telemetry": reread_telemetry,
        "model_routing_simulation": model_routing_simulation,
        "compact_session_history": compact_session_history,
        "runtime_decisions": runtime_decisions,
        "savings": savings,
        "data_sources": [
            {
                "id": "runtime_coverage",
                "label": "Host automation coverage",
                "detail": "Static audit of current hook-backed and instruction-backed optimizer surfaces across the five supported hosts.",
            },
            {
                "id": "trace_recommendations",
                "label": "Trace-based opportunities",
                "detail": "Recommendations generated from stored traces via the session optimizer heuristics.",
            },
            {
                "id": "context_audit",
                "label": "Static context audit",
                "detail": "Prompt-relevant LemonCrow surfaces scanned for approximate token weight and optimization headroom.",
            },
            {
                "id": "quality_score",
                "label": "Recent trace quality scoring",
                "detail": "Multi-signal quality scoring built from recent LemonCrow traces using context fill, delivery, outcome, cache, and redundancy signals.",
            },
            {
                "id": "savings_telemetry",
                "label": "Savings telemetry",
                "detail": "Observed token savings pulled from cost history and live savings event streams.",
            },
            {
                "id": "runtime_decisions",
                "label": "Owned-runtime decision trace",
                "detail": "Redacted proposed-versus-actual policy, provider/tool calls, tokens, cost, verification, and accepted outcomes.",
            },
        ],
    }


# --------------------------------------------------------------------------- #
# App Factory                                                                 #
# --------------------------------------------------------------------------- #


def create_app(store_root: str | Path | None = None, store: StoreBundle | None = None) -> Any:
    """Construct the FastAPI instance."""
    from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware

    def _req_int(value: Any, field: str) -> int:
        """Coerce a request value to int, answering 400 (not 500) on bad input."""
        try:
            return int(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{field} must be an integer") from None

    def _req_dt(value: str, field: str) -> datetime:
        """Parse an ISO-8601 request value, answering 400 (not 500) on bad input."""
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{field} must be an ISO-8601 datetime") from None

    app = FastAPI(
        title="LemonCrow",
        description="Agent Reasoning Runtime API",
        version="0.1.0",
        docs_url="/docs",
    )

    # CORS for the local web UI: loopback origins only (any port). A wildcard
    # origin with credentials would let any website a browser visits drive
    # this API with the user's cookies/credentials.
    app.add_middleware(GZipMiddleware, minimum_size=1_000)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Late load store
    from lemoncrow.infra.storage.factory import create_store

    store_path = Path(store_root or cfg.lemoncrow_root)
    runtime_store = store or create_store(store_path)
    store = runtime_store
    _store_init_lock = threading.Lock()
    _store_initialized = False

    # Daemon-owned savings-aggregate reconciliation (code_warm pattern): keep
    # the persisted day-bucketed savings aggregate folded up to date so MCP
    # session processes and the statusline never scan sessions/** themselves.
    from lemoncrow.core.service.savings_reconcile import start_savings_reconciler

    start_savings_reconciler(store_path)

    def get_store() -> StoreBundle:
        nonlocal _store_initialized
        if not _store_initialized:
            with _store_init_lock:
                if not _store_initialized:  # double-checked locking
                    runtime_store.init()
                    _store_initialized = True
        return runtime_store

    # ------------------------------------------------------------------ #
    # Metadata & Health                                                   #
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}

    @app.get("/ready", tags=["system"])
    def ready_check() -> dict[str, str]:
        return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}

    @app.get("/config", tags=["system"], dependencies=[Depends(verify_api_key)])
    def get_config() -> dict[str, Any]:
        return cfg.as_dict()

    @app.post(
        "/v1/router/claude-code/evaluate",
        tags=["router"],
        dependencies=[Depends(verify_api_key)],
    )
    def evaluate_claude_code_router(payload: HostRouterEvaluateRequest) -> dict[str, Any]:
        return evaluate_host_router_request(
            root=store.knowledge.root,
            path=payload.path,
            model=payload.model,
            messages=payload.messages,
            system=payload.system,
            session_state=payload.session_state,
            mode=payload.mode,
        )

    @app.get("/overview", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_overview(days: int = Query(30)) -> dict[str, Any]:
        """Compatibility: GET /overview -> basic summary stats."""
        from lemoncrow.core.foundation.metrics import summarize
        from lemoncrow.infra.runtime.cost_tracker import CostTracker

        root = Path(cfg.lemoncrow_root)
        store = get_store()
        since = datetime.now(UTC) - timedelta(days=_clamp_days(days))
        summary = summarize(store, since=since)

        # Calculate tokens/cost from traces in database within the time window
        total_raw_tokens = 0
        total_cost_usd = 0.0

        with sqlite3.connect(store.history.db_path) as conn:
            sql = """
                SELECT 
                    SUM(json_extract(payload, '$.input_tokens')),
                    SUM(json_extract(payload, '$.output_tokens')),
                    SUM(json_extract(payload, '$.cached_input_tokens')),
                    SUM(json_extract(payload, '$.thinking_tokens'))
                FROM traces
                WHERE created_at >= ?
            """
            row = conn.execute(sql, (since.isoformat(),)).fetchone()
            if row:
                inp, out, cr, th = row
                total_raw_tokens = (inp or 0) + (out or 0) + (cr or 0) + (th or 0)

            conn.row_factory = sqlite3.Row
            for trace_row in conn.execute("SELECT payload FROM traces WHERE created_at >= ?", (since.isoformat(),)):
                try:
                    payload = json.loads(trace_row["payload"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                total_cost_usd += _trace_cost_from_payload(payload)

        tracker = CostTracker(root)
        savings = tracker.total_savings(since=since)

        return {
            "total_traces": summary.traces_total,
            "total_blocks": summary.blocks_active,
            "total_rubrics": summary.rubrics_total,
            "total_clusters": 0,
            "total_raw_tokens_estimate": total_raw_tokens,
            "estimated_total_cost_usd": max(total_cost_usd, savings["actually_cost_usd"]),
            "estimated_saved_cost_usd": savings["saved_usd"],
            "average_compression_ratio": 1.0,
            "is_estimate": False,
        }

    # ------------------------------------------------------------------ #
    # Traces                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/traces", tags=["traces"], dependencies=[Depends(verify_api_key)])
    def list_traces(
        domain: str | None = Query(None),
        status: str | None = Query(None),
        agent: str | None = Query(None),
        host: str | None = Query(None),
        workspace: str | None = Query(None, description="Filter by exact workspace_path"),
        query: str | None = Query(None),
        days: int | None = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        store = get_store()
        since = datetime.now(UTC) - timedelta(days=_clamp_days(days)) if days else None
        traces = _list_traces_filtered(
            store,
            domain=domain,
            status=status,
            agent=agent,
            host=host,
            workspace=workspace,
            query=query,
            since=since,
            limit=limit,
            offset=offset,
        )
        # Fetch global metrics for the current domain/agent/host filters
        metrics = store.history.get_traces_metrics(domain=domain, agent=agent, host=host, since=since)
        # Distinct workspace values across full history (not just this page)
        # so the frontend's workspace filter dropdown is complete.
        metrics["workspaces"] = _distinct_workspaces(store, domain=domain, agent=agent, host=host, since=since)

        return {"items": [to_jsonable(t) for t in traces], "metrics": metrics}

    @app.get("/v1/traces/{trace_id}", tags=["traces"], dependencies=[Depends(verify_api_key)])
    def get_trace(trace_id: str) -> dict[str, Any]:
        trace = get_store().history.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
        return to_jsonable(trace)

    # ------------------------------------------------------------------ #
    # Context & Rescue                                                   #
    # ------------------------------------------------------------------ #

    def _runtime() -> Any:
        from lemoncrow.gateway.adapters.runtime import ContextRuntime

        return ContextRuntime(root=Path(cfg.lemoncrow_root))

    @app.post(
        "/v1/reasoning/context",
        tags=["context"],
        dependencies=[Depends(verify_api_key)],
        response_model=ContextResponse,
    )
    def reasoning_context(payload: ContextRequest = Body(...)) -> ContextResponse:  # noqa: B008
        result = _runtime().get_context(
            task=payload.task,
            domain=payload.domain,
            files=payload.files,
            tools=payload.tools,
            errors=payload.errors,
            max_blocks=payload.max_blocks,
            token_budget=payload.token_budget,
            dedup=payload.dedup,
            include_telemetry=payload.include_telemetry,
            agent_id=payload.agent_id,
            recall=payload.recall,
        )
        if isinstance(result, dict):
            return ContextResponse.model_validate(result)
        return ContextResponse(context=result)

    @app.post("/v1/reasoning/rescue", tags=["context"], dependencies=[Depends(verify_api_key)])
    def rescue_failure(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008
        task = str(payload.get("task") or "")
        error = str(payload.get("error") or "")
        if not task or not error:
            raise HTTPException(status_code=400, detail="task and error are required")
        result = _runtime().rescue_failure(
            task=task,
            error=error,
            files=list(payload.get("files") or []),
            recent_actions=list(payload.get("recent_actions") or []),
            domain=payload.get("domain"),
        )
        response = to_jsonable(result)
        with contextlib.suppress(Exception):
            analysis = _runtime().core_runtime.analyze_failure_for_error(
                error,
                domain=payload.get("domain"),
            )
            if analysis:
                response["analysis"] = analysis
        return response

    @app.post("/v1/rubrics/run", tags=["rubrics"], dependencies=[Depends(verify_api_key)])
    def run_rubric_gate(payload: dict[str, Any]) -> dict[str, Any]:
        from lemoncrow.core.foundation.rubric_gate import run_rubric

        rubric_id = str(payload.get("rubric_id") or "")
        if not rubric_id:
            raise HTTPException(status_code=400, detail="rubric_id is required")
        rubric = get_store().knowledge.get_rubric(rubric_id)
        if rubric is None:
            raise HTTPException(status_code=404, detail=f"Rubric not found: {rubric_id}")
        checks = payload.get("checks") or {}
        if not isinstance(checks, dict):
            raise HTTPException(status_code=400, detail="checks must be an object")
        return to_jsonable(run_rubric(rubric, checks))

    def _redact_trace_json(value: Any) -> Any:
        from lemoncrow.core.foundation.redaction import redact

        if isinstance(value, str):
            return redact(value)
        if isinstance(value, list):
            return [_redact_trace_json(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _redact_trace_json(item) for key, item in value.items()}
        return value

    def _coerce_trace_validation_passed(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"pass", "passed", "success", "successful", "ok", "true"}:
                return True
            if lowered in {"fail", "failed", "failure", "error", "errored", "false"}:
                return False
        return False

    def _normalize_trace_tool_calls(items: list[Any]) -> list[dict[str, Any]]:
        from lemoncrow.core.foundation.redaction import redact

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                normalized.append({"name": redact(item), "args_hash": "", "count": 1})
                continue
            if isinstance(item, dict):
                raw_count = item.get("count") or 1
                with contextlib.suppress(TypeError, ValueError):
                    raw_count = int(raw_count)
                if not isinstance(raw_count, int):
                    raw_count = 1
                tool_call = {
                    "name": redact(str(item.get("name") or item.get("tool") or "unknown")),
                    "args_hash": redact(str(item.get("args_hash") or "")),
                    "count": raw_count,
                }
                if "args" in item:
                    tool_call["args"] = _redact_trace_json(item["args"])
                if isinstance(item.get("result_summary"), str):
                    tool_call["result_summary"] = redact(item["result_summary"])
                normalized.append(tool_call)
                continue
            normalized.append({"name": redact(str(item)), "args_hash": "", "count": 1})
        return normalized

    def _normalize_trace_validation_results(items: list[Any]) -> list[dict[str, Any]]:
        from lemoncrow.core.foundation.redaction import redact

        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("check") or "validation"
                detail = item.get("detail") or item.get("output") or ""
                passed = item.get("passed")
                if passed is None:
                    passed = item.get("status")
                normalized.append(
                    {
                        "name": redact(str(name)),
                        "passed": _coerce_trace_validation_passed(passed),
                        "detail": redact(str(detail)),
                    }
                )
                continue
            text = redact(str(item))
            lowered = text.lower()
            normalized.append(
                {
                    "name": text,
                    "passed": not any(token in lowered for token in ("fail", "error", "not run")),
                    "detail": "",
                }
            )
        return normalized

    def _normalize_trace_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        from lemoncrow.core.foundation.redaction import redact, redact_list

        def _normalize_trace_learnings(value: Any) -> list[dict[str, Any]]:
            if value is None:
                return []
            if isinstance(value, (str, dict)):
                items = [value]
            elif isinstance(value, list):
                items = value
            else:
                return []

            normalized_items: list[dict[str, Any]] = []
            for item in items:
                if isinstance(item, str):
                    text = redact(item.strip())
                    if text:
                        normalized_items.append({"kind": "note", "text": text})
                    continue
                if not isinstance(item, dict):
                    continue
                raw_text = (
                    item.get("text")
                    or item.get("learning")
                    or item.get("lesson")
                    or item.get("body")
                    or item.get("summary")
                    or ""
                )
                text = redact(str(raw_text).strip())
                if not text:
                    continue
                entry: dict[str, Any] = {"text": text}
                if item.get("kind") is not None:
                    entry["kind"] = redact(str(item["kind"]))
                if item.get("evidence") is not None:
                    entry["evidence"] = redact(str(item["evidence"]))
                promote_to = item.get("promote_to")
                if promote_to is None:
                    promote_to = item.get("target") or item.get("promotion_target")
                if promote_to is not None:
                    entry["promote_to"] = redact(str(promote_to))
                normalized_items.append(entry)
            return normalized_items

        def _normalize_trace_confidence(value: Any) -> str | None:
            if value is None:
                return None
            normalized_value = redact(str(value)).strip().lower()
            if not normalized_value or normalized_value in {"none", "null", "unknown"}:
                return None
            if normalized_value in {"full_live", "mcp_live", "wrapper_live", "imported", "manual"}:
                return normalized_value
            if normalized_value in {"high", "medium", "low"}:
                return "manual"
            return "manual"

        normalized = dict(payload)
        event_recorded = bool(normalized.pop("event_type", None))
        normalized.pop("event_payload", None)
        normalized.pop("prompt", None)
        normalized.pop("response", None)
        normalized.pop("bash_outputs", None)
        normalized.pop("tool_outputs", None)
        normalized.pop("capture_files", None)

        normalized["task"] = redact(str(normalized.get("task") or ""))
        _raw_files = normalized.get("files_touched") or []
        _files_normalized: list[Any] = []
        for _item in _raw_files:
            if isinstance(_item, dict) and "path" in _item:
                _entry: dict[str, Any] = {"path": redact(str(_item["path"]))}
                if _item.get("diff"):
                    _entry["diff"] = str(_item["diff"])
                if _item.get("event"):
                    _entry["event"] = str(_item["event"])
                _files_normalized.append(_entry)
            else:
                _files_normalized.append(redact(str(_item)))
        normalized["files_touched"] = _files_normalized
        normalized["commands_run"] = redact_list([str(item) for item in normalized.get("commands_run") or []])
        normalized["errors_seen"] = redact_list([str(item) for item in normalized.get("errors_seen") or []])
        normalized["diff_summary"] = redact(str(normalized.get("diff_summary") or ""))
        normalized["output_summary"] = redact(str(normalized.get("output_summary") or ""))
        normalized["trace_confidence"] = _normalize_trace_confidence(normalized.get("trace_confidence"))
        normalized["tools_called"] = _normalize_trace_tool_calls(list(normalized.get("tools_called") or []))
        normalized["validation_results"] = _normalize_trace_validation_results(
            list(normalized.get("validation_results") or [])
        )
        normalized["learnings"] = _normalize_trace_learnings(normalized.get("learnings"))
        return normalized, event_recorded

    @app.post("/v1/traces", tags=["traces"], dependencies=[Depends(verify_api_key)])
    def record_trace(payload: dict[str, Any]) -> dict[str, Any]:
        if "id" not in payload:
            payload["id"] = Trace.make_id(payload.get("task", "untitled"), payload.get("agent", "agent"))
        normalized_payload, event_recorded = _normalize_trace_payload(payload)
        trace = Trace.model_validate(normalized_payload)
        get_store().history.record_trace(trace)
        response: dict[str, Any] = {"id": trace.id, "event_recorded": event_recorded}
        if trace.session_id:
            response["session_id"] = trace.session_id
        return response

    # ------------------------------------------------------------------ #
    # Memory (agent long-term memory)                                     #
    # ------------------------------------------------------------------ #

    _mem_store: Any = None

    def _get_mem_store() -> Any:
        nonlocal _mem_store
        if _mem_store is None:
            from lemoncrow.infra.storage.factory import make_memory_store

            _mem_store = make_memory_store(store_path)
        return _mem_store

    def _team_manager_or_none() -> Any | None:
        from lemoncrow.pro.capabilities.team import TeamWorkspaceManager

        manager = TeamWorkspaceManager(store_path)
        return manager if manager.exists() else None

    @app.get("/v1/memory/blocks", tags=["knowledge"], dependencies=[Depends(verify_api_key)])
    def memory_list_or_get(
        agent_id: str | None = None,
        label: str | None = None,
        include_tombstoned: bool = False,
        limit: int = 200,
        user_id: str | None = None,
        shared_only: bool = False,
    ) -> Any:
        mem = _get_mem_store()
        manager = _team_manager_or_none()
        if label is not None:
            block = mem.get_block(agent_id, label, include_tombstoned=include_tombstoned)
            if block is None:
                raise HTTPException(status_code=404, detail=f"Block not found: {label!r}")
            if manager is not None:
                from lemoncrow.pro.capabilities.team import visible_memory_blocks

                visible = visible_memory_blocks([block], manager=manager, user_id=user_id, shared_only=shared_only)
                if not visible:
                    raise HTTPException(status_code=403, detail="block is not visible to this workspace user")
                block = visible[0]
            return block
        blocks = mem.list_blocks(agent_id, include_tombstoned=include_tombstoned, limit=limit)
        if manager is not None:
            from lemoncrow.pro.capabilities.team import visible_memory_blocks

            blocks = visible_memory_blocks(blocks, manager=manager, user_id=user_id, shared_only=shared_only)
        return blocks

    @app.post("/v1/memory/blocks", tags=["knowledge"], dependencies=[Depends(verify_api_key)])
    def memory_upsert_block(payload: dict[str, Any]) -> Any:
        from lemoncrow.core.foundation.memory_models import MemoryBlock
        from lemoncrow.infra.storage.memory_store import MemoryConcurrencyError

        mem = _get_mem_store()
        agent_id = payload.get("agent_id") or "shared"
        label = payload.get("label")
        if not label:
            raise HTTPException(status_code=400, detail="label is required")
        metadata = dict(payload.get("metadata") or {})
        manager = _team_manager_or_none()
        if manager is not None:
            workspace = manager.load_workspace()
            member = manager.require_member(str(payload.get("user_id") or ""), workspace=workspace)
            metadata.setdefault("scope", "private")
            metadata.setdefault("workspace_id", workspace.id)
            metadata.setdefault("owner_user_id", member.user_id)
            if metadata.get("scope") == "shared":
                from lemoncrow.pro.capabilities.team import ensure_shared_memory_write

                ensure_shared_memory_write(member)
        existing = mem.get_block(agent_id, label)
        if existing is None:
            value = str(payload.get("value", ""))
            limit_chars = _req_int(payload.get("limit_chars", 8000), "limit_chars")
            if len(value) > limit_chars:
                raise HTTPException(status_code=400, detail="value exceeds limit_chars")
            block = MemoryBlock(
                agent_id=agent_id,
                label=label,
                value=value,
                limit_chars=limit_chars,
                description=str(payload.get("description", "")),
                read_only=bool(payload.get("read_only", False)),
                pinned=bool(payload.get("pinned", False)),
                metadata=metadata,
            )
        else:
            expected_version = payload.get("expected_version")
            if expected_version is not None and existing.version != _req_int(expected_version, "expected_version"):
                raise HTTPException(
                    status_code=409,
                    detail=f"version conflict: expected {expected_version}, got {existing.version}",
                )
            update: dict[str, Any] = {}
            for field in ("value", "description", "read_only", "pinned", "metadata", "limit_chars"):
                if field in payload and payload[field] is not None:
                    update[field] = payload[field]
            if "metadata" not in update:
                update["metadata"] = metadata or existing.metadata
            else:
                update["metadata"] = metadata
            block = existing.model_copy(update=update)
        actor = str(payload.get("actor") or f"api:{agent_id}")
        try:
            return mem.upsert_block(block, actor=actor)
        except MemoryConcurrencyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/memory/archive", tags=["knowledge"], dependencies=[Depends(verify_api_key)])
    def memory_archive_passage(payload: dict[str, Any]) -> Any:
        import hashlib

        from lemoncrow.core.foundation.memory_models import ArchivalPassage

        mem = _get_mem_store()
        agent_id = payload.get("agent_id") or "shared"
        text = payload.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        valid_sources = ("trace", "block_evict", "user", "tool_output", "file_chunk")
        source = payload.get("source", "user")
        if source not in valid_sources:
            source = "user"
        dedup_hash = hashlib.sha256(f"{agent_id}:{text}".encode()).hexdigest()[:32]
        passage = ArchivalPassage(
            agent_id=agent_id,
            text=str(text),
            source=source,
            source_ref=str(payload.get("source_ref", "")),
            tags=list(payload.get("tags") or []),
            dedup_hash=dedup_hash,
        )
        saved = mem.insert_passage(passage)
        return {"id": saved.id, "dedup_hit": saved.dedup_hit}

    @app.post("/v1/memory/recall", tags=["knowledge"], dependencies=[Depends(verify_api_key)])
    def memory_recall_passages(payload: dict[str, Any]) -> Any:
        mem = _get_mem_store()
        agent_id = payload.get("agent_id")
        query = payload.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        since_str = payload.get("since")
        since_dt: datetime | None = None
        if since_str:
            try:
                since_dt = datetime.fromisoformat(since_str).replace(tzinfo=UTC)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid since: {since_str!r}") from exc
        passages = mem.search_passages(
            agent_id,
            str(query),
            top_k=_req_int(payload.get("top_k", 5), "top_k"),
            tags=list(payload.get("tags") or []) or None,
            since=since_dt,
        )
        return {"passages": [p.model_dump(mode="json") for p in passages]}

    @app.get("/v1/team/workspace", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_workspace_get() -> Any:
        manager = _team_manager_or_none()
        if manager is None:
            raise HTTPException(status_code=404, detail="team workspace not initialized")
        return manager.load_workspace()

    @app.post("/v1/team/workspace", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_workspace_init(payload: dict[str, Any]) -> Any:
        from lemoncrow.pro.capabilities.team import TeamWorkspaceError, TeamWorkspaceManager

        name = str(payload.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        try:
            return TeamWorkspaceManager(store_path).init_workspace(
                name=name,
                admin_email=str(payload.get("admin_email") or "admin@local"),
            )
        except TeamWorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/team/invite", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_invite(payload: dict[str, Any]) -> Any:
        from lemoncrow.pro.capabilities.team import TeamPermissionError, TeamWorkspaceManager

        emails = [str(item).strip().lower() for item in payload.get("emails") or [] if str(item).strip()]
        if not emails:
            raise HTTPException(status_code=400, detail="emails are required")
        manager = TeamWorkspaceManager(store_path)
        try:
            invites = manager.invite_members(
                emails,
                role=str(payload.get("role") or "member"),  # type: ignore[arg-type]
                actor_user_id=str(payload.get("user_id") or "") or None,
            )
        except TeamPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return [invite.model_dump(mode="json") for invite in invites]

    @app.post("/v1/team/join", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_join(payload: dict[str, Any]) -> Any:
        from lemoncrow.pro.capabilities.team import TeamWorkspaceError, TeamWorkspaceManager

        code = str(payload.get("invite_code") or "").strip()
        if not code:
            raise HTTPException(status_code=400, detail="invite_code is required")
        try:
            member = TeamWorkspaceManager(store_path).join_workspace(
                code,
                user_id=str(payload.get("user_id") or "") or None,
            )
        except TeamWorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return member

    @app.post("/v1/team/role", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_role(payload: dict[str, Any]) -> Any:
        from lemoncrow.pro.capabilities.team import (
            TeamPermissionError,
            TeamWorkspaceError,
            TeamWorkspaceManager,
        )

        user_id = str(payload.get("target_user_id") or "").strip().lower()
        role = str(payload.get("role") or "").strip()
        if not user_id or not role:
            raise HTTPException(status_code=400, detail="target_user_id and role are required")
        manager = TeamWorkspaceManager(store_path)
        try:
            return manager.set_role(
                user_id,
                role,  # type: ignore[arg-type]
                actor_user_id=str(payload.get("user_id") or "") or None,
            )
        except TeamPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except TeamWorkspaceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/team/usage", tags=["team"], dependencies=[Depends(verify_api_key)])
    def team_usage(user_id: str | None = None, since: str | None = None) -> Any:
        from lemoncrow.pro.capabilities.team import (
            TeamPermissionError,
            TeamWorkspaceManager,
            summarize_workspace_usage,
        )

        manager = TeamWorkspaceManager(store_path)
        try:
            manager.require_admin(user_id or None)
        except TeamPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        since_dt = _req_dt(since, "since").replace(tzinfo=UTC) if since else None
        return summarize_workspace_usage(store_path, manager=manager, since=since_dt)

    @app.get("/v1/governance/policy", tags=["governance"], dependencies=[Depends(verify_api_key)])
    def governance_get() -> Any:
        from lemoncrow.core.capabilities.governance import load_policy

        return load_policy(store_path)

    @app.post("/v1/governance/policy", tags=["governance"], dependencies=[Depends(verify_api_key)])
    def governance_set(payload: dict[str, Any]) -> Any:
        from lemoncrow.core.capabilities.governance import GovernancePolicy, save_policy
        from lemoncrow.pro.capabilities.team import (
            TeamAuditEvent,
            TeamPermissionError,
            TeamWorkspaceManager,
        )

        manager = TeamWorkspaceManager(store_path)
        try:
            actor = manager.require_admin(str(payload.get("user_id") or "") or None)
        except TeamPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        policy = GovernancePolicy.model_validate(payload.get("policy") or {})
        saved = save_policy(store_path, policy)
        manager.append_audit_event(
            TeamAuditEvent(action="governance.apply", actor_user_id=actor.user_id, details={"source": "api"})
        )
        return saved

    @app.post("/v1/audit/export", tags=["audit"], dependencies=[Depends(verify_api_key)])
    def audit_export(payload: dict[str, Any]) -> Any:
        from lemoncrow.core.capabilities.audit_export import export_audit_bundle
        from lemoncrow.pro.capabilities.team import (
            TeamAuditEvent,
            TeamPermissionError,
            TeamWorkspaceManager,
        )

        out_dir = payload.get("out_dir")
        if not out_dir:
            raise HTTPException(status_code=400, detail="out_dir is required")
        manager = TeamWorkspaceManager(store_path)
        try:
            actor = manager.require_admin(str(payload.get("user_id") or "") or None)
        except TeamPermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        since_raw = str(payload.get("since") or "").strip()
        since_dt = _req_dt(since_raw, "since").replace(tzinfo=UTC) if since_raw else None
        try:
            target_dir = confine_to_root(str(out_dir), store_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"out_dir must stay under {store_path}") from exc
        result = export_audit_bundle(store_path, out_dir=target_dir, since=since_dt)
        manager.append_audit_event(
            TeamAuditEvent(
                action="audit.export",
                actor_user_id=actor.user_id,
                details={"bundle_dir": result["bundle_dir"]},
            )
        )
        return result

    @app.post("/v1/audit/verify", tags=["audit"], dependencies=[Depends(verify_api_key)])
    def audit_verify(payload: dict[str, Any]) -> Any:
        from lemoncrow.core.capabilities.audit_export import verify_audit_bundle

        bundle_dir = payload.get("bundle_dir")
        if not bundle_dir:
            raise HTTPException(status_code=400, detail="bundle_dir is required")
        try:
            target_dir = confine_to_root(str(bundle_dir), store_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"bundle_dir must stay under {store_path}") from exc
        return verify_audit_bundle(store_path, bundle_dir=target_dir)

    # ------------------------------------------------------------------ #
    # Telemetry                                                           #
    # ------------------------------------------------------------------ #

    @app.get("/telemetry/config", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def get_telemetry_config() -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.config import load_telemetry_config

        cfg_telemetry = load_telemetry_config()
        return {
            "remote_enabled": cfg_telemetry.remote_enabled,
            "lexical_frustration_enabled": cfg_telemetry.lexical_frustration_enabled,
        }

    @app.post("/telemetry/config", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def update_telemetry_config(payload: dict[str, Any]) -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.config import save_telemetry_config

        cfg_telemetry = save_telemetry_config(
            remote_enabled=payload.get("remote_enabled"),
            lexical_frustration_enabled=payload.get("lexical_frustration_enabled"),
        )
        return {
            "remote_enabled": cfg_telemetry.remote_enabled,
            "lexical_frustration_enabled": cfg_telemetry.lexical_frustration_enabled,
        }

    @app.post("/telemetry/ack", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def acknowledge_telemetry() -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.banner import mark_acknowledged
        from lemoncrow.core.service.telemetry.config import load_telemetry_config

        mark_acknowledged()
        cfg_telemetry = load_telemetry_config()
        return {
            "remote_enabled": cfg_telemetry.remote_enabled,
            "lexical_frustration_enabled": cfg_telemetry.lexical_frustration_enabled,
        }

    @app.get("/telemetry/local", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def list_local_telemetry(
        limit: int = Query(200, ge=1, le=5000),
        since: float | None = Query(None),
        event: str | None = Query(None),
        host: str | None = Query(None),
    ) -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.local_store import LocalTelemetryStore

        store = LocalTelemetryStore()
        events = store.list_events(limit=limit, since=since, event=event, host=host)
        return {"events": events}

    @app.post("/telemetry/local", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def post_local_telemetry(payload: dict[str, Any]) -> dict[str, bool]:
        from lemoncrow.core.service.telemetry.local_store import LocalTelemetryStore

        event = payload.get("event")
        if not isinstance(event, str) or not event:
            raise HTTPException(status_code=400, detail="event (non-empty string) is required")
        props = payload.get("props", {})
        if not isinstance(props, dict):
            raise HTTPException(status_code=400, detail="props must be an object")
        store = LocalTelemetryStore()
        store.write_event(event=event, props=props, exported=False)
        return {"ok": True}

    @app.get("/telemetry/summary", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def get_telemetry_summary(
        since: float | None = Query(None),
        event: str | None = Query(None),
        host: str | None = Query(None),
    ) -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.local_store import LocalTelemetryStore

        store = LocalTelemetryStore()
        return store.summary(since=since, event=event, host=host)

    @app.get("/telemetry/schema", tags=["telemetry"], dependencies=[Depends(verify_api_key)])
    def get_telemetry_schema() -> dict[str, Any]:
        from lemoncrow.core.service.telemetry.schema import schema_dump

        return schema_dump()

    # ------------------------------------------------------------------ #
    # Analytics & Compatibility Endpoints                                 #
    # ------------------------------------------------------------------ #

    @app.get("/analytics", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_analytics(
        agent: str | None = Query(None),
        category: str | None = Query(None),
        limit: int = Query(1000, ge=1, le=10000),
        grouped: bool = Query(True),
        days: int | None = Query(None),
    ) -> list[dict[str, Any]]:
        """GET /analytics -> granular tool usage dynamically from traces.

        Calculated on the fly from the JSON payloads in the traces table
        to ensure 100% lossless session-level accuracy.
        """
        db_path = store.history.db_path
        # Filters are applied inside _query_analytics_rows, before the row limit.
        return _query_analytics_rows(db_path, grouped=grouped, days=days, limit=limit, agent=agent, category=category)

    @app.get("/analytics/summary", tags=["analytics"], dependencies=[Depends(verify_api_key)])
    def analytics_summary(
        agent: str | None = Query(None),
        model: str | None = Query(None),
        category: str | None = Query(None),
        search: str | None = Query(None),
        limit: int = Query(5000, ge=1, le=10000),
        grouped: bool = Query(True),
        days: int | None = Query(None),
    ) -> dict[str, Any]:
        # Filters are applied inside _query_analytics_rows, before the row limit.
        rows = _query_analytics_rows(
            store.history.db_path,
            grouped=grouped,
            days=days,
            limit=limit,
            agent=agent,
            model=model,
            category=category,
            search=search,
        )
        return _build_analytics_summary(rows, days=days)

    @app.get("/analytics/dashboard", tags=["analytics"], dependencies=[Depends(verify_api_key)])
    def analytics_dashboard(
        days: int = Query(30, ge=1, le=365),
        host: str | None = Query(None),
    ) -> dict[str, Any]:
        """Session-level aggregated analytics for the dashboard.

        Returns daily activity, per-domain, per-host, per-model breakdowns,
        top sessions, and tool-type distributions in one call.
        """
        db_path = store.history.db_path
        start_day = (datetime.now().astimezone().date() - timedelta(days=_clamp_days(days) - 1)).isoformat()
        host_filter = "AND COALESCE(host, agent) = ?" if host else ""
        sql = f"""
            SELECT
                id,
            COALESCE(host, agent) AS host,
                domain,
                json_extract(payload, '$.model') AS model,
                CAST(json_extract(payload, '$.input_tokens') AS INTEGER) AS input_tokens,
                CAST(json_extract(payload, '$.output_tokens') AS INTEGER) AS output_tokens,
                CAST(json_extract(payload, '$.reasoning_output_tokens') AS INTEGER) AS reasoning_output_tokens,
                CAST(json_extract(payload, '$.thinking_tokens') AS INTEGER) AS thinking_tokens,
                CAST(json_extract(payload, '$.cached_input_tokens') AS INTEGER) AS cached_tokens,
                CAST(json_extract(payload, '$.cache_creation_input_tokens') AS INTEGER) AS cache_write_tokens,
                CAST(json_extract(payload, '$.user_prompt_tokens') AS INTEGER) AS user_prompt_tokens,
                payload,
                created_at,
                date(datetime(created_at, 'localtime')) AS day,
                strftime('%Y-%m-%d %H:00', datetime(created_at)) AS hour_bucket
            FROM traces
            WHERE date(datetime(created_at, 'localtime')) >= ?
            {host_filter}
            ORDER BY created_at DESC
        """
        params: list[Any] = [start_day]
        if host:
            params.append(host)

        sessions = []
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            for row in rows:
                d = dict(row)
                payload_obj: dict[str, Any] = {}
                tools_called: list[dict[str, Any]] = []
                try:
                    parsed_payload = json.loads(d.get("payload") or "{}")
                    if isinstance(parsed_payload, dict):
                        payload_obj = parsed_payload
                        raw_tools = payload_obj.get("tools_called") or []
                        if isinstance(raw_tools, list):
                            tools_called = [tool for tool in raw_tools if isinstance(tool, dict)]
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Failed to parse trace payload: %s", exc)

                payload_for_usage = payload_obj or {
                    "model": d.get("model") or "",
                    "input_tokens": d.get("input_tokens") or 0,
                    "output_tokens": d.get("output_tokens") or 0,
                    "reasoning_output_tokens": d.get("reasoning_output_tokens") or 0,
                    "thinking_tokens": d.get("thinking_tokens") or 0,
                    "cached_input_tokens": d.get("cached_tokens") or 0,
                    "cache_creation_input_tokens": d.get("cache_write_tokens") or 0,
                }
                usage_entries = _trace_usage_entries(payload_for_usage)
                model_usages = _trace_model_usages(payload_for_usage)
                session_model = _trace_session_model(payload_for_usage, usage_entries, model_usages)

                in_t = d.get("input_tokens") or 0
                out_t = d.get("output_tokens") or 0
                reasoning_out_t = min(d.get("reasoning_output_tokens") or 0, out_t)
                think_t = d.get("thinking_tokens") or 0
                cache_r = d.get("cached_tokens") or 0
                cache_w = d.get("cache_write_tokens") or 0

                cost = _trace_cost_from_payload(payload_for_usage)

                sessions.append(
                    {
                        "id": d["id"],
                        "session_key": str(payload_obj.get("session_id") or d["id"]),
                        "host": d["host"] or "unknown",
                        "domain": d["domain"] or "unknown",
                        "model": session_model,
                        "model_usages": model_usages,
                        "day": d.get("day") or "",
                        "hour_bucket": d.get("hour_bucket") or "",
                        "created_at": d.get("created_at") or "",
                        "input_tokens": in_t,
                        "output_tokens": out_t,
                        "reasoning_output_tokens": reasoning_out_t,
                        "visible_output_tokens": max(out_t - reasoning_out_t, 0),
                        "reasoning_output_ratio": (round(reasoning_out_t / out_t, 4) if out_t else 0.0),
                        "thinking_tokens": think_t,
                        "cached_tokens": cache_r,
                        "cache_write_tokens": cache_w,
                        "user_prompt_tokens": d.get("user_prompt_tokens") or 0,
                        "cost": cost,
                        "tools_called": tools_called,
                    }
                )

        def _tool_call_count(session: dict[str, Any]) -> int:
            total = 0
            for tool in session.get("tools_called") or []:
                if not isinstance(tool, dict):
                    continue
                total += int(tool.get("count") or 1)
            return total

        def _tool_output_tokens(session: dict[str, Any]) -> int:
            total = 0
            for tool in session.get("tools_called") or []:
                if not isinstance(tool, dict):
                    continue
                total += int(tool.get("output_tokens") or 0)
            return total

        def _has_usage_signal(session: dict[str, Any]) -> bool:
            model = str(session.get("model") or "").strip()
            return any(
                (
                    bool(model),
                    (session.get("output_tokens") or 0) > 0,
                    (session.get("thinking_tokens") or 0) > 0,
                    float(session.get("cost") or 0.0) > 0,
                    _tool_call_count(session) > 0,
                    _tool_output_tokens(session) > 0,
                )
            )

        def _dashboard_session_rank(session: dict[str, Any]) -> tuple[Any, ...]:
            return (
                1 if _has_usage_signal(session) else 0,
                float(session.get("cost") or 0.0),
                int(session.get("output_tokens") or 0),
                int(session.get("thinking_tokens") or 0),
                int(session.get("input_tokens") or 0) + int(session.get("cached_tokens") or 0),
                _tool_call_count(session),
                _tool_output_tokens(session),
                str(session.get("created_at") or ""),
            )

        session_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for session in sessions:
            key = (
                str(session.get("host") or "unknown"),
                str(session.get("session_key") or session["id"]),
            )
            session_groups.setdefault(key, []).append(session)

        dashboard_sessions: list[dict[str, Any]] = []
        for (_, session_key), session_rows in session_groups.items():
            chosen = max(session_rows, key=_dashboard_session_rank)
            if not _has_usage_signal(chosen):
                continue
            collapsed = dict(chosen)
            collapsed["session_key"] = session_key
            dashboard_sessions.append(collapsed)
        dashboard_sessions.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)

        daily: dict[str, dict[str, Any]] = {}
        hourly: dict[str, dict[str, Any]] = {}
        for s in dashboard_sessions:
            day = s["day"]
            if day not in daily:
                daily[day] = {
                    "date": day,
                    "sessions": 0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            daily[day]["sessions"] += 1
            daily[day]["cost"] += s["cost"]
            daily[day]["input_tokens"] += s["input_tokens"]
            daily[day]["output_tokens"] += s["output_tokens"]

            hour = str(s.get("hour_bucket") or "")
            if hour:
                if hour not in hourly:
                    hourly[hour] = {
                        "date": hour,
                        "sessions": 0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    }
                hourly[hour]["sessions"] += 1
                hourly[hour]["cost"] += s["cost"]
                hourly[hour]["input_tokens"] += s["input_tokens"]
                hourly[hour]["output_tokens"] += s["output_tokens"]
        daily_list = sorted(daily.values(), key=lambda x: x["date"])
        hourly_list = sorted(hourly.values(), key=lambda x: x["date"])

        by_domain: dict[str, dict[str, Any]] = {}
        for s in dashboard_sessions:
            dom = s["domain"]
            if dom not in by_domain:
                by_domain[dom] = {"domain": dom, "sessions": 0, "cost": 0.0}
            by_domain[dom]["sessions"] += 1
            by_domain[dom]["cost"] += s["cost"]
        by_domain_list = sorted(by_domain.values(), key=lambda x: x["cost"], reverse=True)[:30]
        for r in by_domain_list:
            r["avg_cost"] = r["cost"] / r["sessions"] if r["sessions"] else 0.0

        by_host: dict[str, dict[str, Any]] = {}
        for s in dashboard_sessions:
            h = _host_family(s["host"])
            if h not in by_host:
                by_host[h] = {
                    "host": h,
                    "sessions": 0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                }
            by_host[h]["sessions"] += 1
            by_host[h]["cost"] += s["cost"]
            by_host[h]["input_tokens"] += s["input_tokens"]
            by_host[h]["cached_tokens"] += s["cached_tokens"]
        by_host_list = sorted(by_host.values(), key=lambda x: x["cost"], reverse=True)
        for r in by_host_list:
            total_in = r["input_tokens"] + r["cached_tokens"]
            r["cache_pct"] = (r["cached_tokens"] / total_in * 100) if total_in else 0.0

        by_model: dict[str, dict[str, Any]] = {}
        for s in dashboard_sessions:
            seen_models: set[str] = set()
            for usage in s.get("model_usages") or []:
                mdl = str(usage.get("model") or "unknown") or "unknown"
                if mdl not in by_model:
                    by_model[mdl] = {
                        "model": mdl,
                        "sessions": 0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_tokens": 0,
                    }
                if mdl not in seen_models:
                    by_model[mdl]["sessions"] += 1
                    seen_models.add(mdl)
                by_model[mdl]["cost"] += _model_usage_cost(usage)
                by_model[mdl]["input_tokens"] += int(usage.get("input_tokens") or 0)
                by_model[mdl]["output_tokens"] += int(usage.get("output_tokens") or 0)
                by_model[mdl]["cached_tokens"] += int(usage.get("cached_input_tokens") or 0)
        by_model_list = sorted(by_model.values(), key=lambda x: x["cost"], reverse=True)
        for r in by_model_list:
            total_in = r["input_tokens"] + r["cached_tokens"]
            r["cache_pct"] = (r["cached_tokens"] / total_in * 100) if total_in else 0.0

        host_model_overview: dict[tuple[str, str], dict[str, Any]] = {}
        for s in dashboard_sessions:
            host_name = _host_family(s["host"])
            session_model = str(s.get("model") or "")
            usage_rows = list(s.get("model_usages") or [])
            overview_models = {str(usage.get("model") or "") or "unknown" for usage in usage_rows}
            attributed_model = session_model
            if not attributed_model and len(overview_models) == 1:
                attributed_model = next(iter(overview_models))
            seen_pairs: set[tuple[str, str]] = set()
            for usage in usage_rows:
                model_name = str(usage.get("model") or "") or "unknown"
                key = (host_name, model_name)
                if key not in host_model_overview:
                    host_model_overview[key] = {
                        "host": host_name,
                        "model": model_name,
                        "sessions": 0,
                        "user_typed_tokens": 0,
                        "base_context_tokens": 0,
                        "cached_prompt_tokens": 0,
                        "cache_write_tokens": 0,
                        "billable_output_tokens": 0,
                        "tool_output_tokens": 0,
                        "thinking_tokens": 0,
                        "tool_calls": 0,
                        "cost": 0.0,
                    }
                row = host_model_overview[key]
                if key not in seen_pairs:
                    row["sessions"] += 1
                    seen_pairs.add(key)
                row["base_context_tokens"] += int(usage.get("input_tokens") or 0)
                row["cached_prompt_tokens"] += int(usage.get("cached_input_tokens") or 0)
                row["cache_write_tokens"] += int(usage.get("cache_creation_input_tokens") or 0)
                row["billable_output_tokens"] += int(usage.get("output_tokens") or 0)
                row["thinking_tokens"] += int(usage.get("thinking_tokens") or 0)
                row["cost"] += _model_usage_cost(usage)
                if attributed_model and model_name == attributed_model:
                    row["user_typed_tokens"] += s["user_prompt_tokens"]
                    row["tool_output_tokens"] += _tool_output_tokens(s)
                    row["tool_calls"] += _tool_call_count(s)
        host_model_overview_list = sorted(
            host_model_overview.values(),
            key=lambda row: (row["cost"], row["sessions"]),
            reverse=True,
        )

        from lemoncrow.infra.runtime.session_report import read_total_savings_from_events

        _lemoncrow_root = Path(cfg.lemoncrow_root)

        # Per-session LemonCrow savings — sum routing/compaction events keyed
        # by session_id. Display savings (content[].saved) live in the host
        # transcript and aren't aggregated here.
        _session_savings: dict[str, float] = {}
        for _s in dashboard_sessions:
            _sk = str(_s.get("session_key") or _s["id"])
            if _sk in _session_savings:
                continue
            try:
                _session_savings[_sk] = read_total_savings_from_events(_sk, _lemoncrow_root)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                _session_savings[_sk] = 0.0

        top_sessions_clean = [
            {
                "id": s.get("session_key") or s["id"],
                "host": s["host"],
                "domain": s["domain"],
                "model": s["model"],
                "date": s["created_at"][:10] if s["created_at"] else "",
                "cost": s["cost"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "cached_tokens": s["cached_tokens"],
                "lemoncrow_savings_usd": _session_savings.get(str(s.get("session_key") or s["id"]), 0.0),
            }
            for s in sorted(dashboard_sessions, key=lambda x: x["cost"], reverse=True)[:20]
        ]

        _CORE = {
            "Read",
            "Edit",
            "Write",
            "Grep",
            "Glob",
            "ListDir",
            "NotebookRead",
            "NotebookEdit",
            "TodoRead",
            "TodoWrite",
            "WebSearch",
            "WebFetch",
            "MultiEdit",
            "view",
            "read_file",
            "replace",
            "apply_patch",
        }
        _SHELL = {"Bash", "bash", "run_shell_command", "exec_command", "shell", "run_code"}

        core_tools: dict[str, dict[str, Any]] = {}
        shell_tools: dict[str, dict[str, Any]] = {}
        mcp_tools: dict[str, dict[str, Any]] = {}

        for s in dashboard_sessions:
            for tool in s["tools_called"]:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name") or ""
                calls = tool.get("count") or 1
                tin = tool.get("input_tokens") or 0
                tout = tool.get("output_tokens") or 0

                if name in _CORE:
                    bucket = core_tools
                elif name in _SHELL:
                    bucket = shell_tools
                elif name.startswith("mcp__") or name.startswith("mcp:"):
                    bucket = mcp_tools
                else:
                    continue

                if name not in bucket:
                    bucket[name] = {"name": name, "calls": 0, "input_tokens": 0, "output_tokens": 0}
                bucket[name]["calls"] += calls
                bucket[name]["input_tokens"] += tin
                bucket[name]["output_tokens"] += tout

        _total_savings = round(sum(_session_savings.values()), 6)
        _total_cost_window = round(sum(float(session["cost"]) for session in dashboard_sessions), 6)

        return {
            "summary": {
                "total_cost": _total_cost_window,
                "projected_monthly_cost": round(
                    _total_cost_window * (30 / max(days, 1)),
                    6,
                ),
                "total_sessions": len(dashboard_sessions),
                "total_lemoncrow_savings_usd": _total_savings,
                "savings_pct": (round(_total_savings / _total_cost_window * 100, 2) if _total_cost_window > 0 else 0.0),
            },
            "daily": daily_list,
            "hourly": hourly_list,
            "by_domain": by_domain_list,
            "by_host": by_host_list,
            "by_model": by_model_list,
            "host_model_overview": host_model_overview_list,
            "top_sessions": top_sessions_clean,
            "tools": {
                "core": sorted(core_tools.values(), key=lambda x: x["calls"], reverse=True)[:15],
                "shell": sorted(shell_tools.values(), key=lambda x: x["calls"], reverse=True)[:10],
                "mcp": sorted(mcp_tools.values(), key=lambda x: x["calls"], reverse=True)[:20],
            },
        }

    @app.get("/plans", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_plans(limit: int = 50) -> list[dict[str, Any]]:
        """Compatibility: GET /plans -> derives plan records from traces."""
        traces = get_store().history.list_traces(limit=limit)
        result = []
        for t in traces:
            if t.session_id:
                result.append(
                    {
                        "trace_id": t.id,
                        "domain": t.domain,
                        "task": t.task,
                        "status": t.status,
                        "plan_checks": [{"name": "plan_valid", "passed": t.status == "success"}],
                    }
                )
        return result

    @app.get("/pricing", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def get_pricing_table() -> dict[str, Any]:
        """GET /pricing -> returns the current model pricing table."""
        from lemoncrow.core.capabilities.pricing import _load_pricing_table

        return _load_pricing_table()

    # ------------------------------------------------------------------ #
    # Raw artifact HTML viewer                                           #
    # ------------------------------------------------------------------ #

    def _render_raw_artifact_html(artifact: Any, content: str) -> str:
        """Render raw artifact content as a standalone HTML page for browser viewing."""
        import html as _html

        meta = artifact.model_dump(mode="json")
        # Escape ``</`` so a ``</script>`` inside the JSON literal can't break out
        # of the inline <script> below (json.dumps does NOT escape '/'). Without
        # this, artifact content containing ``</script><img src=x onerror=...>``
        # executes on the API origin — stored XSS.
        meta_json = json.dumps(meta, default=str, indent=2).replace("</", "<\\/")

        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

        # Attempt to detect JSONL: if >= 80% of non-empty lines parse as JSON
        lines = content.rstrip("\n").split("\n")
        json_lines = 0
        non_empty = 0
        for ln in lines:
            if ln.strip():
                non_empty += 1
                try:
                    json.loads(ln)
                    json_lines += 1
                except (json.JSONDecodeError, ValueError):
                    pass
        is_jsonl = non_empty > 0 and (json_lines / non_empty) >= 0.8

        # Escape content as a JS-safe string via JSON. ``</`` → ``<\/`` so a
        # ``</script>`` in the artifact body can't terminate the inline <script>
        # and inject markup (json.dumps does not escape '/'). See meta_json above.
        content_js = json.dumps(content).replace("</", "<\\/")

        short_id = artifact.id[:24] + "…" if len(artifact.id) > 24 else artifact.id

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Raw Artifact · {_html.escape(short_id)}</title>
<style>
  *, *::before, *::after {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html {{ font-size: 14px; }}
  body {{
    background: #0a0a0a;
    color: #d4d4d4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px;
  }}
  a {{ color: #61afef; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* ── Metadata card ── */
  .meta {{
    background: #141416;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 6px 20px;
  }}
  .meta dt {{
    font-size: 9px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #555;
    margin-top: 4px;
  }}
  .meta dd {{
    font-family: "SF Mono", "Fira Code", "Cascadia Code", "JetBrains Mono", Menlo, monospace;
    font-size: 11px;
    color: #b0b0b0;
    word-break: break-all;
  }}
  .meta .id-val {{ color: #61afef; }}
  .meta .src-val {{ color: #e5c07b; }}
  .meta .kind-val {{ color: #98c379; }}
  .meta .redacted-yes {{ color: #e06c75; }}
  .meta .redacted-no {{ color: #98c379; }}
  .meta .title-row {{
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 8px;
    margin-bottom: 4px;
    border-bottom: 1px solid #222;
  }}
  .meta .title-row h1 {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #888;
  }}
  .meta .title-row .artifact-id {{
    font-family: "SF Mono", "Fira Code", Menlo, monospace;
    font-size: 10px;
    color: #61afef;
    background: #1e1e24;
    padding: 2px 8px;
    border-radius: 4px;
  }}
  .meta .line-count {{
    margin-left: auto;
    font-family: "SF Mono", "Fira Code", Menlo, monospace;
    font-size: 10px;
    color: #888;
    background: #1e1e24;
    padding: 2px 8px;
    border-radius: 4px;
  }}

  /* ── Toolbar ── */
  .toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .toolbar button {{
    background: #1e1e24;
    border: 1px solid #333;
    color: #b0b0b0;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }}
  .toolbar button:hover {{
    background: #2a2a32;
    border-color: #555;
    color: #e0e0e0;
  }}
  .toolbar button:active {{
    transform: scale(0.97);
  }}
  .toolbar .copy-btn {{ color: #98c379; }}
  .toolbar .copy-btn:hover {{ border-color: #98c379; }}
  .toolbar .status {{
    font-size: 10px;
    color: #666;
    margin-left: auto;
  }}

  /* ── Content area ── */
  .content-wrap {{
    background: #0d0d0f;
    border: 1px solid #1a1a1e;
    border-radius: 6px;
    overflow: hidden;
  }}
  .line {{
    display: flex;
    min-height: 21px;
    border-bottom: 1px solid #111113;
    font-family: "SF Mono", "Fira Code", "Cascadia Code", "JetBrains Mono", Menlo, monospace;
    font-size: 11px;
    line-height: 1.6;
    transition: background 0.1s;
  }}
  .line:hover {{
    background: #121216;
  }}
  .line:last-child {{ border-bottom: none; }}
  .line-num {{
    flex-shrink: 0;
    width: 48px;
    padding: 0 10px;
    text-align: right;
    color: #333;
    font-size: 10px;
    user-select: none;
    border-right: 1px solid #16161a;
    padding-right: 12px;
    background: #0a0a0c;
  }}
  .line-content {{
    flex: 1;
    padding: 0 14px;
    white-space: pre-wrap;
    word-break: break-all;
    overflow-x: auto;
  }}
  .line-content.raw-text {{
    color: #888;
    font-style: italic;
  }}

  /* ── JSON syntax highlighting ── */
  .hl-key {{ color: #61afef; }}
  .hl-str {{ color: #98c379; }}
  .hl-num {{ color: #d19a66; }}
  .hl-bool {{ color: #c678dd; }}
  .hl-null {{ color: #c678dd; font-style: italic; }}
  .hl-bracket {{ color: #777; }}
  .hl-punc {{ color: #666; }}
  .hl-comment {{ color: #555; font-style: italic; }}

  /* ── Responsive ── */
  @media (max-width: 640px) {{
    body {{ padding: 12px; }}
    .meta {{ grid-template-columns: 1fr 1fr; padding: 12px; }}
    .line-num {{ width: 36px; padding: 0 6px; font-size: 9px; }}
    .line-content {{ font-size: 10px; padding: 0 8px; }}
  }}

  /* ── Empty state ── */
  .empty {{
    padding: 40px 20px;
    text-align: center;
    color: #555;
    font-size: 12px;
  }}

  /* ── Loading / error ── */
  .error {{ color: #e06c75; padding: 20px; }}
</style>
</head>
<body>

<!-- Metadata -->
<dl class="meta">
  <div class="title-row">
    <h1>⧩ Raw Artifact</h1>
    <span class="artifact-id">{_html.escape(short_id)}</span>
    <span class="line-count">{line_count} line{"s" if line_count != 1 else ""}</span>
  </div>
  <div>
    <dt>Source</dt>
    <dd class="src-val">{_html.escape(str(meta.get("source", "")))}</dd>
  </div>
  <div>
    <dt>Kind</dt>
    <dd class="kind-val">{_html.escape(str(meta.get("kind", "")))}</dd>
  </div>
  <div>
    <dt>Created</dt>
    <dd>{_html.escape(str(meta.get("created_at", "")))}</dd>
  </div>
  <div>
    <dt>Redacted</dt>
    <dd class="redacted-{"yes" if meta.get("redacted") else "no"}">{"Yes" if meta.get("redacted") else "No"}</dd>
  </div>
  <div>
    <dt>Original Bytes</dt>
    <dd>{_html.escape(str(meta.get("byte_count_original", "")))}</dd>
  </div>
  <div>
    <dt>Redacted Bytes</dt>
    <dd>{_html.escape(str(meta.get("byte_count_redacted", "")))}</dd>
  </div>
  <div>
    <dt>Relative Path</dt>
    <dd>{_html.escape(str(meta.get("relative_path", "") or "—"))}</dd>
  </div>
  <div>
    <dt>Source Session</dt>
    <dd>{_html.escape(str(meta.get("source_session_id", "")))}</dd>
  </div>
  <div>
    <dt>Source Path</dt>
    <dd>{_html.escape(str(meta.get("source_path", "")) or "—")}</dd>
  </div>
  <div>
    <dt>Source File Mtime</dt>
    <dd>{_html.escape(str(meta.get("source_file_mtime", "")) or "—")}</dd>
  </div>
</dl>

<!-- Toolbar -->
<div class="toolbar">
  <button class="copy-btn" onclick="copyContent()">⎘ Copy All</button>
  <button onclick="toggleWrap()">↔ Wrap</button>
  <button onclick="toggleCollapsed()">⊟ Collapse All</button>
  <span class="status" id="status"></span>
</div>

<!-- Lines rendered by JS -->
<div class="content-wrap" id="root"></div>

<script>
(function() {{
  const CONTENT = {content_js};
  const IS_JSONL = {json.dumps(is_jsonl)};
  const META = {meta_json};

  let wrapEnabled = false;
  let allCollapsed = false;

  function escapeHtml(s) {{
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }}

  function colorizeJSON(obj) {{
    const formatted = JSON.stringify(obj, null, 2);
    // Tokenize the pretty-printed JSON
    let s = escapeHtml(formatted);
    // Keys: "key":
    s = s.replace(/(&quot;[^&]*&quot;)\\s*:/g, '<span class="hl-key">$1</span>:');
    s = s.replace(/(:\\s*)(&quot;[^&]*&quot;)/g, '$1<span class="hl-str">$2</span>');
    s = s.replace(/(,\\s*)(&quot;[^&]*&quot;)/g, '$1<span class="hl-str">$2</span>');
    s = s.replace(/(:\\s*)(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)/g, '$1<span class="hl-num">$2</span>');
    s = s.replace(/(,\\s*)(-?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)/g, '$1<span class="hl-num">$2</span>');
    s = s.replace(/(:\\s*)(true|false)/g, '$1<span class="hl-bool">$2</span>');
    s = s.replace(/(,\\s*)(true|false)/g, '$1<span class="hl-bool">$2</span>');
    s = s.replace(/(:\\s*)(null)/g, '$1<span class="hl-null">$2</span>');
    s = s.replace(/(,\\s*)(null)/g, '$1<span class="hl-null">$2</span>');
    // Brackets/punctuation
    s = s.replace(/([\\[\\]{{}}])/g, '<span class="hl-bracket">$1</span>');
    return s;
  }}

  function render() {{
    const root = document.getElementById('root');
    const lines = CONTENT.split('\\n');
    // Remove trailing empty line from split if content ends with newline
    if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();

    if (lines.length === 0 || (lines.length === 1 && lines[0] === '')) {{
      root.innerHTML = '<div class="empty">∅ empty artifact</div>';
      return;
    }}

    const fragments = [];
    let jsonOk = 0;
    let jsonFail = 0;

    for (let i = 0; i < lines.length; i++) {{
      const line = lines[i];
      const trimmed = line.trim();
      let contentHtml;

      if (IS_JSONL && trimmed) {{
        try {{
          const parsed = JSON.parse(trimmed);
          contentHtml = colorizeJSON(parsed);
          jsonOk++;
        }} catch (e) {{
          contentHtml = '<span class="hl-comment">// JSON parse error: ' + escapeHtml(e.message) + '</span>\\n' + escapeHtml(line);
          jsonFail++;
        }}
      }} else {{
        contentHtml = escapeHtml(line);
      }}

      fragments.push(
        '<div class="line">' +
          '<span class="line-num">' + (i + 1) + '</span>' +
          '<span class="line-content' + (contentHtml ? '' : ' raw-text') + '">' + (contentHtml || '\\u00A0') + '</span>' +
        '</div>'
      );
    }}

    root.innerHTML = fragments.join('');

    const statusEl = document.getElementById('status');
    if (IS_JSONL) {{
      const total = jsonOk + jsonFail;
      statusEl.textContent = jsonOk + '/' + total + ' JSON lines parsed' + (jsonFail ? ', ' + jsonFail + ' failed' : '');
    }} else {{
      statusEl.textContent = lines.length + ' lines';
    }}
  }}

  window.copyContent = function() {{
    navigator.clipboard.writeText(CONTENT).then(() => {{
      const btn = document.querySelector('.copy-btn');
      const orig = btn.textContent;
      btn.textContent = '✓ Copied!';
      setTimeout(() => btn.textContent = orig, 1500);
    }});
  }};

  window.toggleWrap = function() {{
    wrapEnabled = !wrapEnabled;
    document.querySelectorAll('.line-content').forEach(el => {{
      el.style.whiteSpace = wrapEnabled ? 'pre-wrap' : 'pre';
      el.style.wordBreak = wrapEnabled ? 'break-all' : 'normal';
    }});
    const btn = document.querySelector('.toolbar button:nth-child(2)');
    btn.textContent = wrapEnabled ? '↔ No Wrap' : '↔ Wrap';
  }};

  window.toggleCollapsed = function() {{
    allCollapsed = !allCollapsed;
    document.querySelectorAll('.line').forEach((el, i) => {{
      const content = el.querySelector('.line-content');
      if (allCollapsed && i > 10) {{
        el.style.display = 'none';
      }} else {{
        el.style.display = '';
      }}
    }});
    const btn = document.querySelector('.toolbar button:nth-child(3)');
    if (allCollapsed) {{
      // Show a "show all" row
      const root = document.getElementById('root');
      const showAll = document.createElement('div');
      showAll.className = 'line';
      showAll.style.cursor = 'pointer';
      showAll.style.justifyContent = 'center';
      showAll.style.padding = '12px';
      showAll.style.color = '#61afef';
      showAll.style.fontSize = '11px';
      showAll.id = 'show-all-row';
      showAll.textContent = '⊞ Show all ' + document.querySelectorAll('.line').length + ' lines';
      showAll.onclick = function() {{ allCollapsed = false; toggleCollapsed(); }};
      root.appendChild(showAll);
      btn.textContent = '⊞ Expand All';
    }} else {{
      const row = document.getElementById('show-all-row');
      if (row) row.remove();
      btn.textContent = '⊟ Collapse All';
    }}
  }};

  render();
}})();
</script>
</body>
</html>"""

    # ------------------------------------------------------------------ #
    # Raw artifacts                                                       #
    # ------------------------------------------------------------------ #

    @app.get("/raw-artifacts/{artifact_id}", tags=["artifacts"], dependencies=[Depends(verify_api_key)])
    def get_raw_artifact(artifact_id: str) -> dict[str, Any]:
        """Return metadata for a stored raw artifact."""
        store_inst = get_store()
        artifact = store_inst.history.get_raw_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Raw artifact not found: {artifact_id}")
        return artifact.model_dump(mode="json")

    @app.get(
        "/raw-artifacts/{artifact_id}/content",
        tags=["artifacts"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_raw_artifact_content(
        artifact_id: str,
        accept: str = Header(None),
    ) -> Any:
        """Return the raw artifact content as plain text or a pretty HTML page."""
        from fastapi.responses import HTMLResponse, PlainTextResponse

        store_inst = get_store()
        artifact = store_inst.history.get_raw_artifact(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Raw artifact not found: {artifact_id}")
        try:
            content = store_inst.history.read_raw_artifact_content(artifact)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Content file not found on disk") from exc

        # Serve a pretty-printed HTML page when a browser requests it
        # (skip for very large files to avoid browser memory issues)
        MAX_HTML_SIZE = 5 * 1024 * 1024  # 5 MB
        if accept and "text/html" in accept and len(content) <= MAX_HTML_SIZE:
            return HTMLResponse(_render_raw_artifact_html(artifact, content))

        return PlainTextResponse(content, media_type="text/plain")

    def _file_read_roots() -> list[Path]:
        """Directories the file endpoints are allowed to serve from.

        The store itself, the daemon's own workspace, every workspace LemonCrow
        has actually recorded a session for, plus anything the operator opts in
        through ``LEMONCROW_FILE_READ_ROOTS``. Without a list like this the
        endpoints below served *any* path on the box -- and ``verify_api_key``
        is a no-op unless ``LEMONCROW_REQUIRE_AUTH=true``, so "local caller"
        means any process or any page served from any localhost port.
        """
        roots: list[Path] = []
        seen: set[Path] = set()

        def add(candidate: Path | str) -> None:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                resolved = Path(candidate).expanduser().resolve()
                if resolved not in seen and resolved.is_dir():
                    seen.add(resolved)
                    roots.append(resolved)

        add(store_path)
        with contextlib.suppress(Exception):
            add(resolve_workspace_root())
        for raw in os.environ.get("LEMONCROW_FILE_READ_ROOTS", "").split(os.pathsep):
            if raw.strip():
                add(raw.strip())
        with contextlib.suppress(Exception):
            with sqlite3.connect(get_store().history.db_path) as conn:
                for (workspace_path,) in conn.execute(
                    "SELECT DISTINCT workspace_path FROM traces "
                    "WHERE workspace_path IS NOT NULL AND workspace_path != ''"
                ):
                    add(workspace_path)
        return roots

    def _confined_read_path(path: str) -> Path:
        """Resolve *path* (symlinks included) inside one of the allowed roots."""
        for allowed in _file_read_roots():
            with contextlib.suppress(ValueError):
                return confine_to_root(path, allowed)
        raise HTTPException(
            status_code=403,
            detail="path is outside the allowed roots (set LEMONCROW_FILE_READ_ROOTS to widen them)",
        )

    @app.get("/v1/files/content", tags=["files"], dependencies=[Depends(verify_api_key)])
    def get_file_content(path: str) -> Any:
        """Return local file content with a browser-appropriate media type."""
        from fastapi.responses import FileResponse

        file_path = _confined_read_path(path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
        media_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        return FileResponse(file_path, media_type=media_type, filename=file_path.name)

    @app.get("/v1/files/projection", tags=["files"], dependencies=[Depends(verify_api_key)])
    def get_file_projection(
        path: str,
        view: str = "compact",
        range: str | None = None,
        max_lines: int = 200,
    ) -> dict[str, Any]:
        """Return structured projection metadata for a file read."""
        from lemoncrow.gateway.adapters.mcp_server import tool_smart_read

        payload: dict[str, Any] = {"path": str(_confined_read_path(path)), "include_meta": True}
        if view == "compact":
            pass
        elif view == "exact":
            payload["full"] = True
        elif view == "summary":
            payload["max_lines"] = max_lines
        elif view == "range":
            if not range:
                raise HTTPException(status_code=400, detail="range is required when view=range")
            payload["range"] = range
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported projection view: {view}")
        return cast(dict[str, Any], tool_smart_read(payload))

    @app.get("/ledgers/{session_id}", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_ledger(session_id: str) -> dict[str, Any]:
        """Compatibility: GET /ledgers/{session_id} -> returns run ledger data.

        First checks for a live RunLedger JSON file (written by the reasoning
        runtime).  When that is absent, falls back to an imported Trace that
        carries the same session_id so that sessions imported from Claude / Codex /
        OpenCode / Copilot are still surfaced here.
        """
        from lemoncrow.core.foundation.paths import find_session_dir
        from lemoncrow.infra.runtime.run_ledger import RunLedger

        # Canonical per-session ledger: sessions/YYYY/MM/DD/<host>/<id>/run.json
        # (the old flat runs/<id>.json location was retired).
        _session_root = find_session_dir(Path(cfg.lemoncrow_root), session_id)
        ledger_path = _session_root / "run.json" if _session_root else None
        snap = None
        if ledger_path is not None and ledger_path.exists():
            try:
                ledger = RunLedger.load(ledger_path)
                snap = ledger.snapshot()
            except Exception:
                # The detail stays in the log: str(exc) on a ledger read leaks
                # absolute store paths, and a JSONDecodeError echoes file
                # content back to an unauthenticated local caller.
                logging.exception("Recovered from broad exception handler")
                return {"session_id": session_id, "error": "failed to load run ledger"}

        # Always check for a trace to fetch the full conversation history.
        # Imported sessions from Claude/Codex/OpenCode/Copilot use the Trace as source of truth.
        store_inst = get_store()

        # 1. Try direct ID match (high performance)
        trace = store_inst.history.get_trace(session_id)

        # 2. Try host-prefixed IDs (standard for imported sessions)
        if trace is None:
            from lemoncrow.gateway.hosts.session_parsers.registry import (
                SUPPORTED_SESSION_IMPORT_HOSTS,
            )

            for host in SUPPORTED_SESSION_IMPORT_HOSTS:
                trace = store_inst.history.get_trace(f"{host}-{session_id}")
                if trace:
                    break

        # 3. Slower fallback: search for the session_id inside payloads
        if trace is None:
            with sqlite3.connect(store_inst.history.db_path) as conn:
                # Use json_extract for efficient searching
                row = conn.execute(
                    "SELECT payload FROM traces WHERE json_extract(payload, '$.session_id') = ?",
                    (session_id,),
                ).fetchone()
                if row:
                    trace = Trace.model_validate_json(coerce_trace_json(row[0]))

        conversations: list[dict[str, Any]] = []
        source_files: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        seen_source_files: set[tuple[str, str]] = set()
        if trace:
            if trace.raw_artifact_ids:
                # Reconstruct from all raw artifacts (main session + subagents)
                for art_id in trace.raw_artifact_ids:
                    artifact = store_inst.history.get_raw_artifact(art_id)
                    if artifact:
                        scope = "subagent" if "subagents/" in str(artifact.relative_path).replace("\\", "/") else "main"
                        if artifact.source_path:
                            source_file_key = (artifact.source_path, artifact.id)
                            if source_file_key not in seen_source_files:
                                source_files.append({"path": artifact.source_path, "artifact_id": artifact.id})
                                seen_source_files.add(source_file_key)
                        try:
                            raw_content = store_inst.history.read_raw_artifact_content(artifact)
                            from lemoncrow.gateway.hosts.session_parsers._session_parser import (
                                parse_session_turns,
                            )

                            artifact_turns = parse_session_turns(raw_content, artifact.source)
                            artifact_label = "main"
                            if scope == "subagent":
                                artifact_label = (
                                    next(
                                        (
                                            str(turn.get("subagent_name") or "").strip()
                                            for turn in artifact_turns
                                            if str(turn.get("subagent_name") or "").strip()
                                        ),
                                        "",
                                    )
                                    or Path(artifact.relative_path).stem
                                )
                            artifacts.append(
                                {
                                    "id": artifact.id,
                                    "source": artifact.source,
                                    "kind": artifact.kind,
                                    "relative_path": artifact.relative_path,
                                    "source_path": artifact.source_path,
                                    "scope": scope,
                                    "label": artifact_label,
                                }
                            )
                            for turn in artifact_turns:
                                turn["artifact_id"] = artifact.id
                                turn["artifact_source"] = artifact.source
                                turn["artifact_kind"] = artifact.kind
                                turn["artifact_label"] = artifact_label
                                turn["source_scope"] = scope
                                if scope == "subagent" and not turn.get("subagent_name"):
                                    turn["subagent_name"] = artifact_label
                            conversations.extend(artifact_turns)
                        except Exception as exc:
                            artifacts.append(
                                {
                                    "id": artifact.id,
                                    "source": artifact.source,
                                    "kind": artifact.kind,
                                    "relative_path": artifact.relative_path,
                                    "source_path": artifact.source_path,
                                    "scope": scope,
                                    "label": "main" if scope == "main" else Path(artifact.relative_path).stem,
                                }
                            )
                            logger.error(
                                "Failed to reconstruct conversations from artifact %s (source=%s): %s",
                                art_id,
                                getattr(artifact, "source", "?"),
                                exc,
                                exc_info=True,
                            )
                conversations.sort(key=_conversation_sort_key)

            # Calculate turn costs on the fly using backend pricing
            from lemoncrow.core.capabilities.pricing import get_model_pricing

            report = None
            with contextlib.suppress(Exception):
                from lemoncrow.infra.runtime.session_report import load_report

                report = load_report(session_id, Path(cfg.lemoncrow_root))

            started_model = (
                getattr(report, "started_model", None)
                or next(iter(getattr(report, "models_used", {}) or {}), None)
                or trace.model
                or "_default"
            )
            for turn in conversations:
                t_toks = turn.get("tokens") or {}
                pricing = get_model_pricing(str(turn.get("model") or started_model or "_default"))
                turn["cost"] = pricing.cost_usd(
                    input_tokens=t_toks.get("in", 0),
                    output_tokens=t_toks.get("out", 0),
                    cache_read_tokens=t_toks.get("cache_read", 0),
                    cache_write_tokens=t_toks.get("cache_write", 0),
                )

        if snap:
            if conversations:
                snap["conversations"] = conversations
            if source_files:
                snap["source_files"] = source_files
            if artifacts:
                snap["artifacts"] = artifacts
            return snap

        if trace:
            return {
                "session_id": trace.session_id or session_id,
                "trace_id": trace.id,
                "status": trace.status,
                "task": trace.task,
                "agent": trace.agent,
                "domain": trace.domain,
                "created_at": trace.created_at.isoformat() if trace.created_at else None,
                "files_touched": trace.files_touched,
                "commands_run": trace.commands_run,
                "errors_seen": trace.errors_seen,
                "tools_called": [tc.model_dump() for tc in trace.tools_called],
                "conversations": conversations,
                "source_files": source_files,
                "artifacts": artifacts,
                "raw_artifact_ids": trace.raw_artifact_ids,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "reasoning_output_tokens": trace.reasoning_output_tokens,
                "thinking_tokens": trace.thinking_tokens,
                "cached_input_tokens": trace.cached_input_tokens,
                "cache_creation_input_tokens": trace.cache_creation_input_tokens,
                "user_prompt_tokens": trace.user_prompt_tokens,
                "model": trace.model,
                "note": "Imported from session logs." if trace.raw_artifact_ids else "Live run trace.",
                "trace": trace.model_dump(mode="json"),
            }

        return {"session_id": session_id, "status": "not_found"}

    # ------------------------------------------------------------------ #
    # MCP & Hosts                                                         #
    # ------------------------------------------------------------------ #

    @app.get("/mcp/status", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def mcp_status() -> list[dict[str, Any]]:
        try:
            from lemoncrow.gateway.adapters.mcp_server import (
                TOOLS,
                _tool_description,
                _tool_mode,
                _tool_visible_to_llm,
            )

            return [
                {
                    "tool_name": name,
                    "available": True,
                    "description": _tool_description(spec),
                    "is_dev": bool(spec.get("is_dev")),
                    "mode": _tool_mode(spec),
                    "enum_params": _extract_enum_params(cast(dict[str, Any], spec.get("inputSchema") or {})),
                }
                for name, spec in TOOLS.items()
                if _tool_visible_to_llm(name, spec)
            ]
        except Exception as exc:
            logging.exception("Recovered from broad exception handler")
            logger.warning("Failed to load MCP tool status: %s", exc, exc_info=True)
            return []

    @app.get("/hosts", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def list_hosts() -> list[dict[str, Any]]:
        import yaml

        store = get_store()
        seen_hosts = set(store.history.get_traces_metrics()["hosts"])
        import_stats = _host_import_stats(store)
        root = Path(__file__).parent.parent.parent.parent.parent
        configs_dir = root / "src" / "lemoncrow" / "gateway" / "hosts" / "configs"

        hosts: list[dict[str, Any]] = []
        for host_id in _HOST_ORDER:
            config_path = configs_dir / f"{host_id}.yaml"
            payload: dict[str, Any] = {}
            if config_path.exists():
                loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                if isinstance(loaded, dict):
                    payload = loaded

            install_script = root / "scripts" / f"install_{host_id}.sh"
            label = _HOST_LABEL_OVERRIDES.get(host_id) or str(payload.get("name") or host_id)
            description = _HOST_DESCRIPTION_OVERRIDES.get(host_id) or str(payload.get("description") or "")
            if host_id == "hermes":
                label = "Hermes Agent (global-only)"
            stats = import_stats.get(host_id, {})
            hosts.append(
                {
                    "host_id": host_id,
                    "label": label,
                    "status": "active" if host_id in seen_hosts else "not_detected",
                    "active_domains": [],
                    "mcp_tools": [],
                    # Real per-host last-import time + count, not host
                    # detection -- see _host_import_stats()'s docstring for
                    # why RawArtifact.created_at (not Trace.created_at) is
                    # the honest field here.
                    "last_import_at": stats.get("last_import_at"),
                    "imported_session_count": int(stats.get("imported_session_count") or 0),
                    "lemoncrow_version": None,
                    "description": description or None,
                    "install_command": (f"bash scripts/{install_script.name}" if install_script.exists() else None),
                }
            )
        return hosts

    # ------------------------------------------------------------------ #
    # Skills                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/skills", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def list_skills() -> list[dict[str, Any]]:
        from lemoncrow.core.environment import skill_visible

        root = Path(__file__).parent.parent.parent.parent.parent
        skills: list[dict[str, Any]] = []
        skills_dir = root / "integrations" / "skills"
        if skills_dir.exists():
            for skill_dir in sorted(skills_dir.iterdir()):
                if skill_dir.is_dir():
                    if not skill_visible(skill_dir.name):
                        continue
                    md = skill_dir / "SKILL.md"
                    if md.exists():
                        content = md.read_text(encoding="utf-8")
                        desc = ""
                        if content.startswith("---"):
                            end = content.find("---", 3)
                            if end > 0:
                                for line in content[3:end].split("\n"):
                                    if line.startswith("description:"):
                                        desc = line.split(":", 1)[1].strip()
                        skills.append({"name": skill_dir.name, "description": desc, "content": content})
        return skills

    @app.get("/skills/{name}", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def get_skill(name: str) -> dict[str, Any]:
        from lemoncrow.core.environment import skill_visible

        # skill_visible() is a denylist, so it is not a path guard: without
        # safe_segment a `name` of `../../..` reads any SKILL.md on the box and
        # doubles as a filesystem existence oracle (404 vs 200).
        try:
            name = safe_segment(name, field="skill name")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Skill not found") from exc
        if not skill_visible(name):
            raise HTTPException(status_code=404, detail=f"Skill is hidden from the public host surface: {name}")
        root = Path(__file__).parent.parent.parent.parent.parent
        skills_dir = (root / "integrations" / "skills").resolve()
        try:
            md = confine_to_root(skills_dir / name / "SKILL.md", skills_dir)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=f"Skill not found: {name}") from exc
        if not md.exists():
            raise HTTPException(status_code=404, detail=f"Skill not found: {name}")
        content = md.read_text(encoding="utf-8")
        return {"name": name, "description": "", "content": content}

    # ------------------------------------------------------------------ #
    # Agents                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/agents", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def list_agents() -> list[dict[str, Any]]:
        import re

        import yaml

        root = Path(__file__).parent.parent.parent.parent.parent
        agents_dir = root / "integrations" / "claude" / "plugin" / "agents"
        result: list[dict[str, Any]] = []
        if not agents_dir.exists():
            return result
        for path in sorted(agents_dir.glob("*.md")):
            if path.name.endswith(".dev.md"):
                continue
            text = path.read_text(encoding="utf-8")
            # Parse YAML frontmatter between first two --- markers
            fm: dict[str, Any] = {}
            body = text
            if text.startswith("---"):
                end = text.find("---", 3)
                if end > 0:
                    try:
                        fm = yaml.safe_load(text[3:end]) or {}
                    except Exception:
                        logging.exception("Recovered from broad exception handler")
                        fm = {}
                    body = text[end + 3 :].strip()
            # Strip generator comment from body
            body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL).strip()
            tools_raw = fm.get("tools", [])
            if isinstance(tools_raw, str):
                tools_raw = [tools_raw]
            result.append(
                {
                    "id": path.stem,
                    "name": fm.get("name", path.stem),
                    "description": fm.get("description", ""),
                    "tools": tools_raw,
                    "color": fm.get("color", "neutral"),
                    "model": fm.get("model"),
                    "file": str(path.relative_to(root)),
                    "content": body,
                }
            )
        return result

    # ------------------------------------------------------------------ #
    # Rubrics                                                             #
    # ------------------------------------------------------------------ #

    @app.get("/v1/rubrics", tags=["rubrics"], dependencies=[Depends(verify_api_key)])
    def list_rubrics(domain: str | None = Query(None)) -> list[dict[str, Any]]:
        return [to_jsonable(r) for r in get_store().knowledge.list_rubrics(domain=domain)]

    @app.get("/v1/rubrics/{rubric_id}", tags=["rubrics"], dependencies=[Depends(verify_api_key)])
    def get_rubric(rubric_id: str) -> dict[str, Any]:
        rubric = get_store().knowledge.get_rubric(rubric_id)
        if rubric is None:
            raise HTTPException(status_code=404, detail=f"Rubric not found: {rubric_id}")
        return to_jsonable(rubric)

    # ------------------------------------------------------------------ #
    # Blocks (compat)                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/blocks", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_blocks() -> list[dict[str, Any]]:
        return [to_jsonable(b) for b in get_store().knowledge.list_blocks()]

    @app.get("/blocks/{block_id}", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_block(block_id: str) -> dict[str, Any]:
        block = get_store().knowledge.get_block(block_id)
        if block is None:
            raise HTTPException(status_code=404, detail=f"Block not found: {block_id}")
        return to_jsonable(block)

    # ------------------------------------------------------------------ #
    # Savings & Calls (compat)                                            #
    # ------------------------------------------------------------------ #

    @app.get("/savings", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_savings() -> dict[str, Any]:
        from lemoncrow.infra.runtime.cost_tracker import CostTracker

        return CostTracker(Path(cfg.lemoncrow_root)).total_savings()

    @app.get("/v1/savings/summary", tags=["metrics"], dependencies=[Depends(verify_api_key)])
    def savings_summary_v1(window_days: int = Query(14)) -> dict[str, Any]:
        return _savings_summary_payload(store_path, window_days=window_days, store=get_store())

    @app.get("/v1/optimizations/summary", tags=["metrics"], dependencies=[Depends(verify_api_key)])
    def optimizations_summary(window_days: int = Query(14)) -> dict[str, Any]:
        return _optimizations_summary_payload(Path(cfg.lemoncrow_root), get_store(), window_days=window_days)

    @app.get("/calls", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_calls(limit: int = Query(200)) -> list[dict[str, Any]]:
        from lemoncrow.infra.runtime.cost_tracker import load_cost_history

        history = load_cost_history(Path(cfg.lemoncrow_root))
        ops = history.get("operations", {})
        all_calls: list[dict[str, Any]] = []
        for op_key, entry in ops.items():
            for call in entry.get("calls", []):
                all_calls.append(
                    {
                        "session_id": call.get("session_id", ""),
                        "domain": entry.get("domain"),
                        "task": entry.get("task_sample"),
                        "operation": call.get("operation"),
                        "model": call.get("model"),
                        "input_tokens": call.get("input_tokens"),
                        "output_tokens": call.get("output_tokens"),
                        "cache_read_tokens": call.get("cache_read_tokens"),
                        "cost_usd": call.get("cost_usd"),
                        "lessons_used": call.get("lessons_used"),
                        "op_key": op_key,
                        "at": call.get("at"),
                    }
                )
        all_calls.sort(key=lambda c: c.get("at") or "", reverse=True)
        return all_calls[:limit]

    # ------------------------------------------------------------------ #
    # Clusters (compat)                                                   #
    # ------------------------------------------------------------------ #

    @app.get("/clusters", tags=["compat"], dependencies=[Depends(verify_api_key)])
    def compat_clusters() -> list[dict[str, Any]]:
        return []

    # ------------------------------------------------------------------ #
    # Watchdogs                                                           #
    # ------------------------------------------------------------------ #

    @app.get("/watchdogs/config", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def get_watchdog_config() -> dict[str, Any]:
        from lemoncrow.core.foundation.watchdog_profiles import frontend_watchdog_profile_config

        return frontend_watchdog_profile_config(Path(cfg.lemoncrow_root))

    @app.post("/watchdogs/config", tags=["ops"], dependencies=[Depends(verify_api_key)])
    def update_watchdog_config(payload: dict[str, Any]) -> dict[str, Any]:
        from lemoncrow.core.foundation.watchdog_profiles import (
            frontend_watchdog_profile_config,
            save_watchdog_profile_config,
        )

        try:
            save_watchdog_profile_config(
                Path(cfg.lemoncrow_root),
                active_profile=payload.get("active_profile"),
                profiles=payload.get("profiles"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return frontend_watchdog_profile_config(Path(cfg.lemoncrow_root))

    # ------------------------------------------------------------------ #
    # Memory passages                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/v1/memory/passages", tags=["knowledge"], dependencies=[Depends(verify_api_key)])
    def list_memory_passages(
        agent_id: str = Query(...),
        limit: int = Query(25, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        from lemoncrow.infra.storage.factory import make_memory_store

        mem = make_memory_store(Path(cfg.lemoncrow_root))
        passages = mem.list_passages(agent_id, limit=limit)
        return [p.model_dump(mode="json") for p in passages]

    # ------------------------------------------------------------------ #
    # Week-2 routes — Sessions, Memory facts, Insights, Outcomes,        #
    # Reports (spec 06)                                                   #
    # ------------------------------------------------------------------ #

    def _parse_session_datetime(value: Any) -> datetime | None:
        if isinstance(value, (int, float)):
            with contextlib.suppress(OSError, OverflowError, ValueError):
                return datetime.fromtimestamp(float(value) / 1000, tz=UTC)
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                with contextlib.suppress(OSError, OverflowError, ValueError):
                    return datetime.fromtimestamp(int(stripped) / 1000, tz=UTC)
                return None
            with contextlib.suppress(ValueError):
                parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                # Offset-less ISO timestamps (no "Z"/"+HH:MM") parse naive; treat
                # them as UTC so comparisons against aware datetimes never raise.
                return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return None

    def _conversation_sort_key(turn: dict[str, Any]) -> tuple[int, float, str]:
        parsed = _parse_session_datetime(turn.get("at"))
        if parsed is not None:
            return (0, parsed.timestamp(), "")
        raw_at = turn.get("at")
        if isinstance(raw_at, str):
            return (1, 0.0, raw_at.strip())
        return (2, 0.0, "")

    _TRACE_LOOKUP_NEGATIVE_TTL_SECONDS = 20.0
    _trace_lookup_negative_cache: dict[str, float] = {}
    _trace_lookup_negative_cache_lock = threading.Lock()

    def _load_trace_for_session(session_id: str) -> Trace | None:
        store = get_store()

        now = time.monotonic()
        with _trace_lookup_negative_cache_lock:
            expiry = _trace_lookup_negative_cache.get(session_id)
        if expiry is not None:
            if expiry > now:
                return None
            with _trace_lookup_negative_cache_lock:
                _trace_lookup_negative_cache.pop(session_id, None)

        from lemoncrow.gateway.hosts.session_parsers.registry import (
            SUPPORTED_SESSION_IMPORT_HOSTS,
        )

        # Direct id + every host-prefixed variant in ONE indexed lookup instead
        # of N sequential get_trace() calls (each of which also runs its own
        # json_extract fallback scan on miss).
        candidate_ids = [session_id, *(f"{host}-{session_id}" for host in SUPPORTED_SESSION_IMPORT_HOSTS)]
        placeholders = ",".join("?" for _ in candidate_ids)
        with sqlite3.connect(store.history.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, payload FROM traces WHERE id IN ({placeholders})",
                candidate_ids,
            ).fetchall()
            by_id = {row[0]: row[1] for row in rows}
            payload = next((by_id[cid] for cid in candidate_ids if cid in by_id), None)
            if payload is None:
                # Slower fallback: session_id embedded in the payload rather
                # than the trace id itself. Relies on the traces json_extract
                # index for performance.
                row = conn.execute(
                    "SELECT payload FROM traces WHERE json_extract(payload, '$.session_id') = ?",
                    (session_id,),
                ).fetchone()
                payload = row[0] if row else None

        if payload is None:
            with _trace_lookup_negative_cache_lock:
                if len(_trace_lookup_negative_cache) > 2000:
                    _trace_lookup_negative_cache.clear()
                _trace_lookup_negative_cache[session_id] = now + _TRACE_LOOKUP_NEGATIVE_TTL_SECONDS
            return None
        return Trace.model_validate_json(coerce_trace_json(payload))

    def _reconstruct_trace_conversations(
        session_id: str,
        trace: Trace,
        store_inst: StoreBundle | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        store_inst = store_inst or get_store()
        conversations: list[dict[str, Any]] = []
        source_files: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        seen_source_files: set[tuple[str, str]] = set()
        raw_usage_summary: dict[str, Any] = {
            "started_model": None,
            "models_used": {},
            "total_turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "thinking_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "usage_entries": [],
        }

        if not trace.raw_artifact_ids:
            return conversations, source_files, artifacts, raw_usage_summary

        from lemoncrow.gateway.hosts.session_parsers._session_parser import (
            _extract_claude_session_id,
            attach_lemoncrow_sidecar_savings,
            extract_session_usage_summary,
            parse_session_turns,
        )

        for art_id in trace.raw_artifact_ids:
            artifact = store_inst.history.get_raw_artifact(art_id)
            if artifact is None:
                continue
            scope = "subagent" if "subagents/" in str(artifact.relative_path).replace("\\", "/") else "main"
            if artifact.source_path:
                source_file_key = (artifact.source_path, artifact.id)
                if source_file_key not in seen_source_files:
                    source_files.append({"path": artifact.source_path, "artifact_id": artifact.id})
                    seen_source_files.add(source_file_key)
            try:
                # Single read of the raw artifact backs both the turn-by-turn
                # conversation reconstruction and the usage summary below —
                # avoids parsing every transcript twice per request.
                raw_content = store_inst.history.read_raw_artifact_content(artifact)
                artifact_turns = parse_session_turns(raw_content, artifact.source)
                # Join host sidecar (real per-call savings written by the MCP
                # server) onto LemonCrow MCP tool turns. Claude Code strips the
                # in-response `saved` block when persisting, so this is the
                # only reliable source for per-tool savings in the UI.
                if artifact.source == "claude":
                    artifact_session_id = _extract_claude_session_id(raw_content)
                    if artifact_session_id:
                        from lemoncrow.core.foundation.paths import default_store_root

                        attach_lemoncrow_sidecar_savings(artifact_turns, artifact_session_id, default_store_root())
                artifact_label = "main"
                if scope == "subagent":
                    artifact_label = (
                        next(
                            (
                                str(turn.get("subagent_name") or "").strip()
                                for turn in artifact_turns
                                if str(turn.get("subagent_name") or "").strip()
                            ),
                            "",
                        )
                        or Path(artifact.relative_path).stem
                    )
                artifacts.append(
                    {
                        "id": artifact.id,
                        "source": artifact.source,
                        "kind": artifact.kind,
                        "relative_path": artifact.relative_path,
                        "source_path": artifact.source_path,
                        "scope": scope,
                        "label": artifact_label,
                    }
                )
                for turn in artifact_turns:
                    turn["artifact_id"] = artifact.id
                    turn["artifact_source"] = artifact.source
                    turn["artifact_kind"] = artifact.kind
                    turn["artifact_label"] = artifact_label
                    turn["source_scope"] = scope
                    if scope == "subagent" and not turn.get("subagent_name"):
                        turn["subagent_name"] = artifact_label
                conversations.extend(artifact_turns)

                try:
                    summary = extract_session_usage_summary(raw_content, artifact.source)
                except Exception as usage_exc:
                    logger.error(
                        "Failed to summarize imported artifact %s for session %s: %s",
                        art_id,
                        session_id,
                        usage_exc,
                        exc_info=True,
                    )
                else:
                    if raw_usage_summary["started_model"] is None and summary.get("started_model"):
                        raw_usage_summary["started_model"] = summary["started_model"]
                    for key in (
                        "total_turns",
                        "input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                        "thinking_tokens",
                        "cached_input_tokens",
                        "cache_creation_input_tokens",
                    ):
                        raw_usage_summary[key] += int(summary.get(key, 0) or 0)
                    for model_id, count in dict(summary.get("models_used") or {}).items():
                        raw_usage_summary["models_used"][model_id] = int(
                            raw_usage_summary["models_used"].get(model_id, 0)
                        ) + int(count or 0)
                    raw_usage_summary["usage_entries"].extend(list(summary.get("usage_entries") or []))
            except Exception as exc:
                artifacts.append(
                    {
                        "id": artifact.id,
                        "source": artifact.source,
                        "kind": artifact.kind,
                        "relative_path": artifact.relative_path,
                        "source_path": artifact.source_path,
                        "scope": scope,
                        "label": "main" if scope == "main" else Path(artifact.relative_path).stem,
                    }
                )
                logger.error(
                    "Failed to reconstruct conversations from artifact %s for session %s: %s",
                    art_id,
                    session_id,
                    exc,
                    exc_info=True,
                )

        conversations.sort(key=_conversation_sort_key)
        return conversations, source_files, artifacts, raw_usage_summary

    def _imported_session_fingerprint(
        session_id: str,
        trace: Trace,
        store_inst: StoreBundle,
        root: Path | None,
        artifact_fp_lookup: dict[str, tuple[str, int]] | None = None,
    ) -> tuple[Any, ...]:
        """Cheap staleness signal for the cache below.

        Raw artifact content is immutable once ingested (content-addressed by
        sha256), so per-artifact (mtime, size) never changes for a given
        artifact id without a new id being appended — no need to read/parse
        the transcript to detect a stale entry. The savings sidecars feeding
        total_lemoncrow_savings_usd are stat()'d too so that number stays live.

        ``artifact_fp_lookup``, when given, is a pre-batched
        (mtime, size)-by-artifact-id map (see _bulk_raw_artifact_fingerprints)
        so callers iterating many sessions can fetch every artifact's
        fingerprint in one query instead of one store.get_raw_artifact() call
        — and hence one fresh sqlite3 connection — per artifact.
        """
        artifact_fp: list[tuple[str, int]] = []
        for art_id in trace.raw_artifact_ids:
            if artifact_fp_lookup is not None:
                artifact_fp.append(artifact_fp_lookup.get(art_id, ("", -1)))
                continue
            artifact = store_inst.history.get_raw_artifact(art_id)
            if artifact is None:
                artifact_fp.append(("", -1))
                continue
            mtime = artifact.source_file_mtime.isoformat() if artifact.source_file_mtime else ""
            artifact_fp.append((mtime, artifact.byte_count_original))

        def _stat_fp(path: Path) -> tuple[float, int]:
            try:
                st = path.stat()
            except OSError:
                return (0.0, -1)
            return (st.st_mtime, st.st_size)

        savings_fp = (0.0, -1)
        sidecar_fp = (0.0, -1)
        if root is not None:
            from lemoncrow.core.foundation.paths import find_session_dir

            savings_fp = _stat_fp(root / "live_savings_events.jsonl")
            session_root = find_session_dir(root, session_id) or flat_session_dir(root, session_id)
            if session_root is not None:
                sidecar_fp = _stat_fp(session_root / "savings.jsonl")
        return (trace.id, tuple(artifact_fp), savings_fp, sidecar_fp)

    _imported_session_payload_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    _imported_session_payload_cache_lock = threading.Lock()

    def _build_imported_session_payload(
        session_id: str,
        trace: Trace,
        store_inst: StoreBundle | None = None,
        root: Path | None = None,
        artifact_fp_lookup: dict[str, tuple[str, int]] | None = None,
    ) -> dict[str, Any]:
        from lemoncrow.infra.runtime.session_report import (
            _derive_vendor,
            read_total_savings_from_events,
        )

        def _prefer_positive_int(authoritative: int, fallback: int) -> int:
            return authoritative if authoritative > 0 else fallback

        def _prefer_positive_float(authoritative: float, fallback: float) -> float:
            return authoritative if authoritative > 0 else fallback

        store_inst = store_inst or get_store()

        # Unchanged sessions cost zero parsing on re-poll: list_sessions is
        # hit by the frontend every ~30s, and re-parsing the full raw
        # transcript each time is the dominant cost of that endpoint.
        fingerprint = _imported_session_fingerprint(session_id, trace, store_inst, root, artifact_fp_lookup)
        with _imported_session_payload_cache_lock:
            cached = _imported_session_payload_cache.get(session_id)
        if cached is not None and cached[0] == fingerprint:
            return dict(cached[1])

        conversations, _, _, raw_usage_summary = _reconstruct_trace_conversations(session_id, trace, store_inst)
        trace_payload = to_jsonable(trace)
        trace_usage_entries = _trace_usage_entries(trace_payload)
        trace_model_usages = _trace_model_usages(trace_payload)

        # Seed from the real conversation timestamps, not trace.created_at:
        # that field is when this trace record was written (import/ingest
        # time), which — for a session imported after it finished — is at or
        # after every real turn. Seeding both bounds with it and only ever
        # raising ended_at (never lowering it) pins ended_at at import time
        # forever even though the true last turn is earlier. Track the real
        # min/max across turns instead and fall back to trace.created_at only
        # when no turn has a parseable timestamp at all.
        started_at: datetime | None = None
        ended_at: datetime | None = None
        active_duration_seconds = 0.0
        current_active_start: datetime | None = None

        reconstructed_models_used: dict[str, int] = {}
        reconstructed_started_model: str | None = None
        reconstructed_input_tokens = 0
        reconstructed_output_tokens = 0
        reconstructed_cache_read_tokens = 0
        reconstructed_cache_write_tokens = 0
        reconstructed_input_token_cost_usd = 0.0
        reconstructed_output_token_cost_usd = 0.0
        reconstructed_cache_read_cost_usd = 0.0
        reconstructed_cache_write_cost_usd = 0.0
        reconstructed_total_turns = 0
        tool_costs: dict[str, dict[str, float]] = {}

        for turn in conversations:
            parsed_at = _parse_session_datetime(turn.get("at"))
            if parsed_at is not None:
                if started_at is None or parsed_at < started_at:
                    started_at = parsed_at
                if ended_at is None or parsed_at > ended_at:
                    ended_at = parsed_at
                if turn.get("kind") == "user_message":
                    current_active_start = parsed_at
                elif current_active_start is not None:
                    chunk = (parsed_at - current_active_start).total_seconds()
                    if 0 < chunk < 3600:
                        active_duration_seconds += chunk
                    current_active_start = parsed_at

            tokens = turn.get("tokens") or {}
            turn_input = int(tokens.get("in") or 0)
            turn_output = int(tokens.get("out") or 0)
            turn_cache_read = int(tokens.get("cache_read") or 0)
            turn_cache_write = int(tokens.get("cache_write") or 0)
            if turn_input or turn_output or turn_cache_read or turn_cache_write:
                reconstructed_total_turns += 1

            model_id = str(turn.get("model") or "").strip()
            countable_model_id = (
                model_id if model_id and model_id.lower() not in {"<synthetic>", "_default", "unknown"} else ""
            )
            if countable_model_id:
                if reconstructed_started_model is None:
                    reconstructed_started_model = countable_model_id
                reconstructed_models_used[countable_model_id] = reconstructed_models_used.get(countable_model_id, 0) + 1

            turn_input_cost = _llm_usage_cost(model_id or "_default", input_tokens=turn_input)
            turn_output_cost = _llm_usage_cost(model_id or "_default", output_tokens=turn_output)
            turn_cache_read_cost = _llm_usage_cost(model_id or "_default", cache_read_tokens=turn_cache_read)
            turn_cache_write_cost = _llm_usage_cost(model_id or "_default", cache_write_tokens=turn_cache_write)
            turn_total_cost = turn_input_cost + turn_output_cost + turn_cache_read_cost + turn_cache_write_cost
            turn["cost"] = round(turn_total_cost, 8)

            reconstructed_input_tokens += turn_input
            reconstructed_output_tokens += turn_output
            reconstructed_cache_read_tokens += turn_cache_read
            reconstructed_cache_write_tokens += turn_cache_write
            reconstructed_input_token_cost_usd += turn_input_cost
            reconstructed_output_token_cost_usd += turn_output_cost
            reconstructed_cache_read_cost_usd += turn_cache_read_cost
            reconstructed_cache_write_cost_usd += turn_cache_write_cost

            if turn_total_cost > 0:
                tool_name = str(
                    turn.get("tool_name")
                    or turn.get("subagent_name")
                    or (
                        "assistant"
                        if turn.get("kind") == "agent_message"
                        else "shell" if turn.get("kind") == "shell_command" else turn.get("kind") or "session"
                    )
                )
                bucket = tool_costs.setdefault(tool_name, {"calls": 0.0, "cost_usd": 0.0})
                bucket["calls"] += 1
                bucket["cost_usd"] += turn_total_cost

        if started_at is None:
            started_at = trace.created_at
        if ended_at is None:
            ended_at = trace.created_at
        duration_seconds = max(0.0, (ended_at - started_at).total_seconds())
        if active_duration_seconds <= 0:
            active_duration_seconds = duration_seconds

        def _usage_cost_breakdown(
            usage_entries: list[dict[str, Any]],
            *,
            fallback_model: str | None,
        ) -> dict[str, float]:
            input_cost = 0.0
            output_cost = 0.0
            cache_read_cost = 0.0
            cache_write_cost = 0.0
            for entry in usage_entries:
                model_id = str(entry.get("model") or fallback_model or "_default")
                input_cost += _llm_usage_cost(model_id, input_tokens=int(entry.get("input_tokens") or 0))
                output_cost += _llm_usage_cost(model_id, output_tokens=int(entry.get("output_tokens") or 0))
                cache_read_cost += _llm_usage_cost(
                    model_id, cache_read_tokens=int(entry.get("cached_input_tokens") or 0)
                )
                cache_write_cost += _llm_usage_cost(
                    model_id,
                    cache_write_tokens=int(entry.get("cache_creation_input_tokens") or 0),
                )
            return {
                "input_token_cost_usd": input_cost,
                "output_token_cost_usd": output_cost,
                "cache_read_cost_usd": cache_read_cost,
                "cache_write_cost_usd": cache_write_cost,
                "total_cost_usd": input_cost + output_cost + cache_read_cost + cache_write_cost,
            }

        trace_started_model = (
            _trace_session_model(
                trace_payload,
                trace_usage_entries,
                trace_model_usages,
            )
            or None
        )
        trace_models_used = _trace_models_used_from_payload(
            trace_payload,
            trace_usage_entries,
            trace_model_usages,
        )
        trace_input_tokens = int(trace_payload.get("input_tokens") or 0)
        trace_output_tokens = int(trace_payload.get("output_tokens") or 0)
        trace_cache_read_tokens = int(trace_payload.get("cached_input_tokens") or 0)
        trace_cache_write_tokens = int(trace_payload.get("cache_creation_input_tokens") or 0)
        trace_total_turns = len([entry for entry in trace_usage_entries if entry.get("kind") != "tool"])
        if trace_total_turns <= 0 and (
            trace_input_tokens > 0
            or trace_output_tokens > 0
            or trace_cache_read_tokens > 0
            or trace_cache_write_tokens > 0
            or bool(trace_models_used)
        ):
            trace_total_turns = 1

        trace_cost_breakdown = _trace_cost_breakdown_from_payload(trace_payload)
        trace_input_token_cost_usd = float(trace_cost_breakdown["input_token_cost_usd"])
        trace_output_token_cost_usd = float(trace_cost_breakdown["output_token_cost_usd"])
        trace_cache_read_cost_usd = float(trace_cost_breakdown["cache_read_cost_usd"])
        trace_cache_write_cost_usd = float(trace_cost_breakdown["cache_write_cost_usd"])
        trace_total_cost_usd = _trace_cost_from_payload(trace_payload)
        if trace_total_cost_usd <= 0 and (
            trace_input_tokens > 0
            or trace_output_tokens > 0
            or trace_cache_read_tokens > 0
            or trace_cache_write_tokens > 0
        ):
            pricing_model = trace_started_model or reconstructed_started_model or trace.model or "_default"
            trace_input_token_cost_usd = _llm_usage_cost(pricing_model, input_tokens=trace_input_tokens)
            trace_output_token_cost_usd = _llm_usage_cost(pricing_model, output_tokens=trace_output_tokens)
            trace_cache_read_cost_usd = _llm_usage_cost(pricing_model, cache_read_tokens=trace_cache_read_tokens)
            trace_cache_write_cost_usd = _llm_usage_cost(pricing_model, cache_write_tokens=trace_cache_write_tokens)
            trace_total_cost_usd = round(
                trace_input_token_cost_usd
                + trace_output_token_cost_usd
                + trace_cache_read_cost_usd
                + trace_cache_write_cost_usd,
                8,
            )

        authoritative_started_model = str(raw_usage_summary.get("started_model") or "").strip() or None
        authoritative_models_used = dict(raw_usage_summary.get("models_used") or {})
        authoritative_input_tokens = int(raw_usage_summary.get("input_tokens") or 0)
        authoritative_output_tokens = int(raw_usage_summary.get("output_tokens") or 0)
        authoritative_cache_read_tokens = int(raw_usage_summary.get("cached_input_tokens") or 0)
        authoritative_cache_write_tokens = int(raw_usage_summary.get("cache_creation_input_tokens") or 0)
        authoritative_total_turns = int(raw_usage_summary.get("total_turns") or 0)
        has_authoritative_usage = (
            authoritative_total_turns > 0
            or authoritative_input_tokens > 0
            or authoritative_output_tokens > 0
            or authoritative_cache_read_tokens > 0
            or authoritative_cache_write_tokens > 0
            or bool(authoritative_models_used)
        )
        authoritative_cost_breakdown = _usage_cost_breakdown(
            list(raw_usage_summary.get("usage_entries") or []),
            fallback_model=authoritative_started_model
            or trace_started_model
            or reconstructed_started_model
            or trace.model,
        )

        started_model = (
            authoritative_started_model or reconstructed_started_model or trace_started_model or (trace.model or None)
        )
        models_used = authoritative_models_used or reconstructed_models_used or trace_models_used
        if not models_used and started_model:
            models_used = {started_model: 1}

        input_tokens = (
            authoritative_input_tokens
            if has_authoritative_usage
            else _prefer_positive_int(trace_input_tokens, reconstructed_input_tokens)
        )
        output_tokens = (
            authoritative_output_tokens
            if has_authoritative_usage
            else _prefer_positive_int(trace_output_tokens, reconstructed_output_tokens)
        )
        cache_read_tokens = (
            authoritative_cache_read_tokens
            if has_authoritative_usage
            else _prefer_positive_int(trace_cache_read_tokens, reconstructed_cache_read_tokens)
        )
        cache_write_tokens = (
            authoritative_cache_write_tokens
            if has_authoritative_usage
            else _prefer_positive_int(trace_cache_write_tokens, reconstructed_cache_write_tokens)
        )
        total_turns = (
            authoritative_total_turns
            if authoritative_total_turns > 0
            else trace_total_turns if trace_total_turns > 0 else reconstructed_total_turns
        )

        input_token_cost_usd = (
            float(authoritative_cost_breakdown["input_token_cost_usd"])
            if has_authoritative_usage
            else _prefer_positive_float(trace_input_token_cost_usd, reconstructed_input_token_cost_usd)
        )
        output_token_cost_usd = (
            float(authoritative_cost_breakdown["output_token_cost_usd"])
            if has_authoritative_usage
            else _prefer_positive_float(trace_output_token_cost_usd, reconstructed_output_token_cost_usd)
        )
        cache_read_cost_usd = (
            float(authoritative_cost_breakdown["cache_read_cost_usd"])
            if has_authoritative_usage
            else _prefer_positive_float(trace_cache_read_cost_usd, reconstructed_cache_read_cost_usd)
        )
        cache_write_cost_usd = (
            float(authoritative_cost_breakdown["cache_write_cost_usd"])
            if has_authoritative_usage
            else _prefer_positive_float(trace_cache_write_cost_usd, reconstructed_cache_write_cost_usd)
        )
        total_cost_usd = (
            round(float(authoritative_cost_breakdown["total_cost_usd"]), 6)
            if has_authoritative_usage
            else _prefer_positive_float(
                trace_total_cost_usd,
                round(
                    reconstructed_input_token_cost_usd
                    + reconstructed_output_token_cost_usd
                    + reconstructed_cache_read_cost_usd
                    + reconstructed_cache_write_cost_usd,
                    6,
                ),
            )
        )

        trace_tool_costs = [
            {
                "tool": str(tool.name or "unknown"),
                "calls": int(tool.count or 1),
                "cost_usd": round(
                    _analytics_event_cost(
                        started_model or "_default",
                        "tool_call",
                        int(tool.input_tokens or 0),
                        int(tool.output_tokens or 0),
                    ),
                    6,
                ),
            }
            for tool in trace.tools_called
        ]
        top_tools_by_cost = sorted(
            (
                trace_tool_costs
                if trace_tool_costs
                else [
                    {
                        "tool": name,
                        "calls": int(values["calls"]),
                        "cost_usd": round(values["cost_usd"], 6),
                    }
                    for name, values in tool_costs.items()
                ]
            ),
            key=lambda item: cast(float, item["cost_usd"]),
            reverse=True,
        )[:5]

        result: dict[str, Any] = {
            "session_id": session_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": duration_seconds,
            "active_duration_seconds": active_duration_seconds,
            "vendor": _derive_vendor(models_used),
            "agent_settings": trace.agent_settings,
            "skills": trace.skills,
            "telemetry": trace.telemetry,
            "raw_artifact_ids": trace.raw_artifact_ids,
            "total_turns": total_turns,
            "total_cost_usd": total_cost_usd,
            "total_lemoncrow_savings_usd": (
                read_total_savings_from_events(session_id, root) if root is not None else 0.0
            ),
            "label": None,
            "models_used": models_used,
            "started_model": started_model,
            "cost_status": (
                "estimated"
                if (
                    total_turns > 0
                    or input_tokens > 0
                    or output_tokens > 0
                    or cache_read_tokens > 0
                    or cache_write_tokens > 0
                )
                else "unavailable"
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cache_read_tokens,
            "tool_call_count": sum(int(tool.count or 1) for tool in trace.tools_called),
            "input_token_cost_usd": round(input_token_cost_usd, 6),
            "cache_write_cost_usd": round(cache_write_cost_usd, 6),
            "cache_read_cost_usd": round(cache_read_cost_usd, 6),
            "output_token_cost_usd": round(output_token_cost_usd, 6),
            "cache_write_tokens": cache_write_tokens,
            "cache_read_tokens": cache_read_tokens,
            "routing_downtiered_turns": 0,
            "routing_savings_usd": 0.0,
            "compact_events": 0,
            "compact_savings_estimate_usd": 0.0,
            "context_compression_savings_usd": 0.0,
            "context_compression_tool_calls": 0,
            "tool_savings": [],
            "top_tools_by_cost": top_tools_by_cost,
        }
        with _imported_session_payload_cache_lock:
            if len(_imported_session_payload_cache) > 2000:
                _imported_session_payload_cache.clear()
            _imported_session_payload_cache[session_id] = (fingerprint, dict(result))
        return result

    def _build_session_payload(report: Any, trace: Trace | None = None) -> dict[str, Any]:
        active_trace = trace or _load_trace_for_session(report.session_id)
        estimated_payload = (
            _build_imported_session_payload(report.session_id, active_trace) if active_trace is not None else None
        )

        models_used = dict(report.models_used) or dict((estimated_payload or {}).get("models_used", {}))
        started_model = report.started_model or (estimated_payload or {}).get("started_model")

        input_tokens = int(report.input_tokens or (estimated_payload or {}).get("input_tokens", 0))
        output_tokens = int(report.output_tokens or (estimated_payload or {}).get("output_tokens", 0))
        cache_read_tokens = int(report.cache_read_tokens or (estimated_payload or {}).get("cache_read_tokens", 0))
        cache_write_tokens = int(report.cache_write_tokens or (estimated_payload or {}).get("cache_write_tokens", 0))
        duration_seconds = float(report.duration_seconds or (estimated_payload or {}).get("duration_seconds", 0.0))
        active_duration_seconds = float(
            report.active_duration_seconds or (estimated_payload or {}).get("active_duration_seconds", 0.0)
        )
        total_cost_usd = float(report.total_cost_usd)
        input_token_cost_usd = float(report.input_token_cost_usd)
        output_token_cost_usd = float(report.output_token_cost_usd)
        cache_read_cost_usd = float(report.cache_read_cost_usd)
        cache_write_cost_usd = float(report.cache_write_cost_usd)
        top_tools_by_cost = [{"tool": t, "calls": c, "cost_usd": v} for t, c, v in report.top_tools_by_cost]

        cost_status = (
            "recorded" if (total_cost_usd > 0 or report.total_turns > 0 or bool(report.models_used)) else "unavailable"
        )
        if report.cost_estimated and total_cost_usd > 0:
            # build_report backfilled the total from pricing-derived buckets
            # because the ledger recorded no cost -- that is an estimate.
            cost_status = "estimated"

        if estimated_payload is not None:
            if total_cost_usd <= 0 and float(estimated_payload["total_cost_usd"]) > 0:
                total_cost_usd = float(estimated_payload["total_cost_usd"])
                input_token_cost_usd = float(estimated_payload["input_token_cost_usd"])
                output_token_cost_usd = float(estimated_payload["output_token_cost_usd"])
                cache_read_cost_usd = float(estimated_payload["cache_read_cost_usd"])
                cache_write_cost_usd = float(estimated_payload["cache_write_cost_usd"])
                top_tools_by_cost = list(estimated_payload["top_tools_by_cost"])
                cost_status = "estimated"
            elif not top_tools_by_cost:
                top_tools_by_cost = list(estimated_payload["top_tools_by_cost"])
            if not started_model:
                started_model = estimated_payload.get("started_model")
            if not models_used:
                models_used = dict(estimated_payload.get("models_used", {}))
            estimated_active = float(estimated_payload.get("active_duration_seconds", 0.0))
            if estimated_active > 0 and (active_duration_seconds <= 0 or active_duration_seconds >= duration_seconds):
                active_duration_seconds = estimated_active

        if not started_model and models_used:
            started_model = next(iter(models_used))

        return {
            "session_id": report.session_id,
            "started_at": report.started_at.isoformat(),
            "ended_at": report.ended_at.isoformat() if report.ended_at else None,
            "duration_seconds": duration_seconds,
            "active_duration_seconds": active_duration_seconds,
            "vendor": report.vendor,
            "agent_settings": report.agent_settings,
            "skills": report.skills,
            "telemetry": report.telemetry,
            "raw_artifact_ids": report.raw_artifact_ids,
            "total_turns": report.total_turns or int((estimated_payload or {}).get("total_turns", 0)),
            "total_cost_usd": total_cost_usd,
            "total_lemoncrow_savings_usd": report.total_lemoncrow_savings_usd,
            "label": None,
            "models_used": models_used,
            "started_model": started_model,
            "cost_status": cost_status,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cache_read_tokens,
            "tool_call_count": report.tool_call_count or int((estimated_payload or {}).get("tool_call_count", 0)),
            "input_token_cost_usd": input_token_cost_usd,
            "cache_write_cost_usd": cache_write_cost_usd,
            "cache_read_cost_usd": cache_read_cost_usd,
            "output_token_cost_usd": output_token_cost_usd,
            "cache_write_tokens": cache_write_tokens,
            "cache_read_tokens": cache_read_tokens,
            "routing_downtiered_turns": report.routing_downtiered_turns,
            "routing_savings_usd": report.routing_savings_usd,
            "compact_events": report.compact_events,
            "compact_savings_estimate_usd": report.compact_savings_estimate_usd,
            "context_compression_savings_usd": report.context_compression_savings_usd,
            "context_compression_tool_calls": report.context_compression_tool_calls,
            "tool_savings": report.tool_savings,
            "top_tools_by_cost": top_tools_by_cost,
        }

    # Caches for GET /v1/sessions — the frontend polls this every ~30s and the
    # dataset is dominated by sessions that already finished, so most polls
    # should cost almost nothing:
    #  - run-file payload cache: keyed on the run.json file's own
    #    (mtime, size). A completed session's run.json never changes again,
    #    so a repeat poll skips build_report() entirely for it — profiling
    #    showed build_report() -> _read_compact_savings() re-reading and
    #    re-parsing the *entire* live_savings_events.jsonl file once PER RUN
    #    FILE was the dominant warm-path cost (proportional to run-file count
    #    x savings-log size, on every single request).
    #  - run-files listing cache: short TTL so a burst of requests (multiple
    #    tabs/components loading at once) shares one glob+stat walk of
    #    sessions/**/run.json instead of re-globbing per request.
    _run_file_session_payload_cache: dict[str, tuple[tuple[float, int], dict[str, Any]]] = {}
    _run_file_session_payload_cache_lock = threading.Lock()
    _run_files_list_cache: dict[tuple[str, int], tuple[float, list[Any]]] = {}
    _RUN_FILES_LIST_CACHE_TTL_SECONDS = 10.0

    def _cached_list_run_files(root: Path, *, days: int, cutoff: datetime) -> list[Any]:
        from lemoncrow.infra.runtime.session_report import list_run_files

        cache_key = (str(root), days)
        now_monotonic = time.monotonic()
        cached_listing = _run_files_list_cache.get(cache_key)
        if cached_listing is not None and now_monotonic - cached_listing[0] < _RUN_FILES_LIST_CACHE_TTL_SECONDS:
            return cached_listing[1]
        files = list_run_files(root, since=cutoff)
        _run_files_list_cache[cache_key] = (now_monotonic, files)
        return files

    @app.get("/v1/sessions", tags=["sessions"], dependencies=[Depends(verify_api_key)])
    def list_sessions(
        since: str = Query("7d", description="Time window, e.g. '7d', '30d'"),
        limit: int = Query(200, ge=1, le=1000),
    ) -> list[dict[str, Any]]:
        """List recent sessions with headline cost/savings fields."""
        from datetime import timedelta

        from lemoncrow.infra.runtime.session_report import build_report

        root = Path(cfg.lemoncrow_root)

        # Parse since
        days = 7
        if since.endswith("d"):
            with contextlib.suppress(ValueError):
                days = int(since[:-1])
        cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(days))

        files = _cached_list_run_files(root, days=days, cutoff=cutoff)
        results: list[dict[str, Any]] = []
        seen_session_ids: set[str] = set()
        for f in files[:limit]:
            try:
                stat_result = f.stat()
                run_file_fingerprint = (stat_result.st_mtime, stat_result.st_size)
                cache_key = str(f)
                with _run_file_session_payload_cache_lock:
                    cached_entry = _run_file_session_payload_cache.get(cache_key)
                if cached_entry is not None and cached_entry[0] == run_file_fingerprint:
                    payload = dict(cached_entry[1])
                else:
                    snap: dict[str, Any] = json.loads(f.read_text(encoding="utf-8"))
                    report = build_report(snap, root)
                    # build_report()'s started_at prefers the oldest *surviving*
                    # ledger event, but RunLedger.events is a bounded/evicted list
                    # (see run_ledger.py _MAX_RETAINED_EVENTS) — for a long
                    # session the true first event can be evicted, drifting
                    # started_at forward until it looks like processing time. The
                    # ledger's own created_at is written once at session start
                    # and is never evicted, so it's always <= the true first
                    # event and is the authoritative floor.
                    snap_created_at = _parse_session_datetime(snap.get("created_at"))
                    if snap_created_at is not None and snap_created_at < report.started_at:
                        report.started_at = snap_created_at
                    payload = _build_session_payload(report)
                    payload["updated_at"] = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC).isoformat()
                    with _run_file_session_payload_cache_lock:
                        if len(_run_file_session_payload_cache) > 4000:
                            _run_file_session_payload_cache.clear()
                        _run_file_session_payload_cache[cache_key] = (run_file_fingerprint, dict(payload))
                results.append(payload)
                seen_session_ids.add(payload["session_id"])
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue
        store_inst = get_store()
        imported_traces = store_inst.history.list_traces(since=cutoff, limit=max(limit * 5, limit))
        # Batch every imported session's raw-artifact staleness fingerprints
        # in one query up front instead of one store.get_raw_artifact() call
        # (and hence one fresh sqlite3 connection) per artifact per session.
        artifact_fp_lookup = _bulk_raw_artifact_fingerprints(
            store_inst,
            {art_id for trace in imported_traces for art_id in trace.raw_artifact_ids},
        )
        for trace in imported_traces:
            sid = trace.session_id or trace.id
            if sid in seen_session_ids:
                continue
            try:
                payload = _build_imported_session_payload(
                    sid, trace, store_inst, root=root, artifact_fp_lookup=artifact_fp_lookup
                )
                # trace.created_at is when this trace record was ingested
                # (import time), not the session's real last activity — use
                # the payload's own ended_at (the real last turn timestamp,
                # or trace.created_at when no turn has a parseable one) so a
                # session that finished long before it was imported doesn't
                # sort as if it just happened.
                payload["updated_at"] = payload.get("ended_at") or trace.created_at.isoformat()
                results.append(payload)
                seen_session_ids.add(sid)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue
            if len(results) >= limit:
                break

        def _session_sort_key(item: dict[str, Any]) -> float:
            def _ts(value: Any) -> float:
                if not value:
                    return 0.0
                with contextlib.suppress(TypeError, ValueError):
                    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                    return parsed.timestamp()
                return 0.0

            # Effective last-activity: an in-progress session has no ended_at
            # yet, so fall back to updated_at (run-ledger mtime) and finally
            # started_at — matching the frontend comparator (ended_at ||
            # updated_at) so a running session isn't sorted below history and
            # truncated out by results[:limit].
            return _ts(item.get("ended_at")) or _ts(item.get("updated_at")) or _ts(item.get("started_at"))

        results.sort(key=_session_sort_key, reverse=True)
        if len(results) > limit:
            results = results[:limit]
        return results

    @app.get(
        "/v1/sessions/{session_id}",
        tags=["sessions"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_session(session_id: str) -> dict[str, Any]:
        """Full session report for a single session_id."""
        from lemoncrow.infra.runtime.session_report import load_report

        root = Path(cfg.lemoncrow_root)
        report = load_report(session_id, root)
        if report is None:
            trace = _load_trace_for_session(session_id)
            if trace is None:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
            return _build_imported_session_payload(session_id, trace, root=root)
        return _build_session_payload(report)

    @app.get(
        "/v1/memory/facts",
        tags=["memory"],
        dependencies=[Depends(verify_api_key)],
    )
    def list_memory_facts(
        vendor: str | None = Query(None, description="Filter by vendor: claude, codex, gemini"),
    ) -> list[dict[str, Any]]:
        """List cross-vendor memory facts, optionally filtered by vendor."""
        from lemoncrow.pro.capabilities.cross_vendor_memory.registry import MemoryRegistry

        registry = MemoryRegistry()
        facts = registry.by_vendor(vendor) if vendor else registry.all_facts()
        return [
            {
                "fact_id": f.fact_id,
                "vendor": f.vendor,
                "source_path": str(f.source_path),
                "source_kind": f.source_kind,
                "content": f.content,
                "line_number": f.line_number,
                "captured_at": f.captured_at.isoformat(),
                "raw_meta": f.raw_meta,
            }
            for f in facts
        ]

    @app.get(
        "/v1/memory/facts/{fact_id}",
        tags=["memory"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_memory_fact(fact_id: str) -> dict[str, Any]:
        """Get a single cross-vendor memory fact by its stable fact_id."""
        from lemoncrow.pro.capabilities.cross_vendor_memory.registry import MemoryRegistry

        registry = MemoryRegistry()
        fact = registry.show(fact_id)
        if fact is None:
            raise HTTPException(status_code=404, detail=f"Fact '{fact_id}' not found")
        return {
            "fact_id": fact.fact_id,
            "vendor": fact.vendor,
            "source_path": str(fact.source_path),
            "source_kind": fact.source_kind,
            "content": fact.content,
            "line_number": fact.line_number,
            "captured_at": fact.captured_at.isoformat(),
            "raw_meta": fact.raw_meta,
        }

    # 60-second LRU cache keyed on (since, root_str, time_bucket). The time
    # bucket shards the cache per TTL window so a stale entry simply misses
    # (recomputing only THAT window) and the LRU maxsize bounds growth — instead
    # of a manual timestamp dict that grew unbounded per distinct user `since`
    # value and a cache_clear() that wiped every window on any one's expiry.
    import functools

    @functools.lru_cache(maxsize=64)
    def _cached_insights(since_str: str, root_str: str, _bucket: int) -> dict[str, Any]:
        from datetime import timedelta

        from lemoncrow.pro.runtime.insights import build_insights

        root = Path(root_str)
        days = 7
        if since_str.endswith("d"):
            with contextlib.suppress(ValueError):
                days = int(since_str[:-1])
        now = datetime.now(UTC)
        since_dt = now - timedelta(days=_clamp_days(days))
        window = build_insights(root, since=since_dt, until=now)
        return {
            "since": window.since.isoformat(),
            "until": window.until.isoformat(),
            "session_count": window.session_count,
            "total_duration_seconds": window.total_duration_seconds,
            "total_cost_usd": window.total_cost_usd,
            "total_lemoncrow_savings_usd": window.total_lemoncrow_savings_usd,
            "cost_by_vendor": window.cost_by_vendor,
            "cost_by_tool": window.cost_by_tool,
            "cost_by_model": window.cost_by_model,
            "top_sessions": [
                {
                    "session_id": s.session_id,
                    "cost_usd": s.cost_usd,
                    "label": s.label,
                    "duration_seconds": s.duration_seconds,
                }
                for s in window.top_sessions
            ],
            "outcomes_summary": {
                "route_decisions": window.outcomes_summary.route_decisions,
                "route_avg_score": window.outcomes_summary.route_avg_score,
                "compact_events": window.outcomes_summary.compact_events,
                "compact_avg_score": window.outcomes_summary.compact_avg_score,
                "sessions_with_high_extra_reads": window.outcomes_summary.sessions_with_high_extra_reads,
            },
            "opportunities": [
                {
                    "kind": o.kind,
                    "message": o.message,
                    "estimated_savings_usd": o.estimated_savings_usd,
                    "sessions_affected": o.sessions_affected,
                }
                for o in window.opportunities
            ],
        }

    import time as _time

    _INSIGHTS_CACHE_TTL = 60.0

    @app.get("/v1/insights", tags=["insights"], dependencies=[Depends(verify_api_key)])
    def get_insights(since: str = Query("7d")) -> dict[str, Any]:
        """Weekly insights window — cost, top sessions, opportunities. Cached 60s."""
        root_str = str(cfg.lemoncrow_root)
        # Integer TTL bucket: entries older than one TTL fall in a prior bucket
        # (natural miss) and drop out of the LRU; each window keeps its own TTL.
        bucket = int(_time.monotonic() // _INSIGHTS_CACHE_TTL)
        return _cached_insights(since, root_str, bucket)

    @app.get(
        "/v1/outcomes/summary",
        tags=["outcomes"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_outcomes_summary(since: str = Query("7d")) -> dict[str, Any]:
        """Aggregated route + compact outcome scores across all recent sessions."""
        from datetime import timedelta

        from lemoncrow.infra.runtime.session_report import list_run_files
        from lemoncrow.pro.runtime.outcome_capture import load_outcomes_from_state

        root = Path(cfg.lemoncrow_root)
        days = 7
        if since.endswith("d"):
            with contextlib.suppress(ValueError):
                days = int(since[:-1])
        cutoff = datetime.now(UTC) - timedelta(days=_clamp_days(days))

        files = list_run_files(root, since=cutoff)
        route_scores: list[float] = []
        compact_scores: list[float] = []
        high_extra_reads: list[str] = []

        for f in files:
            # files are canonical sessions/.../<id>/run.json paths; the session
            # id is the parent dir name and outcomes sit beside run.json.
            session_id = f.parent.name
            state_path = f.parent / "outcomes.json"
            if not state_path.exists():
                continue
            try:
                outcomes = load_outcomes_from_state(state_path)
            except Exception:
                logging.exception("Recovered from broad exception handler")
                continue

            for entry in outcomes.get("route_outcomes", []):
                ow = entry.get("outcome_window") or {}
                sc = ow.get("outcome_score")
                if sc is not None:
                    route_scores.append(float(sc))

            for entry in outcomes.get("compact_outcomes", []):
                ow = entry.get("outcome_window") or {}
                sc = ow.get("outcome_score")
                if sc is not None:
                    compact_scores.append(float(sc))
                er = ow.get("extra_read_rate")
                if er is not None and float(er) > 0.3:
                    high_extra_reads.append(session_id)

        return {
            "route_decisions": len(route_scores),
            "route_avg_score": round(sum(route_scores) / len(route_scores), 4) if route_scores else 0.0,
            "compact_events": len(compact_scores),
            "compact_avg_score": round(sum(compact_scores) / len(compact_scores), 4) if compact_scores else 0.0,
            "sessions_with_high_extra_reads": list(set(high_extra_reads)),
        }

    @app.get(
        "/v1/outcomes/{session_id}",
        tags=["outcomes"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_outcomes_for_session(session_id: str) -> list[dict[str, Any]]:
        """All outcome entries (route + compact) for a single session.

        A session with no recorded outcomes is a normal empty state, not an
        error — return [] so the dashboard doesn't log a 404 for every
        session opened.
        """
        from lemoncrow.core.foundation.paths import find_session_dir
        from lemoncrow.pro.runtime.outcome_capture import load_outcomes_from_state

        root = Path(cfg.lemoncrow_root)
        _session_root = find_session_dir(root, session_id) or flat_session_dir(root, session_id)
        if _session_root is None:
            return []
        state_path = _session_root / "outcomes.json"
        if not state_path.exists():
            return []
        try:
            outcomes = load_outcomes_from_state(state_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to load outcomes") from exc

        entries: list[dict[str, Any]] = []
        for entry in outcomes.get("route_outcomes", []):
            entries.append({"kind": "route", **entry})
        for entry in outcomes.get("compact_outcomes", []):
            entries.append({"kind": "compact", **entry})
        return entries

    @app.get("/v1/reports", tags=["reports"], dependencies=[Depends(verify_api_key)])
    def list_reports() -> list[dict[str, Any]]:
        """List all published benchmark reports from reports/index.json."""
        index_path = Path("reports") / "index.json"
        if not index_path.exists():
            return []
        try:
            index: list[dict[str, Any]] = json.loads(index_path.read_text())
            return index
        except (OSError, json.JSONDecodeError):
            return []

    @app.get(
        "/v1/reports/{week}",
        tags=["reports"],
        dependencies=[Depends(verify_api_key)],
    )
    def get_report(week: str) -> dict[str, Any]:
        """Get a published benchmark report (markdown + json) for a specific ISO week."""
        # Validate week format to prevent path traversal
        import re

        if not re.fullmatch(r"\d{4}-W\d{2}", week):
            raise HTTPException(status_code=400, detail="Invalid week format — expected YYYY-Www")

        reports_root = Path("reports").resolve(strict=False)
        report_dir = (reports_root / week).resolve(strict=False)
        md_path = report_dir / "benchmark.md"
        json_path = report_dir / "benchmark.json"

        if not report_dir.is_relative_to(reports_root):
            raise HTTPException(status_code=400, detail="Invalid week format — expected YYYY-Www")

        if not md_path.exists():
            raise HTTPException(status_code=404, detail=f"Report for week '{week}' not found")

        try:
            markdown_content = md_path.read_text(encoding="utf-8")
            json_data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Failed to read report files") from exc

        return {"week": week, "markdown": markdown_content, "json": json_data}

    @app.get("/v1/workflow/current", tags=["workflow"], dependencies=[Depends(verify_api_key)])
    def get_workflow_current() -> dict[str, Any]:
        detail = workflow_runtime_detail(_read_workflow_session_state())
        detail["workspace_root"] = str(resolve_workspace_root())
        return detail

    @app.post("/v1/workflow/current/pause", tags=["workflow"], dependencies=[Depends(verify_api_key)])
    def post_workflow_pause(payload: WorkflowSnapshotActionRequest) -> dict[str, Any]:
        state = _read_workflow_session_state()
        detail = pause_workflow_runtime(state, pause_reason=str(payload.reason or ""))
        _write_workflow_session_state(state)
        detail["workspace_root"] = str(resolve_workspace_root())
        return detail

    @app.post("/v1/workflow/current/stop", tags=["workflow"], dependencies=[Depends(verify_api_key)])
    def post_workflow_stop(payload: WorkflowSnapshotActionRequest) -> dict[str, Any]:
        state = _read_workflow_session_state()
        detail = stop_workflow_runtime(state, stop_reason=str(payload.reason or ""))
        _write_workflow_session_state(state)
        detail["workspace_root"] = str(resolve_workspace_root())
        return detail

    @app.get("/v1/swarm/launch/options", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_launch_options(
        project_root: str | None = Query(default=None),
        spec_path: str | None = Query(default=None),
    ) -> dict[str, Any]:
        root = Path(cfg.lemoncrow_root)
        try:
            selected_project_root = _coerce_project_root(root, project_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        file_options = _iter_swarm_spec_candidates(selected_project_root)
        default_spec = _default_swarm_spec_name(selected_project_root)
        selected_spec = (spec_path or default_spec).strip() or default_spec
        if selected_spec not in file_options:
            file_options.insert(0, selected_spec)
        elif file_options and file_options[0] != selected_spec:
            file_options = [
                selected_spec,
                *[item for item in file_options if item != selected_spec],
            ]
        spec_document = _load_swarm_spec_document(selected_project_root, selected_spec)

        return {
            "project_roots": [
                {
                    "path": str(candidate),
                    "label": candidate.name or str(candidate),
                    "full_path": str(candidate),
                    "has_program_md": any((candidate / name).is_file() for name in _SWARM_DEFAULT_SPEC_NAMES),
                }
                for candidate in _swarm_candidate_project_roots(root)
            ],
            "selected_project_root": str(selected_project_root),
            "files": [
                {
                    "path": relative_path,
                    "is_default": relative_path == default_spec,
                    "exists": (selected_project_root / relative_path).is_file(),
                }
                for relative_path in file_options
            ],
            "selected_spec_path": selected_spec,
            "spec_document": spec_document,
            "providers": [
                {
                    "id": "cli",
                    "label": "CLI runner",
                    "supported": True,
                    "reason": None,
                    "model_placeholder": None,
                    "credential_hint": None,
                },
                {
                    "id": "openai",
                    "label": "OpenAI API",
                    "supported": True,
                    "reason": None,
                    "model_placeholder": "gpt-4o-mini",
                    "credential_hint": "Uses server env only: LEMONCROW_OPENAI_API_KEY / OPENAI_API_KEY (+ optional LEMONCROW_OPENAI_BASE_URL).",
                },
                {
                    "id": "litellm",
                    "label": "LiteLLM",
                    "supported": True,
                    "reason": None,
                    "model_placeholder": "openai/gpt-4o-mini",
                    "credential_hint": "Uses server env only via LiteLLM credentials; no secrets are stored in swarm state.",
                },
            ],
            "runners": list_swarm_runner_profiles(),
            "defaults": {
                "provider": "cli",
                "runner": "claude",
                "runs": 3,
                "continuous": True,
                "max_waves": 5,
                "keep_worktrees": True,
                "effort": "high",
            },
            "notes": {
                "default_spec": default_spec,
                "default_spec_missing": not any(
                    (selected_project_root / name).is_file() for name in _SWARM_DEFAULT_SPEC_NAMES
                ),
                "effort_behavior": "Effort is recorded in swarm metadata today; built-in CLI profiles do not auto-inject runner-specific effort flags.",
                "provider_credentials": "Provider-backed swarm workers inherit credentials from the server environment. API keys and base URLs are never written into child commands or persisted swarm state.",
            },
        }

    @app.post("/v1/swarm/runs", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def post_swarm_run(payload: SwarmLaunchRequest) -> dict[str, Any]:
        root = Path(cfg.lemoncrow_root)
        try:
            provider_env = _build_swarm_provider_env(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            project_root = _coerce_project_root(root, payload.project_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            resolved_spec_path, resolved_spec_source, spec_resolution, used_program_md = _materialize_swarm_spec(
                project_root=project_root,
                spec_path=payload.spec_path,
                spec_mode=payload.spec_mode,
                spec_content=payload.spec_content,
            )
            if payload.provider == "cli":
                runner_options = shlex.split(payload.runner_options) if payload.runner_options.strip() else []
                child_command = resolve_swarm_child_command(
                    runner=payload.runner,
                    runner_model=payload.runner_model,
                    runner_args=runner_options,
                    child_command=(),
                    prompt_template=(
                        "The authoritative task spec is stored at {spec}.\n\n"
                        "<task_spec>\n"
                        "{spec_contents}\n"
                        "</task_spec>\n\n"
                        "Work directly in the current repository, make only the requested "
                        "changes, do not commit, and print a concise summary of what you "
                        "changed or why you left it unchanged."
                    ),
                )
                runner_name, runner_model = resolve_swarm_runner_metadata(
                    runner=payload.runner,
                    runner_model=payload.runner_model,
                    child_command=child_command,
                )
            else:
                child_command = resolve_swarm_provider_command(payload.provider)
                runner_name = payload.provider
                runner_model = payload.model or ""
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        state, state_path = initialize_swarm_run(
            root=root,
            repo_root=project_root,
            spec_path=resolved_spec_path,
            spec_source_path=resolved_spec_source,
            spec_resolution=spec_resolution,
            used_program_md=used_program_md,
            runner_name=runner_name,
            runner_model=runner_model,
            child_command=child_command,
            runs=payload.runs,
            validation_commands=[],
            keep_worktrees=payload.keep_worktrees,
            detached=True,
            continuous=payload.continuous,
            launch_provider=payload.provider,
            launch_effort=payload.effort,
            evaluator_backend=payload.evaluator_backend,
            evaluator_model=payload.evaluator_model or "",
            max_waves=payload.max_waves if payload.continuous else 1,
            max_evaluator_failures=payload.max_evaluator_failures,
        )
        if payload.provider != "cli":
            state.limitations.append(
                "Provider-backed swarm children only run bounded tool loops plus structural git-diff validation unless explicit validation commands are configured."
            )
        coordinator_pid, log_path = spawn_swarm_coordinator(
            root,
            project_root,
            state_path,
            env_overrides=provider_env,
        )
        state.coordinator_pid = coordinator_pid
        state.coordinator_log_path = str(log_path)
        save_swarm_state(state_path, state)
        return {
            "run_id": state.run_id,
            "status": "running",
            "state_path": str(state_path),
            "coordinator_pid": coordinator_pid,
            "log_path": str(log_path),
        }

    @app.get("/v1/swarm/runs", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_runs() -> list[dict[str, Any]]:
        root = Path(cfg.lemoncrow_root)
        states = list_swarm_runs(root)
        payload: list[dict[str, Any]] = []
        for state in states:
            latest_wave = state.waves[-1] if state.waves else None
            spec_payload = build_swarm_spec_payload(state)
            payload.append(
                {
                    "run_id": state.run_id,
                    "status": state.status,
                    "mode": state.mode,
                    "repo_root": state.repo_root,
                    "repo_label": Path(state.repo_root).name,
                    "runner_name": state.runner_name,
                    "runner_model": state.runner_model,
                    "launch_provider": state.launch_provider,
                    "launch_effort": state.launch_effort,
                    "evaluator_backend": state.evaluator_backend,
                    "evaluator_model": state.evaluator_model,
                    "convergence_status": state.convergence_status,
                    "convergence_summary": state.convergence_summary,
                    "next_wave_directives": list(state.next_wave_directives),
                    "current_wave": state.current_wave,
                    "max_waves": state.max_waves,
                    "max_runs": state.max_runs or state.runs,
                    "planned_runs": latest_wave.planned_runs if latest_wave else 0,
                    "planning_mode": latest_wave.planning_mode if latest_wave else state.planning_mode,
                    "accepted_child_ids": state.accepted_child_ids,
                    "primary_winner_child_id": state.primary_winner_child_id or state.winner_child_id,
                    "failed_children": [child.child_id for child in state.children if child.status == "failed"],
                    "running_children": [
                        {
                            "child_id": child.child_id,
                            "activity": child.current_activity,
                            "last_output_at": child.last_output_at,
                        }
                        for child in state.children
                        if child.status == "running"
                    ],
                    "spec_title": spec_payload["title"],
                    "spec_excerpt": spec_payload["excerpt"],
                    "spec_resolution": state.spec_resolution,
                    "used_program_md": state.used_program_md,
                    "created_at": state.created_at,
                    "updated_at": state.updated_at,
                }
            )
        return payload

    @app.get("/v1/swarm/runs/{run_id}", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_run(run_id: str) -> dict[str, Any]:
        state_path = resolve_state_path(Path(cfg.lemoncrow_root), run_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown swarm run: {run_id}")
        state = load_swarm_state(state_path)
        return {
            "run": state.model_dump(mode="json"),
            "spec": build_swarm_spec_payload(state),
            "export": build_swarm_export_payload(state),
            "apply": build_swarm_apply_payload(state),
        }

    @app.get("/v1/swarm/runs/{run_id}/logs", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_logs(
        run_id: str,
        child_id: str | None = None,
        stderr: bool = False,
        tail: int = 40,
    ) -> dict[str, Any]:
        try:
            content = read_swarm_log(
                Path(cfg.lemoncrow_root),
                run_id,
                child_id=child_id,
                stderr=stderr,
                tail=tail,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": run_id,
            "child_id": child_id,
            "stderr": stderr,
            "tail": tail,
            "content": content,
        }

    @app.get("/v1/swarm/runs/{run_id}/export", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_export(run_id: str) -> dict[str, Any]:
        state_path = resolve_state_path(Path(cfg.lemoncrow_root), run_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown swarm run: {run_id}")
        return build_swarm_export_payload(load_swarm_state(state_path))

    @app.get("/v1/swarm/runs/{run_id}/apply", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def get_swarm_apply(run_id: str, wave: int | None = None, child_id: str | None = None) -> dict[str, Any]:
        state_path = resolve_state_path(Path(cfg.lemoncrow_root), run_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown swarm run: {run_id}")
        state = load_swarm_state(state_path)
        try:
            return build_swarm_apply_payload(state, wave_index=wave, child_id=child_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/swarm/runs/{run_id}/stop", tags=["swarm"], dependencies=[Depends(verify_api_key)])
    def post_swarm_stop(run_id: str, cleanup: bool = False) -> dict[str, Any]:
        state_path = resolve_state_path(Path(cfg.lemoncrow_root), run_id)
        if not state_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown swarm run: {run_id}")
        state = stop_swarm_run(root=Path(cfg.lemoncrow_root), state_path=state_path, cleanup=cleanup)
        return state.model_dump(mode="json")

    # ------------------------------------------------------------------ #
    # Local code map                                                     #
    # ------------------------------------------------------------------ #

    def _code_map_context(project_root: str | None) -> tuple[Any, Path]:
        from lemoncrow.core.service import code_map

        try:
            resolved = code_map.resolve_project_root(project_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return code_map.get_engine(str(resolved)), resolved

    @app.get("/v1/code-map/projects", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_projects() -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        return {"projects": code_map.list_projects()}

    @app.get("/v1/code-map/overview", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_overview(project_root: str | None = Query(default=None)) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, resolved = _code_map_context(project_root)
        return code_map.build_overview(engine, resolved)

    @app.get("/v1/code-map/full", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_full(project_root: str | None = Query(default=None)) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, resolved = _code_map_context(project_root)
        return code_map.build_full_graph(engine, resolved)

    @app.get("/v1/code-map/search", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_search(
        q: str = Query(min_length=1, max_length=240),
        project_root: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1),
    ) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, _ = _code_map_context(project_root)
        return {"query": q, "results": code_map.search_symbols(engine, q, limit=min(limit, 40))}

    @app.get("/v1/code-map/neighborhood", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_neighborhood(
        symbol_id: str = Query(min_length=1, max_length=240),
        project_root: str | None = Query(default=None),
        depth: int = Query(default=1, ge=1),
        limit: int = Query(default=80, ge=4),
    ) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, _ = _code_map_context(project_root)
        try:
            return code_map.build_neighborhood(
                engine,
                symbol_id,
                depth=min(depth, 2),
                limit=min(limit, 160),
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/code-map/symbol", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_symbol(
        symbol_id: str = Query(min_length=1, max_length=240),
        project_root: str | None = Query(default=None),
    ) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, _ = _code_map_context(project_root)
        try:
            return code_map.symbol_detail(engine, symbol_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/code-map/activity", tags=["code-map"], dependencies=[Depends(verify_api_key)])
    def get_code_map_activity(
        project_root: str | None = Query(default=None),
        after: str | None = Query(default=None, max_length=80),
        limit: int = Query(default=60, ge=1),
    ) -> dict[str, Any]:
        from lemoncrow.core.service import code_map

        engine, resolved = _code_map_context(project_root)
        return code_map.recent_activity(
            Path(cfg.lemoncrow_root),
            resolved,
            after=after,
            limit=min(limit, 200),
            engine=engine,
        )

    return app


_lazy_app: Any | None = None


def __getattr__(name: str) -> Any:
    """Lazy ``app`` module attribute (PEP 562).

    Building the FastAPI instance at import time started background threads
    (savings reconciler) as an import side effect. uvicorn runs through the
    factory pattern (see ``main``); any legacy ``api:app`` reference still
    works via this deferred, cached constructor.
    """
    global _lazy_app
    if name == "app":
        if _lazy_app is None:
            _lazy_app = create_app()
        return _lazy_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in _LOOPBACK_HOSTS


def main(
    host: str | None = None,
    port: int | None = None,
    *,
    reload: bool = False,
) -> None:
    """Launch the service with uvicorn.

    Used by ``lc service start`` CLI command and the ``lemoncrow-service``
    entrypoint.

    Binding a non-loopback interface fails closed: auth is required (requests
    are rejected until LEMONCROW_API_KEY is set) unless LEMONCROW_REQUIRE_AUTH is
    explicitly disabled.
    """
    import sys

    import uvicorn

    _host = host or cfg.host
    _port = port or cfg.port

    # cfg.require_auth keys off LEMONCROW_SERVICE_HOST; feed it the host that is
    # actually being bound so an explicit ``--host 0.0.0.0`` fails closed
    # (auth required) instead of only warning. An explicit
    # LEMONCROW_REQUIRE_AUTH still wins in either direction.
    os.environ["LEMONCROW_SERVICE_HOST"] = _host
    os.environ["LEMONCROW_SERVICE_PORT"] = str(_port)

    if not _is_loopback(_host):
        if not cfg.require_auth:
            sys.stderr.write(
                f"\nWarning: host={_host!r} is non-loopback and LEMONCROW_REQUIRE_AUTH "
                "was explicitly disabled.\n"
                "Local memory, traces, and configuration will be exposed to the network.\n"
                "Remove the LEMONCROW_REQUIRE_AUTH override and set "
                "LEMONCROW_API_KEY=<a-long-random-secret> for authenticated access.\n"
            )
        elif not cfg.api_key:
            sys.stderr.write(
                f"\nNote: host={_host!r} is non-loopback, so auth is required, "
                "but LEMONCROW_API_KEY is empty.\n"
                "All protected routes will return 503 until "
                "LEMONCROW_API_KEY=<a-long-random-secret> is set.\n"
            )

    uvicorn.run(
        "lemoncrow.core.service.api:create_app",
        factory=True,
        host=_host,
        port=_port,
        reload=reload,
    )
