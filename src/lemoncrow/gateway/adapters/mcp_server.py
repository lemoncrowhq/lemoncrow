"""MCP server (stdio JSON-RPC) for the LemonCrow context runtime.

Implements a minimal subset of the Model Context Protocol sufficient for
Codex / Claude Code to discover and call the runtime tools.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import struct
import sys
import tempfile
import threading
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from pydantic import Field, ValidationError

from lemoncrow import __version__ as lemoncrow_version
from lemoncrow.core.capabilities.default_definitions import DefaultRegistry, build_default_registry
from lemoncrow.core.capabilities.host_runners import resolve_swarm_runner_command
from lemoncrow.core.capabilities.model_settings import normalize_model_for_host, resolve_host_model
from lemoncrow.core.capabilities.workflow_runtime_state import (
    coerce_workflow_review_decision as _coerce_workflow_review_decision,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    pause_workflow_runtime as _pause_workflow_runtime,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    require_active_workflow_runtime as _require_active_workflow_runtime,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    stop_workflow_runtime as _stop_workflow_runtime,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    workflow_runtime_state as _workflow_runtime_state,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    workflow_runtime_status as _coerce_workflow_runtime_status,
)
from lemoncrow.core.capabilities.workflow_runtime_state import (
    write_workflow_runtime_state as _write_workflow_runtime_state,
)
from lemoncrow.core.environment import mcp_tool_description, mcp_tool_mode, mcp_tool_visible_to_llm
from lemoncrow.core.foundation.memory_models import ArchivalPassage, MemoryBlock
from lemoncrow.core.foundation.models import RawArtifact, Trace, to_jsonable
from lemoncrow.core.foundation.redaction import redact
from lemoncrow.core.foundation.rubric_gate import run_rubric
from lemoncrow.gateway.adapters.mcp import ledger as _ledger
from lemoncrow.gateway.adapters.mcp.bash import (  # noqa: F401  (registers bash tool + re-exports)
    _BASH_STATS_MAX_KEYS,
    _BASH_STATS_PRUNE_TO,
    _DEFAULT_BASH_SOFT_TIMEOUT,
    BASH_TOOL_INPUT_SCHEMA,
    _bash_command_key,
    _bash_omitted_tokens_saved,
    _record_bash_command_stats,
    _render_bash_text,
    _run_bash_tool,
)
from lemoncrow.gateway.adapters.mcp.bash import (
    tool_bash as tool_bash,
)
from lemoncrow.gateway.adapters.mcp.deferral import (  # noqa: F401  (re-exported for back-compat)
    _defer_bash_enabled,
    _defer_web_fetch_enabled,
    _deferral_context,
    _deferral_supported,
    _deferred_completion_executor,
    _DeferredResult,
)
from lemoncrow.gateway.adapters.mcp.deferral import (
    _Deferred as _Deferred,
)
from lemoncrow.gateway.adapters.mcp.framework import (  # noqa: F401  (re-exported for back-compat)
    _COERCE_UNCHANGED,
    _annotation_base_types,
    _coerce_json_strings,
    _coerce_str_to_annotation,
    _slim_schema,
    _ToolArgumentError,
    mcp_tool,
)
from lemoncrow.gateway.adapters.mcp.framework import (
    TOOLS as TOOLS,
)
from lemoncrow.gateway.adapters.mcp.fs_access import (  # noqa: F401  (re-exported for back-compat)
    _CLAUDE_ADDITIONAL_DIRS_CACHE,
    _claude_additional_dirs,
)
from lemoncrow.gateway.adapters.mcp.ledger import (  # noqa: F401  (re-exported for back-compat)
    _MAX_HTTP_SESSION_LEDGERS,
    _detect_agent,
    _get_claude_session_id,
    _get_product_session_id,
    _get_realtime_context,
    _http_session_ledgers,
    _http_session_ledgers_lock,
    _ledger_for_session,
    _mcp_window_id,
    _record_full_read,
    _request_ledger,
    _request_session_identity,
    _resolve_live_session_id,
    _workspace_ws_hash,
)
from lemoncrow.gateway.adapters.mcp.ledger import (
    _clear_request_ledger as _clear_request_ledger,
)
from lemoncrow.gateway.adapters.mcp.ledger import (
    _clear_request_session as _clear_request_session,
)
from lemoncrow.gateway.adapters.mcp.ledger import (
    _get_ledger as _get_ledger,
)
from lemoncrow.gateway.adapters.mcp.ledger import (
    _set_request_ledger as _set_request_ledger,
)
from lemoncrow.gateway.adapters.mcp.ledger import (
    _set_request_session as _set_request_session,
)
from lemoncrow.gateway.adapters.mcp.session_state import (  # noqa: F401  (re-exported for back-compat)
    _MCP_ID,
    _MCP_SESSION_FILE_LOCK,
    _forget_mcp_managed_bash,
    _lemoncrow_root,
    _mcp_session_file,
    _mutate_mcp_managed_bash,
    _record_mcp_managed_bash,
)
from lemoncrow.gateway.adapters.mcp.smart_state import (  # noqa: F401  (re-exported for back-compat)
    _STATE_LOCK,
    _acquire_smart_state_flock,
    _read_smart_state,
    _release_smart_state_flock,
    _smart_state_path,
    _tool_call_tokens_saved,
    _write_smart_state,
)
from lemoncrow.gateway.adapters.mcp.tools_commodity import (  # noqa: F401  (registers commodity tools + re-exports)
    tool_mcp,
    tool_web_fetch,
)
from lemoncrow.gateway.adapters.mcp_branding import icon_metadata
from lemoncrow.infra.runtime.run_ledger import (
    RunLedger,
    outcomes_path,
)
from lemoncrow.infra.storage.factory import make_memory_store
from lemoncrow.infra.storage.memory_store import MemoryConcurrencyError, MemorySidecarUnavailable
from lemoncrow.pro.capabilities.code_context.diversity import demote_doc_overflow, query_wants_docs
from lemoncrow.pro.capabilities.owned_execution_lanes import (
    OwnedExecutionError,
    execute_owned_prompt,
)
from lemoncrow.pro.capabilities.owned_execution_routing import (
    NoFeasibleRouteError,
    OwnedCachePolicy,
    OwnedRouteRequest,
    select_owned_route,
)

if TYPE_CHECKING:
    from lemoncrow.gateway.adapters.runtime import ContextRuntime
    from lemoncrow.pro.capabilities.archival_recall import ArchivalRecallCapability
    from lemoncrow.pro.capabilities.memory import MemoryService
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

logger = logging.getLogger(__name__)


def _version_key(version: str) -> tuple[int, ...]:
    """Dotted version -> comparable int tuple (non-numeric chunks count as 0)."""
    parts: list[int] = []
    for chunk in version.split("."):
        match = re.match(r"\d+", chunk)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def _warm_pricing_table() -> None:
    """Pre-build the LiteLLM-backed pricing table off the response path.

    The first tools/call prepares a model recommendation
    (``_prepare_model_recommendation`` -> ``select_owned_route``), which builds
    the counterfactual routing pricing table: a full parse of the LiteLLM
    catalog plus ``pricing.yaml`` overrides (~100-200ms of pure-Python work).
    Warming it on a daemon thread keeps that parse out of the first tool
    response.  Cache semantics are unchanged — the per-process ``lru_cache``
    and its ``override_pricing()`` invalidation still apply; this only moves
    when the first build happens.  Fail-open.
    """
    try:
        from lemoncrow.pro.capabilities.counterfactual.pricing import load_pricing_table

        load_pricing_table()
    except Exception:
        logger.debug("pricing table pre-warm failed", exc_info=True)


# Spawned at import (top of module, so the parse overlaps the rest of this
# module's own import work) rather than in main(), so every entry point that
# dispatches through _handle (stdio serve, HTTP adapter, SDK/embedded direct
# dispatch) gets the warm start before its first tool response.  The
# counterfactual pricing module is already fully imported at this point (see
# the owned_execution_routing import above), so the thread never contends on
# import locks.
threading.Thread(target=_warm_pricing_table, name="lemoncrow-pricing-warm", daemon=True).start()

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "lc"
# Human-readable name reported in the initialize response's serverInfo — purely
# cosmetic (shown in host UIs like Claude Code's "User MCPs" connection list).
# Tool-name rendering (mcp__{SERVER_NAME}__*) and self-detection both key off
# the .mcp.json config entry name ("lc"), not this field, so it's safe to
# diverge from SERVER_NAME for display purposes.
SERVER_DISPLAY_NAME = "lemoncrow"
SERVER_VERSION = lemoncrow_version
# Injected into the host's system prompt via the MCP initialize response — the
# one steering surface every MCP client reads automatically, even hosts and
# subagents that never receive LemonCrow's persona files. Tool descriptions alone
# demonstrably do not change behavior: agents default to bash grep/cat and use
# the indexed tools ~5% of the time. This string carries the FULL generic tool
# discipline: on hosts that render it (Claude Code), the generated personas
# ship only the host-specific remainder (delegation targets + tool-name
# mapping) — see _CLAUDE_TOOL_DISCIPLINE in scripts/sync_agent_context.py.
# Hosts that ignore MCP instructions (Codex — openai/codex#6148 closed
# not-planned — and opencode) plus owned lanes keep the full persona block.
SERVER_INSTRUCTIONS = (
    "LemonCrow replaces the grep→read→re-read loop.\n"
    "- Lead with `code_search`: one call = ranked matches (top source inline, "
    "rest as path:Lx-Ly pointers) + related_symbols + candidate_files. Inline "
    "source = already read; never shell-grep or re-verify indexed results.\n"
    "- Known path/symbols → `read`: ONE call, files=[...], exact :Lx-Ly ranges, "
    "never the same file twice. Need, not might-need: a speculative :full costs "
    "more than the turn it saves. Never cat/sed/head/tail.\n"
    "- ALL edits in ONE `edit` edits[] array; prefer {path: 'f.py:Lx-Ly', new} over old/new.\n"
    "- Independent calls (any tools) → ONE message.\n"
    "- `bash` = execution only (tests, git, builds). Large output → a file, never inline prose.\n"
    "- Graphical data (plots, pixel grids, UI) → write a PNG and `read` it; "
    "don't infer visuals from raw bytes."
)
CONTEXT_WINDOW_TOKENS = 200_000
COMPACT_ADVISORY_THRESHOLD = 60.0
AUTO_COMPACT_THRESHOLD = 80.0
HANDOVER_THRESHOLD = 95.0
AUTO_COMPACT_MIN_TURNS = 15
# Bypass the min-turns gate when utilisation already exceeds this level —
# a few very large turns can fill the window just as fast as many small ones.
AUTO_COMPACT_HIGH_UTIL_OVERRIDE = 90.0

# Direct host integrations can advertise this eager core without making normal
# coding operations pay a broker/search round trip. Rare tools stay available
# behind the deterministic tool broker. The default remains full for backwards
# compatibility and for clients that explicitly prefer one large schema prefix.
_CORE_MCP_TOOLS = frozenset(
    {
        "bash",
        "code_search",
        "compact",
        "context",
        "edit",
        "grep",
        "memory",
        "read",
        "relations",
        "tool",
        "verify",
        "web_fetch",
    }
)

# --------------------------------------------------------------------------- #
# Tool Registry Decorator                                                     #
# --------------------------------------------------------------------------- #

# TOOLS registry now lives in mcp.framework (imported above).


def _tool_description(spec: dict[str, Any]) -> str:
    base = mcp_tool_description(
        str(spec.get("name", "") or ""),
        str(spec.get("description", "") or ""),
    )
    if spec.get("name") == "mcp":
        # Enrich (never spawn -- disk-cache read only) with server names seen
        # by the last `mcp(op="list")` call, if any, so the advertised
        # description hints at what is actually configured this workspace.
        with contextlib.suppress(Exception):
            from lemoncrow.gateway.adapters import mcp_proxy

            names = mcp_proxy.cached_server_names()
            if names:
                base = f"{base}\n\nLast known configured servers: {', '.join(names)}."
    return base


def _tool_visible_to_llm(tool_name: str, spec: dict[str, Any]) -> bool:
    return mcp_tool_visible_to_llm(tool_name)


def _mcp_tool_profile() -> str:
    value = os.environ.get("LEMONCROW_MCP_TOOL_PROFILE", "full").strip().lower()
    return value if value in {"core", "full"} else "full"


def _tool_profile_exposes(tool_name: str) -> bool:
    if _mcp_tool_profile() == "core":
        return tool_name in _CORE_MCP_TOOLS
    # The broker is unnecessary in full mode and would invite an avoidable call.
    return tool_name != "tool"


def _tool_mode(spec: dict[str, Any]) -> str:
    return mcp_tool_mode(str(spec.get("name", "") or ""))


# _COERCE_UNCHANGED, the argument-coercion helpers, _ToolArgumentError and the
# @mcp_tool decorator now live in mcp.framework (imported above).


# G13 — caller-selectable output encoding for read/search/grep. NOT published to
# LLMs (auto already picks the optimal encoding); kept as a power/CLI/benchmark
# knob, accepted by the handler but stripped from the advertised schema. Defined
# before the tool handlers so the @mcp_tool decorator can resolve the Annotated
# default at import time. The handler ignores this arg; the MCP dispatcher reads
# `args["format"]` and applies the N6-gated N7 columnar encoding. `auto`
# (default) keeps today's byte-compatible output.
_FORMAT_FIELD = Field(
    default="auto",
    description=(
        "Output encoding: auto (default, unchanged), json (force raw JSON), or compact (N6-gated columnar encoding)."
    ),
)


# --------------------------------------------------------------------------- #
# session_state.json helpers                                                  #
# --------------------------------------------------------------------------- #

# moved to mcp.ledger (imported/re-exported near the top).
# Per-request ledger override for the HTTP transport: set on the worker thread
# that runs _handle so concurrent HTTP clients each accumulate their own run.json
# instead of co-mingling into the process-global _current_ledger.
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
_product_session_started_at: float | None = None
_last_plan_hash_by_session: dict[str, str] = {}
_last_plan_by_session: dict[str, dict[str, Any]] = {}
_last_blocked_plan_hash_by_session: dict[str, str] = {}

# --------------------------------------------------------------------------- #
# Trajectory monitor state (per session)                                      #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _MonitorSession:
    """Per-session DifficultyFSM + step history for trajectory monitoring."""

    fsm: Any = dataclasses.field(default=None)
    steps: list[str] = dataclasses.field(default_factory=list)
    composite: float = 0.0
    _call_count: int = 0

    def __post_init__(self) -> None:
        if self.fsm is None:
            from lemoncrow.pro.capabilities.monitors.fsm import DifficultyFSM

            self.fsm = DifficultyFSM()


_monitor_sessions: dict[str, _MonitorSession] = {}
_MAX_MONITOR_STEPS = 25
_MAX_MONITOR_SESSIONS = 64
# Serializes mutation of the shared _monitor_sessions map / _MonitorSession
# state: the dispatcher runs context calls on a thread pool, so two concurrent
# calls for one session must not race the setdefault/eviction/steps mutation.
_MONITOR_LOCK = threading.Lock()


def _advance_monitors(session_id: str, task: str, original_task: str) -> tuple[float, bool]:
    """Advance per-session trajectory monitors; return (composite, skip_etraces).

    Guards itself behind the bench kill-switch so monitors don't interfere with
    benchmark runs.  Runs ``evaluate_all`` once every ``monitor_cooldown_steps``
    calls (as determined by the FSM state) to amortise the regex cost.
    """
    try:
        from lemoncrow.bench.mode import is_off as _bench_is_off

        if _bench_is_off():
            return 0.0, False

        from lemoncrow.pro.capabilities.monitors import evaluate_all
        from lemoncrow.pro.capabilities.monitors.fsm import score_step

        with _MONITOR_LOCK:
            ms = _monitor_sessions.setdefault(session_id, _MonitorSession())
            if len(_monitor_sessions) > _MAX_MONITOR_SESSIONS:
                # Bound the session map: a marathon process seeing many session
                # ids must not leak _MonitorSession objects. Evict oldest (skip current).
                for _stale in list(_monitor_sessions)[: len(_monitor_sessions) - _MAX_MONITOR_SESSIONS]:
                    if _stale != session_id:
                        _monitor_sessions.pop(_stale, None)
            ms.steps.append(task)
            if len(ms.steps) > _MAX_MONITOR_STEPS:
                ms.steps = ms.steps[-20:]
            ms.fsm.transition(score_step(task))
            ms._call_count += 1
            cooldown = ms.fsm.monitor_cooldown_steps
            run_eval = ms._call_count % cooldown == 0 or ms._call_count == 1
            steps_snapshot = list(ms.steps)
        if run_eval:
            result = evaluate_all(steps_snapshot, task=original_task)
            ms.composite = result.composite
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return 0.0, False

    return ms.composite, ms.fsm.skip_etraces


# LemonCrow-internal MCP process identity — generated once at import, never changes.
# SessionStart hook finds this file and writes the Claude session UUID + model into it.
# _get_claude_session_id() reads it once then caches in _cached_claude_session_id.
# _MCP_ID moved to mcp.session_state (imported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).
# _current_context_state cache: session id -> (stat-signature, (ctx, model)).
# That probe runs on every savings-bearing tool call; without this it re-tails
# and JSON-parses a 64 KB transcript window every time. Keyed on the candidate
# transcripts' (path, mtime_ns, size) so any new turn / rewrite invalidates it.
_CONTEXT_STATE_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], tuple[int, str]]] = {}
_STDOUT_LOCK = threading.Lock()
# _STATE_LOCK moved to mcp.smart_state (imported/re-exported).
# Per-file edit locks: concurrent edit calls (the MCP dispatcher runs a thread
# pool) that touch the same file must not interleave snapshot/apply/write, or one
# write clobbers the other (lost update). _EDIT_PATH_LOCKS maps a resolved file
# path to its Lock; _EDIT_PATH_LOCKS_GUARD serializes registry mutation only.
_EDIT_PATH_LOCKS: dict[str, threading.Lock] = {}
_EDIT_PATH_LOCKS_GUARD = threading.Lock()
# Per-session-id cache for the savings sidecar path; set on first write so the
# path stays stable if the process runs past midnight (date partition fixed at
# session-start time, not re-derived on every call).
_SAVINGS_SIDECAR_PATH_BY_SID: dict[str, Path] = {}
_DEFAULT_MCP_MAX_WORKERS = 16

# --------------------------------------------------------------------------- #
# Search verdict state (per session) -- mirrors the monitor-session pattern.   #
# Turns an empty search/grep result into an honest signal (found/missed/absent/ #
# dark) plus a soft circuit-breaker note, so the agent stops searching for the  #
# right reason instead of spiraling on hard tasks.                             #
# --------------------------------------------------------------------------- #
_search_history_sessions: dict[str, Any] = {}
_MAX_SEARCH_HISTORY_SESSIONS = 64
_SEARCH_HISTORY_LOCK = threading.Lock()

# Per-session identical-call counters for the spiral nudge (loop_review). Mirrors
# the SearchHistory registry above: bounded LRU-ish map, evict oldest past the
# cap so a long-lived process seeing many session ids cannot leak trackers.
_loop_tracker_sessions: dict[str, Any] = {}
_MAX_LOOP_TRACKER_SESSIONS = 64
_LOOP_TRACKER_LOCK = threading.Lock()


def _count_search_hits(payload: dict[str, Any]) -> int:
    """Hit count for a `search` payload (items pre-view, matches post-view)."""
    for key in ("matches", "items", "ranked_files"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _count_grep_hits(payload: dict[str, Any]) -> int:
    """Hit count for a `grep` payload (ranked matches, else non-empty text blocks)."""
    matches = payload.get("matches")
    if isinstance(matches, list):
        return len(matches)
    content = payload.get("content")
    if isinstance(content, list):
        return sum(
            1
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip()
        )
    return 0


def _search_cascade_enabled() -> bool:
    """Phase 3 cascade-on-empty toggle (default on; set 0/false to disable)."""
    return os.environ.get("LEMONCROW_SEARCH_CASCADE_ON_EMPTY", "1").strip().lower() not in {"0", "false", "no", ""}


def _apply_search_verdict(
    result: dict[str, Any],
    *,
    query: str,
    hit_count: int,
    channels: Any | None = None,
) -> dict[str, Any]:
    """Stamp verdict (+ next hint, + breaker note) onto a search/grep result.

    Boundary-only: per-session reformulation memory drives missed->absent, and a
    consecutive run of unproductive searches trips the soft breaker. The engine
    packing path is untouched.
    """
    if not isinstance(result, dict) or not (query or "").strip():
        return result
    from lemoncrow.pro.capabilities.code_context.search_verdict import (
        BREAKER_NOTE,
        ChannelHealth,
        SearchHistory,
        compute_verdict,
    )

    found = hit_count > 0
    session_id = _get_claude_session_id() or "_global"
    with _SEARCH_HISTORY_LOCK:
        history = _search_history_sessions.get(session_id)
        if history is None:
            history = SearchHistory()
            _search_history_sessions[session_id] = history
            if len(_search_history_sessions) > _MAX_SEARCH_HISTORY_SESSIONS:
                # Bound the map: a marathon process seeing many session ids must
                # not leak SearchHistory objects. Evict oldest (skip current).
                for stale in list(_search_history_sessions)[
                    : len(_search_history_sessions) - _MAX_SEARCH_HISTORY_SESSIONS
                ]:
                    if stale != session_id:
                        _search_history_sessions.pop(stale, None)
        prior = history.prior_empties()
    verdict = compute_verdict(
        hit_count=hit_count,
        query=query,
        channels=channels if isinstance(channels, ChannelHealth) else ChannelHealth(),
        prior_empties=prior,
    )
    with _SEARCH_HISTORY_LOCK:
        history.record(query, found=found)
        tripped = history.breaker_tripped()
    result["verdict"] = verdict.verdict
    if verdict.next:
        result["next"] = verdict.next
    if tripped and not found:
        result["breaker_note"] = BREAKER_NOTE
    return result


def _append_search_verdict_footer(text: str | None, result: dict[str, Any]) -> str | None:
    """Append verdict / fallback / breaker lines to a search|grep render.

    Single chokepoint so the model sees the honest-empty signal regardless of
    which underlying renderer produced the body text.
    """
    parts: list[str] = []
    fallback = result.get("text_fallback")
    if isinstance(fallback, list) and fallback:
        locs = "; ".join(f"{item.get('path')}:{item.get('line')}" for item in fallback[:8] if isinstance(item, dict))
        if locs:
            parts.append(f"[search:tfallback] {locs}")
    verdict = result.get("verdict")
    if isinstance(verdict, str) and verdict not in {"", "found"}:
        nxt = str(result.get("next") or "").strip()
        if verdict in {"missed", "absent"}:
            parts.append("no matches")  # minimal, vanilla-style -- no prefix, no nudge
        else:
            parts.append(f"[search:{verdict}]" + (f" {nxt}" if nxt else ""))
    breaker = result.get("breaker_note")
    if isinstance(breaker, str) and breaker:
        parts.append(f"[search:budget] {breaker}")
    if not parts:
        return text
    footer = "\n".join(parts)
    return f"{text}\n{footer}" if text else footer


def _edit_path_locks(resolved_paths: list[Path]) -> list[threading.Lock]:
    """Return locks for *resolved_paths*, ordered deterministically to avoid
    deadlock when an edit batch touches several files at once."""
    keys = sorted({str(p) for p in resolved_paths})
    with _EDIT_PATH_LOCKS_GUARD:
        return [_EDIT_PATH_LOCKS.setdefault(key, threading.Lock()) for key in keys]


_MAX_MCP_MAX_WORKERS = 64

# Single JSON-RPC frames larger than the host's stdout guard (~16 MiB in Claude
# Code) make the client disconnect the entire MCP server, and it does not
# auto-reconnect mid-session. Cap per-result text well under that, and keep a
# hard frame ceiling as a backstop so no single message can ever trip the guard.
_DEFAULT_MAX_RESULT_BYTES = 6 * 1024 * 1024
_MAX_WIRE_BYTES = 14 * 1024 * 1024
# Context-hygiene bound (chars). Distinct from the multi-MB wire guards above,
# which only keep a JSON-RPC frame under the host's stdout limit: this caps a
# single runaway tool result (huge log, minified file, unbounded grep) that
# would otherwise flood the host prompt -- and the host re-pays for those bytes
# on every later turn. ~64k tokens. Set LEMONCROW_MCP_COMPACT_RESULT_CHARS=0 to
# disable.
_DEFAULT_COMPACT_RESULT_CHARS = 256 * 1024
# Recoverable tool-output cap (chars). Results from spill-supported tools above
# this size are persisted in full and replaced by a bounded head+tail summary
# carrying a retrieval reference. Keep this separate from the legacy lossy
# compaction threshold so tools without spill support retain their prior budget.
# Set LEMONCROW_MCP_SPILL_RESULT_CHARS=0 to disable the char-gated spill cap.
_DEFAULT_SPILL_RESULT_CHARS = 2 * 1024
# Per-tool overrides of the recoverable char cap. bash output (test runs, diffs,
# build logs, git status) is routinely needed inline during the debug loop, so it
# gets a larger inline budget than sql, whose payloads are rarely needed
# byte-complete. Tools absent here use _DEFAULT_SPILL_RESULT_CHARS -- or are
# exempt from the cap entirely (see _SPILL_CHAR_CAP_TOOLS; e.g. web_fetch, which
# spills and truncates itself). An explicit LEMONCROW_MCP_SPILL_RESULT_CHARS env
# value overrides this map for every tool.
# Per-tool inline char budget before spill fires. bash stays small (shell output
# is cheap to re-run); code_search needs more headroom (structured + expensive to
# regenerate).
_SPILL_RESULT_CHARS_BY_TOOL = {
    "bash": 8 * 1024,
    "code_search": 20 * 1024,
}
# Per-read inline budget (bytes). A single file read larger than this is returned
# as a line-aligned prefix plus an EXACT continuation range, instead of being
# handed to the host whole -- where the host's own MCP-output guard would dump it
# to a temp file and force the agent to re-read the file in blind ranges. Keep it
# below the host limit (~50KB). Set LEMONCROW_READ_INLINE_BUDGET_BYTES=0 to disable.
_DEFAULT_READ_INLINE_BUDGET_BYTES = 40 * 1024
_DEFAULT_READ_BATCH_BUDGET_BYTES = 24 * 1024

# Per-session count of partial (range) reads per resolved path. Paging one file
# by hand 3+ times wastes a turn per chunk (and the chunks pile up in context
# anyway), so the third partial read of a path escalates to a full read. Keyed by
# resolved path; a :full read of that path resets it. See _smart_read_single.

# Honest-baseline caps for savings accounting. Savings are "input tokens kept
# out of the host prompt vs what the host would actually have paid" -- never vs
# an unbounded firehose. Vanilla Claude Code itself truncates Bash output at
# ~30k chars, and the host inlines at most ~50KB of MCP tool output before
# dumping the rest to a file the model never pays for. Both constants bound the
# NAIVE side of a savings computation only; they never change returned bytes.
# bash symbols moved to mcp.bash (imported/registered near the top).
_HOST_INLINE_RESULT_CHARS = 50 * 1024


def _service_backed_state() -> bool:
    return True


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


def _emit_mcp_session_start() -> None:
    global _product_session_started_at
    if _product_session_started_at is not None:
        return
    _register_mcp_session()  # register LemonCrow MCP ID so SessionStart hook can find us
    from importlib.metadata import PackageNotFoundError, version

    from lemoncrow.core.foundation.identity import get_anon_id, platform_payload
    from lemoncrow.core.service.telemetry import emit_product

    try:
        service_version = version("lemoncrow")
    except PackageNotFoundError:
        service_version = SERVER_VERSION
    # OTel is initialized lazily on first emit_product_log call.
    _product_session_started_at = time.perf_counter()
    emit_product(
        "session_start",
        agent_host=_detect_agent(),
        lemoncrow_version=service_version,
        anon_id=get_anon_id(),
        session_id=_get_product_session_id(),
        **platform_payload(),
    )


def _emit_mcp_session_end(exit_reason: str = "success") -> None:
    if _product_session_started_at is None:
        return
    from lemoncrow.core.service.telemetry import emit_product
    from lemoncrow.core.service.telemetry.schema import bucket_duration_s

    elapsed = max(0.0, time.perf_counter() - _product_session_started_at)
    emit_product(
        "session_end",
        session_id=_get_product_session_id(),
        duration_s_bucket=bucket_duration_s(elapsed),
        exit_reason=exit_reason,
    )


def _match_mcp_lexical(args: dict[str, Any]) -> None:
    from lemoncrow.core.service.telemetry.frustration import match_frustration

    for key in ("task", "query", "user_goal", "error"):
        value = args.get(key)
        if isinstance(value, str):
            match_frustration(value, surface="mcp_prompt", session_id=_get_product_session_id())


def _emit_playbook_retrieved(scored: list[Any], domain: str | None) -> None:
    from lemoncrow.core.service.telemetry import emit_product
    from lemoncrow.core.service.telemetry.schema import hash_identifier

    for rank, item in enumerate(scored, start=1):
        block = getattr(item, "block", None)
        emit_product(
            "playbook_retrieved",
            block_id_hash=hash_identifier(str(getattr(block, "id", ""))),
            domain=str(getattr(block, "domain", domain or "")),
            retrieval_score=float(getattr(item, "score", 0.0)),
            rank=rank,
            session_id=_get_product_session_id(),
        )


# --------------------------------------------------------------------------- #
# Tool implementations                                                        #
# --------------------------------------------------------------------------- #


# _lemoncrow_root moved to mcp.session_state (imported near the top).


def _make_outcome_writer(led: RunLedger) -> Any:
    """Return a FileStateWriter for outcomes alongside the run file, or None."""
    with contextlib.suppress(Exception):
        from lemoncrow.pro.runtime.outcome_capture import FileStateWriter

        root = led._root
        if root is not None:
            return FileStateWriter(outcomes_path(root, led.agent or "claude", led.session_id))
    return None


# --------------------------------------------------------------------------- #
# Zero-config background service                                              #
# --------------------------------------------------------------------------- #


def _detect_default_branch(repo: Path) -> str | None:
    """Detect the remote default branch (main/master) for *repo*."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "remote", "show", "origin"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("HEAD branch:"):
                branch = stripped.split(":")[-1].strip()
                if branch:
                    return branch
    except Exception:
        logging.exception("Recovered from broad exception handler")
        logger.warning(
            "Suppressed exception in _detect_default_branch",
            exc_info=True,
        )
    # Fallback: try main then master
    for candidate in ("main", "master"):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", f"origin/{candidate}"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            logging.exception("Recovered from broad exception handler")
            continue
    return None


_log = logging.getLogger("lemoncrow.mcp")


def _installed_cli_version() -> str | None:
    """Return the version reported by the installed ``lc`` executable."""
    from lemoncrow.core.foundation.update_state import installed_cli_version

    return installed_cli_version()


def _check_auto_update() -> None:
    """Check git remote for a newer version and auto-update if found.

    Compares the version in the remote repo's ``pyproject.toml`` against the
    currently installed version.  If they differ, pulls the repo and runs
    the install script.  Logs errors and emits telemetry on failure but
    never blocks the MCP server.

    Opt-in. Set ``LEMONCROW_AUTO_UPDATE=1`` (or true/yes/on) to enable startup
    git-checkout auto-update; it is OFF by default so the runtime never contacts
    the network unless the user asks.
    """
    import re
    import subprocess

    if os.environ.get("LEMONCROW_AUTO_UPDATE", "").strip().lower() not in ("1", "true", "yes", "on"):
        return

    # Dev/source installs (`make dev` writes <root>/.dev_mode) must never
    # auto-update: a git fetch/merge on the live checkout would fight an
    # in-progress refactor and can clobber uncommitted work. Production installs
    # have no marker. Read the marker fresh (not the cached _is_dev_mode()) so
    # the decision tracks the current root.
    try:
        if (_lemoncrow_root() / ".dev_mode").exists():
            _log.debug("auto-update skipped: dev install (.dev_mode present)")
            return
    except OSError:
        pass

    _log.info("checking for auto-update...")

    try:
        # Determine the repo directory
        install_dir = os.environ.get("LEMONCROW_INSTALL_DIR", "")
        if install_dir:
            repo = Path(install_dir)
            _log.debug("repo from LEMONCROW_INSTALL_DIR: %s", repo)
        else:
            repo = Path(__file__).resolve().parents[4]
            _log.debug("repo from file path: %s", repo)

        if not (repo / ".git").exists():
            _log.debug("not a git checkout - skipping auto-update")
            return  # Not a git checkout, nothing to auto-update

        # The MCP server can remain alive across source updates, leaving the
        # imported ``lemoncrow_version`` at the version it started with. Query
        # the installed executable instead.
        local_version = _installed_cli_version() or lemoncrow_version

        # Fetch latest remote info
        _log.info("fetching latest remote refs from origin...")
        result = subprocess.run(
            ["git", "fetch", "--tags", "--prune", "origin"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            _log.warning("git fetch exited %d: %s", result.returncode, result.stderr.strip())
            return

        default_branch = _detect_default_branch(repo)
        if default_branch is None:
            _log.warning("could not detect default remote branch")
            return

        # Read remote version from pyproject.toml
        result = subprocess.run(
            ["git", "show", f"origin/{default_branch}:pyproject.toml"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            _log.warning(
                "could not read remote pyproject.toml (exit %d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return

        match = re.search(r'^version\s*=\s*"([^"]+)"', result.stdout, re.MULTILINE)
        if not match:
            _log.warning("could not parse version from remote pyproject.toml")
            return

        remote_version = match.group(1)
        if _version_key(remote_version) <= _version_key(local_version):
            _log.info("remote version is not newer; skipping auto-update")
            return

        # Newer version detected - pull and reinstall
        _log.info("newer version detected - pulling %s/%s ...", default_branch, default_branch)
        subprocess.run(
            ["git", "pull", "--ff-only", "origin", default_branch],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        install_script = repo / "scripts" / "install.sh"
        if install_script.exists():
            _log.info("running install script...")
            subprocess.run(
                ["bash", str(install_script), "--local"],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            _log.info("auto-update complete")

            # Write update-state so SessionStart hooks can notify the user.
            # The server's imported version may be stale after the install.
            try:
                from lemoncrow.core.foundation.update_state import write_update_state

                write_update_state(
                    previous_version=local_version,
                    current_version=_installed_cli_version() or remote_version,
                    method="git",
                )
            except Exception:
                _log.exception("failed to write update state")
        else:
            _log.warning("install script not found at %s", install_script)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        _log.exception("auto-update failed")
        with contextlib.suppress(Exception):
            from lemoncrow.core.service.telemetry import emit_product

            emit_product(
                "mcp_auto_update_failed",
                current_version=lemoncrow_version,
                session_id=_get_product_session_id(),
            )


def _run_worker_tick_safe(root: Path) -> None:
    """Process up to 20 pending jobs for *root*.  Run in a daemon thread."""
    try:
        from lemoncrow.core.service.worker import Worker
        from lemoncrow.infra.storage.factory import create_store

        store = create_store(root)
        store.init()
        worker = Worker(store=store)
        for _ in range(20):
            if worker.run_once() is None:
                break
    except Exception:
        logging.exception("Recovered from broad exception handler")
        logger.warning(
            "Suppressed exception in _run_worker_tick_safe",
            exc_info=True,
        )


_last_worker_spawn_time: float = 0.0
_WORKER_SPAWN_THROTTLE_SECS: float = 30.0


def _spawn_worker_if_idle(root: Path) -> None:
    """Spawn a worker thread at most once per throttle window to avoid thread storms."""
    import time

    global _last_worker_spawn_time
    # Serialize the check-and-set so two concurrent light-pool callers can't both
    # pass the throttle window and spawn redundant workers. Spawn the thread
    # OUTSIDE the lock so thread creation doesn't run under _STATE_LOCK.
    with _STATE_LOCK:
        now = time.monotonic()
        if now - _last_worker_spawn_time < _WORKER_SPAWN_THROTTLE_SECS:
            return
        _last_worker_spawn_time = now
    try:
        threading.Thread(
            target=_run_worker_tick_safe,
            args=(root,),
            daemon=True,
        ).start()
    except RuntimeError:
        # Thread creation failed (e.g. OS thread-limit pressure). Don't let the
        # already-advanced throttle clock suppress the next 30s of ticks -- reset
        # it so the next caller can retry the spawn.
        with _STATE_LOCK:
            _last_worker_spawn_time = 0.0
        logging.exception("Recovered from broad exception handler")


_runtime_cache: ContextRuntime | None = None
_context_budget_recorder: Any = None


def _runtime() -> ContextRuntime:
    global _runtime_cache

    from lemoncrow.gateway.adapters.runtime import ContextRuntime

    with _STATE_LOCK:
        if _runtime_cache is None:
            _runtime_cache = ContextRuntime(_lemoncrow_root())
    return _runtime_cache


def _reset_runtime_cache_for_testing() -> None:
    global _product_session_started_at
    global _runtime_cache, _remote_client, _context_budget_recorder
    global _last_worker_spawn_time
    _ledger._current_ledger = None
    _ledger._realtime_ctx = None
    _ledger._product_session_id = None
    _product_session_started_at = None
    _runtime_cache = None
    _remote_client = None
    _context_budget_recorder = None
    _last_worker_spawn_time = 0.0
    _ledger._WINDOW_SID_CACHE = None
    _ledger._MCP_WINDOW_ID = None
    _ledger._MCP_WINDOW_ID_RESOLVED = False
    _last_plan_hash_by_session.clear()
    _last_plan_by_session.clear()
    _CONTEXT_STATE_CACHE.clear()
    _COMPACT_ADVISE_CACHE.clear()
    _last_blocked_plan_hash_by_session.clear()
    _code_engine_cache.clear()
    _scoped_context_cache.clear()


def _live_savings_events_path() -> Path:
    return _lemoncrow_root() / "live_savings_events.jsonl"


# Cap the analytics log: full-file readers (audit_export, advisor, dashboard,
# session_report) O(n)-scan it per render, so unbounded growth is a real cost.
_LIVE_SAVINGS_MAX_BYTES = 8 * 1024 * 1024
_live_savings_dir_ready = False
_live_savings_append_count = 0
# Serializes the dir-ready flag, rotation counter, size-check/rotate, and the
# append so concurrent dispatcher-pool threads can't lose an event on rotation.
_LIVE_SAVINGS_LOCK = threading.Lock()


def _append_live_savings_event(event: dict[str, Any]) -> None:
    """Append a routing / compaction analytics event.

    Display savings ride the MCP response's content[].saved field into the
    transcript and are summed from there. This file remains the log for
    audit_export and cross_vendor_routing.advisor only.
    """
    global _live_savings_dir_ready, _live_savings_append_count
    path = _live_savings_events_path()
    line = json.dumps(event, sort_keys=True) + "\n"
    with _LIVE_SAVINGS_LOCK:
        if not _live_savings_dir_ready:
            # mkdir once per process instead of a syscall on every tool call.
            path.parent.mkdir(parents=True, exist_ok=True)
            _live_savings_dir_ready = True
        _live_savings_append_count += 1
        if _live_savings_append_count % 128 == 0:
            # Periodic size-based rotation (keep one prior generation) so the log
            # cannot grow without bound; checked rarely to avoid a per-call stat().
            try:
                if path.exists() and path.stat().st_size > _LIVE_SAVINGS_MAX_BYTES:
                    path.replace(path.parent / (path.name + ".1"))
            except OSError:
                logging.exception("Recovered from broad exception handler")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def _workspace_savings_path() -> Path:
    """Side log for per-session savings on Copilot CLI and other non-Claude hosts."""

    from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

    workspace = os.environ.get("LEMONCROW_WORKSPACE_ROOT") or os.getcwd()
    return resolve_workspace_store_dir(workspace_root=Path(workspace)) / "session_savings.jsonl"


# session-file + managed-bash helpers moved to mcp.session_state (imported near the top).


def _workspace_bridge_file() -> Path:
    """Workspace-shared identity relay (``session_id`` + ``model``).

    Written by the SessionStart hook. This is the ONLY per-workspace shared
    slot; it carries just the identity bridge used as a resolution fallback.
    All per-session *runtime* state lives under ``sessions/<id>/`` -- see
    :func:`_workspace_session_state_file` -- so concurrent sessions in one
    workspace never clobber each other's workflow/phase/credit state.
    """

    from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

    ws = os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
    return resolve_workspace_store_dir(workspace_root=Path(ws)) / "session_state.json"


def _workspace_session_state_file() -> Path:
    """Per-session runtime state (workflow runtime, ``session_phase``, credit
    hints), keyed by the resolved live session id.

    Previously a single ``workspaces/<hash>/session_state.json`` slot that every
    concurrent session in the workspace overwrote -- so two sessions running
    workflows in one repo corrupted each other's run state. Keying by the live
    session id isolates them. Falls back to the workspace bridge file before a
    session id is known (early startup / hostless callers).
    """
    sid = _resolve_live_session_id()
    if sid:
        from lemoncrow.core.foundation.paths import session_dir

        return session_dir(_lemoncrow_root(), _detect_agent(), sid) / "runtime_state.json"
    return _workspace_bridge_file()


def _read_workspace_session_bridge() -> tuple[str, str]:
    """Read ``(claude_session_id, model)`` from the workspace identity relay.

    Reads the workspace bridge file directly (not the now-per-session runtime
    state), since this is itself a *fallback* source of the live session id.
    """
    try:
        path = _workspace_bridge_file()
        if not path.is_file():
            return "", ""
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return "", ""
        sid = str(data.get("session_id") or "").strip()
        model = str(data.get("model") or "").strip()
        return sid, model
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return "", ""


# Window-anchored live-session resolution. Cached on this window's identity-file
# mtime so the common path (no SessionStart since last call) is a single stat;
# re-resolves only when SessionStart rewrote the file -- i.e. on startup /
# resume / clear / compact.
# moved to mcp.ledger (imported/re-exported near the top).
# This MCP process's (window_pid, window_btime), memoized: the claude-window
# ancestor is fixed for the server's whole life, so walk /proc only once.
# moved to mcp.ledger (imported/re-exported near the top).
# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


# moved to mcp.ledger (imported/re-exported near the top).


def _claude_session_id() -> str:
    """Live session UUID for *this* MCP server process's window.

    Resolved by :func:`_resolve_live_session_id`, which anchors to the window
    process so it stays correct across ``/clear`` and never adopts a sibling
    session's id from the shared workspace bridge. Empty for non-Claude hosts.
    """
    return _resolve_live_session_id()


def _read_workspace_session_state() -> dict[str, Any]:
    try:
        path = _workspace_session_state_file()
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return {}


def _write_workspace_session_state(state: dict[str, Any]) -> None:
    path = _workspace_session_state_file()
    if path == _workspace_bridge_file():
        # No live session id yet (pre-SessionStart / hostless): do NOT write the
        # shared workspace bridge -- that single slot is SessionStart's to own.
        # Runtime state persists only once a per-session home exists.
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: str | None = None
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(state, handle, indent=2)
            tmp_path = handle.name
        Path(tmp_path).replace(path)
    except Exception:
        logging.exception("Recovered from broad exception handler")


# Process-local hint for the fast path in _process_tool_accounting: when False, a
# call that is neither a read nor a credit-enabled code-intel tool has no pending
# credits to age, so it can skip the session_state read+write entirely. Safe
# because one MCP server process owns the session's pending list; re-derived
# after every real pass.
_tool_accounting_pending_hint: bool = True


def _process_tool_accounting(name: str, args: dict[str, Any], result: Any, rid: Any) -> None:
    """Per-call savings accounting that never touches the model-facing response.

    Three honest corrections share ONE session_state read/write:

    1. Read baseline de-dup: outline/range reads both credit ``tokens_saved``
       against the full-file baseline; crediting that more than once per file per
       session double-counts (you can only avoid reading a file once). The 2nd+
       baseline read of a file has its saving zeroed (via the thread-local)
       BEFORE budget recording. Switch: LEMONCROW_READ_BASELINE_DEDUP=0.
    2. Code-intel avoided-read credit: a deferred, observed credit booked to the
       sidecar after an unread observation window
       (``LEMONCROW_CODE_INTEL_CREDIT_AGE`` ticks). Switch: LEMONCROW_CODE_INTEL_CREDIT=0.
    3. Code counterfactual netting: _finish_code_result's vanilla grep+read
       credit draws on the SAME per-session credited-file set as (1), so a file
       surfaced by two code calls — or surfaced then outline-read — bills its
       content avoidance exactly once. Same switch as (1).

    Fast path: a call that is neither a read nor a credit-enabled code-intel tool,
    with no pending code-intel credits to age, skips all session_state I/O.
    """
    global _tool_accounting_pending_hint
    code_intel_on = os.environ.get("LEMONCROW_CODE_INTEL_CREDIT", "1") != "0"
    read_dedup_on = os.environ.get("LEMONCROW_READ_BASELINE_DEDUP", "1") != "0"
    if not code_intel_on and not read_dedup_on:
        return
    try:
        from lemoncrow.core.capabilities import code_intel_credit, read_baseline_credit
        from lemoncrow.pro.capabilities import context_dedup

        is_read = name == "read"
        is_code_intel = code_intel_on and name in code_intel_credit.CODE_INTEL_TOOLS
        counterfactual = getattr(_tool_call_counterfactual, "value", None) if read_dedup_on else None
        if not is_read and not is_code_intel and counterfactual is None and not _tool_accounting_pending_hint:
            return  # nothing to net or record, and no pending credit to age

        # An in-band error means the handler did not actually surface usable
        # content, so its saving is illusory: don't credit it. Covers a
        # top-level `error` and any per-file `error` in a batch read.
        result_errored = isinstance(result, dict) and bool(result.get("error"))

        # The session_state read-modify-write (and the process-wide pending hint)
        # must be serialized: atomic os.replace prevents a torn file but not a
        # lost update when two tool calls interleave RMW. _STATE_LOCK is an
        # RLock, so reentrant helper calls below are safe.
        with _STATE_LOCK:
            state = _read_workspace_session_state()
            # Epoch guard: a compaction resets both ledgers so credits cannot leak
            # across context windows.  A session-id guard resets read_baseline_credited
            # when the Claude Code session changes (new session in same workspace without
            # an intervening compaction), preventing a prior session's credited-file list
            # from zeroing savings in the fresh context window.
            epoch = context_dedup.current_epoch()
            # Use the workspace bridge session_id (updated by the SessionStart hook
            # on every /clear) in preference to _claude_session_id() (the env var,
            # frozen at MCP process launch).  This lets /clear in the same terminal
            # window be detected and the dedup lists be reset for the new context.
            current_sid = str(state.get("session_id") or "").strip() or _claude_session_id()
            epoch_changed = state.get("code_intel_epoch") != epoch
            session_changed = bool(current_sid) and state.get("code_intel_session_id") != current_sid
            if epoch_changed or session_changed:
                state = code_intel_credit.reset_pending(state)
                state = read_baseline_credit.reset(state)
                state["code_intel_epoch"] = epoch
                if current_sid:
                    state["code_intel_session_id"] = current_sid

            if is_read:
                if result_errored:
                    # Errored read earned nothing -> zero its saving outright.
                    _tool_call_tokens_saved.value = 0
                    if isinstance(result, dict):
                        result.pop("tokens_saved", None)
                elif read_dedup_on and isinstance(result, dict):
                    files_out = result.get("files")
                    if isinstance(files_out, list):
                        # Batch read: de-dup each file independently and re-sum the
                        # surviving per-file savings instead of bypassing dedup.
                        surviving = 0
                        for entry in files_out:
                            if not isinstance(entry, dict) or entry.get("error"):
                                continue
                            state, credit = read_baseline_credit.should_credit(
                                state, entry.get("path"), entry.get("mode")
                            )
                            if not credit:
                                entry.pop("tokens_saved", None)
                            else:
                                surviving += int(entry.get("tokens_saved", 0) or 0)
                        _tool_call_tokens_saved.value = surviving
                    else:
                        mode = result.get("mode")
                        path = result.get("path") or args.get("path")
                        state, credit = read_baseline_credit.should_credit(state, path, mode)
                        if not credit:
                            # Baseline already counted this session -> zero the
                            # saving in BOTH the thread-local and the result so
                            # neither the budget recorder nor the `saved` field
                            # (which reads result['tokens_saved'] first) re-counts it.
                            _tool_call_tokens_saved.value = 0
                            result.pop("tokens_saved", None)
                if code_intel_on and not result_errored:
                    read_paths: list[str] = []
                    single = args.get("path")
                    if isinstance(single, str) and single:
                        read_paths.append(single)
                    files = args.get("files")
                    if isinstance(files, list):
                        for entry in files:
                            if isinstance(entry, str) and entry:
                                read_paths.append(entry)
                            elif isinstance(entry, dict):
                                ep = entry.get("path")
                                if isinstance(ep, str) and ep:
                                    read_paths.append(ep)
                    state = code_intel_credit.consume_reads(state, read_paths)
            elif is_code_intel and isinstance(result, dict) and not result_errored:
                paths = code_intel_credit.extract_credited_paths(name, result)
                state = code_intel_credit.record_pending(state, name, paths)

            if counterfactual and not result_errored and isinstance(result, dict):
                # Net the code counterfactual against the shared credited-file
                # set: each file's content avoidance bills once per session,
                # whether a code tool surfaced it or a read avoided it.
                surviving_chars = 0
                for cf_path, cf_chars in dict(counterfactual.get("per_file_chars") or {}).items():
                    state, cf_credit = read_baseline_credit.should_credit_path(state, cf_path)
                    if cf_credit:
                        surviving_chars += int(cf_chars)
                netted = max(0, surviving_chars - int(counterfactual.get("returned_chars") or 0)) // 4
                final = max(int(counterfactual.get("floor_tokens") or 0), netted)
                if final > 0:
                    result["tokens_saved"] = final
                else:
                    result.pop("tokens_saved", None)
                _tool_call_tokens_saved.value = final

            pending_credits: list[dict[str, Any]] = []
            if code_intel_on:
                threshold = int(os.environ.get("LEMONCROW_CODE_INTEL_CREDIT_AGE", "8"))
                state, credits = code_intel_credit.tick_and_credit(state, threshold=threshold)
                pending_credits = list(credits)

            _write_workspace_session_state(state)
            _tool_accounting_pending_hint = bool(state.get("code_intel_pending"))
        # _append_savings writes a separate sidecar (not session_state) and
        # re-enters _STATE_LOCK via _current_context_state, so run it AFTER the
        # session_state lock releases -- keeping the sidecar append + transcript
        # parsing out of the session_state critical section.
        for credit_entry in pending_credits:
            # Deferred credit for an EARLIER call -> sidecar only; never the
            # current response's `saved` field.
            _append_savings(credit_entry["tool"], 0, 1, rid=str(rid))
    except Exception:
        logging.exception("Recovered from broad exception handler")


def _default_workflow_agent_executor(
    step: Any,
    prompt: str,
    context_state: Any,
    *,
    route: Mapping[str, Any] | None = None,
) -> Any:
    import subprocess

    from lemoncrow.core.capabilities.workflow_spawn import build_spawn_envelope, compile_prompt_text
    from lemoncrow.pro.capabilities.cross_vendor_routing.configuration import RouteConfigError
    from lemoncrow.pro.capabilities.cross_vendor_routing.router import NoFeasibleRouteError
    from lemoncrow.pro.capabilities.owned_execution_cache_affinity import (
        cache_affinity_hint,
        latest_cache_affinity,
    )

    workspace = _workspace_root().resolve()
    defaults = build_default_registry()
    decision: Any = None
    route_args = route if isinstance(route, Mapping) else {}
    route_mode = str(route_args.get("mode") or "native").strip() or "native"
    explicit_requested = any(str(route_args.get(field) or "").strip() for field in ("provider", "model", "runner"))
    explicit_requested = explicit_requested or route_mode == "explicit"
    cache_policy: OwnedCachePolicy = "fresh" if str(getattr(step, "context_mode", "") or "") == "fresh" else "inherit"
    compiled_prompt = compile_prompt_text(prompt)
    spawn_plan = context_state.spawn_plan_for_step(str(getattr(step, "step_id", "") or ""))
    spawn_envelope = build_spawn_envelope(
        step_id=str(getattr(step, "step_id", "") or ""),
        role_id=str(getattr(step, "role_id", "") or "general"),
        compiled_prompt=compiled_prompt,
        spawn_group_id=str(spawn_plan.get("spawn_group_id") or ""),
        cache_scope_id=str(spawn_plan.get("cache_scope_id") or ""),
        cache_policy=cache_policy,
    )
    affinity_state = (
        latest_cache_affinity(context_state.step_results, context_state.step_order) if cache_policy == "inherit" else {}
    )
    route_state = {
        "workflow_step": str(getattr(step, "step_id", "") or ""),
        "expected_input_tokens": max(1000, len(spawn_envelope.prompt) // 4),
        "session_phase": "execute",
        "spawn_group_id": spawn_envelope.spawn_group_id,
        "cache_scope_id": spawn_envelope.cache_scope_id,
        **cache_affinity_hint({"cache_affinity": affinity_state}),
    }
    if route_mode != "native":
        try:
            decision = _select_owned_execution_route(
                tool_name="agent",
                task_text=prompt,
                mode=route_mode,
                provider=str(route_args.get("provider") or ""),
                model=str(route_args.get("model") or ""),
                runner=str(route_args.get("runner") or ""),
                cache_policy=cache_policy,
                session_state=route_state,
            )
        except (RouteConfigError, NoFeasibleRouteError) as exc:
            if explicit_requested or route_mode == "auto":
                error = f"owned route selection failed: {exc}"
                return {
                    "status": "failed",
                    "output": "",
                    "output_json": {},
                    "execution_receipt": _native_workflow_execution_receipt(
                        defaults=defaults,
                        role_id=str(getattr(step, "role_id", "") or "general"),
                        compiled_prompt=compiled_prompt,
                        spawn_envelope=spawn_envelope.to_dict(),
                        status="failed",
                        error=error,
                        route_mode=route_mode,
                        attempted_route=True,
                    ),
                    "error": error,
                }
    if decision is not None:
        ledger = _get_ledger()
        try:
            execution = execute_owned_prompt(
                spawn_envelope.prompt,
                root=_lemoncrow_root(),
                tool_name="agent",
                task_text=spawn_envelope.prompt,
                decision=decision,
                host_agent=_detect_agent(),
                session_state=route_state,
                allow_fallback=decision.mode == "auto",
                cache_policy=cache_policy,
                compiled_prompt=compiled_prompt.to_dict(),
                spawn_metadata=spawn_envelope.to_dict(),
            )
        except OwnedExecutionError as exc:
            return {
                "status": "failed",
                "output": "",
                "output_json": {},
                "execution_receipt": exc.receipt.to_dict(),
                "duration_seconds": exc.receipt.duration_seconds,
                "cost_usd": exc.receipt.cost_usd,
                "error": str(exc),
            }
        ledger.record_call(
            operation="owned_execution",
            model=execution.receipt.executed_model,
            input_tokens=execution.receipt.input_tokens,
            output_tokens=execution.receipt.output_tokens,
            cache_read_tokens=execution.receipt.cache_read_input_tokens,
            cache_write_tokens=execution.receipt.cache_write_input_tokens,
            modeled_cache_read_tokens=execution.receipt.modeled_cache_read_input_tokens,
            cost_usd=execution.receipt.cost_usd,
            stable_prefix_hash=execution.receipt.stable_prefix_hash,
            prefix_invalidated_reason=execution.receipt.prefix_invalidated_reason,
            cache_evidence=execution.receipt.cache_evidence,
            phase="workflow",
        )
        return {
            "status": "done",
            "output": execution.output,
            "output_json": _parse_workflow_agent_output(execution.output),
            "execution_receipt": execution.receipt.to_dict(),
            "duration_seconds": execution.receipt.duration_seconds,
            "cost_usd": execution.receipt.cost_usd,
        }
    runner = decision.runner if decision is not None else _workflow_runner_profile()
    model = (
        decision.model
        if decision is not None
        else _workflow_runner_model(
            defaults,
            role_id=str(getattr(step, "role_id", "") or "general"),
            workspace=workspace,
            runner=runner,
        )
    )
    lane_key = ":".join(part for part in (spawn_envelope.spawn_group_id, spawn_envelope.role_id) if part)
    observed_lane = context_state.observed_host_lane(lane_key) if lane_key else {}
    selected_runner = str(observed_lane.get("runner") or runner)
    selected_model = str(observed_lane.get("model") or model or "")
    if lane_key and not observed_lane:
        context_state.record_host_lane(lane_key, {"runner": selected_runner, "model": selected_model})
    command = resolve_swarm_runner_command(
        runner=selected_runner,
        runner_model=selected_model,
        runner_args=(),
        child_command=(),
        prompt_template=spawn_envelope.prompt,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
            timeout=48 * 60 * 60,  # 48h hard ceiling so a hung host CLI can't wedge the run forever
        )
    except subprocess.TimeoutExpired:
        duration_seconds = time.perf_counter() - started
        error = f"native workflow spawn ({selected_runner}) timed out after 48h"
        return {
            "status": "failed",
            "output": "",
            "output_json": {},
            "execution_receipt": _native_workflow_execution_receipt(
                defaults=defaults,
                runner=selected_runner,
                model=selected_model,
                role_id=str(getattr(step, "role_id", "") or "general"),
                compiled_prompt=compiled_prompt,
                spawn_envelope=spawn_envelope.to_dict(),
                status="failed",
                duration_seconds=duration_seconds,
                observed_fields=_observed_host_fields(
                    spawn_envelope=spawn_envelope.to_dict(),
                    selected_runner=selected_runner,
                    selected_model=selected_model,
                ),
                unverified_fields=_unverified_host_fields(selected_model=selected_model),
                error=error,
                route_mode=route_mode,
            ),
            "error": error,
        }
    duration_seconds = time.perf_counter() - started
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        error = (completed.stderr or output or f"{selected_runner} exited with {completed.returncode}").strip()
        return {
            "status": "failed",
            "output": output,
            "output_json": {},
            "execution_receipt": _native_workflow_execution_receipt(
                defaults=defaults,
                runner=selected_runner,
                model=selected_model,
                role_id=str(getattr(step, "role_id", "") or "general"),
                compiled_prompt=compiled_prompt,
                spawn_envelope=spawn_envelope.to_dict(),
                status="failed",
                duration_seconds=duration_seconds,
                observed_fields=_observed_host_fields(
                    spawn_envelope=spawn_envelope.to_dict(),
                    selected_runner=selected_runner,
                    selected_model=selected_model,
                ),
                unverified_fields=_unverified_host_fields(selected_model=selected_model),
                error=error,
                route_mode=route_mode,
            ),
            "error": error,
        }
    return {
        "status": "done",
        "output": output,
        "output_json": _parse_workflow_agent_output(output),
        "execution_receipt": _native_workflow_execution_receipt(
            defaults=defaults,
            runner=selected_runner,
            model=selected_model,
            role_id=str(getattr(step, "role_id", "") or "general"),
            compiled_prompt=compiled_prompt,
            spawn_envelope=spawn_envelope.to_dict(),
            status="done",
            duration_seconds=duration_seconds,
            observed_fields=_observed_host_fields(
                spawn_envelope=spawn_envelope.to_dict(),
                selected_runner=selected_runner,
                selected_model=selected_model,
            ),
            unverified_fields=_unverified_host_fields(selected_model=selected_model),
            route_mode=route_mode,
        ),
    }


def _workflow_runner_profile() -> str:
    detected = _detect_agent()
    if detected in {"claude", "codex", "copilot", "opencode", "lemoncode"}:
        return detected
    return "claude"


def _workflow_runner_model(
    defaults: DefaultRegistry,
    *,
    role_id: str = "general",
    workspace: Path | None = None,
    runner: str | None = None,
) -> str | None:
    resolved_runner = runner or _workflow_runner_profile()
    workspace_model = resolve_host_model(
        resolved_runner,
        role_id,
        workspace_root=workspace,
        fallback=None,
    )
    if workspace_model:
        return normalize_model_for_host(resolved_runner, workspace_model)
    configured = str(_get_mcp_model() or os.environ.get("LEMONCROW_MODEL") or "").strip()
    if configured:
        return normalize_model_for_host(resolved_runner, configured)
    return None


def _native_workflow_execution_receipt(
    *,
    defaults: DefaultRegistry,
    status: str,
    runner: str | None = None,
    model: str | None = None,
    role_id: str = "",
    compiled_prompt: Any | None = None,
    spawn_envelope: dict[str, Any] | None = None,
    duration_seconds: float = 0.0,
    observed_fields: tuple[str, ...] = (),
    unverified_fields: tuple[str, ...] = (),
    error: str = "",
    route_mode: str = "native",
    attempted_route: bool = False,
) -> dict[str, Any]:
    resolved_runner = runner or _workflow_runner_profile()
    resolved_model = model or _workflow_runner_model(defaults) or ""
    resolved_provider = _provider_for_model(resolved_model) if resolved_model else ""
    expose_selection = attempted_route or route_mode == "native"
    compiled = compiled_prompt if hasattr(compiled_prompt, "stable_prefix_hash") else None
    envelope = dict(spawn_envelope or {})
    requested_fields = tuple(str(field) for field in envelope.get("requested_fields", ()))
    honored_fields = ("prompt",)
    dropped_fields = tuple(field for field in requested_fields if field not in honored_fields)
    return {
        "status": status,
        "mode": route_mode,
        "role_id": role_id,
        "selected_provider": resolved_provider if expose_selection else "",
        "selected_model": resolved_model if expose_selection else "",
        "selected_runner": resolved_runner if expose_selection else "",
        "selected_transport": "host-cli" if expose_selection else "",
        "executed_provider": "",
        "executed_model": "",
        "executed_runner": resolved_runner if status == "done" else "",
        "executed_transport": "host-cli" if status == "done" else "",
        "request_id": "",
        "duration_seconds": duration_seconds,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "modeled_cache_read_input_tokens": 0,
        "stable_prefix_hash": getattr(compiled, "stable_prefix_hash", ""),
        "stable_prefix_tokens": getattr(compiled, "stable_prefix_tokens", 0),
        "dynamic_tokens": getattr(compiled, "dynamic_tokens", 0),
        "prefix_invalidated_reason": "cache_policy_fresh" if str(envelope.get("cache_policy") or "") == "fresh" else "",
        "cache_evidence": "hint_only" if getattr(compiled, "stable_prefix_hash", "") else "none",
        "cache_capability": "hint_only" if getattr(compiled, "stable_prefix_hash", "") else "none",
        "spawn_group_id": str(envelope.get("spawn_group_id") or ""),
        "cache_scope_id": str(envelope.get("cache_scope_id") or ""),
        "cache_policy": str(envelope.get("cache_policy") or "inherit"),
        "eligible_for_reuse": bool(
            getattr(compiled, "stable_prefix_hash", "") and str(envelope.get("cache_policy") or "inherit") != "fresh"
        ),
        "reuse_observed": False,
        "spawn_latency_ms": int(duration_seconds * 1000),
        "requested_fields": list(requested_fields),
        "honored_fields": list(observed_fields or honored_fields),
        "dropped_fields": list(dropped_fields),
        "observed_fields": list(observed_fields),
        "unverified_fields": list(unverified_fields),
        "observation_mode": "runtime-observed",
        "cost_usd": 0.0,
        "rerouted": False,
        "attempts": [],
        "error": error,
    }


def _observed_host_fields(
    *,
    spawn_envelope: dict[str, Any],
    selected_runner: str,
    selected_model: str,
) -> tuple[str, ...]:
    # Only fields the host CLI actually receives count as observed/honored: the
    # prompt always crosses the process boundary and the model only when a
    # --model flag is emitted. cache_policy / spawn_group_id / cache_scope_id /
    # role_id are never passed to the subprocess (resolve_swarm_runner_command),
    # so listing them as honored would overstate what the host actually did.
    observed = ["prompt"]
    if selected_runner:
        observed.append("selected_runner")
    if selected_model:
        observed.append("selected_model")
    return tuple(observed)


def _unverified_host_fields(*, selected_model: str) -> tuple[str, ...]:
    fields = ["executed_provider", "executed_transport", "reuse_observed"]
    if selected_model:
        fields.append("executed_model")
    return tuple(fields)


def _parse_workflow_agent_output(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _workflow_spawn_summary(step_results: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "step_count": 0,
        "eligible_for_reuse": 0,
        "reuse_observed": 0,
        "spawn_latency_ms": 0,
        "cache_capability_counts": {},
        "host_dropped_fields": {},
    }
    for step_result in step_results.values():
        receipt = getattr(step_result, "execution_receipt", None)
        if not isinstance(receipt, Mapping):
            continue
        if not any(
            key in receipt
            for key in (
                "cache_capability",
                "spawn_group_id",
                "cache_scope_id",
                "requested_fields",
                "dropped_fields",
            )
        ):
            continue
        summary["step_count"] += 1
        summary["eligible_for_reuse"] += int(bool(receipt.get("eligible_for_reuse", False)))
        summary["reuse_observed"] += int(bool(receipt.get("reuse_observed", False)))
        summary["spawn_latency_ms"] += int(receipt.get("spawn_latency_ms", 0) or 0)
        capability = str(receipt.get("cache_capability") or "").strip()
        if capability:
            counts = cast(dict[str, int], summary["cache_capability_counts"])
            counts[capability] = int(counts.get(capability, 0) or 0) + 1
        dropped = receipt.get("dropped_fields")
        if isinstance(dropped, list | tuple):
            for field in dropped:
                field_name = str(field).strip()
                if not field_name:
                    continue
                drop_counts = cast(dict[str, int], summary["host_dropped_fields"])
                drop_counts[field_name] = int(drop_counts.get(field_name, 0) or 0) + 1
    return summary if summary["step_count"] else {}


def _select_owned_execution_route(
    *,
    tool_name: str,
    task_text: str,
    mode: str,
    provider: str,
    model: str,
    runner: str,
    cache_policy: OwnedCachePolicy = "inherit",
    session_state: Mapping[str, Any] | None = None,
) -> Any:
    return select_owned_route(
        _lemoncrow_root(),
        OwnedRouteRequest(
            tool_name=tool_name,
            task_text=task_text,
            mode="explicit" if mode == "explicit" else "auto",
            provider=provider.strip().lower(),
            model=model.strip(),
            runner=runner.strip().lower(),
            host_agent=_detect_agent(),
            cache_policy="fresh" if cache_policy == "fresh" else "inherit",
            session_state=dict(session_state or {}),
        ),
    )


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip().lower().replace(".", "-")


def _provider_for_model(model_id: str) -> str:
    normalized = _normalize_model_id(model_id)
    if not normalized:
        return ""
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("gemini"):
        return "google"
    return "unknown"


_WORKFLOW_SPAWN_DEPTH_LIMIT = 8
# Per-worker-thread workflow spawn depth. The dispatcher runs a thread pool, so
# tracking depth in os.environ races across parallel workflow steps and can leak
# a stale value into the process env; a threading.local isolates it per thread.
_workflow_spawn_depth: threading.local = threading.local()
# Tools that themselves spawn sub-agents/workflows: invoking them from a
# workflow step opens an unbounded recursive spawn path, so they are blocked.
_WORKFLOW_SPAWNING_TOOLS = frozenset({"workflow"})


def _default_workflow_tool_executor(step: Any, args: dict[str, Any], context_state: Any) -> Any:
    if step.tool in _WORKFLOW_SPAWNING_TOOLS:
        raise ValueError(f"workflow steps cannot invoke spawning tool {step.tool!r} (unbounded recursion)")
    spec = TOOLS.get(step.tool)
    if spec is None:
        raise ValueError(f"unknown workflow tool: {step.tool}")
    depth = getattr(_workflow_spawn_depth, "value", 0)
    if depth >= _WORKFLOW_SPAWN_DEPTH_LIMIT:
        raise ValueError(
            f"workflow spawn depth limit ({_WORKFLOW_SPAWN_DEPTH_LIMIT}) exceeded; aborting recursive tool execution"
        )
    handler = cast(Callable[[dict[str, Any]], Any], spec["handler"])
    _workflow_spawn_depth.value = depth + 1
    try:
        return handler(args)
    finally:
        _workflow_spawn_depth.value = depth


def _default_workflow_shell_executor(step: Any, command: str, forked_context: dict[str, Any]) -> Any:
    if getattr(step, "fork_from", None):
        # The shell tool handler has no parameter to receive forked context, so a
        # fork_from on a shell step would be silently dropped. Reject it loudly
        # rather than give the author a false sense that the fork took effect.
        raise ValueError("workflow shell steps do not support fork_from (the shell tool cannot receive forked context)")
    spec = TOOLS.get("bash")
    if spec is None:
        raise ValueError("bash tool not registered")
    handler = cast(Callable[[dict[str, Any]], Any], spec["handler"])
    return handler({"command": command})


def _run_owned_workflow(arguments: dict[str, Any]) -> dict[str, Any]:
    from lemoncrow.core.capabilities.workflow_context import WorkflowContextState
    from lemoncrow.core.capabilities.workflow_runner import WorkflowRunner
    from lemoncrow.core.capabilities.workflow_schema import workflow_definition_from_mapping

    resume = bool(arguments.get("resume", False))
    # Hold _STATE_LOCK across the whole read-modify-write: atomic os.replace
    # prevents a torn file but not a lost update when a concurrent handler
    # interleaves its own session_state RMW. _STATE_LOCK is an RLock, so the
    # reentrant helper calls below are safe.
    with _STATE_LOCK:
        session_state = _read_workspace_session_state()
        runtime_state = _workflow_runtime_state(session_state)

        workflow_raw = arguments.get("workflow")
        if resume and not isinstance(workflow_raw, Mapping):
            workflow_raw = runtime_state.get("workflow")
        if not isinstance(workflow_raw, Mapping):
            raise ValueError("workflow run requires workflow mapping")
        route_raw = arguments.get("route")
        if resume and not isinstance(route_raw, Mapping):
            route_raw = runtime_state.get("route")
        route = dict(route_raw) if isinstance(route_raw, Mapping) else {}
        review_raw = arguments.get("plan_review")
        plan_review = dict(review_raw) if isinstance(review_raw, Mapping) else {}
        review_decision = _coerce_workflow_review_decision(plan_review)
        definition = workflow_definition_from_mapping(workflow_raw)
        workflow_state = (
            dict(session_state.get("workflow") or {}) if isinstance(session_state.get("workflow"), dict) else {}
        )
        runner_state = (
            WorkflowContextState.from_mapping(runtime_state.get("runner")) if resume else WorkflowContextState()
        )
        runner = WorkflowRunner(
            agent_executor=lambda step, prompt, context_state: _default_workflow_agent_executor(
                step,
                prompt,
                context_state,
                route=route,
            ),
            tool_executor=_default_workflow_tool_executor,
            shell_executor=_default_workflow_shell_executor,
        )
        ledger = _get_ledger()
        result = runner.run(
            definition,
            context_state=runner_state,
            ledger=ledger,
            plan_review_decision=review_decision,
        )
        spawn_summary = _workflow_spawn_summary(result.step_results)
        created_at = str(runtime_state.get("created_at") or "").strip() if resume else ""
        runtime_state = {
            "run_id": result.run_id,
            "workflow_id": definition.workflow_id,
            "workflow": dict(workflow_raw),
            "route": dict(route),
            "status": result.status,
            "step_order": list(result.step_order),
            "current_step": result.paused_step_id
            or result.failed_step_id
            or (result.step_order[-1] if result.step_order else ""),
            "failed_step_id": result.failed_step_id or "",
            "paused_step_id": result.paused_step_id or "",
            "artifact_ids": [],
            "created_at": created_at or datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "runner": runner_state.to_dict(),
        }
        if spawn_summary:
            runtime_state["spawn_summary"] = dict(spawn_summary)
        if result.status == "awaiting_review":
            workflow_state["current_step"] = "review"
            workflow_state["session_phase"] = "review"
            runtime_state["plan_review"] = {
                "decision": review_decision or "pending",
                "paused_step_id": result.paused_step_id or "",
                "workflow_id": definition.workflow_id,
            }
            ledger.record_workflow_event(
                "plan_review",
                {
                    "workflow_step": "review",
                    "review_decision": "pending",
                    "workflow_id": definition.workflow_id,
                    "step_id": result.paused_step_id or "",
                },
            )
        elif result.status == "review_rejected":
            workflow_state["current_step"] = "review"
            workflow_state["session_phase"] = "review"
            runtime_state["plan_review"] = {
                "decision": review_decision or "revise",
                "paused_step_id": result.paused_step_id or "",
                "workflow_id": definition.workflow_id,
            }
            ledger.record_workflow_event(
                "plan_review",
                {
                    "workflow_step": "review",
                    "review_decision": review_decision or "revise",
                    "workflow_id": definition.workflow_id,
                    "step_id": result.paused_step_id or "",
                },
            )
        else:
            workflow_state["current_step"] = "execution"
            workflow_state["session_phase"] = "execute"
            if review_decision:
                runtime_state["plan_review"] = {
                    "decision": review_decision,
                    "workflow_id": definition.workflow_id,
                }
            if review_decision:
                ledger.record_workflow_event(
                    "plan_review",
                    {
                        "workflow_step": "review",
                        "review_decision": review_decision,
                        "workflow_id": definition.workflow_id,
                    },
                )
        workflow_state["current_task"] = {
            "workflow_id": definition.workflow_id,
            "run_id": result.run_id,
            "step_id": result.paused_step_id
            or result.failed_step_id
            or (result.step_order[-1] if result.step_order else ""),
        }
        workflow_state["task_outputs"] = {
            step_id: step_result.to_dict() for step_id, step_result in result.step_results.items()
        }
        if spawn_summary:
            workflow_state["spawn_summary"] = dict(spawn_summary)
            ledger.record_workflow_event("spawn_summary", dict(spawn_summary))
        if result.status in {"awaiting_review", "review_rejected"}:
            workflow_state["plan_review"] = {
                "decision": review_decision or "pending",
                "paused_step_id": result.paused_step_id or "",
                "workflow_id": definition.workflow_id,
            }
        elif review_decision:
            workflow_state["plan_review"] = {
                "decision": review_decision,
                "workflow_id": definition.workflow_id,
            }
        else:
            workflow_state.pop("plan_review", None)
        workflow_state["updated_at"] = datetime.now(UTC).isoformat()
        session_state["workflow"] = workflow_state
        _write_workflow_runtime_state(session_state, runtime_state)
        _write_workspace_session_state(session_state)
        ledger.persist()
    receipt = {
        "run_id": result.run_id,
        "status": result.status,
        "step_count": len(result.step_order),
        "artifact_ids": [],
    }
    if spawn_summary:
        receipt["spawn_summary"] = dict(spawn_summary)
    if result.failed_step_id:
        receipt["failed_step_id"] = result.failed_step_id
    if result.paused_step_id:
        receipt["paused_step_id"] = result.paused_step_id
    return receipt


WORKFLOW_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": ["run", "status", "inspect", "pause", "resume", "stop"],
        },
        "workflow": {"type": "object"},
        "run_id": {"type": "string"},
        "route": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["native", "auto", "explicit"]},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "runner": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "plan_review": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "revise", "rerun"]},
            },
            "additionalProperties": False,
        },
        "pause_reason": {"type": "string"},
        "stop_reason": {"type": "string"},
    },
    "required": ["op"],
    "additionalProperties": False,
}


@mcp_tool(name="agent")
def tool_agent(
    prompt: Annotated[
        str,
        Field(description="Full task/instruction for the sub-agent."),
    ],
    budget: Annotated[
        str,
        Field(description="Cost/quality tier: 'cheap' | 'balanced' | 'best'. Default 'balanced'."),
    ] = "balanced",
    provider: Annotated[
        str,
        Field(description="Force a provider (e.g. 'anthropic'); empty = auto-select from configured vendors."),
    ] = "",
    model: Annotated[
        str,
        Field(description="Force a model id; empty = auto-pick by budget."),
    ] = "",
    cache_policy: Annotated[
        str,
        Field(
            description="'inherit' = share the prompt-cache scope with prior owned spawns (cheaper); 'fresh' = new scope."
        ),
    ] = "inherit",
) -> dict[str, Any]:
    """Spawn an LemonCrow-owned sub-agent, return its result.

    Runs on LemonCrow's owned-execution runtime: picks provider + model from
    credentials already configured (provider API key when present, else the
    installed host CLI), executes the prompt, shares a prompt-cache scope with
    sibling spawns when ``cache_policy='inherit'``. Prefer over the host
    ``Agent`` tool when LemonCrow should control the sub-agent's model, cost,
    and cache affinity.
    """
    from lemoncrow.pro.capabilities.cross_vendor_routing.configuration import RouteConfigError

    root = _workspace_root()
    session_state = _read_workspace_session_state()
    norm_cache: OwnedCachePolicy = "fresh" if str(cache_policy).strip().lower() == "fresh" else "inherit"
    use_explicit = bool(provider.strip() and model.strip())
    request = OwnedRouteRequest(
        tool_name="agent",
        task_text=prompt,
        mode="explicit" if use_explicit else "auto",
        budget=cast(Any, str(budget).strip().lower() or "balanced"),
        provider=provider.strip(),
        model=model.strip(),
        host_agent=_detect_agent(),
        cache_policy=norm_cache,
        session_state=session_state,
    )
    try:
        decision = select_owned_route(root, request)
    except (NoFeasibleRouteError, RouteConfigError) as exc:
        return {
            "isError": True,
            "status": "no_route",
            "message": (
                f"No owned-execution route available: {exc}. Configure a route config (route.yaml) "
                "plus a provider API key in the environment or an installed host CLI, and enable "
                "owned routing."
            ),
        }
    try:
        result = execute_owned_prompt(
            prompt,
            root=root,
            tool_name="agent",
            task_text=prompt,
            decision=decision,
            host_agent=_detect_agent(),
            session_state=session_state,
            cache_policy=norm_cache,
        )
    except OwnedExecutionError as exc:
        return {
            "isError": True,
            "status": "failed",
            "message": str(exc),
            "receipt": exc.receipt.to_dict(),
        }
    receipt = result.receipt
    return {
        "status": receipt.status,
        "output": result.output,
        "provider": receipt.executed_provider,
        "model": receipt.executed_model,
        "transport": receipt.executed_transport,
        "cost_usd": receipt.cost_usd,
        "tokens": {
            "input": receipt.input_tokens,
            "output": receipt.output_tokens,
            "cache_read": receipt.cache_read_input_tokens,
            "cache_write": receipt.cache_write_input_tokens,
        },
        "cache": {
            "evidence": receipt.cache_evidence,
            "reuse_observed": receipt.reuse_observed,
            "scope_id": receipt.cache_scope_id,
        },
    }


@mcp_tool(name="workflow", input_schema=WORKFLOW_TOOL_INPUT_SCHEMA)
def tool_workflow(
    op: str,
    workflow: dict[str, Any] | None = None,
    run_id: str | None = None,
    route: dict[str, Any] | None = None,
    plan_review: dict[str, Any] | None = None,
    pause_reason: str | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Run or inspect LemonCrow's durable workflow runtime.

    Ops:
      run     — execute a workflow synchronously from fresh runtime state
      status  — persisted runtime state for this workspace
      inspect — spawn/cache receipts for the persisted runtime
      pause   — mark persisted runtime paused (live synchronous call not cancelled)
      resume  — continue persisted runtime with its stored workflow + route
      stop    — mark persisted runtime stopped (live synchronous call not cancelled)
    """
    normalized_op = op.strip().lower()
    if normalized_op == "run":
        return _run_owned_workflow({"workflow": workflow or {}, "route": route or {}, "plan_review": plan_review or {}})
    # Hold _STATE_LOCK across the read so the pause/stop read-modify-write below
    # cannot lose a concurrent handler's session_state update. _STATE_LOCK is an
    # RLock; the resume branch reacquires it reentrantly via _run_owned_workflow.
    with _STATE_LOCK:
        session_state = _read_workspace_session_state()
        if normalized_op == "status":
            return _coerce_workflow_runtime_status(session_state)
        if normalized_op == "inspect":
            return _inspect_workflow_runtime(session_state)
        if normalized_op not in {"pause", "resume", "stop"}:
            return {
                "isError": True,
                "status": "unsupported_op",
                "message": f"unsupported workflow op: {op}",
            }
        _require_active_workflow_runtime(session_state, run_id or "")
        if normalized_op == "resume":
            arguments: dict[str, Any] = {"resume": True, "plan_review": plan_review or {}}
            if workflow is not None:
                arguments["workflow"] = workflow
            if route is not None:
                arguments["route"] = route
            return _run_owned_workflow(arguments)
        if normalized_op == "pause":
            _pause_workflow_runtime(
                session_state,
                run_id=run_id or "",
                pause_reason=str(pause_reason or ""),
            )
            _write_workspace_session_state(session_state)
            return _coerce_workflow_runtime_status(session_state)
        if normalized_op == "stop":
            _stop_workflow_runtime(
                session_state,
                run_id=run_id or "",
                stop_reason=str(stop_reason or ""),
            )
            _write_workspace_session_state(session_state)
            return _coerce_workflow_runtime_status(session_state)
    raise AssertionError(f"unreachable workflow op: {op!r}")  # op is guaranteed pause/resume/stop above


def _inspect_workflow_runtime(session_state: dict[str, Any]) -> dict[str, Any]:
    from lemoncrow.core.capabilities.workflow_context import WorkflowContextState

    status = _coerce_workflow_runtime_status(session_state)
    runtime_state = _workflow_runtime_state(session_state)
    runner_state = WorkflowContextState.from_mapping(runtime_state.get("runner"))
    step_spawns: list[dict[str, Any]] = []
    for step_id in runner_state.step_order:
        step_result = runner_state.step_results.get(step_id)
        if step_result is None:
            continue
        receipt = step_result.execution_receipt
        if not isinstance(receipt, Mapping):
            continue
        if not any(
            key in receipt
            for key in (
                "cache_capability",
                "spawn_group_id",
                "cache_scope_id",
                "requested_fields",
                "dropped_fields",
            )
        ):
            continue
        step_spawns.append(
            {
                "step_id": step_id,
                "status": step_result.status,
                "mode": str(receipt.get("mode") or ""),
                "role_id": str(receipt.get("role_id") or ""),
                "cache_capability": str(receipt.get("cache_capability") or ""),
                "eligible_for_reuse": bool(receipt.get("eligible_for_reuse", False)),
                "reuse_observed": bool(receipt.get("reuse_observed", False)),
                "spawn_latency_ms": int(receipt.get("spawn_latency_ms", 0) or 0),
                "spawn_group_id": str(receipt.get("spawn_group_id") or ""),
                "cache_scope_id": str(receipt.get("cache_scope_id") or ""),
                "requested_fields": list(receipt.get("requested_fields") or []),
                "honored_fields": list(receipt.get("honored_fields") or []),
                "dropped_fields": list(receipt.get("dropped_fields") or []),
            }
        )
    spawn_summary = runtime_state.get("spawn_summary") if isinstance(runtime_state.get("spawn_summary"), dict) else {}
    return {
        **status,
        "spawn_summary": spawn_summary,
        "step_spawns": step_spawns,
    }


def _is_registrable_workspace(ws: Path) -> bool:
    """Reject ``$HOME`` and ``/`` as workspace roots.

    Registering either lets the code-warm daemon (``code_warm.py``) try to
    reindex that entire tree on every ~15s poll, forever: such a root never
    finishes indexing (or writes a ``session_state.json``), so it also never
    becomes eligible for ``lc code prune`` to reclaim -- it just burns a
    CPU core indefinitely for as long as the registering process stays
    alive. Concretely: a host launched with cwd/``CLAUDE_WORKSPACE_ROOT``
    pointed at ``$HOME`` (or ``/``) rather than a repo.
    """
    return ws != Path.home() and ws != Path(ws.anchor)


def _register_mcp_session() -> None:
    """Create this MCP process's registration file if it doesn't exist yet."""
    f = _mcp_session_file()
    if f.exists():
        return
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        from lemoncrow.core.foundation.paths import is_recognized_workspace as _is_recognized_workspace

        configured = os.environ.get("CLAUDE_WORKSPACE_ROOT", "").strip()
        if configured:
            # An explicit host env var is authoritative (matches
            # resolve_workspace_root's own precedence) -- still subject to
            # the $HOME/`/` sanity check below, but not the git-repo check.
            ws_path = Path(configured).resolve()
        else:
            cwd = Path(os.getcwd()).resolve()
            if not _is_recognized_workspace(cwd):
                _log.warning(
                    "Refusing to register MCP session workspace %s: not a git "
                    "repository and not registered via `lc init` -- would "
                    "make the code-warm daemon index an arbitrary directory "
                    "forever. Run `lc init` here, or launch from inside a "
                    "git repository.",
                    cwd,
                )
                return
            ws_path = cwd
        if not _is_registrable_workspace(ws_path):
            _log.warning(
                "Refusing to register MCP session workspace %s: it's $HOME "
                "or / itself, not a single project -- would make the "
                "code-warm daemon reindex it forever. Set "
                "CLAUDE_WORKSPACE_ROOT to the specific repo.",
                ws_path,
            )
            return
        ws = str(ws_path)
        from lemoncrow.core.foundation.paths import workspace_key as _workspace_key

        data = {
            "lemoncrow_mcp_id": _MCP_ID,
            "pid": os.getpid(),
            "workspace": ws,
            "workspace_hash": _workspace_key(ws),
            "started_at": datetime.utcnow().isoformat(),
            "claude_session_id": "",
            "model": "",
            "managed_bash": [],
        }
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        # Best-effort sidecar registration; a failed write must not break startup.
        _log.debug("MCP session registration write failed", exc_info=True)


def _unregister_mcp_session() -> None:
    """Remove this MCP process's registration file on clean shutdown."""
    try:
        _mcp_session_file().unlink(missing_ok=True)
    except OSError:
        _log.debug("MCP session registration cleanup failed", exc_info=True)


# moved to mcp.ledger (imported/re-exported near the top).


def _get_mcp_model() -> str:
    """Return the model string last written by SessionStart, or empty string."""
    if not _ledger._cached_claude_session_id:
        # Try to populate both caches via workspace bridge read.
        _get_claude_session_id()

    # Re-read model from workspace bridge on each call — SessionStart may fire
    # again on resume/compact with a different model. Only trust it when the
    # bridge belongs to this session; otherwise a sibling session sharing the
    # workspace could hand us a wrong model. The live transcript model (preferred
    # in _append_savings) covers the common case; this is a pre-first-turn fallback.
    sid, model = _read_workspace_session_bridge()
    if sid and model and sid == _resolved_host_session_id():
        _ledger._cached_mcp_model = model
        return _ledger._cached_mcp_model

    # Backward-compatible fallback to MCP session file.
    try:
        f = _mcp_session_file()
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _ledger._cached_mcp_model = str(data.get("model") or "").strip()
    except (OSError, json.JSONDecodeError):
        _log.debug("MCP model read failed", exc_info=True)
    return _ledger._cached_mcp_model


# Native per-session id env vars for non-Claude hosts, tried in order after the
# window-anchored Claude resolver. Module-level so _resolved_host_session_id and
# _get_host_session_sidecar_path share one source of truth.
_HOST_SESSION_ENVS: list[tuple[str, str]] = [
    ("CODEX_SESSION_ID", "codex"),
    ("OPENCODE_SESSION_ID", "opencode"),
    ("GITHUB_COPILOT_SESSION_ID", "copilot"),
    ("CURSOR_SESSION_ID", "cursor"),
    ("CURSOR_TRACE_ID", "cursor"),
    ("HERMES_SESSION_ID", "hermes"),
    ("ANTIGRAVITY_SESSION_ID", "antigravity"),
    ("AGY_SESSION_ID", "antigravity"),
]


def _workspace_bridge_session_id() -> str:
    """Session id from the workspace-shared ``session_state.json`` bridge.

    Codex/OpenCode never set a launch-time session env var for this (long-lived)
    MCP server process (see ``_HOST_SESSION_ENVS`` -- that assumption doesn't
    hold for either host in practice). Their hooks instead refresh
    ``workspaces/<hash>/session_state.json`` with the live session_id on every
    hook event (``plugin_runtime._write_codex_session_state`` /
    ``_write_opencode_session_state``). This is the same file
    ``savings_summary._resolve_workspace_session_id`` reads as a read-side
    fallback; here it closes the write-side gap so savings rows land under the
    real session instead of the unattributed quarantine ledger.

    The slot is workspace-shared and last-writer-wins across every host's
    hooks, so the sid is only trusted when the writer stamped a ``host`` label
    that matches this process's host -- otherwise an OpenCode server could
    adopt a Claude/Codex sid (or vice versa) and write savings under a phantom
    session. Claude never uses this fallback at all: it has a window-anchored
    resolver, and adopting a shared slot would cross-contaminate concurrent
    windows in one repo (see :func:`_resolved_host_session_id`). Legacy bridge
    files without a ``host`` stamp are rejected (fail closed -> quarantine).
    """
    try:
        host = _detect_agent()
        if not host or host == "claude":
            # Claude resolves via the window-anchored resolver; a workspace-shared
            # slot would cross-contaminate concurrent windows in one repo.
            return ""
        from lemoncrow.core.foundation.paths import resolve_workspace_store_dir

        ws = os.environ.get("CLAUDE_WORKSPACE_ROOT") or os.getcwd()
        path = resolve_workspace_store_dir(workspace_root=Path(ws)) / "session_state.json"
        if not path.is_file():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ""
        if str(data.get("host") or "").strip() != host:
            # Written by another host's hook (or a legacy hook with no stamp):
            # fail closed -> callers divert to the quarantine ledger.
            return ""
        return str(data.get("session_id") or "").strip()
    except Exception:
        return ""


def _resolved_host_session() -> tuple[str, str]:
    """Resolved ``(session_id, host)`` for the current host, or empty strings."""
    # Singleton daemon: the caller's identity rides in the request headers. The
    # daemon serves every window of a workspace and is not a child of any of
    # them, so its own window/env resolution below names whichever session
    # started it -- crediting every other window's savings to that one session
    # (or, with no launch env id, to the unattributed quarantine ledger) while
    # the live window's statusline reads its own empty sidecar and shows $0.
    req_sid, req_host = _request_session_identity()
    if req_sid:
        return req_sid, req_host or _detect_agent()
    sid = _resolve_live_session_id()
    if sid:
        return sid, "claude"
    claude_sid = os.environ.get("CLAUDE_CODE_SESSION_ID", "").strip()
    if claude_sid:
        return claude_sid, "claude"
    for env_var, host in _HOST_SESSION_ENVS:
        env_sid = os.environ.get(env_var, "").strip()
        if env_sid:
            return env_sid, host
    bridge_sid = _workspace_bridge_session_id()
    if bridge_sid:
        return bridge_sid, _detect_agent()
    return "", ""


def _resolved_host_session_id() -> str:
    """Resolved per-session id for the current host, or ``""`` when unknown.

    Window-anchored live id (Claude — stable across /clear, isolated per window)
    first, then the native host session env vars. An empty result is a hard
    signal that callers MUST fail closed: attributing savings to a
    workspace-shared slot would cross-contaminate concurrent windows in one repo
    (the exact mixing this guards against).
    """
    return _resolved_host_session()[0]


def _get_host_session_sidecar_path() -> Path:
    """Return the per-session savings sidecar path for the current host.

    The sidecar MUST land under the *currently active* session, because the
    readers (Stop hook, statusline, session report) are always handed the
    active session id by the host and look there.  The MCP server process is
    long-lived and survives many /clear cycles, so its launch-time
    ``CLAUDE_CODE_SESSION_ID`` goes stale the moment the user first runs /clear
    and stays stale for the rest of the process's life.

    Resolution is delegated to :func:`_resolved_host_session_id` (window-anchored
    Claude id, then native host env ids). When nothing resolves it falls back to
    the workspace-shared path -- but callers that must not cross-attribute gate
    on :func:`_resolved_host_session_id` first: :func:`_append_savings` diverts
    to the per-workspace quarantine ledger and :func:`_write_statusline_sidecar`
    skips, so neither ever writes that shared slot.
    """
    sid, host = _resolved_host_session()
    if sid:
        cache_key = f"{host}:{sid}"
        if cache_key not in _SAVINGS_SIDECAR_PATH_BY_SID:
            from lemoncrow.core.foundation.paths import session_dir

            _SAVINGS_SIDECAR_PATH_BY_SID[cache_key] = (
                session_dir(_lemoncrow_root(), host or _detect_agent(), sid) / "savings.jsonl"
            )
        return _SAVINGS_SIDECAR_PATH_BY_SID[cache_key]
    return _workspace_savings_path()


def _current_context_state() -> tuple[int, str]:
    """Measured (context size, model) from the host transcript's last usage entry.

    Context size is input + cache_read + cache_creation tokens of the most
    recent usage entry; model is the one that produced it — the per-turn ground
    truth, unlike the SessionStart bridge which goes stale when the user
    switches models mid-session via /model. Returns (0, "") when no
    transcript/usage is available. Callers must treat 0/"" as "unknown" and
    skip pricing — never synthesize values.

    Cached on the candidate transcripts' (path, mtime_ns, size) signature: this
    runs on every savings-bearing tool call, so we re-tail and JSON-parse the
    64 KB window only when a transcript actually changed.
    """
    try:
        from lemoncrow.core.capabilities.savings_summary import (
            claude_transcript_candidates,
            is_real_model,
        )

        sid = _claude_session_id()
        if sid:
            candidates = list(claude_transcript_candidates(sid))
            sig_parts: list[tuple[str, int, int]] = []
            for cand in candidates:
                try:
                    st = os.stat(cand)
                except OSError:
                    continue
                sig_parts.append((str(cand), st.st_mtime_ns, st.st_size))
            sig = tuple(sig_parts)

            with _STATE_LOCK:
                cached = _CONTEXT_STATE_CACHE.get(sid)
            if cached is not None and cached[0] == sig:
                return cached[1]

            from lemoncrow.gateway.hosts.context_state import _tail_lines

            result: tuple[int, str] = (0, "")
            for cand in candidates:
                try:
                    tail_lines = _tail_lines(cand)
                except OSError:
                    continue
                best = 0
                best_model = ""
                for line in tail_lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # first line of the tail window may be partial
                    msg = entry.get("message") or {}
                    usage = msg.get("usage") if isinstance(msg, dict) else None
                    if not isinstance(usage, dict):
                        continue
                    ctx = (
                        int(usage.get("input_tokens", 0) or 0)
                        + int(usage.get("cache_read_input_tokens", 0) or 0)
                        + int(usage.get("cache_creation_input_tokens", 0) or 0)
                    )
                    if ctx > 0:
                        best = ctx
                        candidate = str(msg.get("model") or "").strip()
                        if is_real_model(candidate):
                            best_model = candidate
                if best > 0:
                    result = (best, best_model)
                    break

            with _STATE_LOCK:
                _CONTEXT_STATE_CACHE[sid] = (sig, result)
            return result

        # --- Non-Claude hosts (OpenCode, Codex): read from workspace bridge ---
        bridge_sid, host = _resolved_host_session()
        if bridge_sid and host and host != "claude":
            cache_key = f"bridge:{host}:{bridge_sid}"
            from lemoncrow.core.foundation.paths import session_dir

            # Freshness signature over the bridge stats file (mtime+size) so a
            # cached entry is recomputed when the host plugin updates it.
            stats_path = session_dir(_lemoncrow_root(), host, bridge_sid) / "stats.json"
            try:
                bst = os.stat(stats_path)
                sig = ((str(stats_path), bst.st_mtime_ns, bst.st_size),)
            except OSError:
                sig = ()
            with _STATE_LOCK:
                cached = _CONTEXT_STATE_CACHE.get(cache_key)
            if cached is not None and cached[0] == sig:
                return cached[1]
            result = _bridge_context_state(bridge_sid, host)
            if result[0] > 0:
                with _STATE_LOCK:
                    _CONTEXT_STATE_CACHE[cache_key] = (sig, result)
            return result

    except Exception:
        logging.exception("Recovered from broad exception handler")
        _log.debug("context state probe failed", exc_info=True)
    return 0, ""


def _bridge_context_state(session_id: str, host: str) -> tuple[int, str]:
    """Read context tokens from the workspace bridge or host-specific data.

    For non-Claude hosts (OpenCode, Codex) that don't produce Claude-format
    transcript JSONL, this reads the session stats file maintained by the host
    plugin.
    """
    try:
        from lemoncrow.core.foundation.paths import session_dir

        root = _lemoncrow_root()
        stats_path = session_dir(root, host, session_id) / "stats.json"
        if not stats_path.is_file():
            return 0, ""
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0, ""
        if not isinstance(stats, dict):
            return 0, ""
        # Prefer the bounded current-turn context size (this turn's real,
        # context-window-capped request size -- Codex populates this from its
        # transcript's last_token_usage; see _codex_transcript_snapshot).
        # Falling back to state["usage"] is a LAST RESORT: that field is a
        # cumulative running total across the whole session and grows
        # unboundedly turn over turn, which would price every avoided call as
        # if it re-sent the entire session's tokens instead of one request.
        ctx = int(stats.get("current_turn_context_tokens", 0) or 0)
        if ctx <= 0:
            _usage = stats.get("usage")
            usage = _usage if isinstance(_usage, dict) else {}
            ctx = (
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_read_tokens", 0) or 0)
                + int(usage.get("cache_write_tokens", 0) or 0)
            )
        model = str(stats.get("last_model") or stats.get("model") or "").strip()
        return ctx, model
    except Exception:
        _log.debug("bridge context state probe failed for %s/%s", host, session_id, exc_info=True)
    return 0, ""


def _price_avoided_calls_usd(
    model: str,
    calls_saved: int,
    ctx_tokens: int,
    *,
    avg_output_tokens: int = 0,
    long_context: bool = False,
) -> float:
    """Price avoided tool-call round trips: context re-read + the turn's output.

    Each avoided call is an API round trip that would have (a) re-read the
    current context (ctx_tokens, measured from the host transcript) at the
    cache-read rate, and (b) produced the session's average per-turn output
    (avg_output_tokens, measured) — billed at the output rate and
    re-entering context as a cache write on the next turn. Per-request flat
    semantics, at the >200k premium rate when the live window itself ran
    long-context. Unknown model or unmeasured context → 0.0 (no guess).
    """
    if calls_saved <= 0 or ctx_tokens <= 0 or not model or model == "_default":
        return 0.0
    from lemoncrow.core.capabilities.pricing import get_model_pricing

    pricing = get_model_pricing(model)
    if pricing is None or not pricing.known or pricing.cache_read <= 0:
        return 0.0
    out = max(0, int(avg_output_tokens))
    return pricing.request_cost_usd(
        cache_read_tokens=int(calls_saved) * int(ctx_tokens),
        output_tokens=int(calls_saved) * out,
        cache_write_tokens=int(calls_saved) * out,
        long_context=long_context,
    )


def _session_avg_output_tokens() -> int:
    """Measured average output tokens per assistant turn for the live session.

    0 = unknown (no transcript yet); callers price the output component of an
    avoided roundtrip only when this is measured — never a synthesized value.
    """
    try:
        from lemoncrow.core.capabilities.savings_summary import (
            claude_transcript_candidates,
            read_transcript_stats,
        )

        sid = _claude_session_id()
        if not sid:
            return 0
        paths = list(claude_transcript_candidates(sid))
        if not paths:
            return 0
        stats = read_transcript_stats(paths[0])
        if stats is None:
            return 0
        # output_tokens (thinking + text + tool_use JSON — everything billed at
        # the output rate) includes subagent transcripts, so divide by ALL
        # assistant turns (main + subagent) or the average inflates whenever
        # subagents ran.
        turns = stats.turns + sum(len(sub) for sub in stats.subagent_turn_timestamps)
        if turns <= 0:
            return 0
        return max(0, int(stats.output_tokens / turns))
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return 0


def _savings_long_context(model: str, ctx_tokens: int) -> bool:
    """True when the live window is past *model*'s >200k premium threshold.

    Savings earned there are worth the same premium rates the cost side bills
    for those requests (long-context parity). Models without a premium tier
    on their rate card (threshold 0) always return False.
    """
    if ctx_tokens <= 0 or not model or model == "_default":
        return False
    try:
        from lemoncrow.core.capabilities.pricing import get_model_pricing

        pricing = get_model_pricing(model)
    except Exception:
        return False
    if pricing is None or not pricing.known:
        return False
    threshold = pricing.long_context_threshold()
    return threshold > 0 and ctx_tokens > threshold


def _append_savings(tool_name: str, tokens_saved: int, calls_saved: int, rid: str = "") -> None:
    """Append one per-call savings row to ``sessions/<id>/savings.jsonl``.

    This single sidecar is the source of truth read by the live statusline,
    the stop hook, and the session report. Each row carries the raw
    ``tokens``/``calls`` plus the pre-priced ``cost_saved_usd`` /
    ``calls_usd`` so analytics readers need not re-price.
    """
    # A latency-profile run (`lc perf`) drives `_handle` with synthetic
    # probe calls; those aren't work the agent avoided, so don't credit them.
    if os.environ.get("LEMONCROW_TOOL_PROFILE_PATH"):
        return
    if tokens_saved <= 0 and calls_saved <= 0:
        return
    # Fail closed for ATTRIBUTION, not for ACCOUNTING: with no resolvable
    # session id the sidecar path would fall back to a workspace-shared file
    # that every concurrent window in this repo reads, cross-attributing one
    # window's savings to another. But dropping the row silently understates
    # every historical window (whole resumed sessions have lost 100+ credited
    # calls this way). Route it to a per-workspace quarantine ledger under
    # sessions/ instead: the day-bucketed aggregate counts any savings.jsonl
    # there (see savings_summary._scan_savings_files), while per-session
    # readers glob by exact session id and never see it.
    unattributed = not _resolved_host_session_id()
    if unattributed:
        _log.debug("savings append unattributed: unresolved session id (quarantine ledger)")
    else:
        _register_mcp_session()
    ts = datetime.utcnow().isoformat()
    # Per-turn model truth from the transcript beats the SessionStart bridge,
    # which goes stale when the user switches models mid-session via /model.
    ctx_tokens, live_model = _current_context_state()
    model = live_model or _get_mcp_model()
    # Long-context parity: past the model's >200k premium threshold every
    # request bills at premium rates, so tokens/calls saved there are worth
    # the premium too. Stamped on the row so carry pricing
    # (savings_summary._carry_credit) applies the same rate later.
    long_ctx = _savings_long_context(model, ctx_tokens)
    calls_usd = 0.0
    if calls_saved > 0 and ctx_tokens > 0:
        calls_usd = round(
            _price_avoided_calls_usd(
                model,
                calls_saved,
                ctx_tokens,
                avg_output_tokens=_session_avg_output_tokens(),
                long_context=long_ctx,
            ),
            6,
        )
    cost_saved = round(_price_tokens_saved_usd(model, tokens_saved, long_context=long_ctx), 6)
    try:
        if unattributed:
            from lemoncrow.core.foundation.paths import session_dir

            path = (
                session_dir(_lemoncrow_root(), _detect_agent(), f"unattributed-{_workspace_ws_hash()}")
                / "savings.jsonl"
            )
        else:
            path = _get_host_session_sidecar_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "tool": tool_name,
            # Field names match the in-response `saved: {tokens, calls}` shape.
            # The file lives under sessions/<id>/ so "savings" is implicit
            # from context — no need to suffix the keys.
            "tokens": int(tokens_saved),
            "calls": int(calls_saved),
            "model": model,
            "ts": ts,
        }
        if long_ctx:
            entry["long_context"] = True
        if unattributed:
            # Diagnosable marker: the row counts in windowed totals but belongs
            # to no session; readers that key by session ignore this file.
            entry["unattributed"] = True
        # Pre-priced USD so the session report / analytics need not re-price.
        # Omitted when zero to keep rows lean; readers treat missing as 0.
        if cost_saved > 0:
            entry["cost_saved_usd"] = cost_saved
        if calls_usd > 0:
            entry["calls_usd"] = calls_usd
            entry["ctx_tokens"] = ctx_tokens
        if rid:
            entry["rid"] = rid
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        # Fold the new row into the in-memory historical window totals (O(1)).
        # Invalidating instead would force a full sessions/** re-scan on the
        # next statusline render for every append; the cache's normal TTL
        # still refreshes the totals to pick up other processes' writes.
        try:
            from lemoncrow.core.capabilities.savings_summary import _bump_historical_savings_cache as _bump_cache

            _bump_cache(entry)
        except ImportError:
            pass
    except Exception:
        logging.exception("Recovered from broad exception handler")
        # Best-effort savings sidecar; a failed write must not break the tool call.
        _log.debug("savings sidecar append failed", exc_info=True)
        return
    # Refresh the statusline sidecar so statusline.sh can read it without a
    # subprocess.  Rate-limited internally; never raises.
    _write_statusline_sidecar()


def _append_workspace_savings(tool_name: str, tokens_saved: int, calls_saved: int, rid: str = "") -> None:
    """Backward-compat shim — delegates to _append_savings."""
    _append_savings(tool_name, tokens_saved, calls_saved, rid=rid)


_STATUSLINE_SIDECAR_MIN_INTERVAL: float = 5.0  # seconds — rate-limit transcript reads
_statusline_wake = threading.Event()
_statusline_worker_lock = threading.Lock()
_statusline_worker_started = False
# (session_id, host) pairs awaiting a segment write. Captured on the DISPATCH
# thread — under the singleton daemon the request session context is a
# thread-local the worker thread cannot see, and one daemon serves many
# windows, so the worker must be told which sessions to render for.
_statusline_pending: set[tuple[str, str]] = set()
_statusline_pending_lock = threading.Lock()


def _statusline_sidecar_loop() -> None:
    """Daemon worker: compute + write the statusline sidecar off the tool-call thread.

    savings_frames on a long session costs real CPU (sidecar decode + carry
    credit); computed inline in the tool handler it landed as multi-second
    stalls on whatever query ran concurrently (profiled at 3-4s on a large
    dev session before the _window_carry rewrite). Wake-ups coalesce: at most
    one compute per _STATUSLINE_SIDECAR_MIN_INTERVAL, latest wins.
    """
    while True:
        _statusline_wake.wait()
        _statusline_wake.clear()
        with _statusline_pending_lock:
            pending = sorted(_statusline_pending)
            _statusline_pending.clear()
        for sid, host in pending:
            try:
                _write_statusline_sidecar_now(sid, host)
            except Exception:
                _log.debug("statusline sidecar write failed", exc_info=True)
        # Rate-limit AFTER the write: the first event renders immediately and a
        # burst coalesces into one trailing refresh (wakes set during compute or
        # sleep re-enter the loop, so the sidecar never goes stale after a burst).
        time.sleep(_STATUSLINE_SIDECAR_MIN_INTERVAL)


def _write_statusline_sidecar() -> None:
    """Signal the statusline worker; never compute on the caller's thread."""
    global _statusline_worker_started
    # Resolve the session HERE, on the dispatch thread that still carries the
    # request context; the worker thread would resolve the daemon's own window.
    sid, host = _resolved_host_session()
    if not sid:
        return
    with _statusline_pending_lock:
        _statusline_pending.add((sid, host or _detect_agent()))
    if not _statusline_worker_started:
        with _statusline_worker_lock:
            if not _statusline_worker_started:
                threading.Thread(
                    target=_statusline_sidecar_loop, name="lemoncrow-statusline-sidecar", daemon=True
                ).start()
                _statusline_worker_started = True
    _statusline_wake.set()


def _write_statusline_sidecar_now(session_id: str = "", host: str = "") -> None:
    """Write the current savings segment to sessions/<resolved_id>/statusline_segment.

    Runs on the statusline worker thread after every burst of savings events so
    statusline.sh can read the pre-computed segment directly without spawning a
    subprocess.  Uses the same resolved session id as the savings sidecar, so
    the statusline and MCP server are always in sync regardless of /clear or
    --resume.
    """
    # Fail closed: an unresolved session id would otherwise resolve to the
    # workspace-shared slot and publish one window's segment for every sibling.
    sid = session_id or _resolved_host_session_id()
    if not sid:
        return
    try:
        from lemoncrow.core.foundation.paths import session_dir

        seg_dir = session_dir(_lemoncrow_root(), host or _detect_agent(), sid)
        from lemoncrow.core.capabilities.savings_summary import savings_frames

        frames = savings_frames(session_id=sid)
        if frames:
            seg_dir.mkdir(parents=True, exist_ok=True)
            # All frames, one per line: statusline.sh picks by wall clock so
            # rotation continues BETWEEN sidecar writes instead of freezing on
            # whichever single frame was current at write time.
            (seg_dir / "statusline_frames").write_text("\n".join(frames) + "\n", encoding="utf-8")
            # Legacy single-frame sidecar for older installed statusline.sh.
            idx = int(time.time() // 5) % len(frames)
            (seg_dir / "statusline_segment").write_text(frames[idx], encoding="utf-8")
    except Exception:
        _log.debug("statusline sidecar write failed", exc_info=True)


_dev_mode_cache: bool | None = None


def _mcp_debug_enabled() -> bool:
    if os.environ.get("LEMONCROW_MCP_DEBUG", "0") not in ("0", "", "false", "no"):
        return True
    # Auto-enable in dev installations (marker written by make dev / scripts/local.sh).
    # The marker is constant per process, so cache the stat instead of issuing
    # a syscall on every tool call.
    global _dev_mode_cache
    if _dev_mode_cache is None:
        try:
            _dev_mode_cache = (_lemoncrow_root() / ".dev_mode").exists()
        except Exception:
            _dev_mode_cache = False
    return _dev_mode_cache


def _mcp_debug_path(session_id: str = "") -> Path:
    """Return the per-session debug log path.

    When a session_id is known, writes land at::

        ~/.lemoncrow/sessions/<session_id>/mcp_debug.jsonl

    so each Claude Code session has its own isolated log that can be read,
    tailed, or deleted independently.  Falls back to a top-level
    ``mcp_debug_unknown.jsonl`` for the rare case where the session id is not
    yet resolved (early boot calls).
    """
    root = _lemoncrow_root()
    if session_id:
        from lemoncrow.core.foundation.paths import session_dir

        return session_dir(root, _detect_agent(), session_id) / "mcp_debug.jsonl"
    return root / "mcp_debug_unknown.jsonl"


# Argument keys whose values carry credentials/PII (DSNs, tokens, passwords).
# These are masked regardless of length before anything is written to the debug
# log or shipped to telemetry.
_DEBUG_SECRET_KEY_RE = re.compile(
    r"(connection_string|dsn|api[_-]?key|token|secret|password|authorization)",
    re.IGNORECASE,
)

# Keys whose large values are still truncated in the *telemetry* path
# (Langfuse / OTel) to keep event payloads small.  The local mcp_debug.jsonl
# always logs full content — only secrets are redacted there.
_DEBUG_LARGE_KEYS = frozenset({"new_string", "old_string", "content", "prompt", "task", "query"})


def _scrub_args_for_debug(args: dict[str, Any]) -> dict[str, Any]:
    """Sanitize tool args before logging or telemetry emission.

    Secrets (connection_string / dsn / api_key / token / secret / password /
    authorization) are always masked, regardless of value length.
    """

    def _scrub_value(key: str | None, value: Any) -> Any:
        if isinstance(key, str) and _DEBUG_SECRET_KEY_RE.search(key):
            return "<redacted>"
        if isinstance(value, dict):
            return {kk: _scrub_value(kk, vv) for kk, vv in value.items()}
        if isinstance(value, list):
            return [_scrub_value(None, item) for item in value]
        return value

    return {k: _scrub_value(k, v) for k, v in args.items()}


# Cleanup: remove mcp_debug.jsonl files from sessions older than this many days.
_MCP_DEBUG_RETENTION_DAYS = 7
# Track last prune time to avoid pruning every single tool call.
_mcp_debug_last_prune: float = 0.0
_MCP_DEBUG_PRUNE_INTERVAL_S = 3600.0  # prune at most once per hour


def _prune_mcp_debug_logs() -> None:
    """Delete per-session mcp_debug.jsonl files older than _MCP_DEBUG_RETENTION_DAYS.

    Runs at most once per hour (guarded by ``_mcp_debug_last_prune``).  Only
    the debug file itself is removed — the session directory and its other
    artefacts (savings.jsonl, etc.) are left untouched.  Fail-open.
    """
    global _mcp_debug_last_prune
    now = time.time()
    if now - _mcp_debug_last_prune < _MCP_DEBUG_PRUNE_INTERVAL_S:
        return
    _mcp_debug_last_prune = now
    try:
        sessions_dir = _lemoncrow_root() / "sessions"
        if not sessions_dir.is_dir():
            return
        cutoff = now - _MCP_DEBUG_RETENTION_DAYS * 86400
        for debug_file in sessions_dir.glob("*/mcp_debug.jsonl"):
            try:
                if debug_file.stat().st_mtime < cutoff:
                    debug_file.unlink(missing_ok=True)
            except Exception:
                pass
        # Also prune the fallback unknown-session file if it is old.
        unknown = _lemoncrow_root() / "mcp_debug_unknown.jsonl"
        try:
            if unknown.exists() and unknown.stat().st_mtime < cutoff:
                unknown.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception:
        pass  # never disrupt the server


def _tool_profile_path() -> Path | None:
    """Per-run tool-latency sink, or None when profiling is off.

    Set ``LEMONCROW_TOOL_PROFILE_PATH`` to a writable JSONL path (the benchmark
    points it at a file next to each run's .flow capture) to record per-call
    timing scoped to a single run -- unlike the global ``mcp_debug.jsonl``, which
    mixes every workspace and session. Unset in production -> no-op, no I/O.
    """
    raw = os.environ.get("LEMONCROW_TOOL_PROFILE_PATH", "").strip()
    return Path(raw) if raw else None


def _append_tool_profile(
    *,
    tool: str,
    handler_ms: int,
    total_ms: int,
    response_size: int,
    status: str,
    session_id: str = "",
    error: str | None = None,
) -> None:
    """Append one per-call latency record to the per-run profile sink.

    ``handler_ms`` is the tool handler itself; ``total_ms`` covers the whole
    dispatch incl. the post-handler pipeline (render/dedup/spill/compact/token
    ledger), so ``overhead_ms = total_ms - handler_ms`` exposes server-side cost
    the handler-only timer misses. One sub-4KB line per call -> POSIX-atomic
    append, safe across the server's worker threads. Fail-open.
    """
    path = _tool_profile_path()
    if path is None:
        return
    try:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "tool": tool,
            "handler_ms": handler_ms,
            "total_ms": total_ms,
            "overhead_ms": max(0, total_ms - handler_ms),
            "response_size": response_size,
            "status": status,
            "session_id": session_id,
        }
        if error:
            entry["error"] = error
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
    except Exception:
        pass  # never disrupt the server


def _append_mcp_debug_event(
    *,
    tool: str,
    args: dict[str, Any],
    duration_ms: int,
    response_size: int,
    status: str,
    error: str | None = None,
    session_id: str = "",
    rid: str | None = None,
) -> None:
    """Write a per-call debug record to sessions/<session_id>/mcp_debug.jsonl.

    Only active when LEMONCROW_MCP_DEBUG=1 (or in dev-mode installs).  Each
    session writes to its own file so logs are isolated, greppable per-session,
    and cleaned up independently.  Args are fully logged (only credentials are
    redacted) so the debug log is genuinely useful.  Fail-open.
    """
    if not _mcp_debug_enabled():
        return
    try:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "tool": tool,
            "mcp_tool": f"mcp__{SERVER_NAME}__{tool}",
            "args": _scrub_args_for_debug(args),
            "duration_ms": duration_ms,
            "response_size_bytes": response_size,
            "status": status,
            "session_id": session_id,
        }
        if rid is not None:
            entry["rid"] = rid
        if error:
            entry["error"] = error
        path = _mcp_debug_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        # Opportunistic cleanup -- at most once per hour, non-blocking.
        _prune_mcp_debug_logs()
    except Exception:
        pass  # never disrupt the server


# smart_state read/write moved to mcp.smart_state (imported near the top).


def _coerce_saved_tokens(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return int(max(0.0, value))
    if isinstance(value, dict):
        return sum(
            int(max(0.0, float(item_value)))
            for item_value in value.values()
            if isinstance(item_value, (int, float)) and not isinstance(item_value, bool)
        )
    return 0


def _extract_compact_output_tokens_saved(result: dict[str, Any]) -> int:
    return _coerce_saved_tokens(result.get("tokens_saved_vs_naive"))


def _extract_tokens_saved(result: dict[str, Any]) -> int:
    direct = _coerce_saved_tokens(result.get("tokens_saved"))
    if direct > 0:
        return direct
    # Check thread-local written by tool handlers that strip tokens_saved before returning
    tl = getattr(_tool_call_tokens_saved, "value", 0)
    if tl > 0:
        return tl
    return _extract_compact_output_tokens_saved(result)


# bash symbols moved to mcp.bash (imported/registered near the top).


def _trimmed_tokens_saved(pre_chars: int, post_chars: int) -> int:
    """Tokens credited for dispatcher-level result trimming (spill/compact/truncate).

    The trimmed bytes were already rendered and would have entered the host
    prompt 1:1 -- but the host inlines at most ~50KB of MCP output before
    dumping the rest to a file, so the naive baseline is capped there. Returns
    0 when nothing was trimmed or the final text already exceeds that cap.
    """
    return max(0, min(pre_chars, _HOST_INLINE_RESULT_CHARS) - post_chars) // 4


# smart_state flock moved to mcp.smart_state (imported near the top).


def _record_smart_state_savings(tokens_saved: int, calls_avoided: int) -> None:
    if tokens_saved <= 0 and calls_avoided <= 0:
        return
    # Serialize the read-modify-write across threads (_STATE_LOCK) AND sibling
    # MCP processes sharing the machine-global smart_state.json (flock), so a
    # concurrent process can't lose-update the cumulative counters.
    with _STATE_LOCK:
        _flock = _acquire_smart_state_flock()
        try:
            state = _read_smart_state()
            savings = state.get("savings")
            if not isinstance(savings, dict):
                savings = {"calls_avoided": 0, "tokens_saved": 0}
            savings["calls_avoided"] = int(savings.get("calls_avoided", 0) or 0) + max(0, calls_avoided)
            savings["tokens_saved"] = int(savings.get("tokens_saved", 0) or 0) + max(0, tokens_saved)
            state["savings"] = savings
            _write_smart_state(state)
        finally:
            _release_smart_state_flock(_flock)


# ── Per-command bash spend ledger (`lc audit bash`) ─────────────────────
# For every normalized bash command family, accumulate how many chars its
# output still SHIPPED into context after all compaction next to how many it
# OMITTED. The top shipped rows are the compaction gaps worth new filters --
# the `rtk discover` equivalent, built from LemonCrow's own ledger.
# bash symbols moved to mcp.bash (imported/registered near the top).
# bash symbols moved to mcp.bash (imported/registered near the top).
# bash symbols moved to mcp.bash (imported/registered near the top).
# bash symbols moved to mcp.bash (imported/registered near the top).
# bash symbols moved to mcp.bash (imported/registered near the top).
# bash symbols moved to mcp.bash (imported/registered near the top).


# bash symbols moved to mcp.bash (imported/registered near the top).


# bash symbols moved to mcp.bash (imported/registered near the top).


class _NoOpContextBudgetRecorder:
    """No-op recorder for service-backed MCP state."""

    def record(self, **kwargs: Any) -> None:
        pass

    def record_compact_tool_output(self, **kwargs: Any) -> None:
        pass

    def aggregate_run(self, session_id: str) -> Any:
        return {}


def _get_context_budget_recorder() -> Any:
    global _context_budget_recorder
    if _service_backed_state():
        return _NoOpContextBudgetRecorder()
    if _context_budget_recorder is None:
        try:
            from lemoncrow.core.capabilities.telemetry.context_budget import ContextBudgetRecorder
            from lemoncrow.infra.storage.factory import create_store

            store = create_store(_lemoncrow_root())
            store.init()
            _context_budget_recorder = ContextBudgetRecorder(store)
        except Exception:
            logging.exception("Recovered from broad exception handler")
            _context_budget_recorder = _NoOpContextBudgetRecorder()
    return _context_budget_recorder


_REDACTION_PLACEHOLDER_RE = re.compile(r"<redacted[^>]*>")


def _core_runtime() -> Any:
    return _runtime().core_runtime


def _redact_memory_input(text: str, field_name: str) -> str:
    if _REDACTION_PLACEHOLDER_RE.search(text):
        return text
    redacted = redact(text)
    if not text:
        return redacted
    remaining = _REDACTION_PLACEHOLDER_RE.sub("", redacted)
    if len(remaining.strip()) < len(text.strip()) * 0.5:
        raise ValueError(f"{field_name} rejected: likely secret leakage")
    return redacted


_memory_store_tls = threading.local()


def _memory_store() -> Any:
    # Reuse the store (its sqlite connection + idempotent migrations) per thread
    # instead of constructing a fresh one on every memory call. Thread-local
    # because the sqlite connection has check_same_thread=True; keyed by root.
    cache = getattr(_memory_store_tls, "by_root", None)
    if cache is None:
        cache = {}
        _memory_store_tls.by_root = cache
    root = str(_lemoncrow_root())
    store = cache.get(root)
    if store is None:
        store = make_memory_store(_lemoncrow_root())
        cache[root] = store
    return store


def _archival_recall() -> ArchivalRecallCapability:
    from lemoncrow.infra.embeddings.factory import make_embedder
    from lemoncrow.pro.capabilities.archival_recall import ArchivalRecallCapability

    return ArchivalRecallCapability(_memory_store(), make_embedder(), redactor=redact)


def _symbol_recall() -> Any:
    from lemoncrow.core.foundation.history_store import HistoryStore
    from lemoncrow.pro.capabilities.archival_recall.symbol_recall import SymbolRecallCapability

    workspace_root = _workspace_root()
    trace_store = HistoryStore(_lemoncrow_root())
    trace_store.init()
    return SymbolRecallCapability(
        repo_root=workspace_root,
        engine=_code_context_engine(str(workspace_root)),
        memory_store=_memory_store(),
        trace_store=trace_store,
    )


def _workspace_path(file_path: str) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        return p
    workspace = os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd())
    return Path(workspace) / p


# N10 — per-request project isolation. The stdio MCP server normally serves a
# single workspace resolved from env/cwd. A request may instead carry an
# ``Mcp-Project-Path``-style override (via ``params._meta`` or a reserved
# ``project_path`` arg); when present we serve that repo for the duration of the
# request only. The executor runs each request on its own worker thread, so a
# thread-local override is naturally request-scoped and keeps per-repo isolation.
# Absent override -> today's single-workspace behavior is unchanged.
_request_project: threading.local = threading.local()


def _http_project_override_allowed() -> bool:
    """H1 opt-in: whether a wire-supplied project override may be honored at all.

    Default OFF. Resolving an arbitrary host directory from the wire lets a
    caller pivot the server outside its workspace, so acceptance is gated behind
    ``LEMONCROW_HTTP_ALLOW_PROJECT_OVERRIDE`` AND confined to the workspace root.
    """
    return os.environ.get("LEMONCROW_HTTP_ALLOW_PROJECT_OVERRIDE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _project_override_root() -> Path:
    """The allowlisted root a wire-supplied project override must stay inside."""
    root = (
        os.environ.get("CLAUDE_WORKSPACE_ROOT")
        or os.environ.get("LEMONCROW_WORKSPACE_ROOT")
        or os.environ.get("VSCODE_CWD")
        or os.getcwd()
    )
    return Path(root).expanduser().resolve()


def _is_within_root(candidate: Path, root: Path) -> bool:
    """True iff ``candidate`` is ``root`` or nested under it (no parent escape)."""
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _set_request_project(path: str | None) -> str | None:
    """Set the request-scoped project override; return the prior value.

    Only an existing directory is accepted; anything else clears the override so
    a bad header can never silently mis-route to a non-repo path.

    H1 — a wire-supplied path can point ANYWHERE on the host. Acceptance is
    therefore gated behind the explicit ``LEMONCROW_HTTP_ALLOW_PROJECT_OVERRIDE``
    opt-in AND confined to the workspace root: a candidate outside the root is
    rejected (override cleared) so the default never pivots out of bounds.
    """
    prior = getattr(_request_project, "value", None)
    resolved: str | None = None
    if isinstance(path, str) and path.strip() and _http_project_override_allowed():
        try:
            candidate = Path(path).expanduser().resolve()
            if candidate.is_dir() and _is_within_root(candidate, _project_override_root()):
                resolved = str(candidate)
        except OSError:
            resolved = None
    _request_project.value = resolved
    return prior


def _clear_request_project(prior: str | None) -> None:
    _request_project.value = prior


def _extract_request_project(params: dict[str, Any], args: dict[str, Any]) -> str | None:
    """Pull an ``Mcp-Project-Path``-style override from a tools/call request.

    Checks, in order: ``params._meta`` (the MCP metadata channel) under any of
    the conventional keys, then a reserved ``project_path`` argument. The arg is
    popped so it never reaches the tool handler. Returns ``None`` when absent so
    the default single-workspace path is preserved.
    """
    meta = params.get("_meta")
    if isinstance(meta, dict):
        for key in ("mcp-project-path", "projectPath", "project_path", "projectRoot", "project_root"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if isinstance(args, dict):
        value = args.pop("project_path", None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _workspace_root() -> Path:
    # A request-scoped project override (N10) wins for the lifetime of the
    # request; everything else falls back to the single-workspace resolution.
    override = getattr(_request_project, "value", None)
    if isinstance(override, str) and override:
        return Path(override)
    workspace = (
        os.environ.get("CLAUDE_WORKSPACE_ROOT")
        or os.environ.get("LEMONCROW_WORKSPACE_ROOT")
        or os.environ.get("VSCODE_CWD")
        or os.getcwd()
    )
    return Path(workspace)


_last_session_cwd: str | None = None
"""Most recent explicit ``cwd`` seen on a ``bash`` call, or None.

This server is a long-lived process whose own ``os.getcwd()`` is fixed at
launch, and MCP carries no per-call session cwd (we implement no ``roots``
capability). So when a session enters a git worktree, :func:`_workspace_root`
keeps naming the *main checkout*. A bash call's explicit ``cwd`` is the only
place the session's real working directory reaches this process, which is why
bash honored the worktree and edit did not. Recorded here and used to resolve
relative edit paths -- see :func:`_session_worktree_root`.

lc-debt: process-global, so a shared HTTP-mode server hosting two sessions on
the SAME repo could let one session's worktree cwd redirect the other's
relative edits (a foreign repo is already rejected by the
``<root>/.git/worktrees`` check, and the redirect is always disclosed).
Upgrade path: key this by MCP session id once the dispatcher threads one
through, or drop it entirely if the protocol ever carries a per-call cwd.
"""


def _record_session_cwd(name: str, args: Any) -> None:
    """Remember a bash call's cwd, the only session-cwd signal this server gets."""
    global _last_session_cwd
    if name != "bash" or not isinstance(args, dict):
        return
    cwd = args.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        _last_session_cwd = cwd.strip()


def _session_worktree_root(workspace_root: Path) -> Path | None:
    """The linked git worktree the session is working in, else None.

    Detected without spawning git: a linked worktree's ``.git`` is a *file*
    holding ``gitdir: <path>``, and for a worktree of THIS repo that path lives
    under ``<workspace_root>/.git/worktrees/``. A normal checkout has ``.git``
    as a directory, which ends the walk immediately.

    Returns None for every uncertain case -- no recorded cwd, a plain
    directory, a worktree belonging to a different repo -- so resolution falls
    back to the workspace root exactly as before.
    """
    recorded = _last_session_cwd
    if not recorded:
        return None
    try:
        candidate = Path(recorded).expanduser().resolve()
        if not candidate.is_dir():
            return None
        root = workspace_root.resolve()
        worktrees_dir = (root / ".git" / "worktrees").resolve()
        for directory in (candidate, *candidate.parents):
            marker = directory / ".git"
            if marker.is_dir():
                return None  # a normal checkout, not a linked worktree
            if not marker.is_file():
                continue
            gitdir = marker.read_text(encoding="utf-8").strip()
            if not gitdir.startswith("gitdir:"):
                return None
            target = Path(gitdir.split(":", 1)[1].strip()).resolve()
            if not target.is_relative_to(worktrees_dir):
                return None  # a worktree, but of some other repo
            return None if directory == root else directory
    except OSError:
        return None
    return None


# Thread-local slot for passing real tokens_saved from tool handlers to the
# budget recorder without polluting the LLM-facing response dict.
# _tool_call_tokens_saved moved to mcp.smart_state (imported/re-exported).
# Per-call counterfactual breakdown stamped by _finish_code_result when its
# vanilla grep+read arm wins the max(): {"per_file_chars": {path: chars},
# "returned_chars": int, "floor_tokens": int}. _process_tool_accounting nets
# it against the shared per-session credited-file set (read_baseline_credit)
# so one file's content avoidance never bills twice across mechanisms.
_tool_call_counterfactual: threading.local = threading.local()
_tool_call_rendered_text: threading.local = threading.local()
# Raw structured result of the last tool call, stashed per-call for the in-process
# `lc tools call ... --json` CLI to recover the full dict. Never serialized
# into the host-facing MCP response, so the host's main model only sees `content`.
_tool_call_raw_result: threading.local = threading.local()
# Image content blocks produced by reading an image file, drained into the
# tools/call response `content` so the multimodal model receives the actual image
# instead of a text description it can't act on. Reset per tools/call; filled by
# _smart_read_single.
_tool_call_images: threading.local = threading.local()
_MAX_INLINE_IMAGE_BYTES = 4 * 1024 * 1024

_ARCHIVE_SUFFIXES = frozenset({".gz", ".bz2", ".xz", ".zst", ".tgz", ".tbz2"})
_ARCHIVE_MEDIA_TYPES = frozenset(
    {
        "application/zip",
        "application/x-tar",
        "application/x-7z-compressed",
        "application/vnd.rar",
        "application/x-rar-compressed",
    }
)


def _binary_read_message(media_type: str, size_bytes: int, suffix: str = "") -> str:
    """Type-aware guidance for a binary file `read` can't decode as text.

    Every category used to fall through to the same "not UTF-8 text, not
    decoded" dead end regardless of what the file actually was. Each branch
    here instead names a concrete next step the agent can take with tools it
    already has (bash + this same `read` tool) -- e.g. extract a video frame
    and read() *that* as an image, rather than leaving the agent to rediscover
    ffmpeg/pdftoppm/tesseract from scratch the way real trials did.
    """
    base = f"Binary file ({media_type}, {size_bytes} bytes) -- not UTF-8 text, not decoded."
    if media_type.startswith("image/"):
        return base + f" Image too large to inline (> {_MAX_INLINE_IMAGE_BYTES} bytes)."
    if media_type.startswith("video/"):
        return (
            base + " Video not viewable. Extract a frame -- "
            "`ffmpeg -ss <time> -i <file> -frames:v 1 frame.png` -- then read() it."
        )
    if media_type.startswith("audio/"):
        return base + " Audio is not transcribed by this tool; no transcript path is available here."
    if media_type == "application/pdf":
        return base + " PDF not extracted. `pdftotext` for text, or `pdftoppm -png` a page and read() the PNG."
    if media_type in _ARCHIVE_MEDIA_TYPES or suffix in _ARCHIVE_SUFFIXES:
        return (
            base + " Archive contents are not listed by this tool; use bash (e.g. `unzip -l`, `tar -tf`) to inspect it."
        )
    return base


def _bootstrap_context_status(root: Path) -> dict[str, Any]:
    from lemoncrow.core.service.bootstrap_context import bootstrap_status, missing_bootstrap_labels
    from lemoncrow.core.service.jobs import JOB_BOOTSTRAP_CONTEXT
    from lemoncrow.infra.storage.factory import create_store
    from lemoncrow.pro.capabilities.code_context import CodeContextEngine

    repo_root = _workspace_root().resolve()
    repo_id = CodeContextEngine(repo_root).repo_id
    memory_store = _memory_store()
    state = bootstrap_status(memory_store, repo_id)
    store = create_store(root)
    store.init()
    jobs = [
        job
        for job in store.jobs.list_jobs(job_type=JOB_BOOTSTRAP_CONTEXT, limit=200)
        if isinstance(job.get("payload"), dict) and job["payload"].get("repo_id") == repo_id
    ]
    queued = False
    # Only block re-queueing if there is an already-active (pending or running) job.
    # Failed/dead jobs should not permanently prevent retrying bootstrap.
    active_job = next((job for job in jobs if job["status"] in {"pending", "running"}), None)
    job_id: str | None = None
    if state != "warm" and active_job is None:
        job_id = store.jobs.enqueue_job(
            JOB_BOOTSTRAP_CONTEXT,
            {"repo_root": str(repo_root), "repo_id": repo_id},
        )
        queued = True
    status = "warm" if state == "warm" else ("warming" if queued or active_job or job_id else state)
    return {
        "repo_id": repo_id,
        "queued": queued,
        "job_id": job_id,
        "status": status,
        "missing_labels": missing_bootstrap_labels(memory_store, repo_id),
    }


@mcp_tool(name="context")
def tool_get_context(
    task: str,
    domain: str | None = None,
    files: list[str] | None = None,
    keywords: list[str] | None = None,
    excluded_paths: list[str] | None = None,
    tools: list[str] | None = None,
    errors: list[str] | None = None,
    max_blocks: int = 5,
    token_budget: int | None = 2000,
    dedup: bool = True,
    agent_id: str | None = None,
    recall: bool = True,
    mode: Literal["procedures", "symbols", "pull"] = "procedures",
) -> dict[str, Any]:
    """Record task context and retrieve relevant Playbooks for the task.

    Call at task start to seed context with prior procedures, repo bootstrap
    knowledge, and per-agent memory. mode="symbols" returns the most relevant
    code symbols/files from the code index instead; mode="pull" returns scoped
    subtask context (files/keywords/excluded_paths scope it).

    Args: task (required) drives ranking; domain narrows retrieval; files boost
    related blocks; tools/errors rank matching procedure and rescue blocks;
    max_blocks (default 5); token_budget (default 2000, None = unlimited);
    dedup; agent_id loads per-agent memory; recall=False skips memory recall.
    """
    if mode == "symbols":
        engine = _code_context_engine(".")
        return cast(
            dict[str, Any],
            engine.tool_context(
                task=task,
                seed_files=files or [],
                budget_tokens=token_budget or 4000,
                max_symbols=max_blocks,
            ),
        )
    if mode == "pull":
        from lemoncrow.core.capabilities import licensing as _licensing
        from lemoncrow.pro.capabilities.scoped_context import Subtask

        _licensing.require("scoped_context")
        subtask = Subtask(
            description=task,
            affected_paths=files or [],
            keywords=keywords or [],
            excluded_paths=excluded_paths or [],
            budget_tokens=token_budget or 4000,
        )
        return cast(dict[str, Any], _scoped_context_capability(".").pull(subtask).to_dict())
    if mode != "procedures":
        raise ValueError(f"unknown mode: {mode!r}")
    if errors is None:
        errors = []
    if tools is None:
        tools = []
    if keywords is None:
        keywords = []
    if excluded_paths is None:
        excluded_paths = []
    if files is None:
        files = []
    rt = _runtime()
    led = _get_ledger()
    led.task = task
    if domain:
        led.domain = domain
    _match_mcp_lexical({"task": task})

    led.record_tool_call(
        "get_context",
        {
            "task": task,
            "domain": domain,
            "files": files,
            "keywords": keywords,
            "excluded_paths": excluded_paths,
            "tools": tools,
            "errors": errors,
            "max_blocks": max_blocks,
            "token_budget": token_budget,
            "dedup": dedup,
            "agent_id": agent_id,
            "recall": recall,
        },
    )

    bootstrap = _bootstrap_context_status(_lemoncrow_root())
    # Keep workspace resolution consistent between this MCP adapter and the
    # core runtime path resolver so bootstrap status and injected bootstrap
    # context are derived from the same repository.
    workspace_root = str(_workspace_root().resolve())
    previous_workspace_root = os.environ.get("LEMONCROW_WORKSPACE_ROOT")
    os.environ["LEMONCROW_WORKSPACE_ROOT"] = workspace_root

    # Advance trajectory monitors and obtain FSM-derived retrieval hints.
    _monitor_composite, _fsm_skip_etraces = _advance_monitors(_get_product_session_id(), task, led.task or task)

    try:
        payload = rt.get_context(
            task=task,
            domain=domain,
            files=files,
            tools=tools,
            errors=errors,
            max_blocks=max_blocks,
            token_budget=token_budget,
            dedup=dedup,
            agent_id=agent_id,
            recall=recall,
            monitor_composite=_monitor_composite,
            fsm_skip_etraces=_fsm_skip_etraces,
        )
    finally:
        if previous_workspace_root is None:
            os.environ.pop("LEMONCROW_WORKSPACE_ROOT", None)
        else:
            os.environ["LEMONCROW_WORKSPACE_ROOT"] = previous_workspace_root
    result: dict[str, Any] = payload if isinstance(payload, dict) else {"context": payload}
    if bootstrap["status"] != "warm":
        _spawn_worker_if_idle(_lemoncrow_root())
    result["bootstrap"] = bootstrap

    # Wire PrefixCachePlanner: compute the static/dynamic cache split for this
    # turn. This is cache/token-split diagnostics the model does not act on, so
    # it is gated behind the diagnostics opt-in and never emitted in the default
    # model-facing result. Skipping the whole block also avoids the planner cost
    # on the hot path.
    if _mcp_debug_enabled():
        try:
            from lemoncrow.pro.capabilities.prefix_cache.planner import PrefixCachePlanner
            from lemoncrow.pro.capabilities.prompt_compilation.models import (
                BlockKind,
                PromptBlock,
                Stability,
            )

            context_text = result.get("context", "")
            bootstrap_text = (
                result.get("bootstrap", {}).get("context", "") if isinstance(result.get("bootstrap"), dict) else ""
            )
            _recall_count = len(result.get("recalled_passages", []))

            # Build synthetic PromptBlocks from the assembled context pieces
            blocks: list[PromptBlock] = []
            if context_text:
                blocks.append(
                    PromptBlock(
                        id="context",
                        kind=BlockKind.PLAYBOOK,
                        stability=Stability.BRANCH,
                        content=context_text,
                    )
                )
            if bootstrap_text:
                blocks.append(
                    PromptBlock(
                        id="bootstrap",
                        kind=BlockKind.REPO_SUMMARY,
                        stability=Stability.SESSION,
                        content=bootstrap_text,
                    )
                )
            if task:
                blocks.append(
                    PromptBlock(
                        id="task",
                        kind=BlockKind.USER_TASK,
                        stability=Stability.TURN,
                        content=task,
                    )
                )

            if blocks:
                # Compare with prior hash from last llm_call event in ledger
                prior_hash = ""
                call_events = [e for e in led.events if e.payload.get("kind") == "llm_call"]
                if call_events:
                    prior_hash = call_events[-1].payload.get("stable_prefix_hash", "")

                planner = PrefixCachePlanner()
                plan = planner.plan_with_history(blocks, prior_hash or None)
                result["prefix_plan"] = plan.to_dict()
        except Exception:
            logging.exception("Recovered from broad exception handler")
            # Best-effort: never break tool_context due to prefix planning errors.
            _log.debug("prefix-cache planning failed", exc_info=True)

    return result


def _prefix_cache_diagnostics_from_ledger(led: Any) -> dict[str, Any]:
    """Extract prefix cache metrics from recorded llm_call events in the ledger."""
    call_events = [e for e in led.events if e.payload.get("kind") == "llm_call"]
    if not call_events:
        return {
            "turn_count": 0,
            "cache_hit_ratio": 0.0,
            "cache_read_tokens_saved": 0,
            "avg_prefix_tokens": 0,
            "avg_dynamic_tokens": 0,
            "current_prefix_hash": "",
            "prefix_invalidated_reason": "",
        }

    cache_read_totals = [int(e.payload.get("cache_read_tokens", 0)) for e in call_events]
    modeled_cache_read_totals = [int(e.payload.get("modeled_cache_read_tokens", 0)) for e in call_events]
    input_totals = [int(e.payload.get("input_tokens", 0)) for e in call_events]
    prefix_hashes = [e.payload.get("stable_prefix_hash", "") for e in call_events]

    # A turn is a cache "hit" when cache_read_tokens > 0
    eligible = call_events[1:]
    hits = sum(1 for e in eligible if int(e.payload.get("cache_read_tokens", 0)) > 0)
    hit_ratio = round(hits / len(eligible), 4) if eligible else 0.0
    cache_read_saved = sum(cache_read_totals)
    avg_input = int(sum(input_totals) / len(input_totals)) if input_totals else 0

    last = call_events[-1]
    return {
        "turn_count": len(call_events),
        "cache_hit_ratio": hit_ratio,
        "cache_read_tokens_saved": cache_read_saved,
        "modeled_cache_read_tokens_saved": sum(modeled_cache_read_totals),
        "avg_prefix_tokens": avg_input,
        "avg_dynamic_tokens": 0,
        "current_prefix_hash": prefix_hashes[-1] if prefix_hashes else "",
        "prefix_invalidated_reason": last.payload.get("prefix_invalidated_reason", ""),
    }


@mcp_tool(name="rescue")
def tool_rescue_failure(
    task: str,
    error: str,
    domain: str | None = None,
    files: list[str] | None = None,
    recent_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Suggest a rescue procedure for a repeated failure (call after the same approach fails twice).

    Returns: {cluster_id, domain, rescue_type, procedure: [{step, rationale}], rationale, analysis?}.
    """
    if recent_actions is None:
        recent_actions = []
    if files is None:
        files = []
    rt = _runtime()
    led = _get_ledger()
    _match_mcp_lexical({"task": task, "error": error})
    led.record_tool_call(
        "rescue_failure",
        {
            "task": task,
            "error": error,
            "domain": domain,
            "files": files,
            "recent_actions": recent_actions,
        },
    )

    result = rt.rescue_failure(
        task=task,
        error=error,
        files=files,
        domain=domain,
        recent_actions=recent_actions,
    )
    payload = to_jsonable(result)
    with contextlib.suppress(Exception):
        from lemoncrow.core.service.telemetry import emit_product
        from lemoncrow.core.service.telemetry.schema import hash_identifier

        matched = list(payload.get("matched_blocks", []) or []) if isinstance(payload, dict) else []
        emit_product(
            "rescue_offered",
            cluster_id_hash=hash_identifier(str(matched[0] if matched else "unmatched_rescue")),
            rescue_type="playbook" if matched else "summary",
            session_id=_get_product_session_id(),
        )

    # Failure incident analysis from prior failed traces.
    with contextlib.suppress(Exception):
        analysis = rt.core_runtime.analyze_failure_for_error(
            task=task,
            error=error,
            domain=domain,
            lookback=200,
        )
        payload["analysis"] = analysis
        incident = analysis.get("incident") if isinstance(analysis, dict) else None
        if isinstance(incident, dict):
            root_cause = incident.get("root_cause_hypothesis", "")
            if isinstance(root_cause, str) and root_cause:
                led.record(
                    "note",
                    "failure_analysis",
                    {
                        "root_cause": root_cause,
                        "fingerprint": incident.get("fingerprint"),
                        "count": incident.get("count"),
                    },
                )

    return payload


_MAX_TRACE_FILES = int(os.environ.get("LEMONCROW_TRACE_MAX_FILES", "50"))
_MAX_TRACE_FILE_BYTES = int(os.environ.get("LEMONCROW_TRACE_MAX_FILE_BYTES", str(1024 * 1024)))


@mcp_tool(name="trace")
def tool_record_trace(
    agent: str,
    domain: str,
    task: str,
    status: Literal["success", "failed", "partial"],
    errors_seen: list[str] | None = None,
    diff_summary: str = "",
    output_summary: str = "",
    tools_called: list[Any] | None = None,
    validation_results: list[Any] | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    host: str | None = None,
    trace_confidence: str | None = None,
    capture_sources: list[str] | None = None,
    missing_surfaces: list[str] | None = None,
    event_type: str | None = None,
    event_payload: dict[str, Any] | None = None,
    capture_files: list[str] | None = None,
    learnings: list[Any] | None = None,
) -> dict[str, Any]:
    """Record an observable trace of an agent run (status, diffs, tools, validations, learnings) to the run ledger.

    Call once when a task is done so outcomes and lessons persist for later recall.

    Returns: {trace_id, event_recorded}.
    """
    from lemoncrow.core.foundation.redaction import redact, redact_list

    if tools_called is None:
        tools_called = []
    if validation_results is None:
        validation_results = []
    if errors_seen is None:
        errors_seen = []
    if capture_sources is None:
        capture_sources = []
    if missing_surfaces is None:
        missing_surfaces = []
    if event_payload is None:
        event_payload = {}
    if capture_files is None:
        capture_files = []
    if learnings is None:
        learnings = []
    rt = _runtime()
    led = _get_ledger()
    rtc = _get_realtime_context()

    def _redact_json_strings(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, list):
            return [_redact_json_strings(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _redact_json_strings(item) for key, item in value.items()}
        return value

    def _coerce_validation_passed(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"pass", "passed", "success", "successful", "ok", "true"}:
                return True
            if lowered in {"fail", "failed", "failure", "error", "errored", "false"}:
                return False
        return False

    def _normalize_validation_results(items: list[Any]) -> list[dict[str, Any]]:
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
                        "passed": _coerce_validation_passed(passed),
                        "detail": redact(str(detail)),
                    }
                )
                continue
            text = redact(str(item))
            lowered = text.lower()
            passed = not any(token in lowered for token in ("fail", "error", "not run"))
            normalized.append({"name": text, "passed": passed, "detail": ""})
        return normalized

    def _normalize_tool_calls(items: list[Any]) -> list[dict[str, Any]]:
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
                tool_call: dict[str, Any] = {
                    "name": redact(str(item.get("name") or item.get("tool") or "unknown")),
                    "args_hash": redact(str(item.get("args_hash") or "")),
                    "count": raw_count,
                }
                if "args" in item:
                    tool_call["args"] = _redact_json_strings(item["args"])
                if isinstance(item.get("result_summary"), str):
                    tool_call["result_summary"] = redact(item["result_summary"])
                normalized.append(tool_call)
                continue
            normalized.append({"name": redact(str(item)), "args_hash": "", "count": 1})
        return normalized

    def _normalize_learnings(items: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                text = redact(item.strip())
                if text:
                    normalized.append({"kind": "note", "text": text})
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
            normalized.append(entry)
        return normalized

    def _normalize_trace_confidence(value: Any) -> str | None:
        if value is None:
            return None
        normalized = redact(str(value)).strip().lower()
        if not normalized or normalized in {"none", "null", "unknown"}:
            return None
        if normalized in {"full_live", "mcp_live", "wrapper_live", "imported", "manual"}:
            return normalized
        if normalized in {"high", "medium", "low"}:
            # Legacy callers treated this field like a confidence strength rather
            # than a capture provenance. Preserve the trace conservatively.
            return "manual"
        return "manual"

    def _normalize_workflow_trace_payload(raw_event_type: str, raw_payload: dict[str, Any]) -> dict[str, Any] | None:
        normalized_type = redact(raw_event_type).strip().lower()
        payload = _redact_json_strings(raw_payload)
        if not isinstance(payload, dict):
            return None
        if normalized_type == "workflow_state":
            workflow_step = str(payload.get("workflow_step") or payload.get("current_step") or "").strip()
            session_phase = str(payload.get("session_phase") or "").strip()
            result: dict[str, Any] = {}
            if workflow_step:
                result["workflow_step"] = workflow_step
            if session_phase:
                result["session_phase"] = session_phase
            return result or None
        if normalized_type == "plan_review":
            review_decision = str(payload.get("review_decision") or payload.get("decision") or "").strip()
            plan_id = str(payload.get("plan_id") or "").strip()
            workflow_step = str(payload.get("workflow_step") or "").strip()
            result = {}
            if review_decision:
                result["review_decision"] = review_decision
            if plan_id:
                result["plan_id"] = plan_id
            if workflow_step:
                result["workflow_step"] = workflow_step
            return result or None
        if normalized_type == "task_progress":
            task_id = str(payload.get("task_id") or "").strip()
            workflow_step = str(payload.get("workflow_step") or "").strip()
            result = {}
            if task_id:
                result["task_id"] = task_id
            if workflow_step:
                result["workflow_step"] = workflow_step
            for key in ("completed_tasks", "remaining_tasks"):
                value = payload.get(key)
                if isinstance(value, bool):
                    continue
                try:
                    result[key] = max(0, int(value or 0))
                except (TypeError, ValueError):
                    continue
            return result or None
        return None

    # Derive host label from agent string and environment
    def _derive_host(a: str) -> str:
        al = a.lower()
        if "antigravity" in al or "agy" in al or os.environ.get("ANTIGRAVITY_CLI") or os.environ.get("AGY_CLI"):
            return "antigravity"
        if "cursor" in al or os.environ.get("CURSOR_SESSION_ID") or os.environ.get("CURSOR_TRACE_ID"):
            return "cursor"
        if (
            "hermes" in al
            or os.environ.get("HERMES_HOME")
            or os.environ.get("HERMES_SESSION_ID")
            or os.environ.get("HERMES_CLI")
        ):
            return "hermes"
        if "copilot" in al or os.environ.get("COPILOT_CLI"):
            return "copilot"
        if "codex" in al or os.environ.get("CODEX_CLI"):
            return "codex"
        if (
            "opencode" in al
            or os.environ.get("OPENCODE_CLI")
            or os.environ.get("OPENCODE_SESSION_ID")
            or os.environ.get("LEMONCROW_AGENT", "") == "opencode"
        ):
            return "opencode"
        if "claude" in al or os.environ.get("CLAUDE_CODE"):
            return "claude"

        # Default to the agent name if no known host environment is detected
        return "lemoncrow" if al.startswith("lemoncrow:") else al

    normalized_capture_sources = [redact(str(source)) for source in capture_sources]
    normalized_trace_confidence = _normalize_trace_confidence(trace_confidence)
    normalized_missing_surfaces = redact_list([str(value) for value in missing_surfaces])
    if normalized_trace_confidence == "full_live" and not any(
        source in {"hooks", "live_hooks", "plugin_hooks"} for source in normalized_capture_sources
    ):
        normalized_trace_confidence = "mcp_live"
        if "hooks" not in normalized_missing_surfaces:
            normalized_missing_surfaces.append("hooks")

    payload: dict[str, Any] = {
        "agent": agent,
        "domain": domain,
        "task": redact(task),
        "status": status,
        "errors_seen": redact_list([str(v) for v in errors_seen]),
        "diff_summary": redact(diff_summary),
        "output_summary": redact(output_summary),
        "session_id": session_id or run_id or led.session_id,
        "host": redact(host) if host else _derive_host(agent),
        "trace_confidence": normalized_trace_confidence,
        "capture_sources": normalized_capture_sources,
        "missing_surfaces": normalized_missing_surfaces,
    }
    payload["tools_called"] = _normalize_tool_calls(tools_called)
    payload["validation_results"] = _normalize_validation_results(validation_results)
    payload["learnings"] = _normalize_learnings(learnings)

    raw_artifacts: list[str] = []
    pending_artifacts: list[tuple[RawArtifact, str]] = []
    if capture_files:
        source_session_id = (
            _get_product_session_id()
            or os.environ.get("CODEX_SESSION_ID")
            or os.environ.get("OPENCODE_SESSION_ID")
            or "unknown"
        )
        for fpath in capture_files[:_MAX_TRACE_FILES]:
            try:
                p = Path(fpath)
                if not p.is_file():
                    continue
                if p.stat().st_size > _MAX_TRACE_FILE_BYTES:
                    logging.info(
                        "trace capture: skipping %s (%d bytes > cap %d; raise LEMONCROW_TRACE_MAX_FILE_BYTES)",
                        fpath,
                        p.stat().st_size,
                        _MAX_TRACE_FILE_BYTES,
                    )
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                # We redact secrets from files before capture for safety
                redacted_content = redact(content)
                digest = sha256(redacted_content.encode("utf-8", errors="replace")).hexdigest()

                # Use a stable but unique ID for the file artifact
                artifact_id = f"file-{sha256(fpath.encode()).hexdigest()[:12]}-{digest[:12]}"

                artifact = RawArtifact(
                    id=artifact_id,
                    source="mcp",
                    source_session_id=source_session_id,
                    kind="source.code",
                    relative_path=f"{artifact_id}.txt",
                    content_path=f"raw/mcp/{source_session_id}/{artifact_id}.txt",
                    sha256_original=sha256(content.encode()).hexdigest(),
                    sha256_redacted=digest,
                    byte_count_original=len(content.encode("utf-8")),
                    byte_count_redacted=len(redacted_content.encode("utf-8")),
                    redacted=True,
                    source_path=str(p.absolute()),
                    source_file_mtime=datetime.fromtimestamp(p.stat().st_mtime, tz=UTC),
                )
                pending_artifacts.append((artifact, redacted_content))
                raw_artifacts.append(artifact_id)
            except Exception as e:
                logging.exception("Recovered from broad exception handler")
                logger.warning("Failed to capture context file %s: %s", fpath, e)

    if raw_artifacts:
        payload["raw_artifact_ids"] = raw_artifacts

    if "id" not in payload:
        payload["id"] = Trace.make_id(task, agent)

    # Validate BEFORE any store/ledger write so a malformed payload returns a
    # structured error instead of leaving committed artifacts / a workflow event
    # behind with no persisted trace (partial-write inconsistency).
    trace = Trace.model_validate(payload)

    for _artifact, _redacted_content in pending_artifacts:
        rt.store.history.record_raw_artifact(_artifact, _redacted_content)

    if event_type:
        normalized_event_payload = _normalize_workflow_trace_payload(event_type, event_payload)
        if normalized_event_payload is not None:
            led.record_workflow_event(event_type, normalized_event_payload)
        else:
            led.record("note", f"event:{redact(event_type)}", _redact_json_strings(event_payload))

    rt.store.history.record_trace(trace)
    from lemoncrow.pro.capabilities.lesson_promotion import ingest_failed_trace

    ingest_failed_trace(rt.store, trace)

    # Write learnings to archival memory (not Playbooks - those are curated).
    # Each learning is a short sentence the agent synthesises; stored deduped so
    # repeated identical insights across sessions don't accumulate noise.
    if trace.learnings:
        mem = _memory_store()
        for learning in trace.learnings:
            text = redact(learning.text.strip())
            if not text:
                continue
            dedup_hash = sha256(f"{agent}:{text}".encode()).hexdigest()[:32]
            passage = ArchivalPassage(
                agent_id=agent,
                text=text,
                source="trace",
                source_ref=trace.id,
                tags=["learning", domain, learning.kind],
                dedup_hash=dedup_hash,
            )
            with contextlib.suppress(Exception):
                mem.insert_passage(passage)

    led.close(status=status)
    led.persist()

    rtc.persist()

    # Emit to Langfuse if configured (fail-open)
    from lemoncrow.gateway.integrations.langfuse import emit_trace as _lf_emit

    _lf_emit(payload)

    # Kick off a background consolidation tick so knowledge blocks are extracted
    # from this trace without waiting for the daemon's next cycle. Throttled
    # (>=30s) so a burst of trace calls can't spawn a thread storm.
    _spawn_worker_if_idle(_lemoncrow_root())

    # Stable compact receipt.
    return {
        "trace_id": trace.id,
        "event_recorded": bool(event_type),
    }


@mcp_tool(name="verify")
def tool_run_rubric_gate(rubric_id: str, checks: dict[str, Any]) -> Any:
    """Evaluate agent results against a domain rubric. Returns pass|warn|fail with per-check detail."""
    rt = _runtime()
    led = _get_ledger()
    led.record_tool_call("run_rubric_gate", {"rubric_id": rubric_id, "checks": checks})

    rubric = rt.store.knowledge.get_rubric(rubric_id)
    if rubric is None:
        raise ValueError(f"rubric not found: {rubric_id}")

    if rubric_id not in led.active_rubrics:
        led.active_rubrics.append(rubric_id)

    result = run_rubric(rubric, checks)
    led.record("rubric_run", f"Rubric {rubric_id} status: {result.status}", to_jsonable(result))
    return to_jsonable(result)


def _compress_context(session_id: str | None = None) -> Any:
    """Compress the current ledger state into a compact prompt block for context continuation.

    Call when context is heavy; the block preserves decisions and state while dropping stale history.

    Returns: {prompt_block, tokens_before, tokens_after_estimate, tokens_freed, cost_saved_usd}.
    """
    from lemoncrow.pro.runtime.context_compressor import ContextCompressor

    led = _get_ledger()
    if session_id:
        led.session_id = session_id
    state = ContextCompressor().compress(led, preserve_last_n_turns=10, workspace_root=_workspace_root())
    compaction_savings = _session_compaction_savings_payload(
        led,
        state,
        tokens_before=int(led.token_count or 0),
        trigger="compact_session",
        reason="session compaction executed",
    )
    if int(compaction_savings["tokens_saved"]) > 0:
        _append_live_savings_event(compaction_savings)

    with contextlib.suppress(Exception):
        from lemoncrow.pro.runtime import outcome_capture

        outcome_capture.schedule_compact(
            session_id=led.session_id,
            trigger="compact_session",
            tokens_before=int(compaction_savings["tokens_before"]),
            tokens_after=int(compaction_savings["tokens_after_estimate"]),
            must_keep_keywords=list(led.active_playbooks),
            errors_before=len(led.errors_seen) + len(led.repeated_failures),
            writer=_make_outcome_writer(led),
        )

    return {
        "prompt_block": state.to_prompt_block(),
        "tokens_before": int(compaction_savings["tokens_before"]),
        "tokens_after_estimate": int(compaction_savings["tokens_after_estimate"]),
        "tokens_freed": int(compaction_savings["tokens_freed"]),
        "cost_saved_usd": float(compaction_savings["cost_saved_usd"]),
    }


def _memory_upsert_block(
    agent_id: str,
    label: str,
    value: str,
    limit_chars: int = 8000,
    description: str = "",
    read_only: bool = False,
    pinned: bool = False,
    metadata: dict[str, Any] | None = None,
    expected_version: int | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Create or update an editable memory block."""
    from lemoncrow.infra.embeddings.factory import make_embedder

    clean_value = _redact_memory_input(value, "value")
    clean_description = _redact_memory_input(description, "description")
    store = _memory_store()
    existing = store.get_block(agent_id, label)
    version = expected_version if expected_version is not None else (existing.version if existing else 1)
    seed = existing or MemoryBlock(agent_id=agent_id, label=label, value=clean_value)
    block = MemoryBlock(
        id=seed.id,
        agent_id=agent_id,
        label=label,
        value=clean_value,
        limit_chars=limit_chars,
        description=clean_description,
        read_only=read_only,
        metadata=metadata or {},
        pinned=pinned,
        version=version,
        current_history_id=existing.current_history_id if existing else None,
        created_at=seed.created_at,
    )
    from lemoncrow.pro.capabilities.memory_arbitration import arbitrate

    decision = arbitrate(block, store, make_embedder())
    target = None
    if decision.target_block_id:
        for item in store.list_blocks(agent_id, include_tombstoned=True, limit=500):
            if item.id == decision.target_block_id:
                target = item
                break

    if decision.op == "NOOP" and target is not None:
        stored = target
    elif decision.op == "UPDATE" and target is not None:
        stored = store.upsert_block(
            target.model_copy(update={"value": decision.merged_value or clean_value}),
            actor=actor or f"agent:{agent_id}",
            reason=decision.reason,
        )
    elif decision.op == "DELETE" and target is not None:
        store.tombstone_block(target.id, deprecated_by_block_id=block.id, reason=decision.reason)
        stored = store.upsert_block(block, actor=actor or f"agent:{agent_id}", reason=decision.reason)
    else:
        stored = store.upsert_block(block, actor=actor or f"agent:{agent_id}")
    return {
        "id": stored.id,
        "version": stored.version,
        "arbitration": {"op": decision.op, "reason": decision.reason},
    }


def _memory_get_block(agent_id: str | None, label: str) -> dict[str, Any] | None:
    """Retrieve a MemoryBlock by label."""
    block = _memory_store().get_block(agent_id, label)
    return block.model_dump(mode="json") if block is not None else None


def _memory_archive(
    agent_id: str | None,
    text: str,
    source: str,
    source_ref: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Archive long-term memory text for later recall."""
    passage = _archival_recall().archive(
        agent_id=agent_id,
        text=text,
        source=source,  # type: ignore[arg-type]
        source_ref=source_ref,
        tags=tags or [],
    )
    return {"id": passage.id, "dedup_hit": passage.dedup_hit}


def _memory_recall(
    agent_id: str | None,
    query: str,
    top_k: int = 5,
    tags: list[str] | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Recall relevant archival memory passages.

    The recall surface spans two stores: durable memory (``memory.db`` via
    ``MemoryService``) and past-session transcripts (``recall.db``, populated by
    the SessionStart indexer in ``session_recall``). They are separate SQLite
    files with independently scaled scores, so memory passages are listed first,
    then past-session hits are appended.
    """
    result = (
        _memory_service()
        .recall(
            agent_id=agent_id,
            query=query,
            top_k=top_k,
            tags=tags or None,
            since=since,
        )
        .model_dump(mode="json")
    )
    mem_passages = result.get("passages")
    if not isinstance(mem_passages, list):
        mem_passages = []
    session = _session_recall_passages(query, top_k)
    if session:
        # Honor the top_k contract (the field promises "max results") while keeping
        # past-session recall visible: reserve up to a third of the budget for
        # session hits and fill the rest from durable memory.
        reserve = min(len(session), max(1, top_k // 3))
        passages = mem_passages[: max(0, top_k - reserve)] + session[:reserve]
    else:
        passages = mem_passages[:top_k]
    result["passages"] = passages
    if not passages:
        # Helpful state hint instead of a bare empty result, so the model knows
        # memory is working and how to seed it.
        result["hint"] = (
            "No matching memories yet — memory accrues as you work. Store durable facts with "
            "memory(op=store_fact); past-session recall improves as sessions are indexed."
        )
    return result


def _session_recall_passages(query: str, top_k: int) -> list[dict[str, Any]]:
    """Past-session recall hits from ``recall.db``, shaped like
    ``MemoryRecallPassage`` so ``memory(op=recall)`` renders them next to
    ``memory.db`` passages. Best-effort: ``session_recall.recall`` returns ``[]``
    when the index is missing or the embedder is unavailable."""
    from lemoncrow.core.capabilities import session_recall

    return [
        {
            "id": str(hit.get("session") or ""),
            "text": str(hit.get("text") or ""),
            "source_ref": str(hit.get("session") or ""),
            "tags": list(hit.get("tags") or []),
        }
        for hit in session_recall.recall(_lemoncrow_root(), query, top_k=top_k)
    ]


def _memory_service() -> MemoryService:
    from lemoncrow.infra.embeddings.factory import make_embedder
    from lemoncrow.pro.capabilities.memory import MemoryService

    return MemoryService(store=_memory_store(), embedder=make_embedder(), redactor=redact)


def _memory_store_fact(
    *,
    agent_id: str | None,
    subject: str,
    fact: str,
    citations: str,
    reason: str,
    scope: str,
) -> dict[str, Any]:
    """Store a durable fact with Copilot-memory-like fields in LemonCrow memory."""
    return (
        _memory_service()
        .store_fact(
            agent_id=agent_id,
            subject=_redact_memory_input(subject, "subject"),
            fact=_redact_memory_input(fact, "fact"),
            citations=_redact_memory_input(citations, "citations"),
            reason=_redact_memory_input(reason, "reason"),
            scope=scope,
        )
        .model_dump(mode="json")
    )


def _memory_vote_fact(
    *,
    agent_id: str | None,
    fact: str,
    direction: str,
    reason: str,
    scope: str | None,
) -> dict[str, Any]:
    """Vote on an existing stored fact by exact fact text."""
    return (
        _memory_service()
        .vote_fact(
            agent_id=agent_id,
            fact=_redact_memory_input(fact, "fact"),
            direction=direction,
            reason=_redact_memory_input(reason, "reason"),
            scope=scope,
        )
        .model_dump(mode="json")
    )


@mcp_tool(
    name="memory",
    description=("Memory op-dispatch for fact storage/voting and recall."),
    param_aliases={"agent_id": "agent", "top_k": "k"},
)
def tool_memory(
    op: Annotated[
        Literal[
            "recall",
            "recall_symbol",
            "store_fact",
            "vote_fact",
        ],
        Field(
            description=(
                "recall/recall_symbol need query; "
                "store_fact needs subject+fact+citations+reason+scope; "
                "vote_fact needs fact+direction+reason."
            )
        ),
    ],
    agent: Annotated[
        str | None,
        Field(description="Memory namespace; defaults to shared."),
    ] = None,
    query: Annotated[str | None, Field(description="Search query used by recall.")] = None,
    k: Annotated[int, Field(description="Max results to return for recall.")] = 5,
    subject: Annotated[
        str | None,
        Field(description="Fact subject for store_fact (for example: testing, workflow preference)."),
    ] = None,
    fact: Annotated[
        str | None,
        Field(description="Exact fact text for store_fact and vote_fact."),
    ] = None,
    citations: Annotated[
        str | None,
        Field(description="Source citations for store_fact."),
    ] = None,
    reason: Annotated[
        str | None,
        Field(description="Detailed rationale for store_fact and vote_fact."),
    ] = None,
    scope: Annotated[
        Literal["repository", "user"] | None,
        Field(description="Fact scope for store_fact/vote_fact."),
    ] = None,
    direction: Annotated[
        Literal["upvote", "downvote"] | None,
        Field(description="Vote direction for vote_fact."),
    ] = None,
) -> dict[str, Any] | None:
    """Memory op-dispatch: recall, recall_symbol, store_fact, or vote_fact."""

    def require(name: str, current: str | None) -> str:
        if not current:
            raise ValueError(f"{name} is required for memory op={op}")
        return current

    if op == "recall":
        return _memory_recall(
            agent_id=agent,
            query=require("query", query),
            top_k=k,
        )
    if op == "store_fact":
        return _memory_store_fact(
            agent_id=agent,
            subject=require("subject", subject),
            fact=require("fact", fact),
            citations=citations or "",
            reason=reason or "",
            scope=require("scope", scope),
        )
    if op == "vote_fact":
        return _memory_vote_fact(
            agent_id=agent,
            fact=require("fact", fact),
            direction=require("direction", direction),
            reason=require("reason", reason),
            scope=scope,
        )
    if op == "recall_symbol":
        return cast(
            dict[str, Any],
            _symbol_recall().recall_symbol(
                query=require("query", query),
                agent_id=agent,
                top_k=k,
            ),
        )
    raise ValueError(f"unsupported memory op: {op}")


def _render_read_md(result: dict[str, Any]) -> str | None:
    mode = str(result.get("mode") or "")
    projection = result.get("projection")
    notice = ""
    if isinstance(projection, dict):
        raw_notice = str(projection.get("notice") or "").strip()
        if raw_notice:
            notice = raw_notice
    if mode == "directory":
        entries = result.get("entries")
        if isinstance(entries, list):
            return "\n".join(entries)
        return None
    if mode == "summary":
        summary = str(result.get("summary") or "").strip()
        if not summary:
            return None
        return f"{notice}\n\n{summary}" if notice else summary
    if mode in {"range", "full"}:
        content = str(result.get("content") or "")
        if not content:
            return None
        return f"{notice}\n\n{content}" if notice else content
    if mode == "outline":
        path = str(result.get("path") or "?")
        language = str(result.get("language") or "")
        outline = result.get("outline")
        if isinstance(outline, dict):
            rendered = _render_read_outline_md(path, outline, language)
            return f"{notice}\n\n{rendered}" if notice else rendered
        return None
    return None


def _render_symbol_read_md(sym: dict[str, Any]) -> str | None:
    sym_path = str(sym.get("path") or "")
    sym_kind = str(sym.get("kind") or "symbol")
    start = int(sym.get("line") or 0)
    end = int(sym.get("end_line") or start)
    if not sym_path or not start:
        return None
    try:
        src_lines = Path(sym_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        body = "\n".join(src_lines[start - 1 : end])
    except OSError:
        return None
    range_tag = f"L{start}-L{end}" if end > start else f"L{start}"
    return f"### {sym_path}:{range_tag} ({sym_kind})\n{body}"


# Max lines an outline projection renders inline; the tail collapses to a
# "+K more" line (the full structure is one :Lx-Ly or :full read away). A
# 5k-line file's outline can otherwise run hundreds of resident lines.
_READ_OUTLINE_MAX_LINES = 80


def _render_read_outline_md(path: str, outline: dict[str, Any], language: str) -> str:
    # Treesitter/generic: has pre-formatted `text` field
    text = str(outline.get("text") or "").strip()
    if text:
        t_lines = text.splitlines()
        if len(t_lines) > _READ_OUTLINE_MAX_LINES:
            dropped = len(t_lines) - _READ_OUTLINE_MAX_LINES
            t_lines = [
                *t_lines[:_READ_OUTLINE_MAX_LINES],
                f"... +{dropped} more outline lines (use :Lx-Ly or :full)",
            ]
        return "\n".join(t_lines)
    # AST outline: has `symbols`, `imports`, `hint` fields
    lines: list[str] = []
    imports_list = outline.get("imports")
    if isinstance(imports_list, list) and imports_list:
        # Collapse to distinct top-level roots instead of one line per import.
        # On import-heavy files the full list is dozens of lines the model
        # rarely needs; the root set conveys the dependency surface far cheaper.
        roots = sorted({str(imp).split()[0].split(".")[0] for imp in imports_list if str(imp).strip()})
        lines.append(f"imports ({len(imports_list)}): {', '.join(roots)}")
    symbols_list = outline.get("symbols")
    if isinstance(symbols_list, list) and symbols_list:
        lines.append("symbols:")
        for sym in symbols_list[:_READ_OUTLINE_MAX_LINES]:
            if not isinstance(sym, dict):
                continue
            name = str(sym.get("name") or "?")
            kind = str(sym.get("kind") or "?")
            start = int(sym.get("start_line") or 0)
            end = int(sym.get("end_line") or 0)
            loc = f"{start}-{end}" if end > start else str(start)
            # `[function]` is the majority kind -- tagging only the exceptions
            # saves ~2 tokens per line on function-heavy outlines.
            tag = "" if kind == "function" else f" [{kind}]"
            lines.append(f"- {loc}: {name}{tag}")
        if len(symbols_list) > _READ_OUTLINE_MAX_LINES:
            lines.append(f"... +{len(symbols_list) - _READ_OUTLINE_MAX_LINES} more symbols (use :Lx-Ly or :full)")
    return "\n".join(lines) if lines else "(no outline)"


def _render_grep_md(result: dict[str, Any]) -> str | None:
    mode = str(result.get("mode") or result.get("output_mode") or "")
    if mode == "ranked_file_map":
        matches = result.get("matches")
        if not isinstance(matches, list) or not matches:
            return "no matches"
        lines: list[str] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            file_path = str(match.get("file") or "?")
            lines.append(file_path)
            ranges = match.get("ranges")
            if isinstance(ranges, list):
                for r in ranges:
                    lines.append(f"- lines {r}")
        return "\n".join(lines) if lines else "no matches"
    # Non-ranked modes: content is pre-formatted text blocks
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return None


_SECTION_GUTTER_RE = re.compile(r"^(\d+)\t")

# Within a contiguous run, re-anchor the gutter every K lines. K bounds the
# offset arithmetic a model must do to name any line (≤4 lines from the
# nearest anchor above) — small enough that :Lx-Ly edit targets stay read-off
# rather than computed, so the range-edit path doesn't lose to old/new on
# friction. K=5 keeps ~80% of the full-gutter savings.
_GUTTER_ANCHOR_EVERY = 5


def _sparse_gutter(content: str) -> str:
    """Drop line-number prefixes that are derivable from a nearby anchor.

    Search sections are read-oriented: within a contiguous run the first line
    and every ``_GUTTER_ANCHOR_EVERY``-th line keep their number; the lines
    between (numbered exactly previous+1) carry no information and drop their
    prefix. Runs re-anchor after skeleton elisions and truncation markers (any
    non-gutter line resets the counter). Measured on real payloads the full
    gutter is ~10% of the rendered code_search text. The `read` tool keeps its
    full gutter — that surface is edit-oriented.
    """
    out: list[str] = []
    previous: int | None = None
    run_pos = 0
    for line in content.split("\n"):
        match = _SECTION_GUTTER_RE.match(line)
        if match is None:
            out.append(line)
            previous = None
            continue
        number = int(match.group(1))
        consecutive = previous is not None and number == previous + 1
        run_pos = run_pos + 1 if consecutive else 0
        keep = not consecutive or run_pos % _GUTTER_ANCHOR_EVERY == 0
        out.append(line if keep else line[match.end() :])
        previous = number
    return "\n".join(out)


def _compress_candidate_files(paths: list[str]) -> str:
    """Comma-joined candidate_files line, grouping consecutive same-directory
    entries into ``dir/{a,b,c}`` so a shared prefix isn't repeated per file.

    Only ADJACENT same-directory runs group (never reordered into a group):
    candidate_files is the ranked recall surface (MRR-scored by position), so
    grouping non-adjacent entries would shift ranks when a consumer re-splits
    and expands. A caller that splits on top-level commas (treating a
    ``{...}`` span as opaque) and expands each ``dir/{a,b,c}`` segment
    recovers the identical ordered list -- see
    ``benchmarks/codebench/eval_external_provider_mrr.py``'s
    ``_split_candidate_files_line``/``_expand_candidate_segment``.
    """
    segments: list[str] = []
    i, n = 0, len(paths)
    while i < n:
        path = paths[i]
        slash = path.rfind("/")
        if slash == -1:
            segments.append(path)
            i += 1
            continue
        directory = path[: slash + 1]
        run = [path[slash + 1 :]]
        j = i + 1
        while j < n:
            next_path = paths[j]
            next_slash = next_path.rfind("/")
            if next_slash == -1 or next_path[: next_slash + 1] != directory:
                break
            run.append(next_path[next_slash + 1 :])
            j += 1
        segments.append(directory + "{" + ",".join(run) + "}" if len(run) >= 2 else path)
        i = j
    return ", ".join(segments)


def _render_code_search_md(payload: dict[str, Any]) -> str | None:
    """Compact text view of a lean code_search payload.

    Drops the JSON key-noise: sections render as ``## path`` / ``### sym Lx-Ly``
    over sparsely line-numbered source (first line of each contiguous run keeps
    its number; see ``_sparse_gutter``) or their outline pointer, related_symbols
    as one ``path:Lx-Ly kind name`` line each (same-symbol repeats merge their
    ranges), candidate_files as a single comma-joined line. Internal bookkeeping
    (tokens_saved / calls_saved) never reaches the model.
    """
    files = payload.get("files")
    if not isinstance(files, list):
        return None
    parts: list[str] = []
    if not payload.get("exact_match"):
        parts.append("no exact match -- ranked candidates")
    for entry in files:
        if not isinstance(entry, dict):
            continue
        lines = [f"## {entry.get('path') or '?'}"]
        for sec in entry.get("sections") or []:
            if not isinstance(sec, dict):
                continue
            outline = sec.get("outline")
            if outline and "content" not in sec:
                lines.append(str(outline))
                continue
            sym = str(sec.get("qualified_name") or "")
            start, end = sec.get("line"), sec.get("end_line")
            span = f"L{start}-L{end}" if start is not None and end is not None else ""
            header = " ".join(p for p in ("###", sym, span) if p)
            lines.append(header)
            content = str(sec.get("content") or "").rstrip("\n")
            if content:
                lines.append(_sparse_gutter(content))
        parts.append("\n".join(lines))
    related = payload.get("related_symbols")
    if isinstance(related, list) and related:
        merged: dict[tuple[str, str, str], list[str]] = {}
        for sym_entry in related:
            if not isinstance(sym_entry, dict):
                continue
            path = str(sym_entry.get("path") or "?")
            kind = str(sym_entry.get("kind") or "")
            name = str(sym_entry.get("qualified_name") or "?")
            start, end = sym_entry.get("line"), sym_entry.get("end_line")
            if start is None:
                span = ""
            elif end is None or end == start:
                span = f"L{start}"
            else:
                span = f"L{start}-L{end}"
            merged.setdefault((path, kind, name), []).append(span)
        rel_lines = ["related_symbols:"]
        for (path, kind, name), spans in merged.items():
            span_txt = ",".join(s for s in spans if s)
            loc = f"{path}:{span_txt}" if span_txt else path
            # "function" is the majority kind -- tag only the exceptions
            # (same rule as read outlines).
            if kind == "function":
                kind = ""
            rel_lines.append(" ".join(p for p in (loc, kind, name) if p))
        parts.append("\n".join(rel_lines))
    cands = payload.get("candidate_files")
    if isinstance(cands, list) and cands:
        line = "candidate_files: " + _compress_candidate_files([str(c) for c in cands])
        _more = payload.get("candidate_files_more")
        if isinstance(_more, int) and _more > 0:
            line += f" (+{_more} more; pass limit={len(cands) + _more} to see them)"
        parts.append(line)
    if payload.get("truncated"):
        parts.append("(truncated; narrow with paths=)")
    return "\n\n".join(parts) if parts else None


def _render_search_md(result: dict[str, Any]) -> str | None:
    mode = str(result.get("mode") or "chunks")
    if mode == "map":
        outline = str(result.get("outline") or "").strip()
        ranked_raw = result.get("ranked_files")
        ranked = ranked_raw if isinstance(ranked_raw, list) else []
        if not outline and not ranked:
            return None
        # The repo-map `outline` is already plain text; emitting it directly
        # (instead of json.dumps of the whole payload) drops the JSON wrapper and
        # the \n-escaping of every outline line.
        map_lines = ["### repo_map"]
        if outline:
            map_lines.append(outline)
        if ranked:
            map_lines.append("files:")
            for entry in ranked:
                if isinstance(entry, dict):
                    map_lines.append(f"- {entry.get('path') or entry.get('file') or '?'}")
                else:
                    map_lines.append(f"- {entry}")
        return "\n".join(map_lines)
    matches = result.get("matches")
    if not isinstance(matches, list) or not matches:
        return "### search\n- no matches"
    lines: list[str] = ["### search"]
    for match in matches:
        if not isinstance(match, dict):
            continue
        path = str(match.get("path") or "?")
        lines.append(path)
        content = str(match.get("content") or "").strip()
        if content:
            lines.append(content)
        else:
            snippets = match.get("snippets")
            if isinstance(snippets, list):
                for snip in snippets[:3]:
                    if isinstance(snip, dict):
                        snip_content = str(snip.get("content") or "").strip()
                        if snip_content:
                            lines.append(snip_content)
    return "\n".join(lines)


def _render_memory_md(result: dict[str, Any]) -> str | None:
    """Compact recall rendering: one header line per passage (source/tags) plus
    its text body, instead of a JSON list that repeats the field keys on every
    entry and escapes every newline in the passage text. Only recall (which has
    a ``passages`` list) is rendered; store_fact/vote_fact fall back to JSON.
    """
    passages = result.get("passages")
    if not isinstance(passages, list):
        return None
    if not passages:
        return "### memory\n- no passages"
    lines = ["### memory"]
    for passage in passages:
        if not isinstance(passage, dict):
            continue
        ref = (str(passage.get("source_ref") or passage.get("id") or "?").strip()) or "?"
        tags = passage.get("tags")
        tag_str = f" [{', '.join(str(tag) for tag in tags)}]" if isinstance(tags, list) and tags else ""
        lines.append(f"- {ref}{tag_str}")
        text = str(passage.get("text") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _render_verify_md(result: dict[str, Any]) -> str | None:
    """Compact rubric-gate rendering: one line per check instead of a list of dicts
    that repeats name/status/detail keys (detail is empty on every passing check).
    """
    outcomes = result.get("outcomes")
    if not isinstance(outcomes, list):
        return None
    lines = [f"### verify rubric={result.get('rubric_id') or '?'} status={result.get('status') or '?'}"]
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        status = str(outcome.get("status") or "?")
        check_name = str(outcome.get("name") or "?")
        detail = str(outcome.get("detail") or "").strip()
        lines.append(f"- {status} {check_name}" + (f": {detail}" if detail else ""))
    escalations = result.get("escalations")
    if isinstance(escalations, list):
        for escalation in escalations:
            lines.append(f"- escalation: {escalation}")
    return "\n".join(lines)


def _render_sql_md(result: dict[str, Any]) -> str | None:
    """Compact rendering for sql introspection (schema/table/search/relationships/lint).

    Collapses per-column dicts ({cid,name,type,notnull,pk}) -- which repeat their keys
    on every column -- into one line per column. The `query` action is already columnar
    (positional rows + a single header), so it is left as JSON (returns None).
    """

    def _cols(columns: Any) -> list[str]:
        out: list[str] = []
        if isinstance(columns, list):
            for col in columns:
                if not isinstance(col, dict):
                    continue
                parts = [str(col.get("name") or "?"), str(col.get("type") or "")]
                if col.get("pk"):
                    parts.append("pk")
                if col.get("notnull"):
                    parts.append("notnull")
                out.append("  - " + " ".join(p for p in parts if p))
        return out

    def _fks(fks: Any) -> list[str]:
        out: list[str] = []
        if isinstance(fks, list):
            for fk in fks:
                if isinstance(fk, dict):
                    out.append(
                        f"  fk: {fk.get('from_column', '?')} -> {fk.get('table', '?')}.{fk.get('to_column', '?')}"
                    )
        return out

    if "results" in result:  # query: already columnar, leave as JSON
        return None
    if isinstance(result.get("schema"), dict):
        schema = result["schema"]
        lines = [f"### sql schema ({result.get('table_count', len(schema))} tables)"]
        for table, info in schema.items():
            lines.append(f"- {table}")
            if isinstance(info, dict):
                lines.extend(_cols(info.get("columns")))
                lines.extend(_fks(info.get("foreign_keys")))
        return "\n".join(lines)
    if isinstance(result.get("matches"), list):
        lines = ["### sql search"]
        for match in result["matches"]:
            if not isinstance(match, dict):
                continue
            lines.append(f"- {match.get('table', '?')}")
            lines.extend(_cols(match.get("columns")))
            lines.extend(_fks(match.get("foreign_keys")))
        return "\n".join(lines)
    if isinstance(result.get("relationships"), list):
        lines = ["### sql relationships"]
        for rel in result["relationships"]:
            if isinstance(rel, dict):
                lines.append(f"- {rel.get('from', '?')} -> {rel.get('to', '?')}")
        return "\n".join(lines)
    if isinstance(result.get("columns"), list) and "table" in result:
        lines = [f"### sql table {result.get('table', '?')}"]
        lines.extend(_cols(result.get("columns")))
        lines.extend(_fks(result.get("foreign_keys")))
        return "\n".join(lines)
    if "ok" in result:  # lint
        return f"### sql lint: {'ok' if result.get('ok') else (result.get('message') or 'invalid')}"
    if isinstance(result.get("tables"), list):
        tables = result["tables"]
        return "\n".join([f"### sql tables ({result.get('table_count', len(tables))})", *(f"- {t}" for t in tables)])
    return None


def render_tool_result_text(name: str, result: Any) -> str | None:
    """Best-effort compact text rendering of a tool result for model context.

    Shared by the MCP dispatch path and the in-process CLI runtime so both
    hosts send the model identical, minimal text instead of raw dict dumps.
    Returns ``None`` when no renderer applies or it produced nothing — callers
    fall back to the raw string / compact JSON form.
    """
    # "symbols" is a render-name (not a tool): the `symbols` tool was removed,
    # but direct `_op_search` callers (tests, power use) still pass it to fetch
    # the engine's thread-local rendered text. It can't be "search" -- that
    # would flip the live `search` tool from raw JSON to markdown output.
    if name in {"symbols"} | _CODE_INTEL_TOOLS:
        return getattr(_tool_call_rendered_text, "value", None) or None
    if not isinstance(result, dict):
        return None
    payload = result
    text: str | None = None
    if name == "read":
        with contextlib.suppress(Exception):
            files = payload.get("files")
            if isinstance(files, list):
                parts: list[str] = []
                cwd = str(Path.cwd())
                dict_entries = [entry for entry in files if isinstance(entry, dict)]
                # A single-entry read needs no `## path` header -- the caller
                # just named the file (and range); headers only earn their
                # tokens when several entries must be told apart.
                lone_entry = len(dict_entries) == 1
                for entry in dict_entries:
                    entry_path = str(entry.get("path") or "?")
                    if entry_path.startswith(cwd + os.sep):
                        entry_path = entry_path[len(cwd) + 1 :]
                    entry_text = _render_read_md(entry)
                    if entry_text is None:
                        entry_text = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                    if lone_entry:
                        parts.append(entry_text)
                    elif entry.get("mode") == "range":
                        raw_range = str(entry.get("range") or "")
                        range_tag = ":L" + raw_range.replace("-", "-L") if raw_range else ""
                        parts.append(f"## {entry_path}{range_tag}\n{entry_text}")
                    else:
                        parts.append(f"## {entry_path}\n{entry_text}")
                text = "\n\n".join(parts) if parts else None
            else:
                text = _render_read_md(payload)
                if text and payload.get("mode") == "range":
                    raw_range = str(payload.get("range") or "")
                    if raw_range:
                        text = "## L" + raw_range.replace("-", "-L") + "\n" + text
                if text is None:
                    symbols_list = payload.get("symbols")
                    if isinstance(symbols_list, list):
                        sym_parts = [_render_symbol_read_md(s) for s in symbols_list if isinstance(s, dict)]
                        text = "\n\n".join(p for p in sym_parts if p) or None
                    elif "kind" in payload and "line" in payload and "path" in payload:
                        text = _render_symbol_read_md(payload)
    elif name == "grep":
        with contextlib.suppress(Exception):
            text = _render_grep_md(payload)
    elif name == "code_search":
        with contextlib.suppress(Exception):
            text = _render_code_search_md(payload)
    elif name == "search":
        with contextlib.suppress(Exception):
            # mode="symbol" routes through _op_search, which stashes the compact
            # code-intel locator on the thread-local; prefer it when present.
            rendered = getattr(_tool_call_rendered_text, "value", None)
            text = rendered if isinstance(rendered, str) and rendered.strip() else _render_search_md(payload)
    elif name == "bash":
        with contextlib.suppress(Exception):
            text = _render_bash_text(payload)
    elif name == "web_fetch":
        with contextlib.suppress(Exception):
            text = str(payload.get("content") or "")
    elif name == "mcp":
        with contextlib.suppress(Exception):
            text = str(payload.get("content") or "") or None
    elif name == "verify":
        with contextlib.suppress(Exception):
            text = _render_verify_md(payload)
    elif name == "sql":
        with contextlib.suppress(Exception):
            text = _render_sql_md(payload)
    elif name == "explore":
        with contextlib.suppress(Exception):
            from lemoncrow.pro.capabilities.code_context.renderer import _render_explore

            text = _render_explore(payload) or None
    elif name == "memory":
        with contextlib.suppress(Exception):
            text = _render_memory_md(payload)
    elif name == "edit":
        # Clean success renders a MINIMAL one-liner -- "applied path:line" -- so the
        # model stays oriented without re-reading, and without a JSON dump of the
        # internal `calls_saved` key. No applied ranges -> "ok". Actionable results
        # (failures, rollbacks, diagnostics, reviews, fuzzy matches) keep their
        # structured body and render as JSON via the dispatcher fallback.
        # `vcs_status` rides along as a suffix on the one-liner rather than
        # tripping the JSON fallback -- it's always present on a dirty tree
        # (the just-edited file always shows), so treating it like an
        # actionable key would blow the minimal render up on every call.
        keys = set(payload)
        # `resolved_against` rides as a one-liner suffix like vcs_status rather
        # than tripping the JSON fallback -- a redirected edit is still a clean
        # success and should not render as a structured dump.
        base_keys = keys - {"vcs_status", "resolved_against"}
        if base_keys <= {"calls_saved"}:
            text = "ok"
        elif base_keys <= {"applied", "calls_saved"}:
            applied = payload.get("applied") or []
            if applied and all(isinstance(a, str) for a in applied):
                text = "applied " + ", ".join(applied)
            elif not applied:
                text = "ok"
        if text:
            resolved_against = payload.get("resolved_against")
            if isinstance(resolved_against, str) and resolved_against:
                text = f"{text} | resolved against worktree {resolved_against} (from last bash cwd)"
            vcs_raw = payload.get("vcs_status")
            vcs = vcs_raw if isinstance(vcs_raw, dict) else {}
            vcs_lines = vcs.get("lines")
            if vcs_lines:
                text = f"{text} | {vcs.get('source')}: " + "; ".join(vcs_lines)
    if name in {"search", "grep"} and isinstance(result, dict):
        text = _append_search_verdict_footer(text, result)
    return text or None


def _read_dedup_resource(args: dict[str, Any]) -> str:
    """Stable resource key for delta re-reads of a single file.

    Tracks both calling conventions: a one-entry ``files=[...]`` (the form the
    public `read` tool emits) and the legacy ``path=`` form. A multi-file batch
    read renders several bodies into one text and can't be coherently diffed, so
    it stays untracked. The range/full/lines projection is part of the key so
    different views of the same file never cross-diff.
    """
    files = args.get("files")
    if files is not None:
        if not isinstance(files, list) or len(files) != 1:
            return ""  # batch (or malformed) read: not delta-trackable
        entry = files[0]
        if isinstance(entry, str):
            path, range_spec, expand, head, tail, _summary, _outline = _split_file_opts(entry)
            range_spec = range_spec or ""
            lines_spec = "" if head is None else str(head)
            tail_spec = "" if tail is None else str(tail)
            proj_spec = ""
        elif isinstance(entry, dict):
            path = str(entry.get("path") or "")
            range_spec = str(entry.get("range") or "")
            expand = bool(entry.get("full"))
            _ml = entry.get("lines", entry.get("max_lines"))
            lines_spec = "" if _ml is None else str(_ml)
            tail_spec = "" if entry.get("tail") is None else str(entry.get("tail"))
            proj_spec = str(entry.get("projection_kind") or "")
        else:
            return ""
        if not path:
            return ""
        return f"read:{path}:{range_spec}:{lines_spec}:{tail_spec}:{proj_spec}:{int(bool(expand))}"
    path = str(args.get("path") or "")
    if not path:
        return ""
    range_spec = str(args.get("range") or "")
    max_lines = args.get("lines", args.get("max_lines"))
    max_lines_spec = "" if max_lines is None else str(max_lines)
    projection_spec = str(args.get("projection_kind") or "")
    return f"read:{path}:{range_spec}:{max_lines_spec}::{projection_spec}:{int(bool(args.get('full')))}"


_READ_SUGGEST_PRUNE_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".tox",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".next",
        "target",
    }
)


# Same "not a literal path" shape test as the code-context engine's
# _query_is_pathy_literal: whitespace or a regex/FTS metacharacter means the
# query can't be a literal filesystem path, so skip the stat/walk below.
_QUERY_PATH_HINT_RE = re.compile(r"[\s|*()\[\]^$+?\\=<>{}]")


def _candidate_relpaths_for_query(query: str) -> list[str]:
    """Path forms worth an existence check for a path-shaped query: as given
    (leading slash stripped), and with one leading segment dropped -- a
    phantom repo-name prefix (e.g. "/lemoncrow/benchmarks/x.sh" when the real
    relpath is "benchmarks/x.sh") is a common way an agent mistypes a path.
    """
    stripped = query.strip().lstrip("/")
    candidates = [stripped] if stripped else []
    parts = stripped.split("/")
    if len(parts) > 1:
        trimmed = "/".join(parts[1:])
        if trimmed and trimmed not in candidates:
            candidates.append(trimmed)
    return candidates


def _lookup_unique_basename_in_index(engine: Any, basename: str) -> str | None:
    """Unique indexed file whose basename is `basename`, via the code index's
    `files` table -- a couple ms and deterministic, unlike a live filesystem
    walk over a large repo (the `read`-tool suggestion helper below hits its
    250ms/20k-file budget on this repo before finishing the walk, so which
    file it lands on -- if any -- depends on OS directory-iteration order).
    """
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{engine.db_path}?mode=ro", uri=True, timeout=2.0)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute(
            "SELECT file_path FROM files WHERE repo_id = ? AND (file_path = ? OR file_path LIKE '%/' || ?) LIMIT 2",
            (engine.repo_id, basename, basename),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return rows[0][0] if len(rows) == 1 else None


def _resolve_query_as_existing_file(workspace_root: Path, query: str, engine: Any = None) -> str | None:
    """When `query` names an existing repo file, return its workspace-relative
    path so code_search can pin straight to it instead of ranking.

    Tries the query verbatim as a relpath (and with one bogus leading segment
    stripped), then -- when `engine` is given -- falls back to a unique
    basename lookup against the code index. None when the query isn't
    path-shaped or doesn't resolve to exactly one file.
    """
    q = query.strip()
    if not q or _QUERY_PATH_HINT_RE.search(q) or ("/" not in q and "." not in q):
        return None
    root = workspace_root.resolve()
    for candidate in _candidate_relpaths_for_query(q):
        target = (root / candidate).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            continue  # escapes the workspace, e.g. "../../etc/passwd"
        if target.is_file():
            return str(target.relative_to(root))
    if engine is not None:
        basename = Path(q).name
        if basename:
            return _lookup_unique_basename_in_index(engine, basename)
    return None


def _suggest_paths_for_missing(workspace_root: Path, missing: str, *, limit: int = 3) -> list[str]:
    """Workspace-relative paths whose basename matches the missing file's."""
    name = Path(missing).name
    if not name:
        return []
    lowered = name.lower()
    hits: list[str] = []
    scanned = 0
    # Bound the walk by file count AND wall time: a missed read (often a
    # hallucinated path) otherwise walks the whole tree, turning a failed read
    # into a multi-100ms stall on a large monorepo.
    _suggest_deadline = time.monotonic() + 0.25
    try:
        for dirpath, dirnames, filenames in os.walk(workspace_root):
            dirnames[:] = [d for d in dirnames if d not in _READ_SUGGEST_PRUNE_DIRS]
            scanned += len(filenames)
            for fname in filenames:
                if fname.lower() == lowered:
                    with contextlib.suppress(ValueError):
                        hits.append(str((Path(dirpath) / fname).relative_to(workspace_root)))
                    if len(hits) >= limit:
                        return hits
            if scanned > 20_000 or time.monotonic() > _suggest_deadline:
                break
    except OSError:
        return hits
    return hits


# A trailing ":Lx" / ":Lx-Ly" suffix on a read path is parsed as a line
# range (parity with the edit tool's `file_path:Lx-Ly` form). The canonical
# form requires a colon separator and an "L" prefix on every number
# (e.g. "store.py:L60-L100").
_READ_RANGE_SUFFIX = re.compile(r":L?(\d+)(?:-L?(\d+))?$", re.IGNORECASE)


def _split_read_range_suffix(raw_path: str) -> tuple[str, str | None]:
    """Split a trailing :Lx / :Lx-Ly suffix off a read path.

    Returns (path, range_spec) where range_spec is in the read tool's range
    syntax (e.g. "L60-L100"), or (raw_path, None) when there is no suffix.
    """
    match = _READ_RANGE_SUFFIX.search(raw_path)
    if match is None:
        return raw_path, None
    start = match.group(1)
    end = match.group(2)
    range_spec = f"L{start}-L{end}" if end else f"L{start}"
    return raw_path[: match.start()], range_spec


_semantic_file_memory_cache: dict[str, SemanticFileMemoryCapability] = {}
_semantic_file_memory_lock = threading.Lock()


def _semantic_file_memory(root: Path) -> SemanticFileMemoryCapability:
    """Process-cached SemanticFileMemoryCapability, one per LemonCrow root.

    Constructing it builds a BM25/IDF index over the whole file-memory corpus,
    so a fresh instance per ``read`` call rebuilt that index every time (~140ms
    of tokenise + JSON-parse on each read). FileIndex is file-based/stateless and
    SymbolIndex memoizes its IDF by corpus snapshot, so the instance is safe to
    reuse across the server's worker threads and the index refreshes only when
    the underlying files change.
    """
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    key = str(root)
    cap = _semantic_file_memory_cache.get(key)
    if cap is None:
        with _semantic_file_memory_lock:
            cap = _semantic_file_memory_cache.get(key)
            if cap is None:
                cap = SemanticFileMemoryCapability(root)
                _semantic_file_memory_cache[key] = cap
    return cap


# Bounds every `:summary` body (LLM/outline/heuristic tier) so a summary read
# never grows into the same oversized-payload problem it exists to avoid.
_SUMMARY_TARGET_CHARS = 4096


def _read_summary_response(resolved: Path) -> dict[str, Any]:
    """Build the ``:summary`` response for a single file.

    Ladder: internal-LLM gist when configured and it succeeds (``summarized:
    {model}``); else, for a CODE file, the outline projection itself when it
    fits the same char budget (``summarized:outline`` -- an outline strictly
    dominates a hand-rolled gist at equal cost, and returning it here collapses
    the ``:summary`` -> ``:outline`` follow-up hop for the common, normal-sized
    file); else a type-aware heuristic extractive gist (``summarized:
    heuristic``) for everything else, and for a code file whose OWN outline
    overflows the budget (a very large file). An LLM failure of ANY kind falls
    back silently -- a ``:summary`` read must never fail because the optional
    LLM tier did.
    """
    from lemoncrow.pro.capabilities.semantic_file_memory.capability import _read_source_bounded
    from lemoncrow.pro.capabilities.source_projection import SourceProjection
    from lemoncrow.pro.capabilities.tool_supervision.text_summary import heuristic_summary, llm_summary_tier
    from lemoncrow.pro.capabilities.tool_supervision.tool_output_spill import spill_notice

    source_text, _truncated = _read_source_bounded(resolved)
    original_chars = len(source_text)

    body = ""
    verb = ""
    tier = llm_summary_tier(source_text, target_chars=_SUMMARY_TARGET_CHARS)
    if tier is not None:
        body, verb = tier

    if not body:
        cap = _semantic_file_memory(_lemoncrow_root())
        outline_payload = cap.smart_read(resolved, expand=False, outline_threshold=0)
        if outline_payload.get("mode") == "outline":
            language = str(outline_payload.get("language") or "")
            rendered = _render_read_outline_md(str(resolved), dict(outline_payload.get("outline") or {}), language)
            if len(rendered) <= _SUMMARY_TARGET_CHARS:
                body = rendered
                verb = "summarized:outline"
        if not body:
            body = heuristic_summary(source_text, path=resolved, target_chars=_SUMMARY_TARGET_CHARS)
            verb = "summarized:heuristic"

    footer = spill_notice(
        verb=verb,
        original_chars=original_chars,
        kept_chars=len(body),
        path=resolved,
    )
    return {
        "mode": "summary",
        "summary": f"{body}\n\n{footer}",
        "path": str(resolved),
        "projection": SourceProjection.summary().to_dict(),
    }


# Freshness ledger for blind (no-old) range edits: abs path -> (mtime_ns, size)
# stat signature captured whenever `read`/`code_search` serves that file's
# content. A matching signature proves the requesting model saw the file as it
# currently is on disk, i.e. its :Lx-Ly line numbers still index the same
# bytes. Scoped PER SESSION, mirroring _http_session_ledgers: on the
# multi-client HTTP transport each session's reads prove freshness only for
# that session's own blind range edits (client B can no longer pass the
# staleness check on client A's read). The stdio transport has no request
# ledger and keeps one process-wide "_global" bucket -- one MCP server per
# host window, so that state stays exactly window-scoped. Never refreshed on
# edit-writes -- the entry stays pinned to the serve the model actually read,
# which is exactly the snapshot a later range edit must be relocated against.
# Third element: whether this serve gave line numbers that match disk exactly
# (an expand/:full or explicit :Lx-Ly range) vs a lossy projection (bare
# default read -> minified/compact, or :outline/:summary) whose line numbers
# do NOT index the same bytes. A blind range edit built from the latter would
# silently splice the wrong lines.
# Fourth element: a per-line digest snapshot of the file AS SERVED (empty when
# not captured -- lossy serve, oversized file, unreadable). It turns the guard
# from "did this file change at all" into "did the lines this edit names
# change": drift somewhere else in the file no longer rejects the edit, and a
# pure line SHIFT is repaired by locating the served block at its new offset.
_RANGE_READ_SIGS: OrderedDict[str, dict[str, tuple[int, int, bool, bytes]]] = OrderedDict()
_range_read_sigs_lock = threading.Lock()
_MAX_RANGE_READ_SIG_SESSIONS = _MAX_HTTP_SESSION_LEDGERS

# Per-session recent code_search queries, same session-bucketing rationale as
# _RANGE_READ_SIGS above (HTTP: per client session; stdio: one process-wide
# "_global" bucket per host window). Feeds _check_repeat_query's near-
# duplicate nudge -- measured 11 near-duplicate code_search calls in a single
# debt-benchmark rep (2026-07-25): "fail-on-stale stale-days debt_cmd",
# "stale_days fail_on_stale debt stale age git blame", "_line_age_days git
# blame annotate stale", ... -- each re-phrasing re-sent on every later turn's
# resent history, compounding for the rest of the run.
# Timestamped (not just a bare deque): the stdio "_global" bucket is process-
# wide, so it spans EVERY conversation a long-lived daemon happens to serve
# (a daemon can outlive many separate Cursor/Claude chats -- confirmed this
# session). Without a time bound, a brand-new, unrelated conversation whose
# first query happens to share words with the LAST daemon-wide query (from a
# previous, unrelated chat, possibly hours earlier) would be wrongly blanked.
# Bounding to _RECENT_QUERY_WINDOW_SECONDS keeps the check scoped to "this
# same burst of searching", not "this process's entire lifetime".
_RECENT_CODE_SEARCH_QUERIES: OrderedDict[str, deque[tuple[float, str, frozenset[str]]]] = OrderedDict()
_recent_code_search_queries_lock = threading.Lock()
_MAX_RECENT_QUERY_SESSIONS = _MAX_HTTP_SESSION_LEDGERS
_RECENT_QUERY_HISTORY = 3
_REPEAT_QUERY_SIMILARITY_FLOOR = 0.35
_RECENT_QUERY_WINDOW_SECONDS = 180.0
_QUERY_WORD_RE = re.compile(r"[a-z0-9]+")


def _range_read_sigs() -> dict[str, tuple[int, int, bool, bytes]]:
    """The current request's freshness-signature bucket.

    Keyed exactly like _http_session_ledgers: the per-request ledger installed
    by _set_request_ledger names the HTTP session; without one (stdio) the
    process-wide "_global" bucket is used. Bounded LRU, same cap as the
    session-ledger cache.
    """
    led = getattr(_request_ledger, "value", None)
    sid = led.session_id if isinstance(led, RunLedger) and led.session_id else "_global"
    with _range_read_sigs_lock:
        bucket = _RANGE_READ_SIGS.get(sid)
        if bucket is None:
            if len(_RANGE_READ_SIGS) >= _MAX_RANGE_READ_SIG_SESSIONS:
                _RANGE_READ_SIGS.popitem(last=False)
            bucket = {}
            _RANGE_READ_SIGS[sid] = bucket
        else:
            _RANGE_READ_SIGS.move_to_end(sid)
        return bucket


def _query_words(query: str) -> frozenset[str]:
    """Lowercased alnum tokens of length >=3, for cheap query-similarity."""
    return frozenset(w for w in _QUERY_WORD_RE.findall(query.lower()) if len(w) >= 3)


def _recent_code_search_queries() -> deque[tuple[float, str, frozenset[str]]]:
    """The current session's recent-query bucket (see _RECENT_CODE_SEARCH_QUERIES)."""
    led = getattr(_request_ledger, "value", None)
    sid = led.session_id if isinstance(led, RunLedger) and led.session_id else "_global"
    with _recent_code_search_queries_lock:
        bucket = _RECENT_CODE_SEARCH_QUERIES.get(sid)
        if bucket is None:
            if len(_RECENT_CODE_SEARCH_QUERIES) >= _MAX_RECENT_QUERY_SESSIONS:
                _RECENT_CODE_SEARCH_QUERIES.popitem(last=False)
            bucket = deque(maxlen=_RECENT_QUERY_HISTORY)
            _RECENT_CODE_SEARCH_QUERIES[sid] = bucket
        else:
            _RECENT_CODE_SEARCH_QUERIES.move_to_end(sid)
        return bucket


def _check_repeat_query(query: str) -> bool:
    """True when *query* overlaps heavily (Jaccard >= floor) with a query this
    session ran within the last _RECENT_QUERY_WINDOW_SECONDS -- re-phrasing the
    same investigation rarely surfaces anything new. A hard signal, not a
    hint: an explanatory sentence here just adds tokens to a response the
    caller has already shown it skims past and searches again anyway (measured
    across debt-benchmark reps). The caller returns blank results instead --
    always records *query* into the bucket either way. Time-bounded (not just
    the last _RECENT_QUERY_HISTORY calls) so a brand-new, unrelated
    conversation on the same long-lived daemon never gets blanked because of a
    coincidental word overlap with a stale query from a previous chat.
    """
    words = _query_words(query)
    bucket = _recent_code_search_queries()
    now = time.monotonic()
    is_repeat = False
    if words:
        for ts, _prior_text, prior_words in bucket:
            if now - ts > _RECENT_QUERY_WINDOW_SECONDS:
                continue
            union = words | prior_words
            if not union:
                continue
            if len(words & prior_words) / len(union) >= _REPEAT_QUERY_SIMILARITY_FLOOR:
                is_repeat = True
                break
    bucket.append((now, query, words))
    return is_repeat


# Per-line digest: crc32 + byte length, 8 bytes a line. Both halves must match
# for a line to count as unchanged, so a crc collision alone can never pass a
# changed line off as identical.
_LINE_DIGEST = struct.Struct("<II")
_DIGEST_WIDTH = _LINE_DIGEST.size
# Files above this are stat-tracked but not digest-snapshotted: the snapshot is
# held for the whole session, and a 4 MB file is already ~100k lines / 800 KB of
# digests. Oversized files simply fall back to the old stat-only behaviour.
_MAX_SIG_DIGEST_BYTES = 4_000_000
# Lines of served context demanded around a block that MOVED, tried outermost
# first: the wider window is the stronger evidence, the narrower one still
# works when the drift landed immediately above the target.
_RELOCATE_CONTEXT_RADII = (3, 1)
# Most recently served files that keep a signature (and its digest snapshot).
# The stdio bucket is process-wide and a daemon outlives many chats, so this
# caps what a long session can accumulate; evicting only costs a re-read.
_MAX_RANGE_READ_SIG_PATHS = 512


def _line_digests(text: str) -> bytes:
    """Fixed-width per-line fingerprint of *text*, one 8-byte slot per line.

    Split with ``str.splitlines`` so slot N is line N under exactly the same
    rules the range splice in ``rich_edit`` indexes with (which splits with
    ``splitlines(keepends=True)`` -- same boundaries, terminators kept). A
    bytes-level ``split(b"\\n")`` would disagree on form feeds and U+2028 and
    silently shift every later slot.
    """
    pack = _LINE_DIGEST.pack
    crc = zlib.crc32
    out = bytearray()
    for line in text.splitlines():
        raw = line.encode("utf-8", "surrogatepass")
        out += pack(crc(raw), len(raw) & 0xFFFF_FFFF)
    return bytes(out)


def _aligned_occurrences(haystack: bytes, needle: bytes, limit: int = 8) -> list[int]:
    """Line indices where *needle* (a digest run) occurs in *haystack*.

    ``bytes.find`` can land mid-slot on digest bytes that happen to line up, so
    unaligned hits are skipped rather than reported as a line match.
    """
    hits: list[int] = []
    pos = 0
    while len(hits) < limit:
        found = haystack.find(needle, pos)
        if found < 0:
            break
        if found % _DIGEST_WIDTH == 0:
            hits.append(found // _DIGEST_WIDTH)
            pos = found + _DIGEST_WIDTH
        else:
            pos = found + 1
    return hits


def _relocate_served_range(served: bytes, current: bytes, start_line: int, end_line: int) -> tuple[int, int] | None:
    """Where the lines served as L*start*-L*end* live in the file NOW.

    Returns the same range when those lines are byte-identical on disk (the
    file drifted somewhere the edit doesn't touch -- not a reason to reject),
    the shifted range when the block MOVED and its served neighbours moved
    with it, or None when it is gone or ambiguous -- the only case that still
    has to fail.

    A bare content match is deliberately NOT enough to move an edit. A lone
    ``    return None`` deleted from where it was served and coincidentally
    present somewhere else would be "unique" in the file and relocate the
    splice into unrelated code -- silently, which is the exact failure this
    guard exists to prevent. Requiring the surrounding served lines to travel
    with the block makes a move self-evidencing; radii shrink because an edit
    landing just above the target eats the outer context first.
    """
    n_served = len(served) // _DIGEST_WIDTH
    s0 = start_line - 1
    e0 = end_line
    if s0 < 0 or e0 > n_served or s0 >= e0:
        return None
    block = served[s0 * _DIGEST_WIDTH : e0 * _DIGEST_WIDTH]
    if current[s0 * _DIGEST_WIDTH : e0 * _DIGEST_WIDTH] == block:
        return (start_line, end_line)
    for radius in _RELOCATE_CONTEXT_RADII:
        lo = max(0, s0 - radius)
        hi = min(n_served, e0 + radius)
        if lo == s0 and hi == e0:
            continue  # no served context at this radius -- a bare block match
        hits = _aligned_occurrences(current, served[lo * _DIGEST_WIDTH : hi * _DIGEST_WIDTH])
        if len(hits) == 1:
            new_start = hits[0] + (s0 - lo) + 1
            return (new_start, new_start + (e0 - s0) - 1)
    return None


def _retarget_range_edit(edit: dict[str, Any], path: str, start: int, end: int) -> None:
    """Repoint *edit* at L*start*-L*end*, keeping whichever path key it used."""
    suffix = f":L{start}" if start == end else f":L{start}-L{end}"
    edit["file_path" if "file_path" in edit else "path"] = path + suffix


def _record_read_sig(path: Path | str, *, exact: bool = True) -> None:
    """Best-effort content-signature capture for a file just served to the model.

    ``exact`` (default True -- code_search's inline sections and every other
    caller besides ``_smart_read_single`` always carry real disk line numbers)
    marks whether THIS serve's line numbers are trustworthy for a later blind
    :Lx-Ly range edit. A later minified/summary/outline serve of the same path
    overwrites (downgrades) a prior exact record -- the ledger reflects the
    MOST RECENT serve, matching what the model most recently saw.

    An exact serve also snapshots per-line digests, so the guard can later ask
    whether the *edited lines* changed instead of whether the file did. Those
    snapshots are ~8 bytes a line and the stdio bucket is process-wide (one
    daemon outlives many chats), so the bucket is kept to the most recently
    served _MAX_RANGE_READ_SIG_PATHS files -- re-serving a path moves it back
    to the young end rather than leaving it to age out under its first read.
    """
    try:
        p = Path(path).resolve()
        st = p.stat()
        digests = b""
        if exact and st.st_size <= _MAX_SIG_DIGEST_BYTES:
            with contextlib.suppress(OSError):
                digests = _line_digests(p.read_text(encoding="utf-8", errors="replace"))
        bucket = _range_read_sigs()
        bucket.pop(str(p), None)
        bucket[str(p)] = (st.st_mtime_ns, st.st_size, exact, digests)
        while len(bucket) > _MAX_RANGE_READ_SIG_PATHS:
            bucket.pop(next(iter(bucket)), None)
    except OSError:
        pass


def _smart_read_single(
    path: str,
    range: str | None = None,
    expand: bool = False,
    max_lines: int | None = None,
    tail_lines: int | None = None,
    include_meta: bool = False,
    projection_kind: str | None = None,
    summary: bool = False,
    outline: bool = False,
) -> dict[str, Any]:
    """Execute a single-file smart-read.  Called by both the decorated tool and the batch loop.

    View precedence when `expand`/`range`/`outline`/`summary` combine: expand >
    range > outline > summary (most detailed wins; see the resolution below).
    """
    from lemoncrow.pro.capabilities.source_projection import (
        CompactProjectionResult,
        MinifiedProjectionResult,
        SourceProjection,
    )

    target_path = path
    if not target_path:
        raise ValueError("provide path")
    # Support a trailing ":Lx-Ly" line-range suffix on the path
    # itself (e.g. "store.py:L60-L100"); an explicit range= argument wins.
    target_path, suffix_range = _split_read_range_suffix(target_path)
    if range is None and suffix_range is not None:
        range = suffix_range
    # `:full`, an explicit range (suffix or argument), `:outline`, and
    # `:summary` name four whole/partial-file views of increasing detail.
    # Rather than reject a combination as ambiguous, resolve it by PRECEDENCE
    # -- expand > range > outline > summary -- because the calling LLM can
    # always summarize DOWN from a more detailed result for free, but
    # recovering UP from an over-reduced one costs another turn. Downgrading
    # the less-detailed request silently is safe: the response's own `mode`
    # field says which view was actually served.
    if expand:
        outline = False
        summary = False
    elif range is not None:
        outline = False
        summary = False
    elif outline:
        summary = False
    # tail=N: resolve to a concrete line range so the rest of the read path
    # handles it uniformly (range read is already bounded and efficient).
    if tail_lines is not None and range is None and not expand:
        try:
            total = sum(1 for _ in open(_workspace_path(target_path), encoding="utf-8", errors="replace"))
            start = max(1, total - tail_lines + 1)
            range = f"{start}-"
        except OSError:
            pass
    # Reads may target any path the host process can access — a coding agent
    # legitimately reads configs / sibling repos outside the project, and the
    # host's own permission layer gates the tool call. Writes/edits, by contrast,
    # are confined to the workspace (see tool_smart_edit). Relative paths still
    # resolve against the workspace root.
    resolved = _workspace_path(target_path)
    # Freshness ledger: recorded below, once the actual view served (exact vs
    # a lossy projection) is known -- see _record_read_sig's exact= kwarg.
    # A ranged read is served EXACTLY as requested -- never silently widened.
    # (A "3 partial reads -> serve the whole file" escalation used to live here;
    # it misfired on scattered spot-checks and dumped multi-thousand-line files
    # the caller never asked for. Precise slice > vague complete source.)
    # Binary / non-text guard: never silently UTF-8-decode a binary file into
    # mojibake (a PNG used to come back as garbage, which forced agents into
    # pixel-processing hacks). Sniff the head; if it is not valid UTF-8 text,
    # return a structured signal instead of decoding. Images especially must be
    # recognized as such, not mangled into text.
    if resolved.is_file():
        try:
            with open(resolved, "rb") as _bfh:
                _sniff = _bfh.read(8192)
        except OSError:
            _sniff = b""
        _binary = b"\x00" in _sniff
        if _sniff and not _binary:
            try:
                _sniff.decode("utf-8")
            except UnicodeDecodeError:
                # tolerate a multibyte char split at the 8 KiB sniff boundary
                try:
                    _sniff[:-4].decode("utf-8")
                except UnicodeDecodeError:
                    _binary = True
        if _binary and b"\x00" not in _sniff:
            import mimetypes as _mtypes

            # A text-typed file with stray non-UTF8 bytes but no NULs (a build
            # log with raw control bytes, a latin-1 .txt) is mixed text, not a
            # true binary: serve it lossy-decoded (downstream readers open with
            # errors="replace") instead of a "not decoded" dead-end that
            # measurably sent agents into cat/head/wc/strings archaeology.
            if (_mtypes.guess_type(str(resolved))[0] or "").startswith("text/"):
                _binary = False
        if _binary:
            import mimetypes

            _mt = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
            try:
                _sz = resolved.stat().st_size
            except OSError:
                _sz = len(_sniff)
            if _mt.startswith("image/") and 0 < _sz <= _MAX_INLINE_IMAGE_BYTES:
                # Hand the actual image to the multimodal model as an MCP image
                # content block (drained into the response `content` by the
                # tools/call dispatcher), so it can read the image directly rather
                # than resorting to pixel-processing hacks.
                try:
                    import base64

                    _b64 = base64.b64encode(resolved.read_bytes()).decode("ascii")
                    _imgs = getattr(_tool_call_images, "value", None)
                    if not isinstance(_imgs, list):
                        _imgs = []
                        _tool_call_images.value = _imgs
                    _imgs.append({"type": "image", "data": _b64, "mimeType": _mt})
                    return {
                        "mode": "image",
                        "path": str(resolved),
                        "media_type": _mt,
                        "bytes_total": _sz,
                        "message": f"Image ({_mt}, {_sz} bytes) attached for viewing.",
                    }
                except OSError:
                    pass
            _msg = _binary_read_message(_mt, _sz, suffix=resolved.suffix.lower())
            return {
                "mode": "binary",
                "path": str(resolved),
                "media_type": _mt,
                "bytes_total": _sz,
                "message": _msg,
            }
    if summary and resolved.is_file():
        return _read_summary_response(resolved)
    if max_lines is not None and range is None and not expand:
        payload = cast(dict[str, Any], _core_runtime().smart_read(str(resolved), max_lines=max_lines))
        payload.setdefault("mode", "summary")
        payload["projection"] = SourceProjection.summary().to_dict()
        if include_meta:
            return payload
        payload.pop("cache_hit", None)
        payload.pop("tokens_saved", None)
        return payload

    target = resolved

    # Detect directory input early — return a helpful listing instead of a cryptic error.
    if target.is_dir():
        try:
            entries = sorted(
                os.listdir(target),
                key=lambda x: (not (target / x).is_dir(), x.lower()),
            )
        except OSError:
            entries = []
        return {
            "mode": "directory",
            "path": str(target),
            "entries": [(e + "/" if (target / e).is_dir() else e) for e in entries],
            "message": "Directory, not a file. Use `search` (by name) or `grep` (file_glob_patterns) to find files.",
        }

    # Source-side guard: an exact full read (expand) of a very large file would
    # load the whole file into memory and serialize into one oversized JSON-RPC
    # frame, which disconnects the host. Read only a bounded prefix from disk so
    # gigabytes are never materialized. Range reads are inherently bounded, and
    # non-expand reads of code files become cheap outlines, so neither is touched.
    if expand and range is None:
        disconnect_cap = _max_result_bytes()
        inline_budget = _read_inline_budget_bytes()
        try:
            total_bytes = target.stat().st_size
        except OSError:
            total_bytes = 0
        # Tier 1: enormous file — never materialize it (a multi-MB JSON-RPC frame
        # disconnects the host). Read a bounded byte prefix straight from disk.
        if total_bytes > disconnect_cap:
            # Byte-budgeted disconnect guard: the file is too large to decode
            # in full here (the whole point of this tier), so an exact CHAR
            # count for the original isn't available without violating the
            # never-materialize invariant -- total_bytes/disconnect_cap are
            # used as the closest cheap proxy. The recovery guidance is custom
            # (re-read this SAME file at a narrower range) rather than the
            # generic "narrow the query for full" tail: this is not a spill
            # failure, the file itself is the already-known recovery path.
            notice = f"\n\n[lc: truncated {total_bytes}→{disconnect_cap} chars; re-read narrow range=]"
            prefix_bytes = max(0, disconnect_cap - len(notice.encode("utf-8")) - 1024)
            with open(target, "rb") as fh:
                head = fh.read(prefix_bytes)
            _record_read_sig(target, exact=True)
            return {
                "mode": "full",
                "content": head.decode("utf-8", "replace") + notice,
                "path": str(target),
                "projection": SourceProjection.exact().to_dict(),
                "truncated": True,
                "bytes_total": total_bytes,
            }
        # Tier 2: moderately large — fits in memory but exceeds the host's
        # MCP-output limit, which would persist the result to a temp file and
        # force blind range re-reads. Pre-empt that with a line-aligned prefix
        # plus the EXACT continuation range, so a whole-file read costs at most a
        # couple of clean calls and the bytes are never dumped.
        if inline_budget and total_bytes > inline_budget:
            text = target.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            kept: list[str] = []
            used = 0
            for line in lines:
                line_bytes = len(line.encode("utf-8"))
                if kept and used + line_bytes > inline_budget:
                    break
                kept.append(line)
                used += line_bytes
            shown = len(kept)
            total_lines = len(lines)
            if shown < total_lines:
                notice = f'\n\n[lines 1-{shown} of {total_lines}; range="L{shown + 1}-" for rest]'
                _record_read_sig(target, exact=True)
                return {
                    "mode": "full",
                    "content": "".join(kept) + notice,
                    "path": str(target),
                    "projection": SourceProjection.exact().to_dict(),
                    "truncated": True,
                    "bytes_total": total_bytes,
                    "lines_total": total_lines,
                    "lines_shown": shown,
                }

    cap = _semantic_file_memory(_lemoncrow_root())
    try:
        payload = cap.smart_read(target, range_spec=range, expand=expand, outline_threshold=0 if outline else None)
    except FileNotFoundError as exc:
        # Append nearest basename matches (or an authoritative "no such file
        # anywhere") so the model corrects the path instead of retrying it.
        workspace_root = Path(os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd()))
        suggestions = _suggest_paths_for_missing(workspace_root, target_path)
        if suggestions:
            raise FileNotFoundError(f"{exc}. Did you mean: {', '.join(suggestions)}") from exc
        raise FileNotFoundError(
            f"{exc}. No file named {Path(target_path).name!r} found under {workspace_root} — do not retry this path."
        ) from exc
    mode = payload["mode"]
    # `:outline` forces the outline_threshold above; a mode other than
    # "outline" here means the file genuinely has no outline support (plain
    # text, or too trivial for the AST/tree-sitter/generic cascade to earn its
    # savings) -- self-heal instead of silently shipping full content.
    if outline and mode != "outline":
        raise ValueError(
            f"no outline available for {target_path} (not a code file); "
            "use :summary for a gist or read the file directly"
        )
    content = payload.get("content")
    # Whitespace-minify file bodies before they enter the agent's context
    # (token optimization that works under any host/orchestrator). Only the
    # conservative transform is applied (strip trailing whitespace + collapse
    # 3+ blank-line runs), which the fuzzy edit matcher tolerates. Outline mode
    # carries no body, so it is left untouched.
    projection = SourceProjection.outline() if mode == "outline" else SourceProjection.exact()
    projection_saved = 0
    projection_delta: dict[str, Any] | None = None
    projection_result: CompactProjectionResult | MinifiedProjectionResult | None = None
    exact_read = expand or range is not None
    # M9: a non-expand 'full' body (a text/data file with no outline) over the
    # inline budget would otherwise be head+tail compacted downstream, dropping the
    # middle with no way to continue. Pre-empt with a line-aligned prefix plus an
    # EXACT continuation range (the treatment the expand path gives), and mark it
    # exact so the minify projection below leaves the line numbers intact.
    if mode == "full" and not exact_read and isinstance(content, str):
        _inline_budget = _read_inline_budget_bytes()
        if _inline_budget and len(content.encode("utf-8")) > _inline_budget:
            _src_lines = content.splitlines(keepends=True)
            _kept: list[str] = []
            _used = 0
            for _line in _src_lines:
                _lb = len(_line.encode("utf-8"))
                if _kept and _used + _lb > _inline_budget:
                    break
                _kept.append(_line)
                _used += _lb
            _shown = len(_kept)
            if _shown < len(_src_lines):
                _notice = f'\n\n[lines 1-{_shown} of {len(_src_lines)}; range="L{_shown + 1}-" for rest]'
                content = "".join(_kept) + _notice
                payload["content"] = content
                payload["truncated"] = True
                payload["lines_total"] = len(_src_lines)
                payload["lines_shown"] = _shown
                exact_read = True
    # M9b: an explicit range read carries no inline budget of its own — a
    # deliberate :L1-L10000 over a large (often LLM-generated) file would dump
    # the whole slice, which the host re-pays as cache_read every later turn.
    # Bound it like the expand/full paths: keep a line-aligned prefix and hand
    # back the EXACT continuation range, so the rest is one more call rather than
    # open-ended iteration. A normal (small) range read is untouched.
    if mode == "range" and isinstance(content, str):
        _inline_budget = _read_inline_budget_bytes()
        if _inline_budget and len(content.encode("utf-8")) > _inline_budget:
            _src_lines = content.splitlines(keepends=True)
            _kept = []
            _used = 0
            for _line in _src_lines:
                _lb = len(_line.encode("utf-8"))
                if _kept and _used + _lb > _inline_budget:
                    break
                _kept.append(_line)
                _used += _lb
            _shown = len(_kept)
            if _shown < len(_src_lines):
                _start_m = re.match(r"^L?(\d+)", str(range or ""))
                _start = int(_start_m.group(1)) if _start_m else 1
                _last = _start + _shown - 1
                _notice = f'\n\n[lines {_start}-{_last} of the requested range; range="L{_last + 1}-" for the rest]'
                content = "".join(_kept) + _notice
                payload["content"] = content
                payload["truncated"] = True
                payload["lines_total"] = len(_src_lines)
                payload["lines_shown"] = _shown
    _line_refs_trustworthy = exact_read or mode == "outline"
    if isinstance(content, str) and content and mode in ("full", "range") and not exact_read:
        from lemoncrow.pro.capabilities.prompt_compilation.tokens import count_tokens as _count_gutter_tokens
        from lemoncrow.pro.capabilities.source_projection import (
            ProjectionDelta,
            build_compact_projection,
            build_minified_projection,
            language_for_minify,
            minified_line_gutter,
        )

        language = str(payload.get("language") or "")
        # Prefer the tree-sitter minified view (comments and blank lines
        # dropped, then re-parsed); fall back to the conservative compact
        # whitespace transform when minification does not apply. Callers can
        # pin the conservative compact view via projection_kind="compact".
        force_compact = projection_kind == "compact"
        minify_lang = language_for_minify(str(target))
        if minify_lang is not None and not force_compact:
            minified = build_minified_projection(content, minify_lang, include_mapping=True, path=str(target))
            if minified.applied:
                projection_result = minified
                projection = SourceProjection.minified()
        if projection_result is None:
            compact = build_compact_projection(content, language, include_mapping=True, path=str(target))
            if compact.applied:
                projection_result = compact
                projection = SourceProjection.compact()
        if projection_result is not None:
            projection_saved = projection_result.saved_tokens
            projection_delta = ProjectionDelta(
                path=str(payload.get("path", str(target))),
                lang=language,
                original_tokens=projection_result.original_tokens,
                projected_tokens=projection_result.projected_tokens,
            ).to_dict()
            # Prefix every line with its REAL disk line number so a caller's own
            # :Lx-Ly range edit, read straight off the gutter, is already
            # disk-accurate -- no minified<->disk translation needed, and the
            # model never has to reason about which coordinate space it's in.
            if projection_result.mapping is not None:
                guttered = minified_line_gutter(projection_result.content, projection_result.mapping)
                # The "N\t" prefix on every line has its own token cost, paid
                # AFTER build_minified_projection already decided minification
                # was worthwhile on the un-guttered text -- on a repo sweep
                # (2026-07-25) that overhead alone flipped 321/414 files
                # net-negative (minify.py itself: -112 tokens vs raw). Compare
                # the FINAL guttered size against the original, not the
                # pre-gutter projected size, and only ship the gutter (and
                # trust its line numbers) when it still nets a saving.
                guttered_tokens = _count_gutter_tokens(guttered)
                if guttered_tokens < projection_result.original_tokens:
                    content = guttered
                    _line_refs_trustworthy = True
                    projection_saved = projection_result.original_tokens - guttered_tokens
                    projection_delta = ProjectionDelta(
                        path=str(payload.get("path", str(target))),
                        lang=language,
                        original_tokens=projection_result.original_tokens,
                        projected_tokens=guttered_tokens,
                    ).to_dict()
                    projection = (
                        SourceProjection.minified_with_gutter()
                        if projection.view == "minified"
                        else SourceProjection.compact_with_gutter()
                    )
                else:
                    # Gutter overhead wipes out the saving for this file -- serve
                    # the ungutted projection (still smaller than raw per the
                    # projection_saved/projection_delta set above). Line numbers
                    # are sequential projected position again, NOT disk lines,
                    # so `projection`/`_line_refs_trustworthy` stay at their
                    # pre-gutter (untrustworthy) values -- no :Lx-Ly claim.
                    content = projection_result.content
            else:
                content = projection_result.content
    elif mode == "range":
        projection = SourceProjection.range()
    # Freshness ledger: whether THIS serve's line numbers are trustworthy for a
    # later blind :Lx-Ly range edit (see _record_read_sig's exact= kwarg) -- a
    # minified/compact read counts too when it carries a disk-line gutter; a
    # bare read that neither transform actually applied to (too trivial to earn
    # savings) left `projection` at its untouched SourceProjection.exact()
    # default, which is also trustworthy even without a gutter (content is
    # verbatim disk bytes).
    _record_read_sig(resolved, exact=_line_refs_trustworthy or projection.untransformed_text)
    if mode == "full":
        _record_full_read(resolved)
    # Omit null fields: outline/range/language are absent for most reads (e.g. a
    # range read carries no outline, a plain text read no language), and a null
    # key is pure wire noise the model must skip over. Only attach them when set.
    response: dict[str, Any] = {
        "mode": mode,
        "content": content,
        "path": payload.get("path", str(target)),
        "projection": projection.to_dict(),
    }
    for _opt_key in ("outline", "range", "language", "truncated", "lines_total", "lines_shown"):
        _opt_val = payload.get(_opt_key)
        if _opt_val is not None:
            response[_opt_key] = _opt_val
    ts = int(payload.get("tokens_saved", 0) or 0) + projection_saved
    # Always carry the projection mapping when a projection replaced the body, so
    # downstream edits can map projected coordinates back to source even without
    # include_meta (the top-level mode still reads "full"; the mapping is the
    # authoritative signal that the body is transformed).
    if projection_result is not None and projection_result.mapping is not None:
        response["projection_mapping"] = projection_result.mapping.to_dict()
    # A ranged read (range=Lx-Ly / :Lx-Ly) is an exact, unprojected line
    # slice: the model asked for specific lines and already knows the range, so
    # the outline/language/projection scaffolding is pure redundancy. Return only
    # the requested lines (mode/content/path/range). Full and outline reads keep
    # their metadata; include_meta wins for power callers.
    # NOTE (follow-up): cross-turn dedup ("the model already has these exact
    # lines, return nothing") is handled generically by the dispatcher's
    # _DEDUP_TOOLS path; no read-local dedup needed here.
    if mode == "range" and not include_meta:
        response.pop("outline", None)
        response.pop("language", None)
        response.pop("projection", None)
    if include_meta:
        response["cache_hit"] = bool(payload.get("cache_hit", False))
        response["tokens_saved"] = ts
        if projection_delta is not None:
            response["projection_delta"] = projection_delta
    # Always save real savings via thread-local for the budget recorder
    if ts > 0:
        _tool_call_tokens_saved.value = ts
    return response


# Matches range tokens like L10-L20, 10-20, L10-, 10-, L10, 10
_RANGE_TOKEN_RE = re.compile(r"^L?\d+(-L?\d*)?$", re.IGNORECASE)
# A bare identifier or dotted symbol path (Foo, pkg.Class.method) -- used to
# decide whether a stray `query` on read can be promoted to a symbol lookup.
_SYMBOL_LIKE_RE = re.compile(r"^[A-Za-z_][\w.]*$")


def _split_file_opts(s: str) -> tuple[str, str | None, bool, int | None, int | None, bool, bool]:
    """Parse colon-suffixed options off a file path string.

    Recognised suffixes (applied right-to-left, stops at first unknown token):
      :L10-L20 / :10-20 / :L10-  — line range
      :full                       — whole source
      :head=N                     — first N lines
      :tail=N                     — last N lines
      :summary                    — LLM/heuristic gist instead of full content
      :outline                    — force the structural outline regardless of file size

    Keeps paths that happen to contain colons intact.
    """
    expand_out = False
    range_out: str | None = None
    head_out: int | None = None
    tail_out: int | None = None
    summary_out = False
    outline_out = False
    parts = s.split(":")
    while len(parts) > 1:
        tok = parts[-1]
        if tok in ("full", "full=true", "full=1"):
            expand_out = True
            parts.pop()
        elif tok in ("summary", "summary=true", "summary=1"):
            summary_out = True
            parts.pop()
        elif tok in ("outline", "outline=true", "outline=1"):
            outline_out = True
            parts.pop()
        elif tok.startswith("head="):
            try:
                head_out = int(tok[5:])
                parts.pop()
            except ValueError:
                break
        elif tok.startswith("tail="):
            try:
                tail_out = int(tok[5:])
                parts.pop()
            except ValueError:
                break
        elif _RANGE_TOKEN_RE.match(tok):
            range_out = tok
            parts.pop()
        else:
            break
    return ":".join(parts), range_out, expand_out, head_out, tail_out, summary_out, outline_out


def _recover_read_stray_query(args: dict[str, Any], known_params: frozenset[str]) -> dict[str, Any]:
    """Fold a stray ``query``/``q`` on ``read`` into a usable shape.

    ``read`` has no ``query`` param -- but models that just called ``code_search``
    habitually re-send ``query`` on the follow-up read, which would cost a full
    unknown-argument round-trip that throws away the whole call. If the call
    already names a real target (files/symbol/path/range/...), the query is a
    stray label -> drop it and read the target. If the query is the ONLY locator
    and looks like an identifier, promote it to a symbol read (best-effort
    "show me this named thing"); otherwise just drop it.
    """
    if not isinstance(args, dict):
        return args
    stray_keys = [k for k in ("query", "q") if k in args and k not in known_params]
    if not stray_keys:
        return args
    out = {k: v for k, v in args.items() if k not in stray_keys}
    has_target = any(
        out.get(k) not in (None, "", [])
        for k in ("files", "symbol", "path", "filePath", "file_path", "range", "start_line", "offset")
    )
    if not has_target:
        val = args[stray_keys[0]]
        if isinstance(val, str) and _SYMBOL_LIKE_RE.match(val.strip()):
            out["symbol"] = val.strip()
    return out


@mcp_tool(
    name="read",
    hidden_params=(
        "path",
        "range",
        "start_line",
        "end_line",
        "full",
        "lines",
        "projection_kind",
        "format",
        "include_meta",
        "filePath",
        "offset",
        "limit",
        "force",
    ),
    description=(
        "Read files or exact symbols. Batch the paths/ranges you ALREADY need into ONE call: "
        "files=['a.py', 'b.py:L10-L20', 'c.py:full', 'd.py:head=50', 'e.py:tail=20', 'f.py:summary', 'g.py:outline']. "
        "Need, not might-need — a speculative :full costs more than the turn it saves. "
        ":Lx-Ly exact range; :full full source; :summary gist; :outline structure at any "
        "size (mutually exclusive). Whole file → ONE :full or wide range, never "
        "successive narrow ones. One function/test → symbol='name' or ['a','b'], not a "
        "whole-file read. Default line numbers are disk-accurate for editing; :full is "
        "for exact formatting only. Images (png/jpg/gif/webp/bmp ≤4MB) come back "
        "viewable — read to SEE them, don't write OCR/pixel code."
    ),
    param_aliases={
        "max_lines": "lines",
        "filePath": "path",
        "file_path": "path",
    },
    recover_args=_recover_read_stray_query,
)
def tool_smart_read(
    path: str = "",
    range: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    full: bool = False,
    lines: int | None = None,
    include_meta: bool = False,
    files: Annotated[
        list[str | dict[str, Any]] | None,
        Field(description="Entries: path string or {path, range?, full?, ...} dict."),
    ] = None,
    symbol: Annotated[
        str | list[str] | None,
        Field(description="Exact symbol name or names."),
    ] = None,
    projection_kind: str | None = None,
    filePath: str = "",
    offset: int | None = None,
    limit: int | None = None,
    force: Annotated[
        bool,
        Field(
            description=(
                "Hidden: bypass within-session content dedup and re-emit the full "
                "body even if byte-identical to an earlier read this session."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    """Read a file (or batch of files) by path, or a single symbol by name.

    Symbol mode: read(symbol="name") or read(symbol="pkg.Class.method") returns the
    verbatim source of exactly that symbol — direct index lookup, no FTS expansion.
    Pass a list to fetch multiple: read(symbol=["Foo", "Bar.baz"]).

    File modes: outline (structure only — default for files >200 LOC), range
    (range="42-118", "L42-L118", or open-ended "L42-" for an exact line slice),
    full (small files, or any file with full=true), and compact (safe
    whitespace-only transformation of full reads — not byte-identical source).

    Prefer over native `Read`/`cat` unless the file is known to be small;
    outline mode typically saves 50-90% of tokens on large files. Re-read with
    full=true (or a range) before editing against an outline/compact view.

    BATCH: 2+ files you already know you need → one files=[{path, range?}, ...]
    call, not separate turns (each turn re-reads the whole conversation, ~$0.49
    on large context windows). Need, not might-need: a speculative :full costs
    more than the turn it saves, and an over-budget batch ships as outlines.

    Cross-tool: after editing a file via `edit`, don't re-read it — the edit
    response already confirms the change. When you don't yet know which file
    holds something, use `grep` with mode="with_content" to
    discover and read in one step instead of grep-then-read.
    """
    _ = force  # consumed by the dedup dispatcher wrapper (mcp_server._handle), not here
    if files is not None and symbol is not None:
        # Recovery (don't reject): both given means 'this symbol AS DEFINED IN this
        # file' -- the most precise read the model can ask for. Resolve the symbol
        # scoped to the given file instead of costing a turn on a validation error.
        _scope_path: str | None = None
        if files:
            _first = files[0]
            if isinstance(_first, str):
                _scope_path = _split_file_opts(_first)[0] or None
            elif isinstance(_first, dict):
                _scope_path = str(_first.get("path") or "") or None
        try:
            if isinstance(symbol, list):
                return {"symbols": [_op_node(**_parse_symbol(s), path=_scope_path) for s in symbol]}
            return _op_node(**_parse_symbol(symbol), path=_scope_path)
        except Exception:
            symbol = None
    # `filePath` is an accepted alias for `path` (host Read-tool habit); fold it
    # in before any dispatch so both name the same file.
    if not path and filePath:
        path = filePath
    # offset/limit: the built-in Read tool's paging params (offset = 1-based
    # first line, limit = line count). Fold into range so a vanilla-habit call
    # read(file_path=..., offset=.., limit=..) works unchanged.
    if range is None and start_line is None and offset is not None:
        first = max(1, offset)
        range = f"{first}-{first + limit - 1}" if limit is not None else f"{first}-"
    elif limit is not None and range is None and start_line is None and lines is None:
        lines = limit
    # start_line/end_line integer aliases → range string (wins over any suffix in path).
    if start_line is not None and range is None:
        end = end_line or start_line
        range = f"{start_line}-{end}"

    # Symbol addressing: direct index lookup, no file I/O or FTS expansion.
    if symbol is not None:
        if isinstance(symbol, list):
            return {"symbols": [_op_node(**_parse_symbol(s)) for s in symbol]}
        return _op_node(**_parse_symbol(symbol))

    # Batch mode: process each file spec and return aggregated results.
    if files is not None:
        results = []
        # Parallel to `results`: the path when the entry is a whole-file body
        # eligible for budget downgrade, None when the caller already narrowed it.
        entry_specs: list[str | None] = []
        batch_saved = 0
        for item in files:
            if isinstance(item, str):
                raw_path, item_range, item_expand, item_head, item_tail, item_summary, item_outline = _split_file_opts(
                    item
                )
                spec: dict[str, Any] = {"path": raw_path}
                if item_range is not None:
                    spec["range"] = item_range
                if item_expand:
                    spec["full"] = True
                if item_head is not None:
                    spec["lines"] = item_head
                if item_tail is not None:
                    spec["tail"] = item_tail
                if item_summary:
                    spec["summary"] = True
                if item_outline:
                    spec["outline"] = True
            else:
                spec = item  # dict passthrough for internal callers
            spec_path = str(spec.get("path") or spec.get("file_path") or spec.get("filePath") or "")
            if not spec_path:
                results.append({"error": "path is required in each files entry"})
                entry_specs.append(None)
                continue
            entry_specs.append(
                None
                if (
                    spec.get("range") is not None
                    or spec.get("outline")
                    or spec.get("summary")
                    or spec.get("lines") is not None
                    or spec.get("max_lines") is not None
                    or spec.get("tail") is not None
                )
                else spec_path
            )
            # _smart_read_single writes each file's saving to the thread-local
            # (last write wins). Capture it per file, stamp it on the entry so
            # per-entry baseline de-dup can zero an already-credited file, and
            # accumulate the batch total instead of letting the last file clobber
            # the rest. Reset between iterations so a stale value can't bleed in.
            _tool_call_tokens_saved.value = 0
            try:
                single = _smart_read_single(
                    path=spec_path,
                    range=spec.get("range"),
                    expand=bool(spec.get("full", full)),
                    max_lines=spec.get("lines", spec.get("max_lines", lines)),
                    tail_lines=spec.get("tail"),
                    include_meta=include_meta,
                    projection_kind=spec.get("projection_kind", projection_kind),
                    summary=bool(spec.get("summary", False)),
                    outline=bool(spec.get("outline", False)),
                )
                entry_saved = int(getattr(_tool_call_tokens_saved, "value", 0) or 0)
                if entry_saved > 0:
                    single["tokens_saved"] = entry_saved
                    batch_saved += entry_saved
                results.append(single)
            except Exception as exc:
                results.append({"path": spec_path, "error": str(exc)})
        downgraded = _apply_batch_read_budget(results, entry_specs, include_meta)
        _tool_call_tokens_saved.value = batch_saved
        batch_result: dict[str, Any] = {"files": results}
        if downgraded:
            batch_result["budget_downgraded"] = downgraded
            batch_result["notice"] = (
                f"{len(downgraded)} file(s) exceeded the batch read budget and were served as "
                "outlines. Re-read any of them with an explicit :Lx-Ly range (or :full) for the body."
            )
        # Batched reads collapse N would-be single-file read calls into 1 -- the
        # same honest cross-call credit the edit (distinct files) and sql
        # (batched queries) surfaces already get. Errored entries earned nothing.
        ok_entries = sum(1 for entry in results if isinstance(entry, dict) and not entry.get("error"))
        if ok_entries > 1:
            batch_result["calls_saved"] = ok_entries - 1
        return batch_result

    return _smart_read_single(
        path=path,
        range=range,
        expand=full,
        max_lines=lines,
        include_meta=include_meta,
        projection_kind=projection_kind,
    )


def _snapshot_path(raw_path: str) -> str:
    if "#cell=" in raw_path:
        return raw_path.split("#cell=", 1)[0]
    clean, _range, _full, _head, _tail, _summary, _outline = _split_file_opts(raw_path)
    match = re.search(r"#\d+(?:-\d+)?$", raw_path)
    return clean[: match.start()] if match else clean


def _resolve_snapshot_path(raw_path: str, repo_root: Path) -> tuple[str, Path]:
    """Return a ledger display path and workspace-resolved file path for snapshots."""
    clean = _snapshot_path(raw_path)
    candidate = Path(clean)
    resolved = candidate if candidate.is_absolute() else repo_root / candidate
    resolved = resolved.resolve()
    root = repo_root.resolve()
    try:
        display = str(resolved.relative_to(root))
    except ValueError:
        display = str(resolved)
    return display, resolved


def _collect_touched_paths(edits: list[dict[str, Any]], *, repo_root: str | Path | None = None) -> dict[str, Path]:
    """Extract workspace-resolved file paths referenced in edit descriptors."""
    root = Path(repo_root or Path.cwd()).resolve()
    paths: dict[str, Path] = {}
    for edit in edits:
        raw = str(edit.get("file_path") or edit.get("path") or "")
        if not raw and str(edit.get("kind") or "") == "symbol":
            from lemoncrow.pro.capabilities.tool_supervision.symbol_edit import (
                preview_symbol_edit_path,
            )

            with contextlib.suppress(Exception):
                raw = preview_symbol_edit_path(edit, repo_root=root)
        if raw:
            display, resolved = _resolve_snapshot_path(raw, root)
            paths[display] = resolved
    return dict(sorted(paths.items()))


def _snapshot_paths(paths: dict[str, Path]) -> dict[str, tuple[Path, bool, str | None]]:
    """Snapshot each file's pre-edit state for rollback.

    Returns ``(fp, existed, content)``. ``existed`` distinguishes a file that
    was genuinely absent pre-edit (rollback should delete it) from one that
    existed but could not be read (rollback must NOT delete it -- that would be
    silent data loss). ``existed=True`` with ``content is None`` means the file
    was unreadable: rollback skips it rather than deleting or truncating.
    """
    snap: dict[str, tuple[Path, bool, str | None]] = {}
    for display, fp in paths.items():
        existed = fp.exists()
        content: str | None = None
        if existed:
            try:
                content = fp.read_text(encoding="utf-8")
            except Exception:
                logging.exception("Recovered from broad exception handler")
                content = None
        snap[display] = (fp, existed, content)
    return snap


def _looks_like_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = parts[-1]
    return (
        any(part in {"test", "tests", "spec", "specs", "__tests__"} for part in parts[:-1])
        or name.startswith("test_")
        or "_test." in name
        or ".test." in name
        or ".spec." in name
    )


# Files created by this process's edit tool. Tests the agent authored in this
# session are its own work in progress, not a pre-existing contract to protect.
_SESSION_CREATED_FILES: set[str] = set()


# Test-contract weakening guard. The real reward-hack is an agent "fixing" a red
# test by deleting/loosening an assertion or marking it skip/xfail instead of
# fixing the code. We trip ONLY on that signal -- additive edits (new tests or
# assertions) and in-place assertion modifications pass freely. Heuristic and
# language-fuzzy by design: it errs toward allowing, surfacing a counterexample +
# rollback only on a clear net assertion-removal or skip/xfail addition. Operator
# off-switch: LEMONCROW_TEST_CONTRACT_GUARD=0.
_ASSERTION_RE = re.compile(r"\bassert|\bexpect\s*\(|\bEXPECT_|\bASSERT_|\.should\b")
_SKIP_XFAIL_RE = re.compile(
    r"@(?:pytest\.mark\.)?(?:skip|skipif|xfail)\b|@unittest\.skip|\bt\.Skip[a-zA-Z]*\("
    r"|\bxit\s*\(|\bxdescribe\s*\(|\.skip\s*\("
)


def _test_contract_guard_enabled() -> bool:
    """Whether the test-weakening guard runs (operator off-switch, default on)."""
    return os.environ.get("LEMONCROW_TEST_CONTRACT_GUARD", "").strip().lower() not in ("0", "false", "no", "off")


def _classify_test_weakening(old_content: str, new_content: str) -> str | None:
    """Return a reason if old->new weakens a test contract, else None.

    Weakening = a NET removal of assertion lines, or a NET addition of skip/xfail
    markers. In-place assertion modification (changing an expected value) nets to
    zero and is NOT flagged; purely additive edits are NOT flagged.
    """
    import difflib

    removed: list[str] = []
    added: list[str] = []
    for line in difflib.unified_diff(old_content.splitlines(), new_content.splitlines(), lineterm=""):
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    removed_asserts = sum(1 for ln in removed if _ASSERTION_RE.search(ln))
    added_asserts = sum(1 for ln in added if _ASSERTION_RE.search(ln))
    if removed_asserts > added_asserts:
        return f"net removal of {removed_asserts - added_asserts} assertion(s)"
    added_skips = sum(1 for ln in added if _SKIP_XFAIL_RE.search(ln))
    removed_skips = sum(1 for ln in removed if _SKIP_XFAIL_RE.search(ln))
    if added_skips > removed_skips:
        return f"added {added_skips - removed_skips} skip/xfail marker(s)"
    return None


def _detect_test_weakening(
    snapshots: dict[str, tuple[Path, bool, str | None]],
) -> list[dict[str, str]]:
    """Find edits to existing test files that weaken a test contract.

    Pair signal: a genuine test-contract change almost always rides with a
    production-code change in the same batch (the contract moved with the code),
    whereas a reward-hack weakens the test WITHOUT fixing code. So when the batch
    also edits a non-test file, the weakening is treated as a genuine refactor and
    allowed; only test-ONLY weakening is blocked. Skips files absent pre-edit (new
    files) and tests the agent authored this session.
    """
    findings: list[dict[str, str]] = []
    changed_non_test = False
    for path, (fp, _existed, old_content) in snapshots.items():
        try:
            new_content = fp.read_text(encoding="utf-8") if fp.exists() else None
        except Exception:
            logging.exception("Recovered from broad exception handler")
            continue
        if not _looks_like_test_path(path):
            if new_content != old_content:
                changed_non_test = True
            continue
        if old_content is None or str(fp.resolve()) in _SESSION_CREATED_FILES:
            continue
        if new_content is None or new_content == old_content:
            continue
        reason = _classify_test_weakening(old_content, new_content)
        if reason:
            findings.append({"path": path, "reason": reason})
    # A production-code change in the same batch reads as a genuine refactor (the
    # contract moved with the code), not a reward-hack; allow the weakening. Block
    # only a test-ONLY weakening edit.
    if changed_non_test:
        return []
    return findings


def _compute_and_record_diffs(
    snapshots: dict[str, tuple[Path, bool, str | None]],
) -> None:
    """Record a unified diff of each changed file to the ledger (audit/undo).

    Computed inside the edit lock so a concurrent edit can't race the post-apply
    read. Diffs are never surfaced inline to the caller: echoing old+new content
    back into context costs cache-write now and cache-read on every later turn,
    and the agent can read the file on demand if it needs to verify a change.
    """
    import difflib

    led = _get_ledger()
    for path, (fp, existed, old_content) in snapshots.items():
        try:
            new_content = fp.read_text(encoding="utf-8") if fp.exists() else None
        except Exception:
            logging.exception("Recovered from broad exception handler")
            new_content = None
        if not existed and new_content is not None:
            _SESSION_CREATED_FILES.add(str(fp.resolve()))
        if old_content == new_content:
            continue
        old_lines = (old_content or "").splitlines(keepends=True)
        new_lines = (new_content or "").splitlines(keepends=True)
        diff_text = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"))
        if diff_text:
            led.record_file_event(path=path, event="edit", diff=diff_text)
        else:
            led.record_file_event(path=path, event="edit")


# All spellings an LLM might hallucinate for the old/new pair, in priority order.
# The first key found wins; canonical ``old_string``/``new_string`` is always
# tried last (it is the source of truth and is never overwritten by the alias
# logic — see the guard below).
_OLD_ALIASES = (
    "old",
    "old_str",
    "oldStr",
    "old_text",
    "oldText",
    "oldString",
    "search",
    "find",
    "original",
    "before",
    "source",
)
_NEW_ALIASES = (
    "new",
    "new_str",
    "newStr",
    "new_text",
    "newText",
    "newString",
    "replace",
    "replacement",
    "after",
    "target",
    "result",
    "content",
    "contents",
)


def _normalize_edit_aliases(edit: dict[str, Any]) -> dict[str, Any]:
    """Silently promote any LLM alias for old/new to the canonical names.

    The tool advertises ``old``/``new`` (short aliases for the canonical
    ``old_string``/``new_string``), but LLMs frequently hallucinate other
    spellings: ``oldStr``, ``old_str``, ``oldText``, ``old_text``,
    ``oldString``, ``search``/``replace``, ``original``/``replacement``, etc.
    This function maps every known alias to the canonical pair so the
    downstream apply functions always see ``old_string`` / ``new_string``.

    Returns the edit dict unchanged when the canonical names are already
    present; otherwise copies the dict and promotes the first alias found.
    """
    if "old_string" not in edit:
        for key in _OLD_ALIASES:
            if key in edit:
                edit = {**edit, "old_string": edit[key]}
                break
    if "new_string" not in edit:
        for key in _NEW_ALIASES:
            # Strings only: `replace` doubles as the boolean whole-file flag, and
            # a bool/None must stay a flag, not become the replacement content
            # (matters now that `new` is no longer schema-required).
            if isinstance(edit.get(key), str):
                edit = {**edit, "new_string": edit[key]}
                break
    if "new_string" not in edit and isinstance(edit.get("new_file"), str) and edit["new_file"]:
        # `new` sourced from a file: huge payloads (or a preserved /tmp payload
        # from a prior failed edit) are referenced, never re-inlined -- the
        # failure mode this kills is an LLM regenerating 20KB it already wrote.
        _nf = _workspace_path(edit["new_file"])
        try:
            edit = {**edit, "new_string": _nf.read_text(encoding="utf-8", errors="replace")}
        except OSError as exc:
            raise _ToolArgumentError(
                f"new_file '{edit['new_file']}' could not be read: {exc}. "
                "new_file names a readable file whose FULL content becomes path's new "
                "content; to send content directly, pass it inline via new"
            ) from exc
        # {path: x, new_file: y} with no anchor: y's full content becomes x.
        # An explicit file reference is a deliberate whole-file source, so it
        # gets replace semantics (create or overwrite) instead of the fuzzy
        # old/new ladder or the inline-rewrite size/elision heuristics. An
        # old anchor or a :Lx-Ly path suffix still means a targeted splice.
        if (
            "old_string" not in edit
            and "replace" not in edit
            and "overwrite" not in edit
            and re.search(r":(?:L?\d+)(?:-L?\d*)?$", str(edit.get("path", "")), re.IGNORECASE) is None
        ):
            edit = {**edit, "replace": True}
    return edit


def _require_edits(edits: list[dict[str, Any]]) -> None:
    if not edits:
        raise _ToolArgumentError("edits must include at least one descriptor")
    # The schema requires only `path` (so a hidden new_file retry validates
    # client-side); content presence is enforced here, after alias/new_file
    # normalization. replace/overwrite edits pass through -- the downstream
    # empty-truncation guard already rejects zeroing a non-empty file -- but a
    # bare {path, old} must not silently DELETE the matched text.
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict) or edit.get("kind"):
            continue  # structured kinds (symbol/projection/notebook) carry their own fields
        if "new_string" in edit or edit.get("replace") or edit.get("overwrite"):
            continue
        raise _ToolArgumentError(f"edits[{i}] has no replacement content: provide new (or new_file)")


def _edit_verify_enabled(verify_flag: bool) -> bool:
    """Whether the WS1 executing edit gate should run for this call."""
    if verify_flag:
        return True
    val = os.environ.get("LEMONCROW_EDIT_VERIFY", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _restore_snapshots(
    snapshots: dict[str, tuple[Path, bool, str | None]],
    applied_content: dict[str, str | None] | None = None,
) -> list[str]:
    """Restore files to their pre-edit content (used when the verify gate fails).

    Must run under this call's per-path edit locks. When *applied_content* is
    given (this call's post-apply content per display path), a path is restored
    ONLY if its current on-disk content still equals what this call wrote;
    otherwise a concurrent edit committed in the window after this call released
    its lock, so we skip the restore to avoid clobbering that committed write
    (lost update). Returns the display paths skipped due to such a conflict.
    """
    conflicts: list[str] = []
    for display, (fp, existed, old_content) in snapshots.items():
        try:
            if applied_content is not None and display in applied_content:
                current = fp.read_text(encoding="utf-8") if fp.exists() else None
                if current != applied_content[display]:
                    # A concurrent edit moved this file on after we applied; the
                    # pre-edit content is stale, so restoring it would lose that
                    # update. Leave the concurrent write in place.
                    conflicts.append(display)
                    continue
            if not existed:
                if fp.exists():
                    fp.unlink()
            elif old_content is None:
                # Existed pre-edit but was unreadable at snapshot time: we have
                # no content to restore, so leave the file as-is rather than
                # deleting or truncating it (data-loss safe).
                conflicts.append(display)
                continue
            else:
                fp.write_text(old_content, encoding="utf-8")
        except Exception:
            logging.exception("Recovered from broad exception handler")
    return conflicts


def _apply_edit_verify_gate(
    result: dict[str, Any],
    *,
    touched: list[Path],
    snapshots: dict[str, tuple[Path, bool, str | None]],
    applied_content: dict[str, str | None] | None = None,
    checks: list[str] | None,
    rollback: bool,
    timeout_ms: int,
    repo_root: Path,
) -> None:
    """Run mechanical parse/type checks; attach counterexamples and roll back on failure.

    Silent on pass: nothing is attached when the gate passes (a passing check is
    confirmation noise). Output appears only on a failure/fail-open. This gate does
    not run behavioral tests. Fully fail-open: a gate crash never blocks a
    legitimate edit.
    """
    try:
        from lemoncrow.pro.capabilities.verification.edit_gate import run_edit_gate

        checks_seq = tuple(checks) if checks else ("typecheck",)
        counterexamples = run_edit_gate(
            touched,
            repo_root=repo_root,
            checks=checks_seq,
            timeout_s=max(1.0, timeout_ms / 1000),
        )
        errors = [c for c in counterexamples if c.severity == "error"]
        # Silent on pass: a clean gate is the common case (~83% of edits) and its
        # "passed" object is pure confirmation noise. Only attach output when the
        # gate found something actionable. The rollback-on-failure behavior below
        # is unchanged -- only the success *output* is suppressed.
        if not errors:
            return
        result.setdefault("FIXME", {})["mechanical_checks"] = {
            "passed": False,
            "failures": [{k: v for k, v in c.to_dict().items() if k != "severity" and v is not None} for c in errors],
        }
        if rollback:
            # The gate runs outside the per-file edit locks (so verify can't
            # serialize concurrent edits); re-acquire them just for the rollback
            # restore-write so it can't race a concurrent edit to the same file.
            with contextlib.ExitStack() as _rb_locks:
                for _lock in _edit_path_locks(touched):
                    _rb_locks.enter_context(_lock)
                conflicts = _restore_snapshots(snapshots, applied_content)
            if conflicts:
                result["rollback_conflicts"] = conflicts
            result["rolled_back"] = True
            result["applied"] = []
            result["FIXME"]["mechanical_checks"]["rolled_back"] = True
    except Exception:
        logging.exception("Recovered from broad exception handler")
        result.setdefault("FIXME", {})["mechanical_checks"] = {
            "passed": None,
            "error": "mechanical edit gate failed open",
        }


def _contract_review_enabled() -> bool:
    """Whether post-edit contract-literal discovery runs (operator off-switch, default on)."""
    return os.environ.get("LEMONCROW_CONTRACT_REVIEW", "").strip().lower() not in ("0", "false", "no", "off")


def _loop_review_enabled() -> bool:
    """Whether the spiral nudge runs (operator off-switch, default on)."""
    return os.environ.get("LEMONCROW_LOOP_REVIEW", "").strip().lower() not in ("0", "false", "no", "off")


def _loop_nudge_for_call(name: str, args: dict[str, Any]) -> str | None:
    """Record this tool call against its per-session identical-call counter and
    return a one-line spiral nudge once the same call repeats past threshold.

    Narrow, false-positive-free no-progress signal at the MCP boundary.
    Fail-open: any error returns None so a tool call is never broken by it.
    """
    from lemoncrow.pro.capabilities.tool_supervision.loop_review import (
        SessionLoopTracker,
        call_signature,
        repeat_nudge,
    )

    if call_signature(name, args) is None:
        return None
    session_id = _get_claude_session_id() or "_global"
    with _LOOP_TRACKER_LOCK:
        tracker = _loop_tracker_sessions.get(session_id)
        if tracker is None:
            tracker = SessionLoopTracker()
            _loop_tracker_sessions[session_id] = tracker
            if len(_loop_tracker_sessions) > _MAX_LOOP_TRACKER_SESSIONS:
                for stale in list(_loop_tracker_sessions)[: len(_loop_tracker_sessions) - _MAX_LOOP_TRACKER_SESSIONS]:
                    if stale != session_id:
                        _loop_tracker_sessions.pop(stale, None)
        count = tracker.record(name, args)
    return repeat_nudge(name, count)


def _attach_contract_literal_review(
    result: dict[str, Any],
    edits: list[dict[str, Any]],
    *,
    repo_root: Path,
    touched_paths: list[str],
) -> None:
    """Surface remaining occurrences of contract literals this edit removed (config
    keys, wire fields, kwarg names) in files it did NOT touch.

    These parallel consumers have no call-graph edge to the edited site -- a rename
    or deletion of a quoted literal is invisible to symbol-level callers/callees, so
    the agent routinely fixes one site and misses the rest. Surfacing them while the
    edit's hypothesis is still revisable is the post-edit half of finishing a change
    at every site the contract reaches. Fail-open: a crash never affects the edit.
    """
    if not _contract_review_enabled():
        return
    try:
        from lemoncrow.pro.capabilities.tool_supervision.edit_impact import (
            contract_literal_impact,
            decorator_contract_impact,
            signature_change_impact,
            symbol_contract_impact,
        )

        engine = _code_context_engine(str(repo_root))
        impact = contract_literal_impact(
            edits,
            engine=engine,
            repo_root=repo_root,
            touched_paths=touched_paths,
        )
        sites = list(impact["sites"]) if impact else []
        # Semantic dep that literal matching misses: a removed attribute-providing
        # decorator (e.g. @lru_cache) whose methods callers still use elsewhere.
        sites.extend(decorator_contract_impact(edits, engine=engine, touched_paths=touched_paths))
        # Symbol counterpart of the literal pass: a removed/renamed module-level
        # def/class/constant whose name other files still import or reference.
        sites.extend(symbol_contract_impact(edits, engine=engine, repo_root=repo_root, touched_paths=touched_paths))
        # Signature counterpart: a def that gained a required parameter -- call sites
        # in other files that don't pass it now break.
        sites.extend(signature_change_impact(edits, engine=engine, repo_root=repo_root, touched_paths=touched_paths))
        if sites and _defer_edit_hooks():
            sites = _contract_surface_once(sites)
        if sites:
            # Merge, don't clobber: the lint-diagnostics fold may already have
            # written FIXME as {"diagnostics": [...]}. Losing error-severity
            # diagnostics to a contract-sites list would hide must-act signals.
            existing = result.get("FIXME")
            if isinstance(existing, dict):
                existing["sites"] = sites
            else:
                result["FIXME"] = sites
    except Exception:
        logging.exception("Recovered from broad exception handler")


EDIT_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["edits"],
    "additionalProperties": False,
    "properties": {
        "edits": {
            "type": "array",
            "minItems": 1,
            "description": "File edits to apply in one batch.",
            "items": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, optionally suffixed with :Lx or :Lx-Ly.",
                    },
                    "old": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement or new file content.",
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "Create or replace the whole file.",
                    },
                },
            },
        },
    },
}


def _lift_flattened_edit_args(args: dict[str, Any], known_params: frozenset[str]) -> dict[str, Any]:
    """Recover a flattened single-edit call into the canonical ``edits=[...]`` shape.

    LLMs (especially in a cold session that reads only the short tool
    description) regularly emit the edit descriptor at top level --
    ``edit(path=..., new=..., replace=True)`` -- instead of wrapping it in
    ``edits=[{...}]``. The intent is unambiguous: when ``edits`` is absent and
    the stray keys form an edit descriptor (a target path plus new content),
    wrap them into a single-entry ``edits`` list instead of rejecting the call.
    Anything else (a genuine typo, a path with no content) still surfaces the
    unknown-argument error.
    """
    if "edits" in args:
        return args
    stray = {key: value for key, value in args.items() if key not in known_params}
    if not stray:
        return args
    has_path = any(key in stray for key in ("path", "file_path"))
    has_content = any(key in stray for key in ("new_string", "new_body", "new_file", *_NEW_ALIASES))
    if not (has_path and has_content):
        return args
    lifted = {key: value for key, value in args.items() if key in known_params}
    lifted["edits"] = [stray]
    return lifted


def _applied_entry_path(entry: str | dict[str, Any]) -> str | None:
    """Extract the file path from a raw applied entry, tolerating both shapes.

    Entries can be dicts (``{"path": ...}`` / ``{"file": ...}`` / ``{"file_path": ...}``,
    as emitted by ``apply_rich_edits``) or already-compacted
    strings of the form ``"path:line,start-end"`` (as emitted by
    ``_compact_applied_entries``). A ``#10-20`` line suffix on the path is stripped so
    line-scoped edits to the same file collapse onto one path. Returns ``None`` when no
    path can be recovered.
    """
    if isinstance(entry, str):
        # Compacted form "path:spans" or plain "path"; spans only ever follow the
        # final ":" and contain digits/commas/hyphens, so split on the last ":".
        raw = entry.rsplit(":", 1)[0] if ":" in entry else entry
        raw = raw.strip()
    elif isinstance(entry, dict):
        candidate = entry.get("path") or entry.get("file") or entry.get("file_path")
        raw = str(candidate).strip() if candidate is not None else ""
    else:
        return None
    if not raw:
        return None
    # Drop a ":L10-L20" / "#cell=..." line/cell scope suffix so same-file scopes merge.
    raw = re.sub(r":L\d+(-L\d+)?$", "", raw, flags=re.IGNORECASE)
    return raw.split("#", 1)[0] or raw


def _distinct_edited_files(entries: list[Any]) -> int:
    """Count distinct files across applied entries.

    Built-in MultiEdit already batches multiple same-file hunks into one call, so
    LemonCrow's only honest advantage over a competent baseline is cross-file batching.
    Entries whose path cannot be recovered are counted as their own file so a
    legitimate cross-file edit is never under-credited.
    """
    distinct: set[str] = set()
    unparsed = 0
    for entry in entries:
        path = _applied_entry_path(entry)
        if path is None:
            unparsed += 1
        else:
            distinct.add(path)
    return len(distinct) + unparsed


# A caller-supplied edit path may carry its :Lx-Ly scope suffix; the applied
# spans are authoritative, so strip it to avoid "path:L34-L40:34-40".
_EDIT_PATH_RANGE_RE = re.compile(r":L\d+(?:-L\d+)?$")


def _compact_applied_entries(entries: list[dict[str, Any]]) -> list[str | dict[str, Any]]:
    """Group ordinary edit hunks by path; group every edit "kind" by kind too.

    Ordinary hunk edits collapse to one "path:line,line-line" string per file.
    Structural variants (notebook/projection/symbol/whole-file replace) carry a
    "kind" -- instead of one raw dict per file per kind, bucket them under a
    single {label: [...]} entry per kind: a bare path when the entry has nothing
    beyond path+kind (replace/notebook), or the remaining fields (hunks,
    match_mode, projection_kind, ...) alongside path when there's more to see.
    """
    grouped: dict[str, list[str]] = {}
    kind_groups: dict[str, list[Any]] = {}
    special: list[dict[str, Any]] = []
    _ORDINARY_KEYS = frozenset({"path", "hunks", "match_mode", "result"})
    # match_mode values that carry zero text-matching ambiguity and so compact
    # like "exact": "range" is a pure line-number splice (no old_string search
    # at all), not a match outcome to double-check. Only normalized/placeholder/
    # fuzzy -- genuine near-matches to old_string -- warrant staying a loud dict.
    _UNAMBIGUOUS_MODES = frozenset({"exact", "range"})
    # "replace" reads better as its past-tense outcome; other kinds read fine
    # as the kind name itself ("notebook", "projection", "symbol").
    _KIND_LABELS = {"replace": "replaced"}
    for entry in entries:
        kind = entry.get("kind")
        if kind:
            label = _KIND_LABELS.get(kind, kind)
            rest = {k: v for k, v in entry.items() if k not in ("path", "kind")}
            value: Any = {"path": entry.get("path", ""), **rest} if rest else str(entry.get("path", ""))
            kind_groups.setdefault(label, []).append(value)
            continue
        if set(entry) - _ORDINARY_KEYS:
            special.append(entry)
            continue
        # Non-exact fuzzy matches are actionable: keep as dicts so the model
        # sees match_mode and knows to re-read and verify the divergence.
        match_mode = entry.get("match_mode")
        if match_mode and match_mode not in _UNAMBIGUOUS_MODES:
            special.append(entry)
            continue
        path = _EDIT_PATH_RANGE_RE.sub("", str(entry.get("path", "")))
        spans = grouped.setdefault(path, [])
        for hunk in entry.get("hunks") or []:
            start = hunk.get("line_start")
            end = hunk.get("line_end")
            if isinstance(start, int) and isinstance(end, int):
                spans.append(str(start) if start == end else f"{start}-{end}")
    compact = [f"{path}:{','.join(spans)}" if spans else path for path, spans in grouped.items()]
    kind_entries = [{label: values} for label, values in kind_groups.items()]
    return [*compact, *kind_entries, *special]


# Keys whose presence makes an edit result actionable -- the model must see them.
# `calls_saved` is NOT here: it is internal savings accounting, never model-facing
# (the dispatcher pops it before rendering), so it must not keep a result "loud".
_EDIT_ACTIONABLE_KEYS = (
    # All must-act signals consolidated under FIXME: contract sites, lint
    # diagnostics, mechanical check failures. test_weakening always comes with
    # rolled_back; rollback_conflicts is the rare restore-race edge case.
    "FIXME",
    "test_weakening",
    "rollback_conflicts",
)

# Confirmation fields that are pure noise -- success is implied by the call
# returning, rollback/failure are signalled by their own keys, and atomic writes
# are all-or-nothing so the count carries no information. Stripped from EVERY
# edit result, loud or silent.
_EDIT_NOISE_KEYS = ("writes",)


def _edit_result_is_silent_success(result: dict[str, Any]) -> bool:
    """True when an edit succeeded with nothing the model must act on.

    "Applied" is implied by the call succeeding, so the common case needs no
    confirmation body. Still loud (returns False) for: a non-empty `failed`, a
    `rolled_back: true`, remaining error/warning diagnostics, a contract-literal
    review, a failed verify gate, or a fuzzy match (an `applied` entry retains
    `match_mode` only when it was NOT an exact match -- exact ones are stripped
    upstream -- so the agent is told to re-read and verify the divergence).
    """
    if result.get("rolled_back"):
        return False
    if result.get("failed"):  # only a NON-empty failed list is actionable
        return False
    for key in _EDIT_ACTIONABLE_KEYS:
        if result.get(key):
            return False
    # A non-exact (fuzzy) match survives compaction as a dict still carrying
    # `match_mode`; ordinary exact hunks compact to plain "path:line" strings.
    for entry in result.get("applied") or []:
        if isinstance(entry, dict) and entry.get("match_mode"):
            return False
    return True


def _silence_clean_edit_result(result: dict[str, Any]) -> dict[str, Any]:
    """Strip confirmation noise from every edit result; empty a clean success.

    A clean success carries NO body (the call returning IS the confirmation) --
    only `calls_saved` is kept, which the dispatcher reads for savings accounting
    and strips before rendering, so it never reaches the model. A LOUD result
    (failure, rollback, diagnostics, review, failed gate, fuzzy match) keeps its
    actionable content but still sheds the noise fields: `writes` always, an
    empty `failed`, and a false `rolled_back`.
    """
    if _edit_result_is_silent_success(result):
        # Keep a MINIMAL `applied` range echo (path:line, not a diff) so the model
        # stays oriented and does not re-read the file it just edited. Everything
        # else (writes/hooks/empty-failed) is still dropped.
        silent: dict[str, Any] = {}
        applied = result.get("applied")
        # Only echo the compact "path:line" strings (exact edits). Symbol/special
        # edits keep dict entries -- leave those silent rather than dump raw hunks.
        if applied and all(isinstance(a, str) for a in applied):
            silent["applied"] = applied
        if "calls_saved" in result:
            silent["calls_saved"] = result["calls_saved"]
        # A redirected edit is never silent about being redirected: the whole
        # point of the disclosure is that it survives the clean-success path,
        # which is exactly where a wrong worktree inference would hide.
        if "resolved_against" in result:
            silent["resolved_against"] = result["resolved_against"]
        return silent
    for key in _EDIT_NOISE_KEYS:
        result.pop(key, None)
    if result.get("failed") == []:
        result.pop("failed", None)
    if result.get("rolled_back") is False:
        result.pop("rolled_back", None)
    return result


def _reindex_edited_files(repo_root: Path, touched_paths: list[str]) -> None:
    """Immediately refresh the shared code index for files this edit touched.

    Keeps ``search``/``explore`` consistent with the just-applied edit instead of
    waiting for the engine's autosync poll (~10s). Incremental: re-extracts only
    the touched files (O(edited files)), never a full rebuild, and runs against
    the long-lived, process-shared engine so the next code tool call sees the
    change.

    Runs OFF the edit hot path on a daemon thread: a warm per-file reindex still
    costs hundreds of ms, and the edit response must not block on it. The model's
    next tool call is a network+thinking round-trip away, so the refresh lands
    first in practice; the autosync poll is the backstop if it doesn't. Fail-open;
    off-switch LEMONCROW_EDIT_REINDEX=0.
    """
    if not touched_paths:
        return
    if os.environ.get("LEMONCROW_EDIT_REINDEX", "").strip().lower() in ("0", "false", "no", "off"):
        return

    def _run() -> None:
        try:
            _code_context_engine(str(repo_root))._reindex_files(touched_paths)
        except Exception:
            logging.exception("Recovered from broad exception handler")

    threading.Thread(target=_run, name="lemoncrow-edit-reindex", daemon=True).start()


# Max lint/type diagnostics folded into an edit result's FIXME block; the rest
# collapse to a "+K more" line (they are one linter run away, not lost).
#
# Kept deliberately small. This block is appended to EVERY edit result, so it is
# re-billed on every later round-trip of the session -- a 20-line lint dump after
# each of 8 edits is a wall of text the agent pays for repeatedly. A handful is
# enough to signal "this file has findings, act on them"; the tail is one linter
# run away.
_EDIT_DIAG_CAP = 5
# Same rationale as _EDIT_DIAG_CAP: a dirty tree with a long status listing
# must not dump an unbounded VCS report into the edit result.
_EDIT_VCS_CAP = 10

# Retry anchors exist to let the agent re-issue a rejected range edit with an
# `old=` that matches disk -- they are NOT a content channel. Unbounded, a batch
# of rejected edits echoes every target region back verbatim: one measured Cursor
# run returned an 11,433-char rejection payload (28x the 204-char native edit
# result) because four failed edits each carried their full region. A whole-line
# prefix is still an exact substring of the region, so a bounded anchor stays
# directly usable as `old=`.
_RETRY_ANCHOR_CHARS = 400


def _anchor_snippet(text: str, limit: int = _RETRY_ANCHOR_CHARS) -> str:
    """Bound a retry anchor to whole leading lines within ``limit`` chars.

    Truncating on a line boundary keeps the result a valid substring of the
    region, so passing it straight back as ``old=`` still anchors the retry.
    """
    if len(text) <= limit:
        return text
    kept: list[str] = []
    used = 0
    for line in text.splitlines(keepends=True):
        if kept and used + len(line) > limit:
            break
        kept.append(line)
        used += len(line)
    return "".join(kept) if kept else text[:limit]


@mcp_tool(
    name="edit",
    input_schema=EDIT_TOOL_INPUT_SCHEMA,
    description=(
        "Batch file edits: edits=[{path: 'f.py:L10-L14', new}, ...] — many hunks per "
        "call, even same-file (ranges use the original snapshot). {path, old, new} "
        "only without a fresh range. Whole or brand-new file: {path, new, "
        "replace:true}. Minified-view line numbers → add :minified "
        "('f.py:minified:L10-L14'). No re-read after success."
    ),
    param_aliases={"post_edit_hooks": "hooks"},
    # Policy knobs, not agent choices: accepted by name (tests, power use) but
    # not advertised -- the defaults are right for LLM callers.
    hidden_params=(
        "atomic",
        "hooks",
        "post_edit_timeout_ms",
        "verify",
        "verify_checks",
        "verify_rollback",
        "verify_timeout_ms",
    ),
    recover_args=_lift_flattened_edit_args,
)
def tool_smart_edit(
    edits: list[dict[str, Any]],
    atomic: bool = True,
    hooks: bool = True,
    post_edit_timeout_ms: int = 30_000,
    verify: bool = False,
    verify_checks: list[str] | None = None,
    verify_rollback: bool = True,
    verify_timeout_ms: int = 60_000,
) -> dict[str, Any]:
    """Apply many mechanical edits across files in one deterministic call.

    Prefer fresh range edits: {path: "foo.py:L10-L20", new: "..."}. Batch
    many range+new hunks in one call, including multiple hunks in the same file;
    ranges resolve against the original snapshot. Use old_string/new_string only
    when no fresh range is available.

    Descriptor shapes:
      - Rich: {path|file_path, new|new_string, old|old_string?, replace?}
      - Structured: notebook cell, symbol, or projection edits

    Returns ordinary successful hunks as {applied: ["path:line,start-end", ...]};
    failures and edits carrying special metadata remain structured.
    """
    # Resolve the edit root the same way reads do (honors CLAUDE/LEMONCROW
    # workspace env + per-request project override) so write-confinement below
    # matches the active workspace.
    repo_root = _workspace_root()
    # A session that entered a git worktree keeps calling this same long-lived
    # server, whose _workspace_root() still names the MAIN CHECKOUT. A relative
    # edit path therefore resolved to the wrong copy of the file -- and passed
    # write-confinement unnoticed, because in-repo worktrees
    # (<repo>/.claude/worktrees/...) sit *under* that root. Resolve relative
    # paths against the worktree the session is demonstrably working in.
    #
    # This is an INFERENCE (from the last bash cwd), so it is never silent:
    # `resolved_against` rides on the result, survives the clean-success
    # squelch, and renders as a suffix on the one-liner. Absolute paths are
    # unaffected -- _resolve_snapshot_path only applies a root to relative ones.
    _session_worktree = _session_worktree_root(repo_root)
    _edit_root = _session_worktree or repo_root
    edits = [_normalize_edit_aliases(e) for e in edits]
    _require_edits(edits)

    paths = _collect_touched_paths(edits, repo_root=_edit_root)
    # Confine writes to the workspace root plus any additional directories from
    # Claude Code's additionalDirectories setting or LEMONCROW_ADDITIONAL_DIRS env.
    # Read tools accept any absolute path; writes need explicit opt-in.
    # Path("/tmp").resolve() as well as "/tmp": on macOS /tmp is a symlink to
    # /private/tmp, and the candidates below are resolved, so the bare literal
    # never matched and the /tmp allowance was dead on that platform.
    _extra_roots = [*_claude_additional_dirs(repo_root), Path("/tmp"), Path("/tmp").resolve()]
    if _session_worktree is not None:
        _extra_roots.append(_session_worktree)
    _allowed_edit_roots = [repo_root, _edit_root, *_extra_roots]

    _escaped_edit_paths = [
        str(_p) for _p in paths.values() if not any(_p == _r or _p.is_relative_to(_r) for _r in _allowed_edit_roots)
    ]
    if _escaped_edit_paths:
        return {
            "failed": [
                {
                    "paths": _escaped_edit_paths,
                    "error": (
                        "edit path escapes the workspace root; add the directory via "
                        "permissions.additionalDirectories (or the legacy top-level "
                        "additionalDirectories) in settings.json or settings.local.json under "
                        "~/.claude/ or <workspace>/.claude/ (applies immediately, no restart), "
                        "or LEMONCROW_ADDITIONAL_DIRS env on the lc MCP server entry (requires "
                        "reconnecting) to allow edits there"
                    ),
                }
            ],
            "rolled_back": True,
        }
    # Freshness guard for blind range edits (:Lx-Ly / replace_range with no old
    # anchor): the line numbers were copied from an earlier read/code_search,
    # so they index the file AS SERVED then. If this window's server never
    # served the file, or the lines it names no longer hold what was served
    # (formatter, another agent, our own earlier edit call), a blind splice
    # would replace the WRONG lines silently -- old-anchored edits self-verify,
    # blind ranges cannot. A file that merely CHANGED is not stale: the guard
    # relocates the served block first and only rejects what it cannot place
    # unambiguously, with the fix spelled out. LEMONCROW_RANGE_EDIT_GUARD=0
    # disables.
    if os.environ.get("LEMONCROW_RANGE_EDIT_GUARD", "1") != "0":
        from lemoncrow.pro.capabilities.source_projection import language_for_minify, translate_minified_line_range
        from lemoncrow.pro.capabilities.tool_supervision.rich_edit import _parse_target

        _session_sigs = _range_read_sigs()
        _stale: list[dict[str, Any]] = []
        for _i, _ed in enumerate(edits):
            if not isinstance(_ed, dict) or _ed.get("old_string") or _ed.get("replace") or _ed.get("overwrite"):
                continue
            if str(_ed.get("kind") or "") in ("symbol", "projection"):
                continue
            _raw = str(_ed.get("file_path") or _ed.get("path") or "")
            if not _raw:
                continue
            _spec = _parse_target(_raw)
            _is_range = _spec.start_line is not None and "new_string" in _ed
            if not _is_range:
                continue
            _end_line = _spec.end_line or _spec.start_line
            try:
                _rp = _workspace_path(_spec.path).resolve()
                _st = _rp.stat()
            except OSError:
                continue  # missing file fails downstream with a clearer error
            _sig = _session_sigs.get(str(_rp))
            _cur = (_st.st_mtime_ns, _st.st_size)
            _stat_fresh = _sig is not None and _sig[:2] == _cur
            if _stat_fresh and _sig is not None and _sig[2] and not _spec.minified:
                continue
            # Everything below works off current disk content: read it ONCE and
            # share it between the relocation check, the :minified translation
            # and the retry anchor (each used to re-read the file itself).
            try:
                _disk_content: str | None = _rp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                _disk_content = None
            _served_digests = _sig[3] if _sig is not None else b""
            _cur_digests = _line_digests(_disk_content) if _disk_content is not None and _served_digests else b""
            # Drift is not staleness. That the file changed says nothing about
            # the lines THIS edit names: when they are still byte-identical to
            # what was served, the splice lands on exactly the content the
            # model saw, and when the whole block merely MOVED (an insertion
            # above it -- most often this session's own earlier edit), locating
            # it repairs the range instead of costing a re-read turn -- but a
            # move must carry the served neighbours with it, so a block that
            # is gone (or whose neighbourhood can't be found exactly once)
            # still fails rather than guessing. Requires an exact serve (a
            # lossy view's line numbers were never disk lines) and a snapshot.
            if (
                not _spec.minified
                and _sig is not None
                and _sig[2]
                and _served_digests
                and _cur_digests
                and _spec.start_line is not None
                and _end_line is not None
            ):
                _relocated = _relocate_served_range(_served_digests, _cur_digests, _spec.start_line, _end_line)
                if _relocated is not None:
                    if _relocated != (_spec.start_line, _end_line):
                        _retarget_range_edit(_ed, _spec.path, _relocated[0], _relocated[1])
                    continue
            # A range edit needs translation when the ledger says THIS window
            # actually served the file fresh (read/code_search, and the served
            # bytes are still on disk) -- :minified only changes HOW that fresh
            # serve is interpreted (as minified-view line numbers instead of
            # disk-direct ones); it is NOT a bypass of freshness itself. A
            # :minified flag against a file never read this window, or changed
            # since, has no honest anchor to translate against -- translating
            # it against whatever is on disk NOW would silently resolve to the
            # wrong lines if the file drifted (proven: a stale :minified range
            # survived translation against a since-edited file and renamed the
            # wrong function). Re-derive the same projection from current disk
            # content (deterministic when unchanged) and map the range back to
            # real disk lines -- self-verifying, fails closed (None) on
            # anything ambiguous (out of bounds, spans a dropped comment, not
            # minify-eligible); only then does the reject below fire.
            # Content-identical counts as fresh even when mtime/size moved (a
            # touch, or a rewrite of the same bytes): the projection re-derives
            # identically, so the translation is exactly the one the serve had.
            _ledger_fresh = _stat_fresh or (bool(_served_digests) and _served_digests == _cur_digests)
            if _ledger_fresh:
                _translated: tuple[int, int] | None = None
                _lang = language_for_minify(str(_spec.path))
                if _lang is not None and _spec.start_line is not None and _end_line is not None and _disk_content:
                    _translated = translate_minified_line_range(
                        _disk_content, _lang, _spec.start_line, _end_line, path=_spec.path
                    )
                if _translated is not None:
                    _retarget_range_edit(_ed, _spec.path, _translated[0], _translated[1])
                    continue
                _why = (
                    "was explicitly flagged :minified, but that range doesn't resolve unambiguously "
                    "against current disk content"
                    if _spec.minified
                    else "was last served here as a minified/summary view whose line numbers don't match disk"
                )
                _fix = "widen the range to include a touched comment explicitly, or pass old to anchor the edit"
            else:
                _why = (
                    "was explicitly flagged :minified, but wasn't read fresh in this window "
                    "(never served, or the file changed on disk since)"
                    if _spec.minified
                    else (
                        (
                            "changed on disk since this window last read it, and the lines it names no "
                            "longer hold what was served there (that content is gone, or now sits in "
                            "more than one place)"
                            if _served_digests
                            else "changed on disk since this window last read it"
                        )
                        if _sig is not None
                        else "was not served by read/code_search in this window"
                    )
                )
                _fix = "read the exact range first, or pass old to anchor the edit"
            # Ship the exact current disk content around the requested range so
            # the retry doesn't cost a separate read turn: the model can supply
            # old= from this excerpt directly (rewriting a stripped comment back
            # in if it belongs there), or narrow/widen the range itself. Center
            # on the real DISK region -- for a :minified-flagged range, _spec's
            # numbers are minified-space and would point at the wrong lines
            # here, so resolve each endpoint on its own (a single line rarely
            # hits the same dropped-comment ambiguity a wider span does) before
            # falling back to the raw numbers.
            # _spec.start_line/_end_line are only genuinely disk-space numbers
            # when the range wasn't :minified-flagged; for a flagged range that
            # failed to resolve at all below, there's no honest disk anchor to
            # show context around, so retry_with is omitted rather than shown
            # against the wrong lines.
            _ctx_center_start: int | None = _spec.start_line
            _ctx_center_end: int | None = _end_line
            if _spec.minified and _spec.start_line is not None and _end_line is not None:
                _ctx_center_start = _ctx_center_end = None
                _lang_ctx = language_for_minify(str(_spec.path))
                if _lang_ctx is not None and _disk_content:
                    _start_resolved = translate_minified_line_range(
                        _disk_content, _lang_ctx, _spec.start_line, _spec.start_line, path=_spec.path
                    )
                    _end_resolved = translate_minified_line_range(
                        _disk_content, _lang_ctx, _end_line, _end_line, path=_spec.path
                    )
                    if _start_resolved is not None:
                        _ctx_center_start = _start_resolved[0]
                    if _end_resolved is not None:
                        _ctx_center_end = _end_resolved[1]
            _retry_with: dict[str, Any] | None = None
            _cur_lines = _disk_content.splitlines(keepends=True) if _disk_content is not None else None
            if _cur_lines is not None and _ctx_center_start is not None and _ctx_center_end is not None:
                _ctx_start = max(1, _ctx_center_start - 2)
                _ctx_end = min(len(_cur_lines), _ctx_center_end + 2)
                if _ctx_start <= _ctx_end:
                    _retry_with = {
                        "path": f"{_spec.path}:L{_ctx_start}-L{_ctx_end}",
                        "old_string": _anchor_snippet("".join(_cur_lines[_ctx_start - 1 : _ctx_end])),
                        "hint": "disk content now -- retry with old= set to (part of) this, or a corrected :Lx-Ly range",
                    }
            _entry: dict[str, Any] = {
                "edit_index": _i,
                "edit_file": _raw,
                "error": (
                    f"blind range edit rejected: {_spec.path!r} {_why}, so its line numbers may point at different content now -- {_fix}"
                ),
            }
            if _retry_with is not None:
                _entry["retry_with"] = _retry_with
            _stale.append(_entry)
        if _stale:
            return {"applied": [], "failed": _stale, "rolled_back": True}
    # Serialize the snapshot/apply/write critical section per touched file so two
    # concurrent edit calls cannot read-modify-write the same file and lose one
    # update. Locks are ordered by path (inside _edit_path_locks) to avoid
    # deadlock and release on every return below via the ExitStack.
    # This call's post-apply content per display path, captured under the edit
    # lock. The verify gate (which runs after the lock releases) uses it to skip
    # restoring any file a concurrent edit moved on, avoiding a lost update.
    applied_content: dict[str, str | None] = {}
    with contextlib.ExitStack() as _edit_locks:
        for _lock in _edit_path_locks(list(paths.values())):
            _edit_locks.enter_context(_lock)
        snapshots = _snapshot_paths(paths)

        from lemoncrow.pro.capabilities.tool_supervision.rich_edit import apply_rich_edits

        result = apply_rich_edits(edits, atomic=atomic, repo_root=_edit_root, allowed_roots=_extra_roots)

        # Sync the long-lived engine's index-version cache so the next explore
        # call gets a cache miss and re-queries the FTS5 index (which the
        # background reindex thread and apply_rich_edits both keep up to date).
        # Without this the cached version never changes between tool calls, so
        # explore returns stale pre-edit results on every subsequent invocation.
        try:
            _code_context_engine(str(repo_root))._index_version_cached = None
        except Exception:
            pass

        if not result.get("failed") and not result.get("rolled_back"):
            if _test_contract_guard_enabled():
                weakenings = _detect_test_weakening(snapshots)
                if weakenings:
                    _restore_snapshots(snapshots)
                    return {
                        "failed": [
                            {
                                "paths": [w["path"] for w in weakenings],
                                "error": (
                                    "Edit rolled back: it weakened an existing test contract ("
                                    + "; ".join(f"{w['path']}: {w['reason']}" for w in weakenings)
                                    + "). This was a test-only edit -- fix the production code in the SAME edit so "
                                    "the original assertions pass (a genuine contract change that also edits code is "
                                    "allowed)."
                                ),
                            }
                        ],
                        "rolled_back": True,
                        "test_weakening": weakenings,
                    }
            if hooks:
                from lemoncrow.pro.capabilities.tool_supervision.post_edit_hooks import (
                    HookConfig,
                    run_post_edit_hooks,
                )

                try:
                    hook_result = run_post_edit_hooks(
                        [str(p) for p in paths.values()],
                        repo_root=repo_root,
                        # Disable lint-autofix and formatting: a silent
                        # linter/formatter rewriting the agent's just-applied
                        # edit is surprising and unwanted. Diagnostics
                        # (report-only) still run here.
                        config=HookConfig(
                            total_timeout_s=post_edit_timeout_ms / 1000,
                            run_lint_autofix=False,
                            run_format=False,
                            run_organize_imports=False,
                        ),
                    )
                    result["diagnostics"] = [
                        {
                            "file": d.file,
                            "line": d.line,
                            "col": d.col,
                            "severity": d.severity,
                            "message": d.message,
                            "code": d.code,
                            "source": d.source,
                        }
                        for d in hook_result.diagnostics
                    ]
                    result["hooks"] = {
                        "ran": hook_result.steps_ran,
                        "skipped": hook_result.steps_skipped,
                        "failed_steps": hook_result.steps_failed,
                        "total_ms": hook_result.total_ms,
                    }
                    if hook_result.vcs_status:
                        _vcs_lines = hook_result.vcs_status
                        if len(_vcs_lines) > _EDIT_VCS_CAP:
                            _dropped_vcs = len(_vcs_lines) - _EDIT_VCS_CAP
                            _vcs_lines = [*_vcs_lines[:_EDIT_VCS_CAP], f"... +{_dropped_vcs} more"]
                        result["vcs_status"] = {"source": hook_result.vcs_source, "lines": _vcs_lines}
                except Exception as hook_exc:
                    logging.exception("Recovered from broad exception handler")
                    result["hooks"] = {"error": str(hook_exc)}
            # WS1 edit-loop correctness gate: optional executing parse + scoped
            # mypy/pytest verification with rollback. Opt-in via the `verify` arg or
            # the LEMONCROW_EDIT_VERIFY env var; fully fail-open.
            # Diffs are recorded to the ledger (audit/undo) but never surfaced
            # inline: a unified diff echoes old+new content back into context
            # (cache-write now, cache-read on every later turn) for a signal the
            # agent can get on demand by reading the file. The compact `applied`
            # line ranges confirm success; a non-exact match still exposes
            # match_mode on its applied entry, so the agent knows to re-read and
            # verify when a fuzzy match may have diverged from what was asked.
            _compute_and_record_diffs(snapshots)
            for _disp, _fp in paths.items():
                try:
                    applied_content[_disp] = _fp.read_text(encoding="utf-8") if _fp.exists() else None
                except Exception:
                    logging.exception("Recovered from broad exception handler")
                    applied_content[_disp] = None
            _applied = result.get("applied") or []
            # match_mode is only informative when it is not the default exact match.
            for entry in _applied:
                if isinstance(entry, dict) and entry.get("match_mode") == "exact":
                    entry.pop("match_mode", None)

    # WS1 edit-loop correctness gate: optional executing parse + scoped mypy/pytest
    # verification with rollback. Run OUTSIDE the per-file edit locks (released at
    # the end of the with-block above) so a slow verify (up to verify_timeout_ms)
    # can't serialize concurrent edits to the same file; the gate re-acquires the
    # locks only for its rollback restore-write. Opt-in via `verify` or
    # LEMONCROW_EDIT_VERIFY; fully fail-open.
    if not result.get("failed") and not result.get("rolled_back") and _edit_verify_enabled(verify):
        _apply_edit_verify_gate(
            result,
            touched=list(paths.values()),
            snapshots=snapshots,
            applied_content=applied_content,
            checks=verify_checks,
            rollback=verify_rollback,
            timeout_ms=verify_timeout_ms,
            repo_root=repo_root,
        )

    # Fold lint diagnostics (errors/warnings only) into FIXME so all must-act
    # signals surface under one key. Informational notes are dropped as noise.
    if "diagnostics" in result:

        def _diag_in_repo_root(d: dict[str, Any], root: Path) -> bool:
            raw = d.get("file", "")
            if not raw:
                return False
            path = Path(raw)
            if not path.is_absolute():
                path = root / path
            return _is_within_root(path.resolve(), root)

        result["diagnostics"] = [
            d
            for d in result["diagnostics"]
            if d.get("severity") in ("error", "warning") and _diag_in_repo_root(d, repo_root)
        ]
        if not result["diagnostics"]:
            result.pop("diagnostics")
        else:

            def _fmt_diag(d: dict[str, Any], root: Path) -> str:
                raw = d.get("file", "")
                try:
                    rel = str(Path(raw).relative_to(root))
                except ValueError:
                    rel = raw
                loc = f"{rel}:L{d['line']}" if d.get("line") else rel
                code = d.get("code", "")
                msg = d.get("message", "")
                return f"{loc} {code}: {msg}" if code else f"{loc}: {msg}"

            _diag_lines = [_fmt_diag(d, repo_root) for d in result.pop("diagnostics")]
            # Cap: a touched file with many pre-existing findings must not dump
            # an unbounded lint report into the edit result.
            if len(_diag_lines) > _EDIT_DIAG_CAP:
                _dropped = len(_diag_lines) - _EDIT_DIAG_CAP
                _diag_lines = [
                    *_diag_lines[:_EDIT_DIAG_CAP],
                    f"... +{_dropped} more (run the linter for the full list)",
                ]
            result.setdefault("FIXME", {})["diagnostics"] = _diag_lines
    # Strip verbose hooks metadata — callers don't need step details.
    result.pop("hooks", None)

    # Honest cross-file batching credit: Claude Code's built-in MultiEdit already
    # batches multiple hunks within a single file into one call, so collapsing
    # same-file hunks is no saving vs a competent baseline. LemonCrow's genuine
    # advantage is only batching edits across *distinct files*, so credit
    # (distinct files - 1) calls. The dispatcher reads this and writes it into the
    # response's content[].saved.calls field.
    applied_entries = result.get("applied") or []
    distinct_files = _distinct_edited_files(applied_entries)
    if distinct_files > 1:
        result.setdefault("calls_saved", distinct_files - 1)
    # Avoided retry-loop credit: a hunk that landed via non-exact RECOVERY
    # (normalized/placeholder/fuzzy/minified re-match) is an edit a byte-exact
    # vanilla Edit would have rejected — costing a failed call, a re-read, and
    # a retry: 2 extra roundtrips per recovered file (the failed-edit → Read →
    # Edit chain). "range" is a caller-chosen mode and "noop" applied nothing
    # — no credit.
    if not result.get("rolled_back"):
        recovered_files = {
            str(entry.get("path", ""))
            for entry in applied_entries
            if isinstance(entry, dict)
            and str(entry.get("match_mode") or "") in ("normalized", "placeholder", "fuzzy", "minified")
        }
        if recovered_files:
            result["calls_saved"] = int(result.get("calls_saved", 0) or 0) + 2 * len(recovered_files)
    if applied_entries and not result.get("failed") and not result.get("rolled_back"):
        result["applied"] = _compact_applied_entries(applied_entries)
    if not result.get("failed") and not result.get("rolled_back"):
        _attach_contract_literal_review(
            result,
            edits,
            repo_root=repo_root,
            touched_paths=[str(p.relative_to(repo_root)) for p in paths.values() if p.is_relative_to(repo_root)],
        )
        # Incremental: refresh the shared index for the touched files now, so a
        # follow-up search/explore reflects this edit without the autosync lag.
        _reindex_edited_files(repo_root, [str(p) for p in paths.values()])
    # Disclose the worktree redirect. It is an inference, so a wrong one has to
    # be visible in the same breath as the write it misdirected.
    if _session_worktree is not None:
        result["resolved_against"] = str(_session_worktree)
    return _silence_clean_edit_result(result)


SQL_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "connect",
                "tables",
                "schema",
                "table",
                "relationships",
                "search",
                "lint",
                "query",
            ],
            "description": "table/search need name; lint/query need sql or queries[].",
        },
        "name": {
            "type": "string",
            "description": "Target table for action=table, or keyword for action=search.",
        },
        "sql": {
            "type": "string",
            "description": "SQL string for action=lint or action=query.",
        },
        "queries": {
            "type": "array",
            "description": "Batch for action=query: [{name, sql}, ...]. Prefer over repeated calls.",
            "items": {
                "type": "object",
                "required": ["sql"],
                "properties": {
                    "name": {"type": "string"},
                    "sql": {"type": "string"},
                },
            },
        },
        "connection": {
            "type": "string",
            "description": (
                "DSN (sqlite:///path, postgresql://...). If omitted, auto-discovery "
                "from DATABASE_URL/.env requires LEMONCROW_SQL_AUTODISCOVER=1."
            ),
        },
        "write": {
            "type": "boolean",
            "default": False,
            "description": "Permit INSERT/UPDATE/DELETE/DDL on action=query/lint. Off by default; reads always allowed.",
        },
    },
    "required": ["action"],
    "additionalProperties": False,
}


def _sql_autodiscover_enabled() -> bool:
    """Opt-in gate for discovering a sql connection from DATABASE_URL / .env."""
    return os.environ.get("LEMONCROW_SQL_AUTODISCOVER", "").strip().lower() in {"1", "true", "yes", "on"}


@mcp_tool(
    name="sql",
    input_schema=SQL_TOOL_INPUT_SCHEMA,
    description=(
        "SQL op-dispatch: schema introspection (connect/tables/schema/table/"
        "relationships/search), lint, bounded query execution (single `sql` or "
        "`queries[]` batch). Pass connection (DSN); auto-discovery from "
        "DATABASE_URL/.env requires LEMONCROW_SQL_AUTODISCOVER=1. Live introspection/queries = SQLite; "
        "other dialects → driver-required note."
    ),
    param_aliases={"connection_string": "connection", "allow_writes": "write"},
)
def tool_sql(
    action: str,
    name: str | list[str] | None = None,
    sql: str | None = None,
    queries: list[dict[str, str]] | None = None,
    connection: str | None = None,
    max_rows: int = 500,
    timeout_ms: int = 30_000,
    auto_limit: bool = True,
    write: bool = False,
) -> dict[str, Any]:
    """SQL op-dispatch for connect, lint, and bounded query batching.

    Actions:
      connect       — discover database and show schema overview
      tables        — list table names (+ count)
      schema        — columns + foreign keys per table
      table         — one table's columns + foreign keys (needs name)
      relationships — foreign-key graph as {from: "t.col", to: "rt.col"}
      search        — keyword over table/column names -> matching tables with columns + FKs (needs name)
      lint          — validate SQL syntax without executing (needs sql)
      query         — execute SQL (needs sql or queries[{name,sql},...])

    Pass connection explicitly (DSN). Auto-discovery from the DATABASE_URL env
    var or a .env file is opt-in: set LEMONCROW_SQL_AUTODISCOVER=1.
    Live introspection/queries run on SQLite;
    other dialects report a driver-required note.

    Returns: introspection actions return {tables|table_count|schema|columns|foreign_keys|relationships|matches};
    lint -> {ok, message}; query -> {results: [{name, columns, rows, row_count, truncated}], took_ms}.
    """
    from lemoncrow.pro.capabilities.tool_supervision.sql_tool import sql_tool

    if action not in {
        "connect",
        "tables",
        "schema",
        "table",
        "relationships",
        "search",
        "lint",
        "query",
    }:
        return {
            "isError": True,
            "message": "unsupported action: use connect, tables, schema, table, relationships, search, lint, or query",
        }
    if action == "query" and not sql and not queries:
        return {"isError": True, "message": "action='query' requires sql or queries parameter"}
    if not connection and not _sql_autodiscover_enabled():
        return {
            "isError": True,
            "message": (
                "no connection given and auto-discovery is disabled: pass an explicit "
                "connection (e.g. connection='sqlite:///path/to.db') or set "
                "LEMONCROW_SQL_AUTODISCOVER=1 to allow discovering the connection "
                "from DATABASE_URL / .env"
            ),
        }

    result = sql_tool(
        action=action,
        name=name,
        sql=sql,
        queries=queries,
        connection_string=connection,
        max_rows=max_rows,
        timeout_ms=timeout_ms,
        auto_limit=auto_limit,
        allow_writes=write,
        repo_root=os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd()),
    )
    # Batched queries collapse N would-be individual sql calls into 1.
    if isinstance(result, dict) and isinstance(queries, list) and len(queries) > 1:
        result.setdefault("calls_saved", len(queries) - 1)
    return result


_TASK_BOUNDARY_SUCCESS_RE = re.compile(
    r"\b(done|complete|completed|success|successful|passed|tests?\s+pass(?:ed)?|validated|verified|committed|lgtm)\b",
    re.IGNORECASE,
)
_TASK_BOUNDARY_FAILURE_RE = re.compile(
    r"\b(fail(?:ed|ure)?|error|exception|traceback|blocked|todo|not\s+done|not\s+complete)\b",
    re.IGNORECASE,
)


def _ledger_turn_count(led: RunLedger) -> int:
    turn_events = [
        event
        for event in led.events
        if event.kind in {"agent_message", "reasoning", "test_result", "command_result", "tool_result"}
    ]
    if turn_events:
        return len(turn_events)
    return len(led.events)


def _event_text(event: Any) -> str:
    summary = str(getattr(event, "summary", ""))
    payload = getattr(event, "payload", {})
    return f"{summary}\n{json.dumps(payload, ensure_ascii=False, default=str)}"


def _task_boundary_detected(led: RunLedger) -> bool:
    """Return true only when recent ledger events show a clean stopping point."""
    for event in led.events[-3:]:
        text = _event_text(event)
        if _TASK_BOUNDARY_SUCCESS_RE.search(text) and not _TASK_BOUNDARY_FAILURE_RE.search(text):
            if event.kind == "test_result":
                return bool(event.payload.get("passed"))
            if event.kind == "command_result":
                return bool(event.payload.get("ok"))
            # A SUCCESS keyword in free-text (agent_message / reasoning / note)
            # is the agent *talking* about completion, not a structured outcome,
            # so it must NOT count as a boundary (it would auto-compact mid-task).
    return False


def _context_lifecycle_decision(led: RunLedger) -> dict[str, Any]:
    tokens_used = led.token_count
    utilisation_pct = round(100.0 * tokens_used / CONTEXT_WINDOW_TOKENS, 1)
    turn_count = _ledger_turn_count(led)
    boundary = _task_boundary_detected(led)
    should_handover = utilisation_pct >= HANDOVER_THRESHOLD
    # Bypass the min-turns gate when utilisation is already very high - a small
    # number of dense turns (huge tool outputs, large file reads) can fill the
    # window just as fast as many small ones.
    turns_gate_passed = turn_count > AUTO_COMPACT_MIN_TURNS or utilisation_pct >= AUTO_COMPACT_HIGH_UTIL_OVERRIDE
    should_auto_compact = (
        not should_handover and utilisation_pct >= AUTO_COMPACT_THRESHOLD and turns_gate_passed and boundary
    )
    should_advise = utilisation_pct >= COMPACT_ADVISORY_THRESHOLD

    if should_handover:
        reason = "context utilization reached handover threshold"
    elif should_auto_compact:
        reason = "context utilization reached auto-compact threshold at a task boundary"
    elif utilisation_pct >= AUTO_COMPACT_THRESHOLD and not turns_gate_passed:
        reason = f"auto-compact gated: fewer than {AUTO_COMPACT_MIN_TURNS} turns and below {AUTO_COMPACT_HIGH_UTIL_OVERRIDE}% override"
    elif utilisation_pct >= AUTO_COMPACT_THRESHOLD and not boundary:
        reason = "auto-compact waiting for a clean task boundary"
    elif should_advise:
        reason = "advisory threshold reached; no automatic action"
    else:
        reason = "below advisory threshold"

    return {
        "tokens_used": tokens_used,
        "context_window": CONTEXT_WINDOW_TOKENS,
        "utilisation_pct": utilisation_pct,
        "turn_count": turn_count,
        "task_boundary_detected": boundary,
        "should_advise": should_advise,
        "should_auto_compact": should_auto_compact,
        "should_compact": should_auto_compact,
        "should_handover": should_handover,
        "reason": reason,
        "thresholds": {
            "advisory_pct": COMPACT_ADVISORY_THRESHOLD,
            "auto_compact_pct": AUTO_COMPACT_THRESHOLD,
            "handover_pct": HANDOVER_THRESHOLD,
            "auto_compact_min_turns": AUTO_COMPACT_MIN_TURNS,
        },
    }


def _write_handover_packet(led: RunLedger, state: Any) -> Path:
    from lemoncrow.core.foundation.paths import session_dir
    from lemoncrow.pro.runtime.context_compressor import HandoverPacket

    root = _lemoncrow_root()
    run_dir = session_dir(root, led.agent or _detect_agent(), led.session_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    handover_path = run_dir / "HANDOVER.md"
    packet = HandoverPacket.from_ledger(led, state, workspace_root=_workspace_root())
    handover_path.write_text(packet.to_markdown(), encoding="utf-8")
    return handover_path


_COMPACT_ADVISE_CACHE: dict[str, tuple[int, Any]] = {}
_MAX_COMPACT_ADVISE_CACHE = 64
# Serializes cache read/insert/evict under the dispatcher thread pool; the
# expensive compress() call runs outside the lock.
_COMPACT_ADVISE_LOCK = threading.Lock()


def _compact_advise(session_id: str | None = None) -> dict[str, Any]:
    """Advise when to compact and what context to preserve.

    Returns a manifest with:
    - should_advise: bool (true if utilisation >= 60%)
    - should_compact: bool (true if utilisation >= 80%, after min-turn and boundary gates)
    - should_handover: bool (true if utilisation >= 95%)
    """
    try:
        from lemoncrow.pro.runtime.context_compressor import ContextCompressor

        led = _get_ledger()
        if session_id:
            led.session_id = session_id

        lifecycle = _context_lifecycle_decision(led)
        utilisation_pct = float(lifecycle["utilisation_pct"])
        should_compact = bool(lifecycle["should_compact"])
        should_handover = bool(lifecycle["should_handover"])
        # Memoize the full-ledger compression by (session, event count): this
        # advisory entrypoint may be polled repeatedly without the ledger
        # changing, and compress() walks every event each call. compress() is
        # pure (reads events, returns a fresh CompactState), so reuse is safe.
        _ca_key = led.session_id or ""
        with _COMPACT_ADVISE_LOCK:
            _ca_cached = _COMPACT_ADVISE_CACHE.get(_ca_key)
        if _ca_cached is not None and _ca_cached[0] == len(led.events):
            state = _ca_cached[1]
        else:
            state = ContextCompressor().compress(led, preserve_last_n_turns=10, workspace_root=_workspace_root())
            with _COMPACT_ADVISE_LOCK:
                _COMPACT_ADVISE_CACHE[_ca_key] = (len(led.events), state)
                if len(_COMPACT_ADVISE_CACHE) > _MAX_COMPACT_ADVISE_CACHE:
                    # Bound the cache: a marathon process seeing many session ids
                    # must not leak compressed states. Evict oldest (skip current).
                    for _stale in list(_COMPACT_ADVISE_CACHE)[: len(_COMPACT_ADVISE_CACHE) - _MAX_COMPACT_ADVISE_CACHE]:
                        if _stale != _ca_key:
                            _COMPACT_ADVISE_CACHE.pop(_stale, None)
        compaction_savings = _session_compaction_savings_payload(
            led,
            state,
            tokens_before=int(lifecycle["tokens_used"]),
            trigger="compact_advise",
            reason=str(lifecycle["reason"]),
            utilisation_pct=utilisation_pct,
        )

        # Collect preserve_playbooks: top active Playbooks from ledger
        preserve_playbooks = list(set(led.active_playbooks))[:3]

        # Collect pin_memory: pinned MemoryBlocks for this run's agent
        pin_memory: list[str] = []
        try:
            store = _memory_store()
            agent_id = led.agent or "claude"
            pinned = store.list_pinned_blocks(agent_id=agent_id)
            pin_memory = [b.id for b in pinned][:5]
        except Exception:
            logging.exception("Recovered from broad exception handler")
            logger.warning(
                "Suppressed exception in _compact_advise fetching pinned memory",
                exc_info=True,
            )

        # Collect open_files: last 5 files touched
        open_files = led.files_touched[-5:] if led.files_touched else []
        handover_file: str | None = None
        if should_handover:
            handover_file = str(_write_handover_packet(led, state))

        # Build suggested prompt
        if should_handover:
            suggested_prompt = (
                f"Session is at {utilisation_pct}% context utilisation. Read {handover_file} and continue "
                "from a fresh agent context using the host-native agent/subagent mechanism."
            )
        else:
            suggested_prompt = (
                f"Compact this conversation. Context utilisation: {utilisation_pct}%. "
                f"Please preserve these Playbooks: {', '.join(preserve_playbooks) or '(none yet)'}. "
                f"Recently edited files: {', '.join(open_files) or '(none)'}. "
                "Preserve the last 10 raw turns, active errors, and current CLAUDE.md hash."
            )

        # Persist manifest to disk
        try:
            from lemoncrow.core.foundation.paths import session_dir

            root = _lemoncrow_root()
            run_dir = session_dir(root, led.agent or _detect_agent(), led.session_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = run_dir / "compact_manifest.json"
            manifest = {
                "created_at": datetime.now(UTC).isoformat(),
                "session_id": led.session_id,
                "should_compact": should_compact,
                "should_advise": bool(lifecycle["should_advise"]),
                "should_auto_compact": bool(lifecycle["should_auto_compact"]),
                "should_handover": should_handover,
                "utilisation_pct": utilisation_pct,
                "turn_count": int(lifecycle["turn_count"]),
                "task_boundary_detected": bool(lifecycle["task_boundary_detected"]),
                "reason": str(lifecycle["reason"]),
                "thresholds": lifecycle["thresholds"],
                "preserve_playbooks": preserve_playbooks,
                "pin_memory": pin_memory,
                "open_files": open_files,
                "recent_turns": state.recent_turns,
                "claude_md_hash": state.claude_md_hash,
                "active_errors": state.error_fingerprints,
                "handover_file": handover_file,
                "suggested_prompt": suggested_prompt,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            logging.exception("Recovered from broad exception handler")
            logger.warning(
                "Suppressed exception in _compact_advise persisting manifest",
                exc_info=True,
            )

        if should_compact and int(compaction_savings["tokens_saved"]) > 0:
            _append_live_savings_event(compaction_savings)

        return {
            "should_compact": should_compact,
            "should_advise": bool(lifecycle["should_advise"]),
            "should_auto_compact": bool(lifecycle["should_auto_compact"]),
            "should_handover": should_handover,
            "utilisation_pct": utilisation_pct,
            "turn_count": int(lifecycle["turn_count"]),
            "task_boundary_detected": bool(lifecycle["task_boundary_detected"]),
            "reason": str(lifecycle["reason"]),
            "thresholds": lifecycle["thresholds"],
            "preserve_playbooks": preserve_playbooks,
            "pin_memory": pin_memory,
            "open_files": open_files,
            "recent_turns": state.recent_turns,
            "claude_md_hash": state.claude_md_hash,
            "active_errors": state.error_fingerprints,
            "handover_file": handover_file,
            "suggested_prompt": suggested_prompt,
            "tokens_before": int(compaction_savings["tokens_before"]),
            "tokens_after_estimate": int(compaction_savings["tokens_after_estimate"]),
            "tokens_freed": int(compaction_savings["tokens_freed"]),
            "cost_saved_usd": float(compaction_savings["cost_saved_usd"]),
        }
    except Exception:
        logging.exception("Recovered from broad exception handler")
        # Fail-open: return conservative defaults
        return {
            "should_compact": False,
            "should_advise": False,
            "should_auto_compact": False,
            "should_handover": False,
            "utilisation_pct": 0.0,
            "turn_count": 0,
            "task_boundary_detected": False,
            "reason": "Unable to compute compaction advice; proceed conservatively.",
            "thresholds": {
                "advisory_pct": COMPACT_ADVISORY_THRESHOLD,
                "auto_compact_pct": AUTO_COMPACT_THRESHOLD,
                "handover_pct": HANDOVER_THRESHOLD,
                "auto_compact_min_turns": AUTO_COMPACT_MIN_TURNS,
            },
            "preserve_playbooks": [],
            "pin_memory": [],
            "open_files": [],
            "recent_turns": [],
            "claude_md_hash": None,
            "active_errors": [],
            "handover_file": None,
            "suggested_prompt": "Unable to compute compaction advice; proceed with default compaction.",
        }


def _memory_summary(session_id: str) -> dict[str, Any]:
    """Run the sleeptime summarizer for a given run and return a summary.

    Input:
        session_id: The run identifier to summarize.

    Output:
        tokens_pre, tokens_post, summary_md, evicted_event_ids, strategy
    """
    try:
        from lemoncrow.pro.capabilities.context_compression.capability import (
            ContextCompressionCapability,
        )

        led = _get_ledger()
        if session_id:
            led.session_id = session_id

        cap = ContextCompressionCapability()
        result = cap.compress_with_sleeptime(led)

        summary_lines = [f"## Sleeptime Summary - run `{led.session_id}`", ""]
        summary_lines.append(f"- Tokens before: {result.chars_before // 4}")
        summary_lines.append(f"- Tokens after:  {result.chars_after // 4}")
        summary_lines.append(f"- Reduction:     {result.reduction_pct}%")
        if result.dropped:
            summary_lines.append("")
            summary_lines.append("### Evicted events")
            for d in result.dropped[:10]:
                summary_lines.append(f"- [{d.kind}] {d.summary[:100]}")

        return {
            "tokens_pre": result.chars_before // 4,
            "tokens_post": result.chars_after // 4,
            "summary_md": "\n".join(summary_lines),
            "evicted_event_ids": [d.kind for d in result.dropped],
            "strategy": "tfidf",
        }
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        return {"error": str(exc)}


# Thread-local used to pass the active engine into _maybe_attach_code_rendered
# for cold-start bootstrap-note injection without touching every return branch.
_code_engine_for_current_call: threading.local = threading.local()

# Process-level engine cache keyed by resolved repo path.
# Reusing the same engine across tool calls avoids re-opening the SQLite DB
# and restarting autosync threads on every invocation — critical for both
# MCP server performance (persistent process) and benchmark correctness.
_code_engine_cache: dict[str, Any] = {}
_code_engine_cache_lock: threading.Lock = threading.Lock()
_scoped_context_cache: dict[str, Any] = {}
_scoped_context_cache_lock: threading.Lock = threading.Lock()


def _code_context_engine(repo_root: str = ".") -> Any:
    from lemoncrow.pro.capabilities.code_context import CodeContextEngine

    workspace = str(_workspace_root())
    root = Path(repo_root)
    resolved = (root if root.is_absolute() else Path(workspace) / root).resolve()
    cache_key = str(resolved)
    engine = _code_engine_cache.get(cache_key)
    if engine is None:
        with _code_engine_cache_lock:
            engine = _code_engine_cache.get(cache_key)  # re-check under lock
            if engine is None:
                engine = CodeContextEngine(resolved)
                _code_engine_cache[cache_key] = engine
    return engine


def _scoped_context_capability(repo_root: str = ".") -> Any:
    from lemoncrow.pro.capabilities.scoped_context import ScopedContextCapability

    workspace = str(_workspace_root())
    root = Path(repo_root)
    resolved = (root if root.is_absolute() else Path(workspace) / root).resolve()
    cache_key = str(resolved)
    capability = _scoped_context_cache.get(cache_key)
    if capability is None:
        with _scoped_context_cache_lock:
            capability = _scoped_context_cache.get(cache_key)
            if capability is None:
                capability = ScopedContextCapability(_code_context_engine(str(resolved)))
                _scoped_context_cache[cache_key] = capability
    return capability


def _workspace_code_router(repo_root: str = ".") -> Any:
    from lemoncrow.pro.capabilities.code_context.workspace_router import WorkspaceCodeRouter

    workspace = str(_workspace_root())
    root = Path(repo_root)
    resolved = root if root.is_absolute() else Path(workspace) / root
    return WorkspaceCodeRouter(
        repo_root=resolved,
        engine_factory=lambda target_root: _code_context_engine(str(target_root)),
    )


# Fields that are purely internal LemonCrow bookkeeping — never useful to an LLM.
# Keep: repo_name (multi-repo).
_CODE_OP_TOP_STRIP: frozenset[str] = frozenset(
    {
        "symbol_id",
        "cache_hit",
        "rendered_format",
        "repo_id",
        "total_tokens",
        "tokens_saved",
        "provenance",
        "provenance_breakdown",
        "mode",
        "view",
        "has_more_context",
        "suggested_next",
        "explanation",
        "text_search",
    }
)

# Fields to strip from nested item dicts (search results, callers/related lists, etc.).
# Keep: origin (external/internal scope), repo_name (multi-repo workspace).
_CODE_OP_ITEM_STRIP: frozenset[str] = frozenset(
    {
        "symbol_id",
        "start_byte",
        "end_byte",
        "content_hash",
        "repo_id",
        "score",
        "provenance",
    }
)

# Extra top-level keys to drop per-op (in addition to _CODE_OP_TOP_STRIP).
_CODE_OP_EXTRA_STRIP: dict[str, frozenset[str]] = {
    # edges contain only hash IDs — no names or paths; `related` has the useful data
    "callers": frozenset({"edges"}),
    "callees": frozenset({"edges"}),
    # symbol/node ops: byte offsets and hashes are useless to LLMs (same payload
    # shape — both come from engine.tool_symbol). node additionally renders the
    # source body; symbol stays a compact location/signature summary.
    "symbol": frozenset({"start_byte", "end_byte", "content_hash", "score"}),
    "node": frozenset({"start_byte", "end_byte", "content_hash", "score"}),
    # search: `snippet` at top level is just the mode string ("none"/"head"/"full"), not actual code
    "search": frozenset({"snippet"}),
    # context: `symbols` duplicates entry_points with heavy metadata; telemetry/import_neighbors are internal
    "context": frozenset({"telemetry", "import_neighbors", "symbols"}),
    # status: db_path exposes internal filesystem paths
    "status": frozenset({"db_path"}),
}

# List-valued fields whose items should be stripped of internal keys.
_CODE_OP_ITEM_LIST_FIELDS: tuple[str, ...] = (
    "items",
    "related",
    "related_symbols",
    "entry_points",
    "references",
    "symbols",
)


def _strip_code_item(item: dict[str, Any]) -> dict[str, Any]:
    """Strip internal bookkeeping from a single result item."""
    cleaned = {k: v for k, v in item.items() if k not in _CODE_OP_ITEM_STRIP}
    if cleaned.get("origin") == "internal":
        del cleaned["origin"]
    if cleaned.get("qualified_name") and cleaned["qualified_name"] == (
        cleaned.get("name") or cleaned.get("symbol_name")
    ):
        del cleaned["qualified_name"]
    cleaned.pop("role", None)
    return cleaned


def _strip_code_op_response(op: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Remove internal/telemetry fields that waste LLM context."""
    drop = _CODE_OP_TOP_STRIP | _CODE_OP_EXTRA_STRIP.get(op, frozenset())
    result: dict[str, Any] = {k: v for k, v in payload.items() if k not in drop}

    # Save real tokens_saved via thread-local so _record_context_budget_for_tool
    # can read it without polluting the LLM-facing response.
    ts = int(payload.get("tokens_saved", 0) or 0)
    if ts > 0:
        _tool_call_tokens_saved.value = ts

    # Strip internal keys from the target object
    if isinstance(result.get("target"), dict):
        result["target"] = _strip_code_item(result["target"])

    # Strip internal keys from list fields (or dicts of lists, e.g. references grouped by file)
    for field in _CODE_OP_ITEM_LIST_FIELDS:
        value = result.get(field)
        if isinstance(value, list):
            result[field] = [_strip_code_item(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            result[field] = {
                key: (
                    [_strip_code_item(item) if isinstance(item, dict) else item for item in group]
                    if isinstance(group, list)
                    else group
                )
                for key, group in value.items()
            }

    return result


def _maybe_attach_code_rendered(op: str, payload: dict[str, Any], *, render_compact: bool) -> dict[str, Any]:
    # Render first so the markdown uses all original fields (e.g. repo_id for cache_status heading).
    from lemoncrow.pro.capabilities.code_context.renderer import render_code_payload

    rendered = render_code_payload(op, payload)

    # Store in thread-local so _handle can use MD text as the MCP response body.
    # This is the single model-facing channel for the rendered markdown: for
    # code-intel tools render_tool_result_text() returns this value as the MCP
    # response body. Do NOT also stash it in result["rendered"] — that in-JSON
    # copy would ship the same markdown twice to the model.
    _tool_call_rendered_text.value = rendered
    _ = render_compact  # rendered text now travels only via the response body

    # Strip internal fields after rendering — LLMs get clean JSON without duplicating
    # internal bookkeeping that only LemonCrow needs.
    result = _strip_code_op_response(op, payload)

    # Inject cold-start bootstrap note so the LLM knows results may be incomplete.
    if op not in {"index", "status", "cache_status"}:
        engine = getattr(_code_engine_for_current_call, "value", None)
        if engine is not None and not Path(engine.db_path).exists():
            result["bootstrap_note"] = (
                "Repository not yet indexed — results may be incomplete. "
                "Run `lc code index` (or `lc project init`) to bootstrap the index."
            )

    return result


def _code_search_target_item(item: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": item.get("kind"),
        "name": item.get("name") or item.get("symbol_name"),
        "qualified_name": item.get("qualified_name"),
        "path": item.get("path") or item.get("file_path"),
        "repo_name": item.get("repo_name"),
        "origin": item.get("origin"),
        "line": item.get("line") or item.get("start_line"),
        "end_line": item.get("end_line"),
        "signature": item.get("signature"),
        "snippet": item.get("snippet"),
        "deleted_at": item.get("deleted_at"),
        "deleted_at_sha": item.get("deleted_at_sha"),
        "rename_target": item.get("rename_target"),
        "rename_note": item.get("rename_note"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _code_search_target_view(payload: dict[str, Any]) -> dict[str, Any]:
    items = payload.get("items")
    if isinstance(items, list):
        # Copy before mutating so a cached/shared engine payload is never
        # rewritten in place under concurrent callers.
        payload = dict(payload)
        payload["items"] = [_code_search_target_item(item) if isinstance(item, dict) else item for item in items]
    return payload


def _flatten_code_references(references: Any) -> list[dict[str, Any]]:
    if isinstance(references, list):
        return [dict(item) for item in references if isinstance(item, dict)]
    if isinstance(references, dict):
        flattened: list[dict[str, Any]] = []
        for values in references.values():
            if isinstance(values, list):
                flattened.extend(dict(item) for item in values if isinstance(item, dict))
        return flattened
    return []


def _code_search_graph_view(
    engine: Any,
    *,
    query: str,
    search_payload: dict[str, Any],
    view: Literal["graph", "explain"],
    limit: int,
    depth: int,
    budget_tokens: int,
) -> dict[str, Any]:
    items = search_payload.get("items")
    primary = next((item for item in items if isinstance(item, dict)), None) if isinstance(items, list) else None
    if primary is None:
        return {
            "target": None,
            "related": {"imports": [], "usages": [], "callers": [], "callees": []},
        }

    target = _code_search_target_item(primary)
    symbol_args = {
        "query": query,
        "symbol_id": primary.get("symbol_id") or primary.get("id"),
        "qualified_name": primary.get("qualified_name"),
        "symbol_name": primary.get("symbol_name") or primary.get("name"),
        "file_path": primary.get("file_path") or primary.get("path"),
    }
    relation_budget = max(300, budget_tokens // 3)
    usages = engine.tool_usages(
        query=symbol_args["query"],
        symbol_id=symbol_args["symbol_id"],
        qualified_name=symbol_args["qualified_name"],
        symbol_name=symbol_args["symbol_name"],
        file_path=symbol_args["file_path"],
        group_by="none",
        snippet_lines=0,
        limit=limit,
        budget_tokens=relation_budget,
        auto_index=False,
    )
    callers = engine.tool_callers(
        query=symbol_args["query"],
        symbol_id=symbol_args["symbol_id"],
        qualified_name=symbol_args["qualified_name"],
        symbol_name=symbol_args["symbol_name"],
        file_path=symbol_args["file_path"],
        depth=depth,
        limit=limit,
        budget_tokens=relation_budget,
        auto_index=False,
    )
    callees = engine.tool_callees(
        query=symbol_args["query"],
        symbol_id=symbol_args["symbol_id"],
        qualified_name=symbol_args["qualified_name"],
        symbol_name=symbol_args["symbol_name"],
        file_path=symbol_args["file_path"],
        depth=depth,
        limit=limit,
        budget_tokens=relation_budget,
        auto_index=False,
    )
    refs = _flatten_code_references(usages.get("references"))
    imports = [ref for ref in refs if "import" in str(ref.get("edge_kind") or "")]
    import_ids = {id(ref) for ref in imports}
    usage_refs = [ref for ref in refs if id(ref) not in import_ids]
    payload: dict[str, Any] = {
        "target": target,
        "related": {
            "imports": imports,
            "usages": usage_refs,
            "callers": callers.get("related", []),
            "callees": callees.get("related", []),
        },
    }
    if view == "explain":
        payload["items"] = [
            _code_search_target_item(item) if isinstance(item, dict) else item
            for item in cast(list[Any], search_payload.get("items", []))
        ]
    return payload


# Result keys that represent batched discoveries — each item would have
# required its own naive grep/read in a side-by-side baseline.
_CODE_BATCH_KEYS: tuple[str, ...] = (
    "matches",
    "callers",
    "callees",
    "usages",
    "related",
    "edges",
    "references",
    "results",
    "items",
    "files",
    "symbols",
    "routes",
)


# Vanilla Read dumps up to 2000 lines per file (~40 chars/line of typical
# source). Counterfactual reads of surfaced files are capped here per file so
# a giant file can never inflate the credit past what vanilla would inline.
_VANILLA_READ_FILE_CAP_CHARS = 80_000


def _finish_code_result(result: dict[str, Any]) -> dict[str, Any]:
    # Infer calls_saved. One code-intel call replaces a locating grep plus a
    # read per DISTINCT file surfaced -- not one call per returned item (20
    # symbols across 3 files is ~3 avoided reads, not 19 avoided calls).
    # Credit distinct files: (1 grep + N reads) - the 1 call made = N; that
    # holds at N=1 too (grep + read - this call = 1), and single-file results
    # are the common code_search shape, so they must not fall through
    # uncredited. Items without file info credit only the locate scan.
    surfaced_paths: set[str] = set()
    if isinstance(result, dict):
        for key in _CODE_BATCH_KEYS:
            items = result.get(key)
            if isinstance(items, list) and items:
                # Graveyard hits (deleted_at) describe files that no longer
                # exist — vanilla had no grep+read to avoid, so they earn no
                # call or token credit and keep the deleted surface additive.
                live_items = [item for item in items if isinstance(item, dict) and not item.get("deleted_at")]
                if not live_items and any(isinstance(item, dict) for item in items):
                    break
                paths = {str(item.get("path") or item.get("file") or "") for item in live_items}
                paths.discard("")
                surfaced_paths = paths
                if "calls_saved" not in result:
                    result["calls_saved"] = len(paths) if paths else 1
                break
    # Counterfactual token credit: vanilla explores by grepping and then
    # Reading each surfaced file; what it would have inlined (capped per file)
    # minus what this one call actually returned is context we kept out.
    # The engine may already stamp a packing credit (projected vs full symbol
    # source) -- both measure the same avoidance against different baselines,
    # so take the LARGER of the two, never the sum.
    if surfaced_paths and isinstance(result, dict):
        per_file_chars: dict[str, int] = {}
        for p in surfaced_paths:
            try:
                per_file_chars[p] = min(os.stat(p).st_size, _VANILLA_READ_FILE_CAP_CHARS)
            except OSError:
                continue  # non-file / unreadable path: no counterfactual read
        vanilla_chars = sum(per_file_chars.values())
        if vanilla_chars > 0:
            try:
                returned_chars = len(json.dumps(result, default=str))
            except (TypeError, ValueError):
                returned_chars = 0
            counterfactual = max(0, vanilla_chars - returned_chars) // 4
            existing = max(
                _coerce_saved_tokens(result.get("tokens_saved")),
                int(getattr(_tool_call_tokens_saved, "value", 0) or 0),
            )
            if counterfactual > existing:
                result["tokens_saved"] = counterfactual
                # Hand the per-file breakdown to _process_tool_accounting so
                # the credit is netted against the shared per-session
                # credited-file set (one content-avoidance credit per file).
                _tool_call_counterfactual.value = {
                    "per_file_chars": per_file_chars,
                    "returned_chars": returned_chars,
                    "floor_tokens": existing,
                }
    engine = getattr(_code_engine_for_current_call, "value", None)
    if engine is not None and isinstance(result, dict) and "index_status" not in result:
        try:
            if not engine.index_ready():
                result["index_status"] = "warming"
                result.setdefault(
                    "hint",
                    "code index is still building in the background; retry shortly for complete results",
                )
        except Exception:
            logging.exception("Recovered from broad exception handler")
    return result


def _code_engine_at(repo_root: str | None) -> Any:
    engine = _code_context_engine(repo_root or ".")
    _code_engine_for_current_call.value = engine
    return engine


_GRAPH_KINDS: frozenset[str] = frozenset(
    {
        "blast_radius",
        "dead_code",
        "cycles",
        "coupling",
        "centrality",
        # WS10 code health & history (G15/G16/N17): additive, read-only, fail-open.
        "design_gaps",
        "verify_design",
        "pr_risk",
        "commit_provenance",
        "index_docs",
        "recall_docs",
        # WS11 (G17): module-boundary / god-module topology discovery.
        "topology",
    }
)


def _synthesize_edges_for_paths(paths: list[str]) -> list[dict[str, Any]]:
    """Run the bounded N2/N3 edge synthesizer over *paths*; clearly-labelled output.

    Returns a flat list of heuristic edge dicts (provenance="heuristic"). Never
    touches the static call graph -- this is a separate, opt-in addendum.
    """
    from lemoncrow.infra.code_intel.languages import language_for_path
    from lemoncrow.pro.capabilities.code_context.edge_synthesis import synthesize_edges

    out: list[dict[str, Any]] = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_file():
            continue
        lang = language_for_path(candidate)
        language = lang.name if lang is not None else ""
        source = candidate.read_text(encoding="utf-8", errors="replace")
        for edge in synthesize_edges(source, language=language):
            out.append({"file": str(candidate), **edge.to_dict()})
    return out


def _op_graph(
    *,
    kind: str = "blast_radius",
    path: str | None = None,
    paths: list[str] | None = None,
    limit: int = 50,
    synthesize: bool = False,
    query: str | None = None,
    enable: bool | None = None,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    """Agent-facing graph analytics (G3/G6) dispatched by ``kind``.

    * ``blast_radius`` (default) -- reverse-dependency closure + affected tests +
      risk tier for ``path`` (file-level; uses the existing change_impact).
    * ``dead_code`` / ``cycles`` / ``coupling`` -- repo-wide file-graph analytics
      over the semantic file index. Pass ``paths`` to fold those files into the
      index first; otherwise analyses whatever the index already holds.
    * ``centrality`` -- symbol-level call-graph centrality from the code-intel engine.
      Pass ``synthesize=true`` with ``paths`` to additionally return
      heuristic (route/event) edges as a SEPARATE ``synthesized_edges`` list
      (N2/N3); they are never merged into the static call graph.
    """
    if kind not in _GRAPH_KINDS:
        raise ValueError(f"unknown graph kind: {kind!r}; expected one of {sorted(_GRAPH_KINDS)}")

    # WS10 code health & history kinds (G15/G16/N17). Each is additive, read-only
    # analytics that fails open inside its own module; dispatched before the
    # file-graph kinds because they have distinct argument shapes.
    if kind in {"design_gaps", "verify_design", "pr_risk", "commit_provenance", "index_docs", "recall_docs"}:
        return _op_graph_code_health(
            kind=kind, path=path, paths=paths, limit=limit, query=query, enable=enable, repo_root=repo_root
        )

    if kind == "centrality":
        engine = _code_engine_at(repo_root)
        result = cast(dict[str, Any], engine.call_graph_centrality(limit=limit))
        result["kind"] = "centrality"
        if synthesize and paths:
            result["synthesized_edges"] = _synthesize_edges_for_paths(paths)
        return _finish_code_result(result)

    cap = _semantic_file_memory(_lemoncrow_root())
    if paths:
        for raw in paths:
            candidate = Path(raw)
            if candidate.is_file():
                cap.summarize_file(candidate)
    analytics = cap.graph_analytics()
    if kind == "blast_radius":
        if not path:
            raise ValueError("path is required for kind='blast_radius'")
        result = analytics.blast_radius(str(Path(path)))
    elif kind == "dead_code":
        result = analytics.dead_code(limit=limit)
    elif kind == "cycles":
        result = analytics.cycles(limit=limit)
    elif kind == "topology":
        result = analytics.topology(limit=limit)
    else:  # coupling
        result = analytics.coupling(limit=limit)
    result["kind"] = kind
    _ = render_compact  # file-graph analytics render as JSON; no markdown view
    return result


def _op_graph_code_health(
    *,
    kind: str,
    path: str | None,
    paths: list[str] | None,
    limit: int,
    query: str | None,
    enable: bool | None,
    repo_root: str | None,
) -> dict[str, Any]:
    """Dispatch the WS10 code health & history graph kinds (G15/G16/N17).

    Each delegated function is independently fail-open; this seam only resolves
    the repo/lemoncrow roots and routes by ``kind``.
    """
    from lemoncrow.pro.capabilities.code_health import (
        commit_provenance,
        design_gaps,
        index_design_docs,
        pr_risk,
        recall_design_docs,
        verify_design,
    )

    workspace = _workspace_root()
    repo = (Path(repo_root) if repo_root else workspace).resolve()
    lemoncrow_root = _lemoncrow_root()

    if kind == "design_gaps":
        return design_gaps(repo_root=repo, lemoncrow_root=lemoncrow_root, paths=paths)
    if kind == "verify_design":
        return verify_design(repo_root=repo, lemoncrow_root=lemoncrow_root, paths=paths)
    if kind == "pr_risk":
        targets = paths or ([path] if path else [])
        if not targets:
            raise ValueError("pr_risk requires 'paths' (or 'path') -- the changed files")
        return pr_risk(repo_root=repo, lemoncrow_root=lemoncrow_root, paths=targets)
    if kind == "commit_provenance":
        return commit_provenance(repo_root=repo, path=path, limit=limit)
    if kind == "index_docs":
        return index_design_docs(repo_root=repo, lemoncrow_root=lemoncrow_root, paths=paths, enable=enable)
    # recall_docs
    if not query:
        raise ValueError("recall_docs requires 'query'")
    return recall_design_docs(lemoncrow_root=lemoncrow_root, query=query, limit=limit)


def _op_callers(
    *,
    query: str | None = None,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    kind: str | None = None,
    language: str | None = None,
    depth: int = 1,
    limit: int = 20,
    snapshot: bool = False,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not any([query, symbol_id, qualified_name, symbol_name]):
        raise ValueError("query, symbol_id, qualified_name, or symbol_name is required for code callers")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_callers(
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=path,
            kind=kind,
            language=language,
            depth=depth,
            limit=limit,
            snapshot=snapshot,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("callers", payload, render_compact=render_compact))


def _op_callees(
    *,
    query: str | None = None,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    kind: str | None = None,
    language: str | None = None,
    depth: int = 1,
    limit: int = 20,
    snapshot: bool = False,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not any([query, symbol_id, qualified_name, symbol_name]):
        raise ValueError("query, symbol_id, qualified_name, or symbol_name is required for code callees")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_callees(
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=path,
            kind=kind,
            language=language,
            depth=depth,
            limit=limit,
            snapshot=snapshot,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("callees", payload, render_compact=render_compact))


def _op_usages(
    *,
    query: str | None = None,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    kind: str | None = None,
    language: str | None = None,
    file_glob: str | None = None,
    group_by: str = "file",
    snippet_lines: int = 8,
    limit: int = 20,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not any([query, symbol_id, qualified_name, symbol_name]):
        raise ValueError("query, symbol_id, qualified_name, or symbol_name is required for code usages")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_usages(
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=path,
            kind=kind,
            language=language,
            file_glob=file_glob,
            group_by=group_by,
            snippet_lines=3 if snippet_lines == 8 else snippet_lines,
            limit=limit,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("usages", payload, render_compact=render_compact))


def _op_explore(
    *,
    query: str | None = None,
    seed_files: list[str] | None = None,
    max_files: int = 8,
    max_symbols: int = 4,
    include_source: bool = True,
    include_relationships: bool = True,
    line_numbers: bool = True,
    skeletonize: bool = True,
    complete_families: bool | None = None,
    depth: int = 1,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not query:
        raise ValueError("query is required for code explore")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_explore(
            query=query,
            seed_files=seed_files,
            max_files=max_files,
            max_symbols=max_symbols,
            include_source=include_source,
            include_relationships=include_relationships,
            line_numbers=line_numbers,
            skeletonize=skeletonize,
            complete_families=complete_families,
            depth=depth,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("explore", payload, render_compact=render_compact))


def _op_pattern(
    *,
    pattern: str | None = None,
    rewrite: str | None = None,
    language: str | None = None,
    file_glob: str | None = None,
    dry_run: bool = True,
    limit: int = 20,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not pattern:
        raise ValueError("pattern is required for code pattern")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_pattern(
            pattern=pattern,
            rewrite=rewrite,
            language=language,
            file_glob=file_glob,
            dry_run=dry_run,
            limit=limit,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("pattern", payload, render_compact=render_compact))


def _op_search(
    *,
    query: str | None = None,
    mode: str = "auto",
    intent: str = "auto",
    view: Literal["target", "graph", "context", "explain"] = "target",
    kind: str | None = None,
    language: str | None = None,
    snippet: str = "none",
    snippet_lines: int = 8,
    file_glob: str | None = None,
    scope: str = "repo",
    since: str | None = None,
    touched_by: str | None = None,
    provenance: str | None = None,
    seed_files: list[str] | None = None,
    max_symbols: int = 4,
    depth: int = 1,
    limit: int = 20,
    budget_tokens: int = 4000,
    repo: str | None = None,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not query:
        raise ValueError("query is required for code search")
    engine_root = repo_root or "."
    workspace_router = _workspace_code_router(engine_root)
    if repo is not None and not workspace_router.is_configured:
        raise ValueError("repo filter requires .lemoncrow/workspace.toml")
    engine = _code_context_engine(engine_root)
    _code_engine_for_current_call.value = engine
    if workspace_router.is_configured and view in {"graph", "explain", "context"}:
        # The workspace-routed search path only produces a target-shaped payload;
        # graph/explain/context are not routed. Reject explicitly so callers get a
        # clear error instead of a silently-wrong (unrouted, repo-ignoring) result.
        raise ValueError(
            f"view={view!r} is not supported when a workspace is configured (.lemoncrow/workspace.toml); "
            "use view='target' for routed search"
        )
    if view == "context":
        context_payload = engine.tool_context(
            task=query,
            seed_files=seed_files,
            budget_tokens=budget_tokens,
            max_symbols=max_symbols,
        )
        return _finish_code_result(
            _maybe_attach_code_rendered("context", cast(dict[str, Any], context_payload), render_compact=render_compact)
        )
    search_kwargs: dict[str, Any] = {
        "limit": limit,
        "mode": mode,
        "kind": kind,
        "language": language,
        "snippet": snippet,
        "snippet_lines": snippet_lines,
        "file_glob": file_glob,
        "scope": scope,
        "budget_tokens": budget_tokens,
    }
    if scope != "deleted":
        search_kwargs["intent"] = intent
        search_kwargs["seed_files"] = seed_files
    if since is not None:
        search_kwargs["since"] = since
    if touched_by is not None:
        search_kwargs["touched_by"] = touched_by
    if provenance is not None:
        search_kwargs["provenance_filter"] = provenance
    if workspace_router.is_configured:
        routed_payload = cast(
            dict[str, Any],
            workspace_router.route("search", repo=repo, query=query, **search_kwargs),
        )
        routed_hits = _count_search_hits(routed_payload)
        routed_payload = _code_search_target_view(routed_payload)
        routed_result = _finish_code_result(
            _maybe_attach_code_rendered("search", routed_payload, render_compact=render_compact)
        )
        if scope != "repo":
            return routed_result
        return _apply_search_verdict(
            routed_result,
            query=query,
            hit_count=routed_hits,
            channels=engine.search_channel_health(query, mode),
        )
    search_payload = cast(dict[str, Any], engine.tool_search(query, **search_kwargs))
    resolved_mode = str(search_payload.get("mode") or mode)
    primary_hits = _count_search_hits(search_payload)
    text_fallback: list[dict[str, Any]] = []
    if scope == "repo" and primary_hits == 0 and _search_cascade_enabled():
        # Phase 3 -- reactive-serial cascade: ranked search came up empty, so fall
        # through to a literal text scan within the SAME call (catches comments,
        # config, and strings the symbol index does not hold) instead of making
        # the model spend a turn on grep.
        with contextlib.suppress(Exception):
            text_fallback = [
                {"path": match.file_path, "line": match.line} for match in engine.search_text(query, limit=8)
            ]
    if view == "target":
        search_payload = _code_search_target_view(search_payload)
    elif view in {"graph", "explain"}:
        search_payload = _code_search_graph_view(
            engine,
            query=query,
            search_payload=search_payload,
            view=view,
            limit=limit,
            depth=depth,
            budget_tokens=budget_tokens,
        )
    search_result = _finish_code_result(
        _maybe_attach_code_rendered("search", search_payload, render_compact=render_compact)
    )
    if scope != "repo":
        # deleted/external surfaces stay strictly additive -- no verdict stamping.
        return search_result
    if text_fallback:
        search_result["text_fallback"] = text_fallback
    return _apply_search_verdict(
        search_result,
        query=query,
        hit_count=primary_hits or len(text_fallback),
        channels=engine.search_channel_health(query, resolved_mode),
    )


def _op_index(
    *,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    force: bool = False,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_index(
            include_globs=include_globs,
            exclude_globs=exclude_globs,
            force=force,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("index", payload, render_compact=render_compact))


def _op_blame(
    *,
    query: str | None = None,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    include_churn: bool = True,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    if not (query or symbol_id or qualified_name or symbol_name):
        raise ValueError("query, symbol_id, qualified_name, or symbol_name is required for code blame")
    engine = _code_engine_at(repo_root)
    payload = cast(
        dict[str, Any],
        engine.tool_blame(
            query=query,
            symbol_id=symbol_id,
            qualified_name=qualified_name,
            symbol_name=symbol_name,
            file_path=path,
            include_churn=include_churn,
            budget_tokens=budget_tokens,
        ),
    )
    return _finish_code_result(_maybe_attach_code_rendered("blame", payload, render_compact=render_compact))


def _op_node(
    *,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    line: int | None = None,
    budget_tokens: int = 4000,
    repo: str | None = None,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    engine_root = repo_root or "."
    workspace_router = _workspace_code_router(engine_root)
    if repo is not None and not workspace_router.is_configured:
        raise ValueError("repo filter requires .lemoncrow/workspace.toml")
    engine = _code_context_engine(engine_root)
    _code_engine_for_current_call.value = engine
    if workspace_router.is_configured:
        payload = cast(
            dict[str, Any],
            workspace_router.route(
                "symbol",
                repo=repo,
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=path,
                line=line,
                budget_tokens=budget_tokens,
            ),
        )
    else:
        payload = cast(
            dict[str, Any],
            engine.tool_symbol(
                symbol_id=symbol_id,
                qualified_name=qualified_name,
                symbol_name=symbol_name,
                file_path=path,
                line=line,
                budget_tokens=budget_tokens,
            ),
        )
    return _finish_code_result(_maybe_attach_code_rendered("node", payload, render_compact=render_compact))


def _op_cache_status(
    *,
    cache_tool: str | None = None,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    engine = _code_engine_at(repo_root)
    if cache_tool is None:
        payload = cast(dict[str, Any], engine.tool_cache_status(budget_tokens=budget_tokens))
    else:
        payload = cast(
            dict[str, Any],
            engine.tool_cache_status(cache_tool=cache_tool, budget_tokens=budget_tokens),
        )
    return _finish_code_result(_maybe_attach_code_rendered("cache_status", payload, render_compact=render_compact))


def _op_cache_invalidate(
    *,
    cache_tool: str | None = None,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    engine = _code_engine_at(repo_root)
    if cache_tool is None:
        payload = cast(dict[str, Any], engine.tool_cache_invalidate(budget_tokens=budget_tokens))
    else:
        payload = cast(
            dict[str, Any],
            engine.tool_cache_invalidate(cache_tool=cache_tool, budget_tokens=budget_tokens),
        )
    return _finish_code_result(_maybe_attach_code_rendered("cache_invalidate", payload, render_compact=render_compact))


# ------------------------------------------------------------------ #
# Dedicated code-intel tools — each calls its `_op_*` engine wrapper  #
# directly (no multiplexer). Published tools carry focused schemas;   #
# repo/admin ops (index, blame, cache) are registered hidden via            #
# HIDDEN_LLM_TOOLS so tests and power use reach them by name.               #
# ------------------------------------------------------------------ #

# Code-intel tool names whose pre-rendered text (set during the _op_* call) is
# surfaced verbatim by render_tool_result_text. Includes the hidden repo/admin
# ops so `_call("index"/...)` returns rendered text for tests/power use.
_CODE_INTEL_TOOLS: frozenset[str] = frozenset(
    {
        "node",
        "callers",
        "callees",
        "usages",
        "codemod",
        "index",
        "blame",
        "cache",
        "cache_status",
        "cache_invalidate",
    }
)


def _parse_symbol(symbol: str) -> dict[str, Any]:
    """Route a symbol string to the correct engine kwarg based on form."""
    if "." in symbol:
        return {"qualified_name": symbol}
    return {"symbol_name": symbol}


@mcp_tool(name="graph")
def tool_graph(
    kind: str = "blast_radius",
    path: str | None = None,
    paths: list[str] | None = None,
    limit: int = 50,
    synthesize: bool = False,
    query: str | None = None,
    enable: bool | None = None,
) -> dict[str, Any]:
    """Repo graph analytics + code health & history: blast radius, dead code, cycles,
    coupling, centrality, doc/code drift, PR risk, commit provenance, design-doc recall.

    kind:
      - blast_radius (default): reverse-dependency closure + affected tests + risk tier for `path`.
      - dead_code: files with no inbound importers (likely removable), ranked by complexity.
      - cycles: import cycles (strongly-connected components, size >= 2).
      - coupling: per-file afferent/efferent coupling + Martin instability.
      - centrality: top symbols by call-graph centrality (degree + eigenvector).
      - design_gaps (G15): doc-referenced symbols absent from the index (stale/aspirational refs).
      - verify_design (G15): doc-referenced symbols with drifted signatures.
      - pr_risk (G16): blast-radius + complexity + churn + test-gap → 0..1 risk score + tier
        for the changed `paths` (or `path`).
      - commit_provenance (G16): heuristic bugfix/refactor/feature/perf/rename/revert/docs/test
        classification of commits touching `path` (or repo), tagged confidence.
      - index_docs (N17): opt-in heading-tree indexing of Markdown design docs into a SEPARATE
        retrieval store (`enable=true` or LEMONCROW_DOC_INDEXING=1; off by default).
      - recall_docs (N17): design-doc chunks for `query` from the separate doc store.
    `paths` = fold files into the index first (dead_code/cycles/coupling/pr_risk); for
    design_gaps/verify_design/index_docs it selects the docs/dirs to scan.
    `synthesize=true` (with `paths`, kind=centrality) → heuristic route/event edges as a
    SEPARATE `synthesized_edges` list (never merged into the static call graph).
    """
    return _op_graph(
        kind=kind,
        path=path,
        paths=paths,
        limit=limit,
        synthesize=synthesize,
        query=query,
        enable=enable,
    )


@mcp_tool(
    name="codemod",
    description=(
        "AST-shape search and rewrite via ast-grep. Matches structure, not text "
        "(formatting-safe; ignores strings/comments). `$X` = one node, `$$$` = list; "
        "e.g. pattern `$X == None`, rewrite `$X is None`. Scope with `language`/`glob`. "
        "`dry_run=true` (default) previews a diff, false applies. Returns matches "
        "(snippet, path, line); with rewrite: diff + files_changed."
    ),
    param_aliases={"file_glob": "glob"},
)
def tool_pattern(
    pattern: str,
    language: str | None = None,
    glob: str | None = None,
    rewrite: str | None = None,
    limit: int = 20,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Structural code search and safe rewrite (codemod) by AST shape, via ast-grep.

    Use over `grep` when matching code *shape*, not text: it is formatting-
    independent and never matches inside strings or comments. Metavariables:
    `$X` binds one node, `$$$` binds a list -- e.g. `isinstance($X, $Y)`,
    `$X == None`, `requests.get($URL)`.

    Pass `rewrite` to transform every match (the codemod); captured metavariables
    are reusable in the replacement, e.g. pattern `$X == None`, rewrite `$X is None`.
    `dry_run=True` (default) returns a unified-diff preview and writes nothing;
    `dry_run=False` applies the rewrite across all matched files. Scope with
    `language` (e.g. 'python') and `file_glob`.
    Returns: matches (snippet, file_path, line); with `rewrite`, a diff and `files_changed`.
    """
    return _op_pattern(
        pattern=pattern,
        rewrite=rewrite,
        language=language,
        file_glob=glob,
        dry_run=dry_run,
        limit=limit,
    )


# Repo/admin code-intel ops — registered hidden (see HIDDEN_LLM_TOOLS). Not
# surfaced to agents; reachable by name for tests, the CLI, and power use. Each
# delegates straight to its _op_* engine wrapper.
@mcp_tool(name="index")
def tool_index(
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    force: bool = False,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    """Build or refresh the code index for the repo (internal/admin)."""
    return _op_index(
        include_globs=include_globs,
        exclude_globs=exclude_globs,
        force=force,
        budget_tokens=budget_tokens,
        repo_root=repo_root,
        render_compact=render_compact,
    )


@mcp_tool(name="blame")
def tool_blame(
    query: str | None = None,
    symbol_id: str | None = None,
    qualified_name: str | None = None,
    symbol_name: str | None = None,
    path: str | None = None,
    include_churn: bool = True,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    """Git blame / churn summary for a symbol or file (internal/admin)."""
    return _op_blame(
        query=query,
        symbol_id=symbol_id,
        qualified_name=qualified_name,
        symbol_name=symbol_name,
        path=path,
        include_churn=include_churn,
        budget_tokens=budget_tokens,
        repo_root=repo_root,
        render_compact=render_compact,
    )


@mcp_tool(name="statusline_segment")
def tool_statusline_segment(format: str = "segment") -> str:
    """Savings surface for the active session.

    - ``format="segment"`` (default): the pre-computed rotating statusline
      frame. statusline.sh reads the sessions/<id>/statusline_segment sidecar
      directly; this path refreshes that file and returns its content.
    - ``format="markdown"``: the full `lc savings` panel as markdown, for hosts
      that render chat markdown and have no shell to run the CLI.
    - ``format="json"``: the raw savings report payload, JSON-encoded.

    Hidden from tools/list (see HIDDEN_LLM_TOOLS) but callable by exact name
    through the `tool` broker (see _BROKER_HIDDEN_CALLABLE), which is how the
    lemoncrow skill answers "what are my savings?" without a shell.
    """
    fmt = (format or "segment").strip().lower()
    if fmt in {"markdown", "md", "json"}:
        from lemoncrow.core.capabilities.plugin_runtime import build_savings_report
        from lemoncrow.core.capabilities.savings_summary import render_savings_markdown

        payload = build_savings_report(_lemoncrow_root())
        if fmt == "json":
            return json.dumps(payload, indent=2, sort_keys=True, default=str)
        return render_savings_markdown(payload)
    try:
        sidecar = _get_host_session_sidecar_path()
        seg_path = sidecar.parent / "statusline_segment"
        sid = sidecar.parent.name
        from lemoncrow.core.capabilities.savings_summary import savings_segment

        seg = savings_segment(session_id=sid)
        if seg:
            seg_path.write_text(seg, encoding="utf-8")
            return seg
        if seg_path.exists():
            return seg_path.read_text(encoding="utf-8").strip()
    except Exception:
        _log.debug("tool_statusline_segment failed", exc_info=True)
    return ""


@mcp_tool(name="cache")
def tool_cache(
    op: Literal["status", "invalidate"] = "status",
    cache_tool: str | None = None,
    budget_tokens: int = 4000,
    repo_root: str | None = None,
    render_compact: bool = False,
) -> dict[str, Any]:
    """Code-intel cache admin (internal/admin).

    op='status' (default) reports cache hit/miss counters; op='invalidate'
    clears caches, optionally scoped to one `cache_tool`. Folds the former
    cache_status / cache_invalidate tools into one hidden admin face.
    """
    if op == "invalidate":
        return _op_cache_invalidate(
            cache_tool=cache_tool,
            budget_tokens=budget_tokens,
            repo_root=repo_root,
            render_compact=render_compact,
        )
    return _op_cache_status(
        cache_tool=cache_tool,
        budget_tokens=budget_tokens,
        repo_root=repo_root,
        render_compact=render_compact,
    )


@mcp_tool(name="scan")
def tool_scan(
    path: str | None = None,
    include_taint: bool = True,
    include_rules: bool = True,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Security scan (SAST, first iteration) over the repo or a sub-path.

    Runs a small bundled pack of high-signal OWASP/CWE ast-grep rules
    (eval/exec, subprocess shell=True with interpolation, SQL string
    concatenation, hardcoded secrets) plus a BOUNDED intra-procedural Python
    taint check (request/argv/env/input sources reaching exec/subprocess/SQL
    sinks). Scope to a file or directory with `path`; toggle the rule pack or
    taint pass with `include_rules`/`include_taint`.

    This is a first iteration, NOT a full SAST engine: every finding carries a
    `rule_id`, `severity`, and `confidence`, and heuristic findings are flagged
    (`heuristic: true`). It does not claim exhaustiveness.
    Returns: findings (path, line, rule_id, cwe, severity, confidence, message,
    source, heuristic) and a summary count.
    """
    from lemoncrow.core.capabilities.security import scan_repository

    workspace = _workspace_root()
    root_arg = repo_root or "."
    root_path = Path(root_arg)
    resolved_root = (root_path if root_path.is_absolute() else workspace / root_path).resolve()
    paths = [path] if path else None
    findings = scan_repository(
        resolved_root,
        paths=paths,
        include_taint=include_taint,
        include_rules=include_rules,
    )
    severity_counts: dict[str, int] = {}
    for finding in findings:
        sev = str(finding.get("severity", "info"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    return {
        "findings": findings,
        "summary": {
            "total": len(findings),
            "by_severity": severity_counts,
            "heuristic": True,
            "note": "First-iteration SAST: bounded coverage, not exhaustive.",
        },
    }


@mcp_tool(name="orient")
def tool_orient(topic: str | None = None) -> dict[str, Any]:
    """Return LemonCrow's tool-usage playbook on demand (N8).

    One fetch for the optimal tool sequencing -- explore -> navigate -> edit ->
    verify -- and which tool to reach for in each phase, so this guidance need
    not be duplicated in every system prompt. Static and deterministic.

    Pass an optional `topic` (explore, navigate, edit, verify, selection) for a
    single focused section instead of the whole playbook. An unknown topic is
    not an error: it returns the overview plus the list of valid topics.
    Returns: sequence, sections (title/body), topics, and rendered `text`.
    """
    from lemoncrow.core.capabilities.orientation import orientation_playbook

    return orientation_playbook(topic)


# _DeferredResult, _Deferred and the deferral kill-switch/executor helpers now
# live in mcp.deferral (imported above).
# bash symbols moved to mcp.bash (imported/registered near the top).


# bash symbols moved to mcp.bash (imported/registered near the top).


# bash symbols moved to mcp.bash (imported/registered near the top).


# bash symbols moved to mcp.bash (imported/registered near the top).


def _run_native_grep(
    *,
    path: str,
    content_regex: str | None,
    file_glob_patterns: list[str] | None,
    output_mode: Literal[
        "ranked_file_map",
        "file_paths_with_content",
        "file_paths_only",
        "file_paths_with_match_count",
    ],
    lines_before: int,
    lines_after: int,
    ignore_case: bool,
    type: str | None,
    file_limit: int | None,
    lines_per_file: int | None,
    if_modified_since: str | None,
    multiline: bool,
    summary: bool | None,
    context_budget_tokens: int,
    include_meta: bool,
    badge_provider: Callable[[str, list[str]], str | None] | None = None,
) -> dict[str, Any]:
    from lemoncrow.pro.capabilities.tool_supervision.native_search import search_workspace

    return search_workspace(
        path=path,
        content_regex=content_regex,
        file_glob_patterns=file_glob_patterns,
        output_mode=output_mode,
        lines_before=lines_before,
        lines_after=lines_after,
        ignore_case=ignore_case,
        type=type,
        file_limit=file_limit,
        lines_per_file=lines_per_file,
        if_modified_since=if_modified_since,
        max_line_length=1000,
        multiline=multiline,
        summary=summary,
        context_budget_tokens=context_budget_tokens,
        include_metadata=include_meta,
        repo_root=os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd()),
        badge_provider=badge_provider,
    )


# Cap on how many distinct symbols one grep call will badge with call-graph
# counts -- keeps the symbol lookups bounded on large result sets.
_GREP_BADGE_SYMBOL_CAP = 8

# Model-facing grep output-mode names map to the verbose internal output_mode the
# native engine speaks. Short names keep the per-turn schema small.
_GREP_MODE_ALIASES: dict[str, str] = {
    "with_content": "file_paths_with_content",
    "ranked_map": "ranked_file_map",
    "paths_only": "file_paths_only",
    "count_only": "file_paths_with_match_count",
}


# Forgiving mode normalisation: the model prefers self-documenting names
# (file_paths_with_content) over the terse canonical ones, so accept both forms
# plus common variants and default unknowns to 'content' -- grep never 422s on mode.
_GREP_MODE_CANON: dict[str, str] = {
    "with_content": "with_content",
    "ranked_map": "ranked_map",
    "paths_only": "paths_only",
    "count_only": "count_only",
    "content": "with_content",
    "map": "ranked_map",
    "paths": "paths_only",
    "counts": "count_only",
    "file_paths_with_content": "with_content",
    "files_with_content": "with_content",
    "file_content": "with_content",
    "ranked_file_map": "ranked_map",
    "file_map": "ranked_map",
    "ranked": "ranked_map",
    "file_paths_only": "paths_only",
    "file_paths": "paths_only",
    "files": "paths_only",
    "filenames": "paths_only",
    "file_paths_with_match_count": "count_only",
    "file_paths_with_count": "count_only",
    "match_count": "count_only",
    "count": "count_only",
}


def _normalize_grep_mode(mode: object) -> str:
    """Map any reasonable mode spelling to a canonical short name; unknown -> content."""
    return _GREP_MODE_CANON.get(str(mode or "with_content").strip().lower(), "with_content")


def _grep_badge_provider(rel_path: str, symbol_names: list[str]) -> str | None:
    """Inline relation-count line for a file's matched definition symbols.

    Returns a ` · `-joined badge string (or ``None``) that `search_workspace`
    appends to the file's header so call-graph counts ride along the regex
    search. Uses badge_counts_batch: 3 SQL queries for all symbols instead
    of 3*N serial queries.
    """
    if not symbol_names:
        return None
    repo_root = os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd())
    names = symbol_names[:_GREP_BADGE_SYMBOL_CAP]
    try:
        engine = _code_context_engine(repo_root)
        counts = engine.badge_counts_batch(names)
    except Exception:
        return None
    badges: list[str] = []
    for name in names:
        c = counts.get(name, {})
        parts: list[str] = []
        n_callers = int(c.get("callers") or 0)
        n_callees = int(c.get("callees") or 0)
        n_usages = int(c.get("usages") or 0)
        if n_callers:
            parts.append(f"↳{n_callers} callers")
        if n_callees:
            parts.append(f"↰{n_callees} callees")
        if n_usages:
            parts.append(f"⌖{n_usages} usages")
        if parts:
            badges.append(f"{name}  " + " ".join(parts))
    return "  ·  ".join(badges) if badges else None


# --- lean code_search projection --------------------------------------------- #
# code_search is meant to collapse grep->read->edit into a single code_search->edit.
# The engine returns a rich candidate set; handed that whole payload, agents over-
# search and re-read. Project it to a lean, exact view: rank files by best entry-
# point score, return the top files' source verbatim (the code to edit), trim the
# score tail to compact signatures, drop non-actionable metadata. Generic and
# score-relative -- no per-repo tuning. Offline-validated on the real benchmark
# queries: ~64% smaller output, gold-edited file retained in every case.
_LEAN_REL_FLOOR = 0.10
_LEAN_MAX_SOURCE_FILES = 3
_LEAN_MAX_CANDIDATES = 8
# tool_code_search used to expose its own max_files param (hidden from the
# LLM, default 8, 0 real callers ever needed a different value -- narrowing
# happens via paths=/query). Display count was ALREADY fixed regardless: any
# value >=3 gives an identical n_src via min(_LEAN_MAX_SOURCE_FILES, ...)
# below. Collapsed to a plain constant (2026-07-25) rather than a dead
# parameter -- still feeds engine.tool_explore's own internal ranking
# breadth, just no longer externally settable.
_CODE_SEARCH_ENGINE_MAX_FILES = 8
# Candidate-file list is a path-only recall aid (navigation targets). 24 costs
# a measured 0.8-1.5k chars resident per turn -- but a FLAT trim to 10, then 6
# was part of a measured -0.10 overall retrieval MRR regression (2026-07-06):
# the tail is where non-top-1 golds live, so it earns its chars in the general
# (ambiguous / multi-candidate) case. Do not flat-trim this constant without
# re-running `lc eval retrieval`. The symbol map (_LEAN_MAX_CANDIDATES) stays
# tight because each entry carries structure.
# 24 -> 16 -> 8 (2026-07-25): re-testing the flat-trim ceiling now that the
# dominant-case cap, doc-diversity demotion, and inline gating all exist --
# none of which were present for the 2026-07-06 10/6 regression above. 8 sits
# INSIDE that historical regression's range (10 then 6 both regressed -0.10
# MRR combined) -- 16->8 is the real test of whether the old cause still
# applies under the current architecture; 24->16 alone measured -0.0008 MRR
# (noise-floor, hit@1 unchanged). Gated by the same `lc eval retrieval`
# re-run rule; revert to 16 (or 24) if THIS step regresses for real.
_LEAN_MAX_CANDIDATE_FILES = 8
# Dynamic exception, not a flat trim: only when `_lean_code_search_view`'s
# `dominant` signal fires (exact match, no seed scope, one file scoring
# >=4x its nearest rival, or the only scored file) is the tail this cheap to
# cut -- the same exact-single-winner case that already collapses `files` to
# n_src=1. Ambiguous/no-exact-match cases keep the full 8-wide net.
# Deliberately NOT keyed on any per-candidate score (e.g. epscore==0): that
# was tried and falsified by
# test_lean_code_search_view_candidates_keep_symbol_map_paths's fresh.py case
# (epscore==0 means "no entry_points row", not "irrelevant") -- this instead
# reuses the already-validated whole-response `dominant` confidence signal.
# 8 -> 5 (2026-07-25): the dominant case is already the confident
# single-winner path (files already collapse to n_src=1) -- narrower blast
# radius than the general cap, so trimmed further first. Gated by the same
# `lc eval retrieval` re-run rule; revert to 8 if MRR regresses.
_LEAN_CANDIDATE_FILES_DOMINANT = 5
_LEAN_DOMINANT_RATIO = 4.0
# Tried a dynamic quality gate here (drop candidate_files entries with
# epscore==0.0) and reverted before it reached eval: epscore is populated
# ONLY from entry_points/symbol-level matches, so a file reaching `files` via
# a channel that never emits an entry_points row (pure lexical/path match, no
# symbol pinned) scores 0.0 despite being a real candidate --
# test_lean_code_search_view_candidates_keep_symbol_map_paths's `fresh.py`
# case is exactly this, and it's hardened from the same 2026-07-06 MRR
# regression this whole cand_files block warns about. A sound dynamic gate
# needs a real per-candidate confidence score unified across every surfacing
# channel (not just entry_points) -- that's engine-level work, not available
# at this layer today.


def _lean_score(sym: dict[str, Any]) -> float:
    try:
        return float(sym.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _lean_related_symbol(sym: dict[str, Any]) -> dict[str, Any]:
    name = sym.get("qualified_name") or sym.get("name")
    if isinstance(name, str):
        # Markdown headings can index a whole frontmatter block as their "name";
        # a related_symbols entry is a one-line pointer, never a prose dump.
        name = name.split("\n", 1)[0].strip()
        if len(name) > 120:
            name = name[:117] + "..."
    result: dict[str, Any] = {
        "qualified_name": name,
        "path": sym.get("path"),
        "line": sym.get("line"),
        "end_line": sym.get("end_line"),
        "kind": sym.get("kind"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _lean_section(sec: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": sec.get("path"),
        "qualified_name": sec.get("qualified_name") or sec.get("name"),
        "line": sec.get("line"),
        "end_line": sec.get("end_line"),
        "content": sec.get("content", ""),
    }


# Guards the whole-file fallback below against reading a pathological huge
# file into memory -- downstream _outline_lean_view already caps rendered
# size (_CODESEARCH_TOP2_MAX_CHARS), this only bounds the raw read.
_CODESEARCH_FALLBACK_MAX_BYTES = 2_000_000


def _whole_file_fallback_section(workspace_root: Path, relpath: str) -> dict[str, Any] | None:
    """Whole-file source entry for a resolved-path code_search hit whose file
    type carries no symbol index (shell scripts, markdown, config, ...), so
    the engine's normal symbol pipeline returns zero source_sections for it.
    Without this, a fast-path pin to a non-code file would resolve exact_match
    correctly but ship no content -- the caller would still have to `read` it
    themselves. Downstream outline/top2 shaping still bounds the size.
    """
    target = workspace_root / relpath
    try:
        if target.stat().st_size > _CODESEARCH_FALLBACK_MAX_BYTES:
            return None
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return {
        "path": relpath,
        "sections": [
            {
                "path": relpath,
                "qualified_name": None,
                "line": 1,
                "end_line": text.count("\n") + 1,
                "content": text,
            }
        ],
    }


# Baseline for the token credit = the source you'd realistically READ (the
# returned section + comparable surrounding context), not the whole file.
_REALISTIC_READ_FACTOR = 2


# Per-session set of (absolute) file paths already credited a code_search read /
# token saving. A repeat search returning the same file re-reads nothing — you
# already have it — so credit each file once per session, not once per search.
_SEARCH_CREDITED_PATHS: dict[str, set[str]] = {}


def _code_search_section_savings(lean: dict[str, Any], workspace_root: Path) -> tuple[int, int]:
    """Estimate the round-trips a code_search replaced vs. a manual grep+read loop.

    Credited ONCE per file per session: a later search returning a file you
    already pulled re-reads nothing.

    A single ``grep``/``glob`` returns matches across ALL files at once, so the
    primary locate is a 1-for-1 swap with the code_search call — NOT one avoided
    call per file. What the call genuinely avoids is a measured lower bound:
      * one ``Read`` per file whose source came back whole (a section cut short
        with a ``[truncated …]`` marker is a preview you might still open the
        file for, so it isn't counted); plus
      * one extra locate (grep for usages) when the symbol map points to files
        beyond the returned source — a second query you'd otherwise run.
    """
    files = lean.get("files")
    if not isinstance(files, list):
        return 0, 0
    try:
        session = _resolved_host_session_id() or ""
    except Exception:
        session = ""
    credited = _SEARCH_CREDITED_PATHS.setdefault(session, set())

    tokens_saved = 0
    reads_saved = 0
    source_paths: set[str] = set()  # files with source THIS call (nav gating)
    for entry in files:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        sections = entry.get("sections")
        if not isinstance(sections, list) or not sections:
            continue
        file_content = ""
        for section in sections:
            if isinstance(section, dict):
                content = section.get("content")
                if isinstance(content, str):
                    file_content += content
        source_paths.add(raw_path)
        candidate = Path(raw_path)
        full_path = candidate if candidate.is_absolute() else workspace_root / candidate
        key = str(full_path)
        if key in credited:
            continue  # already pulled this session — a repeat search re-reads nothing
        try:
            file_size = min(full_path.stat().st_size, _HOST_INLINE_RESULT_CHARS)
        except OSError:
            continue
        credited.add(key)
        # A Read is avoided only when the source came back whole; a truncated
        # outline is a preview, not a replaced read — keep the count a lower bound.
        if file_content and "[truncated" not in file_content:
            reads_saved += 1
        # Credit the source you'd realistically have read (section + comparable
        # surrounding context), not the entire file: a targeted read opens the
        # symbol's neighborhood, not a 100k-line file.
        actual = len(file_content)
        tokens_saved += max(0, (min(file_size, _REALISTIC_READ_FACTOR * actual) - actual) // 4)

    # One extra locate (grep for usages) is avoided only when the symbol map
    # points to files beyond the ones we returned source for — otherwise it's
    # the same files you already have, not a second query.
    nav_saved = 0
    related = lean.get("related_symbols")
    if isinstance(related, list):
        for sym in related:
            if isinstance(sym, dict):
                p = sym.get("path")
                if isinstance(p, str) and p and p not in source_paths:
                    nav_saved = 1
                    break
    calls_saved = reads_saved + nav_saved
    return tokens_saved, calls_saved


def _lean_code_search_view(
    result: dict[str, Any],
    *,
    max_files: int,
    seed_files: list[str] | None = None,
    query: str = "",
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Collapse engine.tool_explore output into a lean, edit-ready view."""
    if not isinstance(result, dict):
        return result
    # Filetype diversity: unless the query asks for docs, documentation files
    # (.md/.rst/...) may occupy at most a quarter of each ranked window --
    # excess docs are demoted below it, never dropped from the surface order.
    wants_docs = query_wants_docs(query)
    eps = sorted((result.get("entry_points") or []), key=_lean_score, reverse=True)
    files = result.get("files") or []
    exact = bool(result.get("exact_match"))

    epscore_by_path: dict[str, float] = {}
    for e in eps:
        p = e.get("path")
        if p is not None:
            epscore_by_path[p] = max(epscore_by_path.get(p, 0.0), _lean_score(e))

    seed_norm = {s.rstrip("/") for s in (seed_files or [])}

    def is_seed(path: str | None) -> bool:
        if not path:
            return False
        return any(path == s or path.startswith(s + "/") for s in seed_norm)

    def _fscore(f: dict[str, Any]) -> float:
        fp = f.get("path")
        return epscore_by_path.get(fp if isinstance(fp, str) else "", 0.0)

    # A learned explore reranker already produced the file order. Preserve that
    # order through the MCP projection instead of immediately overwriting it
    # with legacy entry-point scores; otherwise the reranker is effectively a
    # no-op on the agent-visible `files + candidate_files` ranking. This changes
    # ordering only — the lean response schema is untouched.
    experiment = result.get("experiment")
    learned_order = isinstance(experiment, dict) and str(experiment.get("name") or "").startswith("explore_reranker_")

    # Reserve up to the top-2 spots for in-scope files that actually matched
    # (epscore > 0), highest first; every other file -- out-of-scope neighbours the
    # whole-repo channels surfaced, plus any 3rd+ in-scope file -- races purely on
    # score below them. An unmatched in-scope file claims no reserved slot, so a
    # 0-match seed lets a neighbour take rank #1.
    if learned_order:
        _reserved = [f for f in files if is_seed(f.get("path")) and _fscore(f) > 0.0][:2]
        _reserved_ids = {id(f) for f in _reserved}
        _rest = [f for f in files if id(f) not in _reserved_ids]
    else:
        _reserved = sorted(
            (f for f in files if is_seed(f.get("path")) and _fscore(f) > 0.0),
            key=_fscore,
            reverse=True,
        )[:2]
        _reserved_ids = {id(f) for f in _reserved}
        _rest = sorted((f for f in files if id(f) not in _reserved_ids), key=_fscore, reverse=True)
    ranked = [*_reserved, *_rest]
    # Dominance controls source volume, not file relevance. Keep it anchored to
    # the legacy entry-point confidence even when the learned model changes file
    # order; otherwise a reranked low-epscore file at rank 1 turns an exact-hit
    # response from one source file into up to `max_files` source files.
    dominance_ranked = sorted(files, key=_fscore, reverse=True) if learned_order else ranked
    top_fs = epscore_by_path.get(dominance_ranked[0].get("path"), 0.0) if dominance_ranked else 0.0
    second_fs = epscore_by_path.get(dominance_ranked[1].get("path"), 0.0) if len(dominance_ranked) > 1 else 0.0
    dominant = (
        exact and not seed_norm and top_fs > 0.0 and (second_fs == 0.0 or top_fs >= _LEAN_DOMINANT_RATIO * second_fs)
    )
    # Ranking surface is sacred: WHICH files carry sections feeds retrieval
    # rank (and the MRR eval). Context cost of extra sections is handled by
    # outline shaping (pointer-only past the cap), never by dropping files
    # here -- gating this on exact_match measured -0.10 overall MRR (2026-07-06).
    n_src = 1 if dominant else min(_LEAN_MAX_SOURCE_FILES, max(1, max_files))

    out_files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for f in ranked[:n_src]:
        secs = [_lean_section(s) for s in (f.get("source_sections") or [])]
        if secs:
            out_files.append({"path": f.get("path"), "sections": secs})
            seen_paths.add(f.get("path"))
    if not out_files:  # never return zero source when some file has sections
        for f in ranked:
            secs = [_lean_section(s) for s in (f.get("source_sections") or [])]
            if secs:
                out_files.append({"path": f.get("path"), "sections": secs})
                seen_paths.add(f.get("path"))
                break
    # Cross-file symbol map: top-K entry points as structured ranges. On
    # multi-file tasks this lets callers navigate related sites without a second
    # symbol-resolution call. Symbols whose exact span already came back as
    # source are skipped -- repeating a returned section as a pointer is noise.
    returned_spans: set[tuple[Any, Any, Any]] = set()
    for out in out_files:
        for sec in out.get("sections") or []:
            returned_spans.add((out.get("path"), sec.get("line"), sec.get("end_line")))
    candidates: list[dict[str, Any]] = []
    seen_sig: set[tuple[Any, Any, Any, Any]] = set()
    for e in eps:
        candidate = _lean_related_symbol(e)
        if (candidate.get("path"), candidate.get("line"), candidate.get("end_line")) in returned_spans:
            continue
        sig = (
            candidate.get("qualified_name"),
            candidate.get("path"),
            candidate.get("line"),
            candidate.get("end_line"),
        )
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        candidates.append(candidate)
        # Scan past the window so the diversity pass below has non-doc
        # candidates to promote when docs dominate the top scores.
        if len(candidates) >= _LEAN_MAX_CANDIDATES * 3:
            break
    if not wants_docs:
        # Effective window: the list head IS the ranked space when the list is
        # shorter than the cap.
        candidates = demote_doc_overflow(
            candidates,
            window=min(_LEAN_MAX_CANDIDATES, len(candidates)),
            path_of=lambda c: str(c.get("path") or ""),
        )
    candidates = candidates[:_LEAN_MAX_CANDIDATES]
    # Exclude ONLY files whose source was returned (seen_paths). Files that
    # appear in the symbol map STAY in candidate_files: candidate_files is the
    # ranked recall surface (the retrieval eval and rank-consumers read files +
    # candidate_files, never the symbol map) -- excluding map-covered paths
    # deleted the top entry-point files from the ranking and collapsed hit@3
    # (part of a measured -0.10 overall MRR, 2026-07-06).
    cand_files: list[str] = []
    for f in ranked:
        p = f.get("path")
        if p and p not in seen_paths and p not in cand_files:
            cand_files.append(p)
    for p in result.get("additional_relevant_files") or []:
        if p and p not in seen_paths and p not in cand_files:
            cand_files.append(p)
    # Recall tails (fused cross-channel matches, then a deeper lexical pass),
    # appended STRICTLY LAST so the primary candidate order is unchanged. NOT
    # gated on exact_match: exact_match only says SOME exact symbol name
    # matched, not that the gold file surfaced -- gating these on it measured
    # part of the same -0.10 MRR regression.
    for p in result.get("fused_recall") or []:
        if p and p not in seen_paths and p not in cand_files:
            cand_files.append(p)
    for p in result.get("deep_recall") or []:
        if p and p not in seen_paths and p not in cand_files:
            cand_files.append(p)

    # Dynamic cap: the exact-single-winner case (`dominant`, same signal that
    # already collapses `files` to one) also shrinks the candidate_files tail
    # -- see _LEAN_CANDIDATE_FILES_DOMINANT for why this is safe where a flat
    # trim was not. Ambiguous/multi-candidate cases keep the full 8-wide net.
    # An explicit max_candidates from the caller overrides both defaults --
    # the agent knows its own confidence better than either heuristic (e.g.
    # "just the top hit" after it's already narrowed the file itself).
    if max_candidates is not None:
        cand_cap = max(1, max_candidates)
    else:
        cand_cap = _LEAN_CANDIDATE_FILES_DOMINANT if dominant else _LEAN_MAX_CANDIDATE_FILES
    if cand_files and not wants_docs:
        cand_files = demote_doc_overflow(cand_files, window=min(cand_cap, len(cand_files)))

    lean: dict[str, Any] = {"exact_match": exact, "files": out_files}
    if candidates:
        lean["related_symbols"] = candidates
    if cand_files:
        lean["candidate_files"] = cand_files[:cand_cap]
        # The cap silently cutting the ranked tail would look identical to
        # "nothing else matched" -- name the count still hidden and how to
        # raise the cap, so the agent can ask for more instead of assuming
        # recall stopped here.
        _hidden = len(cand_files) - cand_cap
        if _hidden > 0:
            lean["candidate_files_more"] = _hidden
    if result.get("truncated"):
        lean["truncated"] = True
    return lean


def _attach_code_search_savings(view: dict[str, Any], workspace_root: Path | None = None) -> dict[str, Any]:
    """Attach tokens_saved/calls_saved AFTER outline shaping.

    Savings must be measured on the view the model actually receives: an
    outlined or head-capped section is a pointer the agent may still read, not
    a replaced Read -- crediting the pre-shaped source would overcount exactly
    when outline mode trims the most.
    """
    root = workspace_root or _workspace_root()
    # Inline source = already read: files whose sections were served count as
    # reads for the blind-range-edit freshness ledger too.
    for _entry in view.get("files") or []:
        if isinstance(_entry, dict) and _entry.get("sections"):
            _p = _entry.get("path")
            if isinstance(_p, str) and _p:
                _record_read_sig(_p if Path(_p).is_absolute() else root / _p)
    tokens_saved, calls_saved = _code_search_section_savings(view, root)
    if tokens_saved > 0:
        view["tokens_saved"] = tokens_saved
    if calls_saved > 0:
        view["calls_saved"] = calls_saved
    return view


@mcp_tool(
    name="code_search",
    description=(
        "Search the indexed codebase. One call returns top matches' source inline "
        "(bounded), lower-ranked or oversized ones as precise path:Lx-Ly pointers "
        "(`read` exactly that range), a related-symbols map (path:Lx-Ly kind name), "
        "and candidate_files. Use instead of grep/find. Inline source = already read. "
        "One broad query beats several narrow rephrasings — combine every term into "
        "a single call; re-querying rarely surfaces anything new."
    ),
    param_aliases={
        "maxFiles": "limit",
        "max_results": "limit",
        "max_files": "limit",
        "max_candidates": "limit",
        "projectPath": "paths",
        "path": "paths",
        "include_paths": "paths",
        "pattern": "query",
        "regex": "query",
    },
    # include_source hidden from the schema: an un-nudged model reaches for it
    # "to be safe" and pulls thousands of resident chars it then re-reads anyway.
    # One extra precise read beats a speculative full-source dump. Hidden callers
    # (tests, power use) still pass it by name.
    # There used to be a SEPARATE max_files param here (hidden, default 8) that
    # looked like it did the same job as this cap -- it didn't: it fed engine
    # breadth + an inline-source count already fixed at 3 regardless of its
    # value (0 real callers ever moved it). Removed as a public/tool-level
    # concept entirely (see _CODE_SEARCH_ENGINE_MAX_FILES) so there is exactly
    # ONE agent-visible count knob. Every vanilla-habit spelling for "how many
    # results" (maxFiles, max_results, max_files, max_candidates) now lands on
    # `limit`, the real parameter.
    hidden_params=("include_source", "force"),
)
def tool_code_search(
    query: Annotated[
        str,
        Field(description="Question, symbol/file names, code terms, or regex."),
    ],
    paths: Annotated[
        str | list[str] | None,
        Field(description="Optional file or directory scope."),
    ] = None,
    include_source: Annotated[
        bool,
        Field(description="Hidden: keep the top-2 matches' source inline (bounded)."),
    ] = False,
    limit: Annotated[
        int | None,
        Field(ge=1, le=32, description="Cap candidate_files entries (default 8, or 5 when one file dominates)."),
    ] = None,
    force: Annotated[
        bool,
        Field(
            description=(
                "Hidden: bypass within-session content dedup and re-emit the full "
                "result even if byte-identical to an earlier call this session."
            )
        ),
    ] = False,
) -> dict[str, Any]:
    """Relevant symbols' source grouped by file + call-graph relations, in one capped call.

    Seeds from the symbol/text index, pins exact-name matches, expands the call graph,
    and renders budgeted, line-numbered, skeletonized source. Treat the returned source
    as already read -- do not re-open those files with `read`.
    """
    _ = force  # consumed by the dedup dispatcher wrapper (mcp_server._handle), not here
    workspace_root = _workspace_root()
    # Check BEFORE doing any search work: a flagged repeat query skips the
    # engine entirely and returns blank -- no explanatory text (measured: the
    # caller skims past prose hints and searches again anyway; blank results
    # cost nothing and say the same thing).
    if _check_repeat_query(query):
        return _attach_code_search_savings({"exact_match": False, "files": []}, workspace_root)
    # Normalise: paths param accepts list, comma-sep string, or single path
    # (the legacy `path` kwarg is folded into `paths` by param_aliases).
    raw = paths
    if isinstance(raw, list):
        seed_list = [p.strip() for p in raw if p and p.strip()]
    elif isinstance(raw, str):
        seed_list = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        seed_list = []
    seed_files = seed_list or None
    engine = _code_context_engine(str(workspace_root))
    # Fast path: a query that's itself an existing repo file (verbatim path, a
    # mistyped path with a bogus leading segment, or a unique basename) is
    # pinned straight to that file instead of ranked -- only when the caller
    # didn't already supply an explicit paths= scope, which takes precedence.
    resolved_path = None if seed_files else _resolve_query_as_existing_file(workspace_root, query, engine)
    explore_seeds = [resolved_path] if resolved_path else seed_files
    result = cast(
        dict[str, Any], engine.tool_explore(query, max_files=_CODE_SEARCH_ENGINE_MAX_FILES, seed_files=explore_seeds)
    )
    # paths= is a SOFT scope at every shape (single file, directory, multi-path):
    # the engine reserves only the top-2 spots for in-scope files and lets the
    # neighbouring files its whole-repo channels surfaced race in below, so there is
    # no hard out-of-scope post-filter. The lean view applies the same reserve-2 cap
    # to the agent-visible file ranking.
    if resolved_path:
        result["exact_match"] = True
    # Project the engine's rich candidate set to a lean, exact view so the agent
    # can go code_search -> edit without grep/read round-trips (seed files are
    # boosted to the top inside the view).
    lean = _lean_code_search_view(
        result,
        max_files=_CODE_SEARCH_ENGINE_MAX_FILES,
        seed_files=explore_seeds,
        query=query,
        max_candidates=limit,
    )
    if resolved_path and not any(
        isinstance(e, dict) and e.get("path") == resolved_path and e.get("sections") for e in (lean.get("files") or [])
    ):
        # Fast-path hit on a file type with no symbol index (e.g. a shell
        # script) -- the engine surfaced it in candidate_files but rendered no
        # source. Inject the whole file directly rather than making the agent
        # follow up with a plain `read`.
        fallback = _whole_file_fallback_section(workspace_root, resolved_path)
        if fallback:
            lean.setdefault("files", []).insert(0, fallback)
    # Outline shaping happens HERE (dict level, before any rendering) so the
    # compact text renderer and the JSON fallback stay in lockstep -- and BEFORE
    # savings, so an outlined section is never credited as a replaced read.
    # Adaptive inline/outline split: on an EXACT match the top-2 sections ship
    # inline (bounded at _CODESEARCH_TOP2_MAX_CHARS) -- the agent nearly always
    # edits the top hit, and a pointer there forces a read round-trip on every
    # search (measured ~1-2 extra turns/run on SWE tasks). Ranked-candidate
    # (no-exact-match) results stay pointer-only: the top hit is a gamble, so
    # the agent reads only what it picks. Content-only shaping -- the ranked
    # file/candidate surface (and retrieval MRR) is untouched either way.
    shaped = _outline_lean_view(lean, keep_top2=include_source or bool(lean.get("exact_match")))
    return _attach_code_search_savings(shaped, workspace_root)


@mcp_tool(
    name="relations",
    description=(
        "Expand one symbol's call-graph relation into the actual list: kind=callers|callees|usages|self. "
        "use this only to see WHICH callers/callees/usages when worth drilling into."
    ),
)
def tool_relations(
    symbol: str,
    kind: str = "usages",
    depth: int = 1,
    limit: int = 20,
) -> dict[str, Any]:
    """Return the actual callers / callees / usages / definition of one symbol.

    `symbol` is a name, qualified path, or symbol id. `kind` selects the relation
    (default usages); `depth` extends callers/callees transitively. The COUNTS for
    these already ride along on `grep` definition matches — use this only to expand
    a count into the concrete list.
    """
    target = _parse_symbol(symbol)
    rel = kind.strip().lower()
    if rel == "callers":
        return _op_callers(**target, depth=depth, limit=limit)
    if rel == "callees":
        return _op_callees(**target, depth=depth, limit=limit)
    if rel in ("usages", "refs", "references"):
        return _op_usages(**target, limit=limit)
    if rel in ("self", "node", "definition"):
        return _op_node(**target)
    raise ValueError(f"unknown kind {kind!r}; use callers, callees, usages, or self")


@mcp_tool(
    name="grep",
    description=(
        "Search code by regex/glob/type. mode='with_content' (default) = discover AND "
        "read matched context in one step. Match on a symbol definition → its "
        "caller/callee/usage counts ride along inline; expand the lists via `relations`."
    ),
    hidden_params=(
        "include_meta",
        "format",
        "lines_per_file",
        "context_budget_tokens",
        "file_limit",
        "if_modified_since",
    ),
    param_aliases={
        "pattern": "regex",
        "content_regex": "regex",
        "file_glob_patterns": "glob",
        "output_mode": "mode",
        "lines_before": "before",
        "lines_after": "after",
        "ignore_case": "i",
        "-i": "i",
        "-B": "before",
        "-A": "after",
    },
)
def tool_grep(
    path: Annotated[
        str,
        Field(description="Workspace path; single file may carry ':Lx-Ly' (e.g. 'store.py:L60-L100') to scope."),
    ] = ".",
    regex: Annotated[
        str | None,
        Field(description="Regex to match contents; for relation mode this is the symbol when `symbol` is omitted."),
    ] = None,
    glob: Annotated[
        str | list[str] | None,
        Field(description="Globs constraining candidate files (e.g. `src/**/*.py`). List or bare string."),
    ] = None,
    mode: Annotated[
        str,
        Field(
            description=(
                "with_content = matched lines+context (default). ranked_map = ranked file "
                "pointers. paths_only = paths. count_only = path + match count. Old "
                "aliases (file_paths_with_content, ...) accepted."
            )
        ),
    ] = "with_content",
    before: Annotated[
        int,
        Field(description="Lines before match."),
    ] = 0,
    after: Annotated[
        int,
        Field(description="Lines after match."),
    ] = 0,
    i: Annotated[
        bool,
        Field(description="Case-insensitive."),
    ] = False,
    type: Annotated[
        str | None,
        Field(description="File-type filter, e.g. `python`."),
    ] = None,
    file_limit: Annotated[
        int | None,
        Field(description="Max files rendered."),
    ] = None,
    lines_per_file: Annotated[
        int | None,
        Field(description="Max matched lines/file (content mode)."),
    ] = 500,
    if_modified_since: Annotated[
        str | None,
        Field(description="Prior result's timestamp; unchanged files are marked/skipped."),
    ] = None,
    multiline: Annotated[
        bool,
        Field(description="Regex spans newlines."),
    ] = False,
    summary: Annotated[
        bool | None,
        Field(description="Omit: auto-summarize large code. `true`: signatures-only. `false`: raw lines."),
    ] = None,
    context_budget_tokens: Annotated[
        int,
        Field(description="Output token budget (default 2000)."),
    ] = 2000,
    include_meta: Annotated[
        bool,
        Field(description="Include file counts and caps."),
    ] = False,
    format: Annotated[Literal["auto", "compact", "json"], _FORMAT_FIELD] = "auto",
) -> dict[str, Any]:
    """Search code by regex/glob/type, with call-graph counts riding along on definition matches.

    Default mode reads matched context inline. Omit `regex` for path/type listings. When a match
    lands on a symbol's definition, its caller/callee/usage counts are appended to that file's
    header (e.g. `orders.py  OrderService  ↳12 callers ↰3 callees ⌖8 usages`) -- no extra call or
    param needed. To expand a non-trivial count into the actual list, use the `relations` tool.
    Returns: results shaped by `mode` (default `content`: matched lines plus context).
    For natural-language/semantic ranking use `search`.
    """
    # Accept a single glob passed as a bare string -- a common shape the model
    # reaches for -- so it does not trip schema validation against the array type.
    if isinstance(glob, str):
        glob = [glob]
    # Short model-facing mode names map to the engine's verbose output_mode.
    native_mode = cast(
        Literal["ranked_file_map", "file_paths_with_content", "file_paths_only", "file_paths_with_match_count"],
        _GREP_MODE_ALIASES.get(_normalize_grep_mode(mode), "file_paths_with_content"),
    )
    # Ride call-graph counts along content-mode regex matches that land on symbol
    # definitions (best-effort; the provider never fails the search).
    badge_provider = _grep_badge_provider if (regex and native_mode == "file_paths_with_content") else None
    payload = _run_native_grep(
        path=path,
        content_regex=regex,
        file_glob_patterns=glob,
        output_mode=native_mode,
        lines_before=before,
        lines_after=after,
        ignore_case=i,
        type=type,
        file_limit=file_limit,
        lines_per_file=lines_per_file,
        if_modified_since=if_modified_since,
        multiline=multiline,
        summary=summary,
        context_budget_tokens=context_budget_tokens,
        include_meta=include_meta,
        badge_provider=badge_provider,
    )
    # Plumb savings via thread-local (read by _extract_tokens_saved) and
    # strip from the LLM-facing payload to keep responses clean.
    ts = int(payload.pop("tokens_saved", 0) or 0)
    if ts > 0:
        _tool_call_tokens_saved.value = ts
    if regex and not payload.get("isError"):
        # Literal grep has no semantic/zoekt channel, so no "dark" verdict -- only
        # found / missed / absent, plus the shared breaker.
        payload = _apply_search_verdict(payload, query=regex, hit_count=_count_grep_hits(payload), channels=None)
    return payload


def _scope_search_matches_to_range(payload: dict[str, Any], line_range: tuple[int, int]) -> None:
    """Restrict ranked-search matches to snippets overlapping [lo, hi].

    A "path:Lx-Ly" search scopes results to that line window. Snippets carry
    line_start/line_end; matches with no overlapping snippet are dropped. Matches
    lacking snippet line data are kept (they cannot be filtered).
    """
    lo, hi = line_range
    matches = payload.get("matches")
    if not isinstance(matches, list):
        return
    kept: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        snippets = match.get("snippets")
        if isinstance(snippets, list) and snippets:
            in_window = [
                snip
                for snip in snippets
                if isinstance(snip, dict)
                and int(snip.get("line_start", 0) or 0) <= hi
                and int(snip.get("line_end", snip.get("line_start", 0)) or 0) >= lo
            ]
            if not in_window:
                continue
            match = {**match, "snippets": in_window}
        kept.append(match)
    payload["matches"] = kept
    payload["match_paths"] = [str(match.get("path")) for match in kept if isinstance(match, dict) and match.get("path")]


@mcp_tool(
    name="search",
    description=(
        "Semantic/embedding code search: relevance-ranked snippets for a natural-language query. "
        "Hidden until an embedding backend is configured; deterministic regex/glob/symbol/map search "
        "lives on `grep`."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language ranked search query.",
            },
            "path": {
                "type": "string",
                "default": ".",
                "description": "Workspace-relative file or directory; a single file may carry ':Lx-Ly' to scope results.",
            },
            "limit": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of ranked files to return.",
            },
        },
        "required": ["query"],
    },
    param_aliases={"max_files": "limit"},
)
def tool_smart_search(
    query: Annotated[
        str | None,
        Field(description="Natural-language ranked search query."),
    ] = None,
    path: Annotated[
        str,
        Field(
            description=(
                "Workspace-relative file or directory to search. A single file may carry a "
                "':Lx-Ly' suffix (e.g. 'store.py:L60-L100') to scope ranked results to that line range."
            ),
        ),
    ] = ".",
    limit: Annotated[
        int,
        Field(description="Maximum number of ranked files to return."),
    ] = 10,
    max_chars_per_file: Annotated[
        int,
        Field(description=("Cap the returned characters per ranked file before the overall token budget is applied.")),
    ] = 2000,
    include_outline: Annotated[
        bool,
        Field(description="Include outline metadata for ranked files when the backend can provide it."),
    ] = True,
    budget_tokens: Annotated[
        int,
        Field(description="Total token budget for ranked search output."),
    ] = 2000,
    include_meta: Annotated[
        bool,
        Field(description="Include backend/cache metadata fields in the response."),
    ] = False,
    format: Annotated[str, _FORMAT_FIELD] = "auto",
) -> dict[str, Any]:
    """Semantic/embedding ranked search over code and docs (hidden until embeddings are wired up).

    Returns relevance-ranked snippets for a natural-language `query`, with read/context
    follow-up handoffs. Deterministic regex/glob/symbol-locate/repo-map search lives on `grep`.
    """
    # A "path:Lx-Ly" suffix scopes ranked results to a line window of one file.
    line_range: tuple[int, int] | None = None
    path, suffix_range = _split_read_range_suffix(path)
    if suffix_range is not None:
        lo_text, _, hi_text = suffix_range.partition("-")
        lo = int(re.sub(r"\D", "", lo_text) or 0)
        hi = int(re.sub(r"\D", "", hi_text) or 0) if hi_text else lo
        if lo:
            line_range = (lo, hi or lo)
    if query is None:
        raise ValueError("query is required for semantic search; use grep for regex/glob/symbol search")
    from lemoncrow.pro.capabilities.grounded_loop.search_first import search_first

    workspace_root = _workspace_root()

    def indexed_search(
        *,
        query: str,
        path: str,
        max_files: int,
        budget_tokens: int,
    ) -> dict[str, Any]:
        requested = Path(path)
        resolved = requested if requested.is_absolute() else workspace_root / requested
        resolved = resolved.resolve()
        file_glob: str | None = None
        if resolved != workspace_root:
            relative = str(resolved.relative_to(workspace_root))
            file_glob = relative if resolved.is_file() else f"{relative}/**"
        return cast(
            dict[str, Any],
            _code_context_engine(str(workspace_root)).tool_search(
                query,
                limit=max(max_files * 4, 20),
                mode="hybrid",
                intent="auto",
                snippet="head",
                snippet_lines=12,
                file_glob=file_glob,
                budget_tokens=budget_tokens,
            ),
        )

    payload = search_first(
        query=query,
        task=query,
        path=path,
        max_files=limit,
        max_chars_per_file=max_chars_per_file,
        include_outline=include_outline,
        budget_tokens=budget_tokens,
        indexed_search=indexed_search,
    )
    if line_range is not None:
        _scope_search_matches_to_range(payload, line_range)
    # Plumb savings via thread-local and strip from the LLM-facing payload.
    ts = int(payload.pop("tokens_saved", 0) or 0)
    if ts > 0:
        _tool_call_tokens_saved.value = ts
    if include_meta:
        return payload
    payload.pop("cache_hit", None)
    payload.pop("backend", None)
    payload.pop("index_age_seconds", None)
    payload.pop("total_tokens", None)
    return payload


@mcp_tool(name="compact")
def tool_compact(
    op: Annotated[
        str,
        Field(
            description=(
                '"compact" (default) = compress the run ledger into a compact session-state '
                'block. "consolidate" (T6) = distill recent findings + prune stale history, '
                "same compaction entrypoint — autonomous lever when context is heavy."
            )
        ),
    ] = "compact",
    session_id: Annotated[
        str | None,
        Field(description="Optional run-ledger session ID override. Usually omit."),
    ] = None,
) -> dict[str, Any]:
    """Compress the full run ledger into a compact session-state block.

    Ops: "compact" (default); "consolidate" = distill + prune via the same
    entrypoint. Spilled tool output → ``read``.
    """
    normalized = (op or "compact").strip().lower()
    # "compact" and "consolidate" share the existing compaction entrypoint
    # (ContextCompressor().compress, via _compress_context) — do NOT reimplement
    # compression here. consolidate distills recent findings AND prunes history;
    # _compress_context already preserves recent turns + decisions and drops
    # stale history, so the lever is the same call surfaced as an agent op.
    result = cast(dict[str, Any], _compress_context(session_id=session_id))
    if normalized == "consolidate":
        result["op"] = "consolidate"
    return result


_TOOL_BROKER_DESCRIPTION = (
    "Deterministic fallback for rare LemonCrow tools hidden by the core profile. "
    "Normal code_search/read/edit/bash/web_fetch calls are already exposed: never "
    "search for those. Use search once for a rare capability, then call its exact name."
)

# Hidden tools (HIDDEN_LLM_TOOLS) reachable by EXACT name via the broker's
# `call` action. `search` still never lists them, so the advertised surface is
# unchanged -- only a caller that already knows the name and arguments (a skill
# that documents the exact call) can reach one. Keep this set tiny.
_BROKER_HIDDEN_CALLABLE = frozenset({"statusline_segment"})


def _tool_broker_handler(args: dict[str, Any]) -> dict[str, Any] | Any:
    """Search or invoke the rare-tool registry without global registration."""
    action = str(args.get("action") or "")
    if action == "search":
        normalized = " ".join(str(args.get("query") or "").lower().split())
        terms = tuple(term for term in re.split(r"\W+", normalized) if term)
        matches: list[tuple[int, str, str]] = []
        for tool_name, spec in TOOLS.items():
            if tool_name in _CORE_MCP_TOOLS:
                continue
            if not _tool_visible_to_llm(tool_name, spec):
                continue
            description = _tool_description(spec)
            haystack = f"{tool_name} {description}".lower()
            score = (100 if normalized == tool_name.lower() else 0) + (
                40 if normalized and normalized in haystack else 0
            )
            score += 5 * sum(1 for term in terms if term in haystack)
            if not normalized or score > 0:
                matches.append((score, tool_name, description[:240]))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return {
            "matches": [{"name": tool_name, "description": description} for _, tool_name, description in matches[:12]]
        }

    if action == "call":
        target = str(args.get("name") or "").strip()
        if not target:
            raise _ToolArgumentError("tool call requires an exact name")
        if target in _CORE_MCP_TOOLS:
            raise _ToolArgumentError(f"{target!r} is already exposed; call it directly")
        call_spec = TOOLS.get(target)
        if call_spec is None or not (_tool_visible_to_llm(target, call_spec) or target in _BROKER_HIDDEN_CALLABLE):
            raise _ToolArgumentError(f"unknown or unavailable tool: {target}")
        arguments = args.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _ToolArgumentError("tool arguments must be an object")
        handler = cast(Callable[[dict[str, Any]], Any], call_spec["handler"])
        return handler(arguments)

    raise _ToolArgumentError(f"unknown broker action: {action}")


_TOOL_BROKER_SPEC: dict[str, Any] = {
    "name": "tool",
    "description": _TOOL_BROKER_DESCRIPTION,
    "handler": _tool_broker_handler,
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "call"]},
            "query": {"type": "string"},
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------- #
# Remote mode & dispatcher                                                    #
# --------------------------------------------------------------------------- #
# Tools that are routed through the remote HTTP service in MCP remote mode.
_REMOTE_TOOLS = frozenset(
    {
        "context",
        "memory",
        "rescue",
        "trace",
        "verify",
    }
)

# Read-only tools for outcome tracking (distinguishes reads from writes).
_READ_TOOLS = frozenset(
    {
        "Read",
        "View",
        "read_file",
        "view",
        "view_range",
        "search_read",
        "grep",
        "glob",
        "cached_grep",
    }
)

# Read-style tools whose byte-identical results may be deduped within a session
# (registered tool names, post-alias). See context_dedup for the mechanism.
_DEDUP_TOOLS = frozenset({"read", "code_search"})


def _defer_edit_hooks() -> bool:
    """Move mutating edit-hooks (format / organize-imports) and contract-site
    re-fires to the Stop hook. Default off. Read fresh each call (never cached, so
    the env can change at runtime and tests can monkeypatch it)."""
    return os.environ.get("LEMONCROW_DEFER_EDIT_HOOKS", "0").strip().lower() in {"1", "true", "on", "yes"}


# Per-session contract-literal sites already surfaced to the agent, so a site it is
# mid-way through fixing does not re-fire on every later edit (noise). Keyed by
# session id; the Stop hook re-checks the final tree for the real omissions.
_CONTRACT_SEEN: dict[str, set[str]] = {}


def _contract_surface_once(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only contract sites not yet surfaced this session; record them."""
    sid = ""
    with contextlib.suppress(Exception):
        sid = _get_ledger().session_id or ""
    seen = _CONTRACT_SEEN.setdefault(sid, set())
    fresh = [s for s in sites if isinstance(s, dict) and s.get("path") not in seen]
    for s in fresh:
        p = s.get("path")
        if isinstance(p, str):
            seen.add(p)
    return fresh


# Flat single-object schema. The Anthropic Messages API rejects a top-level
# oneOf/anyOf/allOf in a tool's input_schema, so Claude Code's MCP client
# silently SKIPS any tool whose schema uses one ("its input schema uses
# top-level oneOf, which the Anthropic API does not accept") -- which is why a
# previous action-branched oneOf shape made `bash` vanish from the tool list
# while every other tool stayed. Per-action requirements (command for run,
# session_id for poll/kill) are enforced in _run_bash_tool, not the schema.
# BASH_TOOL_INPUT_SCHEMA moved to mcp.bash.


# bash symbols moved to mcp.bash (imported/registered near the top).


# tool_web_fetch now lives in mcp.tools_commodity (imported near the top).
# tool_mcp now lives in mcp.tools_commodity (imported near the top).
_remote_client: Any = None


def _get_remote_client() -> Any:
    global _remote_client
    with _STATE_LOCK:
        if _remote_client is None:
            from lemoncrow.gateway.adapters.remote_client import RemoteClient

            _remote_client = RemoteClient()
    return _remote_client


def _dispatch_remote(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if _remote_client is None and not os.environ.get("LEMONCROW_SERVICE_URL"):
        if name == "context":
            # Route through the registered handler so bootstrap job queuing,
            # worker spawn throttle, and session bookkeeping all execute.
            spec = TOOLS.get("context")
            if spec is None:
                raise ValueError("context tool not registered")
            handler = cast(Callable[[dict[str, Any]], dict[str, Any]], spec["handler"])
            return handler(args)
        if name == "rescue":
            rescue_result = _runtime().rescue_failure(
                task=str(args.get("task") or ""),
                error=str(args.get("error") or ""),
                files=cast(list[str], args.get("files") or []),
                recent_actions=cast(list[str], args.get("recent_actions") or []),
                domain=cast(str | None, args.get("domain")),
            )
            return rescue_result.model_dump()
        spec = TOOLS.get(name)
        if spec is None:
            raise ValueError(f"unknown remote tool: {name}")
        handler = cast(Callable[[dict[str, Any]], dict[str, Any]], spec["handler"])
        return handler(args)
    client = _get_remote_client()

    if name == "context":
        context_args = dict(args)
        context_args["files"] = cast(list[str], args.get("files") or [])
        context_args["tools"] = cast(list[str], args.get("tools") or [])
        context_args["errors"] = cast(list[str], args.get("errors") or [])
        return cast(dict[str, Any], client.get_context(context_args))
    if name == "memory":
        return cast(dict[str, Any], client.memory(args))
    if name == "rescue":
        return cast(dict[str, Any], client.rescue_failure(args))
    if name in {"trace", "record"}:
        trace_result = cast(dict[str, Any], client.record_trace(args))
        trace_id = str(trace_result.get("trace_id") or trace_result.get("id") or "")
        event_recorded = bool(trace_result.get("event_recorded"))
        return {"trace_id": trace_id, "event_recorded": event_recorded}
    if name == "verify":
        return cast(dict[str, Any], client.run_rubric_gate(args))
    raise ValueError(f"tool not supported in remote mode: {name}")


# --------------------------------------------------------------------------- #
# MCP Protocol Handling                                                       #
# --------------------------------------------------------------------------- #


def _lever_for_tool(tool_name: str) -> str:
    lowered = tool_name.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered in {"read", "search"} or lowered.endswith("_read") or lowered.endswith("_search"):
        return "search_read"
    if lowered == "edit" or lowered.endswith("_edit"):
        return "batch_edit"
    if lowered == "sql" or lowered.endswith("_sql"):
        return "sql_batch"
    if lowered == "compact" or lowered.endswith("_compact"):
        return "compact_lifecycle"
    if lowered == "memory" or lowered.endswith("_memory"):
        return "scoped_recall"
    if lowered == "context" or lowered.endswith("_context"):
        return "playbook_inject"
    return lowered or "unknown"


def _price_tokens_saved_usd(model: str, tokens_saved: int, *, long_context: bool = False) -> float:
    """Price ``tokens_saved`` at *model*'s INPUT rate. No fallback.

    Saved tokens are bytes LemonCrow kept out of the LLM input — they would
    have been billed as new input tokens at the model in use at that turn,
    at the >200k premium input rate when that request ran long-context.
    If the model is unknown or has no pricing entry, returns 0.0 (no guess).
    """
    if tokens_saved <= 0 or not model or model == "_default":
        return 0.0
    from lemoncrow.core.capabilities.pricing import get_model_pricing

    pricing = get_model_pricing(model)
    if pricing is None or not pricing.known or pricing.input <= 0:
        return 0.0
    return pricing.request_cost_usd(input_tokens=int(tokens_saved), long_context=long_context)


def _classify_read_savings(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
    *,
    tokens_saved: int,
    default_lever: str,
) -> tuple[str, dict[str, Any]]:
    lowered = tool_name.strip().lower().replace("-", "_").replace(" ", "_")
    if lowered not in {"read", "smart_read"}:
        return default_lever, {}

    mode = str(result.get("mode") or "").strip().lower()
    if mode == "outline" and tokens_saved > 0:
        classified = "structure_map"
    elif mode == "range" and tokens_saved > 0:
        classified = "delta_read"
    else:
        classified = default_lever

    path = result.get("path") or args.get("file_path") or args.get("path")
    metadata: dict[str, Any] = {"read_mode": mode or "full"}
    if isinstance(path, str) and path:
        metadata["path"] = path
    range_spec = result.get("range") or args.get("range")
    if isinstance(range_spec, str) and range_spec:
        metadata["range"] = range_spec
    if "cache_hit" in result:
        metadata["cache_hit"] = bool(result.get("cache_hit"))
    return classified, metadata


def _record_context_budget_for_tool(
    tool_name: str,
    args: dict[str, Any],
    led: RunLedger,
    result: dict[str, Any],
    *,
    rendered_text_size: int | None = None,
) -> None:
    try:
        recorder = _get_context_budget_recorder()

        # Model is best-effort for the analytics recorder below; the
        # response-embedded `saved` field carries the per-event truth.
        model = str(getattr(led, "model", "") or os.environ.get("LEMONCROW_MODEL") or "").strip()

        compact_tool_tokens_saved = _extract_compact_output_tokens_saved(result)
        tokens_saved = _extract_tokens_saved(result)
        base_lever = _lever_for_tool(tool_name)
        lever, savings_metadata = _classify_read_savings(
            tool_name,
            args if isinstance(args, dict) else {},
            result,
            tokens_saved=tokens_saved,
            default_lever=base_lever,
        )
        if "cache_hit" in result and "cache_hit" not in savings_metadata:
            savings_metadata["cache_hit"] = bool(result.get("cache_hit"))
        if isinstance(result.get("provenance"), str):
            savings_metadata.setdefault("provenance", str(result["provenance"]))
        op = args.get("op") if isinstance(args, dict) else None
        if isinstance(op, str) and op:
            savings_metadata.setdefault("op", op)

        raw_lever_savings = result.get("tokens_saved")
        lever_savings = raw_lever_savings.copy() if isinstance(raw_lever_savings, dict) else {}
        if compact_tool_tokens_saved > 0 and not lever_savings:
            lever_savings[f"compact_tool_output:{lever}"] = compact_tool_tokens_saved
        elif tokens_saved > 0:
            lever_savings[lever] = max(int(lever_savings.get(lever, 0) or 0), tokens_saved)
        if tool_name:
            lever_savings.setdefault(f"tool:{tool_name}", 0)

        # Lifetime smart-state counters remain useful for cumulative "savings
        # since install" metrics; they're a single integer pair, not a
        # per-event log. Real per-session savings ride the MCP response's
        # content[].saved field into the Claude transcript.
        calls_avoided = _coerce_saved_tokens(result.get("calls_saved"))
        if tokens_saved > 0 or calls_avoided > 0:
            _record_smart_state_savings(tokens_saved=tokens_saved, calls_avoided=calls_avoided)

        actual_output_tokens = int(result.get("total_tokens", 0) or 0)
        if actual_output_tokens <= 0:
            if rendered_text_size is not None:
                actual_output_tokens = max(0, rendered_text_size // 4)
            else:
                actual_output_tokens = max(0, len(json.dumps(result, ensure_ascii=False, default=str)) // 4)

        if compact_tool_tokens_saved > 0 and not isinstance(raw_lever_savings, dict):
            recorder.record_compact_tool_output(
                session_id=led.session_id,
                turn_index=max(0, len(led.events) - 1),
                model=model,
                method=lever,
                tokens_in=actual_output_tokens + compact_tool_tokens_saved,
                tokens_out=actual_output_tokens,
            )
        else:
            recorder.record(
                session_id=led.session_id,
                turn_index=max(0, len(led.events) - 1),
                model=model,
                input_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                output_tokens=actual_output_tokens,
                naive_input_tokens=actual_output_tokens + tokens_saved,
                lever_savings=lever_savings,
                tool_calls=1,
            )
    except Exception:
        logging.exception("Recovered from broad exception handler")
        logger.warning("Suppressed exception while recording context budget", exc_info=True)


_TASK_TEXT_KEYS = ("task", "user_goal", "query", "prompt", "content", "description", "error")


def _task_text_from_args(args: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in _TASK_TEXT_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n".join(parts)


def _workflow_state_from_workspace() -> dict[str, Any]:
    workflow = _read_workspace_session_state().get("workflow")
    return workflow if isinstance(workflow, dict) else {}


def _route_outcome_calibration(tool_name: str, session_state: Mapping[str, Any], led: RunLedger) -> dict[str, Any]:
    from lemoncrow.pro.runtime.outcome_capture import load_outcomes_from_state

    root = led._root
    if root is None:
        return {}
    outcomes = load_outcomes_from_state(outcomes_path(root, led.agent or "claude", led.session_id))
    session_phase = str(session_state.get("session_phase") or "").strip()
    followed: list[float] = []
    unfollowed: list[float] = []
    samples = 0
    for entry in outcomes.get("route_outcomes", []):
        if str(entry.get("tool") or "") != tool_name:
            continue
        scored_state = entry.get("scored_state")
        if not isinstance(scored_state, dict):
            continue
        if session_phase and str(scored_state.get("session_phase") or "") != session_phase:
            continue
        outcome_window = entry.get("outcome_window")
        if not isinstance(outcome_window, dict):
            continue
        raw_score = outcome_window.get("outcome_score")
        if isinstance(raw_score, bool):
            continue
        if isinstance(raw_score, int | float):
            score = float(raw_score)
        elif isinstance(raw_score, str):
            try:
                score = float(raw_score.strip())
            except ValueError:
                continue
        else:
            continue
        samples += 1
        if bool(entry.get("recommendation_followed")):
            followed.append(score)
        else:
            unfollowed.append(score)
    if not followed or not unfollowed:
        return {}
    delta = round(sum(followed) / len(followed) - sum(unfollowed) / len(unfollowed), 4)
    if delta <= 0.0:
        return {"route_outcome_samples": samples}
    return {
        "route_outcome_score_delta": delta,
        "route_outcome_samples": samples,
    }


def _route_enforcement_enabled() -> bool:
    raw = os.environ.get("LEMONCROW_ENFORCE_ROUTE_MODEL")
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def _restore_legacy_route(workflow: dict[str, Any], current_step: str) -> tuple[Any | None, int]:
    from lemoncrow.pro.capabilities.model_routing import ModelRecommendation

    routing = workflow.get("routing")
    if not isinstance(routing, dict):
        return None, 0
    if str(routing.get("step") or "") != current_step:
        return None, 0
    raw = routing.get("recommendation")
    if not isinstance(raw, dict):
        return None, 0
    tier = str(raw.get("tier") or "").strip()
    if tier not in {"cheap", "medium", "expensive"}:
        return None, 0
    typed_tier = cast(Literal["cheap", "medium", "expensive"], tier)
    route_tier = str(raw.get("route_tier") or "")
    if route_tier not in {
        "deterministic",
        "local_slm",
        "cheap_llm",
        "frontier_llm",
        "human_review",
    }:
        route_tier = "frontier_llm" if tier == "expensive" else "cheap_llm"
    typed_route_tier = cast(
        Literal["deterministic", "local_slm", "cheap_llm", "frontier_llm", "human_review"],
        route_tier,
    )
    baseline_tier_raw = str(raw.get("baseline_tier") or "").strip()
    baseline_tier = (
        cast(Literal["cheap", "medium", "expensive"], baseline_tier_raw)
        if baseline_tier_raw in {"cheap", "medium", "expensive"}
        else None
    )
    return (
        ModelRecommendation(
            tier=typed_tier,
            route_tier=typed_route_tier,
            model=str(raw.get("model") or ""),
            reasons=[str(reason) for reason in raw.get("reasons") or []],
            score=int(raw.get("score") or 0),
            cache_affinity_model=str(raw.get("cache_affinity_model") or "") or None,
            cache_cost_usd=float(raw.get("cache_cost_usd") or 0.0),
            quality_gain_usd_estimated=float(raw.get("quality_gain_usd_estimated") or 0.0),
            decision=str(raw.get("decision") or "baseline"),
            baseline_tier=baseline_tier,
            sticky_until_tool_calls=int(raw.get("sticky_until_tool_calls") or 0),
        ),
        max(0, int(routing.get("remaining_tool_calls") or 0)),
    )


def _persist_legacy_route(workflow: dict[str, Any], payload: dict[str, Any], current_step: str) -> None:
    if not current_step:
        return
    tier = str(payload.get("tier") or "").strip()
    model = str(payload.get("model") or "").strip()
    if tier not in {"cheap", "medium", "expensive"} or not model:
        return
    sticky_window = max(0, int(workflow.get("sticky_window") or 0))
    remaining = max(0, int(payload.get("sticky_until_tool_calls") or 0))
    if str(payload.get("decision") or "baseline") != "sticky":
        remaining = sticky_window
    workflow["routing"] = {
        "step": current_step,
        "remaining_tool_calls": remaining,
        "recommendation": {
            "tier": tier,
            "route_tier": payload.get("route_tier"),
            "model": model,
            "reasons": list(payload.get("reasons") or []),
            "score": int(payload.get("score") or 0),
            "cache_affinity_model": payload.get("cache_affinity_model"),
            "cache_cost_usd": float(payload.get("cache_cost_usd") or 0.0),
            "quality_gain_usd_estimated": float(payload.get("quality_gain_usd_estimated") or 0.0),
            "decision": str(payload.get("decision") or "baseline"),
            "baseline_tier": payload.get("baseline_tier"),
            "sticky_until_tool_calls": remaining,
        },
    }
    routing_entry = workflow["routing"]
    # Hold _STATE_LOCK across the read-modify-write so a concurrent handler's
    # session_state update is not lost. _STATE_LOCK is an RLock. Merge only the
    # 'routing' key onto the lock-fresh workflow rather than overwriting the whole
    # sub-dict with the pre-lock snapshot, which would clobber sibling keys
    # (current_task/current_step/task_outputs) a concurrent step just wrote.
    with _STATE_LOCK:
        state = _read_workspace_session_state()
        fresh_workflow = state.get("workflow")
        fresh_workflow = fresh_workflow if isinstance(fresh_workflow, dict) else {}
        fresh_workflow["routing"] = routing_entry
        state["workflow"] = fresh_workflow
        _write_workspace_session_state(state)


def _prepare_model_recommendation(
    tool_name: str,
    args: dict[str, Any],
    led: RunLedger,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    from lemoncrow.core.capabilities.pricing import get_model_pricing
    from lemoncrow.pro.capabilities.cross_vendor_routing.configuration import RouteConfigError
    from lemoncrow.pro.capabilities.cross_vendor_routing.router import NoFeasibleRouteError
    from lemoncrow.pro.capabilities.model_routing import ModelRouter

    session_state = _model_recommendation_state(led, args)
    session_state.update(_route_outcome_calibration(tool_name, session_state, led))
    workflow = _workflow_state_from_workspace()
    current_step = str(session_state.get("workflow_step") or "")
    prior_route, stickiness_remaining = _restore_legacy_route(workflow, current_step)
    estimated_input_tokens = max(1_000, int(session_state.get("expected_input_tokens") or 0))
    try:
        decision = _select_owned_execution_route(
            tool_name=tool_name,
            task_text=_task_text_from_args(args),
            mode="auto",
            provider="",
            model="",
            runner="",
            session_state=session_state,
        )
        led.record("route_decision", f"{decision.mode} route for {tool_name}", decision.to_dict())
        actual_model = str(getattr(led, "model", "") or os.environ.get("LEMONCROW_MODEL") or "").strip()
        actual_vendor = _provider_for_model(actual_model)
        recommendation = {
            **decision.to_dict(),
            "vendor": decision.provider,
            "actual_model": actual_model,
            "actual_vendor": actual_vendor,
            "recommendation_followed": _normalize_model_id(actual_model) == _normalize_model_id(decision.model),
        }
        vs_model = actual_model or "auto"
        cost_saved_usd = 0.0
        if recommendation["model"] != vs_model and vs_model != "auto":
            expensive_pricing = get_model_pricing(vs_model)
            recommended_pricing = get_model_pricing(recommendation["model"])
            cost_saved_usd = max(
                0.0,
                expensive_pricing.cost_usd(input_tokens=estimated_input_tokens)
                - recommended_pricing.cost_usd(input_tokens=estimated_input_tokens),
            )
        payload = {
            "at": datetime.now(UTC).isoformat(),
            "kind": "model_recommendation",
            "lever": "model_routing",
            "session_id": led.session_id,
            "agent": led.agent or _detect_agent(),
            "tool_name": tool_name,
            "tokens_saved": 0,
            "cost_saved_usd": round(cost_saved_usd, 6),
            "vs_model": vs_model,
            "estimated_input_tokens": estimated_input_tokens,
            "configured": True,
            **recommendation,
        }
    except (RouteConfigError, NoFeasibleRouteError) as exc:

        def _record_route_decision(route_payload: dict[str, Any]) -> None:
            led.record(
                "route_decision",
                f"{route_payload.get('decision', 'baseline')} route for {tool_name}",
                route_payload,
            )

        legacy = ModelRouter().recommend(
            tool_name,
            _task_text_from_args(args),
            session_state,
            prior_route=prior_route,
            stickiness_remaining=stickiness_remaining,
            route_decision_sink=_record_route_decision,
        )
        if legacy is None:
            raise NoFeasibleRouteError("bench-off") from None
        vs_model = "auto"
        # The legacy path has no real baseline model to price against ("auto" is
        # unknown to the pricing table -> input cost 0.0), so the old comparison
        # produced a structural always-0.0 saving. Keep it explicit instead of
        # computing a misleading zero. (The owned path above does real cost math.)
        cost_saved_usd = 0.0
        payload = {
            "at": datetime.now(UTC).isoformat(),
            "kind": "model_recommendation",
            "lever": "model_routing",
            "session_id": led.session_id,
            "agent": led.agent or _detect_agent(),
            "tool_name": tool_name,
            "tokens_saved": 0,
            "configured": False,
            "cost_saved_usd": round(cost_saved_usd, 6),
            "estimated_input_tokens": estimated_input_tokens,
            "vs_model": vs_model,
            "error": str(exc),
            **legacy.to_dict(),
        }
    return payload, session_state, workflow, current_step


def _finalize_model_recommendation(
    payload: dict[str, Any],
    *,
    led: RunLedger,
    tool_name: str,
    session_state: Mapping[str, Any],
    workflow: dict[str, Any],
    current_step: str,
    wrapper_applied: bool = False,
    wrapper_model: str | None = None,
) -> dict[str, Any]:
    finalized = dict(payload)
    finalized["route_enforcement_active"] = _route_enforcement_enabled() and finalized.get("configured") is not False
    finalized["wrapper_applied"] = wrapper_applied
    if wrapper_model:
        finalized["wrapper_model"] = wrapper_model
        finalized["executed_model_scope"] = "local_mcp_only"
    if wrapper_applied:
        finalized["recommendation_followed"] = True
    led.record(
        "model_recommendation",
        f"recommend {finalized.get('model', 'unconfigured')} for {tool_name}",
        finalized,
    )
    if finalized.get("recommendation_followed") or float(finalized.get("cost_saved_usd") or 0.0) > 0:
        _append_live_savings_event(finalized)
    else:
        # Unfollowed zero-saving recommendation: keep the advisor-countable core
        # fields, drop the bulky static provider metadata (~80% of the payload).
        _append_live_savings_event(
            {
                key: finalized[key]
                for key in (
                    "at",
                    "kind",
                    "lever",
                    "session_id",
                    "agent",
                    "tool_name",
                    "tokens_saved",
                    "cost_saved_usd",
                    "configured",
                    "model",
                    "vs_model",
                    "tier",
                    "recommendation_followed",
                )
                if key in finalized
            }
        )
    # Mirror the routing saving into the per-session sidecar so the live
    # statusline / stop hook can show it cheaply: sessions/<id>/savings.jsonl is
    # read per render, while live_savings_events.jsonl is too large to scan there.
    # kind="routing" keeps it out of the context-savings tally (never double-counted).
    _routing_usd = float(finalized.get("cost_saved_usd") or 0.0)
    if _routing_usd > 0:
        with contextlib.suppress(Exception):
            _sidecar = _get_host_session_sidecar_path()
            _sidecar.parent.mkdir(parents=True, exist_ok=True)
            _routing_row = {
                "kind": "routing",
                "usd": round(_routing_usd, 6),
                "tool": tool_name,
                "model": str(finalized.get("model") or ""),
                "ts": str(finalized.get("at") or ""),
            }
            with _sidecar.open("a", encoding="utf-8") as _fh:
                _fh.write(json.dumps(_routing_row) + "\n")
            # Fold into the in-memory 1/7/30d window totals (O(1)) so the
            # breakdown reflects routing without waiting for a full re-scan.
            with contextlib.suppress(ImportError):
                from lemoncrow.core.capabilities.savings_summary import (
                    _bump_historical_savings_cache as _bump_routing_cache,
                )

                _bump_routing_cache(_routing_row)
    _persist_legacy_route(workflow, finalized, current_step)

    if finalized.get("configured") is not False:
        from lemoncrow.pro.runtime import outcome_capture

        outcome_capture.schedule_route(
            session_id=led.session_id,
            tool=tool_name,
            recommended_vendor=str(finalized.get("vendor") or ""),
            recommended_tier=str(finalized.get("tier") or ""),
            recommended_model=str(finalized.get("model") or ""),
            actual_vendor=str(finalized.get("actual_vendor") or ""),
            actual_model=str(finalized.get("actual_model") or ""),
            recommendation_followed=bool(finalized.get("recommendation_followed")),
            applied_lessons=[str(item) for item in finalized.get("applied_lessons") or []],
            cost_cap_triggered=bool(finalized.get("cost_cap_triggered")),
            cost_cap_limit_usd_per_session=(
                float(finalized["cost_cap_limit_usd_per_session"])
                if finalized.get("cost_cap_limit_usd_per_session") is not None
                else None
            ),
            scored_state={
                "turn_number": int(session_state.get("turn_number") or 0),
                "prior_errors": len(led.errors_seen) + len(led.repeated_failures),
                "session_phase": str(session_state.get("session_phase") or "explore"),
                "workflow_step": str(session_state.get("workflow_step") or ""),
            },
            writer=_make_outcome_writer(led),
        )

    return finalized


def _latest_cache_affinity_model(led: RunLedger) -> str | None:
    for event in reversed(led.events):
        payload = event.payload
        raw_cache_write_tokens = (
            payload.get("cache_write_tokens")
            or payload.get("cache_creation_input_tokens")
            or payload.get("cache_creation_tokens")
            or 0
        )
        try:
            cache_write_tokens = int(raw_cache_write_tokens)
        except (TypeError, ValueError):
            cache_write_tokens = 0
        model = str(payload.get("model") or "").strip()
        if cache_write_tokens > 0 and model:
            return model
    return None


def _estimate_compacted_state_tokens(state: Any) -> int:
    prompt_block = state.to_prompt_block()
    preserved_chars = len(prompt_block) + sum(len(turn) for turn in state.recent_turns)
    return max(0, preserved_chars // 4)


def _session_compaction_savings_payload(
    led: RunLedger,
    state: Any,
    *,
    tokens_before: int,
    trigger: str,
    reason: str,
    utilisation_pct: float | None = None,
) -> dict[str, Any]:
    tokens_after_estimate = _estimate_compacted_state_tokens(state)
    tokens_freed = max(0, int(tokens_before) - tokens_after_estimate)
    model = (
        _latest_cache_affinity_model(led)
        or str(getattr(led, "model", "") or "").strip()
        or os.environ.get("LEMONCROW_MODEL", "")
    ).strip()
    # A window already past the >200k threshold was billing premium rates on
    # every request — the tokens this compaction freed are worth the same.
    long_ctx = _savings_long_context(model, int(tokens_before))
    cost_saved_usd = round(_price_tokens_saved_usd(model, tokens_freed, long_context=long_ctx), 6)
    utilisation = (
        round(float(utilisation_pct), 1)
        if utilisation_pct is not None
        else round(100.0 * max(0, int(tokens_before)) / CONTEXT_WINDOW_TOKENS, 1)
    )
    return {
        "at": datetime.now(UTC).isoformat(),
        "kind": "session_compaction",
        "lever": "session_compaction",
        "session_id": led.session_id,
        "agent": led.agent or _detect_agent(),
        "model": model,
        "trigger": trigger,
        "reason": reason,
        "tokens_saved": tokens_freed,
        "tokens_freed": tokens_freed,
        "cost_saved_usd": cost_saved_usd,
        "tokens_before": max(0, int(tokens_before)),
        "tokens_after_estimate": tokens_after_estimate,
        "utilisation_pct": utilisation,
    }


def _emit_model_recommendation(tool_name: str, args: dict[str, Any], led: RunLedger) -> dict[str, Any]:
    payload, session_state, workflow, current_step = _prepare_model_recommendation(tool_name, args, led)
    return _finalize_model_recommendation(
        payload,
        led=led,
        tool_name=tool_name,
        session_state=session_state,
        workflow=workflow,
        current_step=current_step,
    )


def _model_recommendation_state(led: RunLedger, args: dict[str, Any]) -> dict[str, Any]:
    tool_call_events = [e for e in led.events if e.kind == "tool_call"]
    recent_tool_calls = [e.payload.get("tool", "") for e in tool_call_events[-10:]]
    turn_number = len(tool_call_events)
    workflow = _workflow_state_from_workspace()
    session_state: dict[str, Any] = {
        "prior_errors": len(led.errors_seen) + len(led.repeated_failures),
        "cache_affinity_model": _latest_cache_affinity_model(led),
        "turn_number": turn_number,
        "recent_tool_calls": recent_tool_calls,
        "session_cost_usd": round(
            sum(
                float((event.payload or {}).get("cost_usd") or 0.0)
                for event in led.events
                if event.kind == "tool_call" and (event.payload or {}).get("kind") == "llm_call"
            ),
            6,
        ),
    }
    workflow_step = str(workflow.get("current_step") or workflow.get("workflow_step") or "").strip()
    if workflow_step:
        session_state["workflow_step"] = workflow_step
    session_phase = str(workflow.get("session_phase") or "").strip()
    if session_phase:
        session_state["session_phase"] = session_phase
    if "max_output_tokens" in args:
        session_state["max_output_tokens"] = args["max_output_tokens"]
    if "budget_tokens" in args:
        session_state["max_output_tokens"] = args["budget_tokens"]
    expected_input_tokens = max(1_000, int(led.token_count or 0) // max(1, _ledger_turn_count(led)))
    session_state["expected_input_tokens"] = expected_input_tokens
    session_state.setdefault("expected_output_tokens", max(1, int(expected_input_tokens * 0.2)))
    return session_state


def _handle(request: dict[str, Any]) -> dict[str, Any] | _Deferred | None:
    rid = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        _emit_mcp_session_start()
        return _ok(
            rid,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {
                    "name": SERVER_DISPLAY_NAME,
                    "title": "LemonCrow",
                    "version": SERVER_VERSION,
                    "description": "Indexed code search, bounded file tools, editing, and verification for coding agents.",
                    "icons": [icon_metadata()],
                },
                "capabilities": {"tools": {}},
                # Read automatically by MCP clients and folded into the host
                # system prompt — the steering surface that reaches every host
                # and subagent, installed personas or not.
                "instructions": "" if _savings_dormant() else SERVER_INSTRUCTIONS,
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        if _savings_dormant():
            # Layer 2 (server-enforced): the savings cap is exhausted -> advertise
            # NO tools. The model never sees them, so there is nothing to call and
            # no user-editable settings file can re-expose them. The session
            # degrades to the host's built-in tools (the lemoncrow:free agent).
            return _ok(rid, {"tools": []})
        tools = [
            {
                "name": n,
                "description": _tool_description(s),
                "inputSchema": s.get("inputSchema", {}),
            }
            for n, s in sorted(TOOLS.items())
            if _tool_profile_exposes(n) and _tool_visible_to_llm(n, s)
        ]
        if _mcp_tool_profile() == "core":
            broker = _TOOL_BROKER_SPEC
            tools.append(
                {
                    "name": "tool",
                    "description": broker["description"],
                    "inputSchema": broker["inputSchema"],
                }
            )
        tools.sort(key=lambda item: str(item["name"]))
        return _ok(rid, {"tools": tools})

    if method == "tools/call":
        name = params.get("name") or ""
        if name == "run":
            name = "bash"
        if _savings_dormant() and (name in TOOLS or name == "tool"):
            # A stale client may still call a formerly advertised tool. Recheck
            # live and reject it rather than grandfathering the connection.
            return _err(
                rid,
                -32601,
                "tool unavailable: LemonCrow anonymous savings cap reached — run `lc account login` "
                "to unlock uncapped Free",
            )
        _tool_call_images.value = []
        args = params.get("arguments") or {}
        # Some MCP clients deliver the whole `arguments` payload as a JSON string
        # instead of an object. mypyc-compiled handlers enforce dict at the boundary
        # and would reject it with "dict object expected; got str", so parse it here.
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
                args = json.loads(args)
        if not isinstance(args, dict):
            args = {}
        spec = _TOOL_BROKER_SPEC if name == "tool" and _mcp_tool_profile() == "core" else TOOLS.get(name)
        if spec is None:
            return _err(rid, -32601, f"unknown tool: {name}")
        if name == "memory" and isinstance(args, dict):
            properties = spec.get("inputSchema", {}).get("properties", {})
            allowed_args = set(properties) if isinstance(properties, dict) else set()
            # Accept declared backward-compat aliases (e.g. agent_id -> agent);
            # the handler remaps them to the current name before validation.
            allowed_args |= set(spec.get("param_aliases", {}) or {})
            unknown_args = sorted(set(args) - allowed_args)
            if unknown_args:
                return _err(
                    rid,
                    -32602,
                    f"unknown arguments for memory tool: {', '.join(unknown_args)}",
                )

        remote_routed = name in _REMOTE_TOOLS
        # mode="symbols" must always run locally (code-intel engine); bypass remote routing
        if name == "context" and isinstance(args, dict) and args.get("mode") == "symbols":
            remote_routed = False
        # N10 — request-scoped project isolation. Honor an Mcp-Project-Path-style
        # override for the lifetime of this request only; absent -> unchanged.
        _prior_project = _set_request_project(_extract_request_project(params, args if isinstance(args, dict) else {}))
        # A bash cwd is the only place the session's real working directory
        # reaches this long-lived process; record it so edit can resolve
        # relative paths against a worktree the session has entered.
        _record_session_cwd(name, args)
        _call_duration_ms: int = 0
        _call_started = time.perf_counter()

        def _finalize_error_response(exc: Exception) -> dict[str, Any]:
            logging.exception("Recovered from broad exception handler")
            if not remote_routed:
                with contextlib.suppress(Exception):
                    from lemoncrow.pro.runtime import outcome_capture

                    led = _get_ledger()
                    outcome_capture.advance(
                        led.session_id,
                        tool_name=name,
                        is_error=True,
                        is_env_error=isinstance(exc, (OSError, IOError)),
                        writer=_make_outcome_writer(led),
                    )
                _err_session_id = getattr(_get_ledger(), "session_id", "") or ""
                with contextlib.suppress(Exception):
                    _append_live_savings_event(
                        {
                            "kind": "tool_call",
                            "tool": name,
                            "status": "error",
                            "error": type(exc).__name__,
                            "duration_ms": _call_duration_ms,
                            "session_id": _err_session_id,
                            "ts": time.time(),
                        }
                    )
                with contextlib.suppress(Exception):
                    _append_mcp_debug_event(
                        tool=name,
                        args=args if isinstance(args, dict) else {},
                        duration_ms=_call_duration_ms,
                        response_size=0,
                        status="error",
                        error=type(exc).__name__,
                        session_id=_err_session_id,
                        rid=str(rid) if rid is not None else None,
                    )
                with contextlib.suppress(Exception):
                    _append_tool_profile(
                        tool=name,
                        handler_ms=_call_duration_ms,
                        total_ms=round((time.perf_counter() - _call_started) * 1000),
                        response_size=0,
                        status="error",
                        error=type(exc).__name__,
                        session_id=_err_session_id,
                    )
                with contextlib.suppress(Exception):
                    from lemoncrow.gateway.integrations.langfuse import emit_tool_call as _lf_emit_tool

                    _lf_emit_tool(
                        tool=name,
                        args=_scrub_args_for_debug(args) if isinstance(args, dict) else {},
                        duration_ms=_call_duration_ms,
                        response_size=0,
                        status="error",
                        error=type(exc).__name__,
                        session_id=_err_session_id,
                    )
            locked = _feature_locked_response(rid, exc)
            if locked is not None:
                return locked
            # Protocol faults keep JSON-RPC error codes: malformed/unknown
            # arguments (-32602 here; unknown tool/method are rejected before
            # dispatch) and the memory sidecar's deliberate 409/503 mapping.
            if isinstance(exc, (_ToolArgumentError, ValidationError)):
                return _err(rid, -32602, str(exc))
            if isinstance(exc, (MemoryConcurrencyError, MemorySidecarUnavailable)):
                return _err(rid, _tool_error_code(exc), str(exc))
            # Per the MCP spec, a tool-EXECUTION failure is a successful JSON-RPC
            # response whose result carries isError=true and the message as a
            # text content block, so the calling model can see it and self-correct.
            return _ok(
                rid,
                {
                    "content": [{"type": "text", "text": str(exc) or type(exc).__name__}],
                    "isError": True,
                },
            )

        def _finalize_response(result: dict[str, Any] | Any) -> dict[str, Any]:
            # Post-handler finalization pipeline. Runs synchronously on the worker
            # for the non-deferred path, and on bash_exec's watcher thread for a
            # deferred foreground bash command. Owns its own error handling so the
            # watcher continuation still emits a proper JSON-RPC error on failure.
            try:
                rendered_text: str | None = None
                _loop_note: str | None = None
                _args = args if isinstance(args, dict) else {}
                _calls_saved_credit = 0
                if not remote_routed:
                    if isinstance(result, dict):
                        result = _clean_tool_result(result, name)

                    # Compute MD text for read-heavy tools
                    rendered_text = render_tool_result_text(name, result)

                    # Pull the internal `calls_saved` credit out of `result` NOW,
                    # before the JSON-dump fallback below can serialize it verbatim.
                    # `render_tool_result_text` has already used it (e.g. edit's
                    # silent-success check); anything that falls through to
                    # `json.dumps(result, ...)` because rendered_text is falsy must
                    # not still be carrying this model-invisible bookkeeping key.
                    if isinstance(result, dict) and "calls_saved" in result:
                        _calls_saved_credit = _coerce_saved_tokens(result.pop("calls_saved", None))

                    # Spiral nudge: surface a soft note when the agent repeats an
                    # identical tool call -- a narrow, false-positive-free
                    # no-progress signal. Fail-open; never blocks.
                    if _loop_review_enabled():
                        with contextlib.suppress(Exception):
                            _loop_note = _loop_nudge_for_call(name, _args)
                            if _loop_note and isinstance(result, dict):
                                # Carry the note as a field for JSON consumers; the
                                # response_text below appends it for the rendered view.
                                result.setdefault("loop_note", _loop_note)

                    # Per-call savings accounting (read baseline de-dup + deferred
                    # code-intel credit). Runs BEFORE budget recording so a zeroed
                    # read saving flows into both the recorder and the `saved` field.
                    # Local-handler path only; never touches the response bytes.
                    _process_tool_accounting(name, _args, result, rid)

                    _record_context_budget_for_tool(
                        name,
                        _args,
                        led,
                        result if isinstance(result, dict) else {"result": result},
                        rendered_text_size=len(rendered_text) if rendered_text else None,
                    )

                    with contextlib.suppress(Exception):
                        from lemoncrow.pro.runtime import outcome_capture

                        outcome_capture.advance(
                            led.session_id,
                            tool_name=name,
                            is_error=False,
                            is_read_tool=name in _READ_TOOLS,
                            writer=_make_outcome_writer(led),
                        )

                    _ok_session_id = getattr(_get_ledger(), "session_id", "") or ""
                    with contextlib.suppress(Exception):
                        _append_live_savings_event(
                            {
                                "kind": "tool_call",
                                "tool": name,
                                "status": "ok",
                                "duration_ms": _call_duration_ms,
                                "session_id": _ok_session_id,
                                "ts": time.time(),
                            }
                        )
                    # Refresh the statusline frames sidecar on EVERY dispatch
                    # (not only savings-bearing calls) so the shell render's
                    # 10s freshness gate keeps passing during active work and
                    # the slow `lc savings --segment` subprocess fallback
                    # never fires. Rate-limited internally to one write per 5s.
                    with contextlib.suppress(Exception):
                        _write_statusline_sidecar()

                response_text: str
                if rendered_text:
                    response_text = rendered_text
                elif isinstance(result, str):
                    response_text = result
                else:
                    response_text = json.dumps(result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

                # Spiral nudge: append once to the assembled body so it survives
                # whichever render path produced response_text (rendered text, raw
                # string, or JSON). Soft signal -- never replaces the result.
                if _loop_note and _loop_note not in response_text:
                    response_text = f"{response_text}\n{_loop_note}"

                # Only pay the full-payload UTF-8 encode when a telemetry sink will
                # consume the byte count; otherwise approximate with the O(1) char len.
                if _mcp_debug_enabled():
                    _ok_response_size = len(response_text.encode("utf-8", errors="replace"))
                else:
                    _ok_response_size = len(response_text)
                _ok_sid = _ok_session_id if not remote_routed else (getattr(_get_ledger(), "session_id", "") or "")
                with contextlib.suppress(Exception):
                    _append_mcp_debug_event(
                        tool=name,
                        args=args if isinstance(args, dict) else {},
                        duration_ms=_call_duration_ms,
                        response_size=_ok_response_size,
                        status="ok",
                        session_id=_ok_sid,
                        rid=str(rid) if rid is not None else None,
                    )
                with contextlib.suppress(Exception):
                    from lemoncrow.gateway.integrations.langfuse import emit_tool_call as _lf_emit_tool

                    _lf_emit_tool(
                        tool=name,
                        args=_scrub_args_for_debug(args) if isinstance(args, dict) else {},
                        duration_ms=_call_duration_ms,
                        response_size=_ok_response_size,
                        status="ok",
                        session_id=_ok_sid,
                    )

                # G13 — caller-selectable output encoding (auto | compact | json).
                # Default `auto` returns response_text unchanged (byte-compatible);
                # `json` forces raw JSON; `compact` applies the N6-gated N7 columnar
                # form. Reads the selector from the always-defined request args (the
                # remote path never sets `_args`), with an explicit, non-default
                # value so today's default bytes are untouched.
                _format_saved = 0
                _fmt = args.get("format") if isinstance(args, dict) else None
                if isinstance(_fmt, str) and _fmt.strip().lower() in {"compact", "json"}:
                    with contextlib.suppress(Exception):
                        from lemoncrow.pro.capabilities.tool_supervision.output_format import apply_output_format

                        _pre_format_chars = len(response_text)
                        response_text, _ = apply_output_format(
                            fmt=_fmt,
                            result=result,
                            rendered_text=response_text,
                        )
                        _format_saved = _trimmed_tokens_saved(_pre_format_chars, len(response_text))

                # Within-session content dedup: if this read-style result is
                # byte-identical to one already returned this session (and the model
                # didn't pass force=true), return a small pointer instead of
                # re-paying input/cache cost to re-emit the same bytes. Reset on
                # compaction via context_dedup's epoch. Kill switch: LEMONCROW_CONTEXT_DEDUP=0.
                dedup_stubbed = False
                if name in _DEDUP_TOOLS and os.environ.get("LEMONCROW_CONTEXT_DEDUP", "1") != "0":
                    with contextlib.suppress(Exception):
                        from lemoncrow.pro.capabilities import context_dedup as _cdedup

                        _dedup_sid = ""
                        with contextlib.suppress(Exception):
                            _dedup_sid = _get_ledger().session_id or ""
                        _dedup_salt = ""
                        if name == "read":
                            with contextlib.suppress(Exception):
                                _dedup_salt = _read_dedup_resource(_args)
                        _dedup_outcome = _cdedup.registry().stub_for(
                            session_id=_dedup_sid,
                            content=response_text,
                            epoch=_cdedup.current_epoch(),
                            force=bool(_args.get("force")),
                            salt=_dedup_salt,
                        )
                        if _dedup_outcome is None and name == "read":
                            _dedup_resource = _read_dedup_resource(_args)
                            if _dedup_resource:
                                _dedup_outcome = _cdedup.registry().delta_for(
                                    session_id=_dedup_sid,
                                    resource=_dedup_resource,
                                    content=response_text,
                                    epoch=_cdedup.current_epoch(),
                                    force=bool(_args.get("force")),
                                )
                        if _dedup_outcome is not None:
                            stub_text, dedup_chars_saved = _dedup_outcome
                            response_text = stub_text
                            dedup_stubbed = True
                            if dedup_chars_saved > 0:
                                _append_workspace_savings(name, dedup_chars_saved // 4, 0, rid=str(rid))
                # Embed per-call savings on the content item so they also ride into
                # the Claude transcript JSONL. NOTE: this is a secondary record —
                # the live statusline/analytics source is the per-session sidecar
                # sessions/<id>/savings.jsonl written by _append_workspace_savings
                # below, not the transcript.
                # Shape: {"tokens": int, "calls": int}. Either may be 0 but the
                # object is omitted entirely when both are 0.
                # First bound, for context hygiene: head+tail compact a single
                # runaway result so it can't flood the host prompt (which the host
                # re-pays for on every later turn). The legacy char-compaction path
                # (_compact_result_text) is deterministic and prefix-cache stable
                # across identical calls; the T7 spill summary below is intentionally
                # NOT -- it embeds a unique filename (timestamp+random), so two identical
                # capped-tool calls yield different host text (spill summaries are
                # <4096 chars and so never get the cache_control marker below anyway).
                # T8 (LEMONCROW_AUTO_COMPACT_OUTPUT, default off): auto-compact an
                # oversized result (AST-aware for code) while preserving the
                # untransformed original in the T7 spill store so it stays
                # reversible. Off -> returns response_text unchanged.
                _spill_args = args if isinstance(args, dict) else {}
                # Measure dispatcher-level trimming (T8 auto-compact, T7 spill,
                # legacy compaction, wire truncation) across the whole chain so
                # the elided bytes are credited as savings below -- they are
                # exactly the "kept out of the host prompt" tokens this
                # accounting exists for.
                _pre_trim_chars = len(response_text)
                response_text = _auto_compact_result_text(response_text, name, _spill_args)
                # T7 (LEMONCROW_TOOL_OUTPUT_SPILL, default on): spill the FULL,
                # UNTRANSFORMED payload BEFORE the legacy char compaction below, gated
                # at the recoverable result cap. This is deliberately separate from
                # the legacy lossy compaction threshold: capped tools return at most
                # 2 KiB by default (full original recoverable via `compact`/`read`),
                # while unsupported tools retain their prior context-hygiene budget.
                # `read` is exempt (see _SPILL_CHAR_CAP_TOOLS): it is the explicit,
                # incremental retrieval surface, so it stays full here and only the
                # multi-MB wire backstop below can spill it.
                # Off / non-capped tools -> no-op, so _compact_result_text runs
                # exactly as before.
                _eff_spill_tool = _effective_spill_tool(name, _spill_args)
                response_text = _spill_oversized_result_text(
                    response_text,
                    _eff_spill_tool,
                    _spill_args,
                    _spill_result_chars(_eff_spill_tool),
                    unit="chars",
                    tools=_SPILL_CHAR_CAP_TOOLS,
                )
                response_text = _compact_result_text(response_text, name)
                # Wire-byte backstop: a spill-worthy result still over the multi-MB
                # frame ceiling after compaction is spilled rather than truncated.
                # When the char-gated spill above already fired, response_text is now
                # a small summary, so this no-ops.
                response_text = _spill_oversized_result_text(response_text, name, _spill_args, _max_result_bytes())
                # Bound the result so one oversized frame can't trip the host's
                # stdout guard and disconnect the server (no mid-session reconnect).
                response_text = _truncate_result_text(response_text, _max_result_bytes(), name)
                _trim_saved = _trimmed_tokens_saved(_pre_trim_chars, len(response_text))
                # N4 — per-tool exact input/output token ledger. Runs HERE, after the
                # spill/compact/truncate bounds above, so output is measured against
                # the FINAL emitted text the host actually receives (a spilled summary,
                # not the pre-spill payload). Additive only -- never touches the
                # response bytes; best-effort so a write failure can't break the call.
                with contextlib.suppress(Exception):
                    from lemoncrow.core.capabilities.tool_token_ledger import record_tool_tokens

                    record_tool_tokens(
                        _lemoncrow_root(),
                        name,
                        input_payload=args,
                        output_payload=response_text,
                    )
                content_item: dict[str, Any] = {
                    "type": "text",
                    "text": response_text,
                }
                # Best-effort cache hint, NOT a measured saving. Tag large results
                # so a host that honors MCP cache_control can checkpoint them for
                # prompt caching. Caveats kept honest on purpose: (1) we do not
                # verify the host actually forwards this; (2) the conversation
                # prefix is already auto-cached by the host, so the marginal gain
                # is small; (3) a cache *write* costs ~25% over input, so a one-off
                # large result that is never re-read pays the write premium for
                # nothing. The ≥4096-char floor (~1024 tokens) is Anthropic's
                # minimum cacheable size.
                # ...and skip it entirely for dedup-eligible tools (read): an exact
                # re-read is elided by the dedup pass below, so the marker can never
                # earn its cache-read payoff there and only risks a redundant
                # breakpoint on top of the host's automatic prefix caching.
                if len(response_text) >= 4096 and name not in _DEDUP_TOOLS:
                    content_item["cache_control"] = {"type": "ephemeral"}
                # When deduped, skip the original per-call savings (they'd otherwise be
                # credited against bytes we just elided).
                if not dedup_stubbed and isinstance(result, dict):
                    saved_tokens = _extract_tokens_saved(result) + _format_saved + _trim_saved
                    saved_calls = _calls_saved_credit
                    if saved_tokens > 0 or saved_calls > 0:
                        content_item["saved"] = {
                            "tokens": int(saved_tokens),
                            "calls": int(saved_calls),
                        }
                        _append_workspace_savings(name, saved_tokens, saved_calls, rid=str(rid))

                response_payload: dict[str, Any] = {"content": [content_item]}
                # Attach image content blocks produced by a read of an image file so
                # the multimodal model receives the image itself alongside the text
                # metadata (the base64 never rides in response_text/JSON above).
                _pending_images = getattr(_tool_call_images, "value", None)
                if isinstance(_pending_images, list) and _pending_images:
                    response_payload["content"].extend(_pending_images)
                    _tool_call_images.value = []
                # Stash the full structured result for the in-process CLI so `tools call
                # ... --json` returns the dict for EVERY tool -- including the ones whose
                # host-facing content is rendered text (read, grep, search, shell, ...).
                # This never goes on response_payload, so the MCP host's main model only
                # ever sees `content`; no structured data rides the wire to any consumer.
                _tool_call_raw_result.value = result if isinstance(result, dict) else None
                with contextlib.suppress(Exception):
                    _append_tool_profile(
                        tool=name,
                        handler_ms=_call_duration_ms,
                        total_ms=round((time.perf_counter() - _call_started) * 1000),
                        response_size=_ok_response_size,
                        status="ok",
                        session_id=_ok_sid,
                    )
                return _ok(rid, response_payload)
            except Exception as exc:
                return _finalize_error_response(exc)

        try:
            if remote_routed:
                _remote_start = time.perf_counter()
                result = _dispatch_remote(name, args)
                _call_duration_ms = round((time.perf_counter() - _remote_start) * 1000)
                if isinstance(result, dict):
                    result = _clean_tool_result(result, name)
                return _finalize_response(result)
            else:
                led = _get_ledger()
                route_payload, route_state, route_workflow, route_step = _prepare_model_recommendation(
                    name,
                    args if isinstance(args, dict) else {},
                    led,
                )
                handler: Callable[[dict[str, Any]], Any] = spec["handler"]
                # (The hidden grep alias file_paths_with_count is handled in
                # _GREP_MODE_CANON, not here -- a dispatcher-level remap of
                # output_mode was a no-op because it tested the new spelling.)
                # (The bench-mode edit-grounding gate that used to sit here was
                # hard-removed 2026-07-03: it had no exemption for creating new
                # files, so it fired on nearly every from-scratch task, burning
                # retries/tokens for zero benefit -- see reports/ for the
                # Harbor-run cost analysis that surfaced this. Its Claude-plugin
                # and Codex PreToolUse-hook twins, and every test asserting any
                # of the three, were removed in the same pass.)
                _tool_call_tokens_saved.value = 0  # reset before handler so stale values can't bleed through
                _tool_call_counterfactual.value = None  # reset before handler
                _tool_call_rendered_text.value = None  # reset before handler
                wrapper_model = (
                    str(route_payload.get("model") or "")
                    if _route_enforcement_enabled() and route_payload.get("configured") is not False
                    else ""
                )
                from lemoncrow.core.capabilities.pricing import active_model_override

                try:
                    _handler_start = time.perf_counter()
                    with active_model_override(wrapper_model or None):
                        result = handler(args)
                    _call_duration_ms = round((time.perf_counter() - _handler_start) * 1000)
                finally:
                    # Runs in finally; a raise here would mask the handler's real
                    # exception, so suppress like the sibling savings-event calls.
                    with contextlib.suppress(Exception):
                        _finalize_model_recommendation(
                            route_payload,
                            led=led,
                            tool_name=name,
                            session_state=route_state,
                            workflow=route_workflow,
                            current_step=route_step,
                            wrapper_applied=bool(wrapper_model),
                            wrapper_model=wrapper_model or None,
                        )

                # Phase 2: a foreground bash command hands back a deferred marker so
                # the pool worker frees immediately; the watcher continuation runs
                # _finalize_response when the command completes.
                if isinstance(result, _DeferredResult):
                    return _Deferred(
                        src=result,
                        finalize=_finalize_response,
                        finalize_error=_finalize_error_response,
                    )
                return _finalize_response(result)
        except Exception as exc:
            return _finalize_error_response(exc)
        finally:
            # Always drop the request-scoped project override (N10).
            _clear_request_project(_prior_project)

    return _err(rid, -32601, f"unknown method: {method}")


# Depth cap for _strip_nulls: deeply nested adversarial JSON (e.g. from web_fetch
# or a sql JSON column) could otherwise blow Python's recursion limit and turn
# benign data into a -32000 error. Beyond this depth we stop recursing and return
# the subtree untouched.
_STRIP_NULLS_MAX_DEPTH = 200


def _strip_nulls(value: Any, _depth: int = 0) -> Any:
    """Recursively remove None and "" values from response values.

    Strips:
      - None values
      - empty string values ""

    Keeps:
      - empty lists [] and dicts {} (semantic — "no items" is info)
      - numeric 0 / 0.0 (meaningful)
      - False (meaningful)
    """
    if _depth >= _STRIP_NULLS_MAX_DEPTH:
        return value
    if isinstance(value, dict):
        return {k: _strip_nulls(v, _depth + 1) for k, v in value.items() if v is not None and v != ""}
    if isinstance(value, list):
        return [_strip_nulls(item, _depth + 1) for item in value]
    return value


def _clean_tool_result(result: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """Apply final response normalization before serialization."""
    _ = tool_name
    result = cast(dict[str, Any], _strip_nulls(result))
    return result


def _ok(rid: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _tool_error_code(exc: Exception) -> int:
    if isinstance(exc, MemoryConcurrencyError):
        return 409
    if isinstance(exc, MemorySidecarUnavailable):
        return 503
    return -32000


def _feature_locked_response(rid: str | int | None, exc: Exception) -> dict[str, object] | None:
    """Return a readable tool result (not a JSON-RPC error) for auth/license issues.

    The LLM sees this as regular tool output and explains it naturally to the
    user — far better than a cryptic error code.
    """
    # Open-source runtime: no feature is gated, so FeatureLocked is never raised.
    # Retained as a no-op for backward-compatible call sites.
    return None


def _mcp_max_workers() -> int:
    raw = os.environ.get("LEMONCROW_MCP_MAX_WORKERS", str(_DEFAULT_MCP_MAX_WORKERS))
    try:
        configured = int(raw)
    except ValueError:
        _log.warning(
            "invalid LEMONCROW_MCP_MAX_WORKERS=%r; using %d",
            raw,
            _DEFAULT_MCP_MAX_WORKERS,
        )
        return _DEFAULT_MCP_MAX_WORKERS
    return max(1, min(configured, _MAX_MCP_MAX_WORKERS))


_DEFAULT_MCP_HEAVY_WORKERS = 6
_MAX_MCP_HEAVY_WORKERS = 32
# Tools that can run for a long time (subprocess, network, mypy/pytest verify, or
# a workflow/agent spawn up to the 48h ceiling). They get a separate small
# executor lane so a burst can't evict cheap, frequent reads/searches from the
# main pool.
# code_search source shaping. UNCONDITIONAL: any section past _CODESEARCH_TOP2_MAX_CHARS
# keeps its head plus a precise L<start>-L<end> pointer -- one un-capped section measured
# 16.5k chars (~7.6k tokens) resident for 40+ turns. DEFAULT-ON (LEMONCROW_CODESEARCH_OUTLINE=0
# disables): large blocks collapse to a pointer-only outline -- code_search LOCATES the exact
# range to `read`; small matches stay inline (cheap, usually all the agent needs); the hidden
# include_source arg keeps bounded source for the top-2 matches.
# candidate_files/related_symbols (already pointers) stay.
_CODESEARCH_OUTLINE = os.environ.get("LEMONCROW_CODESEARCH_OUTLINE", "1").strip().lower() in {"1", "true", "on", "yes"}
# Matches at or below this many chars keep their source inline; larger ones are outlined.
_CODESEARCH_OUTLINE_MAX_CHARS = int(os.environ.get("LEMONCROW_CODESEARCH_OUTLINE_MAX_CHARS") or 400)
# include_source keeps the top-2 matches' source even in outline mode -- but bounded:
# one un-capped keep measured 16.5k chars (~7.6k tokens) resident for 40+ turns of a
# single benchmark run. Past the cap the section keeps its head plus a precise pointer.
_CODESEARCH_TOP2_MAX_CHARS = int(os.environ.get("LEMONCROW_CODESEARCH_TOP2_MAX_CHARS") or 8000)


def _outline_lean_view(lean: dict[str, Any], *, keep_top2: bool) -> dict[str, Any]:
    """Bound code_search section source; optionally outline large sections.

    Dict-level (applied in tool_code_search before rendering). ALWAYS: a section
    past _CODESEARCH_TOP2_MAX_CHARS keeps its head plus a precise L<start>-L<end>
    pointer.

    top2 (a confident call's top-ranked 1-2 sections; `files` is already
    relevance-ordered) keeps keep_top2's existing width and gate -- exact_match
    or the hidden include_source arg. A low-confidence "ranked candidates" call
    (keep_top2=False) instead protects only its SINGLE top-ranked section --
    a genuine top pick is still worth showing even on a gamble, but a
    low-confidence match ranked 2nd or later no longer earns a free pass just
    for being short: the outline collapse below applies to it UNCONDITIONALLY
    regardless of size, closing a gap where size alone (never a confidence
    signal) decided whether a low-rank guess shipped its source for free.
    """
    files = lean.get("files")
    if not isinstance(files, list):
        return lean
    match = 0  # rank across all matched sections (files are relevance-ordered)
    top_n = 2 if keep_top2 else 1
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        for sec in entry.get("sections") or []:
            if not (isinstance(sec, dict) and "content" in sec):
                continue
            top2 = match < top_n
            match += 1
            start, end = sec.get("line"), sec.get("end_line")
            # Outline mode first: a large non-top2 section always collapses to a
            # pointer-only outline (cheaper than any head cut); a small one only
            # collapses when the call isn't confident enough to earn the free pass.
            if (
                _CODESEARCH_OUTLINE
                and not top2
                and (not keep_top2 or len(sec["content"]) > _CODESEARCH_OUTLINE_MAX_CHARS)
            ):
                sym = sec.get("qualified_name") or path or "symbol"
                sec.pop("content", None)
                sec["outline"] = f"{sym} — read {path or sec.get('path', '')}:L{start}-L{end}"
                continue
            if len(sec["content"]) > _CODESEARCH_TOP2_MAX_CHARS:
                # Pathological single section: keep the head, point at the rest.
                sec["content"] = (
                    sec["content"][:_CODESEARCH_TOP2_MAX_CHARS]
                    + f"\n… [truncated — read {path or sec.get('path', '')}:L{start}-L{end}]"
                )
    return lean


_HEAVY_TOOLS = frozenset({"bash", "run", "edit", "web_fetch", "workflow", "agent"})

# Cost classes for per-request executor routing. Plain module-level str
# constants (not an Enum) for mypyc friendliness.
_COST_CPU = "cpu"  # GIL-bound, fast Python handlers: reads, searches, context,
#                    smart_read, trace, memory recall, protocol methods.
_COST_IO = "io"  # blocks on an external subprocess/socket/LLM: bash foreground,
#                  edit-with-verify, web_fetch, memory store_fact.
_COST_DETACHED = "detached"  # long-lived supervised child: workflow, agent,
#                              background bash.


def _mcp_heavy_max_workers() -> int:
    raw = os.environ.get("LEMONCROW_MCP_HEAVY_WORKERS", str(_DEFAULT_MCP_HEAVY_WORKERS))
    try:
        configured = int(raw)
    except ValueError:
        return _DEFAULT_MCP_HEAVY_WORKERS
    return max(1, min(configured, _MAX_MCP_HEAVY_WORKERS))


_DEFAULT_MCP_IO_WORKERS = 32
_MAX_MCP_IO_WORKERS = 128


def _mcp_io_max_workers() -> int:
    """Worker count for the IO lane (subprocess/socket/LLM-blocking tools).

    Reads LEMONCROW_MCP_IO_WORKERS (default 32, max 128). For back-compat, if the
    old heavy knob LEMONCROW_MCP_HEAVY_WORKERS is set and LEMONCROW_MCP_IO_WORKERS is
    not, the heavy value is used for the IO lane.
    """
    raw = os.environ.get("LEMONCROW_MCP_IO_WORKERS")
    if raw is None:
        if os.environ.get("LEMONCROW_MCP_HEAVY_WORKERS") is not None:
            return _mcp_heavy_max_workers()
        raw = str(_DEFAULT_MCP_IO_WORKERS)
    try:
        configured = int(raw)
    except ValueError:
        _log.warning(
            "invalid LEMONCROW_MCP_IO_WORKERS=%r; using %d",
            raw,
            _DEFAULT_MCP_IO_WORKERS,
        )
        return _DEFAULT_MCP_IO_WORKERS
    return max(1, min(configured, _MAX_MCP_IO_WORKERS))


def _classify_cost(req: dict[str, Any]) -> str:
    """Classify a JSON-RPC request into a cost class for executor routing.

    Returns one of _COST_CPU, _COST_IO, _COST_DETACHED.
    """
    if req.get("method") != "tools/call":
        # Protocol methods are trivial; the cheap CPU lane handles them.
        return _COST_CPU
    params = req.get("params") or {}
    if not isinstance(params, dict):
        return _COST_CPU
    name = params.get("name")
    if name == "run":
        name = "bash"
    args = params.get("arguments")
    if not isinstance(args, dict):
        args = {}
    if name in {"workflow", "agent"}:
        return _COST_DETACHED
    if name == "bash":
        backgrounded = args.get("background") is True
        if not backgrounded:
            command = args.get("command")
            if isinstance(command, str):
                stripped = command.rstrip()
                backgrounded = stripped.endswith("&") and not stripped.endswith("&&")
        return _COST_DETACHED if backgrounded else _COST_IO
    if name in {"edit", "web_fetch"}:
        return _COST_IO
    # memory store_fact runs a blocking arbiter LLM call.
    if name == "memory" and args.get("op") == "store_fact":
        return _COST_IO
    return _COST_CPU


def _is_heavy_request(req: dict[str, Any]) -> bool:
    """True if this JSON-RPC request is not a cheap CPU-lane request.

    Retained for back-compat with existing callers/tests; the live routing
    decision in serve() is made by _classify_cost.
    """
    return _classify_cost(req) != _COST_CPU


def _max_result_bytes() -> int:
    raw = os.environ.get("LEMONCROW_MCP_MAX_RESULT_BYTES", str(_DEFAULT_MAX_RESULT_BYTES))
    try:
        configured = int(raw)
    except ValueError:
        _log.warning(
            "invalid LEMONCROW_MCP_MAX_RESULT_BYTES=%r; using %d",
            raw,
            _DEFAULT_MAX_RESULT_BYTES,
        )
        return _DEFAULT_MAX_RESULT_BYTES
    # Floor avoids pathological tiny caps; ceiling keeps capped results safely
    # under the hard wire limit even after JSON string-escaping inflation.
    return max(64 * 1024, min(configured, _MAX_WIRE_BYTES - 1024 * 1024))


def _read_inline_budget_bytes() -> int:
    """Byte budget for a single inline file read before line-aligned truncation.

    Distinct from ``_max_result_bytes`` (the multi-MB host-disconnect wire guard):
    this is the much smaller threshold above which the *host's* MCP-output limit
    would persist the result to a temp file and make the agent re-read in blind
    ranges. Returning a line-aligned prefix with an exact continuation range keeps
    a whole-file read to a couple of clean calls. 0 disables the behavior.
    """
    raw = os.environ.get("LEMONCROW_READ_INLINE_BUDGET_BYTES", str(_DEFAULT_READ_INLINE_BUDGET_BYTES))
    try:
        configured = int(raw)
    except ValueError:
        return _DEFAULT_READ_INLINE_BUDGET_BYTES
    if configured <= 0:
        return 0
    return max(8 * 1024, configured)


def _read_batch_budget_bytes() -> int:
    """Total byte budget for one batched ``read`` call before outline downgrade.

    Distinct from ``_read_inline_budget_bytes`` (a *single* file's ceiling): a
    batch whose members each clear the per-file bar still lands as one very large
    result. Measured against Cursor: eight ``:full`` files in one call returned
    ~8.8k tokens, where the same files read individually and on demand cost ~950
    each -- and several were never needed at all. The per-file guard cannot see
    the aggregate; only the assembled call can. 0 disables the downgrade.
    """
    raw = os.environ.get("LEMONCROW_READ_BATCH_BUDGET_BYTES", str(_DEFAULT_READ_BATCH_BUDGET_BYTES))
    try:
        configured = int(raw)
    except ValueError:
        return _DEFAULT_READ_BATCH_BUDGET_BYTES
    if configured <= 0:
        return 0
    return max(8 * 1024, configured)


def _batch_entry_bytes(entry: object) -> int:
    """Serialized size of one batch entry, used as the budget unit."""
    try:
        return len(json.dumps(entry, default=str))
    except (TypeError, ValueError):
        return 0


def _apply_batch_read_budget(
    results: list[dict[str, Any]],
    entry_specs: list[str | None],
    include_meta: bool,
) -> list[str]:
    """Cap a batch read's total size by downgrading its largest whole-file bodies.

    Downgrades biggest-first until the call fits, so the fewest files lose their
    bodies. Entries the caller narrowed (range/outline/summary/head/tail) are
    never touched -- those were asked for precisely and are already cheap.

    Returns the paths that were downgraded, for the caller to disclose.
    """
    budget = _read_batch_budget_bytes()
    if budget <= 0:
        return []
    total = sum(_batch_entry_bytes(entry) for entry in results)
    if total <= budget:
        return []
    candidates = sorted(
        (
            idx
            for idx, spec_path in enumerate(entry_specs)
            if spec_path is not None and isinstance(results[idx], dict) and not results[idx].get("error")
        ),
        key=lambda idx: _batch_entry_bytes(results[idx]),
        reverse=True,
    )
    downgraded: list[str] = []
    for idx in candidates:
        if total <= budget:
            break
        spec_path = entry_specs[idx]
        if spec_path is None:
            continue
        before = _batch_entry_bytes(results[idx])
        try:
            outlined = _smart_read_single(path=spec_path, outline=True, include_meta=include_meta)
        except Exception:
            continue
        after = _batch_entry_bytes(outlined)
        # An outline that is not actually smaller (tiny file, no extractable
        # structure) would cost tokens now AND a re-read round-trip later.
        if after >= before:
            continue
        results[idx] = outlined
        total -= before - after
        downgraded.append(spec_path)
    return downgraded


def _truncate_result_text(text: str | bytes, limit: int, tool_name: str | None = None) -> str:
    """Bound a tool-result string to *limit* UTF-8 bytes, appending a notice.

    A single oversized result would otherwise serialize into one JSON-RPC frame
    larger than the host's stdout guard, which disconnects the whole server.

    This is the last-resort backstop: tools in ``_SPILL_TOOLS`` are already
    spilled by ``_spill_oversized_result_text`` before this runs, so this
    mainly fires for everything else. When T7 spill is enabled and *tool_name*
    is given, the full pre-truncation text is persisted here too, so the footer
    names a recoverable path instead of the bare spill-failed shape.
    """
    if isinstance(text, bytes):
        # Defensive: this is the last-resort backstop for arbitrary tool output --
        # normalize once here so `.encode("utf-8")` below never raises on bytes
        # handed in despite the str contract.
        text = text.decode("utf-8", errors="replace")
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    from lemoncrow.pro.capabilities.tool_supervision import tool_output_spill

    record = None
    if tool_name and _tool_output_spill_enabled():
        record = tool_output_spill.spill(text, tool_name=tool_name, kind="original")
    notice = "\n\n" + tool_output_spill.spill_notice(
        verb="truncated",
        original_chars=len(text),
        kept_chars=limit,
        path=record.path if record is not None else None,
    )
    headroom = max(0, limit - len(notice.encode("utf-8")))
    head = encoded[:headroom].decode("utf-8", "ignore")
    return head + notice


def _compact_result_chars() -> int:
    """Char threshold above which an oversized tool result is head+tail compacted.

    Distinct from the multi-MB wire guard (``_max_result_bytes``): that one only
    keeps the JSON-RPC frame under the host's stdout limit. This is a context-
    hygiene bound -- a single runaway result otherwise floods the host prompt and
    the host re-pays for it on every later turn. Set
    ``LEMONCROW_MCP_COMPACT_RESULT_CHARS=0`` to disable.
    """
    raw = os.environ.get("LEMONCROW_MCP_COMPACT_RESULT_CHARS", str(_DEFAULT_COMPACT_RESULT_CHARS))
    try:
        configured = int(raw)
    except ValueError:
        _log.warning(
            "invalid LEMONCROW_MCP_COMPACT_RESULT_CHARS=%r; using %d",
            raw,
            _DEFAULT_COMPACT_RESULT_CHARS,
        )
        return _DEFAULT_COMPACT_RESULT_CHARS
    return max(0, configured)


def _spill_result_chars(tool_name: str | None = None) -> int:
    """Strict returned-char cap for recoverably spilled tool outputs.

    The full original is persisted before the bounded summary is emitted, so this
    limit never discards output and does not affect tools without spill support.
    Resolution order: an explicit ``LEMONCROW_MCP_SPILL_RESULT_CHARS`` env value
    wins for every tool (set it to ``0`` to disable the char-gated cap while
    retaining the multi-MB wire backstop); otherwise a per-tool override from
    ``_SPILL_RESULT_CHARS_BY_TOOL`` (e.g. bash gets a larger inline budget);
    otherwise ``_DEFAULT_SPILL_RESULT_CHARS``.
    """
    raw = os.environ.get("LEMONCROW_MCP_SPILL_RESULT_CHARS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            _log.warning(
                "invalid LEMONCROW_MCP_SPILL_RESULT_CHARS=%r; using per-tool defaults",
                raw,
            )
    if tool_name is not None:
        return _SPILL_RESULT_CHARS_BY_TOOL.get(tool_name, _DEFAULT_SPILL_RESULT_CHARS)
    return _DEFAULT_SPILL_RESULT_CHARS


def _compact_result_text(text: str | bytes, tool_name: str) -> str:
    """Head+tail compact a single oversized tool result before it reaches the host.

    Deterministic (no LLM) so identical calls yield identical bytes and never
    bust the host's prefix cache. Keeps the head (command, first error, initial
    context) and the tail (final status/return value) with an omission marker in
    between, then appends a recovery hint. Results within the threshold pass
    through untouched.

    This is the generic backstop for tools OUTSIDE ``_SPILL_TOOLS`` -- those get
    spilled earlier (``_spill_oversized_result_text``) and never reach here at
    full size. When T7 spill is enabled the full pre-compaction text is
    persisted here too, so the recovery hint names a path instead of just
    "narrow the query" regardless of which tool produced the result.
    """
    if isinstance(text, bytes):
        # Defensive: generic backstop for arbitrary tool output -- normalize once
        # here so `.encode("utf-8")` below never raises on bytes handed in
        # despite the str contract.
        text = text.decode("utf-8", errors="replace")
    threshold = _compact_result_chars()
    if threshold <= 0:
        return text
    # Gate on bytes, not just chars: a multibyte/CJK result can sit under the
    # char threshold while its UTF-8 footprint is several times larger and would
    # still flood the host prompt. len(text) is a lower bound on bytes, so only
    # pay the full encode when chars are under but bytes might exceed (x4 worst case).
    _over = len(text) > threshold
    if not _over and len(text) * 4 > threshold:
        _over = len(text.encode("utf-8")) > threshold
    if not _over:
        return text
    from lemoncrow.pro.capabilities.tool_supervision import tool_output_spill
    from lemoncrow.pro.capabilities.tool_supervision.compact_output import compress_tool_output

    target = max(4096, threshold // 4)
    head = int(target * 0.7)
    tail = max(1, target - head)
    compacted = compress_tool_output(text, threshold_chars=target, head_chars=head, tail_chars=tail)
    record = (
        tool_output_spill.spill(text, tool_name=tool_name, kind="original") if _tool_output_spill_enabled() else None
    )
    footer = tool_output_spill.spill_notice(
        verb="compacted",
        original_chars=len(text),
        kept_chars=len(compacted),
        path=record.path if record is not None else None,
    )
    return f"{compacted}\n\n{footer}"


# T7/T8 — tools whose oversized output is worth spilling/compacting reversibly.
# These produce expensive or non-idempotent output (shell side effects, sql
# query cost, large file reads, network fetches) where re-running to recover a
# truncated tail is wasteful or unsafe.
_SPILL_TOOLS = frozenset({"bash", "code_search", "sql", "read", "web_fetch", "mcp"})

# Tools subject to the strict char-gated spill cap (_spill_result_chars). `read`
# is intentionally EXCLUDED: it is the agent's explicit, incremental retrieval
# surface (ranges, full=true, slice windows) and the very tool used to recover
# spilled output, so capping it would defeat the spill-recovery cycle. read keeps
# its own inline budget + outline projection and the multi-MB wire backstop, so
# it is never lossily truncated -- just not force-summarized at 2 KiB.
# `web_fetch` is EXCLUDED for the same reason: web_fetch.py now spills its own
# full rendered page on truncation (_truncate_with_spill) and names the spill
# path directly in the `[truncated ...]` notice, so re-summarizing that notice
# down to a couple KiB here would just bury the recovery hint under a second,
# nested head/tail digest. code_search stays char-capped -- its output isn't
# self-managed the same way.
_SPILL_CHAR_CAP_TOOLS = frozenset({"bash", "code_search", "sql", "mcp"})


# Redirected bash calls (curl->web_fetch, sed -n / cat -> read) take the TARGET
# tool's spill identity, not bash's -- a redirected read keeps read's larger
# incremental-retrieval budget; a redirected fetch keeps web_fetch's own
# self-managed truncation (also exempt from the char cap, like read).
# grep/find_glob bound their own output in-handler (ranked projection / 300-entry
# cap) so they keep the generic bash backstop.
_REWRITE_SPILL_IDENTITY = {
    "read": "read",
    "read_range": "read",
    "web_fetch": "web_fetch",
}


def _effective_spill_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Spill identity for a call: a bash command rewritten to another tool spills
    AS that tool (its budget + semantics); everything else spills as itself."""
    if tool_name != "bash":
        return tool_name
    command = str(args.get("command") or "").strip() if isinstance(args, dict) else ""
    if not command:
        return tool_name
    try:
        from lemoncrow.pro.capabilities.tool_supervision.bash_exec import classify_command

        decision = classify_command(command)
    except Exception:
        return tool_name
    if decision.action == "rewrite" and decision.rewrite_target:
        return _REWRITE_SPILL_IDENTITY.get(decision.rewrite_target, tool_name)
    return tool_name


_CODE_CONTENT_TOOLS = frozenset({"read"})


def _tool_output_spill_enabled() -> bool:
    """T7 flag: spill oversized output instead of discarding the overflow."""
    return os.environ.get("LEMONCROW_TOOL_OUTPUT_SPILL", "1").strip().lower() in {"1", "true", "yes", "on"}


def _auto_compact_output_enabled() -> bool:
    """T8 flag: auto-apply compact_output.compact() to oversized results."""
    return os.environ.get("LEMONCROW_AUTO_COMPACT_OUTPUT", "0").strip().lower() in {"1", "true", "yes", "on"}


# None (not 0.0) baselines so the FIRST attempt always fires: time.monotonic()
# is boot-relative, so a 0.0 baseline on a host whose uptime is below the
# interval reads as "ran <interval> ago" and spuriously skips the first mint/
# report/reconcile -- exactly the fresh-container/CI case the self-heal exists
# for.
_VERDICT_SELF_HEAL_INTERVAL_SECONDS = 120.0
_verdict_self_heal_at: float | None = None
_LEDGER_RECONCILE_INTERVAL_SECONDS = 600.0
_ledger_reconcile_at: float | None = None
_USAGE_REPORT_TICK_INTERVAL_SECONDS = 120.0
_usage_report_tick_at: float | None = None


def _tick_usage_report(root: Path) -> None:
    """Opportunistically push the local savings total to the server, throttled.

    Real reporting is otherwise host-stop-hook-driven (``maybe_report_usage``'s
    own hourly on-disk watermark): a single long-running session that never
    triggers a host stop event would never re-report, so local usage can run
    well past the cap while the signed verdict -- and thus dormancy -- stays
    stale for the entire session (only the next stop-hook, possibly hours
    away, would catch it up). Ticking here, on the MCP request boundary that
    already re-evaluates dormancy on every tool-list/tool-call, bounds that
    lag to ~1 hour even inside one long session. Cheap: maybe_report_usage
    already throttles itself via an on-disk watermark, so most calls here are
    a single fast disk read that no-ops.
    """
    # Open-source runtime: no usage reporting, no cap, no phone-home. The former
    # per-request server report is removed; nothing to tick.
    return None


def _mint_missing_verdict(root: Path) -> bool:
    """Throttled, forced attempt to mint the missing signed cap verdict.

    Dormancy with reason ``no_token`` is recoverable: it means no verdict is
    persisted for the current identity (fresh container with a forwarded
    auth token, a login that happened offline, an expired token on an idle
    host), not that the server judged the cap exhausted. Environments with
    no background reconciler — Docker/Harbor containers, CI — would
    otherwise stay dormant forever. Best-effort and throttled so an offline
    host pays at most one short network attempt per interval.
    """
    global _verdict_self_heal_at
    now = time.monotonic()
    if _verdict_self_heal_at is not None and now - _verdict_self_heal_at < _VERDICT_SELF_HEAL_INTERVAL_SECONDS:
        return False
    _verdict_self_heal_at = now
    try:
        from lemoncrow.core.capabilities.licensing.usage_report import report_usage_once

        return report_usage_once(root, force=True)
    except Exception:
        return False


def _reconcile_ledger_gap(root: Path, server_saved_usd: float) -> None:
    """Throttled: catch up the local savings ledger to a verified server total.

    Only meaningful when the account HAS verified server-side savings ahead
    of what the local ledger shows (see ``reconcile_local_savings_gap`` for
    the max(local, server) semantics) -- for 99% of installs the ledger never
    drifts from the server and this is a no-op scan. Throttled independently
    from the no_token self-heal above: this repairs a display figure, not
    dormancy, so it can run far less often.
    """
    global _ledger_reconcile_at
    now = time.monotonic()
    if _ledger_reconcile_at is not None and now - _ledger_reconcile_at < _LEDGER_RECONCILE_INTERVAL_SECONDS:
        return
    _ledger_reconcile_at = now
    try:
        from lemoncrow.core.capabilities.plugin_runtime import reconcile_local_savings_gap

        reconcile_local_savings_gap(root, server_saved_usd)
    except Exception:
        pass


_DORMANT_REFRESH_INTERVAL_SECONDS = 300.0  # 5 min; the hotpath tolerates staleness
_dormant_lock = threading.Lock()
_dormant_snapshot: bool | None = None
_dormant_refresher_started = False


def _resolve_dormant(root: Path, *, side_effects: bool) -> bool:
    """Resolve the cap dormancy decision.

    With ``side_effects`` (the background refresher only), ALSO run the
    opportunistic usage-report tick, the no_token forced mint, and the ledger
    reconcile — all network / heavy-ledger I/O that must never touch the request
    hotpath.
    """
    from lemoncrow.pro.capabilities.licensing_gate import resolve_cap_verdict

    if side_effects:
        _tick_usage_report(root)
    verdict = resolve_cap_verdict(root)
    if side_effects and verdict.dormant and verdict.reason == "no_token" and _mint_missing_verdict(root):
        verdict = resolve_cap_verdict(root)
    elif side_effects and verdict.verified and verdict.server_saved_usd is not None:
        _reconcile_ledger_gap(root, verdict.server_saved_usd)
    return verdict.dormant


def _refresh_dormant_snapshot() -> bool:
    """Recompute dormancy WITH side effects and cache it. Runs on the background
    refresher thread (and is what the cap-passthrough tests drive directly)."""
    global _dormant_snapshot
    try:
        value = _resolve_dormant(_lemoncrow_root(), side_effects=True)
    except Exception:
        _log.exception("Cap authority refresh failed; hiding LemonCrow tools")
        value = True
    with _dormant_lock:
        _dormant_snapshot = value
    return value


def _dormant_refresh_loop() -> None:
    while True:
        _refresh_dormant_snapshot()
        time.sleep(_DORMANT_REFRESH_INTERVAL_SECONDS)


def _start_dormant_refresher() -> None:
    """Start the background dormancy refresher once (called from ``serve()``).

    Moves every network/ledger side effect of cap resolution off the request
    path onto a ~5-minute background loop; the hotpath then only reads the
    cached bool.
    """
    global _dormant_refresher_started
    # Open-source runtime: no cap, no dormancy, no background verdict minting.
    # The refresher is intentionally not started.
    return


def _snapshot_dormant() -> bool:
    """Cached cap-authority decision for the request hotpath — an in-memory read.

    A background thread (:func:`_start_dormant_refresher`, launched by
    ``serve()``) refreshes it every ~5 min, doing all network/ledger work off
    this path. Until the first refresh lands, resolve once synchronously WITHOUT
    side effects (a cheap local verdict read) so a cold cache still fails closed.
    """
    global _dormant_snapshot
    with _dormant_lock:
        cached = _dormant_snapshot
    if cached is not None:
        return cached
    try:
        value = _resolve_dormant(_lemoncrow_root(), side_effects=False)
    except Exception:
        _log.exception("Cap authority failed; hiding LemonCrow tools")
        value = True
    with _dormant_lock:
        if _dormant_snapshot is None:
            _dormant_snapshot = value
        return _dormant_snapshot


def _savings_dormant() -> bool:
    """Open-source runtime: there is no savings cap and no dormancy — every tool
    is always available. See docs/maintenance-mode-transition.md.
    """
    return False


def _read_path_arg(args: dict[str, Any]) -> str:
    """Best-effort extraction of the path a read-style call targeted."""
    raw = args.get("path") if isinstance(args, dict) else None
    if isinstance(raw, str) and raw:
        # A read path may carry a ':Lx-Ly' line-range suffix; strip it.
        return _split_read_range_suffix(raw)[0]
    return ""


def _is_spill_path(path: str) -> bool:
    """True when ``path`` resolves to a file inside the shared spill directory."""
    from lemoncrow.pro.capabilities.tool_supervision.tool_output_spill import _spill_dir

    try:
        return Path(path).resolve().parent == _spill_dir().resolve()
    except (ValueError, OSError):
        return False


def _auto_compact_result_text(text: str, tool_name: str, args: dict[str, Any]) -> str:
    """T8 — auto-apply compaction to an oversized result, reversibly.

    AST/structure-aware for code (uses the read-side source-projection compact so
    the projected view stays line-diffable); falls back to the deterministic
    head+tail compaction (``compact_output.compact``) for everything else.

    REVERSIBLE: the untransformed original is written to the T7 spill store and a
    recovery hint naming ``read <path>`` is appended, so the dropped detail is
    never lost. Flag-gated by ``LEMONCROW_AUTO_COMPACT_OUTPUT`` (off -> returns
    ``text`` unchanged).
    """
    if not _auto_compact_output_enabled():
        return text
    threshold = _compact_result_chars()
    if threshold <= 0 or len(text) <= threshold:
        return text

    from lemoncrow.pro.capabilities.tool_supervision import compact_output, tool_output_spill

    # 16x reduction from the char threshold: first chars/4 to estimate tokens
    # (the ~4-chars-per-token rule), then /4 again for headroom so the compacted
    # view lands well under the threshold. e.g. 256K chars -> ~16K-token budget.
    budget_tokens = max(256, threshold // 4 // 4)
    compacted_text = text
    method = "compact_output"

    # AST-aware path for code reads: project the source to its compact view.
    # source_projection is a free feature (see licensing/features.py); the
    # has_feature check below always passes but stays as the single seam if
    # that ever changes.
    lang = ""
    from lemoncrow.core.capabilities import licensing as _licensing

    if tool_name in _CODE_CONTENT_TOOLS and _licensing.has_feature("source_projection"):
        with contextlib.suppress(Exception):
            from lemoncrow.infra.code_intel.languages import language_for_path
            from lemoncrow.pro.capabilities.source_projection import build_compact_projection

            lang_record = language_for_path(_read_path_arg(args))
            if lang_record is not None:
                lang = lang_record.name
                projection = build_compact_projection(text, lang)
                # Char-based gate (we budget in chars here): the projection is
                # token-neutral on pure whitespace but still trims bytes, which
                # is what shrinks the host-prompt footprint.
                if len(projection.content) < len(text):
                    compacted_text = projection.content
                    method = f"projection:{lang}"

    if compacted_text is text:
        compacted = compact_output.compact(
            text,
            content_type="file" if tool_name in _CODE_CONTENT_TOOLS else "tool_output",
            budget_tokens=budget_tokens,
        )
        compacted_text = compacted.compacted

    if len(compacted_text) >= len(text):
        return text  # compaction did not help — leave the original untouched.

    record = tool_output_spill.spill(
        text,
        tool_name=tool_name,
        kind="original",
    )
    if record is None:
        # Could not preserve the original -> do NOT lossily compact; return as-is
        # so the downstream wire guard handles it rather than dropping detail
        # irreversibly.
        return text
    footer = tool_output_spill.spill_notice(
        verb=f"compacted:{method}",
        original_chars=len(text),
        kept_chars=len(compacted_text),
        path=record.path,
    )
    return f"{compacted_text}\n\n{footer}"


def _spill_oversized_result_text(
    text: str | bytes,
    tool_name: str,
    args: dict[str, Any],
    limit: int,
    *,
    unit: str = "bytes",
    tools: frozenset[str] = _SPILL_TOOLS,
) -> str:
    """T7 — spill an over-budget result instead of discarding the overflow.

    When a ``bash``/``sql``/``read``/``web_fetch`` result exceeds the budget, the
    legacy path truncates/compacts and the middle is *lost*. Here the full,
    UNTRANSFORMED payload is written to the spill store as plain text and the
    host-facing text becomes a head/tail summary + the path + a ``read`` hint,
    so the agent can pull the rest back without re-running the tool. If the
    target of a ``read`` call is itself a spill file, re-spilling is skipped and
    normal truncation applies instead — no recursive spill chain.

    M1 — the gate ``unit`` selects the budget basis: ``"chars"`` (compared
    against ``len(text)``) lets the spill fire at the legacy char threshold
    (``_compact_result_chars``), i.e. BEFORE ``_compact_result_text`` would have
    dropped the middle; ``"bytes"`` keeps the original wire-byte semantics.

    Enabled by default, explicitly disabled with ``LEMONCROW_TOOL_OUTPUT_SPILL=0``,
    and limited to ``tools`` (default ``_SPILL_TOOLS``; the char-gated call site
    passes ``_SPILL_CHAR_CAP_TOOLS`` to exempt ``read``). Off / ineligible tools
    -> returns ``text`` unchanged so the caller's existing compaction/truncation
    runs exactly as before.
    """
    if isinstance(text, bytes):
        # Defensive: T7 spill backstop for arbitrary tool output -- normalize once
        # here so `.encode("utf-8")` below never raises on bytes handed in
        # despite the str contract.
        text = text.decode("utf-8", errors="replace")
    if not _tool_output_spill_enabled() or tool_name not in tools:
        return text
    # Don't re-spill a read that targets an already-spilled file: let normal
    # truncation apply so there is no recursive spill chain.
    if tool_name == "read" and _is_spill_path(_read_path_arg(args)):
        return text
    measured = len(text) if unit == "chars" else len(text.encode("utf-8"))
    if limit <= 0 or measured <= limit:
        return text

    from lemoncrow.pro.capabilities.tool_supervision import tool_output_spill
    from lemoncrow.pro.capabilities.tool_supervision.compact_output import compress_tool_output

    record = tool_output_spill.spill(
        text,
        tool_name=tool_name,
        kind="tool_output",
    )
    if record is None:
        return text  # spill failed -> fall back to the legacy compaction/truncation.

    # A compact head+tail summary. summary_with_ref applies the final strict cap
    # after reserving room for the recovery path and instructions.
    summary_budget = limit if unit == "chars" else limit // 8
    target = max(256, min(summary_budget, 16384))
    head = int(target * 0.7)
    tail = max(1, target - head)
    summary = compress_tool_output(text, threshold_chars=target, head_chars=head, tail_chars=tail)
    return tool_output_spill.summary_with_ref(
        summary,
        record,
        original_chars=len(text),
        verb="shrunk",
        max_chars=limit if unit == "chars" else None,
    )


def _write_jsonrpc(message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False, sort_keys=True) + "\n"
    # Hard backstop: a frame above the host's ~16 MiB stdout guard disconnects
    # the server. Per-result capping should prevent this, but escaping overhead
    # or non-result frames could still exceed it — replace such a frame with
    # a small error so the session survives instead of dropping.
    if len(payload.encode("utf-8")) > _MAX_WIRE_BYTES and message.get("id") is not None:
        _log.warning(
            "jsonrpc frame exceeds %d bytes; replacing with error to protect the connection",
            _MAX_WIRE_BYTES,
        )
        message = _err(
            message["id"],
            -32000,
            f"result exceeded the {_MAX_WIRE_BYTES} byte MCP frame limit and was dropped to "
            "keep the connection alive; re-request a narrower slice.",
        )
        payload = json.dumps(message, ensure_ascii=False) + "\n"
    with _STDOUT_LOCK:
        sys.stdout.write(payload)
        sys.stdout.flush()


def _handle_and_write(request: dict[str, Any]) -> None:
    # Mark this worker thread as deferral-capable for the duration of _handle so a
    # foreground bash command can hand the worker back and let the watcher finalize
    # the response (see _deferral_supported). Reset in finally so a pooled worker
    # never carries the flag into non-tool work.
    _deferral_context.active = True
    try:
        response = _handle(request)
        if isinstance(response, _Deferred):
            deferred = response

            def _on_complete() -> None:
                try:
                    concrete = deferred.src.collect()
                except Exception as exc:
                    # A failed deferred result (e.g. web_fetch network/SSRF error)
                    # goes through the same tool-error pipeline as the sync path,
                    # for a byte-identical error response.
                    try:
                        resp = deferred.finalize_error(exc)
                    except Exception:
                        _log.exception("deferred MCP error-finalize failed")
                        resp = _err(request.get("id"), -32603, f"internal error: {exc}")
                else:
                    try:
                        resp = deferred.finalize(concrete)
                    except Exception as exc:
                        _log.exception("deferred MCP continuation failed")
                        resp = _err(request.get("id"), -32603, f"internal error: {exc}")
                with contextlib.suppress(Exception):
                    _write_jsonrpc(resp)

            armed = deferred.src.register(_on_complete)
            if armed is False:
                # The work already finished before we could arm the watcher;
                # produce the response now on this worker thread.
                _on_complete()
            return
    except Exception as exc:
        _log.exception("unhandled MCP request failure")
        response = _err(request.get("id"), -32603, f"internal error: {exc}")
    finally:
        _deferral_context.active = False
    if response is not None:
        try:
            _write_jsonrpc(response)
        except OSError:
            _log.exception("failed to write MCP response (connection closed)")
        except Exception:
            _log.exception("failed to write MCP response (unexpected error)")


def serve() -> None:
    # CPU lane: GIL-bound, fast Python handlers (reads, searches, context,
    # smart_read, trace, memory recall, protocol methods). This is the old
    # "light" pool, unchanged in size.
    cpu_executor = ThreadPoolExecutor(
        max_workers=_mcp_max_workers(),
        thread_name_prefix="lemoncrow-cpu",
    )
    # IO lane: tools that block on an external subprocess/socket/LLM
    # (bash foreground, edit-with-verify, web_fetch, memory store_fact).
    # NOTE(phase-1): detached-class requests (workflow/agent/background bash) are
    # routed to this IO lane for now; a later phase will give the detached class
    # true no-slot handling.
    io_executor = ThreadPoolExecutor(
        max_workers=_mcp_io_max_workers(),
        thread_name_prefix="lemoncrow-io",
    )

    def _stdin_reader() -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except (json.JSONDecodeError, RecursionError) as exc:
                # RecursionError (deeply-nested JSON, e.g. a line of many '[')
                # is NOT a JSONDecodeError; without it here one adversarial line
                # kills the reader thread and tears down the whole stdio server.
                _write_jsonrpc(_err(None, -32700, f"parse error: {type(exc).__name__}"))
                continue
            # Valid JSON but not a request object (batch array, string, null):
            # answer -32600 instead of letting .get() kill the reader thread
            # (mirrors the mcp_http.py transport guard).
            if not isinstance(req, dict):
                _write_jsonrpc(_err(None, -32600, "invalid request: expected a JSON object"))
                continue
            # Initialization establishes client capabilities and must complete
            # before later requests can observe them.
            if req.get("method") in {"initialize", "notifications/initialized"}:
                _handle_and_write(req)
                continue
            executor = io_executor if _classify_cost(req) != _COST_CPU else cpu_executor
            executor.submit(_handle_and_write, req)

    _start_dormant_refresher()
    reader = threading.Thread(target=_stdin_reader, daemon=True, name="mcp-stdin-reader")
    reader.start()
    try:
        reader.join()
    finally:
        cpu_executor.shutdown(wait=True, cancel_futures=False)
        io_executor.shutdown(wait=True, cancel_futures=False)
        _emit_mcp_session_end()
        from lemoncrow.core.service.telemetry import shutdown_otel

        shutdown_otel()
        with contextlib.suppress(Exception):
            from lemoncrow.gateway.integrations.langfuse import shutdown as _lf_shutdown

            _lf_shutdown()


# Rotate mcp.log at 10MB, keeping 5 backups (~60MB ceiling) so a long-lived
# daemon's log never grows unbounded.
_MCP_LOG_MAX_BYTES = 10 * 1024 * 1024
_MCP_LOG_BACKUP_COUNT = 5


def _setup_file_logging(root: str | Path) -> None:
    """Configure the lemoncrow.mcp logger to write to a rotating file.

    This ensures logs survive process termination and can be inspected
    via ``lc logs mcp``. Rotates at ``_MCP_LOG_MAX_BYTES`` so a long-lived
    daemon never accumulates an unbounded log.
    """
    from logging.handlers import RotatingFileHandler

    log_dir = Path(root) / "mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mcp.log"

    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=_MCP_LOG_MAX_BYTES,
        backupCount=_MCP_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    mcp_logger = logging.getLogger("lemoncrow.mcp")
    mcp_logger.addHandler(handler)
    mcp_logger.setLevel(logging.DEBUG)


_account_gate_lock = threading.Lock()
_account_gate_state: dict[str, bool] = {}
_LOGIN_COOLDOWN_SECONDS = 3600.0


def _try_seamless_login(lemoncrow_root: Path) -> bool:
    """Best-effort: open the browser once per cooldown window to activate a free account.

    Runs inside a background daemon thread (never the stdio server's main
    request-serving thread), so blocking here for the OAuth round-trip does
    not delay tool responses. Debounced via a marker file so an unattended
    session (offline, no browser, user ignores the tab) doesn't pop a new
    browser tab on every workspace/session start.

    Never fires after an explicit `lc init --no-login` (``store.is_login_declined``)
    -- that opt-out persists until the next successful `lc account login` / `lc init`
    without --no-login, so an unattended install never gets a surprise browser tab.
    """
    # Open-source runtime: never open a browser or solicit login automatically.
    # The optional hosted account is only ever linked via an explicit
    # `lc account login`. See docs/maintenance-mode-transition.md.
    return False


def _ensure_account_activated(lemoncrow_root: Path) -> bool:
    """Gate for auto-provisioning: True once a free LemonCrow account token exists.

    If no token is present, attempts the seamless browser OAuth login exactly
    once per process (guarded by a lock so the several auto-init daemon
    threads spawned at stdio startup share one attempt instead of each
    popping its own browser tab).
    """
    # Open-source runtime: no account is ever required. Auto-init and code-index
    # warm always proceed locally; login is never solicited.
    return True


def _auto_init_workspace() -> None:
    """One-time workspace bootstrap: seed playbooks, add .gitignore, write marker.

    Runs as a daemon thread on stdio MCP startup.  Fail-open so a crash never
    blocks server startup.  Idempotent via ``.workspace_inited`` marker file.
    """
    try:
        from importlib import resources as _resources

        import yaml

        from lemoncrow.core.foundation.models import Playbook, Rubric
        from lemoncrow.core.foundation.paths import (
            ensure_gitignore,
            is_recognized_workspace,
            resolve_workspace_store_dir,
        )
        from lemoncrow.infra.storage.factory import create_store

        lemoncrow_root = _lemoncrow_root()
        ws_root = _workspace_root()

        if not is_recognized_workspace(ws_root):
            # ws_root falls all the way back to cwd when no host set a
            # workspace env var and no git repo was detected -- never
            # auto-create `.lemoncrow/` (with seeded lessons/rubrics) at an
            # arbitrary directory like $HOME or a multi-repo container. Only
            # a git repo or an already `lc init`-registered dir qualifies.
            _log.debug("skipping workspace auto-init: %s is not a recognized workspace", ws_root)
            return

        marker = resolve_workspace_store_dir(lemoncrow_root, ws_root) / ".workspace_inited"
        if marker.exists():
            return

        if not _ensure_account_activated(lemoncrow_root):
            _log.info(
                "LemonCrow account required for workspace auto-init; skipping until `lc account login` completes."
            )
            return

        # --- Seed playbooks and rubrics ---
        store = create_store(lemoncrow_root)
        store.init()

        blocks_dir = _resources.files("lemoncrow") / "infra" / "seed_playbooks"
        rubric_dir = _resources.files("lemoncrow") / "core" / "rubrics"

        for path in sorted(Path(str(p)) for p in blocks_dir.iterdir() if p.name.endswith(".yaml")):
            data = yaml.safe_load(path.read_text("utf-8"))
            if not isinstance(data, dict):
                continue
            if "id" not in data:
                try:
                    data["id"] = Playbook.make_id(data.get("title", ""), data.get("domain", ""))
                except (KeyError, ValueError):
                    continue
            try:
                store.knowledge.upsert_block(Playbook.model_validate(data))
            except (KeyError, ValueError):
                continue

        for path in sorted(Path(str(p)) for p in rubric_dir.iterdir() if p.name.endswith(".yaml")):
            data = yaml.safe_load(path.read_text("utf-8"))
            if not isinstance(data, dict):
                continue
            try:
                store.knowledge.upsert_rubric(Rubric.model_validate(data))
            except (KeyError, ValueError):
                continue

        # --- Add .gitignore ---
        ensure_gitignore(ws_root)

        # --- Write marker ---
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("")
    except Exception:
        logging.exception("auto-init failed, continuing")


def _warm_stdio_code_index() -> None:
    """Warm the single-workspace code-context engine for the stdio MCP path.

    Reuses the service ``_CodeWarmer`` patterns via ``warm_stdio_workspace``.
    Fail-open: any failure is swallowed so stdio server startup is unaffected.

    No-ops when ``_workspace_root()`` isn't a git repo or an ``lc
    init``-registered directory: ``warm_stdio_workspace`` has no such check
    of its own and will fire an ``lc code index`` subprocess against
    *any* directory (e.g. cwd falling all the way through to an unrelated
    multi-repo container or ``$HOME``), which never finishes and burns a CPU
    core indefinitely.
    """
    from lemoncrow.core.foundation.paths import is_recognized_workspace

    ws_root = _workspace_root()
    if not is_recognized_workspace(ws_root):
        _log.debug("skipping stdio code-index warm: %s is not a recognized workspace", ws_root)
        return
    if not _ensure_account_activated(_lemoncrow_root()):
        _log.info("LemonCrow account required for code-index warm; skipping until `lc account login` completes.")
        return
    try:
        from lemoncrow.core.service.code_warm import warm_stdio_workspace

        warm_stdio_workspace(ws_root)
    except Exception:
        logging.exception("Recovered from broad exception handler")
    # Warm the query path of the cached serving engine (page cache, centrality,
    # ANN matrix) so the first tool call never pays a cold-DB spike. Runs on
    # this daemon thread; fail-open.
    try:
        engine = _code_context_engine(str(ws_root))
        _log.info("code query path warmed: %s", engine.warm_query_path())
    except Exception:
        _log.debug("query-path warm failed", exc_info=True)


def _warm_stdio_embedder() -> None:
    """Pre-load the configured embedder so the first semantic query is instant.

    No-op when the default NullEmbedder is active (embedding off by default).
    Only fires when LEMONCROW_CODE_EMBEDDER is set to bge/ollama/openai.
    Fail-open: any failure is logged and ignored.
    """
    try:
        from lemoncrow.infra.embeddings.factory import get_code_embedder

        embedder = get_code_embedder()
        if callable(getattr(embedder, "_load", None)):
            embedder._load()  # type: ignore[attr-defined]
            _log.info("Embedder pre-warmed: %s dim=%s", getattr(embedder, "name", "?"), getattr(embedder, "dim", "?"))
    except Exception:
        _log.debug("Embedder pre-warm failed", exc_info=True)


def _warm_stdio_zoekt_webserver() -> None:
    """Start the zoekt-webserver eagerly for the stdio workspace.

    The persistent ``zoekt-webserver`` keeps the index resident in memory so
    every subsequent query is single-digit-ms.  Starting it here on a daemon
    thread at MCP startup amortises the ~1-2 s shard-load cost over the entire
    session instead of charging it to the first search query.

    Lifecycle: the webserver subprocess is owned by the ``ZoektServer`` instance
    cached in ``get_zoekt_server()``.  ``atexit`` ensures ``server.stop()`` is
    called on clean MCP exit so the child process is reaped properly.  If the
    MCP process is killed or crashes, ``prctl(PR_SET_PDEATHSIG)`` in the child
    ensures the kernel delivers ``SIGTERM`` to the webserver subprocess.

    Fail-open: any error (no index built yet, binary missing) is logged at
    DEBUG and the fallback per-query CLI path remains active.
    """
    try:
        from lemoncrow.infra.code_intel.zoekt.binary import discover_zoekt_binary
        from lemoncrow.infra.code_intel.zoekt.server import get_zoekt_server

        ws = Path(_workspace_root())
        resolution = discover_zoekt_binary(ws)
        if not resolution.available:
            return
        server = get_zoekt_server(ws, resolution=resolution)
        # ensure_started_and_build() registers the binary handle and builds the
        # index if it is missing (e.g. a fresh swarm worktree). Runs on a daemon
        # thread so the build doesn't block the first MCP response.
        server.ensure_started_and_build()
        # _ensure_webserver() starts the persistent HTTP server and waits until
        # the index shards are loaded and queryable (per /api/list readiness).
        url = server._ensure_webserver()
        if url:
            _log.info("zoekt webserver ready at %s", url)
            import atexit

            atexit.register(server.stop)
        else:
            _log.debug("zoekt webserver did not start; using CLI fallback")
    except Exception:
        _log.debug("zoekt pre-warm failed", exc_info=True)


def _shutdown_managed_bash_commands() -> None:
    from lemoncrow.pro.capabilities.tool_supervision.bash_exec import cleanup_managed_commands

    summary = cleanup_managed_commands()
    terminated = summary["terminated"]
    preserved = summary["preserved"]
    if terminated:
        logger.warning(
            "MCP shutdown terminated %d foreground Bash command(s): %s",
            len(terminated),
            ", ".join(str(row["session_id"]) for row in terminated),
        )
    if preserved:
        logger.info(
            "MCP shutdown preserved %d explicit bg=true Bash command(s): %s",
            len(preserved),
            ", ".join(str(row["session_id"]) for row in preserved),
        )


def main() -> None:
    # Phase 1: Absorb wrapper logic into `lc mcp` (zero-config)
    os.environ.setdefault("LEMONCROW_SERVICE_URL", "http://127.0.0.1:8787")
    # The host's own per-window folder wins; only when it says nothing do we
    # detect the git repo root, so global-mode installs on any host still point
    # at a project root rather than the editor's cwd.
    from lemoncrow.core.foundation.paths import host_workspace_root as _host_workspace_root

    _declared_workspace = _host_workspace_root()
    if _declared_workspace is not None:
        os.environ["LEMONCROW_WORKSPACE_ROOT"] = str(_declared_workspace)
    else:
        try:
            import subprocess as _subprocess

            _git_result = _subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if _git_result.returncode == 0:
                os.environ["LEMONCROW_WORKSPACE_ROOT"] = _git_result.stdout.strip()
        except (OSError, _subprocess.SubprocessError):
            _log.debug("git rev-parse workspace-root detection failed", exc_info=True)
    os.environ.setdefault("LEMONCROW_WORKSPACE_ROOT", os.getcwd())
    os.environ.setdefault(
        "LEMONCROW_LESSONS_ROOT", os.path.join(os.environ["LEMONCROW_WORKSPACE_ROOT"], ".lemoncrow/lessons")
    )

    argv = sys.argv[1:]
    if "--version" in argv or "-V" in argv:
        sys.stdout.write(f"lc mcp {SERVER_VERSION}\n")
        return
    if "--root" in argv:
        i = argv.index("--root")
        if i + 1 < len(argv):
            os.environ["LEMONCROW_ROOT"] = argv[i + 1]
    if "--host" in argv:
        i = argv.index("--host")
        if i + 1 < len(argv):
            os.environ["LEMONCROW_AGENT"] = argv[i + 1]

    # Set up file-based logging so logs survive process termination.
    lemoncrow_root = os.environ.get("LEMONCROW_ROOT", str(Path.home() / ".lemoncrow"))
    _setup_file_logging(lemoncrow_root)

    # Register before serve() so the SessionStart hook can find this process
    # and write the Claude session UUID before the first tool call arrives.
    _register_mcp_session()

    # Warm the code-context engine/index once on stdio startup (G10) so the
    # first code-context tool call does not pay cold-start on Zoekt/ast-grep subprocesses. Off the hot path in a daemon thread; fail-open so
    # warming failure never breaks server startup.
    threading.Thread(target=_warm_stdio_code_index, daemon=True).start()
    # Pre-load embedder if explicitly configured (no-op with default NullEmbedder).
    threading.Thread(target=_warm_stdio_embedder, daemon=True).start()
    # Eagerly start the zoekt-webserver so the resident index is queryable from
    # the first explore call.  Lifecycle (atexit stop) wired inside the fn.
    threading.Thread(target=_warm_stdio_zoekt_webserver, daemon=True).start()

    # One-time workspace bootstrap: seed playbooks, add .lemoncrow/.gitignore,
    # and write the .workspace_inited marker.  Daemon thread, fail-open.
    threading.Thread(target=_auto_init_workspace, daemon=True).start()

    _update_thread = threading.Thread(target=_check_auto_update, daemon=True)
    _update_thread.start()
    # Convert process-termination signals into normal Python unwinding so the
    # finally block can terminate every MCP-owned process group. SIGKILL remains
    # uncatchable by definition.
    import signal as _signal

    previous_handlers: list[tuple[_signal.Signals, Any]] = []
    if threading.current_thread() is threading.main_thread():

        def _exit_on_signal(signum: int, _frame: Any) -> None:
            raise SystemExit(128 + signum)

        for signum in (_signal.SIGTERM, _signal.SIGHUP):
            previous_handlers.append((_signal.Signals(signum), _signal.getsignal(signum)))
            _signal.signal(signum, _exit_on_signal)
    try:
        serve()
    finally:
        _shutdown_managed_bash_commands()
        _unregister_mcp_session()
        # If an opt-in auto-update reinstall (git pull + install.sh) is mid-flight
        # when the host disconnects, let it finish rather than killing the daemon
        # thread abruptly and leaving a half-pulled tree / partial install. Returns
        # immediately in the common no-update case (the thread already exited);
        # only blocks when an install is genuinely running (its own cap is ~300s).
        _update_thread.join(timeout=310.0)
        for signum, handler in previous_handlers:
            _signal.signal(signum, handler)
