"""The bash tool + its command-stats/token-savings helpers (public commodity).

Executes shell commands via ``tool_supervision.bash_exec`` and credits output
trimming. Imports the framework/deferral/session/smart_state/fs_access
substrates with no back-dependency on ``mcp_server``.

Extracted verbatim from ``mcp_server.py`` (behaviour-preserving); ``mcp_server``
re-exports these names for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from lemoncrow.gateway.adapters.mcp.deferral import (
    _defer_bash_enabled,
    _deferral_supported,
    _DeferredResult,
)
from lemoncrow.gateway.adapters.mcp.framework import TOOLS, mcp_tool
from lemoncrow.gateway.adapters.mcp.fs_access import _claude_additional_dirs
from lemoncrow.gateway.adapters.mcp.ledger import _check_redundant_file_dump, _request_bridge_id
from lemoncrow.gateway.adapters.mcp.session_state import (
    _forget_mcp_managed_bash,
    _record_mcp_managed_bash,
    live_managed_bash_ids,
)
from lemoncrow.gateway.adapters.mcp.smart_state import (
    _STATE_LOCK,
    _acquire_smart_state_flock,
    _read_smart_state,
    _release_smart_state_flock,
    _tool_call_tokens_saved,
    _write_smart_state,
)

logger = logging.getLogger(__name__)


_VANILLA_BASH_OUTPUT_CHARS = 30_000


_BASH_STATS_MAX_KEYS = 200


_BASH_STATS_PRUNE_TO = 150


_BASH_KEY_RUNNER_PAIRS = {("uv", "run"), ("npm", "run"), ("pnpm", "run"), ("yarn", "run"), ("poetry", "run")}


_BASH_KEY_GROUP_HEADS = frozenset(
    {
        "git",
        "docker",
        "kubectl",
        "oc",
        "cargo",
        "go",
        "npm",
        "pnpm",
        "yarn",
        "bun",
        "uv",
        "pip",
        "pip3",
        "aws",
        "gcloud",
        "az",
        "make",
        "poetry",
        "bundle",
        "gh",
        "terraform",
        "helm",
    }
)


_BASH_KEY_ENV_ASSIGN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")


_BASH_KEY_CD_PREFIX_RE = re.compile(r"^\s*cd\s+[^&|;]+&&\s*")


def _bash_omitted_tokens_saved(polled: dict[str, Any], chars_omitted: int) -> int:
    """Tokens credited for bash output trimming, against an honest baseline.

    Vanilla Claude Code truncates Bash output at ~30k chars itself, so the
    naive cost of the omitted chars is capped at what vanilla would actually
    have put in context: a 10 MB build log is NOT ~2.5M tokens saved. chars/4
    is the standard chars-per-token estimate.
    """
    if chars_omitted <= 0:
        return 0
    shown = len(str(polled.get("stdout") or "")) + len(str(polled.get("stderr") or ""))
    naive = min(shown + chars_omitted, _VANILLA_BASH_OUTPUT_CHARS)
    return max(0, naive - shown) // 4


def _bash_command_key(command: str) -> str:
    """Normalize a shell command to a short aggregation key (``git status``,
    ``uv run pytest``, ``make lint``): leading cd/env prefixes dropped, only
    the head of the first pipeline segment kept, flags and paths stripped."""
    body = _BASH_KEY_CD_PREFIX_RE.sub("", command)
    body = re.split(r"[|;&]", body, maxsplit=1)[0].strip()
    try:
        tokens = shlex.split(body)
    except ValueError:
        tokens = body.split()
    tokens = [t for t in tokens if not _BASH_KEY_ENV_ASSIGN_RE.fullmatch(t)]
    words: list[str] = []
    for tok in tokens[:3]:
        if tok.startswith("-"):
            break
        words.append(tok.rsplit("/", 1)[-1])
    if not words:
        return ""
    take = 2 if words[0] in _BASH_KEY_GROUP_HEADS else 1
    if len(words) >= 2 and (words[0], words[1]) in _BASH_KEY_RUNNER_PAIRS:
        take = 3
    return " ".join(words[:take])[:60]


def _record_bash_command_stats(command: str, *, shipped_chars: int, omitted_chars: int) -> None:
    """Fold one finished bash call into smart_state's per-command spend ledger."""
    key = _bash_command_key(command)
    if not key:
        return
    with _STATE_LOCK:
        _flock = _acquire_smart_state_flock()
        try:
            state = _read_smart_state()
            cmds = state.get("bash_commands")
            if not isinstance(cmds, dict):
                cmds = {}
            row = cmds.get(key)
            if not isinstance(row, dict):
                row = {"calls": 0, "shipped_chars": 0, "omitted_chars": 0}
            row["calls"] = int(row.get("calls", 0) or 0) + 1
            row["shipped_chars"] = int(row.get("shipped_chars", 0) or 0) + max(0, shipped_chars)
            row["omitted_chars"] = int(row.get("omitted_chars", 0) or 0) + max(0, omitted_chars)
            cmds[key] = row
            if len(cmds) > _BASH_STATS_MAX_KEYS:
                # Bounded: keep the biggest shippers -- exactly the rows the
                # audit exists to surface -- and drop the long tail.
                ranked = sorted(
                    cmds.items(),
                    key=lambda kv: int(kv[1].get("shipped_chars", 0) or 0) if isinstance(kv[1], dict) else 0,
                    reverse=True,
                )
                cmds = dict(ranked[:_BASH_STATS_PRUNE_TO])
            state["bash_commands"] = cmds
            _write_smart_state(state)
        finally:
            _release_smart_state_flock(_flock)


# A host that caps MCP tool calls backgrounds the call when it runs long and
# hands back a *task* id of its own -- Claude Code says "moved to the background
# as task k31b54d8q ... use TaskStop with task_id". That id addresses the host's
# task, never a managed shell, so feeding it back as bash(id=...) can only fail;
# the shell handle is the shorter id inside the eventual completion payload.
# Host task ids seen so far are 9 lowercase alphanumerics, managed shell ids 6,
# so the shape is worth calling out -- as a hint only, never as a rejection.
_HOST_TASK_ID_RE = re.compile(r"^[0-9a-z]{9}$")


def _unknown_session_hint(session_id: str) -> str:
    """Trailing clause naming what the caller could have polled instead."""
    parts: list[str] = []
    if _HOST_TASK_ID_RE.match(session_id):
        parts.append("looks like a host background-task id (TaskStop/task output take it), not a shell id")
    live = [sid for sid in live_managed_bash_ids() if sid != session_id]
    parts.append("live sessions: " + ", ".join(live[:5]) if live else "no live sessions in this MCP process")
    return " -- " + "; ".join(parts)


def _default_bash_soft_timeout() -> int:
    try:
        return max(1, int(os.environ.get("LEMONCROW_BASH_SOFT_TIMEOUT", "120")))
    except ValueError:
        return 120


_DEFAULT_BASH_SOFT_TIMEOUT = _default_bash_soft_timeout()


# ── Idle-aware inline wait ───────────────────────────────────────────
# A foreground command blocks this call for up to its `timeout` budget, then a
# running handle is returned (a sub-1h budget never kills the command). Waiting
# the *full* budget on a command that has deadlocked -- a producer/consumer both
# parked, a consumer blocked on a dead socket -- burns wall-clock the model could
# spend reacting. So, past a floor, hand the handle back early once the command
# has made no *progress* for a grace window, where progress = new output OR the
# process group accruing CPU/IO. Keying on CPU/IO (not stdout alone) is what
# separates a genuinely-stuck command (0 output, 0 CPU, parked in a syscall) from
# a silent-but-working build (0 output, CPU/IO climbing): the latter keeps
# refreshing the idle timer and runs to the full budget. Purely additive -- it
# can only return *earlier* than the timeout, never later, and never kills
# anything. Linux-only (needs /proc for the CPU/IO signal); a no-op elsewhere.
def _bash_idle_return_enabled() -> bool:
    raw = os.environ.get("LEMONCROW_BASH_IDLE_RETURN", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _bash_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


# Minimum elapsed wait before an idle command's handle is handed back early. A
# command whose budget is <= this floor is never cut early, so short/default
# waits keep today's behaviour exactly; the idle path only ever bites a call the
# model deliberately budgeted a longer timeout for.
_BASH_IDLE_FLOOR_S = _bash_env_float("LEMONCROW_BASH_IDLE_FLOOR", 120.0)
# No-progress span that flips a still-running command from "working" to "stuck".
_BASH_IDLE_GRACE_S = _bash_env_float("LEMONCROW_BASH_IDLE_GRACE", 75.0)
# How often progress is sampled while waiting.
_BASH_IDLE_TICK_S = _bash_env_float("LEMONCROW_BASH_IDLE_TICK", 3.0)


def _proc_session_cpu_io(sid: int) -> tuple[int, int] | None:
    """Sum CPU ticks (utime+stime) and IO bytes (read+write) over every process
    in session ``sid``. The managed command runs under ``start_new_session``, so
    its whole descendant tree shares the leader pid as session id. Returns
    ``None`` when /proc is unavailable (non-Linux) or the session has no live
    process -- signalling "no progress measurement available" to the caller, which
    then stays conservative and never cuts the command early.
    """
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    total_cpu = 0
    total_io = 0
    found = False
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        # comm (field 2) is parenthesised and may contain spaces/')' -- split on
        # the last ')' so the fixed fields after it tokenise cleanly. rest[0] is
        # field 3 (state); field N maps to rest[N-3].
        close = data.rfind(b")")
        if close == -1:
            continue
        rest = data[close + 2 :].split()
        if len(rest) < 13:
            continue
        try:
            if int(rest[3]) != sid:  # field 6: session id
                continue
        except ValueError:
            continue
        found = True
        try:
            total_cpu += int(rest[11]) + int(rest[12])  # fields 14/15: utime/stime
        except ValueError:
            pass
        try:
            with open(f"/proc/{entry}/io", "rb") as handle:
                for line in handle:
                    if line.startswith((b"read_bytes:", b"write_bytes:")):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_io += int(parts[1])
        except (OSError, ValueError):
            pass
    if not found:
        return None
    return (total_cpu, total_io)


# (output_bytes, cpu_ticks, io_bytes, have_proc)
_BashProgress = tuple[int, int, int, bool]


def _bash_progress_probe(session_id: str, peek_fn: Callable[[str], dict[str, Any]]) -> _BashProgress | None:
    """Snapshot the command's monotonic progress signals, or ``None`` if it is no
    longer running / not peekable."""
    try:
        snap = peek_fn(session_id)
    except Exception:  # a peek failure just means "no reading right now"
        return None
    if snap.get("status") != "running":
        return None
    out_bytes = 0
    for key in ("log_file", "log_file_stderr"):
        path = snap.get(key)
        if isinstance(path, str) and path:
            try:
                out_bytes += os.path.getsize(path)
            except OSError:
                pass
    cpu = 0
    io = 0
    have_proc = False
    pid = snap.get("pid")
    if isinstance(pid, int):
        cpu_io = _proc_session_cpu_io(pid)
        if cpu_io is not None:
            cpu, io = cpu_io
            have_proc = True
    return (out_bytes, cpu, io, have_proc)


def _bash_made_progress(before: _BashProgress, after: _BashProgress) -> bool:
    """True if any monotonic signal advanced between two probes."""
    if after[0] > before[0]:  # new output bytes
        return True
    if before[3] and after[3] and (after[1] > before[1] or after[2] > before[2]):
        return True  # CPU or IO advanced (both probes had the /proc signal)
    return False


def _start_idle_deadline_watcher(
    session_id: str,
    timeout_s: float,
    fired: threading.Event,
    fire: Callable[[], None],
    peek_fn: Callable[[str], dict[str, Any]],
) -> None:
    """Deferred-path replacement for a single ``Timer(timeout, fire)``: fire at
    the absolute ``timeout`` (unchanged, so the MCP response never blocks past
    it) OR early once the command has made no progress for the grace window past
    the floor. Exits without firing if ``fired`` is already set -- natural
    completion won the race. A ``finally`` guarantees the result is never left
    orphaned even if sampling raises.
    """
    floor = min(timeout_s, _BASH_IDLE_FLOOR_S)
    grace = _BASH_IDLE_GRACE_S
    tick = _BASH_IDLE_TICK_S
    idle_on = _bash_idle_return_enabled()

    def _run() -> None:
        try:
            start = time.monotonic()
            idle_since = start
            last = _bash_progress_probe(session_id, peek_fn)
            while True:
                elapsed = time.monotonic() - start
                to_deadline = timeout_s - elapsed
                if to_deadline <= 0:
                    fire()  # absolute budget spent -- fire exactly at the deadline
                    return
                # Wake at the next sample OR the deadline, whichever is sooner, so
                # a small budget is honoured precisely (a coarse tick must never
                # overshoot it). Returns True if natural completion set `fired`.
                if fired.wait(min(tick, to_deadline)):
                    return
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= timeout_s:
                    fire()
                    return
                if not idle_on:
                    continue
                probe = _bash_progress_probe(session_id, peek_fn)
                if probe is None:
                    continue  # transiently unpeekable; completion cb will finish
                if not probe[3]:
                    # No /proc signal: can't tell stuck from silent-working, so
                    # stay conservative and never cut early on this platform.
                    idle_since = now
                    last = probe
                    continue
                if last is None or not last[3] or _bash_made_progress(last, probe):
                    idle_since = now
                last = probe
                if elapsed >= floor and (now - idle_since) >= grace:
                    fire()
                    return
        except Exception:  # a watcher crash must never orphan the deferred result
            logger.debug("bash idle watcher error for %s", session_id, exc_info=True)
        finally:
            if not fired.is_set():
                fire()

    thread = threading.Thread(target=_run, name="lemoncrow-bash-idle", daemon=True)
    thread.start()


def _run_bash_tool(
    command: str = "",
    timeout: int | None = None,
    cwd: str | None = None,
    max_lines: int = 200,
    max_output_tokens: int | None = None,
    background: bool = False,
    session_id: str | None = None,
    action: Literal["run", "poll", "kill", "status", "update", "send"] = "run",
    interactive: bool = False,
    input_text: str | None = None,
    idle_ttl: int | None = None,
) -> dict[str, Any] | _DeferredResult:
    """Execute a shell command and return compact structured output."""
    from lemoncrow.pro.capabilities.tool_supervision.bash_exec import (
        classify_command,
        execute_inline_op,
        peek_managed_command,
        poll_managed_command,
        send_managed_input,
        start_managed_command,
        update_managed_command,
    )

    def _render_grep_stdout(payload: dict[str, Any]) -> str:
        blocks = payload.get("content", [])
        if isinstance(blocks, list):
            texts: list[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
            if texts:
                normalized: list[str] = []
                for line in "\n".join(texts).splitlines():
                    if line.startswith("@@ "):
                        continue
                    normalized.append(line)
                return "\n".join(normalized)
        matches = payload.get("matches")
        if isinstance(matches, list):
            return json.dumps(matches, ensure_ascii=False)
        return json.dumps(payload, ensure_ascii=False)

    workspace = os.environ.get("CLAUDE_WORKSPACE_ROOT", os.getcwd())
    # A misconfigured CLAUDE_WORKSPACE_ROOT -- e.g. a host path that leaked into
    # a container via the environment and does not exist here -- would make every
    # cwd-less command fail with a raw FileNotFoundError from Popen, surfaced as
    # an opaque "MCP error -32000". Fall back to the process cwd (always a real
    # directory) so the command still runs instead of hard-failing.
    if not Path(workspace).is_dir():
        workspace = os.getcwd()
    effective_cwd = cwd or workspace

    if action in {"poll", "kill", "status", "update", "send"}:
        if not session_id:
            raise ValueError(f"session_id is required for shell action={action}")

        def _session_not_found() -> dict[str, Any]:
            return {
                "status": "error",
                "session_id": session_id,
                "stderr": f"unknown shell session: {session_id}{_unknown_session_hint(session_id or '')}",
                "exit_code": -1,
            }

        try:
            if action == "send":
                # Feed an interactive session's stdin and return the output delta.
                # `timeout` here is only how long to wait for that delta -- the
                # session itself lives on under its own idle-TTL.
                return send_managed_input(
                    session_id,
                    input_text or "",
                    wait=float(timeout) if timeout is not None else 30.0,
                )
            if action == "kill":
                result = poll_managed_command(session_id, cancel=True)
                _forget_mcp_managed_bash(session_id)
                return result
            if action == "status":
                # Single non-blocking check -- unlike `poll`, never waits for the
                # command to finish and never reaps the session.
                result = peek_managed_command(session_id)
                if result.get("status") != "running":
                    _forget_mcp_managed_bash(session_id)
                return result
            if action == "update":
                if timeout is None:
                    raise ValueError("timeout is required for shell action=update")
                if timeout <= 0:
                    raise ValueError("timeout must be positive")
                return update_managed_command(session_id, timeout)
            # Block until the managed command finishes, is cancelled, or the
            # caller's optional poll timeout expires. With no timeout, wait
            # indefinitely (subject to the managed command's own deadline).
            delay = 0.02
            poll_deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
            while True:
                poll_result = poll_managed_command(session_id)
                if poll_result.get("status") != "running":
                    _forget_mcp_managed_bash(session_id)
                    return poll_result
                if poll_deadline is not None:
                    remaining = poll_deadline - time.monotonic()
                    if remaining <= 0:
                        return poll_result
                    time.sleep(min(delay, remaining))
                else:
                    time.sleep(delay)
                delay = min(delay * 2, 0.5)
        except KeyError:
            _forget_mcp_managed_bash(session_id)
            return _session_not_found()
    if not command.strip():
        raise ValueError("command is required for shell action=run")
    if timeout is None:
        timeout = _DEFAULT_BASH_SOFT_TIMEOUT

    if interactive and background:
        raise ValueError(
            "interactive=true cannot be combined with bg=true: an interactive session's stdin dies with this MCP session"
        )

    # Only the API's explicit bg=true flag grants permission to survive MCP
    # shutdown. A trailing ampersand still gets the fast managed-command path,
    # but remains owned by this MCP session and is cleaned up with foreground
    # work when the session exits.
    explicit_background = background

    # A trailing `&` (but not `&&`) means "run in background": strip it and
    # force background mode so the command runs as a managed LemonCrow session
    # with a session_id + pid the model can poll/cancel.  Passing the `&`
    # verbatim to `bash -c "cmd &"` would fork an untracked grandchild that
    # exits from bash immediately with empty output and no handle to follow.
    _stripped = command.rstrip()
    if _stripped.endswith("&") and not _stripped.endswith("&&"):
        command = _stripped[:-1].rstrip()
        background = True

    _shell_workspace_root = Path(workspace).resolve()
    policy = classify_command(
        command,
        allowed_write_roots=[_shell_workspace_root, *_claude_additional_dirs(_shell_workspace_root)],
        cwd=effective_cwd,
    )

    if policy.action == "block":
        return {
            "status": "blocked",
            "stderr": policy.reason,
            "exit_code": -1,
            "blocked": True,
            "blocked_reason": policy.reason,
        }

    # Soft external-compactor pass-through (e.g. rtk, when detected and
    # enabled -- see tool_supervision.external_compactors): substitute the
    # binary-prefixed command and let it run through the normal managed-
    # command path below like any other command, so timeout/background/
    # polling behave identically. Never short-circuits -- a failure here is
    # just the wrapped command's own exit code, not a fallback trigger,
    # since re-running a side-effecting command a second time would be wrong.
    # Pre-wrapper form: the per-command spend ledger attributes output cost to
    # the real command family, not to the compactor binary's path.
    _stats_command = command
    if policy.action == "rewrite" and policy.rewrite_target == "external_compactor" and policy.rewrite_payload:
        _rewritten_command = str(policy.rewrite_payload.get("command") or "")
        if _rewritten_command:
            command = _rewritten_command

    # Pipeline seek rewrite (classify_command): `od <bigfile> | tail` -> an
    # in-place `od -j <offset>` that formats only the tail region instead of the
    # whole file. Substitute the command and run it through the normal managed
    # path like external_compactor; carry the note so the model sees the rewrite.
    _pipeline_note = ""
    if policy.action == "rewrite" and policy.rewrite_target == "pipeline_seek" and policy.rewrite_payload:
        _seek_command = str(policy.rewrite_payload.get("command") or "")
        if _seek_command:
            command = _seek_command
            _pipeline_note = str(policy.rewrite_payload.get("note") or "")

    if policy.action == "rewrite" and policy.rewrite_target in {"cat", "head", "tail", "wc"} and policy.rewrite_payload:
        _stdout, _stderr, _exit = execute_inline_op(policy.rewrite_target, policy.rewrite_payload, effective_cwd)
        return {
            "stdout": _stdout,
            "stderr": _stderr,
            "exit_code": _exit,
            "truncated": False,
            "lines_omitted": 0,
            "duration_ms": 0,
        }

    if policy.action == "rewrite" and policy.rewrite_target == "read" and policy.rewrite_payload:
        raw_file_path = str(policy.rewrite_payload.get("file_path") or "").strip()
        if raw_file_path:
            target_path = Path(raw_file_path)
            if not target_path.is_absolute():
                target_path = (Path(effective_cwd) / target_path).resolve()
            read_handler: Callable[[dict[str, Any]], Any] = TOOLS["read"]["handler"]
            rewritten = cast(dict[str, Any], read_handler({"path": str(target_path), "full": True}))
            rewritten_stdout = str(rewritten.get("content") or "")
            return {
                "stdout": rewritten_stdout,
                "stderr": "",
                "exit_code": 0,
                "truncated": False,
                "lines_omitted": 0,
                "duration_ms": 0,
            }

    if policy.action == "rewrite" and policy.rewrite_target == "grep" and policy.rewrite_payload:
        raw_search_path = str(policy.rewrite_payload.get("file_path") or ".")
        content_regex = cast(str | None, policy.rewrite_payload.get("content_regex"))
        ignore_case = bool(policy.rewrite_payload.get("ignore_case", False))
        file_type = cast(str | None, policy.rewrite_payload.get("type"))

        resolved_search_path = Path(raw_search_path)
        if not resolved_search_path.is_absolute():
            resolved_search_path = (Path(effective_cwd) / resolved_search_path).resolve()
        # Glob patterns from the payload (e.g. --include / -g flags) take
        # precedence; fall back to "**/*" for directory-wide searches.
        payload_globs = policy.rewrite_payload.get("glob")
        if payload_globs:
            glob_patterns = payload_globs if isinstance(payload_globs, list) else [payload_globs]
        elif resolved_search_path.is_dir():
            glob_patterns = ["**/*"]
        else:
            glob_patterns = None
        grep_args: dict[str, Any] = {
            # Pass the cwd-resolved absolute path: tool_grep resolves a relative
            # path against CLAUDE_WORKSPACE_ROOT, which would search the wrong
            # directory when the shell call's cwd differs from the workspace.
            "path": str(resolved_search_path),
            "content_regex": content_regex,
            "file_glob_patterns": glob_patterns,
            "ignore_case": ignore_case,
            "summary": False,
            "output_mode": cast(
                Literal[
                    "ranked_file_map",
                    "file_paths_with_content",
                    "file_paths_only",
                    "file_paths_with_match_count",
                ],
                policy.rewrite_payload.get("output_mode", "file_paths_with_content"),
            ),
            "lines_before": int(policy.rewrite_payload.get("lines_before", 0)),
            "lines_after": int(policy.rewrite_payload.get("lines_after", 0)),
        }
        if file_type:
            grep_args["type"] = file_type
        grep_handler: Callable[[dict[str, Any]], Any] = TOOLS["grep"]["handler"]
        rewritten = cast(dict[str, Any], grep_handler(grep_args))
        rewritten_stdout = _render_grep_stdout(rewritten)

        # If the original command had a pipe tail (e.g. ``grep ... | head -20``),
        # feed the grep output through it so the agent gets the trimmed result
        # rather than the full unpiped output.
        pipe_remainder = str(policy.rewrite_payload.get("pipe_remainder") or "")
        if pipe_remainder:
            try:
                import subprocess as _sp

                pipe_proc = _sp.run(
                    ["bash", "-c", pipe_remainder],
                    input=rewritten_stdout,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                rewritten_stdout = pipe_proc.stdout
                if pipe_proc.returncode != 0 and pipe_proc.stderr:
                    rewritten_stdout = rewritten_stdout + pipe_proc.stderr
            except (OSError, ValueError, _sp.TimeoutExpired):
                pass  # fall through with unpiped output on any error

        return {
            "stdout": rewritten_stdout,
            "stderr": "",
            "exit_code": 0,
            "truncated": False,
            "lines_omitted": 0,
            "duration_ms": 0,
        }

    if policy.action == "rewrite" and policy.rewrite_target == "web_fetch" and policy.rewrite_payload:
        _wf_url = str(policy.rewrite_payload.get("url") or "").strip()
        if _wf_url:
            try:
                from lemoncrow.core.capabilities.web_fetch import fetch_url

                _wf = fetch_url(_wf_url)
                _wf_out = _wf.get("content") if isinstance(_wf, dict) else str(_wf)
            except Exception as _wf_exc:
                _wf_out = f"[web_fetch] {_wf_exc}"
            return {
                "stdout": str(_wf_out or ""),
                "stderr": "",
                "exit_code": 0,
                "truncated": False,
                "lines_omitted": 0,
                "duration_ms": 0,
            }

    if policy.action == "rewrite" and policy.rewrite_target == "find_glob" and policy.rewrite_payload:
        _fg_pat = str(policy.rewrite_payload.get("glob") or "*")
        _fg_path = str(policy.rewrite_payload.get("path") or ".")
        try:
            _fg_base = Path(_fg_path) if Path(_fg_path).is_absolute() else (Path(effective_cwd) / _fg_path)
            _fg_hits = sorted(str(p.relative_to(_fg_base)) for p in _fg_base.rglob(_fg_pat) if p.is_file())
        except Exception:
            _fg_hits = []
        _fg_out = "\n".join(_fg_hits[:300]) if _fg_hits else "(no files match)"
        if len(_fg_hits) > 300:
            _fg_out += f"\n... ({len(_fg_hits) - 300} more)"
        return {
            "stdout": _fg_out,
            "stderr": "",
            "exit_code": 0,
            "truncated": False,
            "lines_omitted": 0,
            "duration_ms": 0,
        }

    if policy.action == "rewrite" and policy.rewrite_target == "read_range" and policy.rewrite_payload:
        _rr_spec = str(policy.rewrite_payload.get("spec") or "").strip()
        if _rr_spec and ":" in _rr_spec:
            _rr_fp, _, _rr_rng = _rr_spec.rpartition(":")
            _rr_target = Path(_rr_fp) if Path(_rr_fp).is_absolute() else (Path(effective_cwd) / _rr_fp).resolve()
            try:
                _rr = cast(dict[str, Any], TOOLS["read"]["handler"]({"path": str(_rr_target), "range": _rr_rng}))
                _rr_out = _rr.get("content") if isinstance(_rr, dict) else str(_rr)
            except Exception as _rr_exc:
                _rr_out = f"[read] {_rr_exc}"
            return {
                "stdout": str(_rr_out or ""),
                "stderr": "",
                "exit_code": 0,
                "truncated": False,
                "lines_omitted": 0,
                "duration_ms": 0,
            }

    # Literal-text "search" rewrite: simple rg/grep queries without regex
    # metacharacters are rewritten to the grep tool (content_regex), which
    # handles the search natively without requiring rg/grep on PATH.
    if policy.action == "rewrite" and policy.rewrite_target == "search" and policy.rewrite_payload:
        _sq = str(policy.rewrite_payload.get("query") or "")
        _search_path = str(policy.rewrite_payload.get("path") or ".")
        if _sq:
            _resolved_sp = Path(_search_path)
            if not _resolved_sp.is_absolute():
                _resolved_sp = (Path(effective_cwd) / _search_path).resolve()
            _search_payload: dict[str, Any] = {
                "path": str(_resolved_sp),
                "content_regex": _sq,
                "ignore_case": False,
                "output_mode": "file_paths_with_content",
                "lines_after": 0,
                "lines_before": 0,
            }
            _search_handler: Callable[[dict[str, Any]], Any] = TOOLS["grep"]["handler"]
            _rewritten = cast(dict[str, Any], _search_handler(_search_payload))
            _rewritten_stdout = _render_grep_stdout(_rewritten)
            return {
                "stdout": _rewritten_stdout,
                "stderr": "",
                "exit_code": 0,
                "truncated": False,
                "lines_omitted": 0,
                "duration_ms": 0,
            }

    # One execution model: every command runs as a managed session; the only
    # variable is how long we block inline before returning a poll handle.
    #   background → 0s (detach immediately, poll/cancel by session)
    #   default    → full timeout (block until the command finishes, or hand
    #                back a still-running session handle at the deadline --
    #                see the deferred branch below.  The command itself is NOT
    #                killed just because this call stopped waiting for it.)
    inline_wait = 0.0 if background else float(timeout)

    started = start_managed_command(
        command,
        cwd=effective_cwd,
        timeout=timeout,
        max_lines=max_lines,
        max_chars=max_output_tokens * 4 if max_output_tokens is not None else None,
        note=_pipeline_note,
        explicit_background=explicit_background,
        interactive=interactive,
        idle_ttl=float(idle_ttl) if idle_ttl is not None else None,
        owner=_request_bridge_id(),
    )
    managed_id = str(started.get("session_id") or "")
    if started.get("status") != "running" or not managed_id:
        return started  # blocked by policy
    _record_mcp_managed_bash(started)

    if interactive:
        # An interactive session never finishes on its own -- hand the handle
        # back immediately; the model feeds it with action="send".
        return started

    # Phase 2: foreground deferral. For the block-until-done case (a foreground
    # run, where inline_wait covers the full timeout), hand the pool worker back
    # immediately and let bash_exec's watcher finalize the response when the
    # command completes. Gated by the kill switch AND by _deferral_supported() so
    # synchronous callers (CLI / in-process runtime / direct test calls), which
    # cannot process a deferred marker, keep today's busy-poll behavior.
    if _defer_bash_enabled() and _deferral_supported() and inline_wait >= float(timeout):
        from lemoncrow.pro.capabilities.tool_supervision.bash_exec import (
            peek_managed_command,
            register_completion,
        )

        def _collect() -> dict[str, Any]:
            # A soft-deadline resolution (see _register below) races the real
            # completion callback, so this can run before the command is
            # actually done -- peek first (non-blocking, never reaps) and only
            # fall through to the terminal, reaping poll once it agrees the
            # process has actually finished. Covers a command that's
            # genuinely still running past `timeout` (e.g. a backgrounded
            # server a task wants left running).
            snapshot = peek_managed_command(managed_id)
            if snapshot.get("status") == "running":
                # This only runs via the deadline watcher (register_completion
                # fires on natural completion → status != running). A still-
                # running, NOT-over-budget snapshot therefore means the watcher
                # fired EARLY on the idle rule, not at the absolute deadline --
                # tag it so the model is told "looks stuck", not "still working".
                if not snapshot.get("over_budget"):
                    snapshot["idle_return"] = True
                return snapshot
            # The process has finished when this runs; poll once for the terminal
            # result and apply the identical terminal transforms the inline path
            # does, so the deferred result dict matches the synchronous one.
            polled = poll_managed_command(managed_id)
            _forget_mcp_managed_bash(managed_id)
            polled.pop("session_id", None)
            polled.pop("status", None)
            # Read without dropping: the renderer needs chars_omitted to tell
            # a trivial trim (no footer) from a real one.
            chars_omitted = int(polled.get("chars_omitted", 0) or 0)
            ts = _bash_omitted_tokens_saved(polled, chars_omitted)
            if ts > 0:
                _tool_call_tokens_saved.value = ts
            _record_bash_command_stats(
                _stats_command,
                shipped_chars=len(str(polled.get("stdout") or "")) + len(str(polled.get("stderr") or "")),
                omitted_chars=chars_omitted,
            )
            return polled

        def _register(cb: Callable[[], None]) -> bool:
            fired = threading.Event()

            def _once() -> None:
                if fired.is_set():
                    return
                fired.set()
                cb()

            armed = register_completion(managed_id, _once)
            if not armed:
                return False
            # Deadline watcher (replaces a single Timer(timeout)): fires _once at
            # the absolute `timeout` -- so the MCP response never blocks past it,
            # exactly as before -- OR early once the command has gone idle (no
            # output/CPU/IO progress) past the floor. Never kills anything: the
            # managed session keeps running untouched; the model gets a
            # session_id back to poll again, keep working, or action="kill" it.
            # register_completion still wins the race on natural completion (it
            # sets `fired`, waking the watcher so it exits without firing again).
            _start_idle_deadline_watcher(managed_id, float(timeout), fired, _once, peek_managed_command)
            return True

        return _DeferredResult(collect=_collect, register=_register)

    # When the inline wait covers the full timeout budget, allow a short grace
    # before giving up and returning a running handle -- covers a command that
    # finishes (or that bash_exec's own hard cap kills) right around the
    # deadline, so we return that reaped terminal result instead of a handle
    # to a session that's about to change state anyway.
    if inline_wait >= float(timeout):
        inline_wait = float(timeout) + 10.0
    start_wait = time.monotonic()
    deadline = start_wait + inline_wait
    # Idle early-return for the synchronous path (CLI/tests; the MCP server takes
    # the deferred branch above). Same rule as the deferred watcher: never before
    # the floor, then hand back the running handle once progress has stalled for
    # the grace window. Disabled for sub-floor waits so short commands are
    # untouched.
    idle_floor = min(inline_wait, _BASH_IDLE_FLOOR_S)
    idle_on = _bash_idle_return_enabled() and inline_wait > idle_floor
    idle_since = start_wait
    last_probe: _BashProgress | None = None
    next_probe_at = start_wait + _BASH_IDLE_TICK_S
    delay = 0.02
    polled: dict[str, Any] = started
    while True:
        polled = poll_managed_command(managed_id)
        if polled.get("status") != "running":
            break
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            return polled  # still running at the window edge — poll later
        if idle_on and now >= next_probe_at:
            next_probe_at = now + _BASH_IDLE_TICK_S
            probe = _bash_progress_probe(managed_id, peek_managed_command)
            if probe is not None and probe[3]:
                if last_probe is None or not last_probe[3] or _bash_made_progress(last_probe, probe):
                    idle_since = now
                last_probe = probe
                if (now - start_wait) >= idle_floor and (now - idle_since) >= _BASH_IDLE_GRACE_S:
                    polled["idle_return"] = True  # cut early: looks stuck, not budget-spent
                    return polled  # progress stalled past the grace window
            elif probe is not None:
                idle_since = now  # no /proc signal: stay conservative
                last_probe = probe
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, 0.5)

    # Finished inline: present as a plain synchronous result. The managed
    # session is already reaped, so status/session_id would only invite a
    # useless poll turn; exit_code/stderr carry the terminal state.
    _forget_mcp_managed_bash(managed_id)
    polled.pop("session_id", None)
    polled.pop("status", None)
    # Read without dropping: the renderer needs chars_omitted (see above).
    chars_omitted = int(polled.get("chars_omitted", 0) or 0)
    ts = _bash_omitted_tokens_saved(polled, chars_omitted)
    if ts > 0:
        _tool_call_tokens_saved.value = ts
    _record_bash_command_stats(
        _stats_command,
        shipped_chars=len(str(polled.get("stdout") or "")) + len(str(polled.get("stderr") or "")),
        omitted_chars=chars_omitted,
    )
    return polled


def _render_bash_text(result: dict[str, Any]) -> str:
    """Render shell output as compact text while preserving structured internals."""
    exit_code = result.get("exit_code")
    stdout = str(result.get("stdout") or "")
    stderr = str(result.get("stderr") or "")
    blocked = bool(result.get("blocked"))
    blocked_reason = str(result.get("blocked_reason") or "")
    truncated = bool(result.get("truncated"))
    lines_omitted = result.get("lines_omitted")
    status = str(result.get("status") or "")
    session_id = str(result.get("session_id") or "")
    explicit_background = bool(result.get("explicit_background"))

    parts: list[str] = []
    if "updated" in result:
        # action="update" response -- a distinct shape from the plain
        # running/status payloads below, so render it up front and return.
        remaining_ms = result.get("timeout_remaining_ms")
        if result.get("updated"):
            remaining_txt = f"{int(remaining_ms) // 1000}s" if isinstance(remaining_ms, int) else "?"
            parts.append(f"kill deadline updated, {remaining_txt} left id={session_id}")
        else:
            parts.append(f"update failed: session already {status} id={session_id}")
        return "\n".join(parts).strip()
    if status == "running":
        over_budget = bool(result.get("over_budget"))
        idle_return = bool(result.get("idle_return"))
        if explicit_background:
            parts.append(f"background running id={session_id}; bash(id={session_id}) waits for it")
        elif result.get("interactive"):
            parts.append(f"interactive session id={session_id}")
        elif idle_return:
            # Cut early because progress stalled -- NOT because the budget ran
            # out. Do not steer the model to just re-wait (that re-blocks on a
            # stuck command); tell it the handle is a decision point.
            parts.append(
                f"no progress ~{int(_BASH_IDLE_GRACE_S)}s — likely stuck. id={session_id} still running "
                f"(not killed): bash(id={session_id}, action=kill), or move on"
            )
        elif over_budget:
            parts.append(f"still running id={session_id}; bash(id={session_id}) waits for it — don't sleep-poll")
        else:
            parts.append(f"running id={session_id}")
    elif status and status != "completed":
        # Terminal states (cancelled/timed_out): the session is reaped, its id
        # can never be polled again -- don't ship a dead handle. A clean
        # "completed" is implied by output + exit_code and costs a line.
        # "blocked" with a reason skips the bare state word too -- every
        # blocked_reason already says "blocked".
        if not (status == "blocked" and blocked_reason):
            parts.append(status)
    # Log paths are recovery pointers: folded into the lossy-view marker
    # (tail slice / truncation) instead of standalone log_file= lines. A spill
    # hint already names a full-output path, so logs are skipped there.
    log_file = str(result.get("log_file") or "")
    log_file_stderr = str(result.get("log_file_stderr") or "")
    tail_lines = result.get("tail_lines")
    spill_hint = str(result.get("spill_hint") or "")
    if log_file and log_file_stderr:
        # The two stream logs differ only in suffix -- brace the divergence
        # ({stdout.txt, stderr.txt}) instead of repeating the directory + id.
        i = len(os.path.commonprefix([log_file, log_file_stderr]))
        i = max(log_file.rfind(c, 0, i) + 1 for c in "./")
        if i:
            log_paths = f"{log_file[:i]}{{{log_file[i:]}, {log_file_stderr[i:]}}}"
        else:
            log_paths = f"{log_file} {log_file_stderr}"
    else:
        log_paths = log_file or log_file_stderr
    log_ptr = f"; full: {log_paths}" if log_paths and not spill_hint else ""
    if status == "running" and log_paths:
        parts.append(f"[logs: {log_paths}]")
    if isinstance(tail_lines, int) and tail_lines > 0:
        parts.append(f"[tail: last {tail_lines} lines{log_ptr}]")
    if blocked:
        if status != "blocked":
            header = "blocked"
            if exit_code is not None:
                header = f"{header} (exit_code={exit_code})"
            parts.append(header)
        if blocked_reason:
            parts.append(blocked_reason)
            # Streams that merely echo the reason are noise.
            if stdout.strip() == blocked_reason:
                stdout = ""
            if stderr.strip() == blocked_reason:
                stderr = ""
    elif exit_code not in (None, 0):
        parts.append(f"exit_code={exit_code}")

    if stdout:
        parts.append(stdout)
    if stderr:
        if stdout:
            parts.append("")
        if exit_code in (None, 0) and not blocked:
            parts.append("stderr:")
        parts.append(stderr)
    _chars_omitted = result.get("chars_omitted")
    _trivial_trim = isinstance(_chars_omitted, int) and 0 <= _chars_omitted < 300
    if truncated and not (_trivial_trim and not spill_hint):
        if stdout or stderr:
            parts.append("")
        # Character-only and structured-data sampling can be lossy without a
        # meaningful line count. Surface recovery whenever the result says it
        # was compacted, not only when lines_omitted happens to be positive.
        # Trivial trims (a few progress/blank lines) get no notice at all --
        # the footer would outweigh what it accounts for.
        if spill_hint:
            parts.append(spill_hint)
        elif isinstance(lines_omitted, int) and lines_omitted > 0:
            parts.append(f"[output truncated: {lines_omitted} lines omitted{log_ptr}]")
        else:
            parts.append(f"[output compacted{log_ptr}]")
    rendered = "\n".join(parts).strip()
    if rendered:
        return rendered
    if exit_code is not None:
        return f"exit_code={exit_code}"
    return ""


def _lift_bash_command_list(args: dict[str, Any], known_params: frozenset[str]) -> dict[str, Any]:
    """Recover list-for-string args: command=[...] -> commands, id=[...] -> ids."""
    cmd = args.get("command")
    if isinstance(cmd, list) and "commands" not in args and all(isinstance(c, str) for c in cmd):
        args = dict(args)
        args["commands"] = args.pop("command")
    sid = args.get("id")
    if isinstance(sid, list) and "ids" not in args and all(isinstance(s, str) for s in sid):
        args = dict(args)
        args["ids"] = args.pop("id")
    return args


BASH_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
            "description": "Command, or array run sequentially — own subshell each, all run even if one fails, `## lc:cmd i/n` headers.",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory; cd never persists.",
        },
        "timeout": {
            "type": "integer",
            "default": _DEFAULT_BASH_SOFT_TIMEOUT,
            "description": "Seconds to WAIT. <1hr → handle returned, run continues (also on stall). >1hr → killed. Long builds → 21600.",
        },
        "bg": {
            "type": "boolean",
            "default": False,
            "description": "Detached, returns id now; only mode outliving this MCP session.",
        },
        "id": {
            "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
            "description": "Session id; bash(id=X) blocks till done — never sleep-poll. Array = several per call.",
        },
        "action": {
            "type": "string",
            "enum": ["poll", "status", "kill", "update", "send"],
            "description": "poll = wait (default); status = peek; kill = terminate; update = new deadline via timeout=; send = write input, return output.",
        },
        "interactive": {
            "type": "boolean",
            "default": False,
            "description": "stdin open as a REPL; drive with action=send + input. Dies after 300s idle.",
        },
        "input": {
            "type": "string",
            "description": "Text for action=send; newline appended. Empty = drain only.",
        },
    },
    "additionalProperties": False,
}


@mcp_tool(
    name="bash",
    input_schema=BASH_TOOL_INPUT_SCHEMA,
    description=(
        "Execute git/build/test/install commands; use indexed tools for code reads/search. "
        "cd doesn't persist → set cwd or use absolute paths. timeout waits, never kills. "
        "Servers → bg=true; REPLs → interactive=true. Batch checks with command=[...]."
    ),
    param_aliases={"session_id": "id", "background": "bg", "is_background": "bg"},
    recover_args=_lift_bash_command_list,
)
def tool_bash(
    command: str = "",
    commands: list[str] | None = None,
    ids: list[str] | None = None,
    timeout: int | None = None,
    cwd: str | None = None,
    max_lines: int = 200,
    max_output_tokens: int | None = None,
    bg: bool = False,
    id: str | None = None,
    action: Literal["run", "poll", "kill", "status", "update", "send"] = "run",
    interactive: bool = False,
    input: str | None = None,
    idle_ttl: int | None = None,
) -> str | _DeferredResult:
    """Execute a shell command and return compact text output.

    Prefer LemonCrow read/grep/search tools directly — they are faster and cheaper.
    Use bash only for commands that have no LemonCrow equivalent (git, make, uv, npm, etc.).

    bg=True starts the command in the background, returns its `id`, and is
    the only mode preserved across MCP shutdown. bash(id=x) alone waits for
    that run to finish (poll); action="status"
    peeks without waiting (state + last 10 output lines); action="kill"
    kills it now. `timeout` at start is only ever a soft response budget --
    it does not kill by itself (eventual internal ~1hr backstop still
    applies). action="update" with id= and a new timeout= installs an exact,
    enforced kill deadline for a running job (e.g. to kill something in 5
    minutes: start it, then update it with timeout=300).

    interactive=True keeps the process alive as a REPL session: stdin stays
    open, action="send" with input= feeds it and returns only the new output
    (state persists across sends -- e.g. one `python -u -i -q` session keeps
    heavy imports loaded). The session dies after `idle_ttl` seconds (default
    300) without a send; every send resets the clock.
    """
    if timeout is not None and timeout > 86_400:
        # A wait budget past 24h is a milliseconds value from a host whose
        # native convention is ms (e.g. 120000 meaning 120s), not a real wait.
        timeout = max(1, timeout // 1000)
    if ids and not id and not command and not commands and not interactive:
        # Multi-id: apply poll/status/kill to each session in one call, one
        # block per id — sessions run concurrently, so total wait ≈ slowest.
        act = "poll" if action == "run" else action
        blocks: list[str] = []
        for sid in ids:
            res = _run_bash_tool(
                "",
                timeout=timeout,
                cwd=cwd,
                max_lines=max_lines,
                max_output_tokens=max_output_tokens,
                background=False,
                session_id=sid,
                action=act,
                interactive=False,
                input_text=None,
                idle_ttl=idle_ttl,
            )
            rendered = _render_bash_text(res) if isinstance(res, dict) else str(res)
            blocks.append(f"## lc:id={sid}\n{rendered}")
        return "\n\n".join(blocks)
    if commands and not command and action == "run" and bg and not interactive and not id:
        # bg batch: start each detached, return one compact indexed id list —
        # the caller already knows its own commands, so no echo.
        lines: list[str] = []
        for i, cmd in enumerate(commands, 1):
            res = _run_bash_tool(
                cmd,
                timeout=timeout,
                cwd=cwd,
                max_lines=max_lines,
                max_output_tokens=max_output_tokens,
                background=True,
                session_id=None,
                action="run",
                interactive=False,
                input_text=None,
                idle_ttl=idle_ttl,
            )
            sid = str(res.get("session_id") or "?") if isinstance(res, dict) else "?"
            lines.append(f"{i}: id={sid} running")
        return "\n".join(lines) + "\nbash(id=[...]) waits for all; action='status' peeks; action='kill' stops"
    if commands and not command and action == "run" and not bg and not interactive and not id:
        # Batch: run every command sequentially in its own subshell (matching
        # the no-cd-persistence contract between separate calls), never stop on
        # failure, and mark each command's exit inline so one call surfaces
        # every failing gate at once instead of an && chain hiding the rest.
        n = len(commands)
        script_parts = ["__lc_worst=0"]
        for i, cmd in enumerate(commands, 1):
            # `## lc:cmd i/n` heads each block (same convention as read's
            # `## path` headers); exit is only called out on failure -- a
            # passing gate needs no per-command noise.
            script_parts.append(f"printf '\\n## lc:cmd {i}/{n}\\n'")
            # 2>&1 per subshell: each command's errors stay inside its own
            # block instead of pooling detached in one trailing stderr.
            script_parts.append("(\n" + cmd + "\n) 2>&1")
            script_parts.append(
                f"__lc_rc=$?; if [ $__lc_rc -gt $__lc_worst ]; then __lc_worst=$__lc_rc; fi; "
                f"if [ $__lc_rc -ne 0 ]; then printf '[lc:cmd {i}/{n} FAILED exit=%d]\\n' $__lc_rc; fi"
            )
        script_parts.append("exit $__lc_worst")
        command = "\n".join(script_parts)
    if id and input is not None and action == "run":
        # bash(id=x, input=...) with no explicit action = feed the session.
        action = "send"
    if id and not command and action == "run":
        # bash(id=x) with no explicit action = wait for the run to finish.
        action = "poll"
    if action == "run" and command and not bg and not interactive:
        _dump_notice = _check_redundant_file_dump(command, cwd)
        if _dump_notice:
            return _dump_notice
    result = _run_bash_tool(
        command,
        timeout=timeout,
        cwd=cwd,
        max_lines=max_lines,
        max_output_tokens=max_output_tokens,
        background=bg,
        session_id=id,
        action=action,
        interactive=interactive,
        input_text=input,
        idle_ttl=idle_ttl,
    )
    # Phase 2: a deferred foreground command flows straight through to _handle,
    # which returns a _Deferred sentinel and lets the watcher render the result.
    if isinstance(result, _DeferredResult):
        return result
    return _render_bash_text(result)
