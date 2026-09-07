"""Shell command execution with token-aware output compaction."""

from __future__ import annotations

import atexit
import codecs
import contextlib
import logging
import os
import re
import secrets
import shlex
import signal
import string
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.redaction import redact_tool_output
from lemoncrow.pro.capabilities.tool_supervision import output_delta
from lemoncrow.pro.capabilities.tool_supervision.bash_output_compression import (
    compact_bash_stream,
    terminal_visible_text,
)
from lemoncrow.pro.capabilities.tool_supervision.bash_output_profiles import compact_profiled_output
from lemoncrow.pro.capabilities.tool_supervision.external_compactors import (
    compactor_for_command,
    external_compactors_enabled,
    resolve_compactor,
    rewrite_compactor_command,
)

_ANSI_ESCAPE = re.compile(
    r"\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI: colors, cursor movement, erase
    r"|\x1b\].*?(?:\x07|\x1b\\)"  # OSC (title, hyperlinks), BEL- or ST-terminated
    r"|\x1b[@-Z\\-_]"  # bare two-byte escapes (incl. unterminated openers)
)
_SEARCH_REGEX_METACHARS = re.compile(r"[][{}()|^$*+?\\]")
# Shell file-write patterns: cat > file or cat >> file (write redirect)
_SHELL_FILE_WRITE_RE = re.compile(r"\bcat\s+>>?", re.IGNORECASE)
# Inline interpreter writes: python -c / heredoc scripts that write workspace
# files (open(...,'w'), .write_text(...)) — same edit-tool bypass as cat >.
_INTERP_WRITE_RE = re.compile(
    r"""\bpython[0-9.]*\b.*(?:
        open\([^)]*,\s*['"][wax]b?\+?['"]   # open(path, 'w'/'a'/'x')
        | \.write_text\(
        | \.write_bytes\(
    )""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)
# Literal write-target extraction for the allowed-roots escape hatch (below).
# A write is permitted only when every target is an absolute literal path inside
# an allowed root; any opaque target (variable, f-string, ``.write_text``
# receiver) yields ``None`` so the guard blocks what it cannot verify.
_OPEN_WRITE_TARGET_RE = re.compile(
    r"""open\(\s*(?P<arg>[^,]+?)\s*,\s*['"][wax]b?\+?['"]""",
    re.IGNORECASE | re.VERBOSE,
)
_CAT_REDIRECT_TARGET_RE = re.compile(r"""\bcat\s+>>?\s*(?P<tgt>'[^']*'|"[^"]*"|[^\s'"|;&>]+)""")
_WRITE_METHOD_RE = re.compile(r"\.write_(?:text|bytes)\(", re.IGNORECASE)
_QUOTED_LITERAL_RE = re.compile(r"""^(?P<q>['"])(?P<v>.*)(?P=q)$""", re.DOTALL)
# A shell short-option cluster requesting no-exec parse mode (``-n``, ``-nx``).
# Among bash/sh/zsh/fish single-char invocation options only ``-n`` contains an
# 'n', so a single-dash cluster containing 'n' implies syntax-check-only.
_SHELL_NOEXEC_SHORT_RE = re.compile(r"^-[a-zA-Z]*n[a-zA-Z]*$")


def _strip_ansi(text: str) -> str:
    return terminal_visible_text(_ANSI_ESCAPE.sub("", text))


# Hard ceiling on how many bytes of stdout/stderr are materialized into memory
# from a single command. A runaway child (`cat /dev/zero`, `yes`, `gzip -dc`,
# a chatty build) would otherwise fill the temp file to disk and OOM on a full
# `.read()`. `max_lines` truncation only runs *after* materialization, so the
# cap must happen at read time. Configurable via env, with a 64KiB floor so it
# can never be set so low that ordinary output is mangled.
_MAX_OUTPUT_BYTES = max(
    64 * 1024,
    int(os.environ.get("LEMONCROW_SHELL_MAX_OUTPUT_BYTES", str(4 * 1024 * 1024))),
)

# On-disk ceiling for a managed command's temp spool. `subprocess.Popen` writes
# the child's output straight to the temp file's fd, so the read-time
# `_MAX_OUTPUT_BYTES` cap cannot bound it -- `cat /dev/zero` would fill the disk
# before any poll runs. The spool pump (`_pump_capped`) stops appending once
# this ceiling is reached. Defaults to the output cap (read side then catches
# every truncated spool); a larger value retains more for later inspection.
_MAX_SPOOL_BYTES = max(
    _MAX_OUTPUT_BYTES,
    int(os.environ.get("LEMONCROW_SHELL_MAX_SPOOL_BYTES", str(_MAX_OUTPUT_BYTES))),
)

# Read granularity for `_pump_capped`; large enough to keep the drain loop cheap
# without buffering an unbounded amount per iteration, and used as the size hint
# for `readline()` so a single pathological line (no trailing newline) still has
# a bounded worst-case read instead of buffering unboundedly.
_PUMP_CHUNK_CHARS = 64 * 1024


def _cap_text(text: str) -> tuple[str, bool]:
    """Bound *text* to the output-byte ceiling, returning (text, truncated).

    Truncation is measured in UTF-8 bytes to mirror on-disk size; the returned
    string is cut on a character boundary at or just under the cap.
    """
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= _MAX_OUTPUT_BYTES:
        return text, False
    capped = encoded[:_MAX_OUTPUT_BYTES].decode("utf-8", "replace")
    return capped, True


def _read_capped(handle: Any) -> tuple[str, bool]:
    """Read at most the output-byte ceiling from a seeked temp-file *handle*.

    Reads one character past the cap to detect a larger file without slurping
    it whole, so memory stays bounded regardless of on-disk size. Returns
    (text, truncated).
    """
    chunk = handle.read(_MAX_OUTPUT_BYTES + 1)
    if len(chunk) <= _MAX_OUTPUT_BYTES:
        return chunk, False
    return chunk[:_MAX_OUTPUT_BYTES], True


def _tail_lines_from_file(handle: Any, n: int) -> list[str]:
    """Return up to the last *n* lines currently written to *handle*.

    Text-mode file cursors only support seeking to 0 or a value previously
    returned by `tell()` -- arithmetic offsets (e.g. "tell() - 4096") are
    undefined for encoded streams -- so this reads from the start rather than
    seeking backward from the end, bounded by the same output-byte ceiling as
    a finished read. Restores the writer's append position before returning;
    callers must hold the file's `output_lock` so a concurrent write can't
    land between the seek(0) and the seek-back.
    """
    if n <= 0:
        return []
    pos = handle.tell()
    if pos == 0:
        return []
    handle.seek(0)
    text, _ = _read_capped(handle)
    handle.seek(pos)
    return text.splitlines()[-n:]


def _pump_capped(src: Any, write: Callable[[str], Any], cap: int) -> bool:
    """Copy text from *src* into *write*, appending at most *cap* UTF-8 bytes.

    Reads line-by-line (bounded by `_PUMP_CHUNK_CHARS` per call) until EOF.
    `readline()` returns as soon as a line is available instead of blocking
    until a full fixed-size chunk is read (as a plain buffered `.read(n)` does
    on a non-interactive pipe) -- required so a `status` peek on a still-running
    command sees output as it's produced rather than only once `_PUMP_CHUNK_CHARS`
    has accumulated or the process exits. The size hint still bounds a single
    call's read for pathological output with no newlines. Once the running byte
    count reaches *cap* the overflow is read and discarded rather than written,
    so the source pipe keeps draining (no deadlock when both stdout and stderr
    are large) while the in-memory or on-disk sink stays bounded. Byte
    accounting mirrors `_cap_text`, cutting a straddling chunk on a character
    boundary at or just under the cap. Returns True if the stream exceeded the
    cap.
    """
    written = 0
    truncated = False
    while True:
        chunk = src.readline(_PUMP_CHUNK_CHARS)
        if not chunk:
            break
        if written >= cap:
            truncated = True
            continue
        encoded = chunk.encode("utf-8", "replace")
        if written + len(encoded) <= cap:
            write(chunk)
            written += len(encoded)
            continue
        prefix = encoded[: cap - written].decode("utf-8", "replace")
        if prefix:
            write(prefix)
        written = cap
        truncated = True
    return truncated


class _LossyStreamReader:
    """Line-oriented text view over a child's *binary* pipe with exact
    lossy-decode detection and raw-byte recovery.

    Decodes utf-8 with ``errors="replace"`` -- a strict decode would raise in
    the reader thread on the first invalid byte, silently emptying the whole
    stream (the pre-fix failure mode). Substitution is detected *exactly* by a
    parallel strict incremental checker over the same bytes: a genuine U+FFFD
    already present in valid utf-8 input does NOT flag the stream, which a
    "count U+FFFD in the decoded text" heuristic would misreport.

    When *raw_file* is given, the raw bytes are teed to it (capped at
    *raw_cap*) so a lossy stream stays byte-exact recoverable; ``finalize()``
    drops the copy when the stream decoded cleanly, so clean commands leave no
    extra file behind.
    """

    def __init__(self, raw: Any, *, raw_file: Any = None, raw_path: str = "", raw_cap: int = 0) -> None:
        self._raw = raw
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._checker: Any = codecs.getincrementaldecoder("utf-8")("strict")
        self._raw_file = raw_file
        self.raw_path = raw_path
        self._raw_cap = raw_cap
        self._raw_written = 0
        self._buf = ""
        self._eof = False
        self._finalized = False
        self.lossy = False

    def _observe(self, chunk: bytes, *, final: bool = False) -> None:
        if self._checker is None:
            return
        try:
            self._checker.decode(chunk, final)
        except UnicodeDecodeError:
            self.lossy = True
            self._checker = None  # one flag suffices; stop paying for the strict pass

    def _tee(self, chunk: bytes) -> None:
        if self._raw_file is None or self._raw_written >= self._raw_cap:
            return
        take = chunk[: self._raw_cap - self._raw_written]
        with contextlib.suppress(Exception):
            self._raw_file.write(take)
            self._raw_written += len(take)

    def readline(self, size: int = -1) -> str:
        """Next line (or at most *size* chars), "" at EOF -- the same contract
        `_pump_capped` relies on for text-mode pipes."""
        limit = size if size > 0 else _PUMP_CHUNK_CHARS
        while True:
            newline = self._buf.find("\n")
            if 0 <= newline < limit:
                line, self._buf = self._buf[: newline + 1], self._buf[newline + 1 :]
                return line
            if len(self._buf) >= limit:
                line, self._buf = self._buf[:limit], self._buf[limit:]
                return line
            if self._eof:
                line, self._buf = self._buf, ""
                return line
            chunk = self._raw.readline(limit)
            if not chunk:
                self._eof = True
                self._observe(b"", final=True)  # a truncated multi-byte seq at EOF is lossy too
                self._buf += self._decoder.decode(b"", True)
                continue
            self._tee(chunk)
            self._observe(chunk)
            self._buf += self._decoder.decode(chunk)

    def finalize(self) -> None:
        """Close the raw tee; drop the copy when the stream decoded cleanly.
        Idempotent -- called from the spool thread at EOF and again from reap
        paths to cover the wedged-reader case."""
        if self._finalized:
            return
        self._finalized = True
        if self._raw_file is not None:
            with contextlib.suppress(Exception):
                self._raw_file.close()
        if not self.lossy and self.raw_path:
            with contextlib.suppress(OSError):
                Path(self.raw_path).unlink(missing_ok=True)
            with _LOG_PATHS_LOCK:
                _ALL_LOG_PATHS.discard(self.raw_path)
            self.raw_path = ""


def _lossy_notice(stream: str, raw_path: str) -> str:
    """One-line integrity warning prepended to a stream that decoded lossily,
    so U+FFFD substitution is loud (with a byte-exact recovery path) instead
    of silently readable as real output."""
    recovery = f"; exact bytes: {raw_path}" if raw_path else "; raw bytes were not captured"
    return (
        f"[lc: WARNING {stream} contained non-UTF8 bytes -- invalid bytes shown as U+FFFD, "
        f"text is NOT byte-exact{recovery}]"
    )


# Default line bound for compacted output. At (or above) this default the char
# budget alone decides truncation; an explicitly TIGHTENED max_lines from the
# caller stays a hard line bound (see _compact_result).
_BASH_DEFAULT_MAX_LINES = 200

_OUTPUT_CAP_NOTICE = (
    "\n... (output exceeded {cap} bytes and was truncated by LemonCrow; narrow the command or redirect to a file) ..."
)


def _head_tail_lines(lines: list[str], head: int, tail: int) -> tuple[str, int, int]:
    if len(lines) <= head + tail:
        return "\n".join(lines), 0, 0
    omitted_lines = lines[head : len(lines) - tail]
    omitted = len(omitted_lines)
    omitted_chars = sum(len(line) for line in omitted_lines)
    parts = [*lines[:head], f"... ({omitted} lines omitted) ...", *lines[-tail:]]
    return "\n".join(parts), omitted, omitted_chars


def _bash_spill_enabled() -> bool:
    """Mirrors the MCP dispatch layer's T7 kill switch (``LEMONCROW_TOOL_OUTPUT_SPILL``)."""
    return os.environ.get("LEMONCROW_TOOL_OUTPUT_SPILL", "1").strip().lower() in {"1", "true", "yes", "on"}


# Footer economics: below this many omitted chars the [lc: shrunk ...] footer
# costs more attention than the trim saved -- a "4 passed" result must not
# drag a spill notice for elided progress dots.
_SPILL_NOTICE_MIN_OMITTED_CHARS = 300
# Escalate to the read-the-spill instruction only when the elided middle is
# big enough to plausibly hold the answer.
_SPILL_ESCALATION_MIN_OMITTED_CHARS = 2000


def _spill_hint(full_text: str, kept_chars: int) -> str:
    """Persist the full pre-compaction *full_text* and return the canonical footer.

    Head+tail compaction below (``_head_tail_lines``) keeps a "(N lines
    omitted)" marker but discards the omitted lines for good -- there was no way
    to recover them without re-running the (often expensive, non-idempotent)
    command. Mirrors ``web_fetch._truncate_with_spill``: persist the untouched
    original to the shared T7 spill store and name the path in the canonical
    ``[lc: ...]`` footer (see ``tool_output_spill.spill_notice``).
    *kept_chars* is the char count of the body actually shown (post-compaction),
    for the footer's ORIG->KEPT accounting. Returns "" when spill is disabled,
    *full_text* is empty, or the write fails, so the caller falls back to the
    bare marker.
    """
    if not full_text or not _bash_spill_enabled():
        return ""
    if len(full_text) - kept_chars < _SPILL_NOTICE_MIN_OMITTED_CHARS:
        return ""  # trim too small to be worth a footer at all
    from lemoncrow.pro.capabilities.tool_supervision import tool_output_spill

    record = tool_output_spill.spill(full_text, tool_name="bash", kind="original")
    if record is None:
        return ""
    notice = tool_output_spill.spill_notice(
        verb="shrunk",
        original_chars=len(full_text),
        kept_chars=kept_chars,
        path=record.path,
    )
    # Heavy elision (>70% of the body dropped) means the middle -- not just
    # boilerplate -- is gone. A bare path footer proved too weak a cue: agents
    # concluded from the kept head/tail and never read the spill (measured on
    # payloads whose answer lived in the elided span). Escalate to an explicit
    # instruction in that case only, so routine trims stay one line. Tiny
    # elisions (a pass summary minus progress dots) never warrant the nudge;
    # a large elided middle keeps it even on success -- a successful query's
    # output IS the answer, unlike a gate's.
    if kept_chars < 0.3 * len(full_text) and len(full_text) - kept_chars >= _SPILL_ESCALATION_MIN_OMITTED_CHARS:
        notice += (
            "\n[lc: mostly elided -- if the answer is in the omitted middle, read the spill file before concluding]"
        )
    return notice


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool
    lines_omitted: int
    command: str
    chars_omitted: int = 0
    policy_category: str = "generic"
    policy_action: str = "allow"
    policy_reason: str = ""
    rewrite_target: str | None = None
    rewrite_payload: dict[str, Any] | None = None
    spill_hint: str = ""


@dataclass(frozen=True)
class CommandPolicyDecision:
    category: str
    action: str
    reason: str = ""
    rewrite_target: str | None = None
    rewrite_payload: dict[str, Any] | None = None


@dataclass
class _ManagedCommand:
    command: str
    proc: subprocess.Popen[bytes]
    stdout_file: Any
    stderr_file: Any
    started: float
    timeout: float
    max_lines: int
    max_chars: int | None = None
    # Only an explicit MCP `bg=true` command may survive MCP shutdown. A
    # foreground command that merely exceeded its soft response budget remains
    # owned by the MCP session and is terminated when that session exits.
    explicit_background: bool = False
    # Host session/bridge that launched this command. Lets a single session's
    # foreground shells be reaped when it disconnects from a shared (singleton)
    # MCP daemon, without touching other sessions' commands. "" on the stdio
    # path (one process per session), where process-exit cleanup covers all.
    owner: str = ""
    # Set only by update_managed_command (bash action="update"). Once
    # true, `timeout` is the exact kill deadline instead of a soft response
    # budget; see _effective_deadline_s.
    deadline_explicit: bool = False
    state: str = "running"
    # First-line provenance when _inject_stable_flags modified the executed
    # command; prepended to the compacted stdout at poll time.
    injected_note: str = ""
    reaped: bool = False
    # Drain threads spooling the child's piped output into the temp files, and a
    # flag set when either spool hit the on-disk ceiling. Joining the threads
    # before a read guarantees all surviving bytes are flushed to disk.
    readers: list[threading.Thread] = field(default_factory=list)
    spool_truncated: bool = False
    # Guards stdout_file/stderr_file cursor + write operations shared between
    # the spool drain threads and a `status` peek (the peek repositions the
    # cursor to read a tail without disturbing the writer's append position).
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    # Phase 2 deferred bash: completion callbacks the watcher fires once the
    # process finishes. Snapshotted+cleared under the lock, invoked outside it.
    on_complete: list[Callable[[], None]] = field(default_factory=list)
    # Real filesystem paths for stdout_file/stderr_file when they were opened
    # on-disk (see _BASH_LOG_DIR below) -- "" when a stream fell back to an
    # anonymous tempfile (e.g. the log dir wasn't writable), which has no path.
    stdout_path: str = ""
    stderr_path: str = ""
    # Text-view readers wrapping the child's binary pipes (exact lossy-decode
    # detection + raw-byte tee; see _LossyStreamReader). Their raw spool file
    # is kept only when a stream actually contained invalid UTF-8.
    stdout_reader: Any = None
    stderr_reader: Any = None
    # Interactive REPL session (bash interactive=true): stdin stays open as a
    # private pipe, action="send" (send_managed_input) feeds it, and the
    # session dies once it has gone `idle_ttl` seconds without input -- a
    # sliding window measured from `last_input` (see _effective_deadline_s).
    interactive: bool = False
    idle_ttl: float = 0.0
    last_input: float = 0.0
    # Opaque text-mode tell() cookies consumed by the previous send, so each
    # send returns only the output produced since the last one (including
    # anything that arrived between sends).
    stdout_read_offset: int = 0
    stderr_read_offset: int = 0


_MANAGED_COMMANDS: dict[str, _ManagedCommand] = {}
_MANAGED_COMMANDS_LOCK = threading.Lock()
# Grace period before the watcher reaps a finished-but-never-polled session,
# so a poll that arrives just after completion still finds its output.
_DETACHED_REAP_GRACE_S = 300.0
# How long poll_managed_command / the watcher's own reap wait for the spool
# drain threads to hit EOF once the wrapped process itself has already
# exited, before giving up and shipping whatever's on disk so far. A detached
# descendant that still holds a duplicate of the output pipe open (e.g. a
# server a task explicitly asked to "keep running in the background") would
# otherwise wedge that join forever even though the command we actually ran
# is long dead.
_READER_JOIN_GRACE_S = 2.0
# Real kill deadline for a managed command, independent of the per-call
# `timeout` a caller passes. `timeout` is a *soft* budget -- see
# _run_bash_tool's deferred branch in mcp_server.py -- for how long the MCP
# tool call waits before handing the model a "still running" session handle
# instead of blocking further; it is NOT how long the command itself is
# allowed to keep running. A command a task deliberately backgrounds (start a
# server, `mailman start`, ...) must survive well past a short/default
# `timeout` or every such task loses its own service partway through. This
# hard cap is the actual backstop against a forgotten/orphaned process
# running forever.
_MANAGED_COMMAND_HARD_CAP_S = 3600.0
# Absolute ceiling on an action="update"-installed explicit deadline (see
# _ManagedCommand.deadline_explicit and update_managed_command): the real
# backstop against a caller granting a forgotten background job unbounded
# life one update at a time -- every update is clamped to this no matter how
# many times a session gets extended.
_MAX_EXPLICIT_TIMEOUT_S = 604800.0  # 7 days
# Sliding idle-TTL default for an interactive session (bash interactive=true):
# after this long without a send the watcher kills the session. Every
# action="send" resets the window; the value is clamped to
# [1, _MANAGED_COMMAND_HARD_CAP_S] at session start.
_DEFAULT_IDLE_TTL_S = 300.0
# A send returns once neither stream has grown for this long (quiescence
# framing -- REPL-agnostic, no sentinel injected into the child's language).
_SEND_QUIESCENCE_S = 0.25
_SEND_POLL_SLICE_S = 0.05
# stdout_file/stderr_file (the managed command's only spool -- no separate
# tee/mirror) live here when writable, instead of an anonymous tempfile with
# no filesystem path. One write per stream, on a real path a user can
# `tail -f`. Deliberately NOT deleted at reap -- a command that just finished
# is exactly when someone is most likely to want to look back at its log; it
# survives until this process's own cleanup (_cleanup_all_log_files, atexit)
# or the staleness sweep below.
_BASH_LOG_DIR = Path(tempfile.gettempdir()) / "lemoncrow-bash"
# Every real path this process has opened under _BASH_LOG_DIR, so process-exit
# cleanup removes exactly its own files and nothing a concurrent gateway
# process (a second Claude Code session on the same machine) is still writing.
_ALL_LOG_PATHS: set[str] = set()
_LOG_PATHS_LOCK = threading.Lock()


def _cleanup_all_log_files() -> None:
    """Remove every on-disk log file this process opened. Registered via
    atexit -- this is this process's "session stop": once it exits, nothing
    can tail a log it owned, so there is nothing left to preserve. Doesn't run
    on a hard kill (SIGKILL/crash); the staleness sweep below is the backstop.
    """
    with _LOG_PATHS_LOCK:
        paths = list(_ALL_LOG_PATHS)
    for raw_path in paths:
        with contextlib.suppress(OSError):
            Path(raw_path).unlink(missing_ok=True)


atexit.register(_cleanup_all_log_files)


# Backstop for log files that outlive this process without going through
# _cleanup_all_log_files -- a hard kill/crash (no atexit ever runs), or a
# stale leftover from a previous crashed process sharing the same tmp dir.
# The age cutoff is well past any real command's runtime, so it never touches
# a file another still-running process is actively writing to.
_STALE_LOG_MAX_AGE_S = 3600.0


def _sweep_stale_log_files(directory: Path, *, max_age_s: float = _STALE_LOG_MAX_AGE_S) -> None:
    with contextlib.suppress(OSError):
        cutoff = time.time() - max_age_s
        for entry in directory.iterdir():
            with contextlib.suppress(OSError):
                if entry.is_file() and entry.stat().st_mtime < cutoff:
                    entry.unlink()


def _open_stream_file(session_id: str, stream_name: str) -> tuple[Any, str]:
    """Open the real, named spool file for one stream of a managed command.

    Falls back to an anonymous tempfile (no path, not tailable) if the named
    file can't be opened -- logging must never block a command from running.
    """
    try:
        _BASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        # Command output can carry secrets and is spooled unredacted, so keep the
        # dir and files owner-only rather than world-readable in a shared /tmp.
        # chmod is best-effort (a dir owned by another session's user can't be
        # re-permed, and that's fine — we only created our own).
        with contextlib.suppress(OSError):
            os.chmod(_BASH_LOG_DIR, 0o700)
        _sweep_stale_log_files(_BASH_LOG_DIR)
        path = _BASH_LOG_DIR / f"{session_id}.{stream_name}.txt"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        handle = os.fdopen(fd, "w+", encoding="utf-8")  # lives on managed, closed at reap
    except OSError:
        return tempfile.TemporaryFile(mode="w+", encoding="utf-8"), ""
    with _LOG_PATHS_LOCK:
        _ALL_LOG_PATHS.add(str(path))
    return handle, str(path)


def _open_raw_stream_file(session_id: str, stream_name: str) -> tuple[Any, str]:
    """Best-effort binary spool for the raw (pre-decode) bytes of one stream.

    Kept only when the stream turns out lossy (see _LossyStreamReader.finalize);
    a clean stream's copy is unlinked at EOF. Returns (None, "") when the log
    dir isn't writable -- raw recovery is then unavailable, never blocking.
    """
    try:
        _BASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(_BASH_LOG_DIR, 0o700)
        path = _BASH_LOG_DIR / f"{session_id}.{stream_name}.bin"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        handle = os.fdopen(fd, "wb")
    except OSError:
        return None, ""
    with _LOG_PATHS_LOCK:
        _ALL_LOG_PATHS.add(str(path))
    return handle, str(path)


_SESSION_ID_ALPHABET = string.ascii_lowercase + string.digits  # base-36


def _new_session_id() -> str:
    """Short, typeable id for a managed bash session.

    6-char base-36 (a-z, 0-9) gives 36^6 ≈ 2.2e9 space — plenty to avoid
    collisions with the live-session table checked below.  Shorter and
    more readable than the old 8-char hex slice.  Falls back to a longer
    id only after repeated clashes (effectively unreachable).
    """
    for width in (6, 6, 10, 16):
        candidate = "".join(secrets.choice(_SESSION_ID_ALPHABET) for _ in range(width))
        with _MANAGED_COMMANDS_LOCK:
            if candidate not in _MANAGED_COMMANDS:
                return candidate
    return "".join(secrets.choice(_SESSION_ID_ALPHABET) for _ in range(32))


_LITERAL_UNSAFE_CHARS = "$`\\*?["


def _literal_operand(tok: str) -> bool:
    """True when *tok* is a plain literal path operand -- something the shell
    would pass through unchanged. A variable ($VAR), tilde or glob operand is
    expanded by a real shell but would be taken literally by a rewrite, and
    ``-`` means stdin; any of those must fall back to real execution.
    """
    if not tok or tok.startswith(("~", "-")):
        return False
    return not any(ch in tok for ch in _LITERAL_UNSAFE_CHARS)


def _rewrite_cat(tokens: list[str]) -> CommandPolicyDecision:
    """Rewrite literal-file cat, including multi-file and numbering forms."""
    files: list[str] = []
    number_mode = ""
    seen_double_dash = False
    for tok in tokens[1:]:
        if tok == "--" and not seen_double_dash:
            seen_double_dash = True
            continue
        if tok.startswith("-") and not seen_double_dash:
            if tok in {"-n", "--number"}:
                number_mode = "all"
                continue
            if tok in {"-b", "--number-nonblank"}:
                number_mode = "nonblank"
                continue
            return CommandPolicyDecision(category="file-read", action="allow")
        files.append(tok)
    if not files or not all(_literal_operand(file) for file in files):
        return CommandPolicyDecision(category="file-read", action="allow")
    if len(files) == 1 and not number_mode:
        return CommandPolicyDecision(
            category="file-read",
            action="rewrite",
            reason="LemonCrow read for file content access",
            rewrite_target="read",
            rewrite_payload={"file_path": files[0]},
        )
    return CommandPolicyDecision(
        category="file-read",
        action="rewrite",
        reason="LemonCrow inline cat for literal file concatenation",
        rewrite_target="cat",
        rewrite_payload={"files": files, "number_mode": number_mode},
    )


def _parse_head_tail_n(tokens: list[str], i: int) -> tuple[int | None, int]:
    """Parse a ``-n N`` / ``--lines=N`` / ``-N`` count from *tokens* at position *i*.

    Returns ``(n, new_i)`` where *n* is ``None`` when the token is unrecognised
    (caller should fall back to subprocess) and *new_i* is the next index to
    process.  Negative N and ``+N`` (from-line) forms return ``None``.
    """
    tok = tokens[i]
    if tok in {"-n", "--lines"}:
        if i + 1 >= len(tokens):
            return None, i + 1
        val = tokens[i + 1]
        if val.startswith("+") or not val.lstrip("-").isdigit():
            return None, i + 2
        n = int(val)
        return (None if n < 0 else n), i + 2
    if tok.startswith("--lines="):
        val = tok.split("=", 1)[1]
        if val.startswith("+") or not val.lstrip("-").isdigit():
            return None, i + 1
        n = int(val)
        return (None if n < 0 else n), i + 1
    if tok.startswith("-n") and tok[2:].isdigit():
        # Bundled form: -n80
        return int(tok[2:]), i + 1
    if len(tok) >= 2 and tok[1:].isdigit():
        # GNU legacy short form: head -80 file
        return int(tok[1:]), i + 1
    return None, i + 1  # unrecognised


def _rewrite_head(tokens: list[str]) -> CommandPolicyDecision:
    """Rewrite ``head [-n N] file`` to a Python inline op (no subprocess)."""
    n = 10
    files: list[str] = []
    i = 1
    seen_double_dash = False
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--" and not seen_double_dash:
            seen_double_dash = True
            i += 1
            continue
        if tok.startswith("-") and not seen_double_dash:
            if tok in {"-q", "--quiet", "--silent", "-v", "--verbose", "-z", "--zero-terminated", "-c", "--bytes"}:
                return CommandPolicyDecision(category="file-read", action="allow")
            parsed_n, i = _parse_head_tail_n(tokens, i)
            if parsed_n is None:
                return CommandPolicyDecision(category="file-read", action="allow")
            n = parsed_n
            continue
        files.append(tok)
        i += 1
    if len(files) != 1 or not _literal_operand(files[0]):
        return CommandPolicyDecision(category="file-read", action="allow")
    return CommandPolicyDecision(
        category="file-read",
        action="rewrite",
        rewrite_target="head",
        rewrite_payload={"file": files[0], "n": n},
    )


def _rewrite_tail(tokens: list[str]) -> CommandPolicyDecision:
    """Rewrite ``tail [-n N] file`` to a Python inline op (no subprocess)."""
    n = 10
    files: list[str] = []
    i = 1
    seen_double_dash = False
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--" and not seen_double_dash:
            seen_double_dash = True
            i += 1
            continue
        if tok.startswith("-") and not seen_double_dash:
            # -f/--follow, -s, --pid, --retry, --sleep-interval and byte-mode
            # all require real tail behaviour.
            if tok in {
                "-f",
                "-F",
                "--follow",
                "--retry",
                "-q",
                "--quiet",
                "--silent",
                "-v",
                "--verbose",
                "-z",
                "--zero-terminated",
                "-c",
                "--bytes",
                "-s",
                "--sleep-interval",
                "--pid",
            }:
                return CommandPolicyDecision(category="file-read", action="allow")
            parsed_n, i = _parse_head_tail_n(tokens, i)
            if parsed_n is None:
                return CommandPolicyDecision(category="file-read", action="allow")
            n = parsed_n
            continue
        files.append(tok)
        i += 1
    if len(files) != 1 or not _literal_operand(files[0]):
        return CommandPolicyDecision(category="file-read", action="allow")
    return CommandPolicyDecision(
        category="file-read",
        action="rewrite",
        rewrite_target="tail",
        rewrite_payload={"file": files[0], "n": n},
    )


def _rewrite_wc(tokens: list[str]) -> CommandPolicyDecision:
    """Rewrite ``wc [-l|-c|-w] file`` to a Python inline op (no subprocess)."""
    count_lines = False
    count_bytes = False
    count_words = False
    files: list[str] = []
    i = 1
    seen_double_dash = False
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--" and not seen_double_dash:
            seen_double_dash = True
            i += 1
            continue
        if tok.startswith("-") and not seen_double_dash:
            if tok in {"-l", "--lines"}:
                count_lines = True
            elif tok in {"-c", "--bytes"}:
                count_bytes = True
            elif tok in {"-w", "--words"}:
                count_words = True
            elif tok in {"-m", "--chars"}:
                # Character vs byte count differs for multibyte content;
                # fall back to subprocess for correctness.
                return CommandPolicyDecision(category="file-read", action="allow")
            elif not tok.startswith("--") and len(tok) > 1:
                # Bundled short flags: -lw, -lc, -lwc …
                for ch in tok[1:]:
                    if ch == "l":
                        count_lines = True
                    elif ch == "c":
                        count_bytes = True
                    elif ch == "w":
                        count_words = True
                    else:
                        return CommandPolicyDecision(category="file-read", action="allow")
            else:
                return CommandPolicyDecision(category="file-read", action="allow")
            i += 1
            continue
        files.append(tok)
        i += 1
    if len(files) != 1 or not _literal_operand(files[0]):
        return CommandPolicyDecision(category="file-read", action="allow")
    return CommandPolicyDecision(
        category="file-read",
        action="rewrite",
        rewrite_target="wc",
        rewrite_payload={
            "file": files[0],
            "count_lines": count_lines,
            "count_bytes": count_bytes,
            "count_words": count_words,
        },
    )


def execute_inline_op(
    rewrite_target: str,
    payload: dict[str, Any],
    cwd: str | None = None,
) -> tuple[str, str, int]:
    """Execute a fast-path file-read op in Python, returning (stdout, stderr, exit_code).

    Covers ``cat``, ``head``, ``tail``, and ``wc``. No subprocess is spawned; latency
    is O(microseconds) rather than O(30-50 ms) for fork+exec of bash+head.
    Called from both ``run_command`` and the MCP adapter so they share the same
    implementation.
    """
    if rewrite_target == "cat":
        files = [str(file) for file in payload.get("files") or []]
        number_mode = str(payload.get("number_mode") or "")
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code = 0
        line_number = 1
        for file_arg in files:
            path = Path(file_arg)
            if not path.is_absolute() and cwd:
                path = Path(cwd) / path
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                stderr_parts.append(f"cat: {file_arg}: No such file or directory\n")
                exit_code = 1
                continue
            except PermissionError:
                stderr_parts.append(f"cat: {file_arg}: Permission denied\n")
                exit_code = 1
                continue
            except OSError as exc:
                stderr_parts.append(f"cat: {file_arg}: {exc}\n")
                exit_code = 1
                continue
            if number_mode:
                numbered: list[str] = []
                for line in text.splitlines(keepends=True):
                    should_number = number_mode == "all" or bool(line.rstrip("\r\n"))
                    if should_number:
                        numbered.append(f"{line_number:6}\t{line}")
                        line_number += 1
                    else:
                        numbered.append(line)
                text = "".join(numbered)
            stdout_parts.append(text)
        return "".join(stdout_parts), "".join(stderr_parts), exit_code

    file_arg = str(payload.get("file") or "")
    path = Path(file_arg)
    if not path.is_absolute() and cwd:
        path = Path(cwd) / path

    if rewrite_target == "head":
        n = int(payload.get("n") or 10)
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                lines: list[str] = []
                for _ in range(n):
                    line = fh.readline()
                    if not line:
                        break
                    lines.append(line)
            return "".join(lines), "", 0
        except FileNotFoundError:
            return "", f"head: cannot open '{file_arg}' for reading: No such file or directory\n", 1
        except PermissionError:
            return "", f"head: cannot open '{file_arg}' for reading: Permission denied\n", 1
        except OSError as exc:
            return "", f"head: {file_arg}: {exc}\n", 1

    if rewrite_target == "tail":
        n = int(payload.get("n") or 10)
        _TAIL_CHUNK = 65536
        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if size == 0:
                    return "", "", 0
                buf = b""
                pos = size
                # Read backward in chunks until we have n+1 newlines
                # (+1 because the first chunk may start mid-line).
                while pos > 0:
                    chunk = min(_TAIL_CHUNK, pos)
                    pos -= chunk
                    fh.seek(pos)
                    buf = fh.read(chunk) + buf
                    if buf.count(b"\n") >= n + 1:
                        break
            text = buf.decode("utf-8", errors="replace")
            lines = text.splitlines()
            tail_lines = lines[-n:] if len(lines) >= n else lines
            return ("\n".join(tail_lines) + "\n") if tail_lines else "", "", 0
        except FileNotFoundError:
            return "", f"tail: cannot open '{file_arg}' for reading: No such file or directory\n", 1
        except PermissionError:
            return "", f"tail: cannot open '{file_arg}' for reading: Permission denied\n", 1
        except OSError as exc:
            return "", f"tail: {file_arg}: {exc}\n", 1

    if rewrite_target == "wc":
        count_lines = bool(payload.get("count_lines"))
        count_bytes = bool(payload.get("count_bytes"))
        count_words = bool(payload.get("count_words"))
        # No flags → report lines, words, and bytes (GNU wc default).
        all_counts = not (count_lines or count_bytes or count_words)
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            parts: list[str] = []
            if all_counts or count_lines:
                parts.append(str(text.count("\n")))
            if all_counts or count_words:
                parts.append(str(len(text.split())))
            if all_counts or count_bytes:
                parts.append(str(len(raw)))
            parts.append(file_arg)
            return " ".join(parts) + "\n", "", 0
        except FileNotFoundError:
            return "", f"wc: {file_arg}: No such file or directory\n", 1
        except PermissionError:
            return "", f"wc: {file_arg}: Permission denied\n", 1
        except OSError as exc:
            return "", f"wc: {file_arg}: {exc}\n", 1

    raise ValueError(f"Unknown inline op: {rewrite_target!r}")


# Flags that are structural/formatting and safe to silently ignore during rewrite
# (they don't change what lines are matched or how many).
_GREP_SAFE_IGNORE_FLAGS: frozenset[str] = frozenset(
    {
        "-r",
        "-R",
        "--recursive",
        "--dereference-recursive",
        "-n",
        "--line-number",
        "-H",
        "--with-filename",
        "-h",
        "--no-filename",
        "-E",
        "--extended-regexp",
        "-s",
        "--no-messages",
        "-a",
        "--text",
        "-I",  # rg: skip binary
        "-u",
        "-uu",
        "-uuu",  # rg: --unrestricted
        "--no-ignore",
        "--no-ignore-vcs",
        "--no-ignore-parent",
        "--hidden",
        "--follow",
        "-L",
        "--color",
        "--colour",
        "--colors",
        "--no-color",
        "--no-colour",
        "--color=never",
        "--color=always",
        "--color=auto",
        "--null",  # grep: NUL-terminate output paths (not content)
        "-p",  # rg: --no-ignore-parent short form
    }
)

# Flags that alter which lines are output or their format — we can't faithfully
# replicate these in the MCP grep tool, so fall back to real shell execution.
_GREP_FALLBACK_FLAGS: frozenset[str] = frozenset(
    {
        "-o",
        "--only-matching",
        "-v",
        "--invert-match",
        "-q",
        "--quiet",
        "--silent",
        "-P",
        "--perl-regexp",
        "-z",
        "--null-data",
        "-Z",
        "-p",  # rg --replace short (conflicts with safe ignore above, but -p is rare)
        "--replace",
    }
)


def _rewrite_search(tokens: list[str], command_name: str, cwd: Path | None = None) -> CommandPolicyDecision:
    # A pipe consumer downstream (`grep ... | wc -l`, `| cut -d: -f1`) parses
    # REAL grep's output format; the internal grep tool renders differently
    # (grouped/ranked), so feeding its output through the pipe would compute
    # wrong results. Piped searches run verbatim. Checked per-token substring
    # so a glued pipe (`f|wc`) is caught too; a `|` inside a quoted regex
    # false-positives, which only skips the optimization.
    if any("|" in tok for tok in tokens[1:]):
        return CommandPolicyDecision(category="search", action="allow")
    grep_tokens = tokens

    ignore_case = False
    file_type: str | None = None
    lines_after = 0
    lines_before = 0
    globs: list[str] = []
    list_files_only = False
    count_only = False
    fixed_strings = False
    word_regexp = False
    line_regexp = False
    cleaned: list[str] = []
    seen_double_dash = False
    i = 1
    while i < len(grep_tokens):
        tok = grep_tokens[i]
        if tok == "--":
            seen_double_dash = True
            i += 1
            continue
        if tok.startswith("-") and not seen_double_dash:
            # Flags that alter output semantics we can't replicate → fall back so
            # the agent gets correct (not silently wrong) results.
            flag_stem = tok.split("=", 1)[0]  # strip =value suffix for lookup
            if flag_stem in _GREP_FALLBACK_FLAGS or tok in _GREP_FALLBACK_FLAGS:
                return CommandPolicyDecision(category="search", action="allow")
            if command_name == "grep" and flag_stem in {"-L", "--files-without-match"}:
                # grep -L = files WITHOUT match (rg's -L is --follow); can't replicate.
                return CommandPolicyDecision(category="search", action="allow")
            if tok in {"-c", "--count"}:
                count_only = True
            elif tok in {"-F", "--fixed-strings"}:
                fixed_strings = True
            elif tok in {"-w", "--word-regexp"}:
                word_regexp = True
            elif tok in {"-x", "--line-regexp"}:
                line_regexp = True
            elif tok in {"-i", "--ignore-case"}:
                ignore_case = True
            # Safe structural/formatting flags — skip quietly.
            elif tok in _GREP_SAFE_IGNORE_FLAGS or flag_stem in _GREP_SAFE_IGNORE_FLAGS:
                i += 1
                continue
            # --type=python or --type python or -t python
            elif tok.startswith("--type="):
                file_type = tok.split("=", 1)[1]
            elif tok in {"--type", "-t"} and i + 1 < len(grep_tokens):
                i += 1
                file_type = grep_tokens[i]
            # -A N / --after-context N  (lines after match)
            elif tok in {"-A", "--after-context"} and i + 1 < len(grep_tokens):
                i += 1
                try:
                    lines_after = int(grep_tokens[i])
                except ValueError:
                    pass
            elif tok.startswith("-A") and tok[2:].isdigit():
                lines_after = int(tok[2:])
            elif tok.startswith("--after-context="):
                try:
                    lines_after = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
            # -B N / --before-context N  (lines before match)
            elif tok in {"-B", "--before-context"} and i + 1 < len(grep_tokens):
                i += 1
                try:
                    lines_before = int(grep_tokens[i])
                except ValueError:
                    pass
            elif tok.startswith("-B") and tok[2:].isdigit():
                lines_before = int(tok[2:])
            elif tok.startswith("--before-context="):
                try:
                    lines_before = int(tok.split("=", 1)[1])
                except ValueError:
                    pass
            # -C N / --context N  (symmetric context)
            elif tok in {"-C", "--context"} and i + 1 < len(grep_tokens):
                i += 1
                try:
                    n = int(grep_tokens[i])
                    lines_before = lines_after = n
                except ValueError:
                    pass
            elif tok.startswith("-C") and tok[2:].isdigit():
                n = int(tok[2:])
                lines_before = lines_after = n
            elif tok.startswith("--context="):
                try:
                    n = int(tok.split("=", 1)[1])
                    lines_before = lines_after = n
                except ValueError:
                    pass
            # -l / --files-with-matches / --files-with-match (rg)
            elif tok in {"-l", "--files-with-matches", "--files-with-match"}:
                list_files_only = True
            # --include=glob (grep) or -g glob (rg)
            elif tok.startswith("--include="):
                globs.append(tok.split("=", 1)[1])
            elif tok in {"-g", "--glob"} and i + 1 < len(grep_tokens):
                i += 1
                globs.append(grep_tokens[i])
            elif tok.startswith("-g") and len(tok) > 2:
                globs.append(tok[2:])
            # Bundled short flags (-rn, -rnicFw …): expand char-by-char.
            # Any fallback char → fall back; any unknown char → fall back;
            # all safe → continue.  Value-in-flag forms (-A90) are already
            # handled above and never reach this branch.
            elif not tok.startswith("--") and len(tok) > 2:
                for ch in tok[1:]:
                    single = f"-{ch}"
                    if single in _GREP_FALLBACK_FLAGS:
                        return CommandPolicyDecision(category="search", action="allow")
                    if command_name == "grep" and single == "-L":
                        # grep -L = files WITHOUT match; can't replicate.
                        return CommandPolicyDecision(category="search", action="allow")
                    if ch == "i":
                        ignore_case = True
                    elif ch == "c":
                        count_only = True
                    elif ch == "F":
                        fixed_strings = True
                    elif ch == "w":
                        word_regexp = True
                    elif ch == "x":
                        line_regexp = True
                    elif single not in _GREP_SAFE_IGNORE_FLAGS:
                        return CommandPolicyDecision(category="search", action="allow")
            # Unknown flag: fall back so we don't silently produce wrong output.
            else:
                return CommandPolicyDecision(category="search", action="allow")
            i += 1
            continue
        cleaned.append(tok)
        i += 1

    if not cleaned:
        return CommandPolicyDecision(category="search", action="allow")
    if len(cleaned) > 2:
        # More than pattern + one path: real grep searches every path given;
        # the internal rewrite takes exactly one. Fall back to real grep.
        return CommandPolicyDecision(category="search", action="allow")

    pattern = cleaned[0]
    # GNU grep BRE treats \| as alternation (extension); rg uses Rust regex
    # where \| is a literal backslash+pipe.  Convert so patterns like
    # "foo\|bar" work as expected via the rg backend.
    if command_name == "grep" and not fixed_strings and r"\|" in pattern:
        pattern = pattern.replace(r"\|", "|")
    if fixed_strings:
        pattern = re.escape(pattern)
    if word_regexp:
        pattern = rf"(?<!\w)(?:{pattern})(?!\w)"
    if line_regexp:
        pattern = rf"^(?:{pattern})$"
    path = cleaned[1] if len(cleaned) > 1 else "."
    if len(cleaned) > 1 and not _literal_operand(path):
        # $VAR/~/glob path operands are expanded by a real shell; the rewrite
        # would take them literally and search the wrong place.
        return CommandPolicyDecision(category="search", action="allow")
    # Single-file targets: fall through to shell grep/rg.  The Python rewrite
    # adds value only for directory-wide searches (ranking, context, file caps).
    # For a specific file, real grep is faster, handles pipes/redirections
    # natively, and avoids any Python overhead or GIL contention.
    resolved_path = Path(path)
    if cwd is not None and not resolved_path.is_absolute():
        resolved_path = cwd / resolved_path
    if path != "." and resolved_path.is_file():
        return CommandPolicyDecision(category="search", action="allow")
    if (
        command_name == "rg"
        and not ignore_case
        and file_type is None
        and not globs
        and not list_files_only
        and not count_only
        and not fixed_strings
        and not word_regexp
        and not line_regexp
        and lines_after == 0
        and lines_before == 0
        and len(cleaned) <= 2
        and not _SEARCH_REGEX_METACHARS.search(pattern)
    ):
        return CommandPolicyDecision(
            category="search",
            action="rewrite",
            reason="LemonCrow search for search-first grounding",
            rewrite_target="search",
            rewrite_payload={"query": pattern, "path": path},
        )
    output_mode = (
        "file_paths_with_match_count"
        if count_only
        else "file_paths_only" if list_files_only else "file_paths_with_content"
    )
    payload: dict[str, Any] = {
        "file_path": path,
        "content_regex": pattern,
        "ignore_case": ignore_case,
        "output_mode": output_mode,
        "lines_after": lines_after,
        "lines_before": lines_before,
    }
    if file_type:
        payload["type"] = file_type
    if globs:
        payload["glob"] = globs
    return CommandPolicyDecision(
        category="search",
        action="rewrite",
        reason=f"LemonCrow grep for {command_name} pattern search",
        rewrite_target="grep",
        rewrite_payload=payload,
    )


def _git_subcommand_index(tokens: list[str]) -> int:
    """Index of the git subcommand, skipping leading global options.

    ``git -C <dir> reset --hard`` and ``git --git-dir=x clean -fd`` place the
    subcommand after global options, so a hardcoded ``tokens[1]`` misses it.
    """
    _takes_value = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
    i = 1
    while i < len(tokens) and tokens[i].startswith("-"):
        tok = tokens[i]
        # ``--git-dir=x`` carries its value inline; bare forms consume the next token.
        if tok in _takes_value and "=" not in tok:
            i += 2
        else:
            i += 1
    return i


def _is_git_reset_hard(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git":
        return False
    idx = _git_subcommand_index(tokens)
    return idx < len(tokens) and tokens[idx] == "reset" and "--hard" in tokens[idx + 1 :]


def _is_git_clean_fd(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git":
        return False
    idx = _git_subcommand_index(tokens)
    if idx >= len(tokens) or tokens[idx] != "clean":
        return False
    joined_flags = "".join(tok for tok in tokens[idx + 1 :] if tok.startswith("-"))
    return "f" in joined_flags and "d" in joined_flags


def _is_shell_file_write(command: str) -> bool:
    """Return True for shell file-write patterns that should use the edit tool instead.

    Catches ``cat > file``, ``cat >> file``, and inline interpreter writes
    (``python -c "...open(f,'w').write(...)"`` or python heredocs) before
    shlex.split, which chokes on heredoc syntax.
    """
    return bool(_SHELL_FILE_WRITE_RE.search(command)) or bool(_INTERP_WRITE_RE.search(command))


def _extract_write_targets(command: str) -> list[str] | None:
    """Literal write targets in *command*, or ``None`` if any write op is opaque.

    Returns ``None`` (caller must block) when a detected write cannot be tied to
    a literal path: a ``.write_text``/``.write_bytes`` call, or an ``open`` whose
    first argument is a variable or expression rather than a string literal.
    """
    if _WRITE_METHOD_RE.search(command):
        return None
    targets: list[str] = []
    for match in _OPEN_WRITE_TARGET_RE.finditer(command):
        literal = _QUOTED_LITERAL_RE.match(match.group("arg").strip())
        if literal is None:
            return None
        targets.append(literal.group("v"))
    for match in _CAT_REDIRECT_TARGET_RE.finditer(command):
        target = match.group("tgt").strip().strip("'\"")
        if not target:
            return None
        targets.append(target)
    return targets


def _file_write_within_allowed(command: str, allowed_roots: list[Path] | None) -> bool:
    """True if every write target is a literal path inside *allowed_roots*.

    *allowed_roots* are the directories writes may target — the workspace root
    plus any opt-in directories (``additionalDirectories`` /
    ``LEMONCROW_ADDITIONAL_DIRS``). A relative target resolves against the first
    root (the workspace root). Any opaque target (variable, f-string, or
    ``.write_text`` receiver) makes ``_extract_write_targets`` return ``None``,
    so the guard blocks what it cannot verify.
    """
    if not allowed_roots:
        return False
    targets = _extract_write_targets(command)
    if not targets:
        return False
    roots = [Path(root).resolve() for root in allowed_roots]
    base = roots[0]
    for raw in targets:
        path = Path(os.path.expanduser(raw))
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not any(path == root or path.is_relative_to(root) for root in roots):
            return False
    return True


def _split_command_segments(command: str) -> list[list[str]]:
    """Split a command line into segments on shell control operators.

    ``bash -c`` runs the whole line, so blocklist checks that only inspect
    ``tokens[0]`` are bypassed by chaining (``ok && rm -rf x``) or command
    substitution (``$(rm -rf x)``). Tokenizing the full line and breaking on
    ``; & | && ||``, newlines, and substitution/brace markers yields each
    segment's own leading token for the blocklist checks.
    """
    operators = {";", "&", "|", "&&", "||", "$(", ")", "`", "{", "}"}
    # Newlines separate statements under ``bash -c``, but shlex.split discards
    # them as whitespace -- which would merge a post-newline command into the
    # previous segment and hide its leading token (``echo hi\nrm -rf x``).
    # Convert them to an explicit ``;`` separator before tokenizing.
    command = command.replace("\n", " ; ")
    # Pad control operators and substitution/brace boundaries with whitespace so
    # shlex isolates them even when glued to a token (``a&&rm``, ``true;rm``) and
    # the command inside ``$(...)`` / ``\`...\``` starts a fresh segment.
    # Over-splitting inside a quoted literal only yields extra benign segments;
    # it can never mask a dangerous leading token.
    normalized = re.sub(r"(\$\(|\)|`|\{|\}|&&|\|\||;|&|\|)", r" \1 ", command)
    try:
        tokens = shlex.split(normalized, comments=False)
    except ValueError:
        return []
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in operators:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)
    return segments


def _is_noexec_shell(tokens: list[str]) -> bool:
    """True if a shell interpreter is invoked purely to syntax-check, not run.

    ``bash -n file`` / ``sh -n`` parse the script and exit without executing any
    command, so unlike ``bash -c '...'`` they cannot smuggle a destructive
    command past the per-segment blocklist. Detects ``-n`` standalone or bundled
    (``-nx``) and the ``-o noexec`` long form. Scans options only up to the first
    non-option token (the script path), so ``bash script.sh -n`` — where ``-n``
    belongs to the script, not the shell — is correctly NOT treated as no-exec.
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-") or tok == "--":
            break
        if tok == "-o":
            if i + 1 < len(tokens) and tokens[i + 1] == "noexec":
                return True
            i += 2
            continue
        if not tok.startswith("--") and _SHELL_NOEXEC_SHORT_RE.match(tok):
            return True
        i += 1
    return False


# A shell short-option cluster carrying inline (`-c`) or stdin (`-s`) code.
_SHELL_INLINE_SHORT_RE = re.compile(r"^-[a-zA-Z]*[cs]")


def _inline_shell_payload(tokens: list[str]) -> str | None:
    """The command string of ``bash -c '...'`` (first non-option arg after a
    ``-c`` cluster), or ``None`` for stdin (``-s``) / no inline payload.

    Used to re-tokenize and block-check the payload so the remaining
    destructive-git guards can't be laundered through an inline shell.
    """
    saw_c = False
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--":
            i += 1
            break
        if tok == "-o":
            i += 2  # -o consumes its option value
            continue
        if tok.startswith("-"):
            if not tok.startswith("--") and "c" in tok[1:]:
                saw_c = True
            i += 1
            continue
        break
    if not saw_c or i >= len(tokens):
        return None
    return tokens[i]


_SCRIPT_SCAN_MAX_BYTES = 64 * 1024


def _script_file_target(tokens: list[str], *, cwd: Path | None) -> Path | None:
    """Resolved script path for ``bash <existing script> [args...]``, else None.

    ``bash -c '...'`` (inline) and ``bash -s`` (stdin) stay blocked: their
    command text is opaque to the per-segment blocklist. A script that exists
    on disk is an auditable on-disk artifact — the same risk class as
    ``python file.py`` or ``make``, which the policy already allows — and its
    contents still get the blocklist scan (_scan_script_for_blocked). A
    missing path yields None so the caller blocks (catches ``bash
    <(curl ...)`` styles and typos).
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--":
            i += 1
            break
        if tok.startswith("-"):
            if not tok.startswith("--") and _SHELL_INLINE_SHORT_RE.match(tok):
                return None  # -c / -s (possibly bundled): inline or stdin code
            if tok == "-o":
                i += 2  # -o consumes its option value
                continue
            i += 1
            continue
        break
    if i >= len(tokens):
        return None  # bare `bash`: interactive / stdin
    script = Path(tokens[i])
    if not script.is_absolute():
        if cwd is None:
            return None
        script = cwd / script
    try:
        return script if script.is_file() else None
    except OSError:
        return None


def _scan_script_for_blocked(script: Path, *, cwd: Path | None, visited: set[Path]) -> CommandPolicyDecision | None:
    """Block-check an on-disk script's contents before letting ``bash script`` run.

    Without this, the blocklist is bypassed by writing the dangerous command
    into a file first — ``bash cleanup.sh`` never had its contents inspected.
    Reads a bounded prefix (_SCRIPT_SCAN_MAX_BYTES); this is a static gate,
    not an interpreter. Fail-open on an unreadable file: pre-scan behavior let
    bash surface its own runtime error, and a file this process cannot read
    would fail the same way under bash. ``visited`` breaks the cycle when
    scripts invoke each other (a.sh -> b.sh -> a.sh).
    """
    try:
        resolved = script.resolve()
    except OSError:
        return None
    if resolved in visited:
        return None
    visited.add(resolved)
    try:
        with open(resolved, encoding="utf-8", errors="replace") as handle:
            text = handle.read(_SCRIPT_SCAN_MAX_BYTES)
    except OSError:
        return None  # unreadable: same as pre-scan behavior — bash reports it at runtime
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue  # shebang / full-line comment
        for segment in _split_command_segments(line):
            decision = _block_check_segment(segment, cwd=cwd, visited=visited)
            if decision is not None:
                return CommandPolicyDecision(
                    category=decision.category,
                    action="block",
                    reason=f"blocked command inside script {script}: {decision.reason}",
                )
    return None


_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Wrappers that run a following command; stripping them exposes the real head so
# the blocklist isn't bypassed by ``timeout 5 rm -rf x`` / ``nice rm -rf x`` /
# ``xargs rm -rf x``. After the wrapper we also skip its option flags and numeric
# option-values / durations (``nice -n 10``, ``timeout 5``), but never a
# non-option word, so a real command head (incl. ``/bin/rm``) is never skipped.
_COMMAND_WRAPPERS = frozenset(
    {
        "env",
        "command",
        "sudo",
        "doas",
        "nohup",
        "setsid",
        "stdbuf",
        "unbuffer",
        "nice",
        "ionice",
        "time",
        "timeout",
        "chrt",
        "xargs",
        "watch",
        "proot",
        "flock",
    }
)
# Shell interpreters whose direct execution is blocked (they run arbitrary
# commands). ``busybox`` is handled separately: its applet is the real head.
_SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "fish", "dash", "ash", "ksh", "mksh", "rbash", "csh", "tcsh"})
# Heads that evaluate their remaining arguments as a fresh command line.
_EVAL_WRAPPERS = frozenset({"eval", "exec"})
# Option flags (``-x``) and numeric option-values / durations (``10``, ``5s``) a
# wrapper may carry before its command; safe to skip (never a command name).
_WRAPPER_SKIP_RE = re.compile(r"^(?:-.*|[+-]?\d+(?:\.\d+)?[smhdkKMGT]?)$")


def _strip_command_prefixes(tokens: list[str]) -> list[str]:
    """Strip leading ``VAR=value`` assignments and pass-through wrappers
    (``env``/``sudo``/``timeout``/``nice``/...) to a fixed point so the real
    command head is checked.

    ``env A=1 bash -c`` / ``command rm -rf`` / ``timeout 5 rm -rf`` would
    otherwise hide a dangerous head behind a wrapper token. After a wrapper we
    also skip its option flags and numeric option-values / durations
    (``nice -n 10``, ``timeout 5``) -- but never a non-option word, so a real
    command head (incl. a path-qualified ``/bin/rm``) is never skipped past.
    """
    i = 0
    changed = True
    while changed:
        changed = False
        while i < len(tokens) and _ASSIGN_RE.match(tokens[i]):
            i += 1
            changed = True
        if i < len(tokens) and os.path.basename(tokens[i]).lower() in _COMMAND_WRAPPERS:
            i += 1
            changed = True
            while i < len(tokens) and _WRAPPER_SKIP_RE.match(tokens[i]):
                i += 1
    return tokens[i:]


def _block_check_segment(
    tokens: list[str],
    *,
    cwd: Path | None = None,
    visited: set[Path] | None = None,
) -> CommandPolicyDecision | None:
    """Return a block decision if *tokens* (one segment) is dangerous, else None.

    Only the destructive-git guards remain (``git reset --hard`` / ``git clean
    -fd``). The former inline-shell (``bash -c``) and ``rm -rf`` blocks were
    removed: benchmark transcripts showed each block costing 1-5 recovery
    turns per trial while protecting a disposable working tree, and every
    inline-shell payload is still block-checked below so the git guards can't
    be laundered through ``bash -c 'git reset --hard'``.
    """
    if not tokens:
        return None
    tokens = _strip_command_prefixes(tokens)
    if not tokens:
        return None
    # Normalize the head to its basename so path-qualified invocations
    # (``/bin/bash``, ``/usr/bin/git``) are matched like their bare names.
    tokens = [os.path.basename(tokens[0]), *tokens[1:]]
    head = tokens[0].lower()
    # ``busybox <applet> ...``: the applet (sh/rm/...) is the effective head.
    if head == "busybox" and len(tokens) > 1:
        return _block_check_segment(tokens[1:], cwd=cwd, visited=visited)
    # ``eval``/``exec <words>``: the remaining words run as a fresh command line,
    # so re-tokenize and block-check them (catches ``eval \"git reset --hard\"``).
    if head in _EVAL_WRAPPERS and len(tokens) > 1:
        for inner in _split_command_segments(" ".join(tokens[1:])):
            decision = _block_check_segment(inner, cwd=cwd, visited=visited)
            if decision is not None:
                return decision
        return None
    if head in _SHELL_INTERPRETERS:
        if _is_noexec_shell(tokens):
            return None  # `bash -n` / `-o noexec`: parse-only, runs nothing
        script = _script_file_target(tokens, cwd=cwd)
        if script is not None:
            # `bash existing-script.sh`: on-disk artifact, like `python file.py`
            # — but its contents get the same blocklist scan the command line
            # gets, so a dangerous command can't be laundered through a file.
            return _scan_script_for_blocked(
                script,
                cwd=cwd,
                visited=visited if visited is not None else set(),
            )
        # Inline (`bash -c '...'`) and stdin (`-s`) shells run; the -c payload
        # is re-tokenized and block-checked so the remaining git guards hold.
        payload = _inline_shell_payload(tokens)
        if payload:
            for inner in _split_command_segments(payload):
                decision = _block_check_segment(inner, cwd=cwd, visited=visited)
                if decision is not None:
                    return decision
        return None
    if _is_git_reset_hard(tokens):
        return CommandPolicyDecision(
            category="destructive",
            action="block",
            reason="git reset --hard blocked",
        )
    if _is_git_clean_fd(tokens):
        return CommandPolicyDecision(
            category="destructive",
            action="block",
            reason="git clean -fd blocked",
        )
    return None


# Known-bad shell calls -> ALLOW or REDIRECT-and-execute, never block/message.
# Where a read-only equivalent exists we REWRITE: the equivalent runs behind the
# scenes and its result is returned in the SAME turn (like grep->grep_tool), so no
# turn is wasted. Everything else (incl. sed -i replacements, git navigation) is
# ALLOWED to run -- a git-archaeology *spiral* is caught by the convergence escalation.
_FETCH_URL_RE = re.compile(r"https?://[^\s'\"|>;)]+", re.IGNORECASE)
_FETCH_SETUP_RE = re.compile(
    r"\|\s*(?:sudo\s+)?(?:sh|bash|zsh|pip[0-9]*|python[0-9.]*|tar|unzip|gunzip|apt|apt-get|brew|npm|node|tee)\b"
    r"|\s-[oO]\b|\s--output\b|>\s*\S"
    r"|&&\s*(?:tar|unzip|pip|sh|bash|make|python|\./)",
    re.IGNORECASE,
)
# curl/wget flags that carry request semantics — headers, method, body/data,
# auth, forms, uploads, cookies — which the plain-URL web_fetch rewrite would
# silently drop. Their presence disables the rewrite; the command runs as-is
# (a false positive here only means no rewrite, never a block).
_FETCH_REQUEST_FLAGS_RE = re.compile(
    r"(?:^|\s)(?:"
    r"--(?:header|request|method|data[-a-z]*|json|user|form(?:-string)?|upload-file"
    r"|cookie(?:-jar)?|user-agent|referer|post-data|post-file|body-data|body-file|head)(?:=|\s|$)"
    r"|-[A-Za-z]*[HXduFTbIA][A-Za-z]*(?=\s|$)"
    r")"
)
# A find -name pattern the internal glob engine reproduces exactly: a basename
# glob with no path separator and no shell-expansion/quoting characters.
_FIND_PATTERN_RE = re.compile(r"^[A-Za-z0-9._*?\[\]-]+$")
# sed line-print expression: A[,B]p and nothing else -- scripts, regex
# addresses, multiple expressions and extra flags all run for real.
_SED_EXPR_RE = re.compile(r"^(\d+)(?:,(\d+))?p$")
# A rewrite replaces the ENTIRE command with an internal-tool equivalent, so it
# is only sound when the command IS that single invocation. Chaining (&&, ||,
# ;, &), redirection (>, <), substitution (`...` or $(...)) or a newline means
# a rewrite would silently drop every other part -- e.g.
# `sed -i ... && sed -n ...` must NOT collapse into a read of the (unedited)
# file. A *quoted* operator false-positives here, which only skips the
# optimization: the command then runs verbatim, never wrongly. Bare `|` stays
# allowed -- _rewrite_pipeline and _rewrite_search handle pipe tails explicitly.
_REWRITE_UNSAFE_RE = re.compile(r"[;&<>`\n]|\$\(")


def _rewrite_sed_read(tokens: list[str]) -> CommandPolicyDecision | None:
    """Recognize quiet single-expression numeric line reads across sed spellings."""
    quiet = False
    expressions: list[str] = []
    operands: list[str] = []
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-n", "--quiet", "--silent"}:
            quiet = True
        elif tok in {"-e", "--expression"}:
            if i + 1 >= len(tokens):
                return None
            i += 1
            expressions.append(tokens[i])
        elif tok.startswith("--expression="):
            expressions.append(tok.split("=", 1)[1])
        elif tok in {"-ne", "-en"}:
            quiet = True
            if i + 1 >= len(tokens):
                return None
            i += 1
            expressions.append(tokens[i])
        elif tok.startswith("-"):
            return None
        else:
            operands.append(tok)
        i += 1
    if not quiet:
        return None
    if not expressions and len(operands) == 2:
        expressions.append(operands.pop(0))
    if len(expressions) != 1 or len(operands) != 1:
        return None
    expression, file_arg = expressions[0], operands[0]
    match = _SED_EXPR_RE.match(expression)
    if match is None or not _literal_operand(file_arg) or ":" in file_arg:
        return None
    start = match.group(1)
    end = match.group(2) or start
    return CommandPolicyDecision(
        category="sed-read",
        action="rewrite",
        rewrite_target="read_range",
        rewrite_payload={"spec": f"{file_arg}:L{start}-L{end}"},
    )


def _redirect_known_bad(command: str) -> CommandPolicyDecision | None:
    """Rewrite known-bad read-only calls to the right tool (executed inline). Never blocks.

    Only called for a single unchained, unpiped command (``classify_command``
    gates on ``_REWRITE_UNSAFE_RE`` and ``|``). Each rewrite anchors on the
    head token and requires the exact, fully-understood shape -- anything else
    runs verbatim. A skipped rewrite is always safe; a wrong one never is.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    head_tok = tokens[0].lower()
    if head_tok == "curl" and not _FETCH_SETUP_RE.search(command):
        # curl only: its default prints the body to stdout, which web_fetch
        # reproduces. wget's default DOWNLOADS to a file a later command may
        # depend on, so wget always runs for real.
        if _FETCH_REQUEST_FLAGS_RE.search(command):
            # Headers/method/auth/body present: a plain-URL web_fetch rewrite
            # would silently drop them and fetch the wrong thing — run as-is.
            return None
        urls = _FETCH_URL_RE.findall(command)
        if len(urls) == 1 and "$" not in urls[0]:
            # Exactly one literal URL: a second URL or a $VAR would be
            # dropped/taken literally by the rewrite.
            return CommandPolicyDecision(
                category="web-fetch",
                action="rewrite",
                rewrite_target="web_fetch",
                rewrite_payload={"url": urls[0]},
            )
        return None  # no/multiple/variable URL -> just allow
    if head_tok == "find":
        # Exact shape only: find [PATH] -name PATTERN -type f (any predicate
        # order). Everything else diverges from the internal glob listing:
        # -iname (case), -wholename (path match), -maxdepth/-mtime/... (extra
        # filters), -delete/-exec (side effects), no -type f (real find also
        # lists matching DIRECTORIES) -- all of those run for real.
        rest = tokens[1:]
        path = "."
        if rest and not rest[0].startswith("-"):
            path, rest = rest[0], rest[1:]
        valid = len(rest) == 4
        opts: dict[str, str] = {}
        if valid:
            for flag, val in zip(rest[::2], rest[1::2], strict=True):
                if flag in opts:
                    valid = False
                    break
                opts[flag] = val
        if (
            valid
            and set(opts) == {"-name", "-type"}
            and opts["-type"] == "f"
            and (path == "." or _literal_operand(path))
            and _FIND_PATTERN_RE.match(opts["-name"])
        ):
            return CommandPolicyDecision(
                category="find",
                action="rewrite",
                rewrite_target="find_glob",
                rewrite_payload={"glob": opts["-name"], "path": path},
            )
        return None
    if head_tok == "sed":
        return _rewrite_sed_read(tokens)
    return None  # sed -i / other sed / other find / wget / git navigation -> ALLOW


# Pipeline-aware rewrite (tier 1 detect + tier 2 safe rewrite). A streaming hex
# formatter over a large file piped into `tail` forces the formatter to process
# the ENTIRE file -- `tail` can't SIGPIPE-abort it early the way `head` can, so
# `od bigfile | tail -60` hex-formats all N bytes just to show the end. Rewrite
# to an in-place seek (`od -j <offset>`), which preserves od's absolute byte
# addresses. The naive `tail -c N file | od` does NOT: its addresses restart at
# 0. Deliberately narrow -- od only, one seekable regular file, tail-only
# consumer, no geometry-changing od flags; every other shape returns None and
# runs unchanged (no silent rewrite of a command we can't prove equivalent).
_PIPELINE_SEEK_MIN_BYTES = 8 * 1024 * 1024
_OD_BYTES_PER_LINE = 16  # od/hexdump default row width; the line-mode seek assumes it


def _split_top_level_pipeline(command: str) -> list[list[str]] | None:
    """The ``|``-separated stages of a simple single-line pipeline as token
    lists, or None if *command* is anything more complex -- a newline, ``;``,
    ``&&``, ``||``, ``&``, a redirect, a subshell, or command substitution. Used
    only to spot a narrow, safely-rewritable shape; when in doubt, return None
    and leave the command untouched.
    """
    if any(marker in command for marker in ("\n", "`", "$(")):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        toks = list(lexer)
    except ValueError:
        return None
    forbidden = {"&&", "||", "&", ";", "(", ")", "<", ">", ">>", "<<", "|&", "<>", "&>"}
    if any(tok in forbidden for tok in toks):
        return None
    if "|" not in toks:
        return None
    stages: list[list[str]] = []
    current: list[str] = []
    for tok in toks:
        if tok == "|":
            if not current:
                return None
            stages.append(current)
            current = []
        else:
            current.append(tok)
    if not current:
        return None
    stages.append(current)
    return stages


def _parse_tail_bound(tokens: list[str]) -> tuple[str | None, int]:
    """``('lines'|'bytes', count)`` for a plain bounded ``tail`` reading stdin,
    or ``(None, 0)`` for any shape whose output can't be reproduced by a byte
    seek (``-f``/follow, suffixed or ``+N`` counts, a file operand, ...).
    """
    mode = "lines"
    count = 10
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in {"-c", "--bytes", "-n", "--lines"}:
            if i + 1 >= len(tokens):
                return None, 0
            val = tokens[i + 1]
            if not val.isdigit():
                return None, 0
            mode = "bytes" if tok in {"-c", "--bytes"} else "lines"
            count = int(val)
            i += 2
            continue
        if tok.startswith(("--bytes=", "--lines=")):
            val = tok.split("=", 1)[1]
            if not val.isdigit():
                return None, 0
            mode = "bytes" if tok.startswith("--bytes=") else "lines"
            count = int(val)
            i += 1
            continue
        if tok.startswith("-c") and tok[2:].isdigit():
            mode, count, i = "bytes", int(tok[2:]), i + 1
            continue
        if tok.startswith("-n") and tok[2:].isdigit():
            mode, count, i = "lines", int(tok[2:]), i + 1
            continue
        if re.fullmatch(r"-\d+", tok):  # `tail -60` == `tail -n 60`
            mode, count, i = "lines", int(tok[1:]), i + 1
            continue
        # any other flag (-f/--follow, -q, ...) or a file operand: the consumer
        # isn't a plain bounded stdin tail -- don't rewrite.
        return None, 0
    return mode, count


def _rewrite_pipeline(command: str, cwd: str | Path | None) -> CommandPolicyDecision | None:
    stages = _split_top_level_pipeline(command)
    if stages is None or len(stages) != 2:
        return None
    producer, consumer = stages
    if not producer or not consumer:
        return None
    if producer[0].lower() != "od" or consumer[0].lower() != "tail":
        return None
    # Bail on od flags that change row geometry or already seek/limit the read --
    # the offset math and the printed addresses would no longer line up.
    for tok in producer[1:]:
        if tok.startswith(("-w", "--width", "-N", "--read-bytes", "-j", "--skip-bytes", "-S", "--strings")):
            return None
    # Producer operands that exist as regular files (a flag value like `-A`'s
    # `d` or `-t`'s `x1` doesn't name a real file, so statting also tells the
    # file apart from option arguments without parsing od's grammar). od over
    # multiple files concatenates their dumps -- which a single-file seek can't
    # reproduce -- so require exactly one, and it must be large enough that
    # formatting the whole thing is the actual waste we're avoiding.
    file_operands: list[tuple[str, int]] = []
    for tok in producer[1:]:
        if tok.startswith("-") and tok != "-":
            continue
        path = Path(tok)
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / tok
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        file_operands.append((tok, size))
    if len(file_operands) != 1:
        return None
    file_tok, size = file_operands[0]
    if size <= _PIPELINE_SEEK_MIN_BYTES:
        return None
    mode, count = _parse_tail_bound(consumer)
    if mode is None or count <= 0:
        return None
    if mode == "bytes":
        offset = size - count
    else:
        # Seek one extra row back so we never return FEWER rows than asked (od
        # also prints a trailing address-only line): a safe superset of `tail -n`.
        offset = size - (count + 1) * _OD_BYTES_PER_LINE
    if offset <= 0:
        return None  # requested tail already spans (nearly) the whole file
    # Rebuild: original od invocation minus the file operand (and any lone `--`),
    # plus our skip and the file at the end behind a fresh `--`.
    rebuilt = ["od"]
    for tok in producer[1:]:
        if tok == file_tok or tok == "--":
            continue
        rebuilt.append(tok)
    rebuilt += ["-j", str(offset), "--", file_tok]
    rewritten = " ".join(shlex.quote(tok) for tok in rebuilt)
    note = (
        f"[LemonCrow: `od … | tail` over a {size // (1024 * 1024)}MB file would hex-format the "
        f"whole file; seeked to byte {offset} instead (od -j; absolute addresses preserved)]"
    )
    return CommandPolicyDecision(
        category="file-read",
        action="rewrite",
        reason=note,
        rewrite_target="pipeline_seek",
        rewrite_payload={"command": rewritten, "note": note},
    )


def classify_command(
    command: str, *, allowed_write_roots: list[Path] | None = None, cwd: str | Path | None = None
) -> CommandPolicyDecision:
    resolved_cwd = Path(cwd).resolve() if cwd else None
    # Block checks run per segment: bash -c executes the whole line, so chaining
    # and command substitution must not slip a dangerous segment past tokens[0].
    for segment in _split_command_segments(command):
        blocked = _block_check_segment(segment, cwd=resolved_cwd)
        if blocked is not None:
            return blocked

    # Shell file-writes (`cat > file`, inline `python -c` open/write) must stay
    # inside the allowed write roots; outside them the edit tool is the right
    # surface. The OS temp dir is additionally allowed for the same reason
    # `rm -rf` is allowed there: scratch files are the agent's own.
    if allowed_write_roots and _is_shell_file_write(command):
        targets = _extract_write_targets(command)
        # Fail-open when no write target can be resolved statically (targets
        # is None: a variable/f-string path or a .write_text receiver, or an
        # empty list). Blocking what we cannot parse would break legitimate
        # $TMPDIR-style writes; the destructive blocklist above still applies,
        # and a genuinely wrong opaque path just fails at runtime.
        if targets and not _file_write_within_allowed(command, [*allowed_write_roots, Path(tempfile.gettempdir())]):
            return CommandPolicyDecision(
                category="file-write",
                action="block",
                reason=(
                    "shell write outside the allowed write roots blocked "
                    f"(target(s): {', '.join(targets)}) — use the edit tool to "
                    "create or modify files, or target a path inside the workspace"
                ),
            )

    # Rewrites replace the whole command -- only sound for a single simple
    # invocation (see _REWRITE_UNSAFE_RE). When the command chains, redirects,
    # or substitutes, skip every rewrite and run it verbatim.
    rewrite_safe = _REWRITE_UNSAFE_RE.search(command) is None

    bad = _redirect_known_bad(command) if rewrite_safe and "|" not in command else None
    if bad is not None:
        return bad

    # Pipeline-aware rewrite runs before the single-command tokens[0] dispatch
    # (which never sees into `producer | consumer`). Only fires for the narrow
    # `od <bigfile> | tail` shape; returns None -> normal flow -> untouched.
    pipeline_rewrite = _rewrite_pipeline(command, resolved_cwd) if rewrite_safe else None
    if pipeline_rewrite is not None:
        return pipeline_rewrite

    try:
        tokens = shlex.split(command)
    except ValueError:
        return CommandPolicyDecision(category="generic", action="allow")
    if not tokens:
        return CommandPolicyDecision(category="generic", action="allow")

    head = tokens[0].lower()
    if rewrite_safe:
        if head == "cat":
            return _rewrite_cat(tokens)
        if head == "head":
            return _rewrite_head(tokens)
        if head == "tail":
            return _rewrite_tail(tokens)
        if head == "wc":
            return _rewrite_wc(tokens)
        if head in {"rg", "grep"}:
            return _rewrite_search(tokens, head, resolved_cwd)
    if external_compactors_enabled():
        compactor = compactor_for_command(tokens)
        if compactor is not None:
            resolution = resolve_compactor(compactor.name)
            rewritten_command = rewrite_compactor_command(compactor, resolution, command)
            if rewritten_command:
                return CommandPolicyDecision(
                    category="external-compactor",
                    action="rewrite",
                    reason=f"passed through installed `{compactor.name}` binary for compact output",
                    rewrite_target="external_compactor",
                    rewrite_payload={
                        "compactor": compactor.name,
                        "binary_path": str(resolution.path),
                        "original_command": command,
                        "command": rewritten_command,
                    },
                )
    return CommandPolicyDecision(category="generic", action="allow")


def _terminate_process_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


# Bash output is re-read as cache_read on EVERY later turn, so a fat test/log dump
# is paid for once per remaining turn -- the dominant cost on long tasks. Cap the
# char size (line caps miss long-line output like ``git log --format``) and, for
# test runs, keep the actionable failure section + summary instead of head/tail
# (the FAILURES block sits in the middle and head/tail would drop it).
# Generic stdout budget. 6000 proved a false economy on benchmark transcripts:
# any file view over ~6KB (sed/awk/python-print of a 250-line source file) came
# back middle-elided, and agents burned 6+ turns re-viewing the same file
# instead of reading the spill. 20000 (~5K tokens) ships moderate bodies whole;
# genuinely large output still gets head+tail + spill recovery.
_BASH_STDOUT_CHAR_CAP = 20000
# Test-runner detection spans the mainstream ecosystems (Python, JS/TS, Rust,
# Go, Ruby, PHP, JVM, .NET, Elixir): the failure-extraction path below is what
# keeps a red run actionable inside the char budget, so missing a runner
# silently downgrades its output to blind head/tail.
_TEST_CMD_RE = re.compile(
    r"\b(pytest|py\.test|runtests|nosetests|tox|jest|vitest|mocha|rspec|phpunit|ctest)\b"
    r"|python[0-9.]*\s+-m\s+(unittest|pytest)"
    r"|manage\.py\s+test"
    r"|\b(cargo|go|dotnet|mix)\s+test\b"
    r"|\b(npm|pnpm|yarn|bun)\s+(run\s+)?test\b"
    r"|\bgradlew?\s+[\w:]*test\b"
    r"|\bmvn\b[^|;&]*\btest\b"
    r"|\bplaywright\s+test\b"
)
# Extra fail-start markers only ever move the cut point EARLIER (the first
# matching line wins), so a false positive keeps more context, never less.
_TEST_FAIL_START_RE = re.compile(
    r"^(=+\s*(FAILURES|ERRORS)\s*=+"  # pytest section header
    r"|=+\s*short test summary"  # pytest -q
    r"|FAIL:|ERROR:|FAILED\b"  # unittest / nose
    r"|FAIL\b"  # jest/vitest per-file, go test per-package
    r"|\s*--- FAIL:"  # go test per-test
    r"|\s*failures?:"  # cargo test / rspec section header
    r"|\s*[✕✗●]"  # jest/vitest failure bullets (✕ ✗ ●)
    r"|\s*\d+\)\s"  # mocha / rspec numbered failures
    r")",
    re.IGNORECASE,
)
_TEST_SUMMARY_RE = re.compile(
    r"\d+\s+(passed|failed|failures?|errors?|skipped|passing|failing|pending|examples?)"
    r"|Ran\s+\d+\s+test"
    r"|^(OK|PASS|FAILED?)\b"  # unittest OK, go test ok/PASS/FAIL
    r"|test result:"  # cargo test
    r"|Tests?(\s+Suites)?\s*:\s*\d+",  # jest/vitest "Tests: ...", dotnet "Total tests: N"
    re.IGNORECASE,
)
_TEST_CHAIN_RE = re.compile(r"(?:&&|\|\||[;|])")
_TEST_CD_PREFIX_RE = re.compile(r"^\s*cd\s+[^&|;]+&&\s*")


def _is_test_only_command(command: str) -> bool:
    """True for one test invocation, allowing only a leading ``cd … &&``.

    Summary extraction is intentionally lossy. Applying it to ``pytest && git
    status`` would silently discard the second command's output, so compound
    commands stay on the generic anomaly/head-tail path.
    """
    body = _TEST_CD_PREFIX_RE.sub("", command, count=1).strip()
    return bool(_TEST_CMD_RE.search(body)) and not bool(_TEST_CHAIN_RE.search(body))


def _cap_chars(text: str, max_chars: int) -> str:
    """Keep head + tail of *text* within *max_chars* (long-line safe)."""
    if len(text) <= max_chars:
        return text
    h = max_chars * 3 // 4
    t = max_chars - h
    return f"{text[:h]}\n... ({len(text) - max_chars:,} chars trimmed) ...\n{text[-t:]}"


def _extract_test_output(text: str, max_chars: int = _BASH_STDOUT_CHAR_CAP) -> str:
    """From a test run, keep the actionable failures + summary; drop pass/collection noise."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _TEST_FAIL_START_RE.search(ln)), None)
    if start is not None:  # there are failures -- keep from the first failure marker on
        return _cap_chars("\n".join(lines[start:]), max_chars)
    summary = [ln for ln in lines if _TEST_SUMMARY_RE.search(ln)]
    if summary:  # all green -- the summary line is all the agent needs
        return "\n".join(summary[-3:])
    return _cap_chars(text, max_chars)


# Dedup-with-count: log-style output (retry loops, polling waits, repeated
# warnings, stack frames) often repeats one line dozens or hundreds of times.
# Collapsing a run of identical lines to one copy plus an annotated count is
# information-preserving, so it can run before every other compaction path.
# Runs shorter than the threshold are left alone -- a 2x repeat may be
# meaningful sequence data and saves almost nothing.
_DEDUP_MIN_REPEATS = 3


def _dedupe_repeated_lines(text: str) -> tuple[str, int]:
    """Collapse runs of >= ``_DEDUP_MIN_REPEATS`` identical lines.

    Returns ``(deduped_text, chars_saved)``. Non-blank runs keep one copy plus
    an explicit ``(line repeated N times)`` marker; blank runs collapse
    silently to a single blank line. Unchanged text is returned as-is.
    """
    lines = text.splitlines()
    if len(lines) < _DEDUP_MIN_REPEATS:
        return text, 0
    out: list[str] = []
    changed = False
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        run = j - i
        if run >= _DEDUP_MIN_REPEATS:
            out.append(lines[i])
            if lines[i].strip():
                out.append(f"... (line repeated {run} times) ...")
            changed = True
        else:
            out.extend(lines[i:j])
        i = j
    if not changed:
        return text, 0
    deduped = "\n".join(out)
    if text.endswith("\n"):
        deduped += "\n"
    return deduped, max(0, len(text) - len(deduped))


# Per-command-kind stdout budgets. Bare listings (ls/tree/du/git status ...) are
# enumerations -- mostly noise -- so they get a lean cap; test runs keep more
# (failures are the actionable signal, and truncating them forces a costly
# re-run); everything else keeps the default head+tail cap.
_BASH_LISTING_RE = re.compile(
    r"^\s*(?:cd\s+[^&|;]+&&\s*)?(?:ls|tree|du|df|find|stat|env|printenv|ps"
    r"|git\s+status|git\s+ls-files|git\s+branch)\b",
    re.IGNORECASE,
)
_BASH_LISTING_CHAR_CAP = 2000
_BASH_TEST_CHAR_CAP = 8000


def _bash_output_budget(command: str) -> int:
    """Stdout char budget keyed by command kind (test / listing / generic)."""
    if _TEST_CMD_RE.search(command):
        return _BASH_TEST_CHAR_CAP
    if _BASH_LISTING_RE.search(command):
        return _BASH_LISTING_CHAR_CAP
    return _BASH_STDOUT_CHAR_CAP


# Generalizes _extract_test_output beyond test runners. Any long-running
# command (build, migration, deploy script, linter) can bury its one actionable
# line in the middle of an otherwise-routine log, and blind head/tail -- like
# the FAILURES block for test runs -- would drop exactly that line.
_ANOMALY_LINE_RE = re.compile(
    r"\b(error|exception|traceback|fatal|panic|warning|deprecated|denied|refused|"
    r"failed|failure|segfault|deadlock|cannot|can't|unable to|undefined reference|"
    r"not found|timed out|timeout|out of memory|oom|killed|permission|conflict)\b",
    re.IGNORECASE,
)


def _extract_anomaly_windows(text: str, max_chars: int, *, context: int = 3) -> str | None:
    """For a non-test command, keep a window of context lines around each
    anomaly-marker line instead of blind head/tail. Returns ``None`` when no
    marker is found anywhere in *text*, so the caller falls back to the
    existing head+tail path unchanged -- a clean run's output is untouched.
    """
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if _ANOMALY_LINE_RE.search(ln)]
    if not hits:
        return None
    windows: list[list[int]] = []
    for i in hits:
        start, end = max(0, i - context), min(len(lines), i + context + 1)
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    parts: list[str] = []
    prev_end = 0
    for start, end in windows:
        if start > prev_end:
            parts.append(f"... ({start - prev_end} lines omitted) ...")
        parts.extend(lines[start:end])
        prev_end = end
    if prev_end < len(lines):
        parts.append(f"... ({len(lines) - prev_end} lines omitted) ...")
    return _cap_chars("\n".join(parts), max_chars)


# Suppress-on-success for the rtk-excluded zone. External compactors only wrap
# read-only/idempotent commands (see external_compactors: "a compactor
# invocation must never be the thing that decides whether a side-effecting
# command runs once or twice"), so noisy *mutating* commands -- git push/pull,
# package installs, docker builds -- always reach this module raw. On success
# their output is almost entirely progress/boilerplate: collapse it to an
# `ok:` one-liner post-hoc (the command already ran exactly once) and keep the
# full text recoverable via the spill store. Failures never take this path.
_SUPPRESS_SUCCESS_RE = re.compile(
    r"^\s*(?:cd\s+[^&|;]+&&\s*)?(?:"
    r"git\s+(?:commit|push|pull|fetch|clone)"
    r"|(?:uv|pip3?|pipx|poetry)\s+(?:pip\s+)?(?:install|sync|add|update)"
    r"|(?:npm|pnpm)\s+(?:install|ci|i|add|update)"
    r"|yarn\s+(?:install|add)"
    r"|bundle\s+install"
    r"|cargo\s+(?:install|fetch)"
    r"|docker\s+(?:build|pull|push)"
    r"|docker\s+compose\s+(?:build|pull)"
    r"|make"
    r")\b",
    re.IGNORECASE,
)
# Below this combined size the output is already cheap; collapsing it would
# save little and the `ok:` line plus marker could even inflate it.
_SUPPRESS_SUCCESS_MIN_CHARS = 600
_SUPPRESS_SALIENT_MAX_CHARS = 200
# git commit's summary is its *first* line ("[main abc1234] message"); for
# everything else (installs, pushes, builds) the last line is the summary.
_LEADING_SUMMARY_RE = re.compile(r"^\[[^\]]+\]")


def _suppress_success_summary(command: str, stdout: str, stderr: str, exit_code: int) -> str | None:
    """``ok: <salient line>`` for a noisy side-effecting command that succeeded.

    Returns None (no suppression) unless ALL hold: exit 0, the command is a
    known noisy mutator, the output is big enough to be worth collapsing, and
    nothing in it looks like an error. An error-looking line on exit 0 falls
    through to ``_extract_anomaly_windows``, which keeps context around it.
    """
    if exit_code != 0 or not _SUPPRESS_SUCCESS_RE.search(command):
        return None
    if len(stdout) + len(stderr) <= _SUPPRESS_SUCCESS_MIN_CHARS:
        return None
    if _ANOMALY_LINE_RE.search(stdout) or _ANOMALY_LINE_RE.search(stderr):
        return None
    # git push/clone write everything to stderr; fall back to it when stdout
    # is empty.
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        lines = [ln.strip() for ln in stderr.splitlines() if ln.strip()]
    if not lines:
        return None
    salient = lines[0] if _LEADING_SUMMARY_RE.match(lines[0]) else lines[-1]
    return f"ok: {salient[:_SUPPRESS_SALIENT_MAX_CHARS]}"


# Upstream flag injection -- the lossless tier: when a command has a
# machine-stable compact format behind a flag (`git status --porcelain`,
# `pytest -q --tb=short`), asking the tool for it up front beats any post-hoc
# trimming because nothing has to be thrown away afterwards. Only exact bare
# forms (plus an optional leading `cd X &&`) are touched, only single-segment
# commands, and only when the agent expressed no formatting intent of its
# own; the injected invocation is announced in the first output line so the
# agent knows which format it is reading. The caller executes the injected
# command but keys every ledger (discipline, delta, budgets, spill) on the
# original -- the mapping is deterministic, so semantics stay consistent.
# Kill switch: LEMONCROW_BASH_FLAG_INJECTION=0.
_FLAG_INJECTION_ENV = "LEMONCROW_BASH_FLAG_INJECTION"
_CD_PREFIX_RE = re.compile(r"^\s*cd\s+[^&|;]+&&\s*")
_PYTEST_RUNNER_TOKENS = frozenset({"uv", "run", "poetry", "pipenv", "hatch", "exec", "-m", "sudo", "time"})
_PYTEST_FORMAT_FLAG_RE = re.compile(r"(?:^|\s)(?:-q|--quiet|-v+\b|--verbose|-r[A-Za-z]+|--tb(?:=|\s))")


def _flag_injection_enabled() -> bool:
    return os.environ.get(_FLAG_INJECTION_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _is_pytest_invocation(body: str) -> bool:
    """True when *body* actually invokes pytest -- not merely mentions it
    (``pip install pytest`` must never get test flags appended)."""
    try:
        tokens = shlex.split(body)
    except ValueError:
        return False
    for idx, tok in enumerate(tokens):
        if tok.rsplit("/", 1)[-1] in {"pytest", "py.test"}:
            return all(
                t in _PYTEST_RUNNER_TOKENS or "=" in t or re.fullmatch(r"python[0-9.]*", t) is not None
                for t in tokens[:idx]
            )
    return False


def _inject_stable_flags(command: str) -> tuple[str, str]:
    """``(exec_command, note)``: append machine-stable output flags when safe.

    Returns *command* unchanged (note == "") unless it is a bare ``git
    status`` / ``git log`` or a single-segment pytest invocation without its
    own verbosity/traceback flags.
    """
    if not _flag_injection_enabled():
        return command, ""
    body = _CD_PREFIX_RE.sub("", command).strip()
    if any(ch in body for ch in ("|", ";", "&", ">", "<")):
        return command, ""
    injected: str | None = None
    if body == "git status":
        injected = f"{command.rstrip()} --porcelain=v1 -b"
    elif body == "git log":
        injected = f"{command.rstrip()} --oneline -n 50"
    elif _is_pytest_invocation(body) and not _PYTEST_FORMAT_FLAG_RE.search(body):
        injected = f"{command.rstrip()} -q --tb=short"
    if injected is None:
        return command, ""
    return injected, f"[ran: {injected}]"


def _compact_result(
    *,
    command: str,
    raw_stdout: str,
    raw_stderr: str,
    exit_code: int,
    duration_ms: int,
    max_lines: int,
    max_chars: int | None = None,
) -> RunResult:
    if exit_code != 0:
        head = 20
        tail = max(max_lines - head, 50)
    else:
        head = max(20, max_lines // 4)
        tail = max(max_lines - head, 0)
    # Successful under-budget output can ship whole. Failures still pass through
    # anomaly extraction even below the char cap: hundreds of routine lines
    # around one buried error are technically small but operationally wasteful.
    line_elision_bypass = exit_code == 0 and max_lines >= _BASH_DEFAULT_MAX_LINES
    budget = max_chars if max_chars is not None else _bash_output_budget(command)

    clean_stdout = _strip_ansi(raw_stdout)
    clean_stderr = _strip_ansi(raw_stderr)
    full_stdout = clean_stdout
    full_stderr = clean_stderr
    terminal_chars_stripped = (len(raw_stdout) - len(clean_stdout)) + (len(raw_stderr) - len(clean_stderr))

    clean_stdout, dedup_stdout_chars = _dedupe_repeated_lines(clean_stdout)
    clean_stderr, dedup_stderr_chars = _dedupe_repeated_lines(clean_stderr)
    profile_stdout = compact_profiled_output(command, clean_stdout, budget=budget, exit_code=exit_code)
    profile_stderr = compact_profiled_output(command, clean_stderr, budget=budget, exit_code=exit_code)
    semantic_stdout = compact_bash_stream(profile_stdout.text, budget=budget)
    semantic_stderr = compact_bash_stream(profile_stderr.text, budget=budget)
    clean_stdout = semantic_stdout.text
    clean_stderr = semantic_stderr.text

    lines_omitted = (
        profile_stdout.lines_omitted
        + profile_stderr.lines_omitted
        + semantic_stdout.lines_omitted
        + semantic_stderr.lines_omitted
    )
    chars_omitted = (
        terminal_chars_stripped
        + dedup_stdout_chars
        + dedup_stderr_chars
        + profile_stdout.chars_saved
        + profile_stderr.chars_saved
        + semantic_stdout.chars_saved
        + semantic_stderr.chars_saved
    )
    lossy = profile_stdout.lossy or profile_stderr.lossy or semantic_stdout.lossy or semantic_stderr.lossy
    stderr_folded = False

    if _is_test_only_command(command):
        stdout_compact = _extract_test_output(clean_stdout, max_chars=budget)
        changed = stdout_compact != clean_stdout
        stdout_omitted = max(0, len(clean_stdout.splitlines()) - len(stdout_compact.splitlines()))
        stdout_chars = max(0, len(clean_stdout) - len(stdout_compact))
        lossy = lossy or changed
    elif (suppressed := _suppress_success_summary(command, clean_stdout, clean_stderr, exit_code)) is not None:
        src_lines = len(clean_stdout.splitlines()) + len(clean_stderr.splitlines())
        stdout_compact = f"{suppressed}\n... ({src_lines} output lines suppressed on success) ..."
        stdout_omitted = len(clean_stdout.splitlines())
        stdout_chars = max(0, len(clean_stdout) - len(stdout_compact))
        stderr_folded = True
        lossy = True
    elif line_elision_bypass and len(clean_stdout) <= budget:
        stdout_compact = "\n".join(clean_stdout.splitlines())
        stdout_omitted = 0
        stdout_chars = 0
    elif len(clean_stdout) <= budget and len(clean_stdout.splitlines()) <= head + tail:
        # Fits as-is: no anomaly-windowing on output that needs no compaction
        # at all -- chopping a small failing result around its 'FAILED' line
        # deletes the surrounding context the model actually needs.
        stdout_compact = clean_stdout
        stdout_omitted = 0
        stdout_chars = 0
    else:
        anomaly = _extract_anomaly_windows(clean_stdout, budget)
        if anomaly is not None:
            stdout_compact = anomaly
            stdout_omitted = max(0, len(clean_stdout.splitlines()) - len(anomaly.splitlines()))
            stdout_chars = max(0, len(clean_stdout) - len(anomaly))
            lossy = lossy or anomaly != clean_stdout
        else:
            stdout_compact, stdout_omitted, stdout_chars = _head_tail_lines(clean_stdout.splitlines(), head, tail)
            lossy = lossy or stdout_omitted > 0
            capped = _cap_chars(stdout_compact, budget)
            if capped != stdout_compact:
                stdout_chars += len(stdout_compact) - len(capped)
                stdout_compact = capped
                lossy = True

    if stderr_folded:
        stderr_compact = ""
        stderr_omitted = len(clean_stderr.splitlines())
        stderr_chars = len(clean_stderr)
    elif line_elision_bypass and len(clean_stderr) <= budget:
        stderr_compact = "\n".join(clean_stderr.splitlines())
        stderr_omitted = 0
        stderr_chars = 0
    else:
        stderr_compact, stderr_omitted, stderr_chars = _head_tail_lines(clean_stderr.splitlines(), 100, 100)
        lossy = lossy or stderr_omitted > 0
        capped_stderr = _cap_chars(stderr_compact, budget)
        if capped_stderr != stderr_compact:
            stderr_chars += len(stderr_compact) - len(capped_stderr)
            stderr_compact = capped_stderr
            lossy = True

    lines_omitted += stdout_omitted + stderr_omitted
    chars_omitted += stdout_chars + stderr_chars
    spill_hint = ""
    if lossy:
        full_text = full_stdout
        if full_stderr.strip():
            full_text = f"{full_text}\n\n--- stderr ---\n{full_stderr}" if full_text else full_stderr
        kept_text = stdout_compact
        if stderr_compact.strip():
            kept_text = f"{kept_text}\n\n--- stderr ---\n{stderr_compact}" if kept_text else stderr_compact
        spill_hint = _spill_hint(full_text, len(kept_text))

    return RunResult(
        stdout=redact_tool_output(stdout_compact),
        stderr=redact_tool_output(stderr_compact),
        exit_code=exit_code,
        duration_ms=duration_ms,
        truncated=lossy,
        lines_omitted=lines_omitted,
        chars_omitted=chars_omitted,
        command=command,
        spill_hint=spill_hint,
    )


def compact_host_bash_output(command: str, stdout: str, stderr: str, exit_code: int | None) -> RunResult:
    """Post-hoc compaction entry for HOST-lane (builtin Bash tool) output.

    The same pipeline the MCP bash lane applies after execution -- ANSI strip,
    dedup-with-count, test-failure extraction, suppress-on-success, anomaly
    windows, per-kind char budgets, spill recovery, secret redaction. Used by
    the Claude plugin's PostToolUse hook (bash_output_shrink.py), which owns
    the result of a command that already ran exactly once -- so unlike
    wrapper-style compactors this is safe for side-effecting commands.

    An unknown exit code is treated as failure: suppress-on-success must never
    collapse a run we cannot prove succeeded.
    """
    return _compact_result(
        command=command,
        raw_stdout=stdout,
        raw_stderr=stderr,
        exit_code=exit_code if exit_code is not None else 1,
        duration_ms=0,
        max_lines=200,
    )


def _neutralize_pipe_fds(*streams: Any) -> None:
    """Release the read ends of a wedged reader's pipes without the double-close
    hazard of os.close(fd).

    A reader stuck in readline() still owns the TextIOWrapper wrapping the fd;
    that wrapper's finalizer (or subprocess teardown) closes the fd *again*
    later -- and if the raw fd number was recycled in the meantime, that second
    close() silently closes an unrelated resource (surfaced as stray
    "Bad file descriptor" noise at best, a wrong-fd close at worst). dup2 of
    /dev/null onto the fd instead keeps the number *allocated* (now pointing at
    /dev/null), so the wrapper's eventual close is a valid single close of a
    still-live fd, and a stuck reader's next read sees EOF.

    fileno()/dup2 never touch the BufferedReader lock the reader holds while
    blocked in readline(), so -- unlike stream.close() -- this can't deadlock
    against the in-flight read. The fd is stably owned by the live wrapper, so
    it can't be recycled between fileno() and dup2 either.
    """
    null_fd: int | None = None
    try:
        null_fd = os.open(os.devnull, os.O_RDONLY)
        for stream in streams:
            if stream is None:
                continue
            with contextlib.suppress(Exception):
                os.dup2(null_fd, stream.fileno())
    finally:
        if null_fd is not None:
            with contextlib.suppress(Exception):
                os.close(null_fd)


def _close_managed_process_pipes(managed: _ManagedCommand) -> None:
    _neutralize_pipe_fds(managed.proc.stdout, managed.proc.stderr)


def _finish_managed_readers(managed: _ManagedCommand, grace_s: float) -> bool:
    reader_wedged = _join_readers_within(managed.readers, grace_s)
    if reader_wedged:
        _close_managed_process_pipes(managed)
        _join_readers_within(managed.readers, 0.2)
    return reader_wedged


def _join_readers_within(readers: list[threading.Thread], grace_s: float) -> bool:
    """Join every reader thread against one shared deadline, not `grace_s`
    per reader -- a naive `for r in readers: r.join(timeout=grace_s)` lets N
    readers extend the real bound to N * grace_s, and lets a reader that
    already timed out quietly finish in the background while a later
    reader's join keeps running, making it look non-wedged by the time
    anyone checks. Returns True if any reader is still alive once the
    shared deadline passes.
    """
    deadline = time.monotonic() + grace_s
    for reader in readers:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            reader.join(timeout=remaining)
    return any(reader.is_alive() for reader in readers)


def _effective_deadline_s(managed: _ManagedCommand) -> float:
    """Seconds since start at which the watcher kills the process.

    A start-time ``timeout`` never kills early -- but a caller who explicitly
    asked to wait LONGER than the one-hour backstop must be respected: the
    process lives at least ``timeout`` seconds (a 6h build with timeout=21600
    must not be killed out from under its own waiting call at 1h). For
    default/shorter timeouts the fixed backstop still reaps forgotten
    processes; ``action="update"`` installs an exact deadline.

    An interactive session instead expires ``idle_ttl`` seconds after its most
    recent input (send_managed_input bumps ``last_input``). Each send is a
    deliberate act -- like ``action="update"`` -- so an actively-fed session
    may outlive the fixed hard cap; an *idle* one always dies within
    ``idle_ttl``. The watcher re-reads this every poll slice, so a send moves
    the deadline live.
    """
    if managed.deadline_explicit:
        return float(managed.timeout)
    if managed.interactive and managed.idle_ttl > 0:
        return (managed.last_input - managed.started) + managed.idle_ttl
    return max(_MANAGED_COMMAND_HARD_CAP_S, float(managed.timeout or 0.0))


def _watch_managed_command(session_id: str) -> None:
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
    if managed is None:
        return
    # Polled in short slices rather than one blocking `proc.wait(timeout=...)`
    # so an action="update" call can move the deadline (managed.timeout /
    # managed.deadline_explicit) without restarting this wait -- each
    # iteration re-reads it live.
    poll_slice_s = 1.0
    while True:
        with _MANAGED_COMMANDS_LOCK:
            deadline = managed.started + _effective_deadline_s(managed)
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            try:
                managed.proc.wait(timeout=0)
            except subprocess.TimeoutExpired:
                # Set the terminal state *before* signalling the process --
                # otherwise a concurrent poll_managed_command can observe the
                # (already-dead) process first and mark it "completed" before
                # this thread gets to flag it "timed_out" (same ordering the
                # cancel path already uses, for the same reason).
                with _MANAGED_COMMANDS_LOCK:
                    if managed.state == "running":
                        managed.state = "timed_out"
                _terminate_process_group(managed.proc)
            else:
                with _MANAGED_COMMANDS_LOCK:
                    if managed.state == "running":
                        managed.state = "completed"
            break
        try:
            managed.proc.wait(timeout=min(remaining, poll_slice_s))
        except subprocess.TimeoutExpired:
            continue
        else:
            with _MANAGED_COMMANDS_LOCK:
                if managed.state == "running":
                    managed.state = "completed"
            break

    # Phase 2 deferred bash: the process has finished, so fire any registered
    # completion callbacks now. A callback collects the result and writes the MCP
    # response (it calls poll_managed_command, which reaps the session), so the
    # grace-sleep+reap below then no-ops. Snapshot under the lock; invoke outside
    # it (the callback re-enters poll_managed_command's lock).
    with _MANAGED_COMMANDS_LOCK:
        cbs = list(managed.on_complete)
        managed.on_complete.clear()
    for cb in cbs:
        with contextlib.suppress(Exception):
            cb()

    # The process has finished. If no one polls the result, its temp files and
    # dict entry would leak forever, so reap it after a grace window. A poll that
    # arrives first reaps it under the lock and clears the entry; this then no-ops.
    time.sleep(_DETACHED_REAP_GRACE_S)
    with _MANAGED_COMMANDS_LOCK:
        if _MANAGED_COMMANDS.get(session_id) is not managed or managed.reaped:
            return
        managed.reaped = True
        _MANAGED_COMMANDS.pop(session_id, None)
    # Let the spool drains finish before closing their temp files; the process
    # has already exited, so the pipes normally EOF and the joins return at
    # once. Bounded for the same reason as poll_managed_command's join below --
    # a still-open duplicate of the pipe (e.g. a detached backgrounded server)
    # must not wedge this cleanup thread forever.
    _finish_managed_readers(managed, _READER_JOIN_GRACE_S)
    with contextlib.suppress(Exception):
        managed.stdout_file.close()
    with contextlib.suppress(Exception):
        managed.stderr_file.close()
    for stream_reader in (managed.stdout_reader, managed.stderr_reader):
        if stream_reader is not None:
            stream_reader.finalize()


def _spool_managed_stream(stream: Any, dst_file: Any, managed: _ManagedCommand) -> None:
    """Drain *stream* into *dst_file*, capped at the on-disk spool ceiling.

    Runs for the command's lifetime in a daemon thread; `_pump_capped` stops
    appending once `_MAX_SPOOL_BYTES` is reached but keeps reading to EOF so the
    child never blocks on a full pipe. Flags the session as spool-truncated when
    either stream overflows. Writes go through `managed.output_lock` so a
    concurrent `status` peek can't interleave a cursor move with a write.
    """

    def _locked_write(text: str) -> None:
        with managed.output_lock:
            dst_file.write(text)

    with contextlib.suppress(Exception):
        truncated = _pump_capped(stream, _locked_write, _MAX_SPOOL_BYTES)
        if truncated:
            with _MANAGED_COMMANDS_LOCK:
                managed.spool_truncated = True
    if isinstance(stream, _LossyStreamReader):
        stream.finalize()


def start_managed_command(
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 30,
    max_lines: int = _BASH_DEFAULT_MAX_LINES,
    max_chars: int | None = None,
    note: str = "",
    explicit_background: bool = False,
    interactive: bool = False,
    idle_ttl: float | None = None,
    owner: str = "",
) -> dict[str, Any]:
    """Start a command without blocking the MCP request.

    ``timeout`` is only the soft response budget. The process uses the fixed
    one-hour safety cap unless ``action="update"`` installs an exact deadline.
    Foreground commands remain MCP-session-owned after that budget; only
    ``explicit_background=True`` commands survive MCP shutdown.

    ``interactive=True`` opens the child with a live stdin pipe and a sliding
    ``idle_ttl`` kill window instead of the hard cap -- a long-lived REPL fed
    via ``send_managed_input`` that dies on its own once it goes unused.

    *note*, when given, seeds the managed command's ``injected_note`` (prepended
    to the compacted stdout at poll time) -- used by a caller that rewrote the
    command (e.g. a pipeline seek) to tell the model what actually ran.
    """
    policy = classify_command(command, cwd=cwd)
    if policy.action == "block":
        return {
            "status": "blocked",
            "stderr": policy.reason,
            "exit_code": -1,
            "blocked": True,
            "blocked_reason": policy.reason,
        }

    exec_command, injected_note = _inject_stable_flags(command)
    # A caller-supplied note (e.g. a pipeline rewrite) wins: the rewritten
    # command won't trigger flag injection, and the note explains the transform.
    injected_note = note or injected_note
    session_id = _new_session_id()
    stdout_file, stdout_path = _open_stream_file(session_id, "stdout")
    stderr_file, stderr_path = _open_stream_file(session_id, "stderr")
    stdout_raw, stdout_raw_path = _open_raw_stream_file(session_id, "stdout")
    stderr_raw, stderr_raw_path = _open_raw_stream_file(session_id, "stderr")
    try:
        # Pipe the child's output through drain threads rather than handing the
        # temp-file fds straight to the kernel. A direct fd lets a runaway
        # producer (`cat /dev/zero`) fill the disk before any poll reads it; the
        # spool pump caps each temp file at `_MAX_SPOOL_BYTES` instead.
        # stdin=DEVNULL: the MCP server's stdin is an open JSON-RPC pipe, so
        # inheriting it causes any child that reads stdin (e.g. `sys.stdin.read()`
        # in a python -c snippet) to block forever instead of failing fast.
        # Interactive sessions instead get a private PIPE that action="send"
        # (send_managed_input) feeds -- no inheritance hazard.
        # start_new_session=True calls setsid() in the child, placing it in
        # its own session and process group.  This has two effects:
        #   1. The child is detached from the MCP server's process group --
        #      if the MCP process dies the child is NOT sent SIGHUP and keeps
        #      running (safe background / long-lived server use-case).
        #   2. _terminate_process_group() can cleanly kill the whole subtree
        #      via SIGTERM/SIGKILL to the child's own pgid.
        proc = subprocess.Popen(
            ["bash", "-c", exec_command],
            stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Binary pipes: decoding happens in _LossyStreamReader (utf-8 with
            # errors="replace" -- a strict decode would raise in the reader
            # thread on the first invalid byte, silently emptying the whole
            # stream), so substitution is detected exactly and the raw bytes
            # stay recoverable when a stream turns out non-UTF8.
            cwd=cwd,
            start_new_session=True,
        )
    except Exception:
        stdout_file.close()
        stderr_file.close()
        for raw_handle in (stdout_raw, stderr_raw):
            if raw_handle is not None:
                with contextlib.suppress(Exception):
                    raw_handle.close()
        raise
    stdout_reader = _LossyStreamReader(
        proc.stdout, raw_file=stdout_raw, raw_path=stdout_raw_path, raw_cap=_MAX_SPOOL_BYTES
    )
    stderr_reader = _LossyStreamReader(
        proc.stderr, raw_file=stderr_raw, raw_path=stderr_raw_path, raw_cap=_MAX_SPOOL_BYTES
    )

    now = time.perf_counter()
    managed = _ManagedCommand(
        command=command,
        proc=proc,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        started=now,
        timeout=timeout,
        max_lines=max_lines,
        max_chars=max_chars,
        explicit_background=explicit_background,
        injected_note=injected_note,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_reader=stdout_reader,
        stderr_reader=stderr_reader,
        interactive=interactive,
        idle_ttl=(
            min(max(float(idle_ttl if idle_ttl is not None else _DEFAULT_IDLE_TTL_S), 1.0), _MANAGED_COMMAND_HARD_CAP_S)
            if interactive
            else 0.0
        ),
        last_input=now,
    )
    managed.owner = owner
    managed.readers = [
        threading.Thread(target=_spool_managed_stream, args=(stdout_reader, stdout_file, managed), daemon=True),
        threading.Thread(target=_spool_managed_stream, args=(stderr_reader, stderr_file, managed), daemon=True),
    ]
    for reader in managed.readers:
        reader.start()
    with _MANAGED_COMMANDS_LOCK:
        _MANAGED_COMMANDS[session_id] = managed
    threading.Thread(
        target=_watch_managed_command,
        args=(session_id,),
        daemon=True,
        name=f"lemoncrow-shell-{session_id[:8]}",
    ).start()
    started_payload = {
        "status": "running",
        "session_id": session_id,
        "pid": proc.pid,
        "timeout": timeout,
        "explicit_background": managed.explicit_background,
    }
    if interactive:
        started_payload["interactive"] = True
        started_payload["idle_ttl"] = managed.idle_ttl
    if managed.stdout_path:
        started_payload["log_file"] = managed.stdout_path
    if managed.stderr_path:
        started_payload["log_file_stderr"] = managed.stderr_path
    return started_payload


def cleanup_managed_commands() -> dict[str, list[dict[str, Any]]]:
    """Terminate live MCP-owned commands and preserve explicit background jobs.

    Idempotent and suitable for both the MCP server's ``finally`` block and
    interpreter ``atexit`` cleanup. Process-group termination covers every
    descendant of each foreground shell, not only its immediate PID.
    """
    terminated: list[tuple[str, _ManagedCommand]] = []
    preserved: list[dict[str, Any]] = []
    with _MANAGED_COMMANDS_LOCK:
        for session_id, managed in _MANAGED_COMMANDS.items():
            if managed.proc.poll() is not None:
                continue
            details: dict[str, Any] = {
                "session_id": session_id,
                "pid": managed.proc.pid,
            }
            if managed.stdout_path:
                details["log_file"] = managed.stdout_path
            if managed.stderr_path:
                details["log_file_stderr"] = managed.stderr_path
            if managed.explicit_background:
                preserved.append(details)
                continue
            if managed.state == "running":
                managed.state = "cancelled"
            terminated.append((session_id, managed))

    terminated_details: list[dict[str, Any]] = []
    for session_id, managed in terminated:
        _terminate_process_group(managed.proc)
        details = {"session_id": session_id, "pid": managed.proc.pid}
        if managed.stdout_path:
            details["log_file"] = managed.stdout_path
        if managed.stderr_path:
            details["log_file_stderr"] = managed.stderr_path
        terminated_details.append(details)
    return {"terminated": terminated_details, "preserved": preserved}


def cleanup_commands_for_owner(owner: str) -> dict[str, list[dict[str, Any]]]:
    """Terminate one owner's live foreground commands; preserve explicit bg.

    Called when a single host session (bridge) disconnects from a shared
    singleton MCP daemon: only that session's foreground shells are reaped --
    other sessions' commands and any ``bg=true`` jobs keep running (bg is
    expected to outlive its launching session, matching stdio semantics). No-op
    for a falsy owner (e.g. the stdio path, where nothing tags an owner).
    """
    if not owner:
        return {"terminated": [], "preserved": []}
    terminated: list[tuple[str, _ManagedCommand]] = []
    preserved: list[dict[str, Any]] = []
    with _MANAGED_COMMANDS_LOCK:
        for session_id, managed in _MANAGED_COMMANDS.items():
            if managed.owner != owner or managed.proc.poll() is not None:
                continue
            if managed.explicit_background:
                preserved.append({"session_id": session_id, "pid": managed.proc.pid})
                continue
            if managed.state == "running":
                managed.state = "cancelled"
            terminated.append((session_id, managed))
    terminated_details: list[dict[str, Any]] = []
    for session_id, managed in terminated:
        _terminate_process_group(managed.proc)
        terminated_details.append({"session_id": session_id, "pid": managed.proc.pid})
    return {"terminated": terminated_details, "preserved": preserved}


atexit.register(cleanup_managed_commands)


def _tail_managed_output(managed: _ManagedCommand, n: int) -> tuple[list[str], list[str]]:
    """(stdout_tail, stderr_tail): up to the last *n* lines of each stream."""
    with managed.output_lock:
        return (
            _tail_lines_from_file(managed.stdout_file, n),
            _tail_lines_from_file(managed.stderr_file, n),
        )


_STATUS_TAIL_LINES = 10


def peek_managed_command(session_id: str, *, tail_lines: int = _STATUS_TAIL_LINES) -> dict[str, Any]:
    """Non-blocking status snapshot: state, pid, timing, and a `tail`-style
    look at output collected so far.

    Unlike `poll_managed_command`, this never blocks on the command finishing,
    never reaps the session, and never closes its spool files -- a later
    `poll`/`cancel` on the same session_id still behaves normally.
    """
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
        if managed is None:
            raise KeyError(f"unknown shell session: {session_id}")
        running = managed.proc.poll() is None
        state = managed.state
    # The watcher records "completed" immediately after wait() returns, but a
    # fast command can exit between register_completion() and that state write.
    # Never expose that finished process as still running; poll() will perform
    # the authoritative terminal transition and reader drain when it reaps.
    status = "running" if running else ("completed" if state == "running" else state)
    elapsed_ms = int((time.perf_counter() - managed.started) * 1000)
    stdout_tail, stderr_tail = _tail_managed_output(managed, tail_lines)
    payload: dict[str, Any] = {
        "status": status,
        "session_id": session_id,
        "pid": managed.proc.pid,
        "duration_ms": elapsed_ms,
        "stdout": _strip_ansi("\n".join(stdout_tail)),
        "stderr": _strip_ansi("\n".join(stderr_tail)),
        "tail_lines": tail_lines,
        "explicit_background": managed.explicit_background,
    }
    if managed.stdout_path:
        payload["log_file"] = managed.stdout_path
    if managed.stderr_path:
        payload["log_file_stderr"] = managed.stderr_path
    if any(r is not None and r.lossy for r in (managed.stdout_reader, managed.stderr_reader)):
        payload["output_lossy"] = True
    if running:
        remaining_ms = max(0, managed.timeout * 1000 - elapsed_ms)
        payload["timeout_remaining_ms"] = remaining_ms
        # Distinct signal from a plain mid-flight peek: this command has
        # already burned through its requested soft-timeout budget -- once
        # action="update" has installed an explicit deadline (see
        # _ManagedCommand.deadline_explicit), this is also exactly how long
        # until the real kill, not just a soft nudge.
        payload["over_budget"] = remaining_ms <= 0
    else:
        payload["exit_code"] = managed.proc.returncode
    return payload


def poll_managed_command(session_id: str, *, cancel: bool = False) -> dict[str, Any]:
    """Poll or cancel a managed command."""
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
        if managed is None:
            raise KeyError(f"unknown shell session: {session_id}")
        if cancel and managed.state == "running":
            managed.state = "cancelled"

    if cancel and managed.proc.poll() is None:
        _terminate_process_group(managed.proc)

    if managed.proc.poll() is None:
        elapsed_ms = int((time.perf_counter() - managed.started) * 1000)
        timeout_remaining_ms = max(0, managed.timeout * 1000 - elapsed_ms)
        running_payload = {
            "status": "running",
            "session_id": session_id,
            "pid": managed.proc.pid,
            "duration_ms": elapsed_ms,
            "timeout_remaining_ms": timeout_remaining_ms,
            "over_budget": timeout_remaining_ms <= 0,
            "explicit_background": managed.explicit_background,
        }
        if managed.stdout_path:
            running_payload["log_file"] = managed.stdout_path
        if managed.stderr_path:
            running_payload["log_file_stderr"] = managed.stderr_path
        return running_payload

    # Join the spool drains before reading -- the process is done, so the pipes
    # normally EOF and the threads exit promptly, leaving every surviving byte
    # on disk. Bounded: a detached descendant that still holds the pipe open
    # (e.g. a backgrounded server a task explicitly asked to be left running)
    # would otherwise wedge this join forever even though the command we
    # actually ran has already exited -- ship whatever's on disk so far
    # instead. Join outside the lock: a drain takes the lock to flag
    # truncation. One shared deadline across every reader (not one grace
    # window each) -- see _join_readers_within.
    reader_wedged = _finish_managed_readers(managed, _READER_JOIN_GRACE_S)

    with _MANAGED_COMMANDS_LOCK:
        if managed.reaped:
            # The watcher already reaped this finished session; its temp files are
            # closed. Report completion without re-reading or double-closing.
            raise KeyError(f"unknown shell session: {session_id}")
        if managed.state == "running":
            managed.state = "completed"
        managed.reaped = True
        _MANAGED_COMMANDS.pop(session_id, None)
        # A reader thread may still be alive here (the bounded join above timed
        # out) -- output_lock serializes our read against any in-flight write
        # so we never see a torn buffer.
        with managed.output_lock:
            managed.stdout_file.flush()
            managed.stderr_file.flush()
            managed.stdout_file.seek(0)
            managed.stderr_file.seek(0)
            raw_stdout, stdout_capped = _read_capped(managed.stdout_file)
            raw_stderr, stderr_capped = _read_capped(managed.stderr_file)
            managed.stdout_file.close()
            managed.stderr_file.close()
    for stream_reader in (managed.stdout_reader, managed.stderr_reader):
        if stream_reader is not None:
            stream_reader.finalize()
    output_byte_capped = stdout_capped or stderr_capped or managed.spool_truncated
    if stdout_capped:
        raw_stdout += _OUTPUT_CAP_NOTICE.format(cap=_MAX_OUTPUT_BYTES)
    if stderr_capped:
        raw_stderr += _OUTPUT_CAP_NOTICE.format(cap=_MAX_OUTPUT_BYTES)

    if managed.state == "timed_out":
        exit_code = -1
        if managed.interactive and not managed.deadline_explicit:
            raw_stderr = f"Interactive session idle-expired after {int(managed.idle_ttl)}s without input"
        else:
            raw_stderr = f"Command timed out after {int(_effective_deadline_s(managed))}s"
    elif managed.state == "cancelled":
        exit_code = -1
        raw_stderr = "Command cancelled"
    else:
        exit_code = managed.proc.returncode
    if reader_wedged:
        wedged_note = (
            "note: output ends where this command exited; a child may still be " "running and holding the stream open."
        )
        raw_stderr = f"{raw_stderr}\n{wedged_note}" if raw_stderr else wedged_note
    result = _compact_result(
        command=managed.command,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        exit_code=exit_code,
        duration_ms=int((time.perf_counter() - managed.started) * 1000),
        max_lines=managed.max_lines,
        max_chars=managed.max_chars,
    )
    if managed.injected_note:
        result.stdout = f"{managed.injected_note}\n{result.stdout}" if result.stdout else managed.injected_note
    out_lossy = bool(managed.stdout_reader is not None and managed.stdout_reader.lossy)
    err_lossy = bool(managed.stderr_reader is not None and managed.stderr_reader.lossy)
    if out_lossy:
        notice = _lossy_notice("stdout", managed.stdout_reader.raw_path)
        result.stdout = f"{notice}\n{result.stdout}" if result.stdout else notice
    if err_lossy:
        notice = _lossy_notice("stderr", managed.stderr_reader.raw_path)
        result.stderr = f"{notice}\n{result.stderr}" if result.stderr else notice
    payload = {
        "status": managed.state,
        "session_id": session_id,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
        "duration_ms": result.duration_ms,
        "truncated": result.truncated or output_byte_capped,
        "lines_omitted": result.lines_omitted,
        "chars_omitted": result.chars_omitted,
        "spill_hint": result.spill_hint,
        "explicit_background": managed.explicit_background,
    }
    if managed.stdout_path:
        payload["log_file"] = managed.stdout_path
    if managed.stderr_path:
        payload["log_file_stderr"] = managed.stderr_path
    if out_lossy or err_lossy:
        payload["output_lossy"] = True
        if out_lossy and managed.stdout_reader.raw_path:
            payload["raw_log_file"] = managed.stdout_reader.raw_path
        if err_lossy and managed.stderr_reader.raw_path:
            payload["raw_log_file_stderr"] = managed.stderr_reader.raw_path
    return payload


def update_managed_command(session_id: str, timeout: float) -> dict[str, Any]:
    """Install (or move) a running managed command's *enforced* kill deadline
    (bash action="update").

    Before any update, `timeout` is only the soft response budget and the
    fixed one-hour safety cap controls process lifetime. Calling update is a
    deliberate act: from here on `timeout` is the exact, enforced
    deadline for this session, seconds since it started -- not "N more
    seconds from now" -- so a caller can read `timeout_remaining_ms` off a
    prior peek/poll and reason about the new absolute budget directly.
    Clamped to `_MAX_EXPLICIT_TIMEOUT_S` regardless of how many times a
    session gets extended: that ceiling is the real backstop against a task
    granting a forgotten background job unbounded life one update at a time.
    """
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
        if managed is None:
            raise KeyError(f"unknown shell session: {session_id}")
        if managed.state != "running" or managed.proc.poll() is not None:
            return {"status": managed.state, "session_id": session_id, "updated": False}
        managed.timeout = min(float(timeout), _MAX_EXPLICIT_TIMEOUT_S)
        managed.deadline_explicit = True
        applied = managed.timeout
        elapsed_ms = int((time.perf_counter() - managed.started) * 1000)
    return {
        "status": "running",
        "session_id": session_id,
        "updated": True,
        "timeout": applied,
        "timeout_remaining_ms": max(0, int(applied * 1000) - elapsed_ms),
    }


def _read_stream_delta(handle: Any, offset: int) -> tuple[str, int, bool]:
    """Read everything written to *handle* past *offset*, restoring the
    writer's append position. Returns (text, new_offset, capped).

    Offsets are opaque text-mode tell() cookies (0 or a prior tell() value),
    never arithmetic -- same constraint as _tail_lines_from_file. Caller must
    hold the session's output_lock so a concurrent spool write can't land
    between the seeks.
    """
    handle.flush()
    pos = handle.tell()
    if pos == offset:
        return "", pos, False
    handle.seek(offset)
    text, capped = _read_capped(handle)
    handle.seek(pos)
    return text, pos, capped


def send_managed_input(session_id: str, text: str, *, wait: float = 30.0) -> dict[str, Any]:
    """Feed *text* to an interactive session's stdin and return the output
    delta it produced (bash action="send").

    Framing is quiescence-based: after writing, wait until neither stream has
    grown for `_SEND_QUIESCENCE_S` (or *wait* runs out, or the process exits),
    then return only the bytes produced since the previous send -- output that
    arrived *between* sends is included, nothing is dropped. A child still
    computing past *wait* returns whatever arrived so far; a follow-up *empty*
    send blocks until more output actually arrives (growth-gated, no
    zero-growth quiescence exit) and drains it.

    Every send -- including an empty drain -- resets the session's idle-TTL
    clock (see _effective_deadline_s).
    """
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
        if managed is None:
            raise KeyError(f"unknown shell session: {session_id}")
        if not managed.interactive:
            raise ValueError(f"session {session_id} is not interactive; start it with interactive=true")
        alive = managed.state == "running" and managed.proc.poll() is None

    # The child usually isn't a shell, but the input often is shell-shaped (an
    # interactive bash, a REPL shelling out); the same policy gate as a
    # top-level command costs nothing and closes the obvious escape hatch.
    # Only "block" is honored -- rewrites target one-shot shell commands.
    if text.strip():
        policy = classify_command(text)
        if policy.action == "block":
            return {
                "status": "running" if alive else managed.state,
                "session_id": session_id,
                "blocked": True,
                "blocked_reason": policy.reason,
                "stderr": policy.reason,
                "exit_code": -1,
                "interactive": True,
            }

    sent = False
    if alive and text:
        stdin = managed.proc.stdin
        if stdin is None:
            alive = False
        else:
            try:
                data = text if text.endswith("\n") else text + "\n"
                stdin.write(data.encode("utf-8"))  # binary pipe: see start_managed_command's Popen
                stdin.flush()
                sent = True
            except (OSError, ValueError):
                alive = False  # pipe closed under us: the child just died
    with _MANAGED_COMMANDS_LOCK:
        # Reset the idle clock on every send -- an empty drain is activity too.
        managed.last_input = time.perf_counter()

    def _sizes() -> tuple[Any, Any]:
        with managed.output_lock:
            return managed.stdout_file.tell(), managed.stderr_file.tell()

    send_started = time.perf_counter()
    deadline = send_started + max(0.0, float(wait))
    baseline = _sizes()
    prev = baseline
    last_growth = send_started
    while managed.proc.poll() is None:
        now = time.perf_counter()
        if now >= deadline:
            break
        # An empty send is a pure drain: zero-growth quiescence just means
        # "nothing yet", so keep waiting for output until the budget runs out.
        if (sent or prev != baseline) and now - last_growth >= _SEND_QUIESCENCE_S:
            break
        time.sleep(min(_SEND_POLL_SLICE_S, deadline - now))
        cur = _sizes()
        if cur != prev:
            prev = cur
            last_growth = time.perf_counter()

    with managed.output_lock:
        stdout_delta, managed.stdout_read_offset, out_capped = _read_stream_delta(
            managed.stdout_file, managed.stdout_read_offset
        )
        stderr_delta, managed.stderr_read_offset, err_capped = _read_stream_delta(
            managed.stderr_file, managed.stderr_read_offset
        )

    budget = managed.max_chars if managed.max_chars is not None else _bash_output_budget(managed.command)

    def _compact_delta(delta: str) -> tuple[str, int, int, bool]:
        cleaned = _strip_ansi(delta)
        deduped, dedup_chars = _dedupe_repeated_lines(cleaned)
        profiled = compact_profiled_output(
            managed.command,
            deduped,
            budget=budget,
            exit_code=managed.proc.returncode,
        )
        compacted = compact_bash_stream(profiled.text, budget=budget)
        body = compacted.text
        omitted = profiled.lines_omitted + compacted.lines_omitted
        chars = dedup_chars + profiled.chars_saved + compacted.chars_saved + max(0, len(delta) - len(cleaned))
        lossy = profiled.lossy or compacted.lossy
        lines = body.splitlines()
        if len(lines) > managed.max_lines:
            omitted += len(lines) - managed.max_lines
            trimmed = "\n".join(lines[-managed.max_lines :])
            chars += max(0, len(body) - len(trimmed))
            body = trimmed
            lossy = True
        capped = _cap_chars(body, budget)
        if capped != body:
            chars += len(body) - len(capped)
            body = capped
            lossy = True
        return body.rstrip("\n"), omitted, chars, lossy

    stdout_text, out_omitted, out_chars, out_lossy = _compact_delta(stdout_delta)
    stderr_text, err_omitted, err_chars, err_lossy = _compact_delta(stderr_delta)

    running = managed.proc.poll() is None
    with _MANAGED_COMMANDS_LOCK:
        state = managed.state
        idle_deadline = managed.last_input + managed.idle_ttl
    truncated = out_capped or err_capped or out_lossy or err_lossy
    payload: dict[str, Any] = {
        "status": "running" if running else ("completed" if state == "running" else state),
        "session_id": session_id,
        "pid": managed.proc.pid,
        "stdout": redact_tool_output(stdout_text),
        "stderr": redact_tool_output(stderr_text),
        "duration_ms": int((time.perf_counter() - send_started) * 1000),
        "interactive": True,
        "sent": sent,
        "truncated": truncated,
        "lines_omitted": out_omitted + err_omitted,
        "chars_omitted": out_chars + err_chars,
    }
    if running:
        payload["idle_ttl_remaining_ms"] = max(0, int((idle_deadline - time.perf_counter()) * 1000))
    else:
        payload["exit_code"] = managed.proc.returncode
    if managed.stdout_path:
        payload["log_file"] = managed.stdout_path
    if managed.stderr_path:
        payload["log_file_stderr"] = managed.stderr_path
    if truncated:
        logs = ", ".join(path for path in (managed.stdout_path, managed.stderr_path) if path)
        if logs:
            payload["spill_hint"] = f"[lc: interactive output compacted; full session logs: {logs}]"
    return payload


def register_completion(session_id: str, callback: Callable[[], None]) -> bool:
    """Arm a completion callback for a running managed command (Phase 2).

    Returns ``True`` and appends the callback (the watcher fires it once the
    process finishes) only if the session is known and still running. Returns
    ``False`` if the session is unknown or already finished/reaped -- the caller
    must then fire its own continuation immediately.
    """
    with _MANAGED_COMMANDS_LOCK:
        managed = _MANAGED_COMMANDS.get(session_id)
        if managed is None:
            return False
        if managed.state != "running" or managed.proc.poll() is not None:
            return False
        managed.on_complete.append(callback)
        return True


def run_command(
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 30,
    max_lines: int = _BASH_DEFAULT_MAX_LINES,
) -> RunResult:
    """Execute *command* in bash, return token-compact structured output.

    Optimizations vs. raw subprocess:
    - ANSI escape codes stripped (progress bars, colors → garbage tokens).
    - stdout truncated head+tail: first 25% for context, last 75% for results/errors.
    - stderr head+tail compacted (100/100 lines); omitted lines recoverable via spill.
    - Structured return: LLM checks exit_code first, reads output only if needed.
    """
    policy = classify_command(command, cwd=cwd)
    if policy.action == "block":
        return RunResult(
            stdout="",
            stderr=policy.reason,
            exit_code=-1,
            duration_ms=0,
            truncated=False,
            lines_omitted=0,
            command=command,
            policy_category=policy.category,
            policy_action=policy.action,
            policy_reason=policy.reason,
            rewrite_target=policy.rewrite_target,
            rewrite_payload=policy.rewrite_payload,
        )

    # Fast-path: execute head/tail/wc directly in Python — no fork, no exec,
    # no gate check (we're not spawning a shell).  Latency drops from ~40 ms
    # to <1 ms for these common file-inspection commands.
    if (
        policy.action == "rewrite"
        and policy.rewrite_target in {"cat", "head", "tail", "wc"}
        and policy.rewrite_payload is not None
    ):
        _t0 = time.perf_counter()
        _stdout, _stderr, _exit = execute_inline_op(policy.rewrite_target, policy.rewrite_payload, cwd)
        _dur = int((time.perf_counter() - _t0) * 1000)
        _result = _compact_result(
            command=command,
            raw_stdout=_stdout,
            raw_stderr=_stderr,
            exit_code=_exit,
            duration_ms=_dur,
            max_lines=max_lines,
        )
        _result.policy_category = policy.category
        _result.policy_action = policy.action
        _result.policy_reason = policy.reason
        _result.rewrite_target = policy.rewrite_target
        _result.rewrite_payload = policy.rewrite_payload
        return _result

    exec_source = command
    if policy.action == "rewrite" and policy.rewrite_target == "external_compactor" and policy.rewrite_payload:
        exec_source = str(policy.rewrite_payload.get("command") or command)
    exec_command, injected_note = _inject_stable_flags(exec_source)
    started = time.perf_counter()
    proc: subprocess.Popen[bytes] | None = None
    output_byte_capped = False
    stdout_reader: _LossyStreamReader | None = None
    stderr_reader: _LossyStreamReader | None = None
    try:
        proc = subprocess.Popen(
            ["bash", "-c", exec_command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Binary pipes; _LossyStreamReader decodes utf-8 with errors="replace"
            # (a strict decode would raise inside _drain's suppress() on the first
            # invalid byte, silently emptying the captured stream -- measured:
            # doom's non-UTF8 boot log came back as "" on exit 0) while exactly
            # flagging substitution and teeing raw bytes for recovery.
            cwd=cwd,
            start_new_session=True,
        )
        raw_sid = _new_session_id()
        out_raw, out_raw_path = _open_raw_stream_file(raw_sid, "stdout")
        err_raw, err_raw_path = _open_raw_stream_file(raw_sid, "stderr")
        stdout_reader = _LossyStreamReader(
            proc.stdout, raw_file=out_raw, raw_path=out_raw_path, raw_cap=_MAX_OUTPUT_BYTES
        )
        stderr_reader = _LossyStreamReader(
            proc.stderr, raw_file=err_raw, raw_path=err_raw_path, raw_cap=_MAX_OUTPUT_BYTES
        )
        # Drain both pipes concurrently into bounded in-memory buffers. A plain
        # `communicate()` slurps the child's *entire* output into RAM before any
        # cap runs, so a runaway producer (`yes`, `cat /dev/zero`) OOMs the host.
        # `_pump_capped` stops accumulating at `_MAX_OUTPUT_BYTES` per stream but
        # keeps reading to EOF, and running one thread per stream avoids the
        # pipe-buffer deadlock when both stdout and stderr are large.
        stdout_buf: list[str] = []
        stderr_buf: list[str] = []
        capped = {"stdout": False, "stderr": False}

        def _drain(stream: Any, buf: list[str], key: str) -> None:
            with contextlib.suppress(Exception):
                capped[key] = _pump_capped(stream, buf.append, _MAX_OUTPUT_BYTES)

        readers = [
            threading.Thread(target=_drain, args=(stdout_reader, stdout_buf, "stdout"), daemon=True),
            threading.Thread(target=_drain, args=(stderr_reader, stderr_buf, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()

        def _finish_run_readers() -> None:
            # Bounded join, same hazard the managed path guards against in
            # _finish_managed_readers: a self-daemonizing child can fork a
            # grandchild that inherits and holds this pipe open, leaving _drain
            # blocked in readline() long after `proc` (the wrapping bash) has
            # exited -- an unbounded join here hangs the whole synchronous call
            # forever. Join with a grace; on wedge neutralize the raw fds (dup2
            # /dev/null, never stream.close -- that serializes with the reader's
            # in-flight readline on the BufferedReader lock and would deadlock
            # this thread) and ship whatever was captured before the command's
            # own process exited.
            if proc is None:
                return
            if _join_readers_within(readers, _READER_JOIN_GRACE_S):
                _neutralize_pipe_fds(proc.stdout, proc.stderr)
                _join_readers_within(readers, 0.2)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the group first so the child's pipes close (the common case);
            # a grandchild that setsid'd out of the group can still hold the pipe
            # open, so bound the join the same way the success path does.
            _terminate_process_group(proc)
            _finish_run_readers()
            raise
        _finish_run_readers()
        exit_code = proc.returncode
        raw_stdout = _strip_ansi("".join(stdout_buf))
        raw_stderr = _strip_ansi("".join(stderr_buf))
        stdout_capped = capped["stdout"]
        stderr_capped = capped["stderr"]
        output_byte_capped = stdout_capped or stderr_capped
        if stdout_capped:
            raw_stdout += _OUTPUT_CAP_NOTICE.format(cap=_MAX_OUTPUT_BYTES)
        if stderr_capped:
            raw_stderr += _OUTPUT_CAP_NOTICE.format(cap=_MAX_OUTPUT_BYTES)
    except subprocess.TimeoutExpired:
        exit_code = -1
        raw_stdout = ""
        raw_stderr = f"Command timed out after {timeout}s"
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        exit_code = -1
        raw_stdout = ""
        raw_stderr = str(exc)

    for sync_reader in (stdout_reader, stderr_reader):
        if sync_reader is not None:
            sync_reader.finalize()
    duration_ms = int((time.perf_counter() - started) * 1000)

    if output_delta.observe(command, cwd=cwd, stdout=raw_stdout, stderr=raw_stderr, exit_code=exit_code):
        # Run-and-dedup (execution NEVER skipped): the command really ran and
        # produced byte-identical output to its previous run this session, so
        # every byte is already in the model's context -- ship a marker. The
        # full text is still spilled for recovery, because re-running the
        # command would just yield the marker again.
        total_lines = len(raw_stdout.splitlines()) + len(raw_stderr.splitlines())
        total_chars = len(raw_stdout) + len(raw_stderr)
        anchor_src = raw_stdout if raw_stdout.strip() else raw_stderr
        anchor = next((ln.strip() for ln in anchor_src.splitlines() if ln.strip()), "")
        marker = (
            "unchanged: output byte-identical to this command's previous run this session "
            f'(exit 0, {total_lines} lines, {total_chars} chars; first line: "{anchor[:120]}")'
        )
        full_text = raw_stdout
        if raw_stderr.strip():
            full_text = f"{full_text}\n\n--- stderr ---\n{raw_stderr}" if full_text else raw_stderr
        result = RunResult(
            stdout=marker,
            stderr="",
            exit_code=exit_code,
            duration_ms=duration_ms,
            truncated=True,
            lines_omitted=total_lines,
            chars_omitted=max(0, total_chars - len(marker)),
            command=command,
            spill_hint=_spill_hint(full_text, len(marker)),
        )
    else:
        result = _compact_result(
            command=command,
            raw_stdout=raw_stdout,
            raw_stderr=raw_stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
            max_lines=max_lines,
        )
        if injected_note:
            result.stdout = f"{injected_note}\n{result.stdout}" if result.stdout else injected_note
    if stdout_reader is not None and stdout_reader.lossy:
        notice = _lossy_notice("stdout", stdout_reader.raw_path)
        result.stdout = f"{notice}\n{result.stdout}" if result.stdout else notice
    if stderr_reader is not None and stderr_reader.lossy:
        notice = _lossy_notice("stderr", stderr_reader.raw_path)
        result.stderr = f"{notice}\n{result.stderr}" if result.stderr else notice
    result.truncated = result.truncated or output_byte_capped
    result.policy_category = policy.category
    result.policy_action = policy.action
    result.policy_reason = policy.reason
    result.rewrite_target = policy.rewrite_target
    result.rewrite_payload = policy.rewrite_payload
    return result


__all__ = [
    "CommandPolicyDecision",
    "RunResult",
    "classify_command",
    "execute_inline_op",
    "poll_managed_command",
    "register_completion",
    "run_command",
    "start_managed_command",
]
