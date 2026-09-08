"""Path helpers for separating runtime state from Git-tracked lessons."""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from datetime import date as _date
from datetime import timedelta as _timedelta
from hashlib import sha256 as _sha256
from pathlib import Path

DEFAULT_STORE_DIRNAME = ".lemoncrow"


class WorkspaceNotRegisteredError(RuntimeError):
    """Raised when cwd is neither a git repo nor an explicitly `lc init`-registered
    workspace. Non-git directories are never silently auto-registered."""


# The colon is intentional: agent identifiers are namespaced (``lemoncrow:code``)
# and have always been used as directory segments. A colon cannot escape a
# directory, so admitting it preserves the traversal protection below.
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def safe_segment(value: str, *, field: str = "value") -> str:
    """Accept only a single, traversal-free path segment.

    Ids that reach the filesystem from an HTTP route (session ids, host names,
    skill names) must never carry ``/``, ``..`` or an absolute prefix --
    otherwise ``root / "sessions" / value`` escapes the store entirely. Raises
    rather than sanitising: a caller that silently rewrote an id would read or
    write the *wrong* session, which is worse than a 4xx.

    Raises:
        ValueError: if *value* is not a safe single segment.
    """
    text = str(value)
    if text in {".", ".."} or not _SAFE_SEGMENT.fullmatch(text):
        raise ValueError(f"invalid {field}: {value!r}")
    return text


def flat_session_dir(root: Path | str, session_id: str) -> Path | None:
    """Legacy pre-date-partition ``sessions/<session_id>/`` directory.

    Kept as a read-side fallback for stores written before the
    ``sessions/YYYY/MM/DD/<host>/<id>/`` layout. Returns ``None`` when
    *session_id* is not a safe segment, so a traversal id degrades to "no such
    session" instead of escaping the store.
    """
    try:
        return Path(root) / "sessions" / safe_segment(session_id, field="session_id")
    except ValueError:
        return None


def workspace_key(path: Path | str) -> str:
    """Human-readable workspace directory key.

    Strips the home-directory prefix and joins remaining path parts with ``-``.
    Characters outside ``[a-zA-Z0-9._-]`` are replaced with ``-``; consecutive
    dashes are collapsed.  Paths not under ``$HOME`` use the full absolute path
    (minus the leading ``/``).  Names longer than 120 chars are truncated and a
    6-char hash suffix is appended to avoid collisions.

    Examples::

        /home/alice/Projects/foo  →  Projects-foo
        /tmp/bench/bar            →  tmp-bench-bar
    """
    resolved = Path(path).expanduser().resolve()
    home = Path.home().resolve()
    try:
        rel = resolved.relative_to(home)
        parts = rel.parts
    except ValueError:
        parts = tuple(p for p in resolved.parts if p and p != "/")

    sanitized = [re.sub(r"[^a-zA-Z0-9.\-_]", "-", p) for p in parts if p]
    label = re.sub(r"-{2,}", "-", "-".join(sanitized)).strip("-")

    if len(label) > 120:
        digest = _sha256(str(resolved).encode()).hexdigest()[:6]
        label = label[:110].rstrip("-") + "--" + digest

    return label or _sha256(str(resolved).encode()).hexdigest()[:12]


DEFAULT_LESSONS_DIRNAME = ".lemoncrow/lessons"


def default_store_root() -> Path:
    """Return the default runtime store root for traces and SQLite state."""
    configured = os.environ.get("LEMONCROW_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / DEFAULT_STORE_DIRNAME).resolve()


# Ordered MOST-specific first. The rule, because getting it wrong has now caused
# the same defect under three different variable names: a value set by the host
# per *invocation* names the workspace this process actually serves; a value that
# merely happens to be in the environment was usually inherited from whatever
# shell launched the editor and names a different project entirely. Specific
# always beats inherited -- when adding a var, place it by that rule, not by
# convenience.
#
# Observed failures from getting this wrong:
#   - a window on /tmp/ide-bench/L1 served reads from /tmp/ide-bench/t3
#   - 20 benchmark workspaces all collapsed onto one scratch repo and serialised
#     on its index-write lock (41 min per prompt against a 3 min baseline)
_HOST_WORKSPACE_ENV_VARS = (
    # Cursor, per-invocation
    "CURSOR_WORKSPACE_ROOT",
    # our own pin (see pin_workspace_env) -- authoritative once set
    "LEMONCROW_WORKSPACE_ROOT",
    # Claude Code / Claude Desktop
    "CLAUDE_WORKSPACE_ROOT",
    # VS Code / generic; the editor's cwd, not a workspace -- last resort
    "VSCODE_CWD",
)

# Cursor/VS Code set this on every MCP server launch with the folders of THAT
# window -- the most specific signal available, so it is consulted before any
# of the single-value vars above.
_HOST_WORKSPACE_FOLDERS_VAR = "WORKSPACE_FOLDER_PATHS"


def pin_workspace_env(workspace: str | Path, env: MutableMapping[str, str] | None = None) -> None:
    """Make *workspace* the only answer this process can give.

    A process that serves exactly one workspace (a per-workspace daemon, a
    benchmark arm running in a copied tree) must not let an inherited value win.
    Setting one or two vars is not enough -- whichever var was NOT set still
    outranks or shadows the pin depending on lookup order, which is precisely how
    the same bug recurred under ``CLAUDE_WORKSPACE_ROOT``, ``WORKSPACE_FOLDER_PATHS``
    and ``CURSOR_WORKSPACE_ROOT``. So pin every var this lookup consults, and drop
    the multi-folder one outright: a single-workspace process has no second folder.
    """
    target = str(Path(workspace).expanduser().resolve())
    environ = os.environ if env is None else env
    for var in _HOST_WORKSPACE_ENV_VARS:
        environ[var] = target
    environ.pop(_HOST_WORKSPACE_FOLDERS_VAR, None)


def _split_folder_list(raw: str) -> list[str]:
    """Split a host folder list without guessing its separator."""
    parts = [raw]
    for sep in (os.pathsep, ","):
        parts = [chunk for part in parts for chunk in part.split(sep)]
    return [part.strip() for part in parts if part.strip()]


def host_workspace_root() -> Path | None:
    """Workspace the host says this process serves, or None when it says nothing.

    None means "fall back to detection" (git toplevel, then cwd) -- it is not an
    error. Callers must not substitute cwd for a host answer without saying so.
    """
    raw = os.environ.get(_HOST_WORKSPACE_FOLDERS_VAR, "").strip()
    if raw:
        # Multi-root windows list every folder; take the first that still exists
        # rather than trusting a separator guess.
        for candidate in _split_folder_list(raw):
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path.resolve()
    for var in _HOST_WORKSPACE_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return Path(value).expanduser().resolve()
    return None


def _git_toplevel(start: Path | None = None) -> Path | None:
    """Return the git worktree root containing ``start`` (default: cwd), or None.

    Auto-detects the project root so LemonCrow, run anywhere inside a repository,
    targets the repo root rather than a (possibly nested) working directory.
    Returns ``None`` when git is unavailable or ``start`` is not inside a repo.
    """
    import subprocess

    cwd = Path(start).expanduser().resolve() if start is not None else Path.cwd()
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    top = completed.stdout.strip()
    if not top:
        return None
    return Path(top).resolve()


def _lemoncrow_marker_root(start: Path | None = None) -> Path | None:
    """Return the nearest ancestor of *start* (default: cwd) that was explicitly
    registered via ``lc init`` (i.e. contains a ``.lemoncrow/`` dir), or None.

    Stops before reaching ``$HOME``: ``~/.lemoncrow`` is the global store
    (see ``default_store_root``), not a per-project marker, so it must never
    make an unrelated subdirectory of home resolve as "registered".
    """
    cwd = Path(start).expanduser().resolve() if start is not None else Path.cwd()
    home = Path.home().resolve()
    for candidate in (cwd, *cwd.parents):
        if candidate == home:
            break
        if (candidate / DEFAULT_STORE_DIRNAME).is_dir():
            return candidate
    return None


def resolve_workspace_root(root: Path | str | None = None) -> Path:
    """Resolve the active workspace root used for project-local lessons.

    Precedence:
    1. ``LEMONCROW_WORKSPACE_ROOT`` — explicit, authoritative
    2. Common host workspace env vars (``CLAUDE_WORKSPACE_ROOT``, etc.)
    3. Derive from the *root* path itself (e.g. parent of ``.lemoncrow``)
    4. Git repository root of the current directory (auto-detect)
    5. A directory explicitly registered via ``lc init`` (marker: ``.lemoncrow/``)
    6. Otherwise: raises ``WorkspaceNotRegisteredError``

    Raises:
        WorkspaceNotRegisteredError: cwd is neither inside a git repository nor
            an explicitly ``lc init``-registered directory.
    """
    for env_var in _HOST_WORKSPACE_ENV_VARS:
        configured = os.environ.get(env_var, "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

    derived = _derive_workspace_root(root)
    if derived is not None:
        return derived
    # Auto-detect the repository root of the current directory so LemonCrow,
    # run anywhere inside a repo, targets the repo root rather than a nested
    # subdirectory. Falls through to a directory explicitly registered via
    # `lc init`, then raises for directories that are neither.
    git_root = _git_toplevel()
    if git_root is not None:
        return git_root
    marker_root = _lemoncrow_marker_root()
    if marker_root is not None:
        return marker_root
    raise WorkspaceNotRegisteredError(
        "This directory is not a git repository and has not been registered with "
        "LemonCrow. Run `lc init` here to register it, or run this command from "
        "inside a git repository."
    )


def is_recognized_workspace(path: Path | str | None = None) -> bool:
    """True when *path* (default: cwd) is a git repository or has been
    explicitly registered via ``lc init`` (marker: ``.lemoncrow/``).

    Mirrors the same two checks :func:`resolve_workspace_root` uses for its
    cwd-fallback precedence (steps 4-5) -- exposed standalone so callers that
    only want a yes/no answer (no env-var override, no raise) don't have to
    catch :class:`WorkspaceNotRegisteredError` or risk honoring an env var
    they didn't ask about.
    """
    start = Path(path).expanduser().resolve() if path is not None else None
    return _git_toplevel(start) is not None or _lemoncrow_marker_root(start) is not None


def resolve_lessons_root(root: Path | str | None = None, lessons_root: Path | str | None = None) -> Path:
    """Resolve the Git-tracked lessons root.

    Precedence:
    1. Explicit constructor argument
    2. LEMONCROW_LESSONS_ROOT
    3. <workspace>/.lemoncrow/lessons
    """
    if lessons_root is not None:
        return Path(lessons_root).expanduser().resolve()

    configured = os.environ.get("LEMONCROW_LESSONS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    return (resolve_workspace_root(root) / DEFAULT_LESSONS_DIRNAME).resolve()


def detect_host() -> str:
    """Derive the host/agent label from the runtime environment.

    The single canonical implementation -- every per-session storage path is
    segregated by this value so two hosts (claude/codex/copilot/opencode/...)
    can never collide on the same session_id: each host's own session-id
    namespace is independent, so string equality across hosts is coincidence,
    not identity. Previously re-derived independently in mcp_server.py and
    reimplemented (badly -- via a same-session-id cross-check) in the
    copilot-cli hooks; this is now the only place that sniffs these env vars.

    Checks, in order:
    1. LEMONCROW_AGENT env var (explicit override -- any host can set this)
    2. CLAUDE_CODE -> "claude"
    3. ANTIGRAVITY_SESSION_ID or AGY_SESSION_ID -> "antigravity"
    4. CODEX_SESSION_ID -> "codex"
    5. LEMONCODE_SESSION_ID or LEMONCODE_CLI -> "lemoncode"
    6. OPENCODE_SESSION_ID -> "opencode"
    7. CURSOR_SESSION_ID or CURSOR_TRACE_ID -> "cursor"
    8. HERMES_* -> "hermes"
    9. COPILOT_CLI or GITHUB_COPILOT_SESSION_ID -> "copilot"
    10. Falls back to "claude" (the MCP wrapper ships with the Claude plugin)

    LemonCode is checked before OpenCode: it is a fork that keeps upstream's
    on-disk layout, so a LemonCode session must not be mistaken for one.
    """
    explicit = os.environ.get("LEMONCROW_AGENT", "").strip()
    if explicit:
        return explicit
    if os.environ.get("CLAUDE_CODE"):
        return "claude"
    if (
        os.environ.get("ANTIGRAVITY_SESSION_ID")
        or os.environ.get("AGY_SESSION_ID")
        or os.environ.get("ANTIGRAVITY_CLI")
        or os.environ.get("AGY_CLI")
    ):
        return "antigravity"
    if os.environ.get("CODEX_SESSION_ID") or os.environ.get("CODEX_CLI"):
        return "codex"
    if os.environ.get("LEMONCODE_SESSION_ID") or os.environ.get("LEMONCODE_CLI"):
        return "lemoncode"
    if os.environ.get("OPENCODE_SESSION_ID") or os.environ.get("OPENCODE_CLI"):
        return "opencode"
    if os.environ.get("CURSOR_SESSION_ID") or os.environ.get("CURSOR_TRACE_ID"):
        return "cursor"
    if os.environ.get("HERMES_HOME") or os.environ.get("HERMES_SESSION_ID") or os.environ.get("HERMES_CLI"):
        return "hermes"
    if os.environ.get("COPILOT_CLI") or os.environ.get("GITHUB_COPILOT_SESSION_ID"):
        return "copilot"
    return "claude"


def session_dir(root: Path | str, host: str, session_id: str, *, search_days: int = 3) -> Path:
    """Canonical per-session directory: ``sessions/YYYY/MM/DD/<host>/<session_id>/``.

    Every per-session artifact (run.json, stats.json, events.jsonl,
    mcp_debug.jsonl, runtime_state.json, statusline_segment, savings.jsonl,
    outcomes.json, ...) belongs in this ONE directory -- the single write
    surface for a session, replacing the flat ``sessions/<id>/`` layout most
    writers used, the date-partitioned ``sessions/YYYY/MM/DD/<id>/`` layout
    RunLedger's own persist/load used, and every ad hoc reimplementation of
    either. Host-segregated so two hosts can never collide on the same id
    (see :func:`detect_host`).

    The date is resolved, not blindly recomputed as "today": a session that
    already has a directory within the last *search_days* days keeps it, so a
    session spanning midnight does not silently start writing into a new
    day's folder and orphan its own earlier state. Only a session with no
    existing directory in that window mints today's date (first writer for a
    session pins its folder for every later caller, in-process or not).

    For read-side lookups where the host is not known up front (e.g. a CLI
    command given a bare session id), use :func:`find_session_dir` instead.
    """
    root = Path(root)
    host = safe_segment(host, field="host")
    session_id = safe_segment(session_id, field="session_id")
    today = _date.today()
    for offset in range(search_days):
        d = today - _timedelta(days=offset)
        candidate = root / "sessions" / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d") / host / session_id
        if candidate.exists():
            return candidate
    return root / "sessions" / today.strftime("%Y") / today.strftime("%m") / today.strftime("%d") / host / session_id


def find_session_dir(root: Path | str, session_id: str) -> Path | None:
    """Locate an existing session directory by id alone, host unknown.

    Globs ``sessions/*/*/*/*/`` <session_id> under *root*. session_id is a
    high-entropy id (a host-issued UUID in practice), so a glob match is safe
    and this stays a read-only lookup -- callers that are writing must go
    through :func:`session_dir` with an explicit host instead. Returns the
    first match (there should only ever be one) or ``None``.
    """
    try:
        session_id = safe_segment(session_id, field="session_id")
    except ValueError:
        return None
    root = Path(root)
    sessions_root = root / "sessions"
    if not sessions_root.is_dir():
        return None
    matches = sorted(sessions_root.glob(f"*/*/*/*/{session_id}"))
    return matches[0] if matches else None


def resolve_session_state_path(workspace_root: Path | str | None = None) -> Path:
    """Resolve the path for session-specific state (failures, current run ID).

    Lives under the project's own runtime cache dir (see
    :func:`resolve_workspace_store_dir`) so it travels with the project
    instead of a global store keyed by a path hash.
    """
    return resolve_workspace_store_dir(workspace_root=workspace_root or None) / "session_state.json"


def _derive_workspace_root(root: Path | str | None) -> Path | None:
    if root is None:
        return None

    candidate = Path(root).expanduser().resolve()
    default_home_store = (Path.home() / DEFAULT_STORE_DIRNAME).resolve()
    if candidate == default_home_store:
        return None

    # Do not treat the workspace hash subfolder as a project root
    if "workspaces" in candidate.parts:
        return None

    # .lemoncrow/lessons is two levels deep — peel both parts to reach workspace
    if candidate.name == "lessons" and candidate.parent.name == DEFAULT_STORE_DIRNAME:
        return candidate.parent.parent
    if candidate.name == DEFAULT_STORE_DIRNAME:
        return candidate.parent
    if candidate.parent != candidate:
        return candidate.parent
    return candidate


def confine_to_root(candidate: str | Path, root: str | Path) -> Path:
    """Resolve *candidate* and ensure it stays within *root*.

    Both paths are ``expanduser()``-ed and ``resolve()``-d, which means symlinks
    are followed; a symlink that points outside *root* therefore resolves to an
    out-of-root target and is rejected. The resolved candidate is returned only
    when it is *root* itself or lives beneath it.

    Raises:
        ValueError: if the resolved candidate escapes *root*.
    """
    resolved_root = Path(root).expanduser().resolve()
    resolved_candidate = Path(candidate).expanduser().resolve()
    if resolved_candidate != resolved_root and not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError("path escapes the allowed root")
    return resolved_candidate


def ensure_gitignore(project_root: Path) -> list[str]:
    """Create/update ``.lemoncrow/.gitignore`` to ignore everything inside ``.lemoncrow/``.

    Keeps the ``.lemoncrow/`` directory visible in git for brand awareness while
    preventing cache files, binaries, and other project-local runtime data from
    being committed.  Idempotent: returns a non-empty list on first run (entries
    added) and an empty list on subsequent runs (already correct).
    """
    lemoncrow_dir = project_root / ".lemoncrow"
    lemoncrow_dir.mkdir(parents=True, exist_ok=True)
    gitignore_path = lemoncrow_dir / ".gitignore"
    content = "# LemonCrow runtime data \u2014 keep the directory, ignore its contents\n*\n"
    if gitignore_path.exists() and gitignore_path.read_text("utf-8") == content:
        return []
    gitignore_path.write_text(content, encoding="utf-8")
    return ["*"]


_ensure_gitignore = ensure_gitignore  # compat alias for internal use


def resolve_workspace_store_dir(root: Path | str | None = None, workspace_root: Path | str | None = None) -> Path:
    """Single source of truth for the project-local runtime cache dir.

    Returns ``<workspace_root>/.lemoncrow/workspace/``. Per-project runtime
    state -- the code index, embeddings, session state, blocks/rubrics
    mirrors, and every other rebuildable per-project cache -- lives inside
    the project itself instead of a global store keyed by a path hash.
    Deleting the project (or its ``.lemoncrow/`` dir) cleans everything;
    nothing orphans in the global store.

    When *workspace_root* is given, it is used verbatim -- no git/marker
    resolution, no raising -- because the caller already knows the project
    root (an indexer's ``repo_root`` argument, an env-derived workspace
    path, ...). Otherwise the active workspace is auto-detected via
    :func:`resolve_workspace_root`, using *root* as a legacy hint (may raise
    ``WorkspaceNotRegisteredError`` when cwd is outside any git repo or
    registered workspace).
    """
    ws = Path(workspace_root).expanduser().resolve() if workspace_root is not None else resolve_workspace_root(root)
    return ws / DEFAULT_STORE_DIRNAME / "workspace"


def resolve_store_root_for_workspace(workspace_root: Path | str | None = None) -> Path:
    """Return the per-workspace runtime cache dir, falling back to the global store.

    When a workspace root is known this returns
    ``<workspace_root>/.lemoncrow/workspace/`` (see
    :func:`resolve_workspace_store_dir`) so that sessions and raw artifacts
    live alongside the code index for the same project.  When the workspace
    root cannot be determined the global store root is returned so callers
    never crash.

    Precedence for workspace discovery (when *workspace_root* is not given):
    1. ``LEMONCROW_WORKSPACE_ROOT``
    2. Common host env vars (``CLAUDE_WORKSPACE_ROOT``, etc.)
    3. Current working directory — only when it is a git repository or has been
       explicitly registered via ``lc init`` (marker: ``.lemoncrow/``); otherwise
       falls back to the global store root (this function never raises).
    """
    if workspace_root is None:
        for env_var in _HOST_WORKSPACE_ENV_VARS:
            configured = os.environ.get(env_var, "").strip()
            if configured:
                workspace_root = Path(configured)
                break
        else:
            cwd = Path.cwd()
            if _git_toplevel(cwd) is not None or _lemoncrow_marker_root(cwd) is not None:
                workspace_root = cwd

    if workspace_root is not None:
        return resolve_workspace_store_dir(workspace_root=workspace_root)
    return default_store_root()


__all__ = [
    "DEFAULT_LESSONS_DIRNAME",
    "DEFAULT_STORE_DIRNAME",
    "WorkspaceNotRegisteredError",
    "confine_to_root",
    "default_store_root",
    "detect_host",
    "find_session_dir",
    "flat_session_dir",
    "is_recognized_workspace",
    "resolve_lessons_root",
    "resolve_session_state_path",
    "resolve_store_root_for_workspace",
    "resolve_workspace_root",
    "resolve_workspace_store_dir",
    "safe_segment",
    "session_dir",
    "workspace_key",
]
