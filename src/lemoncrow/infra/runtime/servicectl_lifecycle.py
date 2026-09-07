"""Servicectl background-loop lifecycle (Phase 25-03, QBL-CLI-03).

PID/state files, status payloads, host-status refresh, host-session import,
external-analytics collection, git auto-update, and the periodic tick loop for
the ``lc servicectl`` daemon. Moved verbatim from ``gateway/cli/app.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lemoncrow.infra.runtime.daemon_units import (
    LAUNCHD_USER_DIR,
    STACK_LABEL,
    STACK_UNIT,
    _is_macos,
)

logger = logging.getLogger(__name__)

# Prune stale workspace indexes once a day so they cannot silently pile up.
_WORKSPACE_PRUNE_INTERVAL_SECONDS = 86_400
_WORKSPACE_PRUNE_MAX_AGE_DAYS = 10


def _servicectl_dir(root: Path) -> Path:
    return Path(root) / "servicectl"


def _servicectl_pid_path(root: Path) -> Path:
    return _servicectl_dir(root) / "servicectl.pid"


def _servicectl_log_path(root: Path) -> Path:
    return _servicectl_dir(root) / "servicectl.log"


def _servicectl_state_path(root: Path) -> Path:
    return _servicectl_dir(root) / "state.json"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_servicectl_state(root: Path) -> dict[str, Any]:
    path = _servicectl_state_path(root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_servicectl_state(root: Path, payload: dict[str, Any]) -> None:
    state_path = _servicectl_state_path(root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic tmp + rename (run_ledger.persist pattern): a crash mid-write must
    # never leave a truncated state.json behind.
    tmp = state_path.with_name(f"{state_path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, state_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _read_servicectl_pid(root: Path) -> int | None:
    path = _servicectl_pid_path(root)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _clear_servicectl_pid(root: Path) -> None:
    path = _servicectl_pid_path(root)
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _kill_orphan_servicectl_processes(current_root: Path) -> None:
    """Kill stale/duplicate ``servicectl run`` daemons of THIS root only.

    A daemon whose cmdline names a different ``--root`` belongs to another
    install and is never touched. Duplicates for the current root (left
    behind by a crashed supervisor) are terminated so exactly one daemon
    serves the store; the pid recorded in this root's pidfile is reaped too.
    The /proc cmdline scan only runs on Linux.
    """
    import glob as _glob

    my_pid = os.getpid()
    pidfile_pid = _read_servicectl_pid(current_root)
    if pidfile_pid is not None and pidfile_pid != my_pid and _pid_is_running(pidfile_pid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pidfile_pid, signal.SIGTERM)

    current_root_str = str(current_root.resolve())
    for cmdline_file in _glob.glob("/proc/*/cmdline"):
        try:
            pid = int(cmdline_file.split("/")[2])
        except (ValueError, IndexError):
            continue
        if pid == my_pid or pid == pidfile_pid:
            continue
        try:
            raw = Path(cmdline_file).read_bytes()
        except OSError:
            continue
        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        # Compare the parsed ``--root`` value for exact equality: a bare
        # substring test would let ``~/.lemoncrow`` match (and kill) the daemon
        # of ``~/.lemoncrow-work`` (and vice versa).
        tokens = cmdline.split()
        root_value: str | None = None
        for i, tok in enumerate(tokens):
            if tok == "--root" and i + 1 < len(tokens):
                root_value = tokens[i + 1]
                break
            if tok.startswith("--root="):
                root_value = tok[len("--root=") :]
                break
        if (
            "lemoncrow.gateway.cli" in cmdline
            and "servicectl" in cmdline
            and " run " in cmdline
            and root_value == current_root_str
        ):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGTERM)


def _servicectl_status_payload(root: Path) -> dict[str, Any]:
    state = _read_servicectl_state(root)
    pid = _read_servicectl_pid(root)
    running = bool(pid is not None and _pid_is_running(pid))
    from lemoncrow.infra.storage.factory import create_store

    store = create_store(root)
    store.init()
    job_queue_health = store.jobs.job_queue_health()
    return {
        "running": running,
        "pid": pid,
        "pid_file": str(_servicectl_pid_path(root)),
        "log_file": str(_servicectl_log_path(root)),
        "state_file": str(_servicectl_state_path(root)),
        "last_tick_at": state.get("last_tick_at"),
        # Surfaced so a stale lemoncrow.com/savings is diagnosable without
        # reading state.json: no timestamp here means nothing has pushed.
        "last_usage_report_at": state.get("last_usage_report_at"),
        "last_processed_jobs": state.get("last_processed_jobs", []),
        "last_enqueued_jobs": state.get("last_enqueued_jobs", []),
        "last_imported_sessions": state.get("last_imported_sessions", {}),
        "last_session_import_at": state.get("last_session_import_at"),
        "last_exit_reason": state.get("last_exit_reason"),
        "started_at": state.get("started_at"),
        "subprocess_timeouts": state.get("subprocess_timeouts", {}),
        "job_queue_health": job_queue_health,
    }


def _servicectl_refresh_host_status(root: Path) -> dict[str, str]:
    """Detect host agent CLI tools and persist status for the Docker service.

    Writes to ``{root}/hosts/status.json`` in the same format as
    ``scripts/status.sh --write`` so the API running in Docker can
    consume it via ``_load_host_status_file()``.

    Also writes to the CWD's ``.lemoncrow/hosts/status.json`` if different
    from *root* (handles the common case where Docker mounts the project's
    ``.lemoncrow`` while servicectl uses ``~/.lemoncrow``).

    Runs on the host (inside servicectl) so ``shutil.which()`` can
    find the actual CLI binaries.
    """
    import shutil

    hosts = [
        ("claude", "claude"),
        ("codex", "codex"),
        ("opencode", None),
        ("lemoncode", "lemoncode"),
        ("copilot", None),
        ("antigravity", "agy"),
    ]
    status: dict[str, str] = {}
    for hid, check in hosts:
        if check:
            installed = shutil.which(check) is not None
        elif hid == "opencode":
            installed = shutil.which("opencode") is not None
        elif hid == "copilot":
            installed = shutil.which("code") is not None
        elif hid == "antigravity":
            installed = shutil.which("agy") is not None or shutil.which("antigravity") is not None
        else:
            installed = False
        status[hid] = "installed" if installed else "not_installed"

    def _write_to(hosts_dir: Path) -> None:
        hosts_dir.mkdir(parents=True, exist_ok=True)
        (hosts_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    # Primary: write to servicectl's root
    root_hosts = Path(root) / "hosts"
    _write_to(root_hosts)

    # Secondary: also write to the CWD's ``.lemoncrow/hosts`` when it is a
    # different store dir (Docker mounts the project's ``.lemoncrow`` while
    # servicectl runs against ``~/.lemoncrow``), so the containerized API can
    # read host-tool status too. Best-effort: a read-only CWD must not defeat
    # the primary write above.
    cwd_hosts = Path.cwd() / ".lemoncrow" / "hosts"
    if cwd_hosts.resolve() != root_hosts.resolve():
        with contextlib.suppress(OSError):
            _write_to(cwd_hosts)

    return status


_SESSION_IMPORT_TIMEOUT_SECONDS = 300
_RECALL_INDEX_TIMEOUT_SECONDS = 300
_WORKSPACE_PRUNE_TIMEOUT_SECONDS = 600
# After this many CONSECUTIVE timeouts of the same tick subprocess, its
# periodic key advances anyway (deferring the retry to the next interval) so
# one pathological store cannot starve every other periodic duty.
_TICK_TIMEOUT_BACKOFF_AFTER = 3


def _run_tick_subprocess(
    cmd: list[str], *, timeout: int, what: str
) -> tuple[subprocess.CompletedProcess[bytes] | None, bool]:
    """``subprocess.run`` guarded against timeout/launch failures.

    A tick subprocess (import, recall index, workspace prune) can legitimately
    run long under load. Letting ``TimeoutExpired`` propagate would abort
    ``_servicectl_tick`` before its periodic-job timestamp and state file are
    written. Returns ``(result, timed_out)``: ``(None, True)`` on timeout so
    the tick can keep the periodic key un-advanced and retry with an
    escalated budget, ``(None, False)`` on launch failure.
    """
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout), False
    except subprocess.TimeoutExpired:
        logging.warning("%s subprocess timed out after %ds", what, timeout)
        return None, True
    except OSError:
        logging.exception("failed to launch %s subprocess", what)
        return None, False


def _servicectl_import_sessions(root: Path, *, timeout: int = _SESSION_IMPORT_TIMEOUT_SECONDS) -> dict[str, int] | None:
    """Import host sessions by delegating to the ``lc import`` CLI subprocess.

    Running import out-of-process keeps JSON parsing, importer-level dedup, and
    the ``sync_usage`` upload out of the daemon's heap. ``sync_usage`` is called
    inside the subprocess, so there is no double-upload.

    Returns ``None`` when the subprocess timed out (caller keeps the periodic
    key un-advanced and retries with a bigger budget), ``{}`` on failure.
    """
    result, timed_out = _run_tick_subprocess(
        [sys.executable, "-m", "lemoncrow.gateway.cli", "--root", str(root), "import", "--json"],
        timeout=timeout,
        what="session import",
    )
    if timed_out:
        return None
    if result is None:
        return {}
    if result.returncode != 0:
        logging.warning(
            "session import subprocess failed (rc=%d): %s",
            result.returncode,
            result.stderr[-500:].decode("utf-8", errors="replace").strip(),
        )
        return {}
    try:
        data = json.loads(result.stdout)
        return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        logging.exception("failed to parse session import JSON output")
        return {}


def _servicectl_index_recall(root: Path, *, timeout: int = _RECALL_INDEX_TIMEOUT_SECONDS) -> dict[str, int] | None:
    """Index recent session transcripts via the ``lc session recall index`` CLI subprocess.

    Keeps embedding work and SQLite writes out of the daemon's heap. The subprocess
    is incremental by default (unchanged sessions are skipped).

    Returns ``None`` on subprocess timeout, ``{}`` on failure.
    """
    result, timed_out = _run_tick_subprocess(
        [sys.executable, "-m", "lemoncrow.gateway.cli", "--root", str(root), "session", "recall", "index", "--json"],
        timeout=timeout,
        what="recall index",
    )
    if timed_out:
        return None
    if result is None:
        return {}
    if result.returncode != 0:
        logging.warning(
            "recall index subprocess failed (rc=%d): %s",
            result.returncode,
            result.stderr[-500:].decode("utf-8", errors="replace").strip(),
        )
        return {}
    try:
        data = json.loads(result.stdout)
        return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        logging.exception("failed to parse recall index JSON output")
        return {}


def _servicectl_prune_workspaces(
    root: Path,
    *,
    max_age_days: int = _WORKSPACE_PRUNE_MAX_AGE_DAYS,
    timeout: int = _WORKSPACE_PRUNE_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Remove orphaned / stale workspace indexes via the ``code prune`` CLI subprocess.

    Runs once a day.  Removes orphaned indexes (no ``session_state.json``),
    ``/tmp`` benchmark runs, indexes whose source repo is gone, and — via
    ``--max-age-days`` — indexes inactive for more than ``max_age_days`` days.
    Runs out-of-process to keep the rmtree/walk work off the daemon heap,
    matching the import/recall pattern.

    Returns ``None`` on subprocess timeout, ``{}`` on failure.
    """
    result, timed_out = _run_tick_subprocess(
        [
            sys.executable,
            "-m",
            "lemoncrow.gateway.cli",
            "--root",
            str(root),
            "code",
            "prune",
            "--store-root",
            str(root),
            "--max-age-days",
            str(max_age_days),
            "--json",
        ],
        timeout=timeout,
        what="workspace prune",
    )
    if timed_out:
        return None
    if result is None:
        return {}
    if result.returncode != 0:
        logging.warning(
            "workspace prune subprocess failed (rc=%d): %s",
            result.returncode,
            result.stderr[-500:].decode("utf-8", errors="replace").strip(),
        )
        return {}
    try:
        data = json.loads(result.stdout)
        return data if isinstance(data, dict) else {}
    except Exception:
        logging.exception("failed to parse workspace prune JSON output")
        return {}


def _lemoncrow_version() -> str:
    """Return the installed LemonCrow version string."""
    from lemoncrow.core.foundation.update_state import installed_cli_version

    try:
        return installed_cli_version() or "0.0.0"
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return "0.0.0"


def _git_project_root() -> Path | None:
    """Resolve the git project root from install record or file-path traversal."""
    record_path = Path.home() / ".lemoncrow" / "install_dir"
    if record_path.exists():
        candidate = Path(record_path.read_text(encoding="utf-8").strip())
        # Bare `.git` existence misfires on ~/.lemoncrow/install when an older
        # install left a stray .git there: auto-update would then run git in a
        # non-repository forever instead of taking the release path.
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").exists():
            try:
                if 'name = "lemoncrow"' in (candidate / "pyproject.toml").read_text(encoding="utf-8"):
                    return candidate.resolve()
            except OSError:
                pass
            return candidate.resolve()
    # Fallback: traverse up from this file
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / ".git").exists() and (parent / "pyproject.toml").exists():
            try:
                content = (parent / "pyproject.toml").read_text("utf-8")
                if 'name = "lemoncrow"' in content:
                    return parent
            except OSError:
                pass
    return None


# Distribution channel -- keep in lockstep with scripts/install.sh and
# src/lemoncrow/gateway/cli/commands/update.py.
_GH_REPO = "lemoncrow-lab/lemoncrow"
_RELEASE_LATEST_URL = f"https://github.com/{_GH_REPO}/releases/latest/download"


def _github_latest_version() -> str | None:
    """Fetch the latest release tag from GitHub Releases (e.g. "0.3.5")."""
    import urllib.request

    try:
        req = urllib.request.Request(
            f"https://api.github.com/repos/{_GH_REPO}/releases/latest",
            headers={"Accept": "application/vnd.github.v3+json", "User-Agent": "lemoncrow-update/1.0"},
        )
        resp = urllib.request.urlopen(req, timeout=10)  # nosec - pinned GitHub API URL
        data = json.loads(resp.read().decode())
        tag = data.get("tag_name", "")
        return tag.lstrip("v") or None
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return None


def _version_key(version: str) -> tuple[int, ...]:
    """Dotted version -> comparable int tuple (non-numeric chunks count as 0)."""
    import re

    parts: list[int] = []
    for chunk in version.split("."):
        match = re.match(r"\d+", chunk)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def _is_dev_install(root: Path) -> bool:
    """True for a ``make dev`` (source-checkout) install.

    ``scripts/local.sh`` writes ``<root>/.dev_mode``; production installers never
    do. Dev installs must NEVER auto-update: a git fetch/merge on the live
    checkout fights an in-progress refactor and can clobber uncommitted work.
    """
    try:
        return (root / ".dev_mode").exists()
    except OSError:
        return False


def _detect_auto_update_method() -> tuple[str, str | None]:
    """Detect the install method for auto-update.

    Returns ("git", project_root) for a source checkout, or ("release", None)
    for an end-user install, which updates through the GitHub release installer.
    """
    git_root = _git_project_root()
    if git_root is not None:
        return ("git", str(git_root))
    return ("release", None)


# Auto-update always tracks this remote branch, regardless of which local
# branch is currently checked out. Hardcoded to origin/main by request.
_AUTO_UPDATE_REMOTE = "origin"
_AUTO_UPDATE_BRANCH = "main"


def _update_via_git(project_root: str) -> bool:
    """Update from git: fetch origin/main, fast-forward, sync deps.

    Auto-update always tracks ``origin/main`` regardless of the currently
    checked-out local branch. Returns True only if an update was applied.
    """
    project_root_p = Path(project_root)
    remote_ref = f"{_AUTO_UPDATE_REMOTE}/{_AUTO_UPDATE_BRANCH}"

    subprocess.run(
        ["git", "fetch", "--quiet", _AUTO_UPDATE_REMOTE, _AUTO_UPDATE_BRANCH],
        cwd=project_root_p,
        check=True,
    )

    # Bail out cleanly if the tracking ref is missing (e.g. the remote has no
    # ``main``) instead of raising -- keeps the controller quiet on odd setups.
    verify = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", remote_ref],
        cwd=project_root_p,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        logger.info(f"Auto-update: {remote_ref} not found; skipping git update.")
        return False

    show = subprocess.run(
        ["git", "show", f"{remote_ref}:pyproject.toml"],
        cwd=project_root_p,
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        logger.info(f"Auto-update: could not read {remote_ref}:pyproject.toml; skipping git update.")
        return False

    import re

    match = re.search(r'^version\s*=\s*"([^"]+)"', show.stdout, re.MULTILINE)
    if not match:
        logger.info(f"Auto-update: could not parse version from {remote_ref}:pyproject.toml; skipping git update.")
        return False

    remote_version = match.group(1)
    current_version = _lemoncrow_version()
    if _version_key(remote_version) <= _version_key(current_version):
        logger.info(
            f"Auto-update: remote version {remote_version} is not newer than "
            f"current version {current_version}; skipping git update."
        )
        return False

    res = subprocess.run(
        ["git", "rev-list", f"HEAD..{remote_ref}", "--count"],
        cwd=project_root_p,
        capture_output=True,
        text=True,
        check=True,
    )
    behind_count = int(res.stdout.strip())
    if behind_count == 0:
        return False

    logger.info(f"Auto-update: detected {behind_count} new commits on {remote_ref}. Updating...")

    # Fast-forward only: never clobber local commits. If the checked-out branch
    # has diverged from main it cannot fast-forward -- log and skip rather than
    # raising, so the controller keeps running without error spam.
    merge = subprocess.run(
        ["git", "merge", "--ff-only", "--quiet", remote_ref],
        cwd=project_root_p,
        capture_output=True,
        text=True,
    )
    if merge.returncode != 0:
        logger.warning(
            f"Auto-update: cannot fast-forward to {remote_ref} "
            f"(local branch has diverged); skipping. {merge.stderr.strip()}"
        )
        return False

    if (project_root_p / "uv.lock").exists() or (project_root_p / "pyproject.toml").exists():
        import shutil

        if shutil.which("uv"):
            logger.info("Auto-update: syncing dependencies with uv...")
            subprocess.run(["uv", "sync"], cwd=project_root_p, check=True)
    return True


def _update_via_release() -> bool:
    """Launch a detached installer to reinstall from the latest GitHub release.

    The daemon cannot reinstall itself inline: ``install.sh`` stops running
    LemonCrow processes (this daemon included). So download the published
    ``install.sh`` and run it in a fully detached session -- it outlives this
    process, reinstalls the uv tool from ``lemoncrow-distribution-*.tar.gz``, and
    its own ``run_setup`` restarts the stack on the new code.

    Returns True if an installer was launched (a newer release exists and the
    download succeeded), else False.
    """
    import shutil
    import tempfile
    import urllib.request

    # Opt-in: this path downloads and executes an installer script from the
    # release channel, so it stays OFF unless LEMONCROW_AUTO_UPDATE_RELEASE is
    # explicitly enabled (1/true/yes/on). Absence disables it.
    if os.environ.get("LEMONCROW_AUTO_UPDATE_RELEASE", "").strip().lower() not in ("1", "true", "yes", "on"):
        logger.info(
            "Auto-update: release auto-update is disabled (opt in with LEMONCROW_AUTO_UPDATE_RELEASE=1). "
            "Run 'lc update' manually to update."
        )
        return False
    if not shutil.which("bash"):
        logger.error("Auto-update: bash unavailable; cannot apply release update")
        return False

    latest = _github_latest_version()
    if latest is None:
        logger.error("Auto-update: could not determine latest release version")
        return False
    current = _lemoncrow_version()
    if _version_key(latest) <= _version_key(current):
        return False

    installer_url = f"{_RELEASE_LATEST_URL}/install.sh"
    try:
        fd, tmp_path = tempfile.mkstemp(suffix="-lemoncrow-install.sh")
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(installer_url, timeout=30) as resp:  # nosec
            shutil.copyfileobj(resp, fh)
    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        logger.error(f"Auto-update: failed to download installer ({installer_url}): {exc}")
        return False

    logger.info(f"Auto-update: launching detached installer ({current} -> {latest})")
    # Fully detached: new session so the installer survives this daemon being
    # stopped by the installer's own process-cleanup, plus its later restart.
    # The wrapper deletes the downloaded script once the installer finishes so it
    # does not accumulate in the temp dir across auto-update cycles.
    subprocess.Popen(
        ["bash", "-c", 'bash "$0"; rm -f "$0"', tmp_path],
        env={**os.environ, "LEMONCROW_NON_INTERACTIVE": "1"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return True


def _stack_restart() -> None:
    """Trigger a restart of managed services (systemd or launchd)."""
    if os.environ.get("INVOCATION_ID"):
        logger.info("Auto-update: triggering systemd stack restart...")
        subprocess.run(["systemctl", "--user", "restart", STACK_UNIT], check=False)
    elif _is_macos() and (LAUNCHD_USER_DIR / f"{STACK_LABEL}.plist").exists():
        logger.info("Auto-update: triggering launchd stack restart...")
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{STACK_LABEL}"], check=False)


def _servicectl_check_and_apply_updates(root: Path) -> bool:
    """Check for updates and apply them if available.

    Two install topologies, two behaviours:

    - **git** -- pull + ``uv sync`` inline, write update-state, restart the stack,
      and return True so the caller exits for an immediate restart on new code.
    - **release** -- launch a *detached* installer (see ``_update_via_release``)
      and return False. The installer owns the reinstall and stack restart, so the
      caller must NOT exit here; returning False lets the tick record its check
      timestamp, preventing a relaunch on the next tick before the installer lands.

    Returns True only when the caller should exit for an immediate restart.
    """
    previous_version = _lemoncrow_version()
    method, project_root = _detect_auto_update_method()
    logger.info(f"Auto-update: install method={method}, current version={previous_version}")

    try:
        if method == "git" and project_root:
            if not _update_via_git(project_root):
                logger.info("Auto-update: already up-to-date.")
                return False

            # The daemon's in-process version is unchanged after a git pull;
            # query the installed CLI after sync for the notification state.
            try:
                from lemoncrow.core.foundation.update_state import write_update_state

                write_update_state(
                    previous_version=previous_version,
                    current_version=_lemoncrow_version(),
                    method=method,
                    root=root,
                )
            except Exception as exc:
                logging.exception("Recovered from broad exception handler")
                logger.warning(f"Auto-update: failed to write update state: {exc}")

            _stack_restart()
            logger.info("Auto-update: update applied successfully. Exiting for restart.")
            return True

        # release install: the detached installer reinstalls and restarts the
        # stack itself. Never exit the daemon here -- the installer stops it when
        # ready, and returning False records the check timestamp so we don't
        # launch a second installer on the next tick.
        if _update_via_release():
            logger.info("Auto-update: detached installer launched; it will restart the stack.")
        else:
            logger.info("Auto-update: already up-to-date.")
        return False

    except Exception as exc:
        logging.exception("Recovered from broad exception handler")
        logger.error(f"Auto-update failed: {exc}")
        return False


def _servicectl_tick(
    root: Path,
    *,
    maintenance_interval_seconds: int,
    session_import_interval_seconds: int,
    auto_update: bool = False,
    auto_update_interval_seconds: int = 3600,
) -> dict[str, Any]:
    from lemoncrow.core.service.jobs import JOB_CONSOLIDATE_BLOCKS, JOB_OPTIMIZE, JOB_RETENTION_CLEANUP
    from lemoncrow.infra.storage.factory import create_store
    from lemoncrow.pro.capabilities.optimization import load_automation_config

    SESSION_IMPORT_KEY = "import_host_sessions"

    store = create_store(root)
    store.init()

    # Refresh host agent detection status for the Docker service
    with suppress(Exception):
        _servicectl_refresh_host_status(root)

    # Reclaim registration files for singleton MCP daemons that crashed without
    # cleanup (a clean exit removes its own). Cheap glob; fail-open.
    with suppress(Exception):
        from lemoncrow.gateway.adapters.mcp_daemon import prune_stale_daemons

        prune_stale_daemons(root)

    # Same for per-session registrations: an agent host that leaks its `lc mcp`
    # child leaves a zombie whose registration file otherwise lingers for as
    # long as that host runs. Cheap glob; fail-open.
    with suppress(Exception):
        from lemoncrow.gateway.adapters.mcp.session_state import prune_stale_mcp_sessions

        prune_stale_mcp_sessions(root)

    now = datetime.now(UTC)
    state = _read_servicectl_state(root)
    periodic = state.setdefault("periodic_jobs", {})

    # Consecutive-timeout health per tick subprocess. A timed-out subprocess
    # does NOT advance its periodic key (its work never happened); the next
    # attempt runs with an escalated budget (2x per consecutive timeout,
    # capped at 4x the base). After _TICK_TIMEOUT_BACKOFF_AFTER consecutive
    # timeouts the key advances anyway so one pathological store cannot
    # starve the other periodic duties; the counter stays in state.json as a
    # health signal (surfaced by ``servicectl status``).
    timeouts_raw = state.get("subprocess_timeouts")
    subprocess_timeouts: dict[str, int] = (
        {str(k): int(v) for k, v in timeouts_raw.items() if isinstance(v, (int, float))}
        if isinstance(timeouts_raw, dict)
        else {}
    )

    def _scaled_timeout(key: str, base: int) -> int:
        scale: int = min(4, 1 << subprocess_timeouts.get(key, 0))
        return base * scale

    def _note_subprocess_outcome(key: str, timed_out: bool) -> bool:
        """Track timeout health for *key*; return whether to advance its periodic key."""
        if not timed_out:
            subprocess_timeouts.pop(key, None)
            return True
        subprocess_timeouts[key] = subprocess_timeouts.get(key, 0) + 1
        return subprocess_timeouts[key] >= _TICK_TIMEOUT_BACKOFF_AFTER

    # 0. Check for auto-updates (never on a dev/source checkout -- a git op on the
    #    live tree would clobber uncommitted work mid-refactor).
    if auto_update and not _is_dev_install(root):
        AUTO_UPDATE_KEY = "auto_update_check"
        last_update_raw = periodic.get(AUTO_UPDATE_KEY)
        last_update_at: datetime | None = None
        if isinstance(last_update_raw, str):
            try:
                last_update_at = datetime.fromisoformat(last_update_raw)
            except ValueError:
                last_update_at = None

        if last_update_at is None or (now - last_update_at).total_seconds() >= auto_update_interval_seconds:
            if _servicectl_check_and_apply_updates(root):
                # Exit 3 = restart-needed signal.  When running as a tick subprocess
                # the parent ``servicectl run`` loop detects code 3 and also exits,
                # letting systemd restart the whole controller with the new code.
                sys.exit(3)
            periodic[AUTO_UPDATE_KEY] = now.isoformat()

    def _periodic_timestamp(key: str) -> datetime | None:
        raw = periodic.get(key)
        if not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    last_enqueue_at = _periodic_timestamp(JOB_CONSOLIDATE_BLOCKS)
    last_optimize_enqueue_at = _periodic_timestamp(JOB_OPTIMIZE)

    last_session_import_raw = periodic.get(SESSION_IMPORT_KEY)
    last_session_import_at: datetime | None = None
    if isinstance(last_session_import_raw, str):
        try:
            last_session_import_at = datetime.fromisoformat(last_session_import_raw)
        except ValueError:
            last_session_import_at = None

    if session_import_interval_seconds < 0:
        import_due = False
    elif session_import_interval_seconds == 0 or last_session_import_at is None:
        import_due = True
    else:
        import_due = (now - last_session_import_at).total_seconds() >= session_import_interval_seconds
    imported_sessions: dict[str, int] = {}
    if import_due:
        imported = _servicectl_import_sessions(
            root, timeout=_scaled_timeout(SESSION_IMPORT_KEY, _SESSION_IMPORT_TIMEOUT_SECONDS)
        )
        if _note_subprocess_outcome(SESSION_IMPORT_KEY, imported is None):
            periodic[SESSION_IMPORT_KEY] = now.isoformat()
        imported_sessions = imported or {}

    # Recall indexing (semantic past-session recall) runs on the maintenance cadence.
    RECALL_INDEX_KEY = "index_recall_sessions"
    last_recall_index_at = _periodic_timestamp(RECALL_INDEX_KEY)
    if maintenance_interval_seconds <= 0 or last_recall_index_at is None:
        recall_index_due = True
    else:
        recall_index_due = (now - last_recall_index_at).total_seconds() >= maintenance_interval_seconds
    indexed_recall: dict[str, int] = {}
    if recall_index_due:
        indexed = _servicectl_index_recall(
            root, timeout=_scaled_timeout(RECALL_INDEX_KEY, _RECALL_INDEX_TIMEOUT_SECONDS)
        )
        if _note_subprocess_outcome(RECALL_INDEX_KEY, indexed is None):
            periodic[RECALL_INDEX_KEY] = now.isoformat()
        indexed_recall = indexed or {}

    WORKSPACE_PRUNE_KEY = "prune_workspaces"
    last_workspace_prune_at = _periodic_timestamp(WORKSPACE_PRUNE_KEY)
    workspace_prune_due = (
        last_workspace_prune_at is None
        or (now - last_workspace_prune_at).total_seconds() >= _WORKSPACE_PRUNE_INTERVAL_SECONDS
    )
    pruned_workspaces: dict[str, Any] = {}
    if workspace_prune_due:
        pruned = _servicectl_prune_workspaces(
            root, timeout=_scaled_timeout(WORKSPACE_PRUNE_KEY, _WORKSPACE_PRUNE_TIMEOUT_SECONDS)
        )
        if _note_subprocess_outcome(WORKSPACE_PRUNE_KEY, pruned is None):
            periodic[WORKSPACE_PRUNE_KEY] = now.isoformat()
        pruned_workspaces = pruned or {}

    # Public savings rollup: the schedule (24 h on success, 15 min -> 6 h
    # backoff on failure) and the checkpoint now live in the rollup's own
    # durable state file, so a dropped POST retries in minutes instead of
    # costing a whole day, and a machine that is only briefly awake still
    # catches up on wake. Ticking every pass is cheap: a not-due check is one
    # small JSON read that no-ops.
    PUBLIC_ROLLUP_KEY = "public_rollup"
    public_rollup_checkpoint_day = state.get("public_rollup_checkpoint_day")
    try:
        from lemoncrow.core.service.telemetry.public_rollup import maybe_flush_public_rollup

        rollup_result = maybe_flush_public_rollup(root, legacy_checkpoint_day=public_rollup_checkpoint_day)
        public_rollup_checkpoint_day = rollup_result.get("checkpoint_day") or public_rollup_checkpoint_day
        if rollup_result.get("reason") not in {"not_due", "locked"}:
            periodic[PUBLIC_ROLLUP_KEY] = now.isoformat()
        if rollup_result.get("reason") in {"post_failed", "error"}:
            logger.warning("public rollup flush incomplete: %s", rollup_result)
    except Exception:
        logger.exception("public rollup flush error")

    job_queue_health_before = store.jobs.job_queue_health()
    enqueued: list[str] = []
    if maintenance_interval_seconds <= 0 or last_enqueue_at is None:
        due = True
    else:
        due = (now - last_enqueue_at).total_seconds() >= maintenance_interval_seconds

    if due:
        active_jobs = [
            job
            for job in store.jobs.list_jobs(job_type=JOB_CONSOLIDATE_BLOCKS, limit=200)
            if job["status"] in {"pending", "running", "failed"}
        ]
        if not active_jobs:
            job_id = store.jobs.enqueue_job(
                JOB_CONSOLIDATE_BLOCKS,
                {"dry_run": False, "source": "servicectl"},
            )
            enqueued.append(job_id)
            periodic[JOB_CONSOLIDATE_BLOCKS] = now.isoformat()

    # Jobs-table retention runs on the same maintenance cadence: without it,
    # succeeded/failed rows accumulate forever.
    last_retention_enqueue_at = _periodic_timestamp(JOB_RETENTION_CLEANUP)
    if maintenance_interval_seconds <= 0 or last_retention_enqueue_at is None:
        retention_due = True
    else:
        retention_due = (now - last_retention_enqueue_at).total_seconds() >= maintenance_interval_seconds
    if retention_due:
        active_retention_jobs = [
            job
            for job in store.jobs.list_jobs(job_type=JOB_RETENTION_CLEANUP, limit=200)
            if job["status"] in {"pending", "running", "failed"}
        ]
        if not active_retention_jobs:
            job_id = store.jobs.enqueue_job(
                JOB_RETENTION_CLEANUP,
                {"days": 14, "source": "servicectl"},
            )
            enqueued.append(job_id)
            periodic[JOB_RETENTION_CLEANUP] = now.isoformat()

    automation = load_automation_config(root)
    if automation.enabled:
        if maintenance_interval_seconds <= 0 or last_optimize_enqueue_at is None:
            optimize_due = True
        else:
            optimize_due = (now - last_optimize_enqueue_at).total_seconds() >= maintenance_interval_seconds
        if optimize_due:
            active_optimize_jobs = [
                job
                for job in store.jobs.list_jobs(job_type=JOB_OPTIMIZE, limit=200)
                if job["status"] in {"pending", "running", "failed"}
            ]
            if not active_optimize_jobs:
                job_id = store.jobs.enqueue_job(
                    JOB_OPTIMIZE,
                    {"days": 7, "host": None, "source": "servicectl"},
                )
                enqueued.append(job_id)
                periodic[JOB_OPTIMIZE] = now.isoformat()

    # Process queued jobs in subprocesses so heavy handlers (consolidation,
    # optimization) keep their LLM heap out of the daemon. Each subprocess
    # claims one job atomically, does the work, and exits.
    processed: list[str] = []
    while len(processed) < 20:
        try:
            job_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lemoncrow.gateway.cli",
                    "--root",
                    str(root),
                    "worker",
                    "run-once",
                    "--json",
                ],
                capture_output=True,
                timeout=600,
            )
            if job_result.returncode != 0:
                logging.warning(
                    "worker run-once subprocess failed (rc=%d): %s",
                    job_result.returncode,
                    job_result.stderr[-300:].decode("utf-8", errors="replace").strip(),
                )
                break
            data = json.loads(job_result.stdout)
            job_id = data.get("job_id")
            if not data.get("processed") or job_id is None:
                break  # queue empty
            processed.append(str(job_id))
        except Exception:
            logging.exception("worker run-once subprocess error")
            break

    # Push cumulative savings to the signed-in account.
    #
    # The only other background pusher is the savings-reconciler thread started
    # by the API service (``core/service/api.py`` -> start_savings_reconciler),
    # which lives inside the OPTIONAL visualization stack. An install that never
    # runs ``lc stack start`` -- or whose stack is down -- therefore had no
    # automated channel at all, and lemoncrow.com/savings only advanced on the
    # forced push inside ``lc account login``. The controller is the always-on
    # component, so it reports here too.
    #
    # Cheap and privacy-preserving by construction: maybe_report_usage throttles
    # itself on its own on-disk watermark and returns immediately when no auth
    # token is stored (login is the opt-in), so a logged-out install never
    # phones home. Guarded -- a network failure must never abort the tick.
    usage_reported = False
    try:
        from lemoncrow.core.capabilities.licensing.usage_report import maybe_report_usage

        usage_reported = bool(maybe_report_usage(root))
    except Exception:
        logging.exception("account usage report failed")

    payload = {
        "last_tick_at": now.isoformat(),
        "last_usage_report_at": (now.isoformat() if usage_reported else state.get("last_usage_report_at")),
        "last_processed_jobs": processed,
        "last_enqueued_jobs": enqueued,
        "last_imported_sessions": imported_sessions if import_due else state.get("last_imported_sessions", {}),
        "last_session_import_at": periodic.get(SESSION_IMPORT_KEY),
        "last_indexed_recall": indexed_recall if recall_index_due else state.get("last_indexed_recall", {}),
        "last_recall_index_at": periodic.get(RECALL_INDEX_KEY),
        "last_pruned_workspaces": (
            pruned_workspaces if workspace_prune_due else state.get("last_pruned_workspaces", {})
        ),
        "last_workspace_prune_at": periodic.get(WORKSPACE_PRUNE_KEY),
        "public_rollup_checkpoint_day": public_rollup_checkpoint_day,
        "last_exit_reason": state.get("last_exit_reason"),
        "periodic_jobs": periodic,
        "subprocess_timeouts": subprocess_timeouts,
        "started_at": state.get("started_at"),
        "job_queue_health": store.jobs.job_queue_health(),
    }
    _write_servicectl_state(root, payload)
    job_queue_health = payload["job_queue_health"]
    return {
        "enqueued_jobs": enqueued,
        "processed_jobs": processed,
        "imported_sessions": imported_sessions,
        "session_import_ran": import_due,
        "indexed_recall": indexed_recall,
        "recall_index_ran": recall_index_due,
        "pruned_workspaces": pruned_workspaces,
        "workspace_prune_ran": workspace_prune_due,
        "job_queue_health_before": job_queue_health_before,
        "job_queue_health": job_queue_health,
        "pending_jobs": job_queue_health["active"],
        "tick_at": now.isoformat(),
    }
