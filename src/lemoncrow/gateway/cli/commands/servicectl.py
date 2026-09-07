"""Thin service/worker/servicectl command groups + ``logs`` (Phase 25-03).

Sibling OS/daemon process-control surfaces. Heavy lifecycle logic lives in
``infra/runtime/servicectl_lifecycle.py``. The hidden ``servicectl run`` loop
stays ``hidden=True`` and invocable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from lemoncrow.gateway.cli.commands._shared import _emit
from lemoncrow.gateway.integrations.openmemory_lifecycle import mcp_log_path as _mcp_log_path
from lemoncrow.gateway.integrations.openmemory_lifecycle import (
    openmemory_log_path as _openmemory_log_path,
)
from lemoncrow.gateway.integrations.openmemory_lifecycle import run_compose as _run_compose
from lemoncrow.infra.runtime.daemon_units import (
    CONTROLLER_UNIT,
    LETTA_UNIT,
    MCP_UNIT,
    OPENMEMORY_UNIT,
    STACK_UNIT,
    SYSTEMD_USER_DIR,
    ZOEKT_UNIT,
    _is_linux,
)
from lemoncrow.infra.runtime.servicectl_lifecycle import (
    _clear_servicectl_pid,
    _kill_orphan_servicectl_processes,
    _pid_is_running,
    _read_servicectl_pid,
    _read_servicectl_state,
    _servicectl_dir,
    _servicectl_log_path,
    _servicectl_pid_path,
    _servicectl_status_payload,
    _servicectl_tick,
    _write_servicectl_state,
)
from lemoncrow.infra.runtime.stack_lifecycle import (
    _stack_log_path,
)

logger = logging.getLogger(__name__)

# The tick subprocess's own work can legitimately run long: its budgets
# (import <=300s + recall index <=300s + occasional workspace prune <=600s +
# up to 20 queued jobs at <=600s each) can exceed a short fixed ceiling under
# normal load. 30 minutes is generous enough that only a genuinely stuck tick
# trips it; a timeout is now recoverable (logged, loop continues) rather than
# fatal, so this is a "give up and retry" ceiling, not a completion guarantee.
_TICK_SUBPROCESS_TIMEOUT_SECONDS = 1800


@click.group("service")
def service_group() -> None:
    """Production service commands."""


@service_group.command("start")
@click.option("--host", default=None, help="Bind host (overrides LEMONCROW_SERVICE_HOST).")
@click.option("--port", default=None, type=int, help="Bind port (overrides LEMONCROW_SERVICE_PORT).")
@click.option("--reload", is_flag=True, default=False, help="Enable uvicorn auto-reload.")
def service_start(host: str | None, port: int | None, reload: bool) -> None:
    """Start the LemonCrow HTTP service API."""
    try:
        from lemoncrow.core.service.api import main as service_main
    except ImportError as exc:
        if "cannot import name 'main'" in str(exc):
            raise click.ClickException(
                "The service API 'main' entrypoint is missing. Ensure your 'lc' installation is up to date."
            ) from exc
        raise click.ClickException(
            "Could not start the service API. Ensure all dependencies are installed: uv sync --extra api"
        ) from exc
    service_main(host=host, port=port, reload=reload)


@service_group.command("config")
def service_config() -> None:
    """Print current service configuration (no secret values)."""
    import json

    from lemoncrow.core.service.config import cfg

    click.echo(json.dumps(cfg.as_dict(), indent=2))


# --------------------------------------------------------------------------- #
# Worker group (P6)                                                           #
# --------------------------------------------------------------------------- #


@click.group("worker")
def worker_group() -> None:
    """Worker/job queue commands."""


@worker_group.command("start")
@click.pass_context
def worker_start(ctx: click.Context) -> None:
    """Start the background worker loop."""
    try:
        from lemoncrow.core.service.worker import Worker
    except ImportError as exc:
        raise click.ClickException("Worker dependencies not available.") from exc

    from lemoncrow.infra.storage.factory import create_store

    root = ctx.obj["root"]
    store = create_store(root)
    store.init()
    worker = Worker(store=store)
    click.echo("Worker started. Press Ctrl+C to stop.")
    worker.run()


@worker_group.command("run-once")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def worker_run_once(ctx: click.Context, as_json: bool) -> None:
    """Claim and process one pending job then exit.

    ``--json`` emits ``{"processed": bool, "job_id": str | None}``. The
    servicectl tick drains the queue by looping this subprocess and parsing
    that payload; without the flag the option error (exit 2) aborted every
    drain, so pending jobs never ran on a stock install.
    """
    try:
        from lemoncrow.core.service.worker import Worker
    except ImportError as exc:
        raise click.ClickException("Worker dependencies not available.") from exc

    from lemoncrow.infra.storage.factory import create_store

    root = ctx.obj["root"]
    store = create_store(root)
    store.init()
    worker = Worker(store=store)
    processed = worker.run_once()
    if as_json:
        _emit({"processed": processed is not None, "job_id": processed}, as_json=True)
        return
    if processed:
        click.echo(f"processed job: {processed}")
    else:
        click.echo("no pending jobs")


@worker_group.command("enqueue")
@click.argument("job_type")
@click.option("--payload", default="{}", show_default=True, help="Inline JSON object payload.")
@click.option("--max-attempts", default=3, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def worker_enqueue(
    ctx: click.Context,
    job_type: str,
    payload: str,
    max_attempts: int,
    as_json: bool,
) -> None:
    """Queue one background job."""
    from lemoncrow.infra.storage.factory import create_store

    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid JSON payload: {exc}") from exc
    if not isinstance(payload_data, dict):
        raise click.ClickException("payload must decode to a JSON object")

    store = create_store(ctx.obj["root"])
    store.init()
    job_id = store.jobs.enqueue_job(job_type, payload_data, max_attempts=max_attempts)
    result = {
        "job_id": job_id,
        "job_type": job_type,
        "status": "pending",
    }
    _emit(result, as_json=as_json) if as_json else click.echo(job_id)


@worker_group.command("list")
@click.option("--status", default=None, help="Filter by job status.")
@click.option("--job-type", default=None, help="Filter by job type.")
@click.option("--limit", default=20, type=int, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def worker_list(ctx: click.Context, status: str | None, job_type: str | None, limit: int, as_json: bool) -> None:
    """List queued and processed jobs."""
    from lemoncrow.infra.storage.factory import create_store

    store = create_store(ctx.obj["root"])
    store.init()
    jobs = store.jobs.list_jobs(status=status, job_type=job_type, limit=limit)
    if as_json:
        _emit(jobs, as_json=True)
        return
    if not jobs:
        click.echo("(no jobs)")
        return
    for job in jobs:
        click.echo(f"{job['id']}\t{job['job_type']}\t{job['status']}\tattempts={job['attempts']}")


@click.group("servicectl")
def servicectl_group() -> None:
    """Manage the background offline processing controller."""


@servicectl_group.command("tick")
@click.option("--maintenance-interval-seconds", default=300, show_default=True, type=int)
@click.option("--session-import-interval-seconds", default=1800, show_default=True, type=int)
@click.option("--auto-update", is_flag=True, help="Apply git auto-update if available (exits 3 when applied).")
@click.option("--auto-update-interval-seconds", default=3600, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def servicectl_tick(
    ctx: click.Context,
    maintenance_interval_seconds: int,
    session_import_interval_seconds: int,
    auto_update: bool,
    auto_update_interval_seconds: int,
    as_json: bool,
) -> None:
    """Run one maintenance tick: enqueue due jobs and process pending work."""
    payload = _servicectl_tick(
        ctx.obj["root"],
        maintenance_interval_seconds=maintenance_interval_seconds,
        session_import_interval_seconds=session_import_interval_seconds,
        auto_update=auto_update,
        auto_update_interval_seconds=auto_update_interval_seconds,
    )
    _emit(payload, as_json=as_json) if as_json else click.echo(json.dumps(payload, indent=2))


@servicectl_group.command("start")
@click.option("--interval-seconds", default=300, show_default=True, type=int)
@click.option("--maintenance-interval-seconds", default=300, show_default=True, type=int)
@click.option("--session-import-interval-seconds", default=1800, show_default=True, type=int)
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def servicectl_start(
    ctx: click.Context,
    interval_seconds: int,
    maintenance_interval_seconds: int,
    session_import_interval_seconds: int,
    as_json: bool,
) -> None:
    """Start the detached background controller."""
    root = ctx.obj["root"]
    if (SYSTEMD_USER_DIR / CONTROLLER_UNIT).exists():
        click.echo(
            f"Notice: {CONTROLLER_UNIT} is installed. "
            "Prefer using 'lc systemd restart' or 'systemctl --user restart lemoncrow-controller'."
        )

    _kill_orphan_servicectl_processes(root)
    status = _servicectl_status_payload(root)
    if status["running"]:
        if as_json:
            _emit(status, as_json=True)
        else:
            click.echo(f"already running (pid {status['pid']})")
        return

    _clear_servicectl_pid(root)
    control_dir = _servicectl_dir(root)
    control_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "lemoncrow.gateway.cli",
        "--root",
        str(root),
        "servicectl",
        "run",
        "--interval-seconds",
        str(interval_seconds),
        "--maintenance-interval-seconds",
        str(maintenance_interval_seconds),
        "--session-import-interval-seconds",
        str(session_import_interval_seconds),
    ]
    env = os.environ.copy()
    env["LEMONCROW_ROOT"] = str(root)
    with _servicectl_log_path(root).open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    _servicectl_pid_path(root).write_text(f"{proc.pid}\n", encoding="utf-8")
    state = _read_servicectl_state(root)
    state.update(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "last_exit_reason": None,
        }
    )
    _write_servicectl_state(root, state)
    payload = _servicectl_status_payload(root)
    if as_json:
        _emit(payload, as_json=True)
        return
    click.echo(f"started servicectl (pid {proc.pid})")


def _signal_process_group(pid: int, sig: int) -> None:
    """Signal the controller's whole process group, tolerating a pid that is
    alive but not a group leader.

    The controller starts with ``start_new_session=True`` (gid == pid), so
    ``killpg`` is correct for a genuinely-started controller. But a stale
    pidfile whose pid has been recycled by an unrelated, non-leader process
    makes ``killpg`` raise ``ProcessLookupError`` (no group with that pgid),
    which would otherwise crash ``lc servicectl stop`` with a traceback and
    leave the pidfile in place forever. Fall back to signalling just the pid,
    and treat an already-gone process as success (the caller's running-check
    clears the pidfile).
    """
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


@servicectl_group.command("stop")
@click.option("--force", is_flag=True, help="Use SIGKILL if SIGTERM does not stop the process.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def servicectl_stop(ctx: click.Context, force: bool, as_json: bool) -> None:
    """Stop the detached background controller."""
    root = ctx.obj["root"]
    pid = _read_servicectl_pid(root)
    if pid is None or not _pid_is_running(pid):
        _clear_servicectl_pid(root)
        state = _read_servicectl_state(root)
        state["last_exit_reason"] = "not_running"
        _write_servicectl_state(root, state)
        payload = _servicectl_status_payload(root)
        if as_json:
            _emit(payload, as_json=True)
        else:
            click.echo("servicectl is not running")
        return

    # The controller is started with start_new_session=True (its own process
    # group, gid == pid) and its periodic tick spawns child subprocesses
    # (import/recall-index/workspace-prune) that inherit that group. Signal
    # the whole group, not just the controller pid, or a long-running child
    # (workspace prune can run up to 10 minutes) survives as an orphan and
    # keeps writing into the store after this command reports "stopped".
    _signal_process_group(pid, signal.SIGTERM)
    deadline = time.time() + 5
    while _pid_is_running(pid) and time.time() < deadline:
        time.sleep(0.1)
    if _pid_is_running(pid):
        if not force:
            if as_json:
                _emit(_servicectl_status_payload(root), as_json=True)
            else:
                click.echo(f"servicectl (pid {pid}) did not stop after SIGTERM; retry with --force", err=True)
            ctx.exit(1)
        _signal_process_group(pid, signal.SIGKILL)
        kill_deadline = time.time() + 5
        while _pid_is_running(pid) and time.time() < kill_deadline:
            time.sleep(0.1)
        if _pid_is_running(pid):
            if as_json:
                _emit(_servicectl_status_payload(root), as_json=True)
            else:
                click.echo(f"servicectl (pid {pid}) survived SIGKILL", err=True)
            ctx.exit(1)
    _clear_servicectl_pid(root)
    state = _read_servicectl_state(root)
    state["last_exit_reason"] = "stopped"
    _write_servicectl_state(root, state)
    payload = _servicectl_status_payload(root)
    if as_json:
        _emit(payload, as_json=True)
        return
    click.echo("stopped servicectl")


@servicectl_group.command("status")
@click.option("--json", "as_json", is_flag=True)
@click.pass_context
def servicectl_status(ctx: click.Context, as_json: bool) -> None:
    """Show background controller status. Exits 3 when not running (systemctl convention)."""
    root = ctx.obj["root"]
    payload = _servicectl_status_payload(root)
    if as_json:
        _emit(payload, as_json=True)
        if not payload["running"]:
            ctx.exit(3)
        return

    if payload["running"] and shutil.which("systemctl"):
        pid = payload["pid"]
        # Try to show system-level status for the PID
        for cmd in [
            ["systemctl", "--user", "status", str(pid), "--no-pager"],
            ["systemctl", "status", str(pid), "--no-pager"],
        ]:
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    click.echo(res.stdout)
                    click.echo("-" * 40)
                    break
            except OSError:
                logger.debug("systemctl status probe failed", exc_info=True)

    click.echo(f"running: {str(payload['running']).lower()}")
    click.echo(f"pid: {payload['pid']}")
    click.echo(f"log_file: {payload['log_file']}")
    if payload["last_tick_at"]:
        click.echo(f"last_tick_at: {payload['last_tick_at']}")
    if payload["last_processed_jobs"]:
        click.echo("last_processed_jobs: " + ", ".join(payload["last_processed_jobs"]))
    health = payload.get("job_queue_health") or {}
    if health:
        click.echo(
            "job_queue_health: "
            f"pending={health.get('pending', 0)} "
            f"running={health.get('running', 0)} "
            f"failed={health.get('failed', 0)} "
            f"dead={health.get('dead', 0)} "
            f"stuck_running={health.get('stuck_running', 0)}"
        )
    if not payload["running"]:
        ctx.exit(3)
    if not payload["running"]:
        ctx.exit(3)


@servicectl_group.command("run", hidden=True)
@click.option("--interval-seconds", default=300, show_default=True, type=int)
@click.option("--maintenance-interval-seconds", default=900, show_default=True, type=int)
@click.option("--session-import-interval-seconds", default=1800, show_default=True, type=int)
@click.option("--auto-update", is_flag=True, help="Check for git updates periodically.")
@click.option("--auto-update-interval-seconds", default=7200, show_default=True, type=int)
@click.pass_context
def servicectl_run(
    ctx: click.Context,
    interval_seconds: int,
    maintenance_interval_seconds: int,
    session_import_interval_seconds: int,
    auto_update: bool,
    auto_update_interval_seconds: int,
) -> None:
    """Internal long-running background loop (thin process-manager).

    Each tick runs as a child subprocess so all Python heap allocated during
    the tick is freed on subprocess exit.  This keeps the long-lived run loop
    itself under ~100 MB regardless of how many sessions have been imported or
    jobs processed.
    """
    root = ctx.obj["root"]
    # Self-register the pidfile. ``servicectl start`` writes it for the daemon
    # it spawns, but launchd/systemd units invoke ``servicectl run`` directly --
    # nothing wrote a pid then, so ``servicectl status``/``stop`` reported
    # ``running: false, pid: None`` for a controller that was demonstrably alive.
    _servicectl_dir(root).mkdir(parents=True, exist_ok=True)
    _servicectl_pid_path(root).write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        _run_servicectl_loop(
            root,
            interval_seconds=interval_seconds,
            maintenance_interval_seconds=maintenance_interval_seconds,
            session_import_interval_seconds=session_import_interval_seconds,
            auto_update=auto_update,
            auto_update_interval_seconds=auto_update_interval_seconds,
        )
    finally:
        # Only reap our own entry: a controller that replaced us (auto-update
        # restart) must keep its pid on file.
        if _read_servicectl_pid(root) == os.getpid():
            _clear_servicectl_pid(root)


def _run_servicectl_loop(
    root: Path,
    *,
    interval_seconds: int,
    maintenance_interval_seconds: int,
    session_import_interval_seconds: int,
    auto_update: bool,
    auto_update_interval_seconds: int,
) -> None:
    """Tick forever, each tick in a child subprocess (see ``servicectl_run``)."""
    try:
        while True:
            cmd = [
                sys.executable,
                "-m",
                "lemoncrow.gateway.cli",
                "--root",
                str(root),
                "servicectl",
                "tick",
                "--maintenance-interval-seconds",
                str(maintenance_interval_seconds),
                "--session-import-interval-seconds",
                str(session_import_interval_seconds),
            ]
            if auto_update:
                cmd += [
                    "--auto-update",
                    "--auto-update-interval-seconds",
                    str(auto_update_interval_seconds),
                ]
            try:
                result = subprocess.run(cmd, timeout=_TICK_SUBPROCESS_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "servicectl tick subprocess exceeded %ds; continuing to next interval",
                    _TICK_SUBPROCESS_TIMEOUT_SECONDS,
                )
                time.sleep(max(1, interval_seconds))
                continue
            if result.returncode == 3:
                # Tick subprocess applied an auto-update (exit code 3 = restart needed).
                # Exit here so systemd / the parent restarts this process with new code.
                state = _read_servicectl_state(root)
                state["last_exit_reason"] = "auto_update"
                _write_servicectl_state(root, state)
                raise SystemExit(0)
            time.sleep(max(1, interval_seconds))
    except KeyboardInterrupt:
        state = _read_servicectl_state(root)
        state["last_exit_reason"] = "interrupted"
        _write_servicectl_state(root, state)
        raise SystemExit(0) from None


# ----- background services (systemd / launchd) ------------------------------ #


@click.command("logs")
@click.argument("service", type=click.Choice(["stack", "controller", "letta", "openmemory", "zoekt", "mcp", "all"]))
@click.option("-f", "--follow", is_flag=True, help="Follow log output.")
@click.option("-n", "--lines", default=80, show_default=True, type=int, help="Number of lines to show.")
@click.pass_context
def logs_cmd(ctx: click.Context, service: str, follow: bool, lines: int) -> None:
    """Show logs for an LemonCrow service.

    SERVICE is one of: stack, controller, letta, openmemory, zoekt, mcp, all.

    On Linux with systemd units installed, uses journalctl to read
    the service unit logs (which contain everything the process wrote
    to stdout/stderr). On macOS and when running natively without
    systemd, tails the log file directly.
    """
    root = ctx.obj["root"]

    unit_map: dict[str, str] = {
        "stack": STACK_UNIT,
        "controller": CONTROLLER_UNIT,
        "letta": LETTA_UNIT,
        "openmemory": OPENMEMORY_UNIT,
        "zoekt": ZOEKT_UNIT,
        "mcp": MCP_UNIT,
    }

    services = list(unit_map.keys()) if service == "all" else [service]

    for s in services:
        if service == "all":
            click.echo(f"\n--- {s} logs ---")

        unit = unit_map[s]

        # Linux with systemd unit installed -> journalctl
        if _is_linux() and (SYSTEMD_USER_DIR / unit).exists():
            cmd: list[str] = ["journalctl", "--user", "-u", unit, "-n", str(lines)]
            if follow:
                cmd.append("-f")
            subprocess.run(cmd, check=False)
            continue

        # Native / macOS -> tail the log file
        if s == "stack":
            log_path = _stack_log_path(root)
        elif s == "controller":
            log_path = _servicectl_log_path(root)
        elif s == "letta":
            # Letta runs under Docker Compose -> use compose logs
            args = ["logs"]
            if follow:
                args.append("-f")
            _run_compose(args)
            continue
        elif s == "openmemory":
            log_path = _openmemory_log_path(root)
        elif s == "zoekt":
            log_path = Path(root) / "zoekt" / "zoekt.log"
        elif s == "mcp":
            log_path = _mcp_log_path(root)
        else:
            # unreachable given the Choice validator
            raise click.ClickException(f"unknown service: {s}")

        if not log_path or not log_path.exists():
            click.echo(f"(no {s} logs at {log_path or 'unknown path'})")
            continue

        if follow:
            try:
                subprocess.run(["tail", "-n", str(lines), "-f", str(log_path)], check=True)
            except FileNotFoundError as exc:
                raise click.ClickException("tail is required for --follow log streaming") from exc
            except subprocess.CalledProcessError as exc:
                raise click.ClickException(f"tail exited with code {exc.returncode}") from exc
        else:
            try:
                subprocess.run(["tail", "-n", str(lines), str(log_path)], check=True)
            except FileNotFoundError as exc:
                raise click.ClickException("tail is required for log streaming") from exc
            except subprocess.CalledProcessError as exc:
                raise click.ClickException(f"tail exited with code {exc.returncode}") from exc
        continue
