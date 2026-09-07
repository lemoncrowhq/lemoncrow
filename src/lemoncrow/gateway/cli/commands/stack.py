"""Thin ``lc stack`` command group (Phase 25-03, QBL-CLI-02).

The optional visualization-stack process lifecycle lives in
``infra/runtime/stack_lifecycle.py``. These callbacks are thin wrappers; the
hidden ``stack run`` supervisor stays ``hidden=True`` and invocable.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

import click

from lemoncrow.gateway.cli.commands._shared import _emit
from lemoncrow.gateway.integrations.openmemory_lifecycle import project_root as _project_root
from lemoncrow.infra.runtime.daemon_units import (
    DEFAULT_STACK_FRONTEND_HOST,
    DEFAULT_STACK_FRONTEND_PORT,
    DEFAULT_STACK_SERVICE_HOST,
    DEFAULT_STACK_SERVICE_PORT,
    STACK_UNIT,
    SYSTEMD_USER_DIR,
)
from lemoncrow.infra.runtime.servicectl_lifecycle import (
    _pid_is_running,
)
from lemoncrow.infra.runtime.stack_lifecycle import (
    _clear_stack_pidfiles,
    _ensure_stack_frontend_dependencies,
    _get_node_dir,
    _get_npm_path,
    _port_is_serving,
    _read_stack_state,
    _signal_process_group,
    _stack_dir,
    _stack_frontend_dir,
    _stack_frontend_is_prebuilt,
    _stack_frontend_pid_path,
    _stack_log_path,
    _stack_pid_path,
    _stack_service_pid_path,
    _stack_status_payload,
    _stop_stack_processes,
    _tail_text,
    _write_stack_state,
)


@click.group("stack", hidden=True)
def stack_group() -> None:
    """Manage the optional visualization stack (service + frontend)."""


@stack_group.command("start")
@click.option("--with-docs", is_flag=True, help="Deprecated; docs are no longer managed by lc stack.")
@click.pass_context
def stack_start(ctx: click.Context, with_docs: bool) -> None:
    """Start the optional visualization stack with native processes."""
    root = ctx.obj["root"]
    if (SYSTEMD_USER_DIR / STACK_UNIT).exists():
        click.echo(
            f"Notice: {STACK_UNIT} is installed. "
            "Prefer using 'lc systemd restart' or 'systemctl --user restart lemoncrow-stack'."
        )
    if with_docs:
        click.echo("Notice: docs are no longer part of the managed stack; starting service + frontend only.")

    payload = _stack_status_payload(root)
    if payload["running"]:
        click.echo(f"frontend: {payload['frontend_url']}")
        click.echo(f"service: {payload['service_url']}")
        return

    if payload["runner_running"] or payload["service_running"] or payload["frontend_running"]:
        _stop_stack_processes(root, force=True)

    stack_dir = _stack_dir(root)
    stack_dir.mkdir(parents=True, exist_ok=True)
    project_root = _project_root()
    command = [
        sys.executable,
        "-m",
        "lemoncrow.gateway.cli",
        "--root",
        str(root),
        "stack",
        "run",
    ]
    env = os.environ.copy()
    env["LEMONCROW_ROOT"] = str(root)
    # The supervisor changes cwd to the installed package checkout. Preserve
    # the repository that invoked ``lc stack start`` so map/code APIs resolve
    # the user's workspace instead of LemonCrow's own source directory.
    if not env.get("LEMONCROW_WORKSPACE_ROOT"):
        from lemoncrow.core.foundation.paths import resolve_workspace_root

        with contextlib.suppress(RuntimeError):
            env["LEMONCROW_WORKSPACE_ROOT"] = str(resolve_workspace_root())
    with _stack_log_path(root).open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=project_root,
            start_new_session=True,
            close_fds=True,
        )

    _stack_pid_path(root).write_text(f"{proc.pid}\n", encoding="utf-8")
    deadline = time.time() + 2
    while time.time() < deadline and _pid_is_running(proc.pid):
        time.sleep(0.1)

    if not _pid_is_running(proc.pid):
        tail = _tail_text(_stack_log_path(root))
        detail = f"\n\n{tail}" if tail else ""
        raise click.ClickException(f"native stack failed to start; inspect {_stack_log_path(root)}{detail}")

    payload = _stack_status_payload(root)
    click.echo(f"frontend: {payload['frontend_url']}")
    click.echo(f"service: {payload['service_url']}")


@stack_group.command("stop")
@click.option("--force", is_flag=True, help="Use SIGKILL if the stack does not stop cleanly.")
@click.pass_context
def stack_stop(ctx: click.Context, force: bool) -> None:
    """Stop the optional visualization stack."""
    root = ctx.obj["root"]
    if (SYSTEMD_USER_DIR / STACK_UNIT).exists():
        click.echo(
            f"Notice: {STACK_UNIT} is installed. "
            "Prefer using 'lc systemd uninstall' or 'systemctl --user stop lemoncrow-stack'."
        )
    payload = _stop_stack_processes(root, force=force)
    if payload["running"]:
        raise click.ClickException("stack did not stop cleanly")
    click.echo("stopped stack")


@stack_group.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of text.")
@click.pass_context
def stack_status(ctx: click.Context, as_json: bool) -> None:
    """Show visualization stack process status."""
    root = ctx.obj["root"]
    payload = _stack_status_payload(root)
    if as_json:
        _emit(payload, as_json=True)
        return
    if (SYSTEMD_USER_DIR / STACK_UNIT).exists():
        subprocess.run(["systemctl", "--user", "status", STACK_UNIT, "--no-pager"], check=False)
        click.echo("-" * 40)
    click.echo(f"running: {str(payload['running']).lower()}")
    click.echo(f"runner_pid: {payload['runner_pid']}")
    click.echo(f"service_pid: {payload['service_pid']}")
    click.echo(f"frontend_pid: {payload['frontend_pid']}")
    click.echo(f"log_file: {payload['log_file']}")
    click.echo(f"service: {payload['service_url']}")
    click.echo(f"frontend: {payload['frontend_url']}")
    if payload["last_exit_reason"]:
        click.echo(f"last_exit_reason: {payload['last_exit_reason']}")


@stack_group.command("run", hidden=True)
@click.option("--service-host", default=DEFAULT_STACK_SERVICE_HOST, show_default=True)
@click.option("--service-port", default=DEFAULT_STACK_SERVICE_PORT, show_default=True, type=int)
@click.option("--frontend-host", default=DEFAULT_STACK_FRONTEND_HOST, show_default=True)
@click.option("--frontend-port", default=DEFAULT_STACK_FRONTEND_PORT, show_default=True, type=int)
@click.pass_context
def stack_run(
    ctx: click.Context,
    service_host: str,
    service_port: int,
    frontend_host: str,
    frontend_port: int,
) -> None:
    """Internal long-running supervisor for the optional native stack."""
    root = ctx.obj["root"]
    frontend_dir = _stack_frontend_dir()
    _ensure_stack_frontend_dependencies(frontend_dir)

    _stack_dir(root).mkdir(parents=True, exist_ok=True)
    _stack_pid_path(root).write_text(f"{os.getpid()}\n", encoding="utf-8")
    # systemd's lemoncrow-stack.service (``background service start``) binds the
    # same port. Spawning a second uvicorn there exits 3 on EADDRINUSE, which
    # the loop below treats as a fatal service exit and tears the frontend down
    # with it -- so the UI never stays up. Adopt the live service instead.
    adopted_service = _port_is_serving(service_host, service_port)
    _write_stack_state(
        root,
        {
            "started_at": datetime.now(UTC).isoformat(),
            "last_exit_reason": None,
            "service_url": f"http://localhost:{service_port}",
            "frontend_url": f"http://localhost:{frontend_port}",
            "service_adopted": adopted_service,
            "service_host": service_host,
            "service_port": service_port,
        },
    )

    service_env = os.environ.copy()
    service_env.update(
        {
            "LEMONCROW_ROOT": str(root),
            "LEMONCROW_SERVICE_HOST": service_host,
            "LEMONCROW_SERVICE_PORT": str(service_port),
            "LEMONCROW_REQUIRE_AUTH": "false",
        }
    )
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_URL"] = f"http://localhost:{service_port}"
    if _stack_frontend_is_prebuilt(frontend_dir):
        # Release installs ship a built bundle, not a checkout: no npm, no vite.
        frontend_command = [
            sys.executable,
            "-m",
            "lemoncrow.infra.runtime.static_frontend",
            "--dir",
            str(frontend_dir),
            "--host",
            frontend_host,
            "--port",
            str(frontend_port),
            "--api-url",
            f"http://localhost:{service_port}",
        ]
    else:
        # npm's shebang needs `node` on PATH even when npm itself is invoked by
        # full path (see _get_node_dir) -- matters under a minimal-PATH spawner
        # like launchd, which the frontend can otherwise fail to start under.
        # Resolve npm once and pass it through so node is paired with the same
        # install rather than picked independently (avoids a node/npm mismatch).
        _npm_path = _get_npm_path()
        _node_dir = _get_node_dir(_npm_path)
        if _node_dir:
            frontend_env["PATH"] = f"{_node_dir}:{frontend_env.get('PATH', '')}"
        frontend_command = [
            _npm_path,
            "exec",
            "vite",
            "--",
            "--host",
            frontend_host,
            "--port",
            str(frontend_port),
            "--strictPort",
        ]

    service_proc: subprocess.Popen[bytes] | None = None
    if adopted_service:
        print(
            f"[stack] service already listening on {service_host}:{service_port}; supervising frontend only",
            flush=True,
        )
    else:
        service_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lemoncrow.gateway.cli",
                "--root",
                str(root),
                "service",
                "start",
                "--host",
                service_host,
                "--port",
                str(service_port),
            ],
            cwd=_project_root(),
            env=service_env,
            start_new_session=True,
        )
    frontend_proc = subprocess.Popen(
        frontend_command,
        cwd=frontend_dir,
        env=frontend_env,
        start_new_session=True,
    )
    if service_proc is not None:
        _stack_service_pid_path(root).write_text(f"{service_proc.pid}\n", encoding="utf-8")
    _stack_frontend_pid_path(root).write_text(f"{frontend_proc.pid}\n", encoding="utf-8")

    stopping = False

    def _handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    previous_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    previous_sigint = signal.signal(signal.SIGINT, _handle_stop)

    exit_reason = "stopped"
    exit_code = 0
    try:
        while True:
            service_code = service_proc.poll() if service_proc is not None else None
            frontend_code = frontend_proc.poll()
            if stopping:
                break
            if service_code is not None:
                exit_reason = f"service_exited:{service_code}"
                exit_code = service_code or 1
                break
            if frontend_code is not None:
                exit_reason = f"frontend_exited:{frontend_code}"
                exit_code = frontend_code or 1
                break
            time.sleep(1)
    finally:
        owned = [proc for proc in (frontend_proc, service_proc) if proc is not None]
        for proc in owned:
            if proc.poll() is None:
                _signal_process_group(proc.pid, signal.SIGTERM)
        deadline = time.time() + 5
        while time.time() < deadline and any(proc.poll() is None for proc in owned):
            time.sleep(0.1)
        for proc in owned:
            if proc.poll() is None:
                _signal_process_group(proc.pid, signal.SIGKILL)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        _clear_stack_pidfiles(root)
        state = _read_stack_state(root)
        state["last_exit_reason"] = exit_reason
        state["stopped_at"] = datetime.now(UTC).isoformat()
        _write_stack_state(root, state)

    raise SystemExit(exit_code)
