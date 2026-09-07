"""Regression cover for issue #44's controller findings.

1. ``servicectl tick`` drains the queue by looping ``worker run-once --json``.
   The subcommand had no ``--json`` option, so every drain died with click's
   exit 2 and pending jobs sat in the queue for weeks.
2. ``servicectl run`` is what launchd/systemd units execute directly. It never
   wrote a pidfile (only ``servicectl start`` did), so ``status``/``stop``
   reported ``running: false, pid: None`` for a live controller.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from lemoncrow.gateway.cli import cli
from lemoncrow.infra.runtime.servicectl_lifecycle import _read_servicectl_pid
from tests.helpers import init_store_at


def _invoke(root: Path, *args: str) -> Result:
    return CliRunner().invoke(cli, ["--root", str(root), *args])


def test_worker_run_once_json_on_empty_queue(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store_at(str(root))
    res = _invoke(root, "worker", "run-once", "--json")
    assert res.exit_code == 0, res.output
    assert json.loads(res.output) == {"processed": False, "job_id": None}


def test_worker_run_once_json_reports_processed_job(tmp_path: Path) -> None:
    root = tmp_path / "store"
    init_store_at(str(root))
    enqueued = _invoke(root, "worker", "enqueue", "retention_cleanup", "--json")
    assert enqueued.exit_code == 0, enqueued.output
    job_id = json.loads(enqueued.output)["job_id"]

    res = _invoke(root, "worker", "run-once", "--json")
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload == {"processed": True, "job_id": job_id}

    # The shape the tick's drain loop reads: it stops only on processed=False.
    assert _invoke(root, "worker", "run-once", "--json").output.strip()
    assert json.loads(_invoke(root, "worker", "run-once", "--json").output)["processed"] is False


def test_servicectl_run_registers_its_own_pidfile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "store"
    init_store_at(str(root))
    seen: list[int | None] = []

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(_read_servicectl_pid(root))
        raise KeyboardInterrupt

    monkeypatch.setattr("lemoncrow.gateway.cli.commands.servicectl.subprocess.run", fake_run)

    res = _invoke(root, "servicectl", "run", "--interval-seconds", "1")
    assert res.exit_code == 0, res.output
    # Live controller: its own pid was on file while the loop ran...
    assert seen == [os.getpid()]
    # ...and reaped on exit so a later start is not blocked by a stale entry.
    assert _read_servicectl_pid(root) is None


def _fake_run_empty_queue(cmd: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
    """Keep the tick hermetic: no real import/index/prune/drain subprocesses."""
    if "run-once" in cmd:
        return subprocess.CompletedProcess(cmd, 0, stdout=b'{"processed": false}', stderr=b"")
    return subprocess.CompletedProcess(cmd, 0, stdout=b"{}", stderr=b"")


def test_tick_pushes_usage_to_the_account(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The controller is the always-on account-sync channel.

    Before this, the only background pusher was the savings reconciler thread
    started by the API service -- which lives inside the OPTIONAL visualization
    stack. An install whose stack was down (see the release frontend crash-loop)
    had no automated channel at all, so lemoncrow.com/savings only advanced via
    the forced push inside ``lc account login``.
    """
    from lemoncrow.infra.runtime import servicectl_lifecycle as svc

    monkeypatch.setattr(svc.subprocess, "run", _fake_run_empty_queue)
    seen: list[Path] = []
    monkeypatch.setattr(
        "lemoncrow.core.capabilities.licensing.usage_report.maybe_report_usage",
        lambda target, **_kw: bool(seen.append(Path(target))) or True,
    )

    svc._servicectl_tick(tmp_path, maintenance_interval_seconds=300, session_import_interval_seconds=3600)

    assert seen == [tmp_path]
    state = svc._read_servicectl_state(tmp_path)
    assert state["last_usage_report_at"] == state["last_tick_at"]
    # Surfaced by `servicectl status` so a stale savings page is diagnosable.
    assert svc._servicectl_status_payload(tmp_path)["last_usage_report_at"] == state["last_tick_at"]


def test_tick_survives_a_failing_usage_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An offline host must still complete its tick and persist state."""
    from lemoncrow.infra.runtime import servicectl_lifecycle as svc

    monkeypatch.setattr(svc.subprocess, "run", _fake_run_empty_queue)

    def boom(*_args: object, **_kw: object) -> bool:
        raise OSError("network down")

    monkeypatch.setattr("lemoncrow.core.capabilities.licensing.usage_report.maybe_report_usage", boom)

    svc._servicectl_tick(tmp_path, maintenance_interval_seconds=300, session_import_interval_seconds=3600)

    state = svc._read_servicectl_state(tmp_path)
    assert state["last_tick_at"]
    assert state["last_usage_report_at"] is None
