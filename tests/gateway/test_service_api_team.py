from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from lemoncrow.core.service.api import create_app
from lemoncrow.infra.storage.bundle import build_sqlite_store_bundle
from lemoncrow.pro.capabilities.cross_vendor_memory.audit_log import MemoryAuditLog
from lemoncrow.pro.capabilities.cross_vendor_memory.models import AuditEvent

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

FastAPITestClient = pytest.importorskip(
    "fastapi.testclient",
    reason="FastAPI API tests require the api extra",
).TestClient


def _client_for(root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LEMONCROW_REQUIRE_AUTH", "false")
    return cast("TestClient", FastAPITestClient(create_app(store_root=root)))


def _write_done_session(root: Path, user_id: str, *, cost: float) -> None:
    runs = root / "sessions" / "run-1"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "run.json").write_text(
        json.dumps(
            {
                "session_id": "run-1",
                "status": "done",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "agent_settings": {"user_id": user_id},
                "cost": {"total_cost_usd": cost},
                "events": [],
            }
        ),
        encoding="utf-8",
    )


def test_team_api_workspace_invite_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    client = _client_for(root, monkeypatch)
    created = client.post("/v1/team/workspace", json={"name": "Acme", "admin_email": "admin@example.com"})
    assert created.status_code == 200
    invited = client.post(
        "/v1/team/invite", json={"emails": ["member@example.com"], "role": "member", "user_id": "admin@example.com"}
    )
    assert invited.status_code == 200
    _write_done_session(root, "admin@example.com", cost=0.75)

    usage = client.get("/v1/team/usage", params={"user_id": "admin@example.com"})

    assert usage.status_code == 200
    assert usage.json()["total_cost_usd"] == 0.75


def test_governance_and_audit_api_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    client = _client_for(root, monkeypatch)
    created = client.post("/v1/team/workspace", json={"name": "Acme", "admin_email": "admin@example.com"})
    assert created.status_code == 200
    policy = client.post(
        "/v1/governance/policy",
        json={"user_id": "admin@example.com", "policy": {"redaction_rules": [{"pattern": "secret"}]}},
    )
    assert policy.status_code == 200
    MemoryAuditLog(root).append(
        AuditEvent(
            vendor="claude",
            event="added",
            fact_id="fact-1",
            source_file="memory.md",
            source_line=1,
            content="secret value",
        )
    )
    _write_done_session(root, "admin@example.com", cost=0.25)
    # The HTTP endpoints confine out_dir/bundle_dir to the store root -- an
    # unauthenticated local caller must not be able to create directories and
    # write manifests anywhere on the box. The CLI wrappers stay unconstrained.
    bundle_dir = root / "exports" / "bundle"

    exported = client.post("/v1/audit/export", json={"user_id": "admin@example.com", "out_dir": str(bundle_dir)})
    verified = client.post("/v1/audit/verify", json={"bundle_dir": str(bundle_dir)})

    assert exported.status_code == 200
    assert verified.status_code == 200
    assert verified.json()["valid"] is True

    escaped = tmp_path / "outside"
    assert (
        client.post("/v1/audit/export", json={"user_id": "admin@example.com", "out_dir": str(escaped)}).status_code
        == 400
    )
    assert client.post("/v1/audit/verify", json={"bundle_dir": str(escaped)}).status_code == 400
    assert not escaped.exists()
    assert (
        client.post(
            "/v1/audit/export", json={"user_id": "admin@example.com", "out_dir": str(root / ".." / "outside2")}
        ).status_code
        == 400
    )
