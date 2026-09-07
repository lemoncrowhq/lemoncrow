"""Path-traversal guards on the service API.

Every endpoint here takes an attacker-controlled string that used to reach the
filesystem unvalidated (CodeQL ``py/path-injection``). ``verify_api_key`` is a
no-op unless ``LEMONCROW_REQUIRE_AUTH=true``, so "attacker" means any local
process or any page served from any localhost port.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from lemoncrow.core.foundation.paths import find_session_dir, flat_session_dir, safe_segment, session_dir
from lemoncrow.core.service.api import create_app
from lemoncrow.infra.storage.bundle import build_sqlite_store_bundle

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

FastAPITestClient = pytest.importorskip(
    "fastapi.testclient",
    reason="FastAPI API tests require the api extra",
).TestClient

TRAVERSAL_IDS = [
    "../../../../etc",
    "..",
    ".",
    "/etc/passwd",
    "a/b",
    "a\x00b",
    "",
]


def _client_for(root: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("LEMONCROW_REQUIRE_AUTH", "false")
    return cast("TestClient", FastAPITestClient(create_app(store_root=root)))


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_safe_segment_rejects_traversal(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid session_id"):
        safe_segment(bad, field="session_id")


def test_safe_segment_accepts_real_ids() -> None:
    assert safe_segment("019e4764-8065-7e11-977f-5a208eb55b8c", field="session_id")
    assert safe_segment("claude-019e4764-8065", field="session_id")
    assert safe_segment("run.json", field="session_id")


@pytest.mark.parametrize("bad", TRAVERSAL_IDS)
def test_session_dir_refuses_to_build_an_escaping_path(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        session_dir(tmp_path, "claude", bad)
    # Read-side lookups degrade to "not found" rather than raising.
    assert find_session_dir(tmp_path, bad) is None
    assert flat_session_dir(tmp_path, bad) is None


def test_session_dir_still_builds_the_canonical_layout(tmp_path: Path) -> None:
    built = session_dir(tmp_path, "claude", "019e4764-8065")
    assert built.is_relative_to(tmp_path / "sessions")
    assert built.name == "019e4764-8065"


def test_file_content_endpoint_confines_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    allowed = tmp_path / "project"
    allowed.mkdir()
    (allowed / "inside.txt").write_text("visible", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve", encoding="utf-8")
    # The daemon's own workspace is an allowed root; pin it to `allowed` so the
    # secret beside it is genuinely out of bounds.
    monkeypatch.setenv("LEMONCROW_WORKSPACE_ROOT", str(allowed))
    monkeypatch.setenv("LEMONCROW_FILE_READ_ROOTS", str(allowed))
    client = _client_for(root, monkeypatch)

    ok = client.get("/v1/files/content", params={"path": str(allowed / "inside.txt")})
    assert ok.status_code == 200
    assert ok.text == "visible"

    for escape in (str(secret), str(allowed / ".." / "secret.txt"), "/etc/passwd"):
        denied = client.get("/v1/files/content", params={"path": escape})
        assert denied.status_code == 403, escape


def test_file_content_endpoint_rejects_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    allowed = tmp_path / "project"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve", encoding="utf-8")
    (allowed / "link.txt").symlink_to(secret)
    monkeypatch.setenv("LEMONCROW_WORKSPACE_ROOT", str(allowed))
    monkeypatch.setenv("LEMONCROW_FILE_READ_ROOTS", str(allowed))
    client = _client_for(root, monkeypatch)

    # A prefix check on the unresolved path would let this through.
    assert client.get("/v1/files/content", params={"path": str(allowed / "link.txt")}).status_code == 403


def test_skill_endpoint_rejects_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    client = _client_for(root, monkeypatch)

    for bad in ("..%2f..%2f..%2fintegrations", "../../..", ".."):
        assert client.get(f"/skills/{bad}").status_code == 404, bad


def test_outcomes_endpoint_rejects_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / ".lemoncrow"
    build_sqlite_store_bundle(root).init()
    client = _client_for(root, monkeypatch)

    outside = tmp_path / "sessions" / "evil"
    outside.mkdir(parents=True)
    (outside / "outcomes.json").write_text('{"route_outcomes": [{"leaked": true}]}', encoding="utf-8")

    response = client.get("/v1/outcomes/sessions/..%2f..%2fsessions%2fevil")
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        assert response.json() == []
