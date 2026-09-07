"""Serving the release-shipped prebuilt dashboard bundle (issue #44, cause 2).

The release tarball installs ``~/.lemoncrow/install/frontend`` as built output
with no ``package.json``; the stack used to reject exactly that layout and
crash-loop. These cover the detection switch plus the two behaviours the static
server must reproduce from the Vite dev server: SPA history fallback and
``/api`` proxying.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from lemoncrow.infra.runtime.stack_lifecycle import _stack_frontend_is_prebuilt
from lemoncrow.infra.runtime.static_frontend import _make_handler, _Server


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "frontend"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<html>spa</html>", encoding="utf-8")
    (root / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return root


def _background(server: ThreadingHTTPServer) -> None:
    threading.Thread(target=server.serve_forever, daemon=True).start()


@pytest.fixture
def api_server() -> Iterator[ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # keep pytest output clean
            return

        def _reply(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            sent = self.rfile.read(length).decode() if length else ""
            body = json.dumps({"path": self.path, "method": self.command, "body": sent}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _reply
        do_POST = _reply

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    _background(server)
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def frontend(bundle: Path, api_server: ThreadingHTTPServer) -> Iterator[str]:
    api_url = f"http://127.0.0.1:{api_server.server_address[1]}"
    server = _Server(("127.0.0.1", 0), _make_handler(bundle, api_url))
    _background(server)
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, str]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status, response.read().decode()


def test_prebuilt_detection(bundle: Path) -> None:
    assert _stack_frontend_is_prebuilt(bundle) is True
    (bundle / "package.json").write_text("{}", encoding="utf-8")
    assert _stack_frontend_is_prebuilt(bundle) is False


def test_prebuilt_detection_rejects_empty_dir(tmp_path: Path) -> None:
    assert _stack_frontend_is_prebuilt(tmp_path) is False


def test_serves_bundled_asset(frontend: str) -> None:
    status, body = _get(f"{frontend}/assets/app.js")
    assert status == 200
    assert body == "console.log(1)"


def test_spa_route_falls_back_to_index(frontend: str) -> None:
    status, body = _get(f"{frontend}/map?focus=abc")
    assert status == 200
    assert body == "<html>spa</html>"


def test_api_prefix_is_stripped_and_proxied(frontend: str) -> None:
    status, body = _get(f"{frontend}/api/v1/sessions?limit=2")
    assert status == 200
    assert json.loads(body)["path"] == "/v1/sessions?limit=2"


def test_api_post_body_is_forwarded(frontend: str) -> None:
    request = urllib.request.Request(
        f"{frontend}/api/traces",
        data=b'{"q":1}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode())
    assert payload == {"path": "/traces", "method": "POST", "body": '{"q":1}'}


def test_unreachable_service_returns_502(bundle: Path) -> None:
    # Port 1 is never bound; the proxy must answer rather than hang or crash.
    server = _Server(("127.0.0.1", 0), _make_handler(bundle, "http://127.0.0.1:1"))
    _background(server)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(f"http://127.0.0.1:{server.server_address[1]}/api/health")
        assert excinfo.value.code == 502
    finally:
        server.shutdown()
        server.server_close()


def test_non_api_post_is_rejected(frontend: str) -> None:
    request = urllib.request.Request(f"{frontend}/map", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(request, timeout=10)
    assert excinfo.value.code == 405
