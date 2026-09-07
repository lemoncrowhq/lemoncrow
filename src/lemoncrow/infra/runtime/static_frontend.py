"""Serve the release-shipped, prebuilt dashboard bundle with the stdlib only.

The release tarball installs ``~/.lemoncrow/install/frontend`` as a **built**
Vite bundle (``index.html`` + ``assets/``, no ``package.json`` and no
``src/``). The stack supervisor's dev path -- ``npm exec vite`` inside a source
checkout -- cannot start that, and a release install has no reason to carry a
node toolchain, so the frontend service crash-looped on every release box.

Two things the Vite dev server did that a plain file server does not, and that
this module therefore reimplements:

* **SPA history fallback** -- the UI mounts a ``BrowserRouter``, so ``/map``
  must serve ``index.html`` rather than 404.
* **``/api`` proxying** -- the bundle fetches relative ``/api/...`` URLs and
  relied on Vite's proxy to strip that prefix and forward to the service port.

Requests are proxied fully buffered; the dashboard uses no SSE or WebSocket
endpoints.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

DEFAULT_API_URL = "http://localhost:8787"
API_PREFIX = "/api"
_PROXY_TIMEOUT_SECONDS = 120

# Never forwarded in either direction (RFC 9110 connection-specific headers).
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


def _make_handler(directory: Path, api_url: str) -> type[SimpleHTTPRequestHandler]:
    target_base = api_url.rstrip("/")

    class _BundleHandler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        # ---- routing ---------------------------------------------------- #

        def _is_api(self) -> bool:
            return self.path == API_PREFIX or self.path.startswith(API_PREFIX + "/")

        def _bundle_has_target(self) -> bool:
            # translate_path() already drops the query string and fragment.
            candidate = Path(self.translate_path(self.path))
            if candidate.is_dir():
                return (candidate / "index.html").exists()
            return candidate.exists()

        def _serve_static(self, *, head: bool) -> None:
            if not self._bundle_has_target():
                self.path = "/index.html"  # SPA history fallback
            if head:
                super().do_HEAD()
            else:
                super().do_GET()

        # ---- proxy ------------------------------------------------------ #

        def _proxy(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else None
            request = urllib.request.Request(
                target_base + self.path[len(API_PREFIX) :],
                data=body,
                method=self.command,
            )
            for key, value in self.headers.items():
                if key.lower() in _HOP_BY_HOP or key.lower() in {"host", "content-length"}:
                    continue
                request.add_header(key, value)
            try:
                with urllib.request.urlopen(request, timeout=_PROXY_TIMEOUT_SECONDS) as response:
                    status, headers, payload = response.status, response.headers, response.read()
            except urllib.error.HTTPError as exc:
                status, headers, payload = exc.code, exc.headers, exc.read()
            except (urllib.error.URLError, OSError) as exc:
                self.send_error(502, f"LemonCrow service unreachable at {target_base}: {exc}")
                return
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() in _HOP_BY_HOP or key.lower() == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        # ---- verbs ------------------------------------------------------ #

        def do_GET(self) -> None:
            self._proxy() if self._is_api() else self._serve_static(head=False)

        def do_HEAD(self) -> None:
            self._proxy() if self._is_api() else self._serve_static(head=True)

        def _api_only(self) -> None:
            if self._is_api():
                self._proxy()
            else:
                self.send_error(405, "only /api accepts this method")

        do_POST = _api_only
        do_PUT = _api_only
        do_PATCH = _api_only
        do_DELETE = _api_only
        do_OPTIONS = _api_only

    return _BundleHandler


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(directory: Path, *, host: str, port: int, api_url: str = DEFAULT_API_URL) -> None:
    """Block serving ``directory`` on ``host:port``, proxying ``/api`` to ``api_url``."""
    with _Server((host, port), _make_handler(directory, api_url)) as httpd:
        httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve LemonCrow's prebuilt dashboard bundle.")
    parser.add_argument("--dir", required=True, help="Directory holding the built bundle.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=3125)
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="LemonCrow service base URL.")
    args = parser.parse_args(argv)

    directory = Path(args.dir).expanduser().resolve()
    if not (directory / "index.html").exists():
        print(f"no built frontend bundle in {directory} (index.html missing)", file=sys.stderr)
        return 1
    print(
        f"[stack] serving prebuilt frontend {directory} on {args.host}:{args.port} (api -> {args.api_url})",
        flush=True,
    )
    try:
        serve(directory, host=args.host, port=args.port, api_url=args.api_url)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
