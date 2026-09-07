"""Bridge the G14 LSP client into the N14 residual-resolution hook.

This lives in ``core`` (which may depend on ``infra``) and wires the infra-side
:class:`~lemoncrow.infra.code_intel.lsp.client.LspClient` into the N14 tiering's
injectable ``LspResolver``, so the two workstreams compose:

    resolver = build_lsp_resolver(client, locate=...)
    report = resolve_call_sites(tags, lsp_resolver=resolver)

Strictly opt-in and fail-open: an unavailable client yields a resolver that
returns ``None`` for every site (everything stays residual; the caller keeps
its tree-sitter result).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from lemoncrow.infra.code_intel.lsp.client import LspClient
from lemoncrow.pro.capabilities.code_context.edge_resolution import (
    CallSite,
    LspResolution,
    LspResolver,
)

# Maps a residual CallSite to a zero-based (line, character) position in its
# file. Tree-sitter tags are one-based by line; column data is not carried on a
# CallSite, so the default locator targets column 0 of the call line. Callers
# with richer position data inject a more precise locator.
PositionLocator = Callable[[CallSite], tuple[int, int]]


def _default_locator(site: CallSite) -> tuple[int, int]:
    # CallSite.line is one-based (tree-sitter convention); LSP is zero-based.
    return max(site.line - 1, 0), 0


def _uri_for(file: str) -> str:
    try:
        return Path(file).resolve().as_uri()
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return f"file://{file}"


def build_lsp_resolver(
    client: LspClient,
    *,
    locate: PositionLocator | None = None,
    confidence: float = 0.8,
) -> LspResolver:
    """Return an N14 ``LspResolver`` backed by *client*.

    The resolver calls ``textDocument/definition`` for each residual call site
    and reports the first resolved definition as an ``lsp_resolved`` candidate;
    multiple definitions (overloads / dynamic dispatch) are reported as
    ``lsp_dispatch``. No definitions -> ``None`` (site stays residual).

    Fail-open: if the client is unavailable the resolver short-circuits to
    ``None`` without ever touching the transport.
    """
    locator = locate or _default_locator

    def resolve(site: CallSite) -> LspResolution | None:
        if not client.available:
            return None
        line, character = locator(site)
        locations = client.definition(_uri_for(site.file), line, character)
        if not locations:
            return None
        target = locations[0].uri
        return LspResolution(
            target=target,
            dispatch=len(locations) > 1,
            confidence=confidence,
        )

    return resolve
