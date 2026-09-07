"""Bounded, provenance-tagged call-edge synthesis (N2/N3).

The full dynamic-dispatch + 20+ framework-resolver subsystem is explicitly out
of scope. This module does the *additive, separable* part:

* It recognises a SMALL, fixed set of common indirection patterns and emits
  synthesized edges for them.
* Every edge is tagged ``provenance="heuristic"`` (mirroring the cross_lang
  edge confidence/kind convention) so it is never confused with a static code-intel
  edge.
* It writes NOTHING into ``call_edges`` and is never folded into
  ``callers``/``callees`` traversal. Callers request synthesized edges
  explicitly; default behaviour of every existing tool is byte-for-byte
  unchanged.

Patterns covered (conservative -- only nameable, identifier handlers; inline
anonymous functions/lambdas are skipped because they have no nameable target):

Python:
  1. Flask/Blueprint route   ``@app.route("/path")``        -> ``route:/path``
  2. FastAPI/Flask verb route ``@app.get("/path")``         -> ``GET /path``
  3. Django URLconf          ``path("x/", view_fn)``        -> ``url:x/``
  4. Registry dynamic dispatch ``{"k": handler}`` dict      -> ``dispatch:k``

JS/TS:
  5. Express verb route      ``app.get("/x", handler)``     -> ``GET /x``
  6. NestJS controller       ``@Get("x") method()``         -> ``GET x``
  7. Observer / pub-sub      ``emitter.on("e", h)``,
                             ``bus.subscribe("e", h)``,
                             ``el.addEventListener("e", h)`` -> ``on:e``

Anything outside these patterns is intentionally NOT synthesized; a partial
that never lies is better than a broad one that fabricates edges.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

SynthesizedEdgeKind = Literal[
    "flask_route",
    "http_route",
    "django_route",
    "dispatch",
    "event_handler",
    "nest_route",
]

# HTTP verb methods recognised on Flask/FastAPI decorators and Express calls.
_HTTP_VERBS: frozenset[str] = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
# NestJS controller method decorators map to verbs (subset of the above).
_NEST_VERBS: frozenset[str] = frozenset({"Get", "Post", "Put", "Patch", "Delete", "Head", "Options", "All"})

# Express verb route: ``app.get('/x', handler)`` / ``router.post("/y", handler)``.
# Conservative: handler must be a bare identifier (skips inline ``(req,res)=>``).
_EXPRESS_RE = re.compile(
    r"""\.(?P<verb>get|post|put|patch|delete|head|options)\(\s*"""
    r"""['\"](?P<path>[^'\"]+)['\"]\s*,\s*(?P<handler>[A-Za-z_$][\w$.]*)\s*\)"""
)
# Observer / pub-sub: ``.on('e', h)`` / ``.subscribe('e', h)`` /
# ``.addEventListener('e', h)`` / ``.addListener('e', h)`` with a bare handler.
_OBSERVER_RE = re.compile(
    r"""\.(?P<method>on|subscribe|addEventListener|addListener)\(\s*"""
    r"""['\"](?P<event>[^'\"]+)['\"]\s*,\s*(?P<handler>[A-Za-z_$][\w$]*)\s*\)"""
)
# NestJS controller: ``@Get('x')`` (or ``@Get()``) immediately above
# ``methodName(...)``. Conservative: single-line decorator + method name only.
_NEST_RE = re.compile(
    r"""@(?P<verb>Get|Post|Put|Patch|Delete|Head|Options|All)\(\s*"""
    r"""(?:['\"](?P<path>[^'\"]*)['\"])?\s*\)\s*\n\s*"""
    r"""(?:async\s+)?(?P<handler>[A-Za-z_$][\w$]*)\s*\("""
)


@dataclass(frozen=True)
class SynthesizedEdge:
    """A heuristically-inferred caller->callee edge, clearly labelled as such."""

    caller: str
    callee: str
    kind: SynthesizedEdgeKind
    line: int
    provenance: str = "heuristic"
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "kind": self.kind,
            "line": self.line,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }


def synthesize_edges(source: str, *, language: str) -> list[SynthesizedEdge]:
    """Return synthesized edges for *source*. Empty list on parse failure.

    Pure and side-effect free. ``language`` selects the resolver set; unknown
    languages yield no edges.
    """
    if language == "python":
        return _synthesize_python(source)
    if language in ("javascript", "typescript"):
        return _synthesize_js(source)
    return []


# --------------------------------------------------------------------------- #
# Python resolvers (AST-based; never raise on malformed source).
# --------------------------------------------------------------------------- #
def _synthesize_python(source: str) -> list[SynthesizedEdge]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    edges: list[SynthesizedEdge] = []
    edges.extend(_python_route_edges(tree))
    edges.extend(_python_django_edges(tree))
    edges.extend(_python_dispatch_edges(tree))
    edges.sort(key=lambda e: (e.line, e.caller, e.callee))
    return edges


def _python_route_edges(tree: ast.AST) -> list[SynthesizedEdge]:
    """Flask ``@app.route`` and verb decorators ``@app.get`` / FastAPI ``@app.get``."""
    edges: list[SynthesizedEdge] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            resolved = _route_from_decorator(dec)
            if resolved is None:
                continue
            caller, kind = resolved
            edges.append(SynthesizedEdge(caller=caller, callee=node.name, kind=kind, line=node.lineno))
    return edges


def _route_from_decorator(dec: ast.expr) -> tuple[str, SynthesizedEdgeKind] | None:
    """Return ``(caller, kind)`` for an ``X.route(...)`` or ``X.<verb>(...)`` call."""
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not isinstance(func, ast.Attribute):
        return None
    path = _first_str_arg(dec)
    if path is None:
        return None
    if func.attr == "route":
        return f"route:{path}", "flask_route"
    if func.attr in _HTTP_VERBS:
        return f"{func.attr.upper()} {path}", "http_route"
    return None


def _python_django_edges(tree: ast.AST) -> list[SynthesizedEdge]:
    """Django URLconf: ``path("x/", view)`` / ``re_path(r"^x$", view)``.

    Only a bare callable name (``ast.Name``) view is recorded; class-based view
    ``.as_view()`` calls and inline lambdas are skipped (no single named target).
    """
    edges: list[SynthesizedEdge] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in ("path", "re_path")):
            continue
        if len(node.args) < 2:
            continue
        route = node.args[0]
        view = node.args[1]
        if not (isinstance(route, ast.Constant) and isinstance(route.value, str)):
            continue
        if not isinstance(view, ast.Name):
            continue
        edges.append(
            SynthesizedEdge(
                caller=f"url:{route.value}",
                callee=view.id,
                kind="django_route",
                line=node.lineno,
            )
        )
    return edges


def _python_dispatch_edges(tree: ast.AST) -> list[SynthesizedEdge]:
    """Registry/dispatch dict ``{"name": handler}`` -> ``dispatch:name -> handler``.

    Conservative: only string-literal keys mapped to a bare callable name
    (``ast.Name``) are recorded. Lambda/inline values and non-string keys are
    skipped -- those are not a single nameable callee.
    """
    edges: list[SynthesizedEdge] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not isinstance(value, ast.Name):
                continue
            line = int(getattr(key, "lineno", getattr(node, "lineno", 1)))
            edges.append(
                SynthesizedEdge(
                    caller=f"dispatch:{key.value}",
                    callee=value.id,
                    kind="dispatch",
                    line=line,
                    confidence=0.4,
                )
            )
    return edges


def _first_str_arg(call: ast.Call) -> str | None:
    if not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


# --------------------------------------------------------------------------- #
# JS/TS resolvers (regex-based; conservative, identifier handlers only).
# --------------------------------------------------------------------------- #
def _synthesize_js(source: str) -> list[SynthesizedEdge]:
    edges: list[SynthesizedEdge] = []
    try:
        for match in _EXPRESS_RE.finditer(source):
            verb = match.group("verb").upper()
            path = match.group("path")
            handler = match.group("handler")
            edges.append(
                SynthesizedEdge(
                    caller=f"{verb} {path}",
                    callee=handler,
                    kind="http_route",
                    line=_line_at(source, match.start()),
                )
            )
        for match in _OBSERVER_RE.finditer(source):
            event = match.group("event")
            handler = match.group("handler")
            edges.append(
                SynthesizedEdge(
                    caller=f"on:{event}",
                    callee=handler,
                    kind="event_handler",
                    line=_line_at(source, match.start()),
                )
            )
        for match in _NEST_RE.finditer(source):
            verb = match.group("verb").upper()
            path = match.group("path") or ""
            handler = match.group("handler")
            edges.append(
                SynthesizedEdge(
                    caller=f"{verb} {path}".rstrip(),
                    callee=handler,
                    kind="nest_route",
                    line=_line_at(source, match.start()),
                )
            )
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []
    edges.sort(key=lambda e: (e.line, e.caller, e.callee))
    return edges


def _line_at(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1
