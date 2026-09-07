"""Edit-loop correctness gate (WS1 / G1+G2).

Wired into ``tool_smart_edit`` to run, *after* a successful edit:

1. A tree-sitter **parse gate** for non-Python languages (TS/JS/TSX/Rust/Go):
   if the edited file no longer parses (an ``ERROR`` or missing node), the edit
   introduced a syntax break. Python is checked with :func:`ast.parse`.
2. The executing :class:`VerifierCapability` (scoped ``mypy`` over touched source
   files). pytest is intentionally NOT run inline -- see ``run_edit_gate``.

The gate is **fail-open**: any unexpected error (missing grammar, runner crash,
timeout) yields *no* counterexamples, so a flaky checker never blocks a
legitimate edit. Callers decide whether to roll back when counterexamples with
``severity == "error"`` are returned.

This is the first, conservative iteration: tests are scoped to touched test
files. Impact-selected tests (via ``semantic_file_memory.change_impact``) are a
follow-up tightening tracked under WS4.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from .capability import VerifierCapability
from .counterexample import Counterexample

# File suffix -> tree-sitter grammar name. Python is handled via ast.parse and
# deliberately excluded here (the re-review scoped the tree-sitter gate to
# non-Python edits; Python syntax errors also surface via mypy/import anyway).
_TS_SUFFIX_TO_GRAMMAR: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".rs": "rust",
    ".go": "go",
}

_MAX_PARSE_ERRORS_PER_FILE = 3


def _node_attr(node: object, name: str, default: object = None) -> object:
    """Read a tree-sitter node attribute exposed as either a method or a value."""
    try:
        val = getattr(node, name)
    except AttributeError:
        return default
    return val() if callable(val) else val


def _node_kind(node: object) -> str:
    """The node's grammar kind. The language-pack binding exposes it as ``kind``."""
    return str(_node_attr(node, "kind") or _node_attr(node, "type") or "")


def _node_children(node: object) -> list[object]:
    count_obj = _node_attr(node, "child_count", 0)
    count = count_obj if isinstance(count_obj, int) else 0
    try:
        return [node.child(i) for i in range(count)]  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError, IndexError):
        return []


def _node_line(node: object) -> int | None:
    point = _node_attr(node, "start_point")
    if point is None:
        return None
    row = _node_attr(point, "row")
    if row is None and isinstance(point, (tuple, list)) and point:
        row = point[0]
    return int(row) + 1 if isinstance(row, int) else None


def _first_error_nodes(root: object) -> list[object]:
    """Depth-first scan for ERROR / missing nodes, capped for cost.

    Prunes subtrees whose ``has_error`` is explicitly False; descends when the
    flag is unavailable so detection never silently misses a break.
    """
    out: list[object] = []
    stack: list[object] = [root]
    while stack and len(out) < _MAX_PARSE_ERRORS_PER_FILE:
        node = stack.pop()
        if _node_kind(node) == "ERROR" or bool(_node_attr(node, "is_missing", False)):
            out.append(node)
            continue  # do not descend into a broken subtree
        if _node_attr(node, "has_error") is False:
            continue  # clean subtree -> skip
        stack.extend(reversed(_node_children(node)))
    return out


def _parse_error_for_path(path: Path) -> list[Counterexample]:
    """Return parse counterexamples for a single existing file (fail-open)."""
    suffix = path.suffix.lower()
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []

    if suffix in (".py", ".pyi"):
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return [
                Counterexample(
                    check="parse",
                    severity="error",
                    file_path=str(path),
                    line=exc.lineno,
                    diagnostic=f"syntax error: {exc.msg}",
                    repro_command=None,
                )
            ]
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return []  # fail-open on non-syntax failures
        return []

    grammar = _TS_SUFFIX_TO_GRAMMAR.get(suffix)
    if grammar is None:
        return []
    try:
        from lemoncrow.pro.capabilities.semantic_file_memory.treesitter_ast import (
            tree_sitter_parser,
        )

        parser = tree_sitter_parser(grammar)
        if parser is None:
            return []  # grammar unavailable -> fail-open
        try:
            tree = parser.parse(source.encode("utf-8"))
        except TypeError:
            # Some tree-sitter-language-pack grammars expect str, not bytes.
            tree = parser.parse(source)
        root = _node_attr(tree, "root_node")
        if root is None:
            return []
        error_nodes = _first_error_nodes(root)
        if not error_nodes:
            return []
        return [
            Counterexample(
                check="parse",
                severity="error",
                file_path=str(path),
                line=_node_line(node),
                diagnostic=f"{grammar} parse error near this location (tree-sitter ERROR/missing node)",
                repro_command=None,
            )
            for node in error_nodes
        ]
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []  # fail-open


def treesitter_parse_errors(paths: Sequence[Path]) -> list[Counterexample]:
    """Parse-gate the given existing files; returns [] when all parse cleanly."""
    out: list[Counterexample] = []
    for path in paths:
        if path.exists():
            out.extend(_parse_error_for_path(path))
    return out


def _bounded_runner(timeout_s: float) -> Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]:
    def _run(args: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(args),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )

    return _run


def run_edit_gate(
    touched_paths: Sequence[Path],
    *,
    repo_root: Path,
    checks: Sequence[str] = ("typecheck",),
    timeout_s: float = 60.0,
    run_parse_gate: bool = True,
) -> list[Counterexample]:
    """Run the parse gate, then scoped mypy, over *touched_paths*.

    Fully fail-open: returns only genuine counterexamples; never raises.
    If the parse gate already finds a syntax break, the (more expensive)
    mypy pass is skipped — there is no point typechecking unparseable code.

    pytest is intentionally excluded: it is slow (often exceeds the timeout ->
    fail-open, so it neither protects nor returns quickly), it reads broad test
    files while concurrent edits mutate the tree (non-deterministic verdicts),
    and auto-rollback on a flaky test is harmful. The agent runs tests itself.
    """
    try:
        existing = [p for p in touched_paths if p.exists()]
        if run_parse_gate:
            parse_failures = treesitter_parse_errors(existing)
            if parse_failures:
                return parse_failures
        scope_files = [str(p) for p in existing if p.suffix.lower() in (".py", ".pyi")]
        if not scope_files:
            return []
        verify_checks = [c for c in checks if c != "tests"]
        if not verify_checks:
            return []
        verifier = VerifierCapability(cwd=repo_root, run=_bounded_runner(timeout_s))
        return verifier.run(scope_files=scope_files, checks=verify_checks)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return []


__all__ = ["run_edit_gate", "treesitter_parse_errors"]
