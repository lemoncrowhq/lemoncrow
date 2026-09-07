"""G15 -- doc-vs-code drift / design-gap detection.

Cross-references design docs (Markdown) against the live symbol index built by
:class:`~lemoncrow.pro.capabilities.semantic_file_memory.SemanticFileMemoryCapability`.

Two read-only analyses:

* :func:`design_gaps` -- identifiers referenced in docs (inline ``backtick``
  spans + fenced code blocks) that do NOT exist anywhere in the symbol index.
  These are stale or aspirational doc references.
* :func:`verify_design` -- doc-referenced symbols whose *signature* in the index
  differs from the signature the doc shows (drift). Only reported when the doc
  itself carries a parenthesised signature to compare against, so prose mentions
  never produce false drift.

Conservative by construction: only tokens that look like real code identifiers
(snake_case with an underscore, dotted paths, CamelCase, or call syntax
``name(...)``) are considered, and a single English word with no code shape is
ignored. The index is the source of truth -- if a symbol is absent from the
index it is reported as a gap, never silently treated as present.

Fail-open: any unexpected error logs and yields an empty findings list.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lemoncrow.pro.capabilities.semantic_file_memory import (
        SemanticFileMemoryCapability,
    )

logger = logging.getLogger(__name__)

# Inline code span: `foo`, `Foo.bar`, `do_thing()`.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
# Fenced code block: ```lang\n ... \n``` (captures the body).
_FENCED_BLOCK = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
# A plausible code identifier reference, optionally dotted, optionally a call.
# Requires at least one underscore, dot, or an interior uppercase letter so a
# bare English word ("the", "design") is never treated as a symbol.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_CALL_SUFFIX = re.compile(r"\s*\(([^)]*)\)")

# Tokens that are common prose / Markdown noise even though they pass the loose
# identifier regex -- never treated as code symbols.
_STOPWORDS = frozenset(
    {
        "true",
        "false",
        "none",
        "null",
        "self",
        "cls",
        "todo",
        "note",
        "warning",
        "example",
        "http",
        "https",
    }
)


def _looks_like_symbol(token: str) -> bool:
    """True only for tokens with genuine code *shape* (not bare prose words)."""
    if not token or token.lower() in _STOPWORDS:
        return False
    if token[0].isdigit():
        return False
    has_underscore = "_" in token
    has_dot = "." in token
    # Interior uppercase => CamelCase / acronym mid-word (e.g. ``FooBar``).
    interior_upper = any(c.isupper() for c in token[1:])
    return has_underscore or has_dot or interior_upper


def _leaf_name(token: str) -> str:
    """Last dotted component, e.g. ``mod.Class.method`` -> ``method``."""
    return token.rsplit(".", 1)[-1]


class _DocReference:
    """A single candidate code reference extracted from a doc."""

    __slots__ = ("doc", "doc_signature", "leaf", "line", "token")

    def __init__(self, *, doc: str, line: int, token: str, doc_signature: str | None) -> None:
        self.doc = doc
        self.line = line
        self.token = token
        self.leaf = _leaf_name(token)
        self.doc_signature = doc_signature


def _strip_fenced_blocks(text: str) -> str:
    """Remove fenced code-block *bodies* so inline scanning sees prose only."""
    return _FENCED_BLOCK.sub(lambda _m: "\n", text)


def _extract_references(doc_path: str, text: str) -> list[_DocReference]:
    """Collect plausible code references with 1-based line numbers.

    References come from two sources: inline ``backtick`` spans and the bodies of
    fenced code blocks. Each is filtered through :func:`_looks_like_symbol`.
    """
    refs: list[_DocReference] = []
    lines = text.splitlines()

    # Inline spans -- scan prose (fenced bodies stripped to avoid double counting
    # with the fenced pass below) line by line for accurate line numbers.
    prose = _strip_fenced_blocks(text)
    for idx, line in enumerate(prose.splitlines(), start=1):
        for match in _INLINE_CODE.finditer(line):
            span = match.group(1).strip()
            _collect_from_span(refs, doc=doc_path, line=idx, span=span)

    # Fenced code blocks -- attribute each identifier to its source line by
    # re-locating block bodies in the original text.
    for block in _FENCED_BLOCK.finditer(text):
        body = block.group(1)
        body_start = text.count("\n", 0, block.start(1)) + 1
        for offset, raw in enumerate(body.splitlines()):
            line_no = body_start + offset
            _collect_from_code_line(refs, doc=doc_path, line=line_no, code=raw)

    # Defensive: keep line numbers within file bounds.
    max_line = max(len(lines), 1)
    return [r for r in refs if 1 <= r.line <= max_line]


def _collect_from_span(refs: list[_DocReference], *, doc: str, line: int, span: str) -> None:
    """Pull a symbol (and any call signature) out of one inline backtick span."""
    ident_match = _IDENT.match(span)
    if ident_match is None:
        return
    token = ident_match.group(0)
    if not _looks_like_symbol(token):
        return
    rest = span[ident_match.end() :]
    call = _CALL_SUFFIX.match(rest)
    doc_sig = call.group(1).strip() if call is not None else None
    refs.append(_DocReference(doc=doc, line=line, token=token, doc_signature=doc_sig))


def _collect_from_code_line(refs: list[_DocReference], *, doc: str, line: int, code: str) -> None:
    """Pull identifier references out of one fenced code-block line.

    Only ``name(...)`` call sites and dotted/CamelCase/underscore identifiers are
    captured; bare loop variables and keywords are skipped by the shape filter.
    """
    for match in _IDENT.finditer(code):
        token = match.group(0)
        if not _looks_like_symbol(token):
            continue
        rest = code[match.end() :]
        call = _CALL_SUFFIX.match(rest)
        doc_sig = call.group(1).strip() if call is not None else None
        refs.append(_DocReference(doc=doc, line=line, token=token, doc_signature=doc_sig))


class DocDriftAnalyzer:
    """Cross-reference Markdown design docs against the live symbol index."""

    def __init__(self, capability: SemanticFileMemoryCapability) -> None:
        self._cap = capability
        # name(lower) -> list of indexed symbol records. Built lazily/once.
        self._symbol_table: dict[str, list[dict[str, Any]]] | None = None

    def _symbols_by_name(self) -> dict[str, list[dict[str, Any]]]:
        if self._symbol_table is not None:
            return self._symbol_table
        table: dict[str, list[dict[str, Any]]] = {}
        try:
            for sym in self._cap._symbol_index.all_symbols():
                name = str(sym.get("name", ""))
                if not name:
                    continue
                # Register under the full name AND the leaf component so a doc
                # reference to ``place_order`` resolves the indexed method
                # ``OrderService.place_order`` without a false gap.
                keys = {name.lower(), _leaf_name(name).lower()}
                for key in keys:
                    table.setdefault(key, []).append(sym)
        except Exception:
            logger.exception("Recovered from broad exception handler")
        self._symbol_table = table
        return table

    def _resolve(self, ref: _DocReference) -> list[dict[str, Any]]:
        """Indexed symbol records matching the reference's leaf name (any case)."""
        return self._symbols_by_name().get(ref.leaf.lower(), [])

    def design_gaps(self, doc_paths: list[Path]) -> dict[str, Any]:
        """Report doc references absent from the symbol index (stale/aspirational)."""
        findings: list[dict[str, Any]] = []
        scanned = 0
        checked_refs = 0
        for path in doc_paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            seen: set[tuple[str, int]] = set()
            for ref in _extract_references(str(path), text):
                checked_refs += 1
                if self._resolve(ref):
                    continue
                key = (ref.token, ref.line)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "doc": ref.doc,
                        "line": ref.line,
                        "symbol": ref.token,
                        "kind": "missing_symbol",
                        "detail": f"'{ref.token}' referenced in doc but not found in symbol index",
                    }
                )
        findings.sort(key=lambda f: (str(f["doc"]), int(f["line"]), str(f["symbol"])))
        return {
            "docs_scanned": scanned,
            "references_checked": checked_refs,
            "gap_count": len(findings),
            "gaps": findings,
            "heuristic": True,
            "note": "Conservative: only code-shaped identifiers are checked; prose is ignored.",
        }

    def verify_design(self, doc_paths: list[Path]) -> dict[str, Any]:
        """Report doc-referenced symbols whose signature drifted from the index.

        Only references that themselves carry a parenthesised call signature are
        compared, so a bare symbol mention can never be a false drift. A symbol
        the doc references that is also missing entirely is surfaced as a
        ``missing`` drift kind (it cannot match any signature).
        """
        findings: list[dict[str, Any]] = []
        scanned = 0
        for path in doc_paths:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            for ref in _extract_references(str(path), text):
                if ref.doc_signature is None:
                    continue  # no signature to compare -- not a drift candidate
                matches = self._resolve(ref)
                if not matches:
                    findings.append(
                        {
                            "doc": ref.doc,
                            "line": ref.line,
                            "symbol": ref.token,
                            "kind": "missing",
                            "doc_signature": ref.doc_signature,
                            "detail": f"'{ref.token}(...)' in doc has no matching indexed symbol",
                        }
                    )
                    continue
                if not self._signature_matches(ref.doc_signature, matches):
                    index_sigs = sorted({str(m.get("signature", "")) for m in matches if m.get("signature")})
                    findings.append(
                        {
                            "doc": ref.doc,
                            "line": ref.line,
                            "symbol": ref.token,
                            "kind": "signature_drift",
                            "doc_signature": ref.doc_signature,
                            "index_signatures": index_sigs,
                            "detail": (
                                f"doc shows '{ref.token}({ref.doc_signature})' but index has "
                                f"{index_sigs or ['<no signature>']}"
                            ),
                        }
                    )
        findings.sort(key=lambda f: (str(f["doc"]), int(f["line"]), str(f["symbol"])))
        return {
            "docs_scanned": scanned,
            "drift_count": len(findings),
            "drifts": findings,
            "heuristic": True,
            "note": "Only doc references carrying a call signature are checked for drift.",
        }

    @staticmethod
    def _signature_matches(doc_signature: str, matches: list[dict[str, Any]]) -> bool:
        """True if the doc's parameter names appear in any indexed signature.

        Compares parameter *names* (order-insensitive) rather than exact text so
        that whitespace, type hints, and defaults do not produce false drift.
        Conservative: if the index has no signature recorded at all, we cannot
        prove drift, so we treat it as a match (no false positive).
        """
        doc_params = _param_names(doc_signature)
        for match in matches:
            index_sig = str(match.get("signature", ""))
            if not index_sig:
                return True  # nothing to compare against -- do not claim drift
            index_params = _param_names(index_sig)
            if doc_params <= index_params or index_params <= doc_params:
                return True
        return False


def _param_names(signature: str) -> frozenset[str]:
    """Extract bare parameter names from a signature fragment.

    Accepts either ``a, b: int = 3`` (bare params) or ``def f(a, b)`` style and
    returns the set of leading identifiers, dropping ``self``/``cls``.
    """
    inner = signature
    paren = re.search(r"\(([^)]*)\)", signature)
    if paren is not None:
        inner = paren.group(1)
    names: set[str] = set()
    for part in inner.split(","):
        part = part.strip()
        if not part or part in {"*", "/", "**"}:
            continue
        part = part.lstrip("*")
        ident = _IDENT.match(part)
        if ident is None:
            continue
        name = ident.group(0)
        if name in {"self", "cls"}:
            continue
        names.add(name)
    return frozenset(names)


def _collect_doc_paths(roots: list[Path]) -> list[Path]:
    """Expand roots into a sorted list of Markdown files.

    A root may be a single ``.md`` file or a directory (recursively scanned).
    """
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root.is_file() and root.suffix.lower() in {".md", ".markdown"}:
            key = str(root.resolve())
            if key not in seen:
                seen.add(key)
                out.append(root)
        elif root.is_dir():
            for md in sorted(root.rglob("*.md")):
                key = str(md.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(md)
    return out


def _build_analyzer(
    repo_root: Path,
    lemoncrow_root: Path,
    paths: list[str] | None,
) -> tuple[DocDriftAnalyzer, list[Path]]:
    from lemoncrow.pro.capabilities.semantic_file_memory import SemanticFileMemoryCapability

    cap = SemanticFileMemoryCapability(lemoncrow_root)
    if paths:
        roots = [Path(p) if Path(p).is_absolute() else repo_root / p for p in paths]
    else:
        roots = [repo_root]
    doc_paths = _collect_doc_paths(roots)
    return DocDriftAnalyzer(cap), doc_paths


def design_gaps(
    *,
    repo_root: Path,
    lemoncrow_root: Path,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    """Fail-open entry point for the ``design_gaps`` graph kind."""
    try:
        analyzer, doc_paths = _build_analyzer(repo_root, lemoncrow_root, paths)
        result = analyzer.design_gaps(doc_paths)
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {"docs_scanned": 0, "gap_count": 0, "gaps": [], "heuristic": True, "error": "recovered"}
    result["kind"] = "design_gaps"
    return result


def verify_design(
    *,
    repo_root: Path,
    lemoncrow_root: Path,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    """Fail-open entry point for the ``verify_design`` graph kind."""
    try:
        analyzer, doc_paths = _build_analyzer(repo_root, lemoncrow_root, paths)
        result = analyzer.verify_design(doc_paths)
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {"docs_scanned": 0, "drift_count": 0, "drifts": [], "heuristic": True, "error": "recovered"}
    result["kind"] = "verify_design"
    return result
