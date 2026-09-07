"""Tiered tree-sitter -> LSP call-site resolution (N14).

Architecture (the *selection* logic is the deliverable here, not a live server):

1. Tree-sitter does bulk symbol/edge extraction (already done by
   :mod:`lemoncrow.infra.tree_sitter.tags`). We treat every ``reference`` tag as a
   candidate call site and every ``definition`` tag as a locally-known target.
2. A reference is **resolved locally** when its name matches a tree-sitter
   definition in the same file -> provenance ``tree_sitter``.
3. Everything left is a **RESIDUAL** -- an unqualified call site tree-sitter
   could not pin down. ONLY these residuals are handed to the injectable
   LSP-resolution hook.
4. The hook may return a resolution tagged ``lsp_resolved`` (a definition site)
   or ``lsp_dispatch`` (a dynamic-dispatch candidate). Heuristic synthesized
   edges fold in as ``heuristic``.

Fail-open contract: with no LSP hook (the default), residuals are returned
UNRESOLVED and tree-sitter results are byte-for-byte unchanged. The hook is
pure dependency-injection: a stub satisfies it, so this is fully unit-tested
without any language server.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from lemoncrow.infra.tree_sitter.tags import Tag

ResolutionProvenance = Literal[
    "tree_sitter",
    "lsp_resolved",
    "lsp_dispatch",
    "heuristic",
]

# Tier ordering: lower index = higher precedence (cheaper / more certain first).
_TIER_ORDER: dict[ResolutionProvenance, int] = {
    "tree_sitter": 0,
    "lsp_resolved": 1,
    "lsp_dispatch": 2,
    "heuristic": 3,
}


@dataclass(frozen=True)
class CallSite:
    """An unqualified reference (candidate call site) extracted by tree-sitter."""

    name: str
    file: str
    line: int


@dataclass(frozen=True)
class ResolvedCallSite:
    """A call site annotated with the tier that resolved it."""

    name: str
    file: str
    line: int
    provenance: ResolutionProvenance
    target: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True)
class LspResolution:
    """What an injected LSP hook returns for a single residual call site.

    ``target`` is the resolved definition (e.g. ``"pkg.Type.method"``) or a
    dispatch candidate name. ``dispatch=True`` marks an ambiguous / dynamic
    resolution, tagged ``lsp_dispatch`` instead of ``lsp_resolved``.
    """

    target: str
    dispatch: bool = False
    confidence: float = 0.8


# A resolver maps a residual CallSite -> an LspResolution, or None if it cannot
# resolve it (fail-open: unresolved residuals stay unresolved).
LspResolver = Callable[[CallSite], LspResolution | None]


@dataclass(frozen=True)
class ResolutionReport:
    """Outcome of one tiered resolution pass over a file's call sites."""

    resolved: list[ResolvedCallSite] = field(default_factory=list)
    residual: list[CallSite] = field(default_factory=list)

    def by_provenance(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for site in self.resolved:
            counts[site.provenance] = counts.get(site.provenance, 0) + 1
        if self.residual:
            counts["residual"] = len(self.residual)
        return counts


def compute_residuals(
    tags: Iterable[Tag],
) -> tuple[list[ResolvedCallSite], list[CallSite]]:
    """Split reference tags into (tree-sitter resolved, residual) call sites.

    A reference resolves locally when its name has a matching definition tag in
    the same set; otherwise it is residual. Definition tags never count as call
    sites. Deterministic ordering by (line, name).
    """
    local_defs: set[str] = {t.name for t in tags if t.kind == "definition"}
    resolved: list[ResolvedCallSite] = []
    residual: list[CallSite] = []
    seen: set[tuple[str, str, int]] = set()
    for tag in tags:
        if tag.kind != "reference":
            continue
        key = (tag.name, tag.file, tag.line)
        if key in seen:
            continue
        seen.add(key)
        if tag.name in local_defs:
            resolved.append(
                ResolvedCallSite(
                    name=tag.name,
                    file=tag.file,
                    line=tag.line,
                    provenance="tree_sitter",
                    target=tag.name,
                )
            )
        else:
            residual.append(CallSite(name=tag.name, file=tag.file, line=tag.line))
    resolved.sort(key=lambda r: (r.line, r.name))
    residual.sort(key=lambda c: (c.line, c.name))
    return resolved, residual


def resolve_call_sites(
    tags: Iterable[Tag],
    *,
    lsp_resolver: LspResolver | None = None,
) -> ResolutionReport:
    """Run the full tree-sitter -> LSP tiering over a file's tags.

    ``lsp_resolver`` is invoked ONLY for residual call sites. With no resolver
    (default), residuals are reported unresolved and the tree-sitter tier is
    unchanged -- the strict fail-open guarantee. A resolver that raises is
    swallowed per-site so one bad call site never sinks the pass.
    """
    resolved, residual = compute_residuals(tags)
    if lsp_resolver is None:
        return ResolutionReport(resolved=resolved, residual=residual)

    still_residual: list[CallSite] = []
    for site in residual:
        resolution = _safe_resolve(lsp_resolver, site)
        if resolution is None:
            still_residual.append(site)
            continue
        provenance: ResolutionProvenance = "lsp_dispatch" if resolution.dispatch else "lsp_resolved"
        resolved.append(
            ResolvedCallSite(
                name=site.name,
                file=site.file,
                line=site.line,
                provenance=provenance,
                target=resolution.target,
                confidence=resolution.confidence,
            )
        )
    resolved.sort(key=lambda r: (_TIER_ORDER[r.provenance], r.line, r.name))
    still_residual.sort(key=lambda c: (c.line, c.name))
    return ResolutionReport(resolved=resolved, residual=still_residual)


def _safe_resolve(lsp_resolver: LspResolver, site: CallSite) -> LspResolution | None:
    try:
        return lsp_resolver(site)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        return None
