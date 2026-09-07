"""Agent-facing graph analytics over the semantic file-dependency graph.

All analytics here are pure functions over the data already captured by
:class:`~lemoncrow.pro.capabilities.semantic_file_memory.indexer.FileIndex`
(per-file ``dependency_map``, ``symbol_details``, ``complexity_score``,
``exports``) plus the reverse-dependency graph it derives. No new graph
storage is introduced -- this module only *reads* the existing index.

The four file-level analytics are:

* ``blast_radius`` -- reverse-dependency closure + affected tests + risk tier
  (delegates to the pre-existing ``SymbolIndex.change_impact``).
* ``dead_code``    -- files with no inbound importers that are not tests or
  obvious entrypoints, ranked by complexity (most code wasted first).
* ``cycles``       -- import dependency cycles via Tarjan strongly-connected
  components over the forward dependency graph.
* ``coupling``     -- afferent/efferent coupling and the instability metric
  ``I = Ce / (Ca + Ce)`` per file (Martin's instability).

Symbol/call-graph *centrality* (G6) is intentionally NOT here: it requires the
The ``call_edges`` graph owned by ``CodeContextEngine`` and lives there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .search import SymbolIndex

# Files matching these markers are never reported as dead code: removing them
# would break test discovery or program entry, so "no importer" is expected.
_ENTRYPOINT_BASENAMES = frozenset(
    {
        "__init__.py",
        "__main__.py",
        "conftest.py",
        "setup.py",
        "manage.py",
        "main.py",
        "app.py",
        "cli.py",
        "wsgi.py",
        "asgi.py",
    }
)


def _is_test_path(path: str) -> bool:
    # Component-based, not a loose substring match: a bare ``"/test" in path``
    # wrongly flags any path that merely contains "test" (e.g. pytest tmp dirs
    # like ``/test_run0/`` or a project dir named ``/test-app/``), which would
    # silently drop real files from dead-code analysis.
    norm = path.replace("\\", "/").lower()
    parts = norm.split("/")
    base = parts[-1]
    return (
        any(part in {"test", "tests", "__tests__", "spec", "specs"} for part in parts[:-1])
        or base.startswith("test_")
        or base.endswith("_test.py")
        or ".test." in base
        or ".spec." in base
    )


def _is_entrypoint(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return base in _ENTRYPOINT_BASENAMES


class GraphAnalytics:
    """Read-only analytics over a :class:`SymbolIndex`'s file graph."""

    def __init__(self, symbol_index: SymbolIndex) -> None:
        self._symbol_index = symbol_index
        # FileIndex (private on SymbolIndex by construction); read-only use.
        self._index = symbol_index._index

    # ------------------------------------------------------------------
    # blast_radius (G3) -- delegate to the existing change_impact
    # ------------------------------------------------------------------

    def blast_radius(self, modified_path: str, *, max_transitive_depth: int = 3) -> dict[str, Any]:
        """Reverse-dependency closure + affected tests + risk tier for one file."""
        return self._symbol_index.change_impact(modified_path, max_transitive_depth=max_transitive_depth)

    # ------------------------------------------------------------------
    # dead_code (G3)
    # ------------------------------------------------------------------

    def dead_code(self, *, limit: int = 50) -> dict[str, Any]:
        """Report indexed files with no inbound importers (likely dead modules).

        A file is flagged when nothing in the index imports it AND it is neither
        a test file nor an obvious entrypoint. Results are ranked by total
        complexity (the most code that would be removed) so high-value cleanups
        surface first. Conservative by construction: an importer that lives
        outside the index (e.g. an unindexed entrypoint) will keep a file off
        the list, so this never over-reports.
        """
        entries = self._index.all_entries()
        rdeps = self._index.build_reverse_deps()
        candidates: list[dict[str, Any]] = []
        for path, entry in entries.items():
            if _is_test_path(path) or _is_entrypoint(path):
                continue
            if rdeps.get(path):
                continue  # something imports it
            exports = list(entry.get("exports", []))
            if not exports and not entry.get("symbols"):
                continue  # empty / data file -- not actionable dead code
            complexity = int(entry.get("complexity_score", 0))
            candidates.append(
                {
                    "path": path,
                    "language": entry.get("language"),
                    "exports": exports[:20],
                    "complexity_score": complexity,
                    "lines_total": int(entry.get("lines_total", 0)),
                }
            )
        candidates.sort(key=lambda c: (-int(c["complexity_score"]), str(c["path"])))
        return {
            "analyzed_files": len(entries),
            "dead_file_count": len(candidates),
            "dead_files": candidates[:limit],
            "truncated": len(candidates) > limit,
        }

    # ------------------------------------------------------------------
    # cycles (G3) -- Tarjan SCC over the forward import graph
    # ------------------------------------------------------------------

    def cycles(self, *, limit: int = 50) -> dict[str, Any]:
        """Find import dependency cycles (strongly-connected components, size >= 2).

        Builds the forward graph from each file's ``dependency_map`` (resolved
        local imports) restricted to indexed files, then runs an iterative
        Tarjan SCC so deeply-nested graphs cannot overflow the recursion stack.
        """
        entries = self._index.all_entries()
        graph: dict[str, list[str]] = {}
        for path, entry in entries.items():
            deps = [d for d in entry.get("dependency_map", []) if d in entries and d != path]
            graph[path] = sorted(set(deps))
        sccs = _tarjan_scc(graph)
        cycles = [sorted(scc) for scc in sccs if len(scc) >= 2]
        # Largest, most tangled cycles first; deterministic tie-break by member.
        cycles.sort(key=lambda c: (-len(c), c[0] if c else ""))
        return {
            "analyzed_files": len(entries),
            "cycle_count": len(cycles),
            "cycles": cycles[:limit],
            "truncated": len(cycles) > limit,
        }

    # ------------------------------------------------------------------
    # coupling (G3) -- afferent/efferent + Martin's instability
    # ------------------------------------------------------------------

    def coupling(self, *, limit: int = 50) -> dict[str, Any]:
        """Per-file afferent/efferent coupling and instability ``I = Ce/(Ca+Ce)``.

        * ``afferent`` (Ca)  -- number of indexed files that import this one.
        * ``efferent`` (Ce)  -- number of indexed files this one imports.
        * ``instability``    -- 0.0 (maximally stable, only depended upon) ..
          1.0 (maximally unstable, only depends on others). Files with both
          high Ca and high Ce are coupling hotspots that are hard to change.
        Ranked by total coupling (Ca + Ce) so the riskiest modules surface first.
        """
        entries = self._index.all_entries()
        rdeps = self._index.build_reverse_deps()
        rows: list[dict[str, Any]] = []
        for path, entry in entries.items():
            efferent = len({d for d in entry.get("dependency_map", []) if d in entries and d != path})
            afferent = len({d for d in rdeps.get(path, []) if d in entries and d != path})
            total = afferent + efferent
            if total == 0:
                continue
            instability = efferent / total
            rows.append(
                {
                    "path": path,
                    "afferent": afferent,
                    "efferent": efferent,
                    "total_coupling": total,
                    "instability": round(instability, 4),
                }
            )
        rows.sort(key=lambda r: (-int(r["total_coupling"]), str(r["path"])))
        return {
            "analyzed_files": len(entries),
            "coupled_file_count": len(rows),
            "files": rows[:limit],
            "truncated": len(rows) > limit,
        }

    # ------------------------------------------------------------------
    # topology (G17) -- module-boundary + god-module discovery
    # ------------------------------------------------------------------

    def topology(self, *, limit: int = 50) -> dict[str, Any]:
        """Cluster files into modules (by directory) and surface module topology.

        Fuses the file import graph into module-level edges so the hidden module
        structure is visible: which modules depend on which, and which files are
        coupling hotspots (god-module candidates, reusing ``coupling``). Pure and
        read-only over the existing index.
        """
        entries = self._index.all_entries()

        def _module_of(path: str) -> str:
            norm = path.replace("\\", "/")
            return norm.rsplit("/", 1)[0] if "/" in norm else "."

        efferent: dict[str, set[str]] = {}
        file_counts: dict[str, int] = {}
        for path, entry in entries.items():
            module = _module_of(path)
            file_counts[module] = file_counts.get(module, 0) + 1
            for dep in entry.get("dependency_map", []):
                if dep in entries and dep != path:
                    dep_module = _module_of(dep)
                    if dep_module != module:
                        efferent.setdefault(module, set()).add(dep_module)
        afferent: dict[str, set[str]] = {}
        for module, deps in efferent.items():
            for dep_module in deps:
                afferent.setdefault(dep_module, set()).add(module)

        modules: list[dict[str, Any]] = []
        for module in sorted(file_counts):
            eff = sorted(efferent.get(module, set()))
            modules.append(
                {
                    "module": module,
                    "files": file_counts[module],
                    "depends_on": eff,
                    "efferent_modules": len(eff),
                    "afferent_modules": len(afferent.get(module, set())),
                }
            )
        # Most-connected modules first (likely architectural hubs / god-modules).
        modules.sort(key=lambda r: (-(int(r["efferent_modules"]) + int(r["afferent_modules"])), str(r["module"])))
        hotspots = self.coupling(limit=limit)["files"][:10]
        return {
            "analyzed_files": len(entries),
            "module_count": len(file_counts),
            "modules": modules[:limit],
            "hotspots": hotspots,
            "truncated": len(file_counts) > limit,
        }


def _tarjan_scc(graph: dict[str, list[str]]) -> list[list[str]]:
    """Iterative Tarjan strongly-connected components.

    Stack-safe (no recursion) so pathological deep graphs cannot blow up.
    Returns each SCC as a list of node ids; node order within an SCC is the
    order discovered (callers sort for determinism where it matters).
    """
    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []

    for root in graph:
        if root in indices:
            continue
        # work item: (node, iterator-position into its neighbours)
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, pos = work[-1]
            if pos == 0:
                indices[node] = index_counter
                lowlink[node] = index_counter
                index_counter += 1
                stack.append(node)
                on_stack.add(node)
            neighbours = graph.get(node, [])
            recursed = False
            while pos < len(neighbours):
                nxt = neighbours[pos]
                pos += 1
                if nxt not in indices:
                    work[-1] = (node, pos)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], indices[nxt])
            if recursed:
                continue
            # done exploring `node`
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == indices[node]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == node:
                        break
                result.append(component)
    return result
