"""Symbol-level call-graph centrality (G6).

File-level PageRank already exists in ``repo_map/pagerank.py``; this module is
the genuinely-missing *symbol* / call-graph level. It answers "which symbols
are most important here?" by ranking nodes of the call graph rather than files.

Input is the set of directed call edges already persisted in the engine's
``call_edges`` table -- ``(caller, callee)`` name pairs. We do NOT reinvent
graph storage; the caller passes edges read straight from that table.

Metrics (cheap, deterministic, dependency-free):

* ``in_degree`` / ``out_degree`` / ``degree`` centrality (normalised by N-1).
* ``eigenvector`` -- damped, out-degree-normalized directed PageRank (standard
  formulation, with dangling-node mass redistribution). Field name kept as
  ``eigenvector`` for output-shape compatibility with existing consumers.

Why PageRank and not a plain eigenvector/degree walk: an earlier undirected,
un-normalized power iteration spread each node's score to EVERY neighbour
regardless of how many other neighbours that node had -- a caller with 2 call
sites and a caller with 2,000 call sites contributed the SAME total score to a
shared callee. On a real C codebase that ranked ubiquitous utility functions
(``spin_unlock``, ``IS_ERR``, ``kfree`` -- called from everywhere) as the most
"central" symbols in the repo, which is exactly backwards: being called from
everywhere is what should be normalized away, not rewarded twice. Directed,
out-degree-normalized PageRank fixes this at the algorithm level: a caller's
vote is split across ITS OWN out-edges, so high-fan-out callers (which say
little about any one callee) contribute less per edge than focused callers.

The whole computation is O(iterations x edges); ``max_iterations`` is small and
fixed, so it stays cheap even on large graphs.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Edge = (caller_name, callee_name). Names are qualified-or-short identifiers as
# stored in call_edges; identity is by name (the call graph is name-resolved).
CallEdge = tuple[str, str]


def compute_call_graph_centrality(
    edges: Iterable[CallEdge],
    *,
    limit: int = 50,
    max_iterations: int = 50,
    tolerance: float = 1.0e-6,
    damping: float = 0.85,
) -> dict[str, Any]:
    """Rank call-graph symbols by importance.

    Returns a dict with ``node_count``, ``edge_count`` and a ``ranking`` list
    (most central first) of ``{symbol, in_degree, out_degree, degree,
    eigenvector}`` entries, truncated to ``limit``. Ordering is total and
    deterministic: by descending eigenvector, then descending total degree,
    then symbol name -- so equal-score graphs always rank identically.
    """
    in_deg: dict[str, int] = {}
    out_deg: dict[str, int] = {}
    out_edges: dict[str, list[str]] = {}
    nodes_set: set[str] = set()
    edge_count = 0
    for caller, callee in edges:
        if not caller or not callee or caller == callee:
            continue
        edge_count += 1
        out_deg[caller] = out_deg.get(caller, 0) + 1
        in_deg[callee] = in_deg.get(callee, 0) + 1
        nodes_set.add(caller)
        nodes_set.add(callee)
        out_edges.setdefault(caller, []).append(callee)

    nodes = sorted(nodes_set)
    node_count = len(nodes)
    if node_count == 0:
        return {"node_count": 0, "edge_count": 0, "ranking": [], "truncated": False}

    rank = _pagerank(nodes, out_edges, max_iterations=max_iterations, tolerance=tolerance, damping=damping)
    norm = float(node_count - 1) if node_count > 1 else 1.0

    ranking: list[dict[str, Any]] = []
    for symbol in nodes:
        ind = in_deg.get(symbol, 0)
        outd = out_deg.get(symbol, 0)
        ranking.append(
            {
                "symbol": symbol,
                "in_degree": ind,
                "out_degree": outd,
                "degree": round((ind + outd) / norm, 4),
                "in_degree_centrality": round(ind / norm, 4),
                "out_degree_centrality": round(outd / norm, 4),
                "eigenvector": round(rank.get(symbol, 0.0), 6),
            }
        )
    ranking.sort(
        key=lambda r: (
            -float(r["eigenvector"]),
            -(int(r["in_degree"]) + int(r["out_degree"])),
            str(r["symbol"]),
        )
    )
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "ranking": ranking[:limit],
        "truncated": node_count > limit,
    }


def _pagerank(
    nodes: list[str],
    out_edges: dict[str, list[str]],
    *,
    max_iterations: int,
    tolerance: float,
    damping: float,
) -> dict[str, float]:
    """Standard directed PageRank with dangling-node mass redistribution.

    Score flows caller -> callee, split EVENLY across a caller's own out-edges
    (a caller with 50 call sites contributes far less per edge than one with 2
    -- the out-degree normalization the old undirected walk never had, which
    let pure fan-in dominate regardless of who was calling). A node with no
    out-edges (a leaf callee, or a callee never itself observed calling
    anything) is "dangling": its score is redistributed uniformly across ALL
    nodes each iteration rather than lost -- the standard PageRank fix, else
    rank mass leaks out of the graph and iteration never converges to a stable
    total. L1 mass is conserved every iteration (sums to ~1 over all nodes),
    so results are comparable across repos of different sizes without a
    separate max-normalization step.
    """
    n = len(nodes)
    if n == 0:
        return {}
    score: dict[str, float] = dict.fromkeys(nodes, 1.0 / n)
    base = (1.0 - damping) / n
    dangling = [node for node in nodes if not out_edges.get(node)]
    dangling_set = set(dangling)
    # Redistribution TARGETS exclude the dangling nodes themselves: a leaf that
    # calls nothing (a pure sink -- an absorbing trap for the standard random-
    # surfer model) would otherwise recycle a slice of its OWN just-leaked mass
    # back to itself every iteration, self-reinforcing. Verified analytically on
    # a 5-node toy graph (3 distinct callers -> hub -> 1 leaf) that including
    # sinks in the redistribution target set lets the leaf's fixed-point rank
    # OVERTAKE the hub's (0.380 vs 0.336) purely from this self-credit, even
    # though the hub has 3x the leaf's in-degree and both its callers count --
    # exactly backwards for code-navigation importance. The random-jump term
    # (``base``) still reaches every node uniformly, unchanged; only a sink's
    # OWN leaked mass is redirected to the rest of the graph instead of partly
    # back to itself (and other sinks).
    redistribution_targets = [node for node in nodes if node not in dangling_set] or nodes
    for _ in range(max(1, max_iterations)):
        nxt: dict[str, float] = dict.fromkeys(nodes, base)
        dangling_mass = sum(score[node] for node in dangling)
        if dangling_mass:
            share = damping * dangling_mass / len(redistribution_targets)
            for node in redistribution_targets:
                nxt[node] += share
        for node in nodes:
            callees = out_edges.get(node)
            if not callees:
                continue
            contrib = damping * score[node] / len(callees)
            for callee in callees:
                nxt[callee] += contrib
        delta = sum(abs(nxt[node] - score[node]) for node in nodes)
        score = nxt
        if delta < tolerance:
            break
    return score
