"""Personalized PageRank helpers for repo maps."""

from __future__ import annotations

from lemoncrow.core.foundation._graph import DiGraph, pagerank


def personalized_pagerank(
    graph: DiGraph,
    seed_files: list[str],
    *,
    alpha: float = 0.85,
) -> dict[str, float]:
    if not graph.nodes:
        return {}
    seeds = [seed for seed in seed_files if seed in graph]
    if not seeds:
        seeds = list(graph.nodes)[:1]
    personalization = {node: 0.0 for node in graph.nodes}
    for seed in seeds:
        personalization[seed] = 1.0 / len(seeds)
    return pagerank(graph, alpha=alpha, personalization=personalization, weight="weight")


__all__ = ["personalized_pagerank"]
