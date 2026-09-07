"""High-level repo map builder."""

from __future__ import annotations

from pathlib import Path

from lemoncrow.core.capabilities.repo_map_contract import RepoMapResult as RepoMapResult
from lemoncrow.pro.capabilities.repo_map.budget import count_tokens, fit_to_budget
from lemoncrow.pro.capabilities.repo_map.graph import build_reference_graph
from lemoncrow.pro.capabilities.repo_map.pagerank import personalized_pagerank
from lemoncrow.pro.capabilities.repo_map.render import render_outline


def build_repo_map(
    repo_root: str | Path = ".",
    *,
    seed_files: list[str] | None = None,
    budget_tokens: int = 2000,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
) -> RepoMapResult:
    _ = (include_globs, exclude_globs)
    root = Path(repo_root)
    graph, tags_by_file = build_reference_graph(root)
    scores = personalized_pagerank(graph, seed_files or [])
    ranked_files = [file_name for file_name, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def render(files: list[str]) -> str:
        return render_outline(tags_by_file, files)

    selected, outline = fit_to_budget(ranked_files, render, budget_tokens)
    return RepoMapResult(
        outline=outline,
        ranked_files=selected,
        token_count=count_tokens(outline),
        budget_tokens=budget_tokens,
    )


__all__ = ["RepoMapResult", "build_repo_map"]
