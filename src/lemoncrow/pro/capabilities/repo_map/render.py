"""Rendering for repo map output."""

from __future__ import annotations

from lemoncrow.infra.tree_sitter.tags import Tag


def render_outline(tags_by_file: dict[str, list[Tag]], ranked_files: list[str]) -> str:
    lines: list[str] = []
    for file_name in ranked_files:
        defs = [tag for tag in tags_by_file.get(file_name, []) if tag.kind == "definition"]
        if not defs:
            continue
        lines.append(file_name)
        for tag in defs[:40]:
            lines.append(f"  L{tag.line}: {tag.name}")
    return "\n".join(lines)


__all__ = ["render_outline"]
