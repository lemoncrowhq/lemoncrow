"""Compact benchmark-facing renderers for code context payloads."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

# Matches the "\d+\t" line-number prefix baked into explore source sections.
_LINE_NUM_RE = re.compile(r"^\d+\t")

_CONTEXT_ENTRY_CAP = 8
_CONTEXT_RELATED_CAP = 10
_CONTEXT_CODE_BLOCK_CAP = 3
_CONTEXT_PER_FILE_CAP = 3
_EXPLORE_FILE_SYMBOL_CAP = 8
_NODE_BODY_MAX_LINES = 400
_NODE_BODY_HEAD_LINES = 120


def render_code_payload(op: str, payload: Mapping[str, Any]) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    if payload.get("error"):
        return None
    if op == "search":
        return _render_search(payload)
    if op in {"symbol", "node"}:
        # The compact `symbol` view is location/signature-only; only the `node`
        # view emits the source body.
        return _render_symbol(payload, include_source=(op == "node"))
    if op in {"callers", "callees", "usages"}:
        return _render_relations(op, payload)
    if op == "pattern":
        return _render_pattern(payload)
    if op == "blame":
        return _render_blame(payload)
    if op == "outline":
        return _render_outline(payload)
    if op == "status":
        return _render_status(payload)
    if op == "index":
        return _render_index(payload)
    if op == "cache_status":
        return _render_cache_status(payload)
    if op == "context":
        return _render_context(payload)
    if op == "explore":
        return _render_explore(payload)
    if op == "files":
        return _render_files(payload)
    if op == "routes":
        return _render_routes(payload)
    return None


def _group_rows_by_file(rows: Iterable[tuple[str, int, str]]) -> list[str]:
    """Render ``(path, line, label)`` rows with each file path emitted once.

    Emitting the path once per file (a header line, then indented per-hit
    lines) instead of repeating the full path on every hit is the dominant
    token win for clustered usages/callers/callees/search/pattern results.
    """
    from itertools import groupby

    ordered = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
    out: list[str] = []
    for file_path, group in groupby(ordered, key=lambda row: row[0]):
        out.append(f"- {file_path}")
        for _path, line, label in group:
            loc = str(line) if line > 0 else ""
            if loc and label:
                out.append(f"  - {loc} — {label}")
            elif loc:
                out.append(f"  - {loc}")
            elif label:
                out.append(f"  - {label}")
            else:
                out.append("  - ?")
    return out


def _render_search(payload: Mapping[str, Any]) -> str:
    items = payload.get("items")
    if not isinstance(items, list):
        return "- no matches"
    rows: list[tuple[str, int, str]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        file_path = str(item.get("path") or item.get("file_path") or "?")
        line = int(item.get("line") or item.get("start_line") or 0)
        name = str(item.get("qualified_name") or item.get("name") or item.get("symbol_name") or "?")
        kind = str(item.get("kind") or "?")
        rows.append((file_path, line, f"{name} [{kind}]"))
    if not rows:
        return "- no matches"
    lines: list[str] = []
    lines.extend(_group_rows_by_file(rows))
    return "\n".join(lines)


def _render_symbol(payload: Mapping[str, Any], *, include_source: bool = True) -> str:
    symbol_id = str(payload.get("symbol_id") or "").strip()
    file_path = str(payload.get("path") or payload.get("file_path") or "?")
    start_line = int(payload.get("line") or payload.get("start_line") or 0)
    end_line = int(payload.get("end_line") or 0)
    symbol = str(payload.get("qualified_name") or payload.get("name") or payload.get("symbol_name") or symbol_id or "?")
    kind = str(payload.get("kind") or "?")
    signature = str(payload.get("signature") or "").strip()
    # Header: same format as explore sections — #### path:Lx-Ly — name [kind]
    range_tag = (
        f":L{start_line}-L{end_line}"
        if start_line and end_line >= start_line
        else (f":L{start_line}" if start_line else "")
    )
    header = f"#### {file_path}{range_tag} — {symbol} [{kind}]"
    lines: list[str] = [header]
    if symbol_id:
        lines.append(f"- id: {symbol_id}")
    if signature:
        lines.append(f"- signature: {signature}")
    source = str(payload.get("source") or "")
    if source and include_source:
        language = str(payload.get("language") or "")
        body_lines = source.splitlines()
        if len(body_lines) <= _NODE_BODY_MAX_LINES:
            kept = body_lines
        else:
            kept = body_lines[:_NODE_BODY_HEAD_LINES]
            lines.append(
                f"*first {_NODE_BODY_HEAD_LINES} of {len(body_lines)} lines; "
                f"read L{start_line}-L{end_line} for the rest*"
            )
        if start_line > 0:
            body = "\n".join(f"{start_line + idx}\t{line}" for idx, line in enumerate(kept))
        else:
            body = "\n".join(kept)
        lines.append(f"```{language}" if language else "```")
        lines.append(body)
        lines.append("```")
    return "\n".join(lines)


def _render_relations(op: str, payload: Mapping[str, Any]) -> str:
    lines: list[str] = []

    if op == "usages":
        rows: list[tuple[str, int, str]] = []
        for ref in _flatten_usages(payload.get("references")):
            file_path = str(ref.get("path") or ref.get("file_path") or "?")
            line = int(ref.get("line") or 0)
            caller = str(ref.get("caller") or ref.get("enclosing_qualified_name") or "").strip()
            rows.append((file_path, line, caller))
        if not rows:
            lines.append("- no references")
            return "\n".join(lines)
        lines.extend(_group_rows_by_file(rows))
        return "\n".join(lines)

    related = payload.get("related")
    rows = []
    if isinstance(related, list):
        for item in related:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("qualified_name") or item.get("name") or item.get("symbol_name") or "?")
            file_path = str(item.get("path") or item.get("file_path") or "?")
            line = int(item.get("line") or item.get("start_line") or 0)
            rows.append((file_path, line, name))
    if not rows:
        lines.append("- no related symbols")
        return "\n".join(lines)
    lines.extend(_group_rows_by_file(rows))
    return "\n".join(lines)


def _flatten_usages(references: Any) -> list[Mapping[str, Any]]:
    if isinstance(references, list):
        rows = [item for item in references if isinstance(item, Mapping)]
    elif isinstance(references, Mapping):
        rows = []
        for key in sorted(references.keys(), key=lambda value: str(value)):
            value = references[key]
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, Mapping))
    else:
        rows = []
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("path") or item.get("file_path") or ""),
            int(item.get("line") or 0),
            int(item.get("column") or 0),
        ),
    )


def _render_pattern(payload: Mapping[str, Any]) -> str | None:
    matches = payload.get("matches")
    # Rewrite responses carry a diff/files_changed, not matches -- leave those as
    # JSON so the agent still receives the structured diff. Only search responses
    # (matches is a list, possibly empty) get the compact markdown treatment.
    if not isinstance(matches, list):
        return None
    rows: list[tuple[str, int, str]] = []
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        file_path = str(match.get("path") or match.get("file_path") or "?")
        line = int(match.get("line") or match.get("start_line") or 0)
        snippet = " ".join(str(match.get("snippet") or "").split())[:120]
        rows.append((file_path, line, snippet))
    if not rows:
        return "- no matches"
    lines: list[str] = []
    lines.extend(_group_rows_by_file(rows))
    if payload.get("truncated"):
        total = payload.get("total_matches")
        lines.append(f"- truncated (total_matches={total})" if total is not None else "- truncated")
    return "\n".join(lines)


def _render_blame(payload: Mapping[str, Any]) -> str:
    name = str(payload.get("qualified_name") or payload.get("name") or payload.get("symbol_name") or "?")
    file_path = str(payload.get("path") or payload.get("file_path") or "?")
    start = int(payload.get("line_start") or 0)
    end = int(payload.get("line_end") or 0)
    loc = f"{file_path}:{start}-{end}" if start and end else file_path
    lines = [f"- target: {name} ({loc})"]
    last_author = str(payload.get("last_author") or payload.get("author") or "").strip()
    last_sha = str(payload.get("last_commit_sha") or "").strip()[:10]
    summary = str(payload.get("last_commit_summary") or "").strip()
    if last_sha or last_author:
        head = " ".join(part for part in (last_sha, last_author) if part)
        lines.append(f"- last: {head}" + (f" — {summary}" if summary else ""))
    meta: list[str] = []
    if payload.get("age_days") is not None:
        meta.append(f"age_days={payload['age_days']}")
    if payload.get("distinct_authors") is not None:
        meta.append(f"authors={payload['distinct_authors']}")
    if payload.get("local_edits"):
        meta.append("local_edits=true")
    freshness = str(payload.get("freshness") or "").strip()
    if freshness:
        meta.append(f"freshness={freshness}")
    if meta:
        lines.append("- " + ", ".join(meta))
    hunks = payload.get("hunks")
    if isinstance(hunks, list) and hunks:
        lines.append(f"- hunks ({len(hunks)}):")
        for hunk in hunks:
            if not isinstance(hunk, Mapping):
                continue
            hs = int(hunk.get("start_line") or hunk.get("line") or 0)
            he = int(hunk.get("end_line") or 0)
            rng = f"{hs}-{he}" if hs and he else (str(hs) if hs else "?")
            sha = str(hunk.get("commit_sha") or "").strip()[:10]
            author = str(hunk.get("author_email") or "").strip()
            lines.append(f"  - {rng} " + " ".join(part for part in (sha, author) if part))
    churn = payload.get("churn")
    if isinstance(churn, Mapping) and churn:
        parts = []
        if churn.get("commit_count") is not None:
            parts.append(f"commits={churn['commit_count']}")
        if churn.get("score") is not None:
            parts.append(f"score={churn['score']}")
        if parts:
            lines.append("- churn: " + ", ".join(parts))
    return "\n".join(lines)


def _render_outline(payload: Mapping[str, Any]) -> str | None:
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return None
    total = payload.get("symbol_count")
    lines: list[str] = []
    if isinstance(total, int):
        lines.append(f"- outline: {total} symbols")
    for file_path in sorted(files.keys(), key=str):
        symbols = files[file_path]
        if not isinstance(symbols, list):
            continue
        lines.append(f"- {file_path}")
        for symbol in symbols:
            if not isinstance(symbol, Mapping):
                continue
            name = str(symbol.get("qualified_name") or symbol.get("name") or "?")
            kind = str(symbol.get("kind") or "?")
            start = symbol.get("line_start") or symbol.get("start_line")
            end = symbol.get("line_end") or symbol.get("end_line")
            rng = f"{start}-{end}" if start and end else (str(start) if start else "")
            lines.append(f"  - {rng}: {name} [{kind}]" if rng else f"  - {name} [{kind}]")
    return "\n".join(lines)


def _render_status(payload: Mapping[str, Any]) -> str:
    index = payload.get("index")
    cache = payload.get("cache")
    freshness = payload.get("freshness")
    providers = payload.get("providers")
    files_indexed = int(index.get("files_indexed") or 0) if isinstance(index, Mapping) else 0
    symbols_indexed = int(index.get("symbols_indexed") or 0) if isinstance(index, Mapping) else 0
    entry_count = int(cache.get("entry_count") or 0) if isinstance(cache, Mapping) else 0
    freshness_status = str(freshness.get("status") or "unknown") if isinstance(freshness, Mapping) else "unknown"
    lines = [
        f"- repo: {(payload.get('repo_root') or payload.get('repo_id') or '?')!s}",
        f"- index: files={files_indexed}, symbols={symbols_indexed}",
        f"- cache_entries: {entry_count}",
        f"- freshness: {freshness_status}",
    ]
    if isinstance(providers, list):
        provider_rows = sorted(
            (provider for provider in providers if isinstance(provider, Mapping)),
            key=lambda provider: str(provider.get("name") or ""),
        )
        for provider in provider_rows:
            lines.append(f"- provider:{(provider.get('name') or '?')!s}={(provider.get('status') or 'unknown')!s}")
    return "\n".join(lines)


def _render_index(payload: Mapping[str, Any]) -> str:
    files_indexed = int(payload.get("files_indexed") or 0)
    symbols_indexed = int(payload.get("symbols_indexed") or 0)
    imports_indexed = int(payload.get("imports_indexed") or 0)
    index_version = int(payload.get("index_version") or 0)
    lines = [
        f"- repo: {(payload.get('repo_id') or '?')!s}",
        f"- version: {index_version}",
        f"- counts: files={files_indexed}, symbols={symbols_indexed}, imports={imports_indexed}",
    ]
    return "\n".join(lines)


def _render_cache_status(payload: Mapping[str, Any]) -> str:
    entry_count = int(payload.get("entry_count") or 0)
    total_bytes = int(payload.get("total_bytes") or 0)
    max_bytes = int(payload.get("max_bytes") or 0)
    index_version = int(payload.get("index_version") or 0)
    by_tool = payload.get("entries_by_tool")
    tool_summary = ""
    if isinstance(by_tool, Mapping):
        entries = [f"{key!s}={int(value)}" for key, value in sorted(by_tool.items(), key=lambda item: str(item[0]))]
        if entries:
            tool_summary = ", ".join(entries[:4])
            if len(entries) > 4:
                tool_summary = f"{tool_summary}, +{len(entries) - 4} more"
    lines = [
        f"- repo: {(payload.get('repo_id') or '?')!s}",
        f"- index_version: {index_version}",
        f"- entries: {entry_count}",
        f"- bytes: {total_bytes}/{max_bytes}",
    ]
    if tool_summary:
        lines.append(f"- tools: {tool_summary}")
    return "\n".join(lines)


def _render_context(payload: Mapping[str, Any]) -> str:
    entry_points = _normalize_context_symbols(payload.get("entry_points"), fallback=payload.get("symbols"))
    related_symbols = _normalize_context_symbols(payload.get("related_symbols"), fallback=[])
    code_blocks = _normalize_code_blocks(payload.get("code_blocks"))
    import_neighbors = payload.get("import_neighbors")
    lines: list[str] = []

    if entry_points:
        lines.append("#### entry_points")
        for row in (entry_points or [])[:_CONTEXT_ENTRY_CAP]:
            lines.append(f"- {row['file_path']}:L{row['start_line']} — {row['qualified_name']} [{row['kind']}]")

    if related_symbols:
        lines.append("#### related_symbols")
        for row in related_symbols[:_CONTEXT_RELATED_CAP]:
            lines.append(f"- {row['file_path']}:L{row['start_line']} — {row['qualified_name']} [{row['kind']}]")
    elif isinstance(import_neighbors, list) and import_neighbors:
        lines.append("#### related_symbols")
        for item in sorted(str(value) for value in import_neighbors[:_CONTEXT_RELATED_CAP]):
            lines.append(f"- {item}")

    if code_blocks:
        lines.append("#### code_blocks")
        for block in code_blocks[:_CONTEXT_CODE_BLOCK_CAP]:
            lines.append(
                f"- {block['qualified_name']} ({block['file_path']}:L{block['start_line']}-L{block['end_line']})"
            )
            language = str(block.get("language") or "")
            fence = f"```{language}" if language else "```"
            lines.append(fence)
            lines.append(str(block["source"]).rstrip())
            lines.append("```")

    return "\n".join(lines) if lines else "no context"


def _normalize_context_symbols(items: Any, *, fallback: Any) -> list[dict[str, Any]]:
    source_items = items if isinstance(items, list) and items else fallback
    rows = [item for item in source_items if isinstance(item, Mapping)] if isinstance(source_items, list) else []
    normalized: list[dict[str, Any]] = []
    for item in rows:
        kind = str(item.get("kind") or "?").strip().lower()
        if kind in {"import", "export"}:
            continue
        normalized.append(
            {
                "qualified_name": str(item.get("qualified_name") or item.get("symbol_name") or "?"),
                "file_path": str(item.get("file_path") or "?"),
                "start_line": int(item.get("start_line") or 0),
                "kind": str(item.get("kind") or "?"),
            }
        )
    normalized.sort(
        key=lambda row: (
            row["file_path"],
            row["start_line"],
            row["qualified_name"],
            row["kind"],
        )
    )
    return _cap_symbols_per_file(normalized, max_per_file=_CONTEXT_PER_FILE_CAP)


def _cap_symbols_per_file(rows: list[dict[str, Any]], *, max_per_file: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        file_path = str(row["file_path"])
        seen = counts.get(file_path, 0)
        if seen >= max_per_file:
            continue
        counts[file_path] = seen + 1
        out.append(row)
    return out


def _normalize_code_blocks(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    rows = [item for item in items if isinstance(item, Mapping)]
    normalized: list[dict[str, Any]] = []
    for item in rows:
        normalized.append(
            {
                "qualified_name": str(item.get("qualified_name") or item.get("symbol_name") or "?"),
                "file_path": str(item.get("file_path") or "?"),
                "start_line": int(item.get("start_line") or 0),
                "end_line": int(item.get("end_line") or 0),
                "language": str(item.get("language") or ""),
                "source": str(item.get("source") or "").strip(),
            }
        )
    normalized.sort(
        key=lambda row: (
            row["file_path"],
            row["start_line"],
            row["qualified_name"],
        )
    )
    return normalized


def _render_explore(payload: Mapping[str, Any]) -> str:
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return _render_explore_items(payload)
    parts: list[str] = []
    for file_entry in files:
        if not isinstance(file_entry, Mapping):
            continue
        file_path = str(file_entry.get("file_path") or file_entry.get("path") or "?")
        sections = file_entry.get("source_sections")
        if not isinstance(sections, list):
            continue
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            content = str(section.get("content") or "").rstrip("\n")
            if not content:
                continue
            # Build the range-tagged header:  #### path:Lstart-Lend — name [kind]
            start_line = int(section.get("start_line") or section.get("line") or 0)
            end_line = int(section.get("end_line") or 0)
            range_tag = ""
            if start_line and end_line:
                range_tag = f":L{start_line}-L{end_line}"
            elif start_line:
                range_tag = f":L{start_line}"
            label = ""
            sym_name = str(section.get("symbol_name") or section.get("name") or section.get("qualified_name") or "")
            sym_kind = str(section.get("kind") or "")
            if sym_name:
                label = f" — {sym_name} [{sym_kind}]" if sym_kind else f" — {sym_name}"
            header = f"#### {file_path}{range_tag}{label}"
            # Skeleton notice and query-match tag are inline on the header.
            if section.get("skeleton"):
                header += " · skeleton"
            if section.get("matched"):
                header += " · match"
            lines: list[str] = [header]
            cleaned = "\n".join(_LINE_NUM_RE.sub("", ln) for ln in content.splitlines())
            lines.append(cleaned)
            parts.append("\n".join(lines))
    if parts:
        rel_lines = _render_explore_relationships(payload.get("relationships"))
        if rel_lines:
            parts.append("\n".join(rel_lines))
    extra = payload.get("additional_relevant_files")
    if isinstance(extra, list) and extra:
        extra_block = ["#### additional_relevant_files"]
        extra_block.extend(f"- {path}" for path in extra[:_CONTEXT_RELATED_CAP])
        parts.append("\n".join(extra_block))
    if not parts:
        return "no results"
    if payload.get("exact_match") is False:
        query = str(payload.get("query") or "")
        note = f'*no exact-name match for "{query}"' if query else "*no exact-name match"
        parts.append(note + " — results are nearest FTS; try _node for direct symbol lookup*")
    return "\n\n".join(parts)


def _render_explore_relationships(relationships: Any) -> list[str]:
    if not isinstance(relationships, Mapping):
        return []
    out: list[str] = []
    for op_name in ("callers", "callees", "usages"):
        groups = relationships.get(op_name)
        if not isinstance(groups, list) or not groups:
            continue
        rows: list[tuple[str, int, str]] = []
        for group in groups:
            if not isinstance(group, Mapping):
                continue
            entries = group.get("references") if op_name == "usages" else group.get("related")
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                file_path = str(entry.get("file_path") or entry.get("path") or "?")
                line = int(entry.get("line") or entry.get("start_line") or 0)
                label = str(
                    entry.get("qualified_name")
                    or entry.get("symbol_name")
                    or entry.get("name")
                    or entry.get("caller")
                    or ""
                )
                rows.append((file_path, line, label))
        if rows:
            out.append(f"#### {op_name}")
            out.extend(_group_rows_by_file(rows))
    return out


def _render_explore_items(payload: Mapping[str, Any]) -> str:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return "no results"
    lines: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        file_path = str(item.get("file_path") or item.get("path") or "?")
        name = str(item.get("qualified_name") or item.get("symbol_name") or "")
        source = str(item.get("source") or "").strip()
        lines.append(f"{file_path} — {name}" if name else file_path)
        if source:
            lines.append(source)
    return "\n".join(lines) if lines else "no results"


def _render_files(payload: Mapping[str, Any]) -> str:
    files = payload.get("files")
    if isinstance(files, list):
        return "\n".join(f"- {entry}" for entry in files) if files else "no files"
    if isinstance(files, Mapping):
        return "\n".join(f"- {path}" for path in sorted(str(key) for key in files)) if files else "no files"
    return "no files"


def _render_routes(payload: Mapping[str, Any]) -> str:
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        return "no routes"
    lines: list[str] = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        method = str(route.get("method") or "?").upper()
        path = str(route.get("path") or route.get("route") or "?")
        handler = str(route.get("handler") or route.get("function") or "")
        file_path = str(route.get("file_path") or "")
        line = int(route.get("line") or route.get("start_line") or 0)
        loc = f" ({file_path}:{line})" if file_path and line > 0 else (f" ({file_path})" if file_path else "")
        handler_part = f" — {handler}" if handler else ""
        lines.append(f"- {method} {path}{handler_part}{loc}")
    return "\n".join(lines) if lines else "no routes"
