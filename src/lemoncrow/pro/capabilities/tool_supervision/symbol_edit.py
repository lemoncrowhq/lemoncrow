"""Symbol-scoped edit planning for rich edit descriptors."""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.core.capabilities.tool_supervision_contract import SymbolEditError as SymbolEditError
from lemoncrow.pro.capabilities.code_context import CodeContextEngine


@dataclass(frozen=True)
class ResolvedSymbolEdit:
    """Concrete file edit derived from a symbol descriptor."""

    symbol_id: str
    repo_file_path: str
    scoped_file_path: str
    old_string: str
    new_string: str
    content_hash: str
    mode: str


def preview_symbol_edit_path(edit: dict[str, Any], *, repo_root: str | Path | None = None) -> str:
    """Resolve the file path for a symbol edit before snapshots are taken."""
    return resolve_symbol_edit(edit, repo_root=repo_root).repo_file_path


def resolve_symbol_edit(edit: dict[str, Any], *, repo_root: str | Path | None = None) -> ResolvedSymbolEdit:
    """Resolve a ``kind="symbol"`` descriptor into a concrete file edit."""
    root = Path(repo_root or Path.cwd()).resolve()
    engine = CodeContextEngine(root)
    symbol = _resolve_symbol_payload(edit, engine=engine)
    if str(symbol.get("origin") or "internal") == "external":
        raise SymbolEditError(
            "external_symbol_edit_not_allowed",
            "external dependency symbols are read-only; edit the first-party wrapper or local source instead",
            symbol_id=str(symbol["symbol_id"]),
            file_path=str(symbol["file_path"]),
            origin="external",
        )
    repo_file_path = str(symbol["file_path"])
    abs_path = root / repo_file_path
    current_bytes = abs_path.read_bytes()
    current_hash = hashlib.sha256(current_bytes).hexdigest()
    if current_hash != str(symbol["content_hash"]):
        raise SymbolEditError(
            "stale_target",
            "symbol changed since it was resolved",
            symbol_id=str(symbol["symbol_id"]),
            file_path=repo_file_path,
        )
    start_byte = int(symbol["start_byte"])
    end_byte = int(symbol["end_byte"])
    current_source = current_bytes[start_byte:end_byte].decode("utf-8", errors="replace")
    file_lines = current_bytes.decode("utf-8", errors="replace").splitlines()
    header_indent = ""
    start_line = int(symbol["start_line"])
    end_line = int(symbol["end_line"])
    if 0 < start_line <= len(file_lines):
        header_indent = _leading_whitespace(file_lines[start_line - 1])
    exact_source = "\n".join(file_lines[start_line - 1 : end_line])
    original_source = str(symbol["source"])
    if current_source != original_source:
        raise SymbolEditError(
            "concurrent_edit",
            "symbol source no longer matches the resolved span",
            symbol_id=str(symbol["symbol_id"]),
            file_path=repo_file_path,
        )

    mode = str(edit.get("mode") or "replace")
    preserve_signature = bool(edit.get("preserve_signature", False))
    new_body = str(edit.get("new_body") or "")
    new_string = _apply_symbol_mode(
        source=exact_source,
        new_body=new_body,
        mode=mode,
        preserve_signature=preserve_signature,
        header_indent=header_indent,
    )
    scoped_file_path = f"{repo_file_path}:L{int(symbol['start_line'])}-L{int(symbol['end_line'])}"
    return ResolvedSymbolEdit(
        symbol_id=str(symbol["symbol_id"]),
        repo_file_path=repo_file_path,
        scoped_file_path=scoped_file_path,
        old_string=exact_source,
        new_string=new_string,
        content_hash=str(symbol["content_hash"]),
        mode=mode,
    )


def record_symbol_edit_memory(resolved: ResolvedSymbolEdit) -> None:
    """Persist a small memory block linking the symbol edit to the current trace."""
    from lemoncrow.gateway.adapters.mcp_server import _get_ledger, _memory_upsert_block

    trace_id = _get_ledger().session_id
    _memory_upsert_block(
        agent_id="shared",
        label=f"edits/{resolved.symbol_id}",
        value=trace_id,
        metadata={
            "symbol_id": resolved.symbol_id,
            "file_path": resolved.repo_file_path,
            "mode": resolved.mode,
            "trace_id": trace_id,
        },
    )


def _resolve_symbol_payload(edit: dict[str, Any], *, engine: CodeContextEngine) -> dict[str, Any]:
    symbol_id = _optional_text(edit.get("symbol_id"))
    qualified_name = _optional_text(edit.get("qualified_name"))
    symbol_name = _optional_text(edit.get("symbol_name"))
    name = _optional_text(edit.get("name"))
    file_path = _optional_text(edit.get("file_path")) or _optional_text(edit.get("path"))
    normalized_file = str(Path(file_path).as_posix()) if file_path else None

    if symbol_id:
        try:
            payload = engine.get_symbol(symbol_id=symbol_id)
        except LookupError as exc:
            raise SymbolEditError("symbol_not_found", "symbol_id did not resolve", symbol_id=symbol_id) from exc
        if normalized_file and payload.get("file_path") != normalized_file:
            raise SymbolEditError(
                "symbol_not_found",
                "resolved symbol did not match requested file",
                symbol_id=symbol_id,
                file_path=normalized_file,
            )
        return payload

    target_qualified = qualified_name or (name if name and "." in name else None)
    target_symbol = symbol_name or name
    query = target_qualified or target_symbol
    if not query:
        raise SymbolEditError(
            "invalid_symbol_descriptor",
            "symbol edits require name, symbol_name, qualified_name, or symbol_id",
        )

    candidates = engine.search_symbols(query, limit=50, snippet="none")
    exact = [
        candidate
        for candidate in candidates
        if (
            (target_qualified and candidate.qualified_name == target_qualified)
            or (target_symbol and candidate.symbol_name == target_symbol)
        )
        and (normalized_file is None or candidate.file_path == normalized_file)
    ]
    deduped: dict[str, Any] = {candidate.symbol_id: candidate for candidate in exact}
    matches = list(deduped.values())
    if not matches:
        raise SymbolEditError(
            "symbol_not_found",
            "no matching symbol was found",
            query=query,
            file_path=normalized_file,
        )
    if len(matches) > 1:
        raise SymbolEditError(
            "disambiguation_required",
            "symbol edit target is ambiguous",
            query=query,
            matches=[
                {
                    "symbol_id": candidate.symbol_id,
                    "qualified_name": candidate.qualified_name,
                    "symbol_name": candidate.symbol_name,
                    "file_path": candidate.file_path,
                    "start_line": candidate.start_line,
                }
                for candidate in matches[:10]
            ],
        )
    return engine.get_symbol(symbol_id=matches[0].symbol_id)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_symbol_mode(*, source: str, new_body: str, mode: str, preserve_signature: bool, header_indent: str) -> str:
    if mode == "replace":
        if not preserve_signature:
            return _indent_symbol_block(new_body, header_indent=header_indent)
        return _replace_symbol_body(source, new_body)
    if mode == "prepend":
        return _join_symbol_fragments(_indent_symbol_block(new_body, header_indent=header_indent), source)
    if mode == "append":
        return _join_symbol_fragments(source, _indent_symbol_block(new_body, header_indent=header_indent))
    raise SymbolEditError("invalid_symbol_mode", f"unsupported symbol edit mode: {mode}", mode=mode)


def _replace_symbol_body(source: str, new_body: str) -> str:
    lines = source.splitlines()
    if not lines:
        return new_body
    header = lines[0]
    body_indent = _body_indent(source)
    body = new_body.strip("\n")
    if not body:
        return header
    indented = "\n".join((body_indent + line if line.strip() else line) for line in body.splitlines())
    return f"{header}\n{indented}"


def _join_symbol_fragments(first: str, second: str) -> str:
    if not first:
        return second
    if not second:
        return first
    joiner = "" if first.endswith("\n") else "\n"
    return f"{first}{joiner}{second}"


def _indent_symbol_block(new_body: str, *, header_indent: str) -> str:
    normalized = textwrap.dedent(new_body).strip("\n")
    lines = normalized.splitlines()
    if not lines:
        return normalized
    result: list[str] = []
    for line in lines:
        if not line.strip():
            result.append(line)
            continue
        result.append(header_indent + line)
    return "\n".join(result)


def _leading_whitespace(line: str) -> str:
    stripped = line.lstrip()
    return line[: len(line) - len(stripped)]


def _body_indent(source: str) -> str:
    lines = source.splitlines()
    if len(lines) > 1:
        for line in lines[1:]:
            stripped = line.lstrip()
            if stripped:
                return line[: len(line) - len(stripped)]
    header = lines[0] if lines else ""
    stripped = header.lstrip()
    return header[: len(header) - len(stripped)] + "    "


__all__ = [
    "ResolvedSymbolEdit",
    "SymbolEditError",
    "preview_symbol_edit_path",
    "record_symbol_edit_memory",
    "resolve_symbol_edit",
]
