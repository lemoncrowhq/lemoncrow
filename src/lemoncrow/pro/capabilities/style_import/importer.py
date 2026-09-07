"""Import Markdown style guides into human-reviewed lesson candidates."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.lesson_models import LessonCandidate
from lemoncrow.core.foundation.models import Playbook
from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.embeddings.factory import get_embedder
from lemoncrow.infra.embeddings.null_embedder import NullEmbedder
from lemoncrow.infra.internal_llm import chat
from lemoncrow.infra.storage.bundle import StoreBundle
from lemoncrow.infra.storage.vector import cosine_similarity
from lemoncrow.pro.capabilities.style_import.prompts import (
    STYLE_IMPORT_PROMPT_VERSION,
    STYLE_IMPORT_RESPONSE_SCHEMA,
    build_messages,
)

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_DEFAULT_DOMAIN = "coding"
_SKIP_PARTS = {".git", ".hg", ".svn", "node_modules", "_book", "_site", "dist", "build"}


@dataclass(frozen=True)
class MarkdownChunk:
    file_path: Path
    start_line: int
    end_line: int
    text: str


ChatFn = Callable[..., str | dict[str, Any]]


def import_files(
    paths: Sequence[Path | str],
    domain: str | None = None,
    *,
    store: StoreBundle | None = None,
    write: bool = True,
    limit: int = 25,
    chat_func: ChatFn = chat,
    embedder: Embedder | None = None,
) -> list[LessonCandidate]:
    """Extract procedural lessons from Markdown files.

    Candidates are only written to the existing lesson inbox when ``write`` is
    true. Promotion into Playbooks still requires the existing human-review
    ``lesson_decide`` flow.
    """

    if limit <= 0:
        return []

    resolved_domain = domain or _DEFAULT_DOMAIN
    files = collect_markdown_files(paths)
    active_embedder = embedder or get_embedder()
    existing_blocks = store.knowledge.list_blocks(domain=resolved_domain, include_deprecated=True) if store else []
    existing_vectors = _embed_existing_blocks(existing_blocks, active_embedder)
    candidates: list[LessonCandidate] = []

    for file_path in files:
        for chunk in split_markdown_chunks(file_path):
            if len(candidates) >= limit:
                break
            response = chat_func(
                build_messages(
                    file_path=str(file_path),
                    chunk_range=(chunk.start_line, chunk.end_line),
                    text=chunk.text,
                ),
                json_schema=STYLE_IMPORT_RESPONSE_SCHEMA,
            )
            candidate = _candidate_from_response(
                response,
                chunk=chunk,
                domain=resolved_domain,
                embedder=active_embedder,
                existing_vectors=existing_vectors,
            )
            if candidate is None:
                continue
            candidates.append(candidate)
            if write and store is not None:
                store.lessons.upsert_lesson_candidate(candidate)
        if len(candidates) >= limit:
            break

    return candidates


def collect_markdown_files(paths: Sequence[Path | str]) -> list[Path]:
    """Collect Markdown files from files or directories, skipping generated trees."""

    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and _is_markdown(path) and not _is_skipped(path):
            found.append(path)
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*.md")):
            if child.is_file() and not _is_skipped(child):
                found.append(child)
    return sorted(dict.fromkeys(found), key=lambda item: str(item))


def split_markdown_chunks(path: Path, *, max_tokens: int = 800) -> list[MarkdownChunk]:
    text = path.read_text(encoding="utf-8")
    return split_markdown_text(text, file_path=path, max_tokens=max_tokens)


def split_markdown_text(text: str, *, file_path: Path, max_tokens: int = 800) -> list[MarkdownChunk]:
    """Split Markdown at H2/H3 boundaries while respecting fenced code blocks."""

    lines = text.splitlines()
    if not lines:
        return []

    raw_chunks: list[MarkdownChunk] = []
    current: list[str] = []
    start_line = 1
    in_fence = False

    for line_no, line in enumerate(lines, 1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        is_boundary = bool(_HEADING_RE.match(line)) and not in_fence
        if is_boundary and current:
            raw_chunks.append(
                MarkdownChunk(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=line_no - 1,
                    text="\n".join(current).strip(),
                )
            )
            current = [line]
            start_line = line_no
        else:
            current.append(line)

    if current:
        raw_chunks.append(
            MarkdownChunk(
                file_path=file_path,
                start_line=start_line,
                end_line=len(lines),
                text="\n".join(current).strip(),
            )
        )

    chunks: list[MarkdownChunk] = []
    for chunk in raw_chunks:
        chunks.extend(_split_long_chunk(chunk, max_tokens=max_tokens))
    return [chunk for chunk in chunks if chunk.text.strip()]


def _split_long_chunk(chunk: MarkdownChunk, *, max_tokens: int) -> list[MarkdownChunk]:
    if _approx_tokens(chunk.text) <= max_tokens:
        return [chunk]

    paragraphs = _paragraphs(chunk)
    out: list[MarkdownChunk] = []
    current: list[str] = []
    current_start = chunk.start_line
    current_end = chunk.start_line

    for start_line, end_line, text in paragraphs:
        next_text = "\n\n".join([*current, text]) if current else text
        if current and _approx_tokens(next_text) > max_tokens:
            out.append(
                MarkdownChunk(
                    file_path=chunk.file_path,
                    start_line=current_start,
                    end_line=current_end,
                    text="\n\n".join(current).strip(),
                )
            )
            current = [text]
            current_start = start_line
        else:
            if not current:
                current_start = start_line
            current.append(text)
        current_end = end_line

    if current:
        out.append(
            MarkdownChunk(
                file_path=chunk.file_path,
                start_line=current_start,
                end_line=current_end,
                text="\n\n".join(current).strip(),
            )
        )
    return out


def _paragraphs(chunk: MarkdownChunk) -> list[tuple[int, int, str]]:
    lines = chunk.text.splitlines()
    out: list[tuple[int, int, str]] = []
    current: list[str] = []
    start_line = chunk.start_line
    in_fence = False

    for offset, line in enumerate(lines):
        line_no = chunk.start_line + offset
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence and current:
            out.append((start_line, line_no - 1, "\n".join(current).strip()))
            current = []
            start_line = line_no + 1
            continue
        if not current:
            start_line = line_no
        current.append(line)

    if current:
        out.append((start_line, chunk.end_line, "\n".join(current).strip()))
    return out


def _candidate_from_response(
    response: str | dict[str, Any],
    *,
    chunk: MarkdownChunk,
    domain: str,
    embedder: Embedder | None,
    existing_vectors: list[tuple[Playbook, list[float]]],
) -> LessonCandidate | None:
    data = _response_dict(response)
    if not _truthy(data.get("procedural")):
        return None

    title = _clean_string(data.get("title")) or f"Procedure from {chunk.file_path.name}"
    body = _clean_string(data.get("body"))
    procedure = _string_list(data.get("procedure")) or ([body] if body else [])
    if not body or not procedure:
        return None

    verification = _string_list(data.get("verification"))
    triggers = _string_list(data.get("triggers"))
    confidence = _clamp_confidence(data.get("confidence"))
    block = Playbook(
        id=Playbook.make_id(title, domain),
        title=title,
        domain=domain,
        task_types=["style-guide-import"],
        triggers=list(dict.fromkeys([*triggers, "style-guide-import", chunk.file_path.stem])),
        situation=body,
        procedure=procedure,
        verification=verification,
        failure_signals=[],
        when_not_to_apply="When the imported source no longer represents current team policy.",
    )
    embedder_local = embedder or get_embedder()
    embedding = _embed_block(block, embedder_local)
    near_duplicates = _near_duplicates(block, embedding, existing_vectors)
    fingerprint = _fingerprint(chunk=chunk, body=body)
    return LessonCandidate(
        domain=domain,
        cluster_fingerprint=f"style-guide-import:{fingerprint}",
        kind="new_block",
        proposed_block=block,
        evidence_trace_ids=[],
        body=body,
        evidence={
            "source": "style-guide-import",
            "file_path": str(chunk.file_path),
            "chunk_range": [chunk.start_line, chunk.end_line],
            "prompt_version": STYLE_IMPORT_PROMPT_VERSION,
            "near_duplicates": near_duplicates,
        },
        embedding=embedding,
        embedding_provenance=getattr(embedder_local, "name", embedder_local.__class__.__name__),
        confidence=confidence,
    )


def _response_dict(response: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return {"procedural": False}
    return parsed if isinstance(parsed, dict) else {"procedural": False}


def _embed_existing_blocks(blocks: list[Playbook], embedder: Embedder | None) -> list[tuple[Playbook, list[float]]]:
    embedder_local = embedder or get_embedder()
    out: list[tuple[Playbook, list[float]]] = []
    for block in blocks:
        out.append((block, _embed_block(block, embedder_local)))
    return out


def _embed_block(block: Playbook, embedder: Embedder) -> list[float]:
    text = "\n".join([block.title, block.situation, *block.procedure, *block.verification])
    try:
        vectors = embedder.embed([text])
    except Exception:
        logging.exception("Recovered from broad exception handler")
        vectors = NullEmbedder().embed([text])
    return vectors[0] if vectors else []


def _near_duplicates(
    block: Playbook,
    embedding: list[float],
    existing_vectors: list[tuple[Playbook, list[float]]],
    *,
    threshold: float = 0.9,
) -> list[dict[str, Any]]:
    duplicates: list[dict[str, Any]] = []
    block_tokens = _tokens(" ".join([block.title, block.situation, *block.procedure]))
    for existing, existing_embedding in existing_vectors:
        similarity = 0.0
        if embedding and existing_embedding and len(embedding) == len(existing_embedding):
            similarity = cosine_similarity(embedding, existing_embedding)
        existing_tokens = _tokens(" ".join([existing.title, existing.situation, *existing.procedure]))
        lexical = len(block_tokens & existing_tokens) / max(1, len(block_tokens | existing_tokens))
        score = max(similarity, lexical)
        if score >= threshold:
            duplicates.append(
                {
                    "block_id": existing.id,
                    "title": existing.title,
                    "score": round(score, 4),
                }
            )
    duplicates.sort(key=lambda item: (-float(item["score"]), str(item["block_id"])))
    return duplicates[:5]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_]{3,}", text)}


def _fingerprint(*, chunk: MarkdownChunk, body: str) -> str:
    raw = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{body}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _clean_string(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_string(item) for item in value if _clean_string(item)]
    if isinstance(value, str) and value.strip():
        return [_clean_string(value)]
    return []


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.7
    return max(0.0, min(1.0, number))


def _is_markdown(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"}


def _is_skipped(path: Path) -> bool:
    return bool(_SKIP_PARTS & set(path.parts))


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


__all__ = [
    "MarkdownChunk",
    "collect_markdown_files",
    "import_files",
    "split_markdown_chunks",
    "split_markdown_text",
]
