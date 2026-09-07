"""N17 -- design-doc indexing into a SEPARATE retrieval corpus.

Indexes Markdown design docs (heading-tree chunked, embedder-window-sized) into
a dedicated SQLite store so docs can be recalled on demand *alongside* code
without touching the existing code-retrieval path.

Design decisions / guards:

Separate store (additive)
    Chunks live in their own ``design_docs.sqlite3`` under the LemonCrow root --
    NOT in the symbol index, the code-context engine, or the archival memory
    store. Nothing in the existing retrieval defaults reads this table, so with
    indexing off the code path is byte-identical.

Opt-in
    :func:`index_design_docs` only writes when ``LEMONCROW_DOC_INDEXING`` is
    truthy (or ``enable=True`` is passed explicitly, e.g. from a test). Recall
    is always available against whatever has been indexed.

Heading-tree chunking
    A doc is split at Markdown ATX headings (``#``..``######``); each chunk
    carries its heading breadcrumb (``Parent > Child``) and is further split so
    no chunk exceeds the embedder window (``_MAX_CHUNK_CHARS``).

Embedding reuse + fallback
    Embeddings come from the shared :func:`make_embedder`. When the embedder is
    the null backend (dim 0) or embedding fails, chunks are still stored with an
    empty vector and recall transparently falls back to lexical token overlap,
    so doc recall works offline.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_FILENAME = "design_docs.sqlite3"
_DOC_INDEXING_ENV = "LEMONCROW_DOC_INDEXING"
# Embedder-window-sized chunk cap (characters). Conservative for typical 512/8k
# token windows; large sections are split on paragraph boundaries below this.
_MAX_CHUNK_CHARS = 1200

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_TOKEN = re.compile(r"[a-zA-Z0-9_]+")


def doc_indexing_enabled(env: Any | None = None) -> bool:
    """Return True only when the opt-in design-doc indexing flag is set.

    Default-off: with ``LEMONCROW_DOC_INDEXING`` unset, :func:`index_design_docs`
    writes nothing and existing retrieval is untouched.
    """
    from lemoncrow.core.environment import bool_env

    return bool_env(_DOC_INDEXING_ENV, default=False, env=env)


@dataclass(frozen=True)
class DocChunk:
    """A single heading-scoped, window-sized doc chunk."""

    doc_path: str
    heading_path: str
    line_start: int
    text: str


def _split_oversize(text: str) -> list[str]:
    """Split a section body into <= _MAX_CHUNK_CHARS pieces on blank lines."""
    if len(text) <= _MAX_CHUNK_CHARS:
        return [text] if text.strip() else []
    pieces: list[str] = []
    buf: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        addition = len(para) + 2
        if size + addition > _MAX_CHUNK_CHARS and buf:
            joined = "\n\n".join(buf).strip()
            if joined:
                pieces.append(joined)
            buf, size = [], 0
        buf.append(para)
        size += addition
    joined = "\n\n".join(buf).strip()
    if joined:
        pieces.append(joined)
    return pieces


def chunk_markdown(doc_path: str, text: str) -> list[DocChunk]:
    """Heading-tree chunk a Markdown doc into window-sized :class:`DocChunk`s.

    Each chunk's ``heading_path`` is the breadcrumb of enclosing headings, e.g.
    ``Architecture > Storage``. Body content before the first heading is kept
    under a synthetic ``(intro)`` heading so nothing is dropped.
    """
    lines = text.splitlines()
    chunks: list[DocChunk] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)
    section_lines: list[str] = []
    section_start = 1

    def _breadcrumb() -> str:
        return " > ".join(title for _lvl, title in heading_stack) or "(intro)"

    def _flush(start_line: int) -> None:
        body = "\n".join(section_lines).strip()
        if not body:
            return
        crumb = _breadcrumb()
        for piece in _split_oversize(body):
            chunks.append(
                DocChunk(
                    doc_path=doc_path,
                    heading_path=crumb,
                    line_start=start_line,
                    text=piece,
                )
            )

    for idx, line in enumerate(lines, start=1):
        match = _HEADING.match(line)
        if match is None:
            section_lines.append(line)
            continue
        # New heading: flush the accumulated section under the *current* crumb.
        _flush(section_start)
        section_lines = []
        section_start = idx
        level = len(match.group(1))
        title = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))
    _flush(section_start)
    return chunks


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class DesignDocStore:
    """Separate, persistent SQLite store for design-doc chunks (N17).

    Schema is created lazily on first write/read; the file is wholly separate
    from the code index, so its mere existence never affects code retrieval.
    """

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / _DB_FILENAME

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS design_doc_chunks (
                chunk_id      TEXT PRIMARY KEY,
                doc_path      TEXT NOT NULL,
                heading_path  TEXT NOT NULL,
                line_start    INTEGER NOT NULL,
                text          TEXT NOT NULL,
                tokens        TEXT NOT NULL,
                embedder_name TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                vector_json   TEXT NOT NULL,
                content_hash  TEXT NOT NULL,
                indexed_at    INTEGER NOT NULL
            )
            """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ddc_doc ON design_doc_chunks(doc_path)")
        return conn

    @staticmethod
    def _chunk_id(chunk: DocChunk) -> str:
        raw = f"{chunk.doc_path}::{chunk.line_start}::{chunk.heading_path}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def upsert_doc(
        self,
        chunks: list[DocChunk],
        *,
        vectors: list[list[float]],
        embedder_name: str,
        embedding_dim: int,
    ) -> int:
        """Replace all stored chunks for the docs in *chunks* with fresh rows."""
        if not chunks:
            return 0
        now = int(time.time())
        doc_paths = {c.doc_path for c in chunks}
        with closing(self._connect()) as conn:
            for doc_path in doc_paths:
                conn.execute("DELETE FROM design_doc_chunks WHERE doc_path = ?", (doc_path,))
            import json

            for chunk, vector in zip(chunks, vectors, strict=True):
                content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO design_doc_chunks
                    (chunk_id, doc_path, heading_path, line_start, text, tokens,
                     embedder_name, embedding_dim, vector_json, content_hash, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._chunk_id(chunk),
                        chunk.doc_path,
                        chunk.heading_path,
                        chunk.line_start,
                        chunk.text,
                        " ".join(_tokens(f"{chunk.heading_path} {chunk.text}")),
                        embedder_name,
                        embedding_dim,
                        json.dumps(vector),
                        content_hash,
                        now,
                    ),
                )
            conn.commit()
        return len(chunks)

    def count(self) -> int:
        if not self._path.exists():
            return 0
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT COUNT(*) FROM design_doc_chunks").fetchone()
            return int(row[0]) if row else 0

    def query(
        self,
        query_text: str,
        *,
        query_vector: list[float] | None,
        embedder_name: str,
        embedding_dim: int,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top design-doc chunks for *query_text*.

        Uses cosine over stored vectors when the query has a usable vector AND
        stored vectors share the embedder/dim stamp (N5-style drift guard); else
        falls back to lexical token-overlap so recall works with no embeddings.
        """
        if not self._path.exists():
            return []
        import json

        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT doc_path, heading_path, line_start, text, tokens, "
                "embedder_name, embedding_dim, vector_json FROM design_doc_chunks"
            ).fetchall()
        if not rows:
            return []

        use_vectors = bool(query_vector) and embedding_dim > 0
        scored: list[tuple[float, dict[str, Any]]] = []
        if use_vectors:
            from lemoncrow.infra.storage.vector import cosine_similarity

            assert query_vector is not None
            for doc_path, heading_path, line_start, text, _toks, ename, edim, vjson in rows:
                if ename != embedder_name or int(edim) != embedding_dim:
                    continue  # drift: foreign vector space, skip
                vector = json.loads(vjson)
                if not vector:
                    continue
                score = cosine_similarity(query_vector, vector)
                scored.append((score, self._row_dict(doc_path, heading_path, line_start, text, score, "vector")))

        if not scored:  # no vectors usable -> lexical fallback
            q_tokens = set(_tokens(query_text))
            if not q_tokens:
                return []
            for doc_path, heading_path, line_start, text, toks, _e, _d, _v in rows:
                doc_tokens = set(toks.split())
                if not doc_tokens:
                    continue
                overlap = len(q_tokens & doc_tokens)
                if overlap == 0:
                    continue
                score = overlap / len(q_tokens)
                scored.append((score, self._row_dict(doc_path, heading_path, line_start, text, score, "lexical")))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _score, payload in scored[:limit]]

    @staticmethod
    def _row_dict(
        doc_path: str,
        heading_path: str,
        line_start: int,
        text: str,
        score: float,
        method: str,
    ) -> dict[str, Any]:
        snippet = text if len(text) <= 400 else text[:400] + " ..."
        return {
            "doc": doc_path,
            "heading_path": heading_path,
            "line": line_start,
            "score": round(float(score), 4),
            "method": method,
            "text": snippet,
        }


def _collect_doc_paths(roots: list[Path]) -> list[Path]:
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


def _embed_chunks(chunks: list[DocChunk]) -> tuple[list[list[float]], str, int]:
    """Embed chunk bodies; fail open to empty vectors (lexical recall still works)."""
    try:
        from lemoncrow.infra.embeddings.factory import make_embedder

        embedder = make_embedder()
        dim = int(getattr(embedder, "dim", 0))
        name = str(getattr(embedder, "name", "unknown"))
        if dim <= 0:
            return [[] for _ in chunks], name, 0
        vectors = embedder.embed([c.text for c in chunks])
        return vectors, name, dim
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return [[] for _ in chunks], "unavailable", 0


def index_design_docs(
    *,
    repo_root: Path,
    lemoncrow_root: Path,
    paths: list[str] | None = None,
    enable: bool | None = None,
) -> dict[str, Any]:
    """Index Markdown design docs into the separate doc store (opt-in).

    No-op unless ``LEMONCROW_DOC_INDEXING`` is truthy or ``enable=True`` is passed
    explicitly. Returns a structured receipt either way; the ``enabled`` field
    makes the default-off behaviour observable.
    """
    active = enable if enable is not None else doc_indexing_enabled()
    if not active:
        return {
            "kind": "index_docs",
            "enabled": False,
            "indexed_chunks": 0,
            "docs": 0,
            "note": f"Opt-in: set {_DOC_INDEXING_ENV}=1 to enable design-doc indexing.",
        }
    try:
        if paths:
            roots = [Path(p) if Path(p).is_absolute() else repo_root / p for p in paths]
        else:
            roots = [repo_root]
        doc_paths = _collect_doc_paths(roots)
        all_chunks: list[DocChunk] = []
        for doc in doc_paths:
            try:
                text = doc.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            all_chunks.extend(chunk_markdown(str(doc), text))
        if not all_chunks:
            return {"kind": "index_docs", "enabled": True, "indexed_chunks": 0, "docs": 0}
        vectors, embedder_name, dim = _embed_chunks(all_chunks)
        store = DesignDocStore(lemoncrow_root)
        written = store.upsert_doc(all_chunks, vectors=vectors, embedder_name=embedder_name, embedding_dim=dim)
        return {
            "kind": "index_docs",
            "enabled": True,
            "indexed_chunks": written,
            "docs": len(doc_paths),
            "embedder": embedder_name,
            "embedding_dim": dim,
        }
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {"kind": "index_docs", "enabled": True, "indexed_chunks": 0, "docs": 0, "error": "recovered"}


def recall_design_docs(
    *,
    lemoncrow_root: Path,
    query: str,
    limit: int = 5,
) -> dict[str, Any]:
    """Recall the most relevant design-doc chunks for *query* (read-only).

    Always available against whatever has been indexed; returns an empty result
    when nothing has been indexed yet. Never reads or affects the code index.
    """
    try:
        store = DesignDocStore(lemoncrow_root)
        if store.count() == 0 or not query.strip():
            return {"kind": "recall_docs", "query": query, "result_count": 0, "results": []}
        query_vector: list[float] | None = None
        embedder_name = "unavailable"
        dim = 0
        try:
            from lemoncrow.infra.embeddings.factory import make_embedder

            embedder = make_embedder()
            dim = int(getattr(embedder, "dim", 0))
            embedder_name = str(getattr(embedder, "name", "unknown"))
            if dim > 0:
                query_vector = embedder.embed([query])[0]
        except Exception:
            logger.exception("Recovered from broad exception handler")
        results = store.query(
            query,
            query_vector=query_vector,
            embedder_name=embedder_name,
            embedding_dim=dim,
            limit=limit,
        )
        return {
            "kind": "recall_docs",
            "query": query,
            "result_count": len(results),
            "results": results,
        }
    except Exception:
        logger.exception("Recovered from broad exception handler")
        return {"kind": "recall_docs", "query": query, "result_count": 0, "results": [], "error": "recovered"}
