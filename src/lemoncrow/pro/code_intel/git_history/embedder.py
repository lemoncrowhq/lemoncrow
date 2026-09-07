"""Embedding helper for Context Lineage commit summaries."""

from __future__ import annotations

import struct

from lemoncrow.infra.embeddings.base import Embedder
from lemoncrow.infra.embeddings.factory import embed_documents, get_code_embedder
from lemoncrow.pro.code_intel.git_history.models import CommitSummary

_embedder: Embedder | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = get_code_embedder()
    return _embedder


def embed_summary(summary: CommitSummary) -> bytes | None:
    """Embed the summary text + top-10 files into a float32 BLOB."""
    embedder = _get_embedder()
    if embedder.dim <= 0:
        return None
    text = f"{summary.summary}\n{' '.join(summary.files_touched[:10])}"
    vectors = embed_documents(embedder, [text])
    vec = vectors[0]
    if not vec:
        return None
    return struct.pack(f"{len(vec)}f", *vec)


def decode_embedding(blob: bytes) -> list[float]:
    """Deserialise a BLOB back to list[float]."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def embedding_dim() -> int:
    """Return the current embedder dimension."""
    return _get_embedder().dim


def embedder_name() -> str:
    """Return the current embedder cache/identity string."""
    return _get_embedder().name


__all__ = ["decode_embedding", "embed_summary", "embedder_name", "embedding_dim"]
