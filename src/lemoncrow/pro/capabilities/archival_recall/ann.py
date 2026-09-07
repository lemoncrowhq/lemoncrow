"""Persistent, incrementally-maintained ANN retrieval over archival embeddings (G5).

Archival passages are persisted by the memory store and carry their own vector
provenance (``embedding`` + ``embedding_model`` + ``embedding_provenance``). This
module layers an opt-in approximate-nearest-neighbour pre-filter over that
persisted set so cosine ranking does not have to brute-force every passage,
while preserving exact results.

The HNSW graph is **built once and extended incrementally** -- new eligible
passages are inserted into the existing graph instead of triggering a full
rebuild -- and (when a ``persist_path`` is given) the graph is **persisted to
disk**, so a fresh process loads it instead of paying the full build cost again.
The datasketch HNSW build is superlinear (~64s for 2000 nodes here), so this is
the difference between every CLI query rebuilding the graph and a one-time build
followed by ~instant loads plus cheap deltas.

Guards:

N5 (drift invalidation)
    Only passages whose ``embedding_model`` matches the live embedder model-id
    AND whose embedding dim matches the query are admitted to the ANN. A model
    or dim change discards the graph (in memory and on the next load), so
    neighbours are never recovered from a stale model or a foreign vector space.

Most-recent-N exact tail (mandatory)
    The newest ``recent_exact`` passages are always retained as exact candidates
    regardless of ANN recall, so just-stored memory is never missed by the
    approximate index.

Brute-force fallback (mandatory)
    Exact cosine is the fallback when the eligible set is small, ``datasketch``
    is unavailable, or an ANN query raises -- so a missing optional dependency
    never breaks recall.

Default-safe
    Nothing here runs unless the caller opts in (``LEMONCROW_ANN_RETRIEVAL``); with
    the flag off, archival ranking is byte-identical to today.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any

from lemoncrow.core.foundation.memory_models import ArchivalPassage
from lemoncrow.infra.storage.vector import cosine_similarity

logger = logging.getLogger(__name__)

# HNSW removed (datasketch dropped); brute-force cosine is the permanent fallback.
_HNSW: Any = None

# Below this many eligible passages, exact cosine is faster and exact.
_ANN_MIN_PASSAGES = 16
# Newest passages always kept as exact candidates so just-stored memory is never
# invisible to the approximate index.
_RECENT_EXACT = 8
_ANN_OVERFETCH = 4
# A passage is only ANN-eligible when its stored provenance is a real embedder
# stamp; the legacy stub never carries a usable vector space.
_LEGACY_STUB = "legacy_stub"
# The index accumulates incrementally and never evicts; past this many nodes it
# is rebuilt from the current eligible set (the recency window) so the on-disk
# graph cannot grow without bound.
_MAX_GRAPH_NODES = 20000
# Bump when the persisted payload layout changes (old files are then ignored).
_PERSIST_VERSION = 1


def ann_retrieval_enabled(env: Any | None = None) -> bool:
    """Return True when the opt-in ANN retrieval path is enabled (default off)."""
    from lemoncrow.core.environment import bool_env

    return bool_env("LEMONCROW_ANN_RETRIEVAL", default=False, env=env)


def _ann_distance(a: Any, b: Any) -> float:
    return 1.0 - cosine_similarity(list(a), list(b))


class ArchivalAnnIndex:
    """Opt-in, persistent, incrementally-maintained ANN pre-filter (N5 + tail)."""

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._graph: Any = None
        self._member_ids: set[str] = set()
        self._model_id: str | None = None
        self._dim: int | None = None
        self._persist_path = Path(persist_path) if persist_path else None
        self._loaded = False

    def _eligible(
        self,
        passages: list[ArchivalPassage],
        *,
        model_id: str,
        dim: int,
    ) -> list[ArchivalPassage]:
        """Passages whose vector matches the live model-id and dim (N5 gate)."""
        if dim <= 0 or not model_id:
            return []
        out: list[ArchivalPassage] = []
        for passage in passages:
            if not passage.embedding or len(passage.embedding) != dim:
                continue
            if passage.embedding_provenance == _LEGACY_STUB:
                continue
            if passage.embedding_model != model_id:
                continue
            out.append(passage)
        return out

    def candidate_ids(
        self,
        query_embedding: list[float],
        passages: list[ArchivalPassage],
        *,
        model_id: str,
        dim: int,
        top_k: int,
    ) -> set[str] | None:
        """Return the ANN-narrowed candidate passage ids, or None to mean "all".

        ``None`` signals the caller to keep every passage (small set, ineligible
        set, or HNSW unavailable). With HNSW permanently removed, this method
        always returns ``None`` — the caller scores all passages via brute-force
        cosine, which is exact (not approximate) and covers every passage.
        """
        if not query_embedding or len(query_embedding) != dim:
            return None
        eligible = self._eligible(passages, model_id=model_id, dim=dim)
        if len(eligible) < _ANN_MIN_PASSAGES or _HNSW is None:
            return None
        recent_ids = {p.id for p in sorted(eligible, key=lambda p: (p.created_at, p.id), reverse=True)[:_RECENT_EXACT]}
        ann_ids = self._ann_neighbour_ids(query_embedding, eligible, top_k=top_k, model_id=model_id, dim=dim)
        if ann_ids is None:
            return None
        return ann_ids | recent_ids

    def _ann_neighbour_ids(
        self,
        query_embedding: list[float],
        eligible: list[ArchivalPassage],
        *,
        top_k: int,
        model_id: str,
        dim: int,
    ) -> set[str] | None:
        graph = self._ensure_graph(eligible, model_id=model_id, dim=dim)
        if graph is None:
            return None
        try:
            import numpy as np

            neighbours = graph.query(
                np.asarray(query_embedding, dtype="float64"),
                k=max(top_k * _ANN_OVERFETCH, top_k),
            )
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return None
        return {str(key) for key, _distance in neighbours}

    def _ensure_graph(
        self,
        eligible: list[ArchivalPassage],
        *,
        model_id: str,
        dim: int,
    ) -> Any:
        if _HNSW is None:
            return None
        with self._lock:
            # N5: a model-id / dim change invalidates the whole graph.
            if self._graph is not None and (self._model_id != model_id or self._dim != dim):
                self._reset()
            # Lazy one-time disk load for this process.
            if self._graph is None and not self._loaded:
                self._load_from_disk(model_id=model_id, dim=dim)
            # Bound unbounded growth: rebuild from the current eligible window.
            if self._graph is not None and len(self._member_ids) > _MAX_GRAPH_NODES:
                self._reset()
            if self._graph is None:
                self._graph = _HNSW(distance_func=_ann_distance)
                self._member_ids = set()
                self._model_id = model_id
                self._dim = dim
            added = self._insert_missing(eligible)
            if added:
                self._save_to_disk()
            return self._graph

    def _insert_missing(self, eligible: list[ArchivalPassage]) -> int:
        """Insert eligible passages not already in the graph; return how many."""
        import numpy as np

        added = 0
        try:
            for passage in eligible:
                if passage.id in self._member_ids:
                    continue
                self._graph.insert(passage.id, np.asarray(passage.embedding, dtype="float64"))
                self._member_ids.add(passage.id)
                added += 1
        except Exception:
            logging.exception("Recovered from broad exception handler")
            self._reset()
            return 0
        return added

    def _load_from_disk(self, *, model_id: str, dim: int) -> None:
        self._loaded = True
        path = self._persist_path
        if path is None or not path.exists():
            return
        try:
            data = pickle.loads(path.read_bytes())
        except Exception:
            logging.exception("Recovered from broad exception handler")
            return
        if (
            isinstance(data, dict)
            and data.get("version") == _PERSIST_VERSION
            and data.get("model_id") == model_id
            and data.get("dim") == dim
            and data.get("graph") is not None
        ):
            self._graph = data["graph"]
            self._member_ids = {str(x) for x in (data.get("member_ids") or [])}
            self._model_id = model_id
            self._dim = dim

    def _save_to_disk(self) -> None:
        path = self._persist_path
        if path is None or self._graph is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = pickle.dumps(
                {
                    "version": _PERSIST_VERSION,
                    "model_id": self._model_id,
                    "dim": self._dim,
                    "member_ids": list(self._member_ids),
                    "graph": self._graph,
                }
            )
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, path)
        except Exception:
            logging.exception("Recovered from broad exception handler")

    def _reset(self) -> None:
        self._graph = None
        self._member_ids = set()
        self._model_id = None
        self._dim = None

    def invalidate(self) -> None:
        with self._lock:
            self._reset()
            self._loaded = False


__all__ = ["ArchivalAnnIndex", "ann_retrieval_enabled"]
