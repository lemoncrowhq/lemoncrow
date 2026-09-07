"""Cross-file symbol resolution and BM25 search over cached file summaries."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .indexer import FileIndex

# BM25 tuning constants
_K1 = 1.5
_B = 0.75

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "through",
        "during",
        "self",
        "cls",
        "def",
        "class",
        "return",
        "import",
        "pass",
        "none",
        "true",
        "false",
    }
)


def _tokenise(text: str) -> list[str]:
    tokens = re.findall(r"[a-z][a-z0-9]*", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def _build_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_dl: float,
) -> float:
    dl = len(doc_tokens)
    freq_map: dict[str, int] = {}
    for t in doc_tokens:
        freq_map[t] = freq_map.get(t, 0) + 1
    score = 0.0
    for term in query_tokens:
        if term not in idf:
            continue
        tf = freq_map.get(term, 0)
        numerator = tf * (_K1 + 1)
        denominator = tf + _K1 * (1 - _B + _B * dl / max(avg_dl, 1))
        score += idf[term] * (numerator / denominator)
    return score


class SymbolIndex:
    """
    Cross-file symbol lookup and BM25 full-text search over cached summaries.

    All data comes from :class:`FileIndex` — no disk scanning is performed.

    IDF table is precomputed lazily on the first search call and cached for the
    lifetime of this instance.  This avoids the O(|corpus|) rebuild overhead on
    every query, which is significant for large indexes.
    """

    def __init__(self, index: FileIndex) -> None:
        self._index = index
        # Lazily-built IDF cache: (idf_dict, avg_dl, snapshot_key)
        self._idf_cache: tuple[dict[str, float], float, str] | None = None

    # ------------------------------------------------------------------
    # IDF precomputation
    # ------------------------------------------------------------------

    def _ensure_idf(
        self,
    ) -> tuple[dict[str, float], float, list[tuple[str, dict[str, Any], list[str]]]]:
        """
        Return (idf, avg_dl, docs) — building from scratch only when the index
        has changed.  Staleness is keyed on each entry's content hash, so an
        edit that leaves the entry count unchanged still rebuilds the IDF.
        """
        entries = self._index.all_entries()
        snapshot_key = hashlib.sha1(
            "\n".join(f"{path}\x00{entry.get('content_hash', '')}" for path, entry in sorted(entries.items())).encode()
        ).hexdigest()  # content-aware change detector

        if self._idf_cache is not None and self._idf_cache[2] == snapshot_key:
            idf, avg_dl, _ = self._idf_cache
            # Rebuild docs list (cheap, no IDF recompute)
            docs = self._build_docs(entries)
            return idf, avg_dl, docs

        docs = self._build_docs(entries)
        corpus = [d[2] for d in docs]
        idf = _build_idf(corpus)
        avg_dl = sum(len(d) for d in corpus) / max(len(corpus), 1)
        self._idf_cache = (idf, avg_dl, snapshot_key)
        return idf, avg_dl, docs

    @staticmethod
    def _build_docs(
        entries: dict[str, dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any], list[str]]]:
        docs = []
        for path, entry in entries.items():
            doc_text = " ".join(
                [
                    " ".join(entry.get("symbols", [])),
                    " ".join(entry.get("exports", [])),
                    " ".join(entry.get("imports", [])),
                    entry.get("ast_summary", ""),
                ]
            )
            docs.append((path, entry, _tokenise(doc_text)))
        return docs

    # ------------------------------------------------------------------
    # Symbol resolution
    # ------------------------------------------------------------------

    def resolve_symbol(self, name: str) -> list[dict[str, Any]]:
        """
        Find all definitions of *name* across the cache.

        Returns a ranked list (exact-name matches first, then substring).
        """
        name_lower = name.lower()
        exact: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        for path, entry in self._index.all_entries().items():
            for sym in entry.get("symbol_details", []):
                sym_name: str = str(sym.get("name", ""))
                if sym_name.lower() == name_lower:
                    exact.append({"path": path, **sym})
                elif name_lower in sym_name.lower():
                    partial.append({"path": path, **sym})
        return exact + partial

    def all_symbols(self, *, language: str | None = None) -> list[dict[str, Any]]:
        """Return every symbol across all cached files."""
        results: list[dict[str, Any]] = []
        for path, entry in self._index.all_entries().items():
            if language and entry.get("language") != language:
                continue
            for sym in entry.get("symbol_details", []):
                results.append({"path": path, **sym})
        return results

    # ------------------------------------------------------------------
    # BM25 search over file summaries (precomputed IDF)
    # ------------------------------------------------------------------

    def bm25_search(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """BM25-ranked search over file symbol names, exports, and imports.

        IDF is precomputed and cached -- repeated queries are O(query_tokens x docs).
        """
        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        idf, avg_dl, docs = self._ensure_idf()
        if not docs:
            return []

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for path, entry, doc_tokens in docs:
            score = _bm25_score(query_tokens, doc_tokens, idf, avg_dl)
            if score > 0:
                scored.append((score, path, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "path": path,
                "score": round(score, 4),
                "language": entry.get("language"),
                "symbols": entry.get("symbols", [])[:10],
                "exports": entry.get("exports", [])[:10],
            }
            for score, path, entry in scored[:limit]
        ]

    # ------------------------------------------------------------------
    # Change impact analysis
    # ------------------------------------------------------------------

    def change_impact(self, modified_path: str, *, max_transitive_depth: int = 3) -> dict[str, Any]:
        """
        Estimate the blast radius of modifying *modified_path*.

        Returns:
            direct_importers      - files that directly import the modified file
            transitive_importers  - files reachable via the dependency graph
            affected_tests        - test files in the transitive closure
            risk_level            - 'low' | 'medium' | 'high' | 'critical'
        """
        rdeps = self._index.build_reverse_deps()

        direct: list[str] = rdeps.get(modified_path, [])
        visited: set[str] = set(direct)
        frontier: set[str] = set(direct)
        transitive: list[str] = []

        for _ in range(max_transitive_depth - 1):
            next_frontier: set[str] = set()
            for f in frontier:
                for dep in rdeps.get(f, []):
                    if dep not in visited:
                        visited.add(dep)
                        next_frontier.add(dep)
                        transitive.append(dep)
            frontier = next_frontier
            if not frontier:
                break

        affected_tests = [f for f in visited if "/test" in f or "test_" in f.split("/")[-1]]
        total_impact = len(direct) + len(transitive)
        if total_impact == 0:
            risk = "low"
        elif total_impact <= 3:
            risk = "medium"
        elif total_impact <= 10:
            risk = "high"
        else:
            risk = "critical"

        return {
            "modified_file": modified_path,
            "direct_importers": direct,
            "transitive_importers": transitive,
            "affected_tests": affected_tests,
            "risk_level": risk,
        }
