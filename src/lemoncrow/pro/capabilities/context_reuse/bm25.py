"""BM25 retrieval with stopword filtering and simple stemming."""

from __future__ import annotations

import math
import re

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
        "none",
        "true",
        "false",
        "not",
        "no",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "return",
        "import",
        "pass",
        "def",
        "class",
        "lambda",
        "yield",
        "assert",
        "raise",
        "except",
        "finally",
        "try",
        "as",
        "del",
        "global",
        "nonlocal",
        "async",
        "await",
    }
)


def _stem(word: str) -> str:
    """Very lightweight English stemmer (strips common suffixes)."""
    for suffix in ("ing", "tion", "ation", "ness", "ment", "ity", "ly", "ed", "er", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenise(text: str) -> list[str]:
    """
    Tokenise and filter text for BM25 indexing.

    - Lowercases
    - Splits on non-alphanumeric (camelCase → two tokens via split on uppercase boundary)
    - Removes stopwords
    - Applies lightweight stemmer
    """
    # Split camelCase and snake_case
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-z][a-z0-9]*", text.lower())
    return [_stem(t) for t in tokens if t not in _STOPWORDS and len(t) >= 2]


def build_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    if N == 0:
        return {}
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()}


def bm25_score(
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
        idf_val = idf.get(term, 0.0)
        if idf_val <= 0:
            continue
        tf = freq_map.get(term, 0)
        numerator = tf * (_K1 + 1)
        denominator = tf + _K1 * (1 - _B + _B * dl / max(avg_dl, 1))
        score += idf_val * (numerator / denominator)
    return score
