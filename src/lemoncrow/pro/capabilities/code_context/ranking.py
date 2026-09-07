"""LemonCrow symbol-ranking: IDF weighting + the multi-signal symbol scorer.

The ``adjustment`` scorer -- IDF-weighted term coverage, name/qualified/path
prefix bonuses, per-token name-match boosts, call-graph centrality, seed-import
proximity -- re-ranks candidates after channel/FTS retrieval.

MEASURED IMPACT (full bench A/B, 2026-07-13, 6058 queries): this scorer is worth
**+0.003 overall MRR** (0.6685 with vs 0.6656 without) -- ~+0.02 on the semantic/
sessions NL golds, ~0 (or slightly negative) on exact-name golds. It is NOT the
retrieval moat: the ~0.67 vs commodity ~0.4 gap comes from the channel
architecture in engine._search_symbols_local (exact-name channels, IDF-pruned
discriminative FTS, multi-channel candidate generation), which is NOT here.

Published in full like the rest of the engine — it is cheap tuning, not what
defends the product.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# IDF-weighted lexical coverage constants (see score_symbol_row / idf_weight).
# A name/signature/path match on a rare, discriminative token counts for far more
# than one on a corpus-common token; the cap/floor keep a common term counting a
# little and an ultra-rare one from running away.
_COVERAGE_MULT = 20.0  # score per fully-weighted (rare, on-topic) covered term
_COVERAGE_IDF_CAP = 8.0  # upper bound on a single term's IDF weight
_COVERAGE_IDF_SLOPE = 4.0  # weight = clamp(SLOPE * normalized_idf, 0.3, CAP)

__all__ = [
    "SymbolScoringContext",
    "idf_weight",
    "score_symbol_row",
]


def idf_weight(normalized_idf: float) -> float:
    """Map a corpus-normalized IDF (0..1) to a bounded per-term coverage weight."""
    return min(_COVERAGE_IDF_CAP, max(0.3, _COVERAGE_IDF_SLOPE * normalized_idf))


@dataclass
class SymbolScoringContext:
    """Per-query state the scorer closes over (built once in _search_symbols_local).

    ``tokenize`` / ``is_test_path`` are supplied by the engine so this module does
    not import back into it (no cycle) and does not duplicate tokenization.
    ``importing_files_for_boost`` is passed by reference: the engine may still be
    populating it when the context is built; the set object is shared.
    """

    kind_boosts: Mapping[str, float]
    term_idf: Mapping[str, float]
    terms: Sequence[str]
    term_set: set[str]
    normalized_query_lower: str
    centrality_map: Mapping[str, float]
    importing_files_for_boost: set[str]
    query_mentions_tests: bool
    tokenize: Callable[[str], list[str]]
    is_test_path: Callable[[str], bool]


def score_symbol_row(row: Mapping[str, Any], ctx: SymbolScoringContext) -> float:
    """Score one raw candidate DB row (all columns TEXT -> str) for the query.

    Pure function of ``row`` once ``ctx.importing_files_for_boost`` is populated;
    computed once per symbol per query in the search hot loop.
    """
    symbol_name = str(row["symbol_name"])
    qualified_name = str(row["qualified_name"])
    file_path = str(row["file_path"])
    score = ctx.kind_boosts.get(str(row["kind"]), 0.0)
    symbol_name_lower = symbol_name.lower()
    qualified_name_lower = qualified_name.lower()
    # Fold the doc summary into the matched text so a natural-language query can
    # reach a symbol through its own prose description ("create and configure a
    # Flask app" -> create_app). IDF weighting below keeps common docstring words
    # from inflating the score; only rare, on-topic terms carry real weight.
    lexical_text = f"{symbol_name} {qualified_name} {row['signature']} {row['doc_summary'] or ''}".lower()
    file_path_lower = file_path.lower()
    # Basename-without-extension via slicing (Path(...).stem builds a pathlib
    # object per row, which dominated the profile); identical for '/'-separated
    # stored paths.
    _basename = file_path_lower[file_path_lower.rfind("/") + 1 :]
    _dot = _basename.rfind(".")
    file_name_stem = _basename[:_dot] if _dot > 0 else _basename
    coverage = sum(ctx.term_idf.get(term, 1.0) for term in ctx.terms[:8] if term and term in lexical_text)
    score += coverage * _COVERAGE_MULT
    if symbol_name_lower.startswith(ctx.normalized_query_lower):
        score += 24.0
    if qualified_name_lower.startswith(ctx.normalized_query_lower):
        score += 20.0
    if ctx.normalized_query_lower in file_name_stem:
        score += 22.0
    elif file_name_stem.startswith(ctx.normalized_query_lower[: max(1, min(len(ctx.normalized_query_lower), 8))]):
        score += 10.0
    for term in ctx.terms[:6]:
        if term and term in file_path_lower:
            score += 6.0 * ctx.term_idf.get(term, 1.0)
    # Per-token name match: reward a query TOKEN matching the symbol's OWN name
    # tokens, so multi-term/regex queries ("select_format|CAST") still surface the
    # exactly-named symbol instead of losing to body-coverage / kind-boost noise.
    name_tokens = ctx.tokenize(symbol_name)
    if name_tokens:
        matched = sum(1 for token in name_tokens if token in ctx.term_set)
        if matched == len(name_tokens):
            # Longer multi-token names that fully match are more discriminative
            # ("RewriteContext" vs "get") -- amplify the complete-match bonus.
            base_bonus = 28.0 + 6.0 * len(name_tokens)
            if len(name_tokens) >= 2:
                base_bonus += 12.0 * (len(name_tokens) - 1)
            score += base_bonus
        elif matched:
            score += 9.0 * matched
    # Structural importance (call-graph eigenvector centrality / PageRank),
    # normalized 0..1: central core symbols outrank peripheral textual matches.
    cscore = ctx.centrality_map.get(symbol_name_lower)
    if cscore is None:
        cscore = ctx.centrality_map.get(qualified_name_lower, 0.0)
    score += cscore * 30.0
    # Files that explicitly import the seed are closely related.
    if file_path in ctx.importing_files_for_boost:
        score += 50.0
    if ctx.is_test_path(file_path) and not ctx.query_mentions_tests:
        score -= 90.0
    return score
