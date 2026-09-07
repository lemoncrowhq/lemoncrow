"""Precision / recall / F1 evaluation harness for ranked retrieval (N12).

Reusable, dependency-light scorer mirroring the shape of the memory retrieval
eval (``tests/core/test_retriever_eval.py``): given a ranking of result ids and a
labelled set of relevant ids, it computes precision@k, recall@k, and F1 so a
fusion-weight change can be *measured* rather than guessed at.

Pure and deterministic -- no I/O, no model calls -- so the same labelled set and
ranking always score identically. It is generic over the id type: callers feed
symbol ids, block ids, file paths, anything hashable. Tuning loop:

    weights -> rank candidates -> score against ground truth -> compare F1.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionRecallF1:
    """Precision/recall/F1 for one query at a fixed cutoff ``k``."""

    k: int
    true_positives: int
    retrieved: int
    relevant: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvalCase[T]:
    """One labelled query: a ranking plus the set of relevant ids."""

    case_id: str
    ranking: Sequence[T]
    relevant: frozenset[T]


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def score_ranking[T](ranking: Sequence[T], relevant: Iterable[T], *, k: int) -> PrecisionRecallF1:
    """Score one ranking against its relevant set at cutoff ``k``.

    Precision = TP / retrieved (the top-``k`` slice, deduped by first occurrence),
    recall = TP / relevant, F1 = harmonic mean. ``k`` must be positive. An empty
    relevant set yields recall 0.0 (no relevant item can be recalled); an empty
    retrieval yields precision 0.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")
    relevant_set: frozenset[T] = frozenset(relevant)

    # Top-k slice, deduped by first occurrence so repeats never inflate counts.
    seen: set[T] = set()
    top: list[T] = []
    for item in ranking:
        if item in seen:
            continue
        seen.add(item)
        top.append(item)
        if len(top) >= k:
            break

    retrieved = len(top)
    true_positives = sum(1 for item in top if item in relevant_set)
    relevant_count = len(relevant_set)
    precision = true_positives / retrieved if retrieved else 0.0
    recall = true_positives / relevant_count if relevant_count else 0.0
    return PrecisionRecallF1(
        k=k,
        true_positives=true_positives,
        retrieved=retrieved,
        relevant=relevant_count,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
    )


@dataclass(frozen=True)
class AggregateScore:
    """Macro-averaged precision/recall/F1 across a set of cases."""

    k: int
    case_count: int
    precision: float
    recall: float
    f1: float
    per_case: tuple[PrecisionRecallF1, ...]


def evaluate_cases[T](cases: Sequence[EvalCase[T]], *, k: int) -> AggregateScore:
    """Macro-average precision/recall/F1 over ``cases`` at cutoff ``k``.

    Macro-averaging (mean of per-case metrics) weights every query equally,
    matching how the memory retrieval eval reports recall@k/MRR. With no cases
    every metric is 0.0.
    """
    per_case = tuple(score_ranking(case.ranking, case.relevant, k=k) for case in cases)
    count = len(per_case)
    if not count:
        return AggregateScore(k=k, case_count=0, precision=0.0, recall=0.0, f1=0.0, per_case=())
    precision = sum(item.precision for item in per_case) / count
    recall = sum(item.recall for item in per_case) / count
    f1 = sum(item.f1 for item in per_case) / count
    return AggregateScore(
        k=k,
        case_count=count,
        precision=precision,
        recall=recall,
        f1=f1,
        per_case=per_case,
    )


__all__ = [
    "AggregateScore",
    "EvalCase",
    "PrecisionRecallF1",
    "evaluate_cases",
    "score_ranking",
]
