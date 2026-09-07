"""Near-duplicate detection and semantic deduplication for context compression."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from lemoncrow.core.capabilities._optional_runtime import blake3
from lemoncrow.core.foundation._minhash import MinHash

# Levenshtein is computed over at most this many leading chars for speed. The
# similarity denominator MUST use the same cap: otherwise two long strings that
# share a 120-char prefix but diverge after score a spurious ~1.0.
_EDIT_DISTANCE_MAX_LEN = 120


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (capped at len(a)+len(b) for speed)."""
    if a == b:
        return 0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0
    # Use truncated strings to bound runtime
    a, b = a[:_EDIT_DISTANCE_MAX_LEN], b[:_EDIT_DISTANCE_MAX_LEN]
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def _similarity(a: str, b: str) -> float:
    """Return similarity in [0, 1] (1 = identical)."""
    if a == b:
        return 1.0
    max_len = max(len(a), len(b), 1)
    # _edit_distance only compares the first _EDIT_DISTANCE_MAX_LEN chars for
    # speed, so any characters beyond that window are never measured. Count them
    # as differing — otherwise two long strings that merely share a prefix score
    # a spurious ~1.0. Distance and denominator now span the full length.
    unmeasured = max(0, max_len - _EDIT_DISTANCE_MAX_LEN)
    d = _edit_distance(a, b) + unmeasured
    return 1.0 - d / max_len


def deduplicate_tool_outputs(
    events: list[dict[str, Any]],
    *,
    threshold: float = 0.80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split events into (kept, dropped) based on output similarity.

    Consecutive events of the same kind with very similar summaries are
    deduplicated — only the first occurrence is kept.
    """
    if not events:
        return [], []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    last_by_kind: dict[str, str] = {}
    exact_seen: set[tuple[str, str]] = set()
    minhash_by_kind: dict[str, list[Any]] = {}

    for ev in events:
        kind = str(ev.get("kind", ""))
        summary = str(ev.get("summary", ""))
        digest = _content_digest(kind, summary, ev.get("payload", {}))

        exact_key = (kind, digest)
        if exact_key in exact_seen:
            dropped.append(ev)
            continue

        is_near_duplicate = False
        mh = _build_minhash(summary)
        if mh is not None:
            for prior_mh in minhash_by_kind.get(kind, []):
                if mh.jaccard(prior_mh) >= threshold:
                    is_near_duplicate = True
                    break

        last_summary = last_by_kind.get(kind)
        if not is_near_duplicate and last_summary is not None:
            is_near_duplicate = _similarity(summary, last_summary) >= threshold

        if is_near_duplicate:
            dropped.append(ev)
        else:
            kept.append(ev)
            last_by_kind[kind] = summary
            exact_seen.add(exact_key)
            if mh is not None:
                minhash_by_kind.setdefault(kind, []).append(mh)
    return kept, dropped


def _content_digest(kind: str, summary: str, payload: Any) -> str:
    try:
        payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        logging.exception("Recovered from broad exception handler")
        payload_str = str(payload)
    text = f"{kind}\n{summary}\n{payload_str}".encode("utf-8", errors="replace")
    if blake3 is not None:
        return blake3(text).hexdigest()
    return hashlib.sha256(text).hexdigest()


def _build_minhash(text: str) -> Any | None:
    if MinHash is None:
        return None
    mh = MinHash(num_perm=64)
    for token in _shingles(text):
        mh.update(token.encode("utf-8", errors="replace"))
    return mh


def _shingles(text: str, width: int = 3) -> list[str]:
    compact = " ".join(text.lower().split())
    if len(compact) <= width:
        return [compact]
    return [compact[i : i + width] for i in range(0, len(compact) - width + 1)]
