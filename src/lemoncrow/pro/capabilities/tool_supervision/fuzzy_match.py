"""Fuzzy matching for the rich-edit path — backed by diff-match-patch."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher

from diff_match_patch import diff_match_patch as _DMP

# --------------------------------------------------------------------------- #
# Public data types (kept for backward compat)                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FuzzyCandidate:
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    distance: int
    ratio: float


from lemoncrow.core.capabilities.tool_supervision_contract import (  # noqa: E402
    FuzzyAmbiguousMatchError as FuzzyAmbiguousMatchError,
)

# --------------------------------------------------------------------------- #
# Text normalization helpers                                                   #
# --------------------------------------------------------------------------- #

_WS_RUN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([:;,)\]\}])")


def normalize_for_fuzzy(text: str) -> str:
    """Normalize whitespace to tolerate indentation/trailing differences."""
    lines = text.splitlines()
    normalized_lines = []
    for line in lines:
        expanded = line.expandtabs(8).rstrip()
        collapsed = _WS_RUN.sub(" ", expanded).strip()
        normalized_lines.append(_SPACE_BEFORE_PUNCT.sub(r"\1", collapsed))
    return "\n".join(normalized_lines)


# --------------------------------------------------------------------------- #
# Levenshtein (kept for callers / tests that import it directly)              #
# --------------------------------------------------------------------------- #


def bounded_levenshtein(a: str, b: str, max_distance: int) -> int | None:
    """Return edit distance if <= max_distance, else None."""
    if max_distance < 0:
        return None
    if abs(len(a) - len(b)) > max_distance:
        return None
    if a == b:
        return 0

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        row_min = current[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if current[j] < row_min:
                row_min = current[j]
        if row_min > max_distance:
            return None
        previous = current

    distance = previous[-1]
    return distance if distance <= max_distance else None


# --------------------------------------------------------------------------- #
# Core fuzzy replace — diff-match-patch backed                                #
# --------------------------------------------------------------------------- #

_DMP_THRESHOLD = 0.5

# Minimum similarity between the matched window and old_string before accepting a
# DMP-located replacement.  Values below this floor indicate a bad guess that
# would corrupt the file — we reject and surface a useful error instead.
_FUZZY_SIMILARITY_FLOOR = 0.90


def _make_dmp(content_len: int) -> _DMP:
    dmp = _DMP()
    dmp.Match_Threshold = _DMP_THRESHOLD
    dmp.Match_Distance = max(content_len, 1000)
    dmp.Match_MaxBits = 0  # no pattern-size limit (default 32 breaks long patterns)
    return dmp


def _preserve_window_newline(window: str, new_string: str) -> str:
    """Keep the replaced window's trailing newline when new_string lacks one.

    The fuzzy path replaces whole source lines, so the window always ends with
    a newline (unless at EOF). Dropping it would glue the following line onto
    the last line of new_string — e.g. a function body onto its signature —
    which can still parse and therefore slip past the parse gate.
    An empty new_string is a deletion and intentionally removes the newline.
    """
    if window.endswith("\n") and new_string and not new_string.endswith("\n"):
        return new_string + "\n"
    return new_string


def _find_exact_normalized_candidates(content: str, old_string: str) -> list[FuzzyCandidate]:
    lines = content.splitlines(keepends=True)
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    norm_old = normalize_for_fuzzy(old_string)
    n_old_lines = max(1, len(old_string.splitlines()))
    candidates: list[FuzzyCandidate] = []
    for start_line_idx in range(len(lines)):
        consumed = 0
        end_line_idx = start_line_idx
        while consumed < n_old_lines and end_line_idx < len(lines):
            if lines[end_line_idx].strip():
                consumed += 1
            end_line_idx += 1
        if consumed < n_old_lines:
            continue
        window = "".join(lines[start_line_idx:end_line_idx])
        if normalize_for_fuzzy(window) != norm_old:
            continue
        candidates.append(
            FuzzyCandidate(
                start_line=start_line_idx + 1,
                end_line=end_line_idx,
                start_offset=offsets[start_line_idx],
                end_offset=offsets[end_line_idx],
                distance=0,
                ratio=1.0,
            )
        )
    return candidates


def _count_end_line_idx(lines: list[str], start_idx: int, n_old_lines: int) -> int:
    """Count-based window end: consume n_old_lines non-blank lines from start_idx."""
    consumed = 0
    end_idx = start_idx
    while consumed < n_old_lines and end_idx < len(lines):
        if lines[end_idx].strip():
            consumed += 1
        end_idx += 1
    return end_idx


def _anchor_end_line_idx(
    lines: list[str],
    start_idx: int,
    n_old_lines: int,
    old_string: str,
) -> int:
    """R3: locate the window end by finding the last non-blank line of old_string.

    Candidate anchor lines are ranked by distance from the expected window end
    (start + n_old_lines), so a generic last line (e.g. ")") occurring early in
    the window cannot truncate the replacement region.
    Falls back to the count-based approach when the anchor can't be located.
    """
    old_lines = old_string.splitlines()
    last_anchor_raw = next((line for line in reversed(old_lines) if line.strip()), None)
    if last_anchor_raw:
        norm_last = normalize_for_fuzzy(last_anchor_raw)
        expected_end = start_idx + n_old_lines  # exclusive end if no drift
        search_end = min(start_idx + n_old_lines * 3 + 2, len(lines))
        candidates = [
            i + 1  # exclusive end (line index after the last anchor)
            for i in range(start_idx, search_end)
            if normalize_for_fuzzy(lines[i].rstrip("\n")) == norm_last
        ]
        if candidates:
            return min(candidates, key=lambda end: abs(end - expected_end))
    return _count_end_line_idx(lines, start_idx, n_old_lines)


_FUZZY_AMBIGUITY_MARGIN = 0.05


def _overlapping(a: FuzzyCandidate, b: FuzzyCandidate) -> bool:
    """Whether two candidate line windows share any lines.

    Overlapping windows are boundary variants of the SAME target (e.g. similar
    import lines shifting the window start: 11-92 / 14-92 / 18-92), not genuine
    duplicates -- ambiguity protection is for distinct locations only.
    """
    return a.start_line <= b.end_line and b.start_line <= a.end_line


def _fuzzy_window_candidates(
    content: str,
    lines: list[str],
    offsets: list[int],
    old_string: str,
) -> list[FuzzyCandidate]:
    """Windows whose normalized similarity to old_string clears the floor.

    Scans every plausible start line (a cheap first-line pre-filter keeps the
    full-window scoring proportional to the number of near matches, not the file
    size) and returns candidates sorted best-first. DMP alone returns only the
    first acceptable location, which silently mis-anchors onto the first of
    several similar blocks (duplicated text); ranking all candidates lets the
    caller pick the global best and detect ties.
    """
    old_lines = old_string.splitlines()
    first_anchor = next((line for line in old_lines if line.strip()), "")
    norm_first = normalize_for_fuzzy(first_anchor)
    norm_old = normalize_for_fuzzy(old_string)
    n_old_lines = max(1, len(old_lines))

    def _similarity(start_idx: int, end_idx: int) -> float:
        window = content[offsets[start_idx] : offsets[end_idx]]
        return SequenceMatcher(None, norm_old, normalize_for_fuzzy(window), autojunk=False).ratio()

    candidates: list[FuzzyCandidate] = []
    for start_idx in range(len(lines)):
        norm_line = normalize_for_fuzzy(lines[start_idx].rstrip("\n"))
        if norm_first and SequenceMatcher(None, norm_first, norm_line).quick_ratio() < 0.6:
            continue
        ends = {
            _anchor_end_line_idx(lines, start_idx, n_old_lines, old_string),
            _count_end_line_idx(lines, start_idx, n_old_lines),
        }
        end_idx = max(ends, key=lambda end: _similarity(start_idx, end))
        ratio = _similarity(start_idx, end_idx)
        if ratio >= _FUZZY_SIMILARITY_FLOOR:
            candidates.append(
                FuzzyCandidate(
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    start_offset=offsets[start_idx],
                    end_offset=offsets[end_idx],
                    distance=0,
                    ratio=ratio,
                )
            )
    candidates.sort(key=lambda candidate: candidate.ratio, reverse=True)
    return candidates


def apply_fuzzy_replace(content: str, old_string: str, new_string: str) -> tuple[str, int, int]:
    """Fuzzy-replace old_string with new_string inside content.

    Matching ladder (strict → loose):
      R2  whitespace/typography-normalized exact, unique match
      R3  anchor match: DMP locates start; last non-blank line of old_string
          pins the window end (replaces fragile "count N lines" approach)
      R5  DMP location with similarity gate (≥ _FUZZY_SIMILARITY_FLOOR)

    Returns (new_content, 1-based line_start, 1-based line_end).
    Raises ValueError when no match meets the similarity floor.
    """
    lines = content.splitlines(keepends=True)
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    # R2: exact normalized, unique
    exact_candidates = _find_exact_normalized_candidates(content, old_string)
    if len(exact_candidates) == 1:
        candidate = exact_candidates[0]
        replacement = _preserve_window_newline(content[candidate.start_offset : candidate.end_offset], new_string)
        new_content = content[: candidate.start_offset] + replacement + content[candidate.end_offset :]
        return new_content, candidate.start_line, candidate.end_line
    if len(exact_candidates) > 1:
        # Multiple equally good targets — surface the ambiguity instead of
        # letting DMP pick one arbitrarily.
        raise FuzzyAmbiguousMatchError(exact_candidates)

    # R3 / R5: prefer the globally best-scoring window over DMP's first hit.
    # DMP returns only the first acceptable location, so with duplicated blocks
    # it silently anchors onto the wrong one even when old_string targets a
    # later block. Rank all candidates, take the best, and refuse on a true tie.
    candidates = _fuzzy_window_candidates(content, lines, offsets, old_string)
    if candidates:
        best = candidates[0]
        rivals = [
            other
            for other in candidates[1:]
            if not _overlapping(best, other) and best.ratio - other.ratio <= _FUZZY_AMBIGUITY_MARGIN
        ]
        if rivals:
            raise FuzzyAmbiguousMatchError([best, *rivals])
        replacement = _preserve_window_newline(content[best.start_offset : best.end_offset], new_string)
        new_content = content[: best.start_offset] + replacement + content[best.end_offset :]
        return new_content, best.start_line, best.end_line

    # Fallback: no pre-filtered window cleared the floor (e.g. the first line of
    # old_string diverges too much for the cheap pre-filter). Use DMP's single
    # location for a best-effort match and a helpful similarity message.
    dmp = _make_dmp(len(content))
    match_char = dmp.match_main(content, old_string, 0)
    if match_char == -1:
        raise ValueError("old_string not found in file")

    # Check for a second DMP match — DMP silently picks the first; a second
    # NON-OVERLAPPING hit means the old_string is ambiguous and we must not
    # guess (an overlapping second location is a boundary variant of the same
    # target, not a duplicate).
    second_match = dmp.match_main(content, old_string, match_char + 1)
    if second_match != -1 and second_match >= match_char + max(1, len(old_string)):
        lines_tmp = content.splitlines(keepends=True)
        offsets_tmp: list[int] = [0]
        for _l in lines_tmp:
            offsets_tmp.append(offsets_tmp[-1] + len(_l))
        first_line = max(0, bisect_right(offsets_tmp, match_char) - 1) + 1
        second_line = max(0, bisect_right(offsets_tmp, second_match) - 1) + 1
        n = max(1, len(old_string.splitlines()))
        raise FuzzyAmbiguousMatchError(
            [
                FuzzyCandidate(first_line, first_line + n - 1, match_char, match_char, 0, 1.0),
                FuzzyCandidate(second_line, second_line + n - 1, second_match, second_match, 0, 1.0),
            ]
        )

    start_line_idx = max(0, bisect_right(offsets, match_char) - 1)
    n_old_lines = max(1, len(old_string.splitlines()))
    norm_old = normalize_for_fuzzy(old_string)

    def _window_similarity(end_idx: int) -> float:
        window = content[offsets[start_line_idx] : offsets[end_idx]]
        return SequenceMatcher(None, norm_old, normalize_for_fuzzy(window), autojunk=False).ratio()

    window_ends = {
        _anchor_end_line_idx(lines, start_line_idx, n_old_lines, old_string),
        _count_end_line_idx(lines, start_line_idx, n_old_lines),
    }
    end_line_idx = max(window_ends, key=_window_similarity)

    similarity = _window_similarity(end_line_idx)
    if similarity < _FUZZY_SIMILARITY_FLOOR:
        raise ValueError(
            f"old_string not found in file "
            f"(best match similarity {similarity:.2f} < {_FUZZY_SIMILARITY_FLOOR:.2f}; "
            "re-read the file and supply exact disk content as old_string)"
        )

    region_start = offsets[start_line_idx]
    region_end = offsets[end_line_idx]
    replacement = _preserve_window_newline(content[region_start:region_end], new_string)
    new_content = content[:region_start] + replacement + content[region_end:]
    return new_content, start_line_idx + 1, end_line_idx


def find_best_fuzzy_window(content: str, old_string: str) -> FuzzyCandidate | None:
    """Best-scoring line window in ``content`` matching ``old_string``, or ``None``.

    Locate-only reuse of the R5 ladder rung from :func:`apply_fuzzy_replace` (normalized
    similarity >= ``_FUZZY_SIMILARITY_FLOOR``), for callers that need a fuzzy anchor
    without the splice -- e.g. matching against a minified projection instead of raw
    disk content. Raises :class:`FuzzyAmbiguousMatchError` on a genuine tie, same as
    the full ladder.
    """
    lines = content.splitlines(keepends=True)
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    candidates = _fuzzy_window_candidates(content, lines, offsets, old_string)
    if not candidates:
        return None
    best = candidates[0]
    rivals = [
        other
        for other in candidates[1:]
        if not _overlapping(best, other) and best.ratio - other.ratio <= _FUZZY_AMBIGUITY_MARGIN
    ]
    if rivals:
        raise FuzzyAmbiguousMatchError([best, *rivals])
    return best


def find_fuzzy_candidates(
    content: str,
    old_string: str,
    *,
    distance_ratio: float = 0.05,
) -> list[FuzzyCandidate]:
    """Find candidate line windows — now backed by DMP. Returns 0 or 1 result."""
    dmp = _make_dmp(len(content))
    match_char = dmp.match_main(content, old_string, 0)
    if match_char == -1:
        return []

    lines = content.splitlines(keepends=True)
    offsets: list[int] = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    start_line_idx = max(0, bisect_right(offsets, match_char) - 1)
    n_old_lines = max(1, len(old_string.splitlines()))
    end_line_idx = min(start_line_idx + n_old_lines, len(lines))

    norm_old = normalize_for_fuzzy(old_string)
    window = "".join(lines[start_line_idx:end_line_idx])
    norm_window = normalize_for_fuzzy(window)
    ratio = SequenceMatcher(None, norm_old, norm_window, autojunk=False).ratio()

    return [
        FuzzyCandidate(
            start_line=start_line_idx + 1,
            end_line=end_line_idx,
            start_offset=offsets[start_line_idx],
            end_offset=offsets[end_line_idx],
            distance=0,
            ratio=ratio,
        )
    ]


__all__ = [
    "_FUZZY_SIMILARITY_FLOOR",
    "FuzzyAmbiguousMatchError",
    "FuzzyCandidate",
    "_anchor_end_line_idx",
    "_count_end_line_idx",
    "_find_exact_normalized_candidates",
    "_preserve_window_newline",
    "apply_fuzzy_replace",
    "bounded_levenshtein",
    "find_best_fuzzy_window",
    "find_fuzzy_candidates",
    "normalize_for_fuzzy",
]
