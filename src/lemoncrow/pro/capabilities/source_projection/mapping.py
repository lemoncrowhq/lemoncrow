"""Mapping helpers for compact source projections."""

from __future__ import annotations

from bisect import bisect_right

from lemoncrow.pro.capabilities.source_projection.models import (
    ProjectionMapping,
    ProjectionSegment,
    SourceRange,
)


def build_compact_mapping(
    *,
    source_text: str,
    projected_text: str,
    path: str,
    lang: str,
) -> ProjectionMapping | None:
    """Build a fail-closed source mapping for a compact projection.

    The compact projection must preserve non-whitespace characters in order.
    Any non-whitespace mismatch causes mapping generation to abort.
    """
    segments: list[ProjectionSegment] = []
    line_offsets = _line_offsets(source_text)
    source_pos = 0
    projected_pos = 0
    segment_index = 0

    while source_pos < len(source_text) and projected_pos < len(projected_text):
        if source_text[source_pos] == projected_text[projected_pos]:
            source_start = source_pos
            projected_start = projected_pos
            while (
                source_pos < len(source_text)
                and projected_pos < len(projected_text)
                and source_text[source_pos] == projected_text[projected_pos]
            ):
                source_pos += 1
                projected_pos += 1
            segments.append(
                ProjectionSegment(
                    segment_id=f"seg:{segment_index:04d}",
                    kind="exact",
                    source=_source_range(line_offsets, source_start, source_pos),
                    projected_start=projected_start,
                    projected_end=projected_pos,
                    exact=True,
                )
            )
            segment_index += 1
            continue

        if not source_text[source_pos].isspace():
            return None

        source_start = source_pos
        projected_start = projected_pos
        while source_pos < len(source_text) and source_text[source_pos].isspace():
            source_pos += 1
        while projected_pos < len(projected_text) and projected_text[projected_pos].isspace():
            projected_pos += 1

        segments.append(
            ProjectionSegment(
                segment_id=f"seg:{segment_index:04d}",
                kind="whitespace",
                source=_source_range(line_offsets, source_start, source_pos),
                projected_start=projected_start,
                projected_end=projected_pos,
                exact=False,
            )
        )
        segment_index += 1

    if not _remaining_is_whitespace(source_text, source_pos):
        return None
    if projected_pos != len(projected_text):
        return None

    if source_pos < len(source_text):
        segments.append(
            ProjectionSegment(
                segment_id=f"seg:{segment_index:04d}",
                kind="whitespace",
                source=_source_range(line_offsets, source_pos, len(source_text)),
                projected_start=projected_pos,
                projected_end=projected_pos,
                exact=False,
            )
        )

    if not _reconstructs_projected_text(source_text, projected_text, segments):
        return None

    return ProjectionMapping(
        version="v1",
        projection_kind="compact",
        path=path,
        lang=lang,
        source_length=len(source_text),
        projected_length=len(projected_text),
        source_hash=ProjectionMapping.digest(source_text),
        projected_hash=ProjectionMapping.digest(projected_text),
        source_line_offsets=line_offsets,
        segments=tuple(segments),
    )


def resolve_projected_range(
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> SourceRange | None:
    """Resolve a projected span back to source text.

    This only succeeds when the full projected range lies inside exact segments.
    Any overlap with a lossy whitespace segment returns ``None``.
    """
    if projected_start < 0 or projected_end < projected_start or projected_end > mapping.projected_length:
        return None

    if projected_start == projected_end:
        for segment in mapping.segments:
            if segment.exact:
                continue
            if segment.projected_start <= projected_start <= segment.projected_end:
                return None
        for segment in mapping.segments:
            if not segment.exact:
                continue
            if segment.projected_start <= projected_start <= segment.projected_end:
                delta = projected_start - segment.projected_start
                offset = segment.source.start_offset + delta
                return _source_range(mapping.source_line_offsets, offset, offset)
        return None

    start_segment = _find_segment(mapping, projected_start, inclusive_end=False)
    end_segment = _find_segment(mapping, projected_end - 1, inclusive_end=False)
    if start_segment is None or end_segment is None:
        return None
    if not start_segment.exact or not end_segment.exact:
        return None

    for segment in mapping.segments:
        if segment.projected_end <= projected_start or segment.projected_start >= projected_end:
            continue
        if not segment.exact:
            return None

    source_start = start_segment.source.start_offset + (projected_start - start_segment.projected_start)
    source_end = end_segment.source.start_offset + (projected_end - end_segment.projected_start)
    return _source_range(mapping.source_line_offsets, source_start, source_end)


def suggest_exact_reread_range(
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
    context_lines: int = 1,
) -> str | None:
    """Suggest a small exact reread window around a projected span."""
    if projected_start < 0 or projected_end < projected_start or projected_end > mapping.projected_length:
        return None
    if not mapping.segments:
        return None

    indices = _touching_segment_indices(mapping, projected_start=projected_start, projected_end=projected_end)
    if not indices:
        return None

    start_index = max(0, min(indices) - 1)
    end_index = min(len(mapping.segments) - 1, max(indices) + 1)
    window = mapping.segments[start_index : end_index + 1]
    max_line = _line_for_offset(mapping.source_line_offsets, mapping.source_length)
    start_line = max(1, min(segment.source.start_line for segment in window) - context_lines)
    end_line = min(max_line, max(segment.source.end_line for segment in window) + context_lines)
    return f"L{start_line}-L{end_line}"


def _find_segment(
    mapping: ProjectionMapping,
    projected_offset: int,
    *,
    inclusive_end: bool,
) -> ProjectionSegment | None:
    for segment in mapping.segments:
        start = segment.projected_start
        end = segment.projected_end
        if inclusive_end:
            if start <= projected_offset <= end:
                return segment
            continue
        if start <= projected_offset < end:
            return segment
    return None


def _touching_segment_indices(
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> list[int]:
    if projected_start == projected_end:
        indices = [
            index
            for index, segment in enumerate(mapping.segments)
            if segment.projected_start <= projected_start <= segment.projected_end
        ]
        if indices:
            return indices
        for index, segment in enumerate(mapping.segments):
            if projected_start < segment.projected_start:
                return sorted({max(0, index - 1), index})
        return [len(mapping.segments) - 1]

    indices = [
        index
        for index, segment in enumerate(mapping.segments)
        if not (segment.projected_end <= projected_start or segment.projected_start >= projected_end)
    ]
    if indices:
        return indices
    return _touching_segment_indices(
        mapping,
        projected_start=projected_start,
        projected_end=projected_start,
    )


def _remaining_is_whitespace(text: str, start: int) -> bool:
    return all(char.isspace() for char in text[start:])


def _reconstructs_projected_text(
    source_text: str,
    projected_text: str,
    segments: list[ProjectionSegment],
) -> bool:
    rebuilt: list[str] = []
    for segment in segments:
        if segment.exact:
            rebuilt.append(source_text[segment.source.start_offset : segment.source.end_offset])
            continue
        rebuilt.append(projected_text[segment.projected_start : segment.projected_end])
    return "".join(rebuilt) == projected_text


def _line_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n" and index + 1 < len(text):
            offsets.append(index + 1)
    return tuple(offsets)


def _source_range(line_offsets: tuple[int, ...], start_offset: int, end_offset: int) -> SourceRange:
    return SourceRange(
        start_offset=start_offset,
        end_offset=end_offset,
        start_line=_line_for_offset(line_offsets, start_offset),
        end_line=_line_for_offset(line_offsets, end_offset),
    )


def _line_for_offset(line_offsets: tuple[int, ...], offset: int) -> int:
    return bisect_right(line_offsets, offset) or 1


__all__ = ["build_compact_mapping", "resolve_projected_range", "suggest_exact_reread_range"]
