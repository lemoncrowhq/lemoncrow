"""Projection-aware edit helpers built on compact mapping metadata."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from lemoncrow.pro.capabilities.source_projection.errors import ProjectionEditError
from lemoncrow.pro.capabilities.source_projection.mapping import (
    resolve_projected_range,
    suggest_exact_reread_range,
)
from lemoncrow.pro.capabilities.source_projection.minify import resolve_minified_span
from lemoncrow.pro.capabilities.source_projection.models import ProjectionMapping


def _read_retry(path: str, *, expand: bool = False, range_spec: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"tool": "read", "path": path, "include_meta": True}
    if expand:
        payload["full"] = True
    if range_spec:
        payload["range"] = range_spec
    return payload


def _touching_segment_bounds(
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> tuple[int, int]:
    if projected_start == projected_end:
        indices = [
            index
            for index, segment in enumerate(mapping.segments)
            if segment.projected_start <= projected_start <= segment.projected_end
        ]
        if indices:
            return min(indices), max(indices)
        for index, segment in enumerate(mapping.segments):
            if projected_start < segment.projected_start:
                return max(0, index - 1), index
        return len(mapping.segments) - 1, len(mapping.segments) - 1

    indices = [
        index
        for index, segment in enumerate(mapping.segments)
        if not (segment.projected_end <= projected_start or segment.projected_start >= projected_end)
    ]
    if indices:
        return min(indices), max(indices)
    return _touching_segment_bounds(
        mapping,
        projected_start=projected_start,
        projected_end=projected_start,
    )


def _anchor_text(content: str, mapping: ProjectionMapping, *, index: int, reverse: bool) -> str | None:
    step = -1 if reverse else 1
    stop = -1 if reverse else len(mapping.segments)
    for cursor in range(index, stop, step):
        segment = mapping.segments[cursor]
        if not segment.exact:
            continue
        snippet = " ".join(content[segment.source.start_offset : segment.source.end_offset].split()).strip()
        if not snippet:
            continue
        return snippet[-24:] if reverse else snippet[:24]
    return None


def _selection_context(
    content: str,
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
    range_spec: str,
) -> dict[str, Any]:
    start_index, end_index = _touching_segment_bounds(
        mapping,
        projected_start=projected_start,
        projected_end=projected_end,
    )
    window = mapping.segments[start_index : end_index + 1]
    payload: dict[str, Any] = {
        "line_range": range_spec,
        "segment_kinds": [segment.kind for segment in window],
    }
    before = _anchor_text(content, mapping, index=start_index, reverse=True)
    after = _anchor_text(content, mapping, index=end_index, reverse=False)
    if before:
        payload["before"] = before
    if after:
        payload["after"] = after
    return payload


def _exact_retry_with(
    content: str,
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> dict[str, Any]:
    range_spec = suggest_exact_reread_range(
        mapping,
        projected_start=projected_start,
        projected_end=projected_end,
    )
    if range_spec:
        retry_with = _read_retry(mapping.path, range_spec=range_spec)
        retry_with["selection_context"] = _selection_context(
            content,
            mapping,
            projected_start=projected_start,
            projected_end=projected_end,
            range_spec=range_spec,
        )
        return retry_with
    return _read_retry(mapping.path, expand=True)


def _ambiguous_retry(
    content: str,
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> tuple[str, dict[str, Any]]:
    retry_with = _exact_retry_with(
        content,
        mapping,
        projected_start=projected_start,
        projected_end=projected_end,
    )
    range_spec = retry_with.get("range")
    if isinstance(range_spec, str) and range_spec:
        return (
            f"Choose an exact token span, or re-read with range={range_spec} for untransformed text.",
            retry_with,
        )
    return (
        "Choose an exact token span, or re-read with full=true for untransformed text.",
        retry_with,
    )


def _overlap_retry(
    content: str,
    mapping: ProjectionMapping,
    *,
    projected_start: int,
    projected_end: int,
) -> tuple[str, dict[str, Any]]:
    retry_with = _exact_retry_with(
        content,
        mapping,
        projected_start=projected_start,
        projected_end=projected_end,
    )
    range_spec = retry_with.get("range")
    if isinstance(range_spec, str) and range_spec:
        return (
            f"Re-read with range={range_spec} and reissue non-overlapping exact spans.",
            retry_with,
        )
    return (
        "Re-read with full=true and reissue non-overlapping exact spans.",
        retry_with,
    )


def _validate_projection_mapping(content: str, mapping: ProjectionMapping) -> None:
    if mapping.projection_kind not in {"compact", "minified"}:
        raise ProjectionEditError(
            f"unsupported projection kind: {mapping.projection_kind}",
            code="unsupported_projection_kind",
            hint="Use projection_kind='compact' or 'minified'.",
        )
    if ProjectionMapping.digest(content) != mapping.source_hash:
        raise ProjectionEditError(
            "projection mapping is stale; re-read the file before editing",
            code="stale_projection_mapping",
            hint="Re-read the file to get a fresh compact projection, or use full=true for untransformed text.",
            retry_with=_read_retry(mapping.path, expand=True),
        )


def apply_compact_projection_edits(
    content: str,
    *,
    mapping: ProjectionMapping,
    projected_edits: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int]]]:
    """Apply multiple exact projected spans back onto untransformed source text."""
    _validate_projection_mapping(content, mapping)

    resolved: list[tuple[int, int, int, int, str, int, int]] = []
    for edit in projected_edits:
        projected_start = int(edit["projected_start"])
        projected_end = int(edit["projected_end"])
        new_string = str(edit.get("new_string", ""))
        if mapping.projection_kind == "minified":
            source_range = resolve_minified_span(
                mapping,
                projected_start=projected_start,
                projected_end=projected_end,
            )
        else:
            source_range = resolve_projected_range(
                mapping,
                projected_start=projected_start,
                projected_end=projected_end,
            )
        if source_range is None:
            hint, retry_with = _ambiguous_retry(
                content,
                mapping,
                projected_start=projected_start,
                projected_end=projected_end,
            )
            raise ProjectionEditError(
                "projected range is ambiguous; re-read with full=true or an exact range",
                code="ambiguous_projected_range",
                hint=hint,
                retry_with=retry_with,
            )
        resolved.append(
            (
                source_range.start_offset,
                source_range.end_offset,
                source_range.start_line,
                max(source_range.start_line, source_range.end_line),
                new_string,
                projected_start,
                projected_end,
            )
        )

    resolved.sort(key=lambda item: item[0])
    for previous, current in pairwise(resolved):
        if current[0] < previous[1]:
            hint, retry_with = _overlap_retry(
                content,
                mapping,
                projected_start=min(previous[5], current[5]),
                projected_end=max(previous[6], current[6]),
            )
            raise ProjectionEditError(
                "projected ranges overlap after source resolution",
                code="overlapping_projected_ranges",
                hint=hint,
                retry_with=retry_with,
            )

    updated = content
    hunks: list[tuple[int, int]] = []
    for start_offset, end_offset, start_line, end_line, new_string, _, _ in reversed(resolved):
        updated = updated[:start_offset] + new_string + updated[end_offset:]
        hunks.append((start_line, end_line))
    hunks.reverse()
    return updated, hunks


def apply_compact_projection_edit(
    content: str,
    *,
    mapping: ProjectionMapping,
    projected_start: int,
    projected_end: int,
    new_string: str,
) -> tuple[str, int, int]:
    """Apply a projection-aware edit back onto untransformed source text."""
    updated, hunks = apply_compact_projection_edits(
        content,
        mapping=mapping,
        projected_edits=[
            {
                "projected_start": projected_start,
                "projected_end": projected_end,
                "new_string": new_string,
            }
        ],
    )
    line_start, line_end = hunks[0]
    return updated, line_start, line_end


__all__ = [
    "ProjectionEditError",
    "apply_compact_projection_edit",
    "apply_compact_projection_edits",
]
