"""Source projection capability helpers."""

from .compact import CompactProjectionResult, build_compact_projection
from .edit import ProjectionEditError, apply_compact_projection_edit, apply_compact_projection_edits
from .mapping import build_compact_mapping, resolve_projected_range, suggest_exact_reread_range
from .minify import (
    MinifiedEditError,
    MinifiedProjectionResult,
    apply_minified_edit,
    build_minified_projection,
    language_for_minify,
    minified_line_gutter,
    resolve_minified_span,
    translate_minified_line_range,
)
from .models import (
    ProjectionDelta,
    ProjectionMapping,
    ProjectionSegment,
    SourceProjection,
    SourceRange,
)

__all__ = [
    "CompactProjectionResult",
    "MinifiedEditError",
    "MinifiedProjectionResult",
    "ProjectionDelta",
    "ProjectionEditError",
    "ProjectionMapping",
    "ProjectionSegment",
    "SourceProjection",
    "SourceRange",
    "apply_compact_projection_edit",
    "apply_compact_projection_edits",
    "apply_minified_edit",
    "build_compact_mapping",
    "build_compact_projection",
    "build_minified_projection",
    "language_for_minify",
    "minified_line_gutter",
    "resolve_minified_span",
    "resolve_projected_range",
    "suggest_exact_reread_range",
    "translate_minified_line_range",
]
