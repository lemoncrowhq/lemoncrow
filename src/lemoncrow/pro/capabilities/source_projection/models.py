"""Data models for read-side source projection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

ProjectionView = Literal["summary", "exact", "range", "outline", "compact", "minified"]


@dataclass(frozen=True)
class SourceRange:
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceRange:
        return cls(
            start_offset=int(payload["start_offset"]),
            end_offset=int(payload["end_offset"]),
            start_line=int(payload["start_line"]),
            end_line=int(payload["end_line"]),
        )


@dataclass(frozen=True)
class ProjectionSegment:
    segment_id: str
    kind: Literal["exact", "whitespace", "dropped"]
    source: SourceRange
    projected_start: int
    projected_end: int
    exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "source": self.source.to_dict(),
            "projected_start": self.projected_start,
            "projected_end": self.projected_end,
            "exact": self.exact,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectionSegment:
        kind = str(payload["kind"])
        if kind not in {"exact", "whitespace", "dropped"}:
            raise ValueError(f"unsupported projection segment kind: {kind}")
        return cls(
            segment_id=str(payload["segment_id"]),
            kind=cast(Literal["exact", "whitespace", "dropped"], kind),
            source=SourceRange.from_dict(dict(payload["source"])),
            projected_start=int(payload["projected_start"]),
            projected_end=int(payload["projected_end"]),
            exact=bool(payload["exact"]),
        )


@dataclass(frozen=True)
class ProjectionMapping:
    version: Literal["v1"]
    projection_kind: Literal["compact", "minified"]
    path: str
    lang: str
    source_length: int
    projected_length: int
    source_hash: str
    projected_hash: str
    source_line_offsets: tuple[int, ...]
    segments: tuple[ProjectionSegment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "projection_kind": self.projection_kind,
            "path": self.path,
            "lang": self.lang,
            "source_length": self.source_length,
            "projected_length": self.projected_length,
            "source_hash": self.source_hash,
            "projected_hash": self.projected_hash,
            "source_line_offsets": list(self.source_line_offsets),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectionMapping:
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("projection_mapping.segments must be a list")
        version = str(payload["version"])
        if version != "v1":
            raise ValueError(f"unsupported projection mapping version: {version}")
        projection_kind = str(payload["projection_kind"])
        if projection_kind not in {"compact", "minified"}:
            raise ValueError(f"unsupported projection kind: {projection_kind}")
        return cls(
            version=cast(Literal["v1"], version),
            projection_kind=cast(Literal["compact", "minified"], projection_kind),
            path=str(payload["path"]),
            lang=str(payload["lang"]),
            source_length=int(payload["source_length"]),
            projected_length=int(payload["projected_length"]),
            source_hash=str(payload["source_hash"]),
            projected_hash=str(payload["projected_hash"]),
            source_line_offsets=tuple(int(item) for item in payload.get("source_line_offsets", [])),
            segments=tuple(ProjectionSegment.from_dict(dict(item)) for item in raw_segments),
        )

    @staticmethod
    def digest(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProjectionDelta:
    """Per-read projection telemetry."""

    path: str
    lang: str
    original_tokens: int
    projected_tokens: int
    kind: str = "compact"

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.projected_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "lang": self.lang,
            "kind": self.kind,
            "original_tokens": self.original_tokens,
            "projected_tokens": self.projected_tokens,
            "saved_tokens": self.saved_tokens,
        }


@dataclass(frozen=True)
class SourceProjection:
    """Describes the view returned to the model for a read request."""

    view: ProjectionView
    transformed: bool
    body_complete: bool
    untransformed_text: bool
    notice: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "view": self.view,
            "transformed": self.transformed,
            "body_complete": self.body_complete,
            "untransformed_text": self.untransformed_text,
        }
        if self.notice:
            payload["notice"] = self.notice
        return payload

    @classmethod
    def summary(cls) -> SourceProjection:
        return cls(
            view="summary",
            transformed=True,
            body_complete=False,
            untransformed_text=False,
        )

    @classmethod
    def exact(cls) -> SourceProjection:
        return cls(
            view="exact",
            transformed=False,
            body_complete=True,
            untransformed_text=True,
        )

    @classmethod
    def range(cls) -> SourceProjection:
        return cls(
            view="range",
            transformed=False,
            body_complete=False,
            untransformed_text=True,
        )

    @classmethod
    def outline(cls) -> SourceProjection:
        return cls(
            view="outline",
            transformed=True,
            body_complete=False,
            untransformed_text=False,
            notice="(outline; :full = source)",
        )

    @classmethod
    def compact(cls) -> SourceProjection:
        return cls(
            view="compact",
            transformed=True,
            body_complete=True,
            untransformed_text=False,
            notice="(compact; :full = source)",
        )

    @classmethod
    def compact_with_gutter(cls) -> SourceProjection:
        """Compact, but each line is prefixed with its real disk line number --
        see ``minified_with_gutter``; the same ":Lx-Ly is safe to use directly"
        property applies.
        """
        return cls(
            view="compact",
            transformed=True,
            body_complete=True,
            untransformed_text=False,
            notice="(compact; number = real disk line, safe in :Lx-Ly. Gaps are stripped blank-line runs only -- no code is hidden.)",
        )

    @classmethod
    def minified(cls) -> SourceProjection:
        return cls(
            view="minified",
            transformed=True,
            body_complete=True,
            untransformed_text=False,
            notice="(minified; use :minified:Lx-Ly to edit)",
        )

    @classmethod
    def minified_with_gutter(cls) -> SourceProjection:
        """Minified, but each line is prefixed with its real disk line number
        (see ``minify.minified_line_gutter``) -- unlike ``minified()``, a
        :Lx-Ly range copied straight off what's shown IS disk-accurate, so no
        "differs from disk" warning is needed; only that lines were omitted.
        """
        return cls(
            view="minified",
            transformed=True,
            body_complete=True,
            untransformed_text=False,
            notice="(minified; number = real disk line, safe in :Lx-Ly. Gaps are stripped comments/blank lines only -- no code is hidden.)",
        )
