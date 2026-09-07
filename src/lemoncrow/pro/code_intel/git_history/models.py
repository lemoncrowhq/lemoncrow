"""Typed models for historical symbol ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GraveyardEntry:
    symbol_name: str
    qualified_name: str | None
    file_path: str
    language: str | None
    deleted_at_sha: str
    deleted_at_ts: int
    last_author: str | None
    last_commit_msg: str | None
    rename_target: str | None = None
    signature_hash: str | None = None


FreshnessState = Literal["fresh", "stale"]


@dataclass(frozen=True)
class BlameRequest:
    file_path: str
    line_start: int
    line_end: int
    index_sha: str
    head_sha: str
    include_churn: bool = True
    churn_window_days: int = 180

    def __post_init__(self) -> None:
        if self.line_start < 1:
            raise ValueError("line_start must be >= 1")
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        if not self.index_sha:
            raise ValueError("index_sha is required")
        if not self.head_sha:
            raise ValueError("head_sha is required")


@dataclass(frozen=True)
class BlameHunk:
    start_line: int
    end_line: int
    commit_sha: str
    author_email: str | None
    commit_time: int


@dataclass(frozen=True)
class ChurnStats:
    commit_count: int
    score: float
    window_days: int = 180


@dataclass(frozen=True)
class BlameRangeAnnotation:
    file_path: str
    line_start: int
    line_end: int
    index_sha: str
    head_sha: str
    freshness: FreshnessState
    last_author: str | None
    last_commit_sha: str
    last_commit_summary: str | None
    age_days: int
    local_edits: bool
    hunks: tuple[BlameHunk, ...]
    churn: ChurnStats | None = None


@dataclass(frozen=True)
class CommitRecord:
    """Raw enumerated commit before summarisation."""

    sha: str
    author_date: int  # unix seconds
    message: str  # strip()[:2000]
    files_touched: list[str]
    is_merge: bool


@dataclass(frozen=True)
class CommitSummary:
    """LLM-generated semantic summary of a single commit."""

    sha: str
    author_date: int
    files_touched: list[str]
    summary: str  # ≤200 tokens
    summary_model: str
    prompt_version: str  # "v1"
