"""Literal-only static cross-language edge support for Phase 5."""

from .edges import CrossLangEdge, CrossLangEdgeStore
from .runner import CrossLangRunner

__all__ = ["CrossLangEdge", "CrossLangEdgeStore", "CrossLangRunner"]
