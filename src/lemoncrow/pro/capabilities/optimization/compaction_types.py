"""Four explicit compaction types used by the Optimization Advisor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RiskLevel = Literal["very_low", "medium", "high"]


@dataclass(frozen=True)
class CompactionType:
    id: str
    label: str
    risk: RiskLevel
    mechanism: str
    default_enabled: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "id": self.id,
            "label": self.label,
            "risk": self.risk,
            "mechanism": self.mechanism,
            "default_enabled": self.default_enabled,
        }


ALL_COMPACTION_TYPES: tuple[CompactionType, ...] = (
    CompactionType(
        id="prompt_cache_reorder",
        label="Prompt-cache reorder",
        risk="very_low",
        mechanism="Put stable instructions, tool schemas, and repo maps first to maximize provider cache hits.",
        default_enabled=True,
    ),
    CompactionType(
        id="dedup",
        label="Dedup compaction",
        risk="very_low",
        mechanism="Remove repeated tool outputs, duplicate file reads, and repeated logs.",
        default_enabled=True,
    ),
    CompactionType(
        id="retrieval_filter",
        label="Retrieval filter",
        risk="medium",
        mechanism="Include only retrieval-relevant files and spans, with safe-mode preserving active work context.",
        default_enabled=True,
    ),
    CompactionType(
        id="lossy_summary",
        label="Lossy summary",
        risk="high",
        mechanism="Summarize old discussion, resolved branches, and repeated reasoning.",
        default_enabled=False,
    ),
)
