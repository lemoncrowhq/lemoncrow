"""Block model for the LemonCrow prompt compiler.

PromptBlock, Stability, BlockKind, and DEFAULT_STABILITY are the foundational
types that P1 (compiler), P2 (linter), P3 (providers), and P5 (trace) all
depend on.

Invariants:
- PromptBlock is frozen (immutable). Do not add mutable fields.
- version_hash is sha256(content.encode("utf-8")).hexdigest() — stable across processes.
- cacheable is forced False for TURN and VOLATILE regardless of caller intent.
- Stability override requires stability_override_reason.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any


class Stability(StrEnum):
    """Total stability order for prompt blocks.

    The sort key for cache-prefix planning is STABILITY_ORDER[stability],
    defined in P1 (compiler.py). This enum defines the taxonomy only.
    """

    STATIC = "static"  # tool schemas, system prompt, coding policy
    SESSION = "session"  # repo summary, project conventions
    BRANCH = "branch"  # Playbooks, file summaries for this task
    TURN = "turn"  # user task, current diff, last tool result
    VOLATILE = "volatile"  # timestamps, request IDs, raw logs


class BlockKind(StrEnum):
    """Semantic type of a prompt block."""

    TOOL_SCHEMA = "tool_schema"
    SYSTEM = "system"
    CODING_POLICY = "coding_policy"
    REPO_SUMMARY = "repo_summary"
    PLAYBOOK = "playbook"
    FILE_SUMMARY = "file_summary"
    USER_TASK = "user_task"
    GIT_DIFF = "git_diff"
    TOOL_RESULT = "tool_result"
    SCRATCHPAD = "scratchpad"


# Canonical stability for each block kind.
# A PromptBlock may override this with stability_override_reason.
DEFAULT_STABILITY: dict[BlockKind, Stability] = {
    BlockKind.TOOL_SCHEMA: Stability.STATIC,
    BlockKind.SYSTEM: Stability.STATIC,
    BlockKind.CODING_POLICY: Stability.STATIC,
    BlockKind.REPO_SUMMARY: Stability.SESSION,
    BlockKind.PLAYBOOK: Stability.BRANCH,
    BlockKind.FILE_SUMMARY: Stability.BRANCH,
    BlockKind.USER_TASK: Stability.TURN,
    BlockKind.GIT_DIFF: Stability.TURN,
    BlockKind.TOOL_RESULT: Stability.TURN,
    BlockKind.SCRATCHPAD: Stability.VOLATILE,
}

_NON_CACHEABLE_STABILITY = {Stability.TURN, Stability.VOLATILE}
_ID_RE = re.compile(r"^[a-z0-9_./:-]+$")
COUNTEREXAMPLE_METADATA_KEY = "verification.counterexample"


@dataclass(frozen=True)
class PromptBlock:
    """An immutable, typed prompt block with deterministic hash and token estimate.

    Args:
        id: Unique identifier matching ^[a-z0-9_./:-]+$. Used in cache keys and trace IDs.
        kind: Semantic type of this block (determines default stability).
        content: Non-empty text content of the block.
        stability: Cache stability level. Defaults to DEFAULT_STABILITY[kind].
            If overriding the default, stability_override_reason is required.
        cacheable: Whether this block is eligible for provider-side prefix caching.
            Forced to False for TURN and VOLATILE regardless of caller intent.
        metadata: Optional key-value metadata (labels, source paths, etc.).
        stability_override_reason: Required when stability != DEFAULT_STABILITY[kind].
    """

    id: str
    kind: BlockKind
    content: str
    stability: Stability
    cacheable: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stability_override_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("PromptBlock.content must not be empty")
        if not _ID_RE.match(self.id):
            raise ValueError(f"PromptBlock.id {self.id!r} is invalid; must match ^[a-z0-9_./:-]+$")
        default_stab = DEFAULT_STABILITY[self.kind]
        if self.stability != default_stab and not self.stability_override_reason:
            raise ValueError(
                f"PromptBlock stability {self.stability!r} overrides the default "
                f"{default_stab!r} for kind {self.kind!r}; provide stability_override_reason"
            )
        # Force cacheable=False for TURN and VOLATILE regardless of caller intent.
        # object.__setattr__ is required because the dataclass is frozen=True.
        if self.stability in _NON_CACHEABLE_STABILITY and self.cacheable:
            object.__setattr__(self, "cacheable", False)

    @property
    def version_hash(self) -> str:
        """SHA-256 of content encoded as UTF-8. Deterministic across processes."""
        # Defensive: some callers (SDK adapters ingesting external tool/provider
        # content) hand us raw bytes despite the str contract; bytes has no
        # `.encode`, so guard here rather than let that raise.
        content = self.content if isinstance(self.content, bytes) else self.content.encode("utf-8")
        return sha256(content).hexdigest()

    @property
    def token_estimate(self) -> int:
        """Estimated token count. Uses tiktoken when available, char/4 fallback otherwise."""
        from .tokens import estimate_tokens

        return estimate_tokens(self.content)

    @property
    def is_counterexample(self) -> bool:
        return self.metadata.get(COUNTEREXAMPLE_METADATA_KEY) is True
