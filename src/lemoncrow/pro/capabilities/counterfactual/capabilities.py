"""Per-model capability matrix and turn-requirement inference.

See docs/plans/active/commercial-wedge/W2-counterfactual.md for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass

from lemoncrow.pro.capabilities.counterfactual.pricing import CandidateModel

# ---------------------------------------------------------------------------
# Turn classification
# ---------------------------------------------------------------------------

# Tools that only read — safe to run on cheap/flash models.
_READ_TOOLS: frozenset[str] = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "mcp__lc__read",
        "mcp__lc__search",
        "mcp__lc__context",
        "mcp__lc__memory",
        "mcp__lc__compact",
        "mcp__lc__route",
        # Read-only code-intel tools (split from the former `code` tool).
        "mcp__lc__symbols",
        "mcp__lc__node",
        "mcp__lc__callers",
        "mcp__lc__callees",
        "mcp__lc__usages",
        "mcp__lc__code_search",
        "lemoncrow_smart_read",
        "lemoncrow_smart_search",
        "lemoncrow_get_reasoning_context",
        "search",
        "read",
        "context",
        "compact",
        "route",
        "symbols",
        "node",
        "callers",
        "callees",
        "usages",
        "code_search",
    }
)

# Tools that mutate — keep on high-tier models.
_EDIT_TOOLS: frozenset[str] = frozenset(
    {
        "Edit",
        "Write",
        "Bash",
        "NotebookEdit",
        "mcp__lc__edit",
        "mcp__lc__bash",
        "mcp__lc__sql",
        "mcp__lc__codemod",
        "mcp__lc__trace",
        "mcp__lc__verify",
        "mcp__lc__rescue",
        "lemoncrow_smart_edit",
        "lc_bash",
        "lemoncrow_sql",
        "edit",
        "bash",
        "sql",
        "pattern",
        "trace",
        "verify",
        "rescue",
    }
)


def classify_turn_kind(tool_name: str) -> str:
    """Return 'read', 'edit', or 'agent' for a given tool name."""
    if tool_name in _READ_TOOLS:
        return "read"
    if tool_name in _EDIT_TOOLS:
        return "edit"
    return "agent"


# ---------------------------------------------------------------------------
# Turn requirements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnRequirements:
    turn_kind: str  # "read" | "edit" | "agent"
    requires_tool_use: bool
    min_context_window: int = 0


def infer_turn_requirements(tool_name: str) -> TurnRequirements:
    """Infer what a model must support to handle this tool call."""
    kind = classify_turn_kind(tool_name)
    # All non-trivial turns need tool-use support; pure text read-only turns don't.
    requires_tool_use = kind in ("edit", "agent")
    return TurnRequirements(
        turn_kind=kind,
        requires_tool_use=requires_tool_use,
    )


def supports_turn(candidate: CandidateModel, requirements: TurnRequirements) -> bool:
    """Return True if *candidate* can satisfy *requirements*."""
    if requirements.requires_tool_use and not candidate.supports_tool_use:
        return False
    return requirements.min_context_window <= candidate.context_window


__all__ = [
    "TurnRequirements",
    "classify_turn_kind",
    "infer_turn_requirements",
    "supports_turn",
]
