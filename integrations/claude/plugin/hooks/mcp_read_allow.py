"""PreToolUse auto-allow for lc's genuinely read-only MCP tools.

Claude Code's Plan Mode runs its own approval gate, independent of
``permissions.allow``. It auto-passes only the built-in read tools
(Read/Grep/Glob/WebFetch); a third-party MCP tool is never in that set, so
``mcp__lc__read`` & friends prompt on every call even when ``lc init`` /
``install_claude.sh`` already whole-tool-allowed them. There is no MCP-side
annotation Claude Code honours for this -- the only thing that suppresses the
prompt in *every* mode is a PreToolUse hook returning
``hookSpecificOutput.permissionDecision: "allow"``.

Scope is deliberately narrow. Only lookup-only tools are named here; anything
that can mutate a file, a store, or a shell (``edit``, ``bash``, ``sql``,
``codemod``, ``memory``, ``compact``, ``verify``, ``agent``, ``workflow``) is
omitted and keeps prompting normally. ``tool`` is omitted too: it dispatches to
any rarely-used lc tool by name, including write-capable ones, so allowing it
would launder the whole surface through one decision.

Stays silent (no decision at all) for every other tool, so it never overrides a
user's own deny rule for something outside this list.

Fail-open; opt-out via LEMONCROW_MCP_READ_ALLOW=0.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

# MCP server key as registered by install_claude.sh (user scope: "lc",
# workspace .mcp.json / plugin mcp.json: "lemoncrow").
_SERVERS = frozenset({"lc", "lemoncrow"})

# Read-only lc tools. Each only reads the workspace, the index, or the network;
# none writes files, the store, or runs a command.
_READ_ONLY_TOOLS = frozenset(
    {
        "blame",
        "code_search",
        "context",
        "graph",
        "grep",
        "orient",
        "read",
        "relations",
        "search",
        "web_fetch",
    }
)


def _read_only_tool(tool_name: str) -> str | None:
    """Return the bare lc tool name when ``tool_name`` is an allowed read tool."""
    parts = tool_name.split("__")
    if len(parts) != 3 or parts[0] != "mcp" or parts[1] not in _SERVERS:
        return None
    return parts[2] if parts[2] in _READ_ONLY_TOOLS else None


def _allow(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> int:
    if os.environ.get("LEMONCROW_MCP_READ_ALLOW", "1") == "0":
        return 0
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read() or "{}")
    except (json.JSONDecodeError, TypeError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    tool = _read_only_tool(str(payload.get("tool_name") or ""))
    if tool is None:
        return 0
    _allow(f"lc {tool} is read-only (no writes, no shell); auto-allowed in every mode including Plan Mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
