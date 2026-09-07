"""Structured counterexample model + prompt formatter (M3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from lemoncrow.pro.capabilities.prompt_compilation import (
    COUNTEREXAMPLE_METADATA_KEY,
    BlockKind,
    PromptBlock,
    Stability,
)


@dataclass
class Counterexample:
    """A single check failure rendered as actionable feedback."""

    check: str  # "lint" | "typecheck" | "tests" | "semantic"
    severity: str  # "error" | "warn"
    file_path: str | None
    line: int | None
    diagnostic: str
    expected: str | None = None
    actual: str | None = None
    repro_command: str | None = None

    def to_prompt_block(self) -> str:
        """Render as a structured block the agent can ingest (TURN-channel)."""
        attrs = [f'check="{self.check}"', f'severity="{self.severity}"']
        if self.file_path:
            attrs.append(f'file="{self.file_path}"')
        if self.line is not None:
            attrs.append(f"line={self.line}")
        lines = [f"<counterexample {' '.join(attrs)}>"]
        if self.expected is not None:
            lines.append(f"  expected: {self.expected}")
        if self.actual is not None:
            lines.append(f"  actual:   {self.actual}")
        if self.repro_command:
            lines.append(f"  repro:    {self.repro_command}")
        if self.diagnostic:
            lines.append(f"  diagnostic: {self.diagnostic}")
        lines.append("</counterexample>")
        return "\n".join(lines)

    def to_compiler_block(self) -> PromptBlock:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = sha256(payload.encode("utf-8")).hexdigest()[:16]
        return PromptBlock(
            id=f"counterexample/{self.check}/{digest}",
            kind=BlockKind.TOOL_RESULT,
            content=self.to_prompt_block(),
            stability=Stability.TURN,
            cacheable=False,
            metadata={
                COUNTEREXAMPLE_METADATA_KEY: True,
                "check": self.check,
                "severity": self.severity,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "file_path": self.file_path,
            "line": self.line,
            "diagnostic": self.diagnostic,
            "expected": self.expected,
            "actual": self.actual,
            "repro_command": self.repro_command,
        }
